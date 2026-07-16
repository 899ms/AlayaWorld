"""FlashAlaya demo entry: standalone engine + streaming pipeline + save video.

Lossless optimizations (compile + cudagraphs, flex-attention, perturbation
shortcut, merged weights, meta-init load, seamless overlap-tiling VAE decode)
are ON by default. Lossy approximations (taehv bank decode) are opt-in flags.

Usage:
    PYTORCH_ALLOC_CONF=expandable_segments:True python -m flash_alaya.run \
        [--input playground/case1/case1] [--rounds 5] [--seed 1234]

Input is user-provided (no dataset pipeline), either a scene triplet
    <prefix>_video.mp4 + <prefix>_camera.pt + <prefix>_prompt.txt
or a single .pt dict {"video_pixels": [F,C,H,W] in [-1,1], "caption", "metadata"}
(see load_input_sample below; ready-made cases ship under playground/).
API callers can feed (video_pixels, caption, metadata) directly to
FlashAlayaPipeline.initialize_cache.
"""

import argparse
import time
from pathlib import Path

import torch
import torch.distributed as dist

from flash_alaya.alaya.config.loader import load_config
from flash_alaya.utils.pipeline import FlashAlayaPipeline
from flash_alaya.utils.rollout_utils import (
    apply_joystick_overlay,
    bootstrap_context_parallel,
    build_engine,
    check_input_resolution,
    free_generation_models,
    load_input_sample,
    plan_rollout,
    silence_worker_rank,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="flash_alaya.run",
        description="FlashAlaya standalone inference. Single card: python -m flash_alaya.run; "
        "multi-card Context Parallel: torchrun --nproc_per_node=N (e.g. 2 or 4).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cfg", default="configs/infer.yaml", help="inference config (yaml)")
    p.add_argument("--input", default="playground/case1/case1", help="case/scene prefix (_image.*|_video.mp4 + _camera.pt + _prompt.txt)")
    p.add_argument("--rounds", type=int, default=1000, help="max autoregressive chunks; min(this, trajectory length)")
    p.add_argument("--seed", type=int, default=None, help="fix per-chunk noise seed for reproducible runs")
    p.add_argument(
        "--compile",
        default="reduce-overhead",
        choices=["reduce-overhead", "default", "max-autotune", "none"],
        help="torch.compile mode for the DiT (under CP, reduce-overhead auto-downgrades to default)",
    )
    p.add_argument("--flex-attn", action=argparse.BooleanOptionalAction, default=True, help="fused flex_attention")
    p.add_argument(
        "--ttc",
        action="store_true",
        help="Pathwise Test-Time Correction (arXiv:2602.05871): re-anchor each chunk's "
        "denoise to the initial frame to curb appearance/style drift over long rollouts. "
        "OFF by default; when omitted the output is identical to the baseline. The TTC "
        "knobs (levels / strength / ref_action) live in config under validation.ttc.",
    )
    p.add_argument(
        "--bank-taehv",
        action="store_true",
        help="LOSSY: decode the spatial bank with taehv instead of full VAE (faster); weights from paths.taehv",
    )
    p.add_argument("--video-crf", type=int, default=28, help="h264 crf for saved mp4s (18 near-lossless, 28 small)")
    return p.parse_args()



def main() -> None:
    args = parse_args()
    flex_attn = args.flex_attn and args.compile != "none"
    if args.flex_attn and not flex_attn:
        print("[FlashAlaya] --compile none: flex-attn off (eager math fallback)", flush=True)

    cp_size, is_rank0, args.seed, args.compile = bootstrap_context_parallel(args.seed, args.compile)
    if not is_rank0:
        silence_worker_rank()  # only rank0 prints; workers stay quiet (stderr kept for tracebacks)
    cfg = load_config(args.cfg)  # seamless overlap-tiling decode params come from cfg.runtime

    # Read + validate the input BEFORE loading any model, so a wrong-resolution
    # input fails immediately instead of after the (~40s) engine load + rollout.
    video_pixels, caption, metadata = load_input_sample(
        args.input, image_target_hw=(int(cfg.sample.height), int(cfg.sample.width))
    )
    check_input_resolution(video_pixels, cfg)  # images are auto-fit above; videos must already match

    engine = build_engine(
        cfg, compile_mode=args.compile, compile_aux=False, bank_taehv=args.bank_taehv, verbose=is_rank0
    )

    mode_cfg = next(iter(cfg.validation.modes.values()))
    K = int(mode_cfg.layout.output_latent_frames)
    N = int(
        cfg.layout.history_latent_frames
        if mode_cfg.layout.history_latent_frames is None
        else mode_cfg.layout.history_latent_frames
    )
    gap_steps = int(float(mode_cfg.layout.max_gap_sec or 0.0) * cfg.sample.fps / cfg.sample.temporal_stride)
    cond_end = int(mode_cfg.layout.condition_latent_frames)
    video_pixels, metadata, rounds, max_rounds, needed_latents = plan_rollout(
        cfg, video_pixels, metadata, rounds_cap=args.rounds, K=K, N=N, gap_steps=gap_steps, cond_end=cond_end
    )
    print(
        f"[FlashAlaya] input={args.input} rounds={rounds}/{max_rounds} (cap {args.rounds}) caption={caption[:50]!r}",
        flush=True,
    )

    pipe = FlashAlayaPipeline(
        engine,
        control_modes=list(mode_cfg.control),
        use_memory=bool(mode_cfg.use_memory),
        action_cfg_scale=float(mode_cfg.action_cfg_scale),
        flex_attn=flex_attn,
        seed=args.seed,
        ttc=args.ttc,
        ttc_levels=tuple(int(x) for x in cfg.validation.ttc.levels),
        ttc_strength=float(cfg.validation.ttc.strength),
        ttc_ref_action=bool(cfg.validation.ttc.ref_action),
    )
    if is_rank0:
        _ttc = cfg.validation.ttc
        ttc_str = (
            (
                f"ON levels={','.join(map(str, _ttc.levels))} strength={_ttc.strength} "
                f"ref:{'action' if _ttc.ref_action else 'no-action'}"
            )
            if args.ttc
            else "off"
        )
        print(
            f"[FlashAlaya] compile={args.compile} flex_attn={'on' if flex_attn else 'off'} "
            f"seed={args.seed} ttc={ttc_str}",
            flush=True,
        )

    cache = pipe.initialize_cache(
        video_pixels, caption, metadata, rounds=rounds, K=K, cond_end=cond_end, needed_latents=needed_latents
    )

    for i in range(rounds):
        t0 = time.perf_counter()
        pred = pipe.generate(i, cache)
        t1 = time.perf_counter()
        pipe.finalize(i, cache, pred)
        torch.cuda.synchronize()
        if is_rank0:
            print(
                f"[FlashAlaya] chunk {i + 1}/{rounds}: generate={t1 - t0:.2f}s finalize={time.perf_counter() - t1:.2f}s",
                flush=True,
            )

    if is_rank0:  # only rank0 decodes + saves
        free_generation_models(engine)
        t0 = time.perf_counter()
        frames = pipe.decode(cache)
        if cfg.validation.save_joystick:  # default: overlay the Move/Rotate joystick HUD
            frames = apply_joystick_overlay(cfg, cache, frames)
        out_dir = Path(cfg.run.output_dir) / "flash_alaya"
        out_dir.mkdir(parents=True, exist_ok=True)
        in_stem = Path(args.input).stem
        out_path = out_dir / f"{in_stem}_rounds-{rounds}.mp4"
        engine.write_video(out_path, frames, crf=args.video_crf)
        print(
            f"[FlashAlaya] decode {time.perf_counter() - t0:.1f}s frames={tuple(frames.shape)} -> saved {out_path}",
            flush=True,
        )

    if cp_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
