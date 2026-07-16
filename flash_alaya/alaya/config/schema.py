from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


LTX_DEFAULT_NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, "
    "excessive noise, grainy texture, poor lighting, flickering, motion blur, "
    "distorted proportions, unnatural skin tones, deformed facial features, "
    "asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, "
    "incorrect depth of field, background too sharp, background clutter, "
    "distracting reflections, harsh shadows, inconsistent lighting direction, "
    "color banding, cartoonish rendering, 3D CGI look, unrealistic materials, "
    "uncanny valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, "
    "wrong gaze direction, mismatched lip sync, silent or muted audio, distorted voice, "
    "robotic voice, echo, background noise, off-sync audio, incorrect dialogue, "
    "added dialogue, repetitive speech, jittery movement, awkward pauses, "
    "incorrect timing, unnatural transitions, inconsistent framing, tilted camera, "
    "flat lighting, inconsistent tone, cinematic oversaturation, stylized filters, "
    "or AI artifacts."
)


@dataclass
class RunConfig:
    name: str = "rollout_from_bcd"
    seed: int = 42
    output_dir: str = "./logs_rollout"
    log_dir: str = "./logs/rollout_from_bcd"
    debug: bool = False


@dataclass
class PathsConfig:
    # merged one-file inference weights (DiT + VAE + text-enc + history_encoder);
    # preferred name for inference configs. Falls through to the training-side
    # transformer/base_transformer fields below.
    model: str = ""
    transformer: str = ""
    base_transformer: str = ""
    continue_transformer: str = ""
    resume_checkpoint: str | None = None
    # 仅用 resume_checkpoint 的权重做初始化, 但把 step 重置为 0 (warmup/调度从头).
    # DMD 从某个 SFT ckpt 起步蒸馏时需要: 要权重(teacher+generator), 不要 SFT 的 step 计数.
    resume_reset_step: bool = False
    # DMD 续训: base(transformer.pt) 仍从 resume_checkpoint(SFT ckpt) 读入 generator+teacher,
    # 但 generator LoRA / critic LoRA / action_adaln / step 从这个 dmd_resume 目录读
    # (DMD checkpoint 只存 LoRA, 不含 base)。None 时全部从 resume_checkpoint 读(旧行为)。
    dmd_resume: str | None = None
    stage1_ckpt_dir: str = ""
    stage1_base_transformer: str = ""
    merged_base_dir: str | None = None
    # DMD: real/fake score 模型的 base 权重路径。None 时复用 effective_transformer。
    real_score_model: str | None = None
    vae: str = ""
    gemma: str = ""
    history_encoder: str | None = None
    # DA3 depth model (spatial-memory bank): code repo, HF weights id, HF cache.
    da3_repo: str | None = None
    da3_model: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"
    da3_cache: str | None = None
    # taehv tiny bank decoder weights (only used with run.py --bank-taehv).
    taehv: str = ""
    video_base_dir: str = ""
    annotation_base_dir: str = ""

    @property
    def effective_transformer(self) -> str:
        return self.model or self.continue_transformer or self.transformer or self.base_transformer

    @property
    def effective_vae(self) -> str:
        # VAE lives in the same merged one-file weights as the transformer unless
        # a separate `vae` path is set, so the config need only list it once.
        return self.vae or self.effective_transformer


@dataclass
class DataConfig:
    sources: dict[str, float] = field(default_factory=lambda: {"sekai_game": 1.0})
    use_cache: bool = True
    skip_file_check: bool = True
    abstract_caption_prob: float = 0.0
    require_camera: bool = False
    camera_norm_mode: str = "relic"
    camera_post_relic_scale: float = 0.017
    sekai_game_jsonl: str | None = "sekai_game_walking_lch_smooth_s2p5.jsonl"
    sekai_game_pose_subdir: str | None = "pose_lch_smooth_s2p5"


@dataclass
class SampleConfig:
    height: int = 544
    width: int = 960
    fps: float = 24.0
    temporal_stride: int = 8
    spatial_stride: int = 32


@dataclass
class ConditionLayout:
    type: str = "nearby"
    i2v_prob: float = 0.9
    v2v_prob: float = 0.1
    v2v_ratio_min: float = 0.2
    v2v_ratio_max: float = 0.6


@dataclass
class OutputLayout:
    latent_frames: list[int] = field(default_factory=lambda: [8])
    probs: list[float] = field(default_factory=lambda: [1.0])


@dataclass
class LayoutConfig:
    sink_latent_frames: int = 1
    max_gap_sec: float = 20.0
    history_latent_frames: int = 60
    condition: ConditionLayout = field(default_factory=ConditionLayout)
    output: OutputLayout = field(default_factory=OutputLayout)
    # 当 K=8 时, 从 jsonl 的 valid_k8_starts 字段抽起点; dataloader 同时预 roll gap_steps + cond_mode,
    # 让 K=8 output 精确对齐 video[s, s+8). gap_steps 进 target/cond 的 RoPE PE。
    k8_use_valid_starts: bool = False
    # 同上 K=4 版本, 需要 jsonl 含 valid_k4_starts 字段
    k4_use_valid_starts: bool = False
    # sink_remote: sink frame 从 video 随机一帧加载 (远离 target), 而非 target 紧邻的前一帧。
    # 强迫模型把 sink 当 "global identity anchor", 不能从 sink 局部预测 target → 强化 action 控制学习。
    sink_remote: bool = False
    # sink 离 target 的最小 latent 距离 (= 帧距离 / vae_temporal_factor). 8 latents ≈ 2.7s @ 24fps stride 8.
    sink_remote_min_distance: int = 8


@dataclass
class MemoryConfig:
    compress_t: int = 1
    compress_h: int = 2
    compress_w: int = 2
    lr_compress_t: int = 1
    lr_compress_h: int = 2
    lr_compress_w: int = 2
    gate_init: float = 0.5
    use_self_attn: bool = True
    use_lr_branch: bool = True
    train: bool = True
    drop_prob: float = 0.0


@dataclass
class SpatialMemoryConfig:
    enabled: bool = False
    context_mode: str = "retrieval"
    depth_backend: str = "constant"
    # DA3 path-type settings (repo / weights id / cache) live under paths.da3_*.
    da3_process_res: int = 504
    da3_process_res_method: str = "upper_bound_resize"
    da3_device: str = "auto"
    da3_align_to_input_scale: bool = True
    use_warped_context: bool = True
    retrieval_max_coverage: bool = True
    retrieval_depth_threshold: float = 0.1
    num_context_frames: int = 1
    require_full_context: bool = True
    retrieval_views: int = 1
    cache_stride: int = 1
    skip_recent_latents: int = 1
    downsample: int = 4
    dropout: float = 0.0
    force_all_invalid: bool = False
    constant_depth: float = 1.0
    include_sink: bool = False
    include_nearby: bool = True


@dataclass
class ControlConfig:
    candidates: list[list[str]] = field(default_factory=lambda: [[]])
    probs: list[float] = field(default_factory=lambda: [1.0])
    action_scale: str = "0.14,0.075,0.22,0.17,0.70,0.16"
    action_freq_scale: float = 1000.0
    action_freq_dim_per_axis: int = 32
    action_learning_rate: float | None = None
    action_history_memory: bool = False

    def uses(self, mode: str) -> bool:
        return any(mode in candidate for candidate in self.candidates)


@dataclass
class DriftConfig:
    enabled: bool = True
    noise_mode_prob: float = 0.9
    corrupt_ratio: float = 0.333
    clean_prob: float = 0.1
    downsample_min: float = 0.9
    downsample_max: float = 1.0
    saturation_clean_prob: float = 0.1
    keep_x0: bool = False


@dataclass
class ErrorBankConfig:
    enabled: bool = True
    bank_prob: float = 0.3
    buffer_k: int = 500
    num_grids: int = 50
    warmup_iter: int = 200
    latent_prob: float = 0.9
    clean_prob: float = 0.2
    clean_buffer_update_prob: float = 0.1
    replacement_strategy: str = "l2_batch"
    gamma: float = 1.0
    history_prob: float = 0.9
    spatial_prob: float = 0.9
    nearby_prob: float = 0.9
    modulate_factor: float = 0.2


@dataclass
class AntiDriftConfig:
    drift: DriftConfig = field(default_factory=DriftConfig)
    error_bank: ErrorBankConfig = field(default_factory=ErrorBankConfig)


@dataclass
class DmdConfig:
    """DMD2 风格少步蒸馏配置。第一版只做纯 DMD（不含 GAN / reward / decouple）。

    详见 docs/dmd_distillation_plan.md。三模型：generator=transformer(+LoRA)，
    real/fake score 共用第二个 LTX23Model（base 冻结 + critic LoRA，adapter 开关切换）。
    """

    enabled: bool = False
    # critic:generator 更新比。generator 仅在 global_step % ratio == 0 时更新，critic 每步更新。
    dfake_gen_update_ratio: int = 5
    # 少步生成器的去噪 sigma 列表（σ∈[0,1]，单调递减；末步落到 x0）。对应 HELIOS t=[1000,750,500,250]。
    dmd_sigma_list: list[float] = field(default_factory=lambda: [1.0, 0.75, 0.5, 0.25])
    real_guidance_scale: float = 3.0     # real score 侧 CFG（=1 表示不做 CFG）
    fake_guidance_scale: float = 0.0     # fake score CFG（一般 0，第一版不做）
    critic_lora_rank: int = 128
    # 非标准缩放：scaling=alpha/rank 在初始化时折进 lora_a，forward 不再乘。要求 alpha==rank（见 validate）。
    critic_lora_alpha: float = 128.0
    # critic LoRA 挂载目标。None 时复用 lora.targets。
    critic_lora_targets: list[str] | None = None
    critic_lr: float = 5e-5              # critic 独立学习率
    # 少步采样保 grad 策略：False=随机选一步（HELIOS 默认），True=固定最后一步（CausVid 风格）。
    last_step_only: bool = False
    # real CFG 的 negative prompt（None 时复用 validation.negative_prompt）。
    negative_prompt: str | None = None
    # —— GAN（DMD2 风格,可选；对齐 HELIOS hook 判别器）——
    is_use_gan: bool = False
    gan_start_step: int = 0              # global_step >= 此值才注入 GAN（判别器/对抗项）
    gan_hooks: list[int] = field(default_factory=lambda: [4, 9, 14])  # 判别头挂的 block 索引
    gan_cond_map_dim: int = 768          # 判别头中间通道数
    gan_g_weight: float = 1e-2           # generator 侧对抗项权重（加到 dmd_loss）
    gan_d_weight: float = 1e-2           # critic 侧判别器损失权重
    r1_weight: float = 0.0               # R1 梯度惩罚（real 侧），0=关
    r2_weight: float = 0.0               # R2 梯度惩罚（fake 侧），0=关
    r1_sigma: float = 0.1
    r2_sigma: float = 0.1
    # —— 后置（暂不支持）——
    is_decouple_dmd: bool = False


@dataclass
class TrainingConfig:
    mode: str = "lora"
    adaptive_sigma_shift: bool = False
    adaptive_shift_m_lo: float = 5.0
    adaptive_shift_m_hi: float = 30.0
    adaptive_shift_frame_lo: int = 8
    adaptive_shift_frame_hi: int = 121


@dataclass
class ValidationDatasetConfig:
    source: str = "sekai_game"
    split: str = "val"
    filter: str | None = None
    video_base_dir: str | None = None
    annotation_base_dir: str | None = None


@dataclass
class ValidationLayoutConfig:
    condition: str = "i2v"
    output_latent_frames: int = 8
    condition_latent_frames: int = 1
    history_latent_frames: int | None = None
    max_gap_sec: float | None = None
    height: int | None = None
    width: int | None = None


@dataclass
class ValidationModeConfig:
    dataset: ValidationDatasetConfig = field(default_factory=ValidationDatasetConfig)
    layout: ValidationLayoutConfig = field(default_factory=ValidationLayoutConfig)
    control: list[str] = field(default_factory=list)
    rollout_rounds: int = 1
    use_memory: bool = True
    action_cfg_scale: float = 1.0


@dataclass
class TtcConfig:
    """Pathwise Test-Time Correction (arXiv:2602.05871). Toggled by the run.py
    --ttc switch; these knobs live in config (defaults = the chosen 'L3s1.2').
    levels: denoise target noise levels (sigma*1000, 0..1000) to correct at.
    strength: pull toward the initial-frame anchor (1.0 = plain TTC; >1 over-corrects).
    ref_action: keep the camera action in the reference pass (False = anchor pure
        appearance/style, geometry restored by the resume step)."""
    levels: list[int] = field(default_factory=lambda: [750, 500, 250])
    strength: float = 1.2
    ref_action: bool = False


@dataclass
class ValidationConfig:
    enabled: bool = False
    before_train: bool = False
    interval: int = 200
    first_step: int = 200
    max_samples: int = 1
    dynamic_rounds: bool = False
    sampling_steps: int = 30
    scheduler: str = "shift"
    cfg_scale: float = 3.0
    negative_prompt: str = LTX_DEFAULT_NEGATIVE_PROMPT
    stg_scale: float = 1.0
    stg_blocks: list[int] = field(default_factory=lambda: [28])
    rescale_scale: float = 0.7
    step_dir_suffix: str = ""
    save_videos: bool = True
    save_joystick: bool = True
    video_history_latent_frames: int | None = None
    ttc: TtcConfig = field(default_factory=TtcConfig)
    modes: dict[str, ValidationModeConfig] = field(default_factory=dict)


@dataclass
class LoraConfig:
    enabled: bool = True
    train: bool = True
    rank: int = 128
    alpha: int = 128
    targets: list[str] = field(default_factory=lambda: [
        "attn1.to_q",
        "attn1.to_k",
        "attn1.to_v",
        "attn1.to_out.0",
        "attn2.to_q",
        "attn2.to_k",
        "attn2.to_v",
        "attn2.to_out.0",
        "ff.net.0.proj",
        "ff.net.2",
    ])


@dataclass
class OptimizerConfig:
    batch_size: int = 1
    lr: float = 5e-5
    weight_decay: float = 0.001
    epochs: int = 500
    max_grad_norm: float = 5.0
    warmup_steps: int = 100
    checkpoint_steps: int = 200
    max_checkpoints: int = 5
    log_steps: int = 1
    max_steps: int | None = None


@dataclass
class RuntimeConfig:
    dtype: str = "bf16"
    attention_type: str = "flash_attention_3"
    gradient_checkpointing: bool = True
    vae_chunk_size: int = 33
    vae_decode_chunk_latents: int | None = None
    vae_decode_overlap_latents: int = 0  # >0 => seamless overlap-tiling decode (lossless)
    dataloader_workers: int = 2
    dataloader_pin_memory: bool = False
    dataloader_prefetch_factor: int = 2
    fsdp: bool = True
    norm_by_fps: bool = True
    norm_by_max_frames: bool = True
    positional_embedding_max_pos: str = "20,2048,2048"


@dataclass
class TrainConfig:
    run: RunConfig = field(default_factory=RunConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    spatial_memory: SpatialMemoryConfig = field(default_factory=SpatialMemoryConfig)
    control: ControlConfig = field(default_factory=ControlConfig)
    anti_drift: AntiDriftConfig = field(default_factory=AntiDriftConfig)
    dmd: DmdConfig = field(default_factory=DmdConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "TrainConfig":
        cfg = cls()
        _update_dataclass(cfg, values)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.training.mode not in {"lora", "sft"}:
            raise ValueError("training.mode must be 'lora' or 'sft'")
        if self.training.adaptive_shift_m_lo <= 0 or self.training.adaptive_shift_m_hi <= 0:
            raise ValueError("training adaptive shift m values must be > 0")
        if self.training.adaptive_shift_frame_lo <= 0 or self.training.adaptive_shift_frame_hi <= 0:
            raise ValueError("training adaptive shift frame anchors must be > 0")
        _validate_control(self.control)
        _validate_validation(self.validation)
        if self.layout.condition.type != "nearby":
            raise ValueError("first rewrite only supports layout.condition.type=nearby")
        if self.layout.sink_latent_frames < 0:
            raise ValueError("layout.sink_latent_frames must be >= 0")
        if self.layout.history_latent_frames < 0:
            raise ValueError("layout.history_latent_frames must be >= 0")
        if self.layout.k8_use_valid_starts:
            # valid_k8_starts 现支持 t2v + i2v 混训: dataloader 预 roll gap_steps + cond_mode,
            # 让 K=8 output 精确对齐 video[s, s+8); gap_steps 同时进 RoPE PE。
            # 不支持: history (布局不同), v2v (cond_end 可变)。
            if self.layout.history_latent_frames != 0:
                raise ValueError(
                    "layout.k8_use_valid_starts requires history_latent_frames=0"
                )
            if self.layout.condition.v2v_prob > 0:
                raise ValueError(
                    "layout.k8_use_valid_starts requires v2v_prob=0 (cond_end 可变, 无法对齐)"
                )
            if 8 not in [int(k) for k in self.layout.output.latent_frames]:
                raise ValueError(
                    "layout.k8_use_valid_starts requires 8 in layout.output.latent_frames"
                )
        if self.layout.k4_use_valid_starts:
            # K=4 版本: 跟 k8_use_valid_starts 完全对称
            if self.layout.history_latent_frames != 0:
                raise ValueError(
                    "layout.k4_use_valid_starts requires history_latent_frames=0"
                )
            if self.layout.condition.v2v_prob > 0:
                raise ValueError(
                    "layout.k4_use_valid_starts requires v2v_prob=0 (cond_end 可变, 无法对齐)"
                )
            if 4 not in [int(k) for k in self.layout.output.latent_frames]:
                raise ValueError(
                    "layout.k4_use_valid_starts requires 4 in layout.output.latent_frames"
                )
        if self.layout.sink_remote:
            if self.layout.sink_latent_frames < 1:
                raise ValueError(
                    "layout.sink_remote requires sink_latent_frames >= 1 "
                    "(没有 sink 槽位可填远端帧)"
                )
            if self.layout.sink_remote_min_distance < 0:
                raise ValueError("layout.sink_remote_min_distance must be >= 0")
        if not 0.0 <= float(self.memory.drop_prob) <= 1.0:
            raise ValueError("memory.drop_prob must be in [0, 1]")
        if self.spatial_memory.enabled:
            if self.spatial_memory.context_mode not in {"retrieval", "target_prefix_pixels"}:
                raise ValueError("spatial_memory.context_mode must be one of: retrieval, target_prefix_pixels")
            if self.spatial_memory.depth_backend not in {"constant", "metadata", "da3"}:
                raise ValueError("spatial_memory.depth_backend must be one of: constant, metadata, da3")
            if self.spatial_memory.depth_backend == "da3" and not self.paths.da3_model:
                raise ValueError("paths.da3_model must be set when spatial_memory.depth_backend=da3")
            if self.spatial_memory.da3_process_res <= 0:
                raise ValueError("spatial_memory.da3_process_res must be > 0")
            if self.spatial_memory.retrieval_depth_threshold < 0:
                raise ValueError("spatial_memory.retrieval_depth_threshold must be >= 0")
            if self.spatial_memory.num_context_frames <= 0:
                raise ValueError("spatial_memory.num_context_frames must be > 0")
            if self.spatial_memory.retrieval_views <= 0:
                raise ValueError("spatial_memory.retrieval_views must be > 0")
            if self.spatial_memory.cache_stride <= 0:
                raise ValueError("spatial_memory.cache_stride must be > 0")
            if self.spatial_memory.skip_recent_latents < 0:
                raise ValueError("spatial_memory.skip_recent_latents must be >= 0")
            if self.spatial_memory.downsample <= 0:
                raise ValueError("spatial_memory.downsample must be > 0")
            if not 0.0 <= float(self.spatial_memory.dropout) <= 1.0:
                raise ValueError("spatial_memory.dropout must be in [0, 1]")
            if self.spatial_memory.constant_depth <= 0:
                raise ValueError("spatial_memory.constant_depth must be > 0")
        if len(self.layout.output.latent_frames) != len(self.layout.output.probs):
            raise ValueError("layout.output.latent_frames/probs length mismatch")
        if self.optimizer.checkpoint_steps <= 0:
            raise ValueError("optimizer.checkpoint_steps must be > 0")
        if self.optimizer.max_checkpoints < 0:
            raise ValueError("optimizer.max_checkpoints must be >= 0")
        if not self.paths.effective_transformer and not self.paths.stage1_ckpt_dir:
            raise ValueError("set paths.continue_transformer, paths.transformer, paths.base_transformer, or paths.stage1_ckpt_dir")
        if self.dmd.enabled:
            _validate_dmd(self.dmd)


def _validate_dmd(dmd: "DmdConfig") -> None:
    if dmd.dfake_gen_update_ratio < 1:
        raise ValueError("dmd.dfake_gen_update_ratio must be >= 1")
    if not dmd.dmd_sigma_list:
        raise ValueError("dmd.dmd_sigma_list must be non-empty")
    sigmas = [float(s) for s in dmd.dmd_sigma_list]
    if any(not (0.0 <= s <= 1.0) for s in sigmas):
        raise ValueError("dmd.dmd_sigma_list values must be in [0, 1]")
    if any(sigmas[i] <= sigmas[i + 1] for i in range(len(sigmas) - 1)):
        raise ValueError("dmd.dmd_sigma_list must be strictly decreasing (σ=1 noise → σ small)")
    if dmd.critic_lora_rank <= 0:
        raise ValueError("dmd.critic_lora_rank must be > 0")
    if float(dmd.critic_lora_alpha) != float(dmd.critic_lora_rank):
        # LoRAForwardManager 把 scaling=alpha/rank 折进 lora_a 初值、forward 不再乘（非标准），
        # 且 save() 存的是含 scaling 的权重。alpha!=rank 会让缩放语义偏离标准 LoRA、且跨工具
        # 加载双重缩放。第一版强制相等，避免静默错误（确需不等请改这里）。
        raise ValueError(
            f"dmd.critic_lora_alpha ({dmd.critic_lora_alpha}) must equal critic_lora_rank "
            f"({dmd.critic_lora_rank}); non-standard scaling otherwise (see docs/dmd_distillation_plan.md §4.1)"
        )
    if dmd.real_guidance_scale < 1.0:
        raise ValueError("dmd.real_guidance_scale must be >= 1 (1 = no CFG)")
    if dmd.is_decouple_dmd:
        raise ValueError("dmd.is_decouple_dmd is not supported yet")
    if dmd.is_use_gan:
        if not dmd.gan_hooks:
            raise ValueError("dmd.is_use_gan requires a non-empty dmd.gan_hooks (block indices for discriminator heads)")
        if any(int(h) < 0 for h in dmd.gan_hooks):
            raise ValueError("dmd.gan_hooks must be non-negative block indices")
        if dmd.gan_start_step < 0:
            raise ValueError("dmd.gan_start_step must be >= 0")
        if dmd.gan_cond_map_dim <= 0:
            raise ValueError("dmd.gan_cond_map_dim must be > 0")


def _update_dataclass(obj: Any, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not hasattr(obj, key):
            raise ValueError(f"unknown config key {type(obj).__name__}.{key}")
        current = getattr(obj, key)
        if key == "modes" and isinstance(obj, ValidationConfig):
            if not isinstance(value, dict):
                raise ValueError("validation.modes must be a mapping")
            current.clear()
            for mode_name, mode_value in value.items():
                if not isinstance(mode_name, str) or not mode_name:
                    raise ValueError("validation mode names must be non-empty strings")
                if not isinstance(mode_value, dict):
                    raise ValueError(f"validation.modes.{mode_name} must be a mapping")
                mode_cfg = ValidationModeConfig()
                _update_dataclass(mode_cfg, mode_value)
                current[mode_name] = mode_cfg
            continue
        if key == "dataset" and isinstance(obj, ValidationModeConfig) and isinstance(value, str):
            current.source = value
            continue
        if hasattr(current, "__dataclass_fields__"):
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a mapping")
            _update_dataclass(current, value)
        else:
            setattr(obj, key, value)


def _validate_control(control: ControlConfig) -> None:
    if len(control.candidates) != len(control.probs):
        raise ValueError("control.candidates/probs length mismatch")
    if not control.candidates:
        raise ValueError("control.candidates must contain at least one candidate")
    allowed = {"action"}
    for candidate in control.candidates:
        if not isinstance(candidate, list):
            raise ValueError("each control candidate must be a list, e.g. ['action'] or []")
        unknown = set(candidate) - allowed
        if unknown:
            raise ValueError(f"unknown control modes: {sorted(unknown)}; allowed={sorted(allowed)}")
        if len(candidate) != len(set(candidate)):
            raise ValueError(f"duplicated modes in control candidate: {candidate}")
    total = sum(float(p) for p in control.probs)
    if total <= 0 or any(float(p) < 0 for p in control.probs):
        raise ValueError("control.probs must be non-negative and sum to > 0")


def _validate_validation(validation: ValidationConfig) -> None:
    if not validation.enabled:
        return
    if validation.interval <= 0:
        raise ValueError("validation.interval must be > 0")
    if validation.first_step < 0:
        raise ValueError("validation.first_step must be >= 0")
    if validation.max_samples <= 0:
        raise ValueError("validation.max_samples must be > 0")
    if validation.sampling_steps <= 0:
        raise ValueError("validation.sampling_steps must be > 0")
    if validation.scheduler not in {"shift", "linear_quadratic", "uniform"}:
        raise ValueError("validation.scheduler must be one of: shift, linear_quadratic, uniform")
    if validation.cfg_scale < 1.0:
        raise ValueError("validation.cfg_scale must be >= 1")
    if validation.stg_scale < 0.0:
        raise ValueError("validation.stg_scale must be >= 0")
    if validation.rescale_scale < 0.0:
        raise ValueError("validation.rescale_scale must be >= 0")
    if not validation.modes:
        raise ValueError("validation.modes must contain at least one mode when validation.enabled=true")
    for mode_name, mode in validation.modes.items():
        if mode.rollout_rounds <= 0:
            raise ValueError(f"validation.modes.{mode_name}.rollout_rounds must be > 0")
        if float(mode.action_cfg_scale) < 1.0:
            raise ValueError(f"validation.modes.{mode_name}.action_cfg_scale must be >= 1")
        if mode.layout.condition not in {"hc", "i2v", "v2v"}:
            raise ValueError(f"validation.modes.{mode_name}.layout.condition must be one of: hc, i2v, v2v")
        if mode.layout.output_latent_frames <= 0:
            raise ValueError(f"validation.modes.{mode_name}.layout.output_latent_frames must be > 0")
        if mode.layout.condition_latent_frames < 0:
            raise ValueError(f"validation.modes.{mode_name}.layout.condition_latent_frames must be >= 0")
        unknown = set(mode.control) - {"action"}
        if unknown:
            raise ValueError(f"unknown validation.modes.{mode_name}.control modes: {sorted(unknown)}; allowed=['action']")
        if len(mode.control) != len(set(mode.control)):
            raise ValueError(f"duplicated modes in validation.modes.{mode_name}.control: {mode.control}")
