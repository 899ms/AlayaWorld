"""FlashAlaya: standalone streaming inference pipeline for the alaya world model.

FlashDreams-style execution contract, self-contained (no alaya.trainer):

    engine = InferenceEngine(cfg); engine.setup()
    pipe   = FlashAlayaPipeline(engine)
    cache  = pipe.initialize_cache(...) -> [generate -> finalize] x rounds -> decode

Layout:
    run.py        the single entry point: python -m flash_alaya.run --input <scene>
    utils/        pipeline internals
        engine        model loading (meta-init DiT, VAE, Gemma, history enc, DA3) + basic ops
        pipeline      initialize_cache / generate / finalize / decode
        cache         RolloutCache (one-shot conditions + AR state)
        conditioning  RoPE index grids, t-offsets, sigma schedule, action control
        spatial       spatial memory bank (DA3 depth + forward warp)
        taehv         optional streaming tiny decoder (--taehv-decode)
    ltx2/         LTX2 model stack (modules / configs / utils incl. streaming VAE)
    alaya/        world-model extensions (config, control, memory, model loaders)
"""
from flash_alaya.utils.cache import RolloutCache
from flash_alaya.utils.engine import InferenceEngine
from flash_alaya.utils.pipeline import FlashAlayaPipeline

__all__ = ["FlashAlayaPipeline", "InferenceEngine", "RolloutCache"]
