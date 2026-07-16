from __future__ import annotations

import torch.nn as nn


def maybe_enable_gradient_checkpointing(model: nn.Module, enabled: bool) -> None:
    if not enabled:
        return
    if hasattr(model, "enable_gradient_checkpointing"):
        model.enable_gradient_checkpointing()
        return
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
