from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn


@dataclass
class ModelComponents:
    transformer: nn.Module
    vae_encoder: object
    vae_decoder: nn.Module
    text_encoder: nn.Module
    encode_text: Callable

