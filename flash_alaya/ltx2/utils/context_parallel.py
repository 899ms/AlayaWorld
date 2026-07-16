# -*- coding: utf-8 -*-
# Last modified: rsh 2026-03-26 02:35:37
"""
Context Parallel (Ulysses-style) Implementation for Long Video Training

基于 LingBot-World 论文的 Ulysses 风格 Context Parallel 实现:
- 将输入张量沿时间（序列）维度分区到多个 GPU
- 在 attention 计算时使用 all-to-all 通信重新分配激活
- 每个设备本地计算其序列分片上的 attention

参考: https://arxiv.org/abs/2601.20540 Section 3.3.3
"""

import datetime
import torch
import torch.distributed as dist
from typing import Optional, Tuple
from dataclasses import dataclass

from flash_alaya.ltx2.utils.parallel_states import nccl_info

# 长视频训练需要更长的超时时间（2小时）
_CP_TIMEOUT = datetime.timedelta(hours=2)


@dataclass
class ContextParallelConfig:
    """Context Parallel 配置"""
    enabled: bool = False
    cp_size: int = 1  # Context Parallel 的 GPU 数量
    cp_rank: int = 0  # 当前 GPU 在 CP group 中的 rank
    cp_group: Optional[dist.ProcessGroup] = None


# 全局 CP 配置
_cp_config = ContextParallelConfig()


def initialize_context_parallel(cp_size: int = 1):
    """
    初始化 Context Parallel 组
    
    Args:
        cp_size: Context Parallel 的 GPU 数量。如果为 1 则禁用 CP。
    """
    global _cp_config
    
    if cp_size <= 1:
        _cp_config = ContextParallelConfig(enabled=False, cp_size=1, cp_rank=0, cp_group=None)
        return
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    assert world_size % cp_size == 0, f"world_size ({world_size}) must be divisible by cp_size ({cp_size})"
    
    # 创建 CP 组（使用 2 小时超时，用于长视频训练）
    num_cp_groups = world_size // cp_size
    for i in range(num_cp_groups):
        ranks = list(range(i * cp_size, (i + 1) * cp_size))
        group = dist.new_group(ranks, timeout=_CP_TIMEOUT)
        if rank in ranks:
            _cp_config = ContextParallelConfig(
                enabled=True,
                cp_size=cp_size,
                cp_rank=rank - i * cp_size,
                cp_group=group,
            )
    
    if rank == 0:
        print(f"[Context Parallel] Initialized with cp_size={cp_size}, num_groups={num_cp_groups}")


def warmup_context_parallel(hidden_dim: int = 4096, num_heads: int = 32,
                            seq_len: int = 1024, device: torch.device = None):
    """
    NCCL 通信预热 (对齐 FastVideo 策略)

    在第一次 forward 之前执行 dummy all-to-all，强制 NCCL 完成
    communicator 初始化、ring/tree 拓扑建立。避免首步 forward 时
    NCCL lazy init 导致部分 rank 超时。

    Args:
        hidden_dim: 模型 hidden dimension
        num_heads: attention head 数
        seq_len: dummy 序列长度
        device: GPU 设备
    """
    if not _cp_config.enabled:
        return

    if device is None:
        device = torch.cuda.current_device()

    rank = dist.get_rank()
    head_dim = hidden_dim // num_heads
    cp_size = _cp_config.cp_size
    group = _cp_config.cp_group

    if rank == 0:
        print(f"[CP Warmup] 开始 NCCL 通信预热: cp_size={cp_size}, "
              f"hidden={hidden_dim}, heads={num_heads}, seq={seq_len}")

    # Pattern 1: scatter heads, gather sequence (attention 前)
    dummy = torch.zeros(1, seq_len // cp_size, num_heads, head_dim, device=device, dtype=torch.bfloat16)
    _ = _all_to_all_single(dummy, scatter_dim=2, gather_dim=1, group=group)

    # Pattern 2: scatter sequence, gather heads (attention 后)
    dummy2 = torch.zeros(1, seq_len, num_heads // cp_size, head_dim, device=device, dtype=torch.bfloat16)
    _ = _all_to_all_single(dummy2, scatter_dim=1, gather_dim=2, group=group)

    # Pattern 3: all-gather (用于 gradient sync)
    dummy3 = torch.zeros(seq_len // cp_size, device=device, dtype=torch.bfloat16)
    gathered = [torch.zeros_like(dummy3) for _ in range(cp_size)]
    dist.all_gather(gathered, dummy3, group=group)

    # 全局 barrier 确保所有 rank 完成预热
    dist.barrier()

    del dummy, dummy2, dummy3, gathered
    torch.cuda.empty_cache()

    if rank == 0:
        print(f"[CP Warmup] NCCL 通信预热完成")


def get_cp_config() -> ContextParallelConfig:
    """获取 Context Parallel 配置"""
    return _cp_config


def destroy_context_parallel():
    """销毁 Context Parallel 组"""
    global _cp_config
    _cp_config = ContextParallelConfig()


def _all_to_all_single(
    input_: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group: dist.ProcessGroup,
) -> torch.Tensor:
    """
    All-to-all 通信（用于 Ulysses-style CP）
    
    将 scatter_dim 维度分散，在 gather_dim 维度聚合
    
    Args:
        input_: 输入张量
        scatter_dim: 分散的维度
        gather_dim: 聚合的维度
        group: 进程组
    
    Returns:
        重分布后的张量
    """
    world_size = dist.get_world_size(group)
    
    if world_size == 1:
        return input_
    
    # 将输入沿 scatter_dim 分割 (chunk returns views, must be contiguous for NCCL)
    input_list = [chunk.contiguous() for chunk in torch.chunk(input_, world_size, dim=scatter_dim)]

    # 创建输出缓冲区
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    
    # 执行 all-to-all
    dist.all_to_all(output_list, input_list, group=group)
    
    # 沿 gather_dim 拼接
    return torch.cat(output_list, dim=gather_dim)


class _SeqAllToAll(torch.autograd.Function):
    """
    序列维度的 All-to-All 通信（支持自动梯度）
    
    用于 Ulysses-style Context Parallel
    """
    
    @staticmethod
    def forward(
        ctx,
        input_: torch.Tensor,
        scatter_dim: int,
        gather_dim: int,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.group = group
        return _all_to_all_single(input_, scatter_dim, gather_dim, group)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None, None]:
        # 反向传播时交换 scatter 和 gather 维度
        grad_input = _all_to_all_single(
            grad_output, 
            ctx.gather_dim, 
            ctx.scatter_dim, 
            ctx.group
        )
        return grad_input, None, None, None


def seq_all_to_all(
    input_: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    """
    序列维度的 All-to-All 通信
    
    Args:
        input_: 输入张量
        scatter_dim: 分散的维度
        gather_dim: 聚合的维度
    
    Returns:
        重分布后的张量
    """
    if not _cp_config.enabled:
        return input_
    return _SeqAllToAll.apply(input_, scatter_dim, gather_dim, _cp_config.cp_group)


def scatter_sequence(input_: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    将序列分散到所有 CP GPU
    
    Args:
        input_: [B, S, ...] 或 [B, ..., S, ...]
        dim: 序列维度
    
    Returns:
        分散后的张量 [B, S//cp_size, ...]
    """
    if not _cp_config.enabled:
        return input_
    
    # 确保序列长度可被 cp_size 整除
    seq_len = input_.shape[dim]
    assert seq_len % _cp_config.cp_size == 0, \
        f"Sequence length ({seq_len}) must be divisible by cp_size ({_cp_config.cp_size})"
    
    # 分割并取当前 rank 的部分
    chunks = torch.chunk(input_, _cp_config.cp_size, dim=dim)
    return chunks[_cp_config.cp_rank].contiguous()


def gather_sequence(input_: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """
    从所有 CP GPU 聚合序列
    
    Args:
        input_: 当前 GPU 的序列分片
        dim: 序列维度
    
    Returns:
        聚合后的完整序列
    """
    if not _cp_config.enabled:
        return input_
    
    # All-gather
    world_size = _cp_config.cp_size
    gathered = [torch.empty_like(input_) for _ in range(world_size)]
    dist.all_gather(gathered, input_.contiguous(), group=_cp_config.cp_group)
    
    return torch.cat(gathered, dim=dim)


class _GatherForward(torch.autograd.Function):
    """前向聚合，反向分散（用于损失计算等）"""
    
    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int) -> torch.Tensor:
        ctx.dim = dim
        return gather_sequence(input_, dim)
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return scatter_sequence(grad_output, ctx.dim), None


def gather_for_loss(input_: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """前向聚合用于损失计算（反向自动分散梯度）"""
    if not _cp_config.enabled:
        return input_
    return _GatherForward.apply(input_, dim)


def apply_ulysses_attention(
    q: torch.Tensor,
    k: torch.Tensor, 
    v: torch.Tensor,
    heads: int,
    attention_fn,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    应用 Ulysses-style Context Parallel 的 Attention
    
    1. All-to-all: scatter sequence dim, gather head dim
    2. 执行标准 attention（每个 GPU 处理全部序列的部分 heads）
    3. All-to-all: scatter head dim, gather sequence dim
    
    Args:
        q: Query [B, S, H*D]
        k: Key [B, S, H*D]  
        v: Value [B, S, H*D]
        heads: 注意力头数
        attention_fn: 注意力函数
        mask: 可选的 attention mask
    
    Returns:
        注意力输出 [B, S, H*D]
    """
    if not _cp_config.enabled:
        return attention_fn(q, k, v, heads, mask)
    
    B, S, HD = q.shape
    H = heads
    D = HD // H
    cp_size = _cp_config.cp_size
    
    # 确保 heads 可被 cp_size 整除
    assert H % cp_size == 0, f"Number of heads ({H}) must be divisible by cp_size ({cp_size})"
    
    # 1. Reshape to expose head dimension: [B, S, H, D]
    q = q.view(B, S, H, D)
    k = k.view(B, S, H, D)
    v = v.view(B, S, H, D)
    
    # 2. All-to-all: scatter heads (dim=2), gather sequence (dim=1)
    # [B, S_local, H, D] -> [B, S_full, H//cp_size, D]
    # 每个 GPU 获得全部序列、部分 heads
    q = seq_all_to_all(q, scatter_dim=2, gather_dim=1)
    k = seq_all_to_all(k, scatter_dim=2, gather_dim=1)
    v = seq_all_to_all(v, scatter_dim=2, gather_dim=1)
    
    # 3. Reshape back for attention: [B, S*cp_size, (H//cp_size)*D]
    _, S_full, H_local, _ = q.shape
    q = q.view(B, S_full, H_local * D)
    k = k.view(B, S_full, H_local * D)
    v = v.view(B, S_full, H_local * D)
    
    # 4. Execute attention with local heads
    # mask 也需要调整
    if mask is not None:
        # 聚合 mask 的序列维度（如果需要）
        mask = gather_sequence(mask, dim=-1) if mask.shape[-1] == S else mask
    
    out = attention_fn(q, k, v, H_local, mask)
    
    # 5. Reshape for all-to-all back: [B, S_full, H_local, D]
    out = out.view(B, S_full, H_local, D)
    
    # 6. All-to-all: scatter sequence (dim=1), gather heads (dim=2)
    # [B, S_full, H//cp_size, D] -> [B, S_local, H, D]
    out = seq_all_to_all(out, scatter_dim=1, gather_dim=2)
    
    # 7. Reshape to original format: [B, S, H*D]
    out = out.view(B, S, HD)
    
    return out


def pad_to_cp_divisible(tensor: torch.Tensor, dim: int = 1) -> Tuple[torch.Tensor, int]:
    """
    将张量 pad 到可被 cp_size 整除
    
    Args:
        tensor: 输入张量
        dim: 要 pad 的维度
    
    Returns:
        (padded_tensor, original_length)
    """
    if not _cp_config.enabled:
        return tensor, tensor.shape[dim]
    
    original_length = tensor.shape[dim]
    cp_size = _cp_config.cp_size
    
    if original_length % cp_size == 0:
        return tensor, original_length
    
    pad_length = cp_size - (original_length % cp_size)
    
    # 创建 pad 规格（PyTorch pad 从最后一维开始）
    pad_spec = [0] * (2 * tensor.ndim)
    # dim 对应的 pad 位置（从右到左）
    pad_idx = 2 * (tensor.ndim - 1 - dim)
    pad_spec[pad_idx + 1] = pad_length  # 在该维度末尾 pad
    
    padded = torch.nn.functional.pad(tensor, pad_spec, mode='constant', value=0)
    return padded, original_length


def unpad_from_cp(tensor: torch.Tensor, original_length: int, dim: int = 1) -> torch.Tensor:
    """
    移除 CP padding
    
    Args:
        tensor: padded 张量
        original_length: 原始长度
        dim: pad 的维度
    
    Returns:
        unpadded 张量
    """
    if not _cp_config.enabled or tensor.shape[dim] == original_length:
        return tensor
    
    # 使用 narrow 截取原始长度
    return tensor.narrow(dim, 0, original_length)


# =============================================================================
# 辅助函数：用于训练循环
# =============================================================================

def prepare_cp_inputs(
    video_latent: torch.Tensor,
    audio_latent: Optional[torch.Tensor] = None,
    video_positions: Optional[torch.Tensor] = None,
    audio_positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor], dict]:
    """
    准备 Context Parallel 的输入
    
    将 latent 和 positions 分片到各 GPU
    
    Args:
        video_latent: [B, T, ...] 视频 latent（已 patchified）
        audio_latent: [B, T, ...] 音频 latent（已 patchified）
        video_positions: [B, 1, T, D] 视频位置编码
        audio_positions: [B, 1, T, D] 音频位置编码
    
    Returns:
        分片后的 (video_latent, audio_latent, video_positions, audio_positions, metadata)
    """
    metadata = {
        'video_original_length': video_latent.shape[1] if video_latent is not None else 0,
        'audio_original_length': audio_latent.shape[1] if audio_latent is not None else 0,
    }
    
    if not _cp_config.enabled:
        return video_latent, audio_latent, video_positions, audio_positions, metadata
    
    # 分片 video latent (dim=1 是序列维度)
    if video_latent is not None:
        video_latent = scatter_sequence(video_latent, dim=1)
    
    # 分片 audio latent
    if audio_latent is not None:
        audio_latent = scatter_sequence(audio_latent, dim=1)
    
    # 分片 positions (dim=2 是序列维度)
    if video_positions is not None:
        video_positions = scatter_sequence(video_positions, dim=2)
    
    if audio_positions is not None:
        audio_positions = scatter_sequence(audio_positions, dim=2)
    
    return video_latent, audio_latent, video_positions, audio_positions, metadata


def gather_cp_outputs(
    video_output: torch.Tensor,
    audio_output: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    聚合 Context Parallel 的输出
    
    Args:
        video_output: 视频输出分片
        audio_output: 音频输出分片
    
    Returns:
        聚合后的 (video_output, audio_output)
    """
    if not _cp_config.enabled:
        return video_output, audio_output
    
    video_output = gather_sequence(video_output, dim=1)
    if audio_output is not None:
        audio_output = gather_sequence(audio_output, dim=1)
    
    return video_output, audio_output


def get_cp_world_size() -> int:
    """获取 CP world size"""
    return _cp_config.cp_size if _cp_config.enabled else 1


def get_cp_rank() -> int:
    """获取 CP rank"""
    return _cp_config.cp_rank if _cp_config.enabled else 0


def is_cp_enabled() -> bool:
    """检查 CP 是否启用"""
    return _cp_config.enabled


def compute_cp_divisible_frames(frames: int, cp_size: int, temporal_stride: int = 8) -> int:
    """
    计算可被 CP 和 temporal_stride 整除的帧数
    
    LTX2 VAE 要求 (frames - 1) % 8 == 0
    CP 要求 latent_frames % cp_size == 0
    
    Args:
        frames: 目标帧数
        cp_size: CP size
        temporal_stride: VAE 时间步长
    
    Returns:
        调整后的帧数
    """
    # LTX2 VAE: latent_frames = (frames - 1) // 8 + 1
    # 确保 (frames - 1) % 8 == 0
    frames = ((frames - 1) // temporal_stride) * temporal_stride + 1
    
    # 确保 latent_frames % cp_size == 0
    latent_frames = (frames - 1) // temporal_stride + 1
    if latent_frames % cp_size != 0:
        latent_frames = ((latent_frames // cp_size) + 1) * cp_size
        frames = (latent_frames - 1) * temporal_stride + 1
    
    return frames
