"""LTX History Encoder (Frame Preservation 论文 stage 1 pretrain 用)
=========================================================================

实现参考:
    - arxiv 2512.23851 (Pretraining Frame Preservation for Lightweight
      Autoregressive Video History Embedding)
    - LongWorld 仓库 (Wan 2.2 14B 实现)

输入 (双分支: HR + LR):
    latent: [B, 128, T_h, H_h, W_h]      LTX VAE 输出, 32x spatial / 8x temporal
                                          典型 20s history: T_h=60, H_h=17, W_h=30

输出:
    mem_tokens: [B, N_mem_total, 4096]   (HR mem tokens 在前, LR mem tokens 在后)
    mem_indices_grid: [B, 3, N_mem_total]
                                          (T, H, W) 在原 latent 坐标系, 供主 DiT 算 RoPE

双分支设计:
    HR 分支 (重表征):
        Conv3d 6 级压缩 + 3D self-attn + 1×1 投影 + output_gate
        默认压缩 (2, 2, 2): (60, 17, 30) → (30, 8, 15) = 3600 token
    LR 分支 (粗表征, 论文 Without LR ablation 显示去掉它 PSNR 降 1.7 dB):
        avg_pool3d 时空下采样 + Conv3d 1×1 投影 + output_gate
        默认压缩 (2, 4, 4): (60, 17, 30) → (30, 4, 7) = 840 token

总 mem token: 3600 + 840 = 4440 (跟 LongWorld 6240 同量级)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Causal 3D conv: T 维全 pad 在左侧 (右侧 0 pad), 保证 token at frame t 不依赖 frame > t.
# H, W 维双侧对称 pad. 实现跟 wan.modules.vae2_1.CausalConv3d 一致, inline 避免 wan __init__ 副作用.
class CausalConv3d(nn.Conv3d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],  # W 双侧
            self.padding[1],
            self.padding[1],  # H 双侧
            2 * self.padding[0],
            0,  # T 全在左, 右侧 0
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding)
        return super().forward(x)


class _Conv3dBlock(nn.Module):
    """CausalConv3d + SiLU (跟 LongWorld 一致, 不带 GroupNorm).
    T 维因果 (pad 全在左侧), H/W 维对称 pad."""

    def __init__(self, in_ch, out_ch, stride=(1, 1, 1), kernel_size=(3, 3, 3)):
        super().__init__()
        padding = tuple(k // 2 for k in kernel_size)
        self.conv = CausalConv3d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


class _SelfAttention3D(nn.Module):
    """3D self-attention over flattened (T, H, W) tokens.
    T 维 causal mask: token at frame t 只能 attend 到 frame ≤ t (空间内自由).
    """

    def __init__(self, dim, num_heads=8):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        spatial = H * W
        x_flat = x.permute(0, 2, 3, 4, 1).reshape(B, T * spatial, C)
        residual = x_flat
        x_flat = self.norm(x_flat)
        qkv = self.qkv(x_flat).reshape(B, T * spatial, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # T 维 causal mask: token i 在 frame t_i = i // spatial,
        #   允许 attend to j 当 t_j <= t_i (同 frame 内全互通, 跨 frame 只看更早).
        t_indices = torch.arange(T * spatial, device=x.device) // spatial  # [T*spatial]
        causal_mask = t_indices.unsqueeze(0) <= t_indices.unsqueeze(1)  # [seq, seq]
        attn_mask = torch.zeros(T * spatial, T * spatial, device=x.device, dtype=q.dtype)
        attn_mask = attn_mask.masked_fill(~causal_mask, float("-inf"))

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, T * spatial, C)
        attn_out = self.proj(attn_out)
        out = residual + attn_out
        out = out.reshape(B, T, H, W, C).permute(0, 4, 1, 2, 3).contiguous()
        return out


class VideoHistoryEncoder(nn.Module):
    """双分支 video history encoder.

    Args:
        in_channels: 输入 latent channel (LTX = 128)
        out_channels: 输出 channel = LTX inner_dim (= 4096)
        compress_t/h/w: HR 分支压缩率 (默认 (2, 2, 2))
        lr_compress_t/h/w: LR 分支额外下采样率 (默认 (2, 4, 4))
        gate_init: HR / LR output gate 初值 (默认 0.0, zero-init)
        use_self_attn: HR 分支是否在压缩后接 3D self-attn (默认 True)
        use_lr_branch: 是否启用 LR 分支 (默认 True). 关掉只用 HR 单分支
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 4096,
        compress_t: int = 2,
        compress_h: int = 2,
        compress_w: int = 2,
        lr_compress_t: int = 2,
        lr_compress_h: int = 4,
        lr_compress_w: int = 4,
        gate_init: float = 0.0,
        use_self_attn: bool = True,
        use_lr_branch: bool = True,
        use_camera_pose: bool = False,  # ★ Path C: HistoryEncoder 是否接受相机 pose 输入
        pose_emb_dim: int = 32,  # ★ Path C: pose embedding 通道数 (concat 进 HR 分支)
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.compress_t = compress_t
        self.compress_h = compress_h
        self.compress_w = compress_w
        self.use_lr_branch = use_lr_branch
        self.lr_compress_t = lr_compress_t
        self.lr_compress_h = lr_compress_h
        self.lr_compress_w = lr_compress_w
        self.use_camera_pose = use_camera_pose
        self.pose_emb_dim = pose_emb_dim if use_camera_pose else 0

        # ★ Path C: pose embedder (c2w 12 floats → pose_emb_dim channels)
        # 用一个小 MLP 把 [B, T_h, 12] 升到 [B, T_h, pose_emb_dim], 然后广播到 (T_h, H_h, W_h) 跟 latent concat
        if use_camera_pose:
            self.pose_embedder = nn.Sequential(
                nn.Linear(12, 64),
                nn.SiLU(),
                nn.Linear(64, pose_emb_dim),
            )
        else:
            self.pose_embedder = None

        # ===== HR 分支 (重表征, 严格对齐 paper Fig.3 8-stage 顺序) =====
        # Stage 1: in_channels (+pose_emb_dim) → 64       (channel 缩,不下采样)
        # Stage 2: 64 → 128, stride=(compress_t,1,1)       (T 压)
        # Stage 3: 128 → 256, stride=(1,compress_h,compress_w)  (空间压)
        # Stage 4: 256 → 256, stride=1                    (refine, 新增)
        # Stage 5: 256 → 512, stride=1                    (refine)
        # Stage 6: 512 → 512, stride=1                    (refine)
        # Stage 7: CausalAttn3d (T 维 causal mask, 在最压缩 scale 上)
        # Stage 8: 512 → out_channels (1×1×1 projection)
        # ★ Path C: 启用 pose 时, stage1 输入通道 = in_channels + pose_emb_dim
        _stage1_in = in_channels + self.pose_emb_dim
        self.hr_stage1 = _Conv3dBlock(_stage1_in, 64, stride=(1, 1, 1))
        self.hr_stage2 = _Conv3dBlock(64, 128, stride=(compress_t, 1, 1))
        self.hr_stage3 = _Conv3dBlock(128, 256, stride=(1, compress_h, compress_w))
        self.hr_stage4 = _Conv3dBlock(256, 256, stride=(1, 1, 1))  # refine (新增, 跟 paper 对齐)
        self.hr_stage5 = _Conv3dBlock(256, 512, stride=(1, 1, 1))
        self.hr_stage6 = _Conv3dBlock(512, 512, stride=(1, 1, 1))

        self.use_self_attn = use_self_attn
        if use_self_attn:
            self.hr_attn = _SelfAttention3D(dim=512, num_heads=8)  # Stage 7: causal self-attn

        # Stage 8: 1×1×1 projection (用 CausalConv3d 保持因果性)
        self.hr_proj = CausalConv3d(512, out_channels, kernel_size=1, stride=1, padding=0)

        # ===== 共享 output_gate (跟 LongWorld 对齐, 1 个标量 gate, init=0.0) =====
        # 旧版有独立 hr_output_gate / lr_output_gate, 现合并成单一共享 gate
        # 应用位置: HR+LR add 之后, 整体缩放 (mem = (HR + LR) * output_gate)
        self.output_gate = nn.Parameter(torch.full((1,), float(gate_init)))

        # ===== LR 分支 (粗表征, Y-b' 思路: 复用主 transformer 的 patchify_proj 权重) =====
        # 复用方式: 在 FSDP wrap 前 copy 一份 patchify_proj.weight/bias 到 buffer,
        # forward 直接用 F.linear 调 buffer, 不依赖 transformer (避免 FSDP sharded weight 问题).
        # patchify_proj 在 LoRA 模式下是冻的 (LoRA target 不含 patchify_proj), 副本跟主路径一致.
        if use_lr_branch:
            # 占位 buffer, 待 setup_lr_proj_from_patchify(...) 填充
            self.register_buffer("lr_proj_weight", torch.zeros(out_channels, in_channels), persistent=False)
            self.register_buffer("lr_proj_bias", torch.zeros(out_channels), persistent=False)
            self._lr_proj_initialized = False

    def setup_lr_proj_from_patchify(self, patchify_proj: nn.Module):
        """从主 transformer.patchify_proj 复制权重到 LR 分支 buffer.

        必须在 FSDP wrap 主 transformer 之前调用 (此时 patchify_proj.weight 是完整的, 没分片).
        FSDP wrap 后再调会拿到 sharded weight, 复制结果是错的.
        """
        if not self.use_lr_branch:
            return
        with torch.no_grad():
            w = patchify_proj.weight.detach()
            self.lr_proj_weight = w.clone().to(self.lr_proj_weight.device)
            if patchify_proj.bias is not None:
                b = patchify_proj.bias.detach()
                self.lr_proj_bias = b.clone().to(self.lr_proj_bias.device)
            else:
                self.lr_proj_bias.zero_()
        self._lr_proj_initialized = True

    def _build_indices_grid(
        self,
        B: int,
        T_m: int,
        H_m: int,
        W_m: int,
        compress_t: int,
        compress_h: int,
        compress_w: int,
        device: torch.device,
    ) -> torch.Tensor:
        """计算 mem token 的 (T, H, W) 真实 latent patch bounds.

        Mem token i 代表原 latent 的 [i*compress, (i+1)*compress) 这块, 中点 i*compress + compress/2.
        返回 [B, 3, N, 2] 格式 (start, end), 跟 patchifier.get_patch_grid_bounds 一致,
        让主 DiT 走 get_pixel_coords + causal_fix + fps_norm 全套, RoPE 中点跟 generation 段对齐.
        """
        # T 维 bounds
        t_start = torch.arange(T_m, device=device, dtype=torch.float32) * compress_t
        t_end = t_start + compress_t
        # H 维 bounds
        h_start = torch.arange(H_m, device=device, dtype=torch.float32) * compress_h
        h_end = h_start + compress_h
        # W 维 bounds
        w_start = torch.arange(W_m, device=device, dtype=torch.float32) * compress_w
        w_end = w_start + compress_w

        # meshgrid 跟 generation 段顺序一致 (ij)
        Ts, Hs, Ws = torch.meshgrid(t_start, h_start, w_start, indexing="ij")  # [T_m, H_m, W_m]
        Te, He, We = torch.meshgrid(t_end, h_end, w_end, indexing="ij")

        starts = torch.stack([Ts.flatten(), Hs.flatten(), Ws.flatten()], dim=0)  # [3, N]
        ends = torch.stack([Te.flatten(), He.flatten(), We.flatten()], dim=0)  # [3, N]
        bounds = torch.stack([starts, ends], dim=-1)  # [3, N, 2]
        return bounds.unsqueeze(0).expand(B, -1, -1, -1).contiguous()  # [B, 3, N, 2]

    def forward(
        self,
        latent: torch.Tensor,
        patchify_proj: Optional[nn.Module] = None,
        past_c2w: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            latent: [B, in_channels, T_h, H_h, W_h] (e.g. [1, 128, 60, 17, 30])
            patchify_proj: [废弃] LR 分支已改用内部 buffer, 此参数被忽略
            past_c2w: [B, T_h, 4, 4] (★ Path C, 跟 latent 的 T_h 对齐). use_camera_pose=True 时必传.
                     未启用 pose 时此参数被忽略.

        Returns:
            mem_tokens: [B, N_total, out_channels]  (HR 在前, LR 在后)
            mem_indices_grid: [B, 3, N_total]
        """
        # === Debug: 计数器 + 是否打印 ===
        if not hasattr(self, "_fwd_count"):
            self._fwd_count = 0
        self._fwd_count += 1
        _DBG_INTERVAL = 50
        try:
            import torch.distributed as _dist

            _dbg_rank = _dist.get_rank() if _dist.is_initialized() else 0
        except Exception:
            _dbg_rank = 0
        _do_dbg = _dbg_rank == 0 and self._fwd_count % _DBG_INTERVAL == 1

        B, C, T_h, H_h, W_h = latent.shape

        # ★ Path C: 把 pose embedding concat 到 latent 的 channel 维
        # past_c2w [B, T_h, 4, 4] → 取前 3 行 [B, T_h, 3, 4] → flatten → [B, T_h, 12]
        # → MLP → [B, T_h, pose_emb_dim] → 广播到 [B, pose_emb_dim, T_h, H_h, W_h] → concat
        if self.use_camera_pose:
            assert past_c2w is not None, "use_camera_pose=True 时必须传 past_c2w"
            assert past_c2w.shape[0] == B and past_c2w.shape[1] == T_h, (
                f"past_c2w shape {tuple(past_c2w.shape)} 不匹配 [B={B}, T_h={T_h}, 4, 4]"
            )
            _pose_flat = past_c2w[:, :, :3, :].reshape(B, T_h, 12).to(latent.dtype)
            _pose_emb = self.pose_embedder(_pose_flat)  # [B, T_h, pose_emb_dim]
            # === Debug: pose_emb 统计 (Path C) ===
            if _do_dbg:
                _pe = _pose_emb.detach().float()
                _pf = _pose_flat.detach().float()
                # print(
                #     f"[HistEnc-DBG fwd={self._fwd_count}] pose(Path C) "
                #     f"input(c2w_12) abs|μ|={_pf.abs().mean():.4f} σ={_pf.std():.4f} "
                #     f"max={_pf.abs().max():.3f} | "
                #     f"pose_emb out abs|μ|={_pe.abs().mean():.4f} σ={_pe.std():.4f} "
                #     f"min={_pe.min():+.3f} max={_pe.max():+.3f} (dim={self.pose_emb_dim})",
                #     flush=True,
                # )
            # 广播到空间: [B, T_h, D] → [B, D, T_h, 1, 1] → expand 到 [B, D, T_h, H_h, W_h]
            _pose_grid = _pose_emb.permute(0, 2, 1).unsqueeze(-1).unsqueeze(-1)  # [B, D, T_h, 1, 1]
            _pose_grid = _pose_grid.expand(B, self.pose_emb_dim, T_h, H_h, W_h)
            latent_with_pose = torch.cat([latent, _pose_grid], dim=1)  # [B, in_ch + D, T_h, H, W]
        else:
            latent_with_pose = latent

        # ===== HR 分支 forward (8-stage paper Fig.3) =====
        x = self.hr_stage1(latent_with_pose)
        x = self.hr_stage2(x)  # Stage 2: T 压
        x = self.hr_stage3(x)  # Stage 3: H, W 压
        x = self.hr_stage4(x)  # Stage 4: refine
        x = self.hr_stage5(x)  # Stage 5: refine
        x = self.hr_stage6(x)  # Stage 6: refine
        if self.use_self_attn:
            x = self.hr_attn(x)  # Stage 7: causal self-attn
        x = self.hr_proj(x)  # Stage 8: 1×1 proj → [B, out_channels, T_hr, H_hr, W_hr]

        _, _, T_hr, H_hr, W_hr = x.shape
        hr_mem_tokens = x.permute(0, 2, 3, 4, 1).reshape(B, T_hr * H_hr * W_hr, self.out_channels)
        # === Debug: HR mem 统计 (gate 应用挪到 add 之后, 这里只记 raw HR 输出) ===
        if _do_dbg:
            _hr_raw = hr_mem_tokens.detach().float()
            _hr_raw_stats = (
                _hr_raw.abs().mean().item(),
                _hr_raw.std().item(),
                _hr_raw.min().item(),
                _hr_raw.max().item(),
            )
        hr_indices = self._build_indices_grid(
            B,
            T_hr,
            H_hr,
            W_hr,
            self.compress_t,
            self.compress_h,
            self.compress_w,
            device=latent.device,
        )

        if not self.use_lr_branch:
            # 单 HR 模式: 也走 output_gate (共享 gate)
            return hr_mem_tokens * self.output_gate, hr_indices

        if not self._lr_proj_initialized:
            raise RuntimeError(
                "LR 分支需要先调用 history_encoder.setup_lr_proj_from_patchify(transformer.patchify_proj). "
                "在 FSDP wrap 主 transformer 之前调用 (否则 patchify_proj.weight 已分片, 复制结果错)."
            )

        # ===== LR 分支 forward (复用 patchify_proj 权重 buffer) =====
        # 用 trilinear 插值下采样 (跟 LongWorld 一致, 信息保留比 avg_pool 好)
        # 部分 PyTorch 版本 trilinear 在 bf16 上支持有限, cast to float32 算完转回
        _orig_dtype = latent.dtype
        latent_fp32 = latent.float() if latent.dtype != torch.float32 else latent
        # ceil 公式: 跟 HR Conv3d(kernel=3, padding=1, stride=N) 在奇数维度上的下采样结果对齐
        # (HR 走 floor((D+2-3)/N)+1 = ceil(D/N), LR 这边也按 ceil 算, 保证 HR/LR shape 一致)
        target_T = (T_h + self.lr_compress_t - 1) // self.lr_compress_t
        target_H = (H_h + self.lr_compress_h - 1) // self.lr_compress_h
        target_W = (W_h + self.lr_compress_w - 1) // self.lr_compress_w
        lr_latent = F.interpolate(
            latent_fp32,
            size=(target_T, target_H, target_W),
            mode="trilinear",
            align_corners=False,
        )
        lr_latent = lr_latent.to(_orig_dtype)
        # [B, in_channels, T_lr, H_lr, W_lr] → [B, T_lr*H_lr*W_lr, in_channels]
        _, _, T_lr, H_lr, W_lr = lr_latent.shape
        lr_tokens = lr_latent.permute(0, 2, 3, 4, 1).reshape(B, T_lr * H_lr * W_lr, self.in_channels)
        # 用 buffer 调 F.linear (相当于复用主 transformer patchify_proj 但避开 FSDP sharded weight)
        lr_mem_tokens = F.linear(
            lr_tokens.to(self.lr_proj_weight.dtype),
            self.lr_proj_weight,
            self.lr_proj_bias,
        ).to(_orig_dtype)
        # === Debug: LR mem 统计 (gate 应用挪到 add 之后, 这里只记 raw LR 输出) ===
        if _do_dbg:
            _lr_raw = lr_mem_tokens.detach().float()
            _lr_raw_stats = (
                _lr_raw.abs().mean().item(),
                _lr_raw.std().item(),
                _lr_raw.min().item(),
                _lr_raw.max().item(),
            )
        lr_indices = self._build_indices_grid(
            B,
            T_lr,
            H_lr,
            W_lr,
            self.lr_compress_t,
            self.lr_compress_h,
            self.lr_compress_w,
            device=latent.device,
        )

        # ===== HR + LR element-wise add → 共享 output_gate (跟 LongWorld 对齐, PFP paper add 设计) =====
        # HR/LR 输出 shape 严格一致 (compress_t/h/w 三维相等, LR ceil 公式跟 HR Conv3d 输出对齐)
        # 单一共享 gate 应用在 add 之后, 跟 LongWorld 一致, HR/LR 等地位
        assert hr_mem_tokens.shape == lr_mem_tokens.shape, (
            f"HR shape {tuple(hr_mem_tokens.shape)} != LR shape {tuple(lr_mem_tokens.shape)}. "
            f"add 要求二者完全一致 — 检查 compress/lr_compress 三维是否对应相等."
        )
        mem_sum = hr_mem_tokens + lr_mem_tokens
        mem_tokens = mem_sum * self.output_gate
        mem_indices = hr_indices  # add 后只有一组 token, indices 用 HR 的 (HR/LR 同 grid)

        # === Debug: 合并后 output_gate 应用前后统计 (HR + LR pre/post gate) ===
        if _do_dbg:
            _sum_raw = mem_sum.detach().float()
            _sum_pre_stats = (_sum_raw.abs().mean().item(), _sum_raw.std().item())
            _mem_post = mem_tokens.detach().float()
            _mem_post_stats = (_mem_post.abs().mean().item(), _mem_post.std().item())
            # print(
            #     f"[HistEnc-DBG fwd={self._fwd_count}] output_gate={self.output_gate.item():+.6f} "
            #     f"| HR raw abs|μ|={_hr_raw_stats[0]:.4f} σ={_hr_raw_stats[1]:.4f} "
            #     f"min={_hr_raw_stats[2]:+.3f} max={_hr_raw_stats[3]:+.3f} "
            #     f"| LR raw abs|μ|={_lr_raw_stats[0]:.4f} σ={_lr_raw_stats[1]:.4f} "
            #     f"min={_lr_raw_stats[2]:+.3f} max={_lr_raw_stats[3]:+.3f} "
            #     f"| sum(HR+LR) pre-gate abs|μ|={_sum_pre_stats[0]:.4f} σ={_sum_pre_stats[1]:.4f} "
            #     f"| mem post-gate abs|μ|={_mem_post_stats[0]:.4f} σ={_mem_post_stats[1]:.4f}",
            #     flush=True,
            # )
        return mem_tokens, mem_indices

    def output_token_count(self, T_h: int, H_h: int, W_h: int) -> int:
        """工具函数: 给定输入 latent shape, 返回 mem token 总数.

        HR/LR add 融合 → token 数等于 HR (= LR, 二者同 shape), 不再相加.
        HR Conv3d(k=3,p=1,stride=N) 在奇数维度上等价 ceil(D/N), LR 也用 ceil 对齐.
        """

        def _ceil_div(a, b):
            return (a + b - 1) // b

        return _ceil_div(T_h, self.compress_t) * _ceil_div(H_h, self.compress_h) * _ceil_div(W_h, self.compress_w)
