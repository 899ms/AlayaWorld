# Streaming Decoder 改进计划(未实施)

> 目标:交互式生成场景下,rollout 每出一个 K=4 latent 块就立刻解码出像素帧,
> 跨块无缝、显存恒定。记录于 2026-06-10,实施待定。

## 1. 背景与已确认的事实

代码事实(均已核实):

- 解码器 `VideoDecoder`(`ltx2/modules/vae.py`)按 **非因果** 配置运行:
  checkpoint config `causal_decoder: False` → 每个 `CausalConv3d` 走双侧对称
  padding(`forward(causal=False)`),输出帧 t 依赖输入的过去与未来。
- 除卷积外全部是逐位置算子:`norm_layer: pixel_norm`、
  `timestep_conditioning: False`、`inject_noise` 未开启、无 attention
  → 网络是纯局部、有限感受野的 conv 网络。
- 时间感受野半径逐层累计 ≈ **30 个 latent 帧**(大头在 1×/2× 低分辨率段:
  conv_in + res_x(4) + res_x(6) 等 ~23 convs 在 1× 域)。
- 帧数公式 `(d-1)*8+1` 来自 `DepthToSpaceUpsample` 每级 ×2 后砍首帧,与
  causal 与否无关。
- 当前 `_decode_latent_to_video_frames`(chunk=4, 无状态, 丢首帧)是 **有损**
  的:每块边界塌缩关键帧 + 跳变(实测 `compare.log`: 丢 64 帧, 边界
  mean|diff|≈25/255)。
- 编码侧 `StreamingVAEEncoder`(`fastvideo/ltx2_streaming_vae.py`)已用
  **逐层 feat cache** 实现分块==整段的无损流式;decoder 改进即其镜像。

## 2. 硬结论

**交互式(streaming)+ 与非因果整段逐位相等,不可兼得。**
非因果解码器中帧 t 的"正确"像素依赖未来 ~30 latent 帧;交互式下未来尚未生成。
任何精确方案的延迟下界 = 感受野的未来半径(≈30 latent ≈ 10s 视频)。
因此交互式必须选择某种近似;离线出片另有精确方案(见 §5)。

## 3. 交互式候选方案

### A. causal 模式 + 逐层 feat cache(首选实验)

- `VideoDecoder.forward` 已有 `causal` 开关;传 `causal=True` → 全部 conv
  只看过去。
- 配 `StreamingVAEDecoder`(镜像 `StreamingVAEEncoder` 的三件套:
  `_apply_conv_with_cache` / 逐块驱动 / 首块特殊处理):
  - 第一块:最左侧"复制首帧"padding(= causal 整段行为,全片仅此一次);
  - 后续块:每个 conv 层的左上下文 = 上一块该层尾部 2 帧真实特征(cache),
    不是重新 padding;
  - `DepthToSpaceUpsample` 的"砍首帧"只在第一块做 → 后续块输出满 d*8 帧,
    总帧数与整段一致,**不再丢帧**。
- 性质:零 lookahead、跨块逐位自洽(流式 ≡ causal 整段)。
- 风险:权重按非因果训练,causal 推理感受野砍半,**质量待实测**。
- 前置实验(十分钟级):用 `pred_latent.pt` 整段解 causal=True vs False,
  出并排对比 + diff,人工判断质量损失是否可接受。

### B. 流式 + 1 块 lookahead(质量最稳的全 VAE 方案)

- 生成块 i 后解码块 i-1:左侧用层级 cache,右侧用块 i 当 halo。
- 延迟 = 1 chunk ≈ 1.3s;右 halo 4 latent < 理论半径 30,但卷积影响随距离
  指数衰减,边界误差应极小(近似,视觉无缝)。
- 实现 = 方案 A 的 cache 框架 + 右侧 halo 输入、输出丢 halo 区。

### C. 流式重叠 + 梯形融合(实现最快的兜底)

- 每块带上一块尾部 N latent 重叠解码,重叠区线性加权融合
  (官方 LTX-2 `tiled_decode` 的流式化,参考 ltx-core 的 `tiling.py`)。
- 零额外延迟;近似;重叠区重复计算(~N/K 额外开销)。

### D. 轻量因果流式解码器 taehv / LightTAE(真·实时路线)

- FlashDreams 交互管线(LingBot Fast)的选择:为 streaming 设计的因果小
  解码器,零延迟逐块出帧,比全 VAE 快约一个量级。
- workspace 已有 `taehv/` 与 `LTX-2/run_ltx_taehv.py`(适配尝试已存在);
  需确认对 LTX-2.3 latent 的质量。
- 推荐组合:taehv 出交互画面(实时) + 离线整段全 VAE 重解"高清回放"。
- 顺带:spatial bank 喂 DA3 的 `_decode_latent_to_bank_pixels` 也应换
  taehv(深度估计对像素质量不敏感,纯赚速度)。

## 4. 建议路线

1. 跑方案 A 的前置质量实验(causal=True vs False 整段对比)。
2. 质量可接受 → 实现 `StreamingVAEDecoder`(A);不可接受 → B(同一 cache
   框架加 1 块 lookahead)。
3. 并行评估 D(taehv)作为实时交互主路径;A/B 产物降级为"高质量流式"档。
4. 接入 `FlashAlayaPipeline`:`finalize()` 后调用流式解码器逐块出帧
   (替代攒满后 `decode()` 一次性出片);bank/DA3 解码切 taehv。

## 5. 附:离线(非交互)的精确分块方案

仅为完整记录——离线出片若要省显存且与整段逐位相等:
**按 stage 混合分块**:stage0/1(1×/2× 域,感受野大头、显存便宜)整段算;
stage2/3(4×/8× 域,显存大头、stage 内半径仅 ~5)时间分块 + 小 halo,
输出丢 halo 拼接。预计峰值 71GB → ~10GB 级,逐位相等。
验证方法:`max|混合分块 - whole|` 用 `pred_latent.pt` 实测应为 0(或 bf16
非确定性 1e-3 内)。

## 6. 验收标准(实施时)

- 流式解码器对同一 latent 序列:总帧数 = `(D-1)*8+1`(不丢帧);
- 方案 A:流式输出 vs causal 整段输出 `max|diff| = 0`(逐位自洽);
- 块边界处 per-frame mean|diff|(vs 非因果整段参照)无可见跳变
  (对照现状 chunk=4 的 ~25/255);
- 单块解码延迟与显存:报告 ms/块 与峰值 VRAM;
- 接入 pipeline 后端到端交互延迟(生成+解码)报告。
