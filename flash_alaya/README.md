# FlashAlaya 推理

## 单卡

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True \
python -m flash_alaya.run --input playground/case1/case1
```

## 多卡（Context Parallel，N ∈ {2, 4}）

```bash
OMP_NUM_THREADS=1 PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun --nproc_per_node=4 -m flash_alaya.run --input playground/case1/case1
```

## 说明

- 配置：默认 `configs/infer.yaml`，换配置用 `--cfg`。所有权重/模型路径都在该文件的 `paths:` 下。
- 输入：`--input <prefix>`，需要 `<prefix>_camera.pt` + `<prefix>_prompt.txt`，外加二选一：
  - `<prefix>_video.mp4`：视频，按原样作为历史；
  - `<prefix>_image.<png/jpg/...>`：单张首帧，自动复制到相机轨迹长度来填历史。
  **分辨率必须等于配置里的 `sample.height/width`（默认 544×960）**，否则启动即报错。
  模型启动需要约 **5.4s（17 latent / 129 帧）历史**，最短输入约 **7s** 才能出 ≥1 个 chunk
  （图片输入会自动凑够）。
- 输出：`outputs/flash_alaya/<input>_rounds-N.mp4`，默认带 Move/Rotate 摇杆 HUD
  （关掉：配置里设 `validation.save_joystick: false`）。
- 常用参数：`--rounds`（自回归步数上限）、`--seed`（复现）、`--no-flex-attn`、`--compile none`。
