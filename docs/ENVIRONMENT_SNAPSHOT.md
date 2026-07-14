# 环境快照

采集时间：2026-07-14，采集于审计驱动训练之前。

| 项目 | 值 |
| --- | --- |
| Conda 环境 | `yolo` |
| Python | 3.10.19 |
| PyTorch | 2.5.1+cu121 |
| PyTorch CUDA 运行时 | 12.1 |
| CUDA 可用 | 是 |
| PyTorch 可见 GPU | NVIDIA GeForce RTX 4090 D |
| NVIDIA 驱动 / `nvidia-smi` CUDA | 580.159.03 / 13.0 |
| 本地 Ultralytics | 8.3.215 |

`nvidia-smi` 当时显示 8 张 RTX 4090 D；GPU 0 基本空闲，GPU 1、4–7 有其他计算任务。每次实验均须记录实际设备，并在启动前重新检查可用性。可移植 Conda 规格在 `environment.yml`，已删除机器相关的 `prefix`。
