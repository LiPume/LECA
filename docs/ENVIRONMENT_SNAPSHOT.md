# Environment snapshot

Captured on 2026-07-14 before audit-driven training.

| Item | Value |
| --- | --- |
| Conda environment | `yolo` |
| Python | 3.10.19 |
| PyTorch | 2.5.1+cu121 |
| PyTorch CUDA runtime | 12.1 |
| CUDA available | Yes |
| GPU visible to PyTorch | NVIDIA GeForce RTX 4090 D |
| NVIDIA driver / `nvidia-smi` CUDA | 580.159.03 / 13.0 |
| Local Ultralytics source version | 8.3.215 |

`nvidia-smi` showed eight RTX 4090 D devices. At capture time GPU 0 was almost idle, while GPUs 1 and 4--7 had other active compute processes. Every experiment must record the actual selected device and re-check availability immediately before launch.

The complete portable Conda specification is in `environment.yml`; it intentionally omits the machine-specific `prefix`.
