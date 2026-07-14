import ultralytics
from ultralytics import YOLO
import torch
import csv
import os
import numpy as np

print(f"--- 正在加载的 ultralytics 库位于: {ultralytics.__file__} ---")

# --- 1. 定义回调函数 ---
def on_train_epoch_end(trainer):
    """
    在每个 epoch 结束时被调用。
    用来记录 LECA 模块中的 alpha, beta, gamma 参数值。
    """
    model = trainer.model
    
    # 用于存储当前 epoch 所有 LECA 层的参数值
    alphas, betas, gammas = [], [], []
    
    # 遍历模型的所有子模块
    for name, module in model.named_modules():
        # 通过属性判断是否是你的 LECA 模块 (只要有 alpha, beta, gamma 参数)
        if hasattr(module, 'alpha') and hasattr(module, 'beta') and hasattr(module, 'gamma'):
            # 注意：参数可能在 GPU 上，需要转到 CPU 并取 item
            # 这里我们记录所有 LECA 层参数的【平均值】，代表整体趋势
            # 如果你想研究特定层，可以根据 name 筛选
            alphas.append(module.alpha.detach().cpu().item())
            betas.append(module.beta.detach().cpu().item())
            gammas.append(module.gamma.detach().cpu().item())

    # 如果找到了参数，记录到 CSV
    if alphas:
        avg_alpha = np.mean(alphas)
        avg_beta = np.mean(betas)
        avg_gamma = np.mean(gammas)
        epoch = trainer.epoch + 1
        
        # CSV 文件路径保存到训练目录下
        save_dir = trainer.save_dir
        csv_path = os.path.join(save_dir, 'leca_params.csv')
        
        # 如果是第一个 epoch，创建文件并写入表头
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['epoch', 'alpha', 'beta', 'gamma'])
            writer.writerow([epoch, avg_alpha, avg_beta, avg_gamma])

# --- 2. 加载模型 ---
# 确保你的 yaml 里使用的是你定义了 LUECA 的那个版本
model = YOLO("yolo11n.yaml") 

# --- 3. 注册回调函数 ---
model.add_callback("on_train_epoch_end", on_train_epoch_end)

# --- 4. 开始训练 ---
train_results = model.train(
    data="screw.yaml",
    epochs=200,
    imgsz=640,
    device=5,
    workers=8,
    pretrained="yolo11n.pt",
    name="car_bolt_leca_analysis", # 改个名字方便区分
    exist_ok=True,
    patience=20,
    val=True,
)