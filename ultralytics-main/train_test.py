import ultralytics
from ultralytics import YOLO
import torch
import csv
import os
import numpy as np

print(f"--- 正在加载的 ultralytics 库位于: {ultralytics.__file__} ---")

def on_train_epoch_end(trainer):
    m = trainer.model
    # 兼容 DataParallel/DDP 包装
    if hasattr(m, "module"):
        m = m.module

    alphas, betas, gammas, names = [], [], [], []

    for name, module in m.named_modules():
        if all(hasattr(module, k) for k in ("alpha", "beta", "gamma")):
            # 确认是可学习参数，避免误匹配
            if isinstance(module.alpha, torch.nn.Parameter) and isinstance(module.beta, torch.nn.Parameter) and isinstance(module.gamma, torch.nn.Parameter):
                alphas.append(module.alpha.detach().float().cpu().item())
                betas.append(module.beta.detach().float().cpu().item())
                gammas.append(module.gamma.detach().float().cpu().item())
                names.append(name)

    if not alphas:
        return

    epoch = trainer.epoch + 1
    save_dir = str(trainer.save_dir)
    csv_path = os.path.join(save_dir, "leca_params.csv")
    file_exists = os.path.isfile(csv_path)

    # 统计：均值+标准差（能体现收敛/稳定性），再额外记录“第一层/最后一层”
    a_mean, a_std = float(np.mean(alphas)), float(np.std(alphas))
    b_mean, b_std = float(np.mean(betas)), float(np.std(betas))
    g_mean, g_std = float(np.mean(gammas)), float(np.std(gammas))

    a_first, b_first, g_first = alphas[0], betas[0], gammas[0]
    a_last,  b_last,  g_last  = alphas[-1], betas[-1], gammas[-1]

    with open(csv_path, mode="a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow([
                "epoch",
                "alpha_mean","alpha_std","beta_mean","beta_std","gamma_mean","gamma_std",
                "alpha_first","beta_first","gamma_first",
                "alpha_last","beta_last","gamma_last",
                "num_leca"
            ])
        w.writerow([
            epoch,
            a_mean,a_std,b_mean,b_std,g_mean,g_std,
            a_first,b_first,g_first,
            a_last,b_last,g_last,
            len(alphas)
        ])

model = YOLO("yolo11n.yaml")

model.add_callback("on_train_epoch_end", on_train_epoch_end)




# Train the model
train_results = model.train(
    data="screw.yaml",  # 数据集配置文件
    epochs=200,
    imgsz=640,
    device=5,
    workers=8,  
    pretrained="yolo11n.pt",
    name="car_bolt_leca_basen",
    exist_ok=True,
    patience=20,
    val=True, 
)

MODEL_PATH = r"/home/lzx/car_bolt_detection/ultralytics-main/runs/detect/car_bolt_leca_basen/weights/best.pt"
DATA_YAML_PATH = r"/home/lzx/car_bolt_detection/ultralytics-main/dataset/hardData/YOLODataset/hard_test_set.yaml"
model = YOLO(MODEL_PATH)

print(f"正在对 {DATA_YAML_PATH} 指定的 'test' 集进行评估...")
results = model.val(
    data=DATA_YAML_PATH,
    split='test',
    conf=0.0001,
    iou=0.5,
)


print("\n评估完成！")
print("--- 自动计算的指标 (mAP, Precision, Recall) ---")
print(f"Overall mAP@0.5: {results.box.map50}")  # 打印 mAP@0.5
print(f"Overall Precision (P): {results.box.mp}")  # 打印 Mean Precision
print(f"Overall Recall (R): {results.box.mr}")  # 打印 Mean Recall
