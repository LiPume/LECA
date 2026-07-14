import csv
import torch
from ultralytics import YOLO
# 引入你的 LECA 类 (根据实际路径修改)
from ultralytics.nn.modules.attention import LECA

# 敏感性配置
configs = {
    'Default':  {'alpha': 0.02, 'beta': 0.04, 'gamma': 0.01},
    'Beta_0.02':   {'alpha': 0.02, 'beta': 0.02, 'gamma': 0.01}, 
    'Beta_0.06':   {'alpha': 0.02, 'beta': 0.06, 'gamma': 0.01}, 
    'Alpha_0.01':   {'alpha': 0.01, 'beta': 0.04, 'gamma': 0.01}, 
    'Alpha_0.03':   {'alpha': 0.03, 'beta': 0.04, 'gamma': 0.01}, 
    'Gamma_0.005':  {'alpha': 0.02, 'beta': 0.04, 'gamma': 0.005},
    'Gamma_0.015':  {'alpha': 0.02, 'beta': 0.04, 'gamma': 0.015},
}

csv_file = "sensitivity_analysis_hard.csv"
# 初始化 CSV
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    # 记下来：Config名，参数，以及 Hard Set 上的 mAP
    writer.writerow(['Config_Name', 'Alpha', 'Beta', 'Gamma', 'Hard_mAP50', 'Hard_mAP50-95'])

print(f"开始敏感性分析 (Target: Hard Test Set)...")

for name, conf in configs.items():
    print(f"\n>>> 运行配置: {name} {conf}")

    # 1. 修改类变量
    LECA.INIT_ALPHA = conf['alpha']
    LECA.INIT_BETA = conf['beta']
    LECA.INIT_GAMMA = conf['gamma']

    # 2. 构建模型
    model = YOLO("yolo11n.yaml") 

    try:
        # 3. 训练 (用普通数据训练)
        # 设 epochs=100, patience=15 足够了
        model.train(
            data="screw.yaml",  # 训练用普通数据
            epochs=200,         # 不需要太久
            patience=20,        # 早停是关键
            imgsz=640,
            device=0,
            workers=4,
            project="runs/sensitivity",
            name=f"sens_{name}",
            exist_ok=True,
            verbose=False
        )

        print(f"   训练完成，开始在 Hard Test Set 上评估...")

        # 4. 【关键步骤】在 Hard Test Set 上验证
        # 假设你有一个 screw_hard.yaml，里面的 'val' 指向你的困难数据集路径
        # 如果没有，你需要新建一个 yaml，把 val: 指向你的 hard test 图片文件夹
        metrics = model.val(
            data="/home/lzx/car_bolt_detection/ultralytics-main/dataset/hardData/YOLODataset/hard_test_set.yaml",
            split='test') 
        
        # 5. 获取 Hard Set 的指标
        map50 = metrics.results_dict.get('metrics/mAP50(B)', 0)
        map5095 = metrics.results_dict.get('metrics/mAP50-95(B)', 0)

        # 6. 写入 CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([name, conf['alpha'], conf['beta'], conf['gamma'], map50, map5095])
            
        print(f"   --> {name} Hard Set mAP50: {map50:.4f}")

    except Exception as e:
        print(f"   !!! {name} 失败: {e}")

print(f"\n所有实验结束，结果已保存至 {csv_file}")