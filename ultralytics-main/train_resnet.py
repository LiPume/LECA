from ultralytics import YOLO
import torch

# 确保能找到你的 LECA
from ultralytics.nn.modules.attention import LECA

# 数据集配置文件
data_yaml = "screw.yaml" 
# 【重要】你的困难测试集 yaml (如果 screw.yaml 里没有 test 指向 hard set，就新建一个)
hard_test_yaml = "/home/lzx/car_bolt_detection/ultralytics-main/dataset/hardData/YOLODataset/hard_test_set.yaml" # 或者 "screw_hard.yaml"

def run_experiment(yaml_file, run_name):
    print(f"\n{'='*20} Running: {run_name} {'='*20}")
    
    # 加载模型
    model = YOLO(yaml_file, task='detect')
     # 1. 训练 (ResNet 比较重，建议多跑点 epoch 或者开早停)
    model.train(
        data=data_yaml,
        epochs=200,      # 100轮 + 早停
        patience=20,
        imgsz=640,
        device=0,
        batch=4,        # ResNet50 显存占用较大，如果 OOM 就调小 batch
        project="runs/resnet_exp",
        pretrained="rtdetr-l.pt",
        name=run_name,
        exist_ok=True,
        verbose=True
    )
    
    # 2. 在 Hard Set 上验证
    print(f"Validating {run_name} on Hard Test Set...")
    # 注意：这里假设你的 yaml 里 split='test' 指向的是 hard set
    # 或者你可以手动指定 data='hard_set.yaml'
    metrics = model.val(data=hard_test_yaml, split='test') 
    
    map50 = metrics.results_dict.get('metrics/mAP50(B)', 0)
    print(f"--> {run_name} Result: mAP50 = {map50:.4f}")
    return map50

if __name__ == "__main__":
    # 1. 跑 Baseline (不加 LECA)
    # 请先创建 resnet50_base.yaml (把上面 yaml 里的 LECA 删掉)
    acc_base = run_experiment("/home/lzx/car_bolt_detection/ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet50.yaml", "resnet50_baseline")
    
    # 2. 跑 LECA (加 LECA)
    acc_leca = run_experiment("/home/lzx/car_bolt_detection/ultralytics-main/ultralytics/cfg/models/rt-detr/rtdetr-resnet50_leca.yaml", "resnet50_leca")
    
    print(f"\n{'='*30} Final Comparison {'='*30}")
    print(f"ResNet50 Baseline: {acc_base:.4f}")
    print(f"ResNet50 + LECA:   {acc_leca:.4f}")
    
    improvement = acc_leca - acc_base
    print(f"Improvement:       {'+' if improvement>0 else ''}{improvement*100:.2f}%")
    
    if improvement > 0:
        print("\n结论: 恭喜！LECA 在 ResNet 骨干上也有提升，泛化性得到证明！")
    else:
        print("\n注意: 提升不明显，可能需要调整 ResNet 的 LECA 插入位置或参数。")