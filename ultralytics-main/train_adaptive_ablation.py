import os
import subprocess
import time
import sys
import torch
from ultralytics import YOLO

# ================= 🚀 实验配置中心 =================
# 1. 种子设置 (保证可复现性的关键)
SEED = 42

# 2. 硬件分配 (根据你的空闲显卡填)
# 建议使用 4 张卡，如果没有那么多，可以分批跑
GPUS = [4,5,6,7]  # 举例：使用 0,1,2,3 四张卡

# 3. 数据集和模型
DATA_YAML = "screw.yaml"
test_YAML = r"/home/lzx/car_bolt_detection/ultralytics-main/dataset/hardData/YOLODataset/hard_test_set.yaml"
MODEL_YAML = "yolo11n.yaml" # 使用上面创建的新 yaml

# 4. 训练参数 (统一标准)
TRAIN_ARGS = {
    "epochs": 200,
    "imgsz": 640,
    "batch": 16,
    "workers": 4,
    "patience": 20,
    "pretrained": "yolo11n.pt",
    "exist_ok": True,
    "optimizer": "auto",
    "deterministic": True, # [核心] 开启确定性模式
    "seed": SEED           # [核心] 固定种子
}

# 5. 消融配置 (定义实验组) -- 只有 Var 和 Rec，去掉 BRI 相关的实验
experiments = [
    {
        "name": "1_Adapt_Baseline4",
        "desc": "Adaptive ECA Only (Learned Weights Forced to 0)",
        "env": {"A_ULECA_VAR": "0", "A_ULECA_REC": "0"}
    },
    {
        "name": "2_Adapt_Var4",
        "desc": "Adaptive + Variance Only",
        "env": {"A_ULECA_VAR": "1", "A_ULECA_REC": "0"}
    },
    {
        "name": "3_Adapt_Rec4",
        "desc": "Adaptive + Rec Only",
        "env": {"A_ULECA_VAR": "0", "A_ULECA_REC": "1"}
    },
    {
        "name": "4_Adapt_Var_Rec4",
        "desc": "Adaptive + Var + Rec",
        "env": {"A_ULECA_VAR": "1", "A_ULECA_REC": "1"}
    }
]

# ====================================================
def run_training_task(exp, gpu_id):
    """启动单个训练进程"""
    cmd = ["yolo", "detect", "train"]
    
    # 添加基础参数
    cmd.extend([f"model={MODEL_YAML}", f"data={DATA_YAML}"])
    for k, v in TRAIN_ARGS.items():
        cmd.append(f"{k}={v}")
        
    # 添加实验特定参数
    cmd.append(f"name={exp['name']}")
    cmd.append(f"device={gpu_id}")
    cmd.append("project=runs/adaptive_ablation") # 统一保存路径
    
    # 准备环境变量
    env = os.environ.copy()
    env.update(exp['env'])
    
    print(f"▶️  [GPU {gpu_id}] 启动: {exp['name']} ({exp['desc']})")
    
    # 启动子进程
    # stdout=subprocess.DEVNULL 可以隐藏海量日志，建议开启
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return p

def evaluate_results():
    """所有训练结束后，统一评估"""
    print("\n📊 正在收集实验结果 (Hard Test Set)...")
    print(f"{'Method':<25} | {'Precision':<10} | {'Recall':<10} | {'mAP@0.5':<10}")
    print("-" * 65)
    
    results = []
    for exp in experiments:
        # 寻找最佳权重
        best_pt = f"runs/adaptive_ablation/{exp['name']}/weights/best.pt"
        
        if not os.path.exists(best_pt):
            print(f"{exp['name']:<25} | ❌ Training Failed (No best.pt)")
            continue
            
        try:
            # 加载模型进行验证
            model = YOLO(best_pt)
            metrics = model.val(
                data=test_YAML,
                split='test',
                batch=16,
                device=GPUS[0], # 用第一张卡验证即可
                conf=0.001,     # [关键] 保持低阈值以评估 Recall
                iou=0.5,
                plots=False,
                verbose=False
            )
            
            p = metrics.results_dict['metrics/precision(B)']
            r = metrics.results_dict['metrics/recall(B)']
            map50 = metrics.results_dict['metrics/mAP50(B)']
            
            print(f"{exp['name']:<25} | {p:.4f}     | {r:.4f}     | {map50:.4f}")
            results.append((exp['name'], p, r, map50))
            
        except Exception as e:
            print(f"{exp['name']:<25} | ❌ Eval Error")

    return results

def main():
    print(f"🚀 开始 Adaptive ULECA 并行消融实验 (SEED={SEED})")
    print(f"🔥 使用 GPU: {GPUS}")
    print("=" * 60)

    processes = []
    
    # 1. 并行启动所有任务
    for i, exp in enumerate(experiments):
        if i >= len(GPUS):
            print("❌ GPU 数量不足，无法完全并行！请增加 GPUS 列表或减少实验组。")
            break
        p = run_training_task(exp, GPUS[i])
        processes.append(p)
        time.sleep(5) # 间隔启动，防止争抢文件资源

    print("\n⏳ 所有任务已在后台运行，请耐心等待 (约 30-50 分钟)...")
    
    # 2. 等待所有任务结束
    for p in processes:
        p.wait()
        
    print("\n✅ 所有训练任务完成！")
    
    # 3. 自动评估
    evaluate_results()

if __name__ == "__main__":
    main()
