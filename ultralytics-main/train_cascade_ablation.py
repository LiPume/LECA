import os
import subprocess
import time
import sys
import torch
from ultralytics import YOLO

# ================= 🚀 级联消融实验配置 =================
SEED = 42
GPUS = [4, 5, 6, 7] # 确认使用这4张卡

DATA_YAML = "screw.yaml"
# 验证集路径 (Hard Test Set)
TEST_YAML = "/home/lzx/car_bolt_detection/ultralytics-main/dataset/hardData/YOLODataset/hard_test_set.yaml"
MODEL_YAML = "yolo11n.yaml" 

TRAIN_ARGS = {
    "epochs": 200,
    "imgsz": 640,
    "batch": 16,
    "workers": 3,          # 防止崩溃
    "patience": 20,
    "pretrained": "yolo11n.pt",
    "exist_ok": True,
    "deterministic": True, # [核心] 必须开启
    "seed": SEED           # [核心] 必须固定
}

# 消融配置 (注意环境变量名变成了 C_ULECA_*)
experiments = [
    {
        "name": "1_Cas_Baseline1",
        "desc": "Cascade Baseline (ECA Only)",
        "env": {"C_ULECA_VAR": "0", "C_ULECA_REC": "0", "C_ULECA_BRI": "0"}
    },
    {
        "name": "2_Cas_Var1",
        "desc": "Cascade + Variance",
        "env": {"C_ULECA_VAR": "1", "C_ULECA_REC": "0", "C_ULECA_BRI": "0"}
    },
    {
        "name": "3_Cas_Var_Rec1",
        "desc": "Cascade + Var + Rec",
        "env": {"C_ULECA_VAR": "1", "C_ULECA_REC": "1", "C_ULECA_BRI": "0"}
    },
    {
        "name": "4_Cas_Full1",
        "desc": "Cascade Full (Var+Rec+Bri)",
        "env": {"C_ULECA_VAR": "1", "C_ULECA_REC": "1", "C_ULECA_BRI": "1"}
    }
]

# ====================================================

def run_training_task(exp, gpu_id):
    cmd = ["yolo", "detect", "train"]
    cmd.extend([f"model={MODEL_YAML}", f"data={DATA_YAML}"])
    for k, v in TRAIN_ARGS.items():
        cmd.append(f"{k}={v}")
        
    cmd.append(f"name={exp['name']}")
    cmd.append(f"device={gpu_id}")
    cmd.append("project=runs/cascade_ablation") 
    
    env = os.environ.copy()
    env.update(exp['env'])
    
    os.makedirs("logs_cascade", exist_ok=True)
    log_file = open(f"logs_cascade/{exp['name']}.log", "w")
    
    print(f"▶️  [GPU {gpu_id}] 启动: {exp['name']}")
    
    p = subprocess.Popen(cmd, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    return p, log_file

def evaluate_results():
    print("\n📊 正在收集 Cascade 实验结果 (Hard Test Set)...")
    print(f"{'Method':<25} | {'Precision':<10} | {'Recall':<10} | {'mAP@0.5':<10}")
    print("-" * 65)
    
    for exp in experiments:
        best_pt = f"runs/cascade_ablation/{exp['name']}/weights/best.pt"
        
        if not os.path.exists(best_pt):
            print(f"{exp['name']:<25} | ❌ Failed (No weights)")
            continue
            
        try:
            # 注入环境，确保验证时结构一致
            os.environ.update(exp['env'])
            
            model = YOLO(best_pt)
            metrics = model.val(
                data=TEST_YAML,
                split='test',
                batch=16,
                device=GPUS[0],
                conf=0.001,     # 保持低阈值
                iou=0.5,
                plots=False,
                verbose=False
            )
            
            p = metrics.results_dict['metrics/precision(B)']
            r = metrics.results_dict['metrics/recall(B)']
            map50 = metrics.results_dict['metrics/mAP50(B)']
            
            print(f"{exp['name']:<25} | {p:.4f}     | {r:.4f}     | {map50:.4f}")
            
        except Exception as e:
            print(f"{exp['name']:<25} | ❌ Eval Error")

def main():
    print(f"🚀 开始 Cascade ULECA 并行消融实验 (SEED={SEED})")
    print("📄 日志位置: logs_cascade/*.log")
    print("=" * 60)

    processes = []
    open_files = []
    
    for i, exp in enumerate(experiments):
        if i >= len(GPUS): break
        p, f = run_training_task(exp, GPUS[i])
        processes.append(p)
        open_files.append(f)
        time.sleep(5)

    print("\n⏳ 任务运行中...")
    
    for p in processes: p.wait()
    for f in open_files: f.close()
        
    print("\n✅ 训练完成！开始评估...")
    evaluate_results()

if __name__ == "__main__":
    main()