import os
import subprocess
import time

# ================= 🚀 配置区域 =================
# 显卡分配 (请填入你空闲的 4 张显卡 ID)
GPUS = [3, 5, 6, 7] 

# 基础训练命令参数
BASE_CMD = [
    "yolo", "detect", "train",
    "model=yolo11n.yaml",           # 确保这里指像你修改过结构的 yaml
    "data=/home/lzx/car_bolt_detection/ultralytics-main/ultralytics/cfg/datasets/screw.yaml", # 数据集
    "epochs=200",                 
    "imgsz=640",
    "batch=16",                     # 保持一致
    "workers=8",
    "pretrained=yolo11n.pt",        # 加速收敛
    "exist_ok=True",
    "patience=20",                  # [重要] 加上早停，保持一致
    "val=True"  # 训练完成自动验证
]

# 定义 4 组消融实验
experiments = [
    {
        "name": "1_ECA_Only",
        "env": {"ULECA_USE_VAR": "0", "ULECA_USE_REC": "0", "ULECA_USE_BRI": "0"},
        "gpu": GPUS[0]
    },
    {
        "name": "2_ECA_Var",
        "env": {"ULECA_USE_VAR": "1", "ULECA_USE_REC": "0", "ULECA_USE_BRI": "0"},
        "gpu": GPUS[1]
    },
    {
        "name": "3_ECA_Var_Rec",
        "env": {"ULECA_USE_VAR": "1", "ULECA_USE_REC": "1", "ULECA_USE_BRI": "0"},
        "gpu": GPUS[2]
    },
    {
        "name": "4_ULECA_Full",
        "env": {"ULECA_USE_VAR": "1", "ULECA_USE_REC": "1", "ULECA_USE_BRI": "1"},
        "gpu": GPUS[3]
    }
]

def main():
    processes = []
    print(f"🚀 准备在 GPU {GPUS} 上并行启动 {len(experiments)} 个训练任务...")
    print("=" * 60)
    
    for exp in experiments:
        # 1. 准备环境变量
        current_env = os.environ.copy()
        current_env.update(exp["env"])
        
        # 2. 组装命令
        cmd = BASE_CMD + [
            f"name={exp['name']}",
            f"device={exp['gpu']}",
            "project=runs/ablation_retrain" # 结果单独存一个文件夹
        ]
        
        print(f"▶️  启动任务: {exp['name']}")
        print(f"    GPU: {exp['gpu']} | Config: {exp['env']}")
        
        # 3. 异步启动
        # stdout=subprocess.DEVNULL 可以让控制台清静点，不打印训练日志
        # 如果你想看日志，去掉 stdout 参数即可
        p = subprocess.Popen(cmd, env=current_env)
        processes.append(p)
        
        time.sleep(5) # 间隔几秒，防止文件写入冲突

    print("\n✅ 所有任务已在后台启动！")
    print("   请使用 'nvidia-smi' 查看显卡占用。")
    print("   训练大概需要 30-50 分钟。完成后请去 runs/ablation_retrain 查看结果。")
    
    # 等待所有任务完成
    for p in processes:
        p.wait()
        
    print("\n🎉 全部训练结束！")

if __name__ == "__main__":
    main()


