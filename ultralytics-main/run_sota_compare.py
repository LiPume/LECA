import os
import subprocess
import time

# ================= 🚀 配置区域 =================
# 分配两张空闲显卡
GPUS = [3, 5] 

# 定义实验
experiments = [
    {
        "name": "SOTA_SimAM",
        "model": "yolo11n-simam.yaml",
        "gpu": GPUS[0]
    },
    {
        "name": "SOTA_CoordAtt",
        "model": "yolo11n-coordatt.yaml",
        "gpu": GPUS[1]
    }
]

BASE_CMD = [
    "yolo", "detect", "train",

    "data=screw.yaml",
    "epochs=200", 
    "imgsz=640",
    "batch=16",
    "workers=8",
    "pretrained=yolo11n.pt",
    "exist_ok=True",
    "patience=20",
    "project=runs/sota_compare"  # 结果单独存
]

def main():
    processes = []
    print(f"🚀 开始 SOTA 对比训练 (SimAM vs CoordAtt)...")
    print("=" * 60)
    
    for exp in experiments:
        cmd = BASE_CMD + [
            f"model={exp['model']}",
            f"name={exp['name']}",
            f"device={exp['gpu']}"
        ]
        
        print(f"▶️  启动: {exp['name']} on GPU {exp['gpu']}")
        # 异步启动
        p = subprocess.Popen(cmd)
        processes.append(p)
        time.sleep(5)

    print("\n✅ 任务已启动，请等待训练完成...")
    
    for p in processes:
        p.wait()
    
    print("\n🎉 训练结束！请收集 mAP 数据填入 Table 1。")

if __name__ == "__main__":
    main()
