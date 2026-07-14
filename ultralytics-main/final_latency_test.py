import torch
import time
import os
import pandas as pd
from ultralytics import YOLO

# ================= 🚀 配置区域 =================
# 1. 定义三个模型的路径
MODELS = {
    'ULECA (Ours)':        '/home/lzx/car_bolt_detection/ultralytics-main/runs/detect/car_bolt_LECA7/weights/best.pt'
}

# 2. 指定显卡 (找一张空闲的)
DEVICE = torch.device('cuda:5') 

# 3. 测试参数
IMG_SIZE = 640
WARMUP_RUNS = 200   # 热身次数
TEST_RUNS = 1000    # 测试次数 (取平均更准)
# ==============================================

def measure_latency(model_path, model_name):
    if not os.path.exists(model_path):
        print(f"❌ 错误: 找不到文件 {model_path}")
        return None

    # 加载模型
    print(f"\n🔄 正在加载: {model_name} ...")
    try:
        model = YOLO(model_path)
        model.to(DEVICE)
        model.eval() # 切换到评估模式
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None

    # 准备 Dummy Input (B=1, C=3, H=640, W=640)
    input_tensor = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(DEVICE)

    # 1. Warmup (让 GPU 进入高性能状态)
    print(f"🔥 Warming up ({WARMUP_RUNS} runs)...")
    with torch.no_grad():
        for _ in range(WARMUP_RUNS):
            _ = model(input_tensor)

    # 2. 正式计时 (使用 CUDA Event)
    print(f"⏱️  Benchmarking ({TEST_RUNS} runs)...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    with torch.no_grad():
        for _ in range(TEST_RUNS):
            _ = model(input_tensor)
    end_event.record()
    
    # 等待 GPU 完成
    torch.cuda.synchronize()

    # 计算结果
    total_time_ms = start_event.elapsed_time(end_event)
    avg_latency = total_time_ms / TEST_RUNS
    fps = 1000.0 / avg_latency
    
    # 获取参数量 (Millions) 和 FLOPs (Giga)
    # YOLOv11 的 info() 会打印，这里我们手动获取一下大概值
    params = sum(p.numel() for p in model.parameters()) / 1e6

    print(f"✅ 完成: {avg_latency:.3f} ms | {fps:.1f} FPS")
    return {
        'Method': model_name,
        'Params (M)': f"{params:.2f}",
        'Latency (ms)': avg_latency,
        'FPS': fps
    }

def main():
    print(f"🚀 开始真实推理延迟测试 (Device: {DEVICE})")
    print("=" * 60)
    
    results = []
    
    # 按顺序测试
    for name, path in MODELS.items():
        res = measure_latency(path, name)
        if res:
            results.append(res)
    
    # 打印最终表格
    print("\n\n" + "="*60)
    print("📄 FINAL LATENCY COMPARISON (Table Y)")
    print("="*60)
    print(f"{'Method':<25} | {'Params(M)':<10} | {'Latency':<10} | {'FPS':<10}")
    print("-" * 60)
    
    base_latency = 0
    for i, res in enumerate(results):
        lat = res['Latency (ms)']
        if i == 0: base_latency = lat
        
        # 计算相对于 Baseline 的增加量
        diff = lat - base_latency
        diff_str = f"(+{diff:.2f}ms)" if diff > 0 else "(-)"
        
        print(f"{res['Method']:<25} | {res['Params (M)']:<10} | {lat:.3f}ms | {res['FPS']:.1f}")
    
    print("="*60)

if __name__ == "__main__":
    main()
