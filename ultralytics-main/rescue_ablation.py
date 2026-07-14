import torch
from ultralytics import YOLO
import sys
import os

# ================= 🚑 救援配置区域 =================
# 1. 必须指向你 Windows 上训练的那个"好"模型 (0.975那个)
# 请确认路径！不要用刚才崩掉的Linux模型！
MODEL_PATH = '/home/lzx/car_bolt_detection/ultralytics-main/runs/detect/car_bolt_LECA7/weights/best.pt'

# 2. 数据集配置 (Linux路径版)
DATA_YAML = '/home/lzx/car_bolt_detection/ultralytics-main/dataset/hardData/YOLODataset/hard_test_set.yaml'

# 3. 指定显卡
DEVICE = '5'
# ====================================================

def set_leca_config(model, config):
    """动态设置开关"""
    for m in model.model.modules():
        if m.__class__.__name__ in ['LECA', 'ULECA']:
            m.use_var = config.get('use_var', True)
            m.use_rec = config.get('use_rec', True)
            m.use_bri = config.get('use_bri', True)

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型: {MODEL_PATH}")
        return
    
    print(f"🔄 加载 Windows 冠军模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 定义消融配置
    configs = {
        'Baseline (ECA)':   {'use_var': False, 'use_rec': False, 'use_bri': False},
        '+ Variance':       {'use_var': True,  'use_rec': False, 'use_bri': False},
        '+ Var & Rec':      {'use_var': True,  'use_rec': True,  'use_bri': False},
        'Full ULECA':       {'use_var': True,  'use_rec': True,  'use_bri': True}
    }

    # 我们要测试的阈值列表
    # 0.001: YOLO默认，Recall最高，ULECA优势最大
    # 0.01:  折中
    # 0.1:   你之前的设置 (导致ULECA失效)
    conf_list = [0.001, 0.01] 

    print("\n🚑 开始敏感度扫描... 寻找最佳数据...")
    
    # 用于存储所有结果以便最后汇总
    all_results = []

    for conf in conf_list:
        print(f"\n\n🔶 测试阈值 Conf = {conf} | IoU = 0.5")
        print("=" * 60)
        print(f"{'Method':<20} | {'Precision':<10} | {'Recall':<10} | {'mAP@0.5':<10}")
        print("-" * 60)
        
        for name, cfg in configs.items():
            # 设置开关
            set_leca_config(model, cfg)
            
            # 运行验证
            try:
                metrics = model.val(
                    data=DATA_YAML,
                    split='test',
                    project='runs/rescue_mission',
                    name=f'conf_{conf}_{name}',
                    batch=16,
                    device=DEVICE,
                    
                    # [关键] 动态调整阈值
                    conf=conf, 
                    iou=0.5,
                    
                    plots=False,
                    verbose=False
                )
                
                p = metrics.results_dict['metrics/precision(B)']
                r = metrics.results_dict['metrics/recall(B)']
                map50 = metrics.results_dict['metrics/mAP50(B)']
                
                print(f"{name:<20} | {p:.4f}     | {r:.4f}     | {map50:.4f}")
                
                # 保存本次结果
                all_results.append({
                    'conf': conf,
                    'name': name,
                    'p': p,
                    'r': r,
                    'map': map50
                })
                
            except Exception as e:
                print(f"{name:<20} | ❌ Error: {e}")

    # ================= 打印最终汇总表格 =================
    print("\n\n" + "="*80)
    print("📋 最终敏感度扫描汇总表 (Rescue Mission Summary)")
    print("="*80)
    print(f"{'Conf':<10} | {'Method':<20} | {'Precision':<10} | {'Recall':<10} | {'mAP@0.5':<10}")
    print("-" * 80)
    
    for res in all_results:
        print(f"{res['conf']:<10} | {res['name']:<20} | {res['p']:.4f}     | {res['r']:.4f}     | {res['map']:.4f}")
    
    print("="*80)
    print("提示：请寻找 Full ULECA > Baseline (ECA) 的那组 Conf，直接填入论文表格！")

if __name__ == "__main__":
    main()