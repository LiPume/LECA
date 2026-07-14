import torch
from ultralytics import YOLO
import os
import sys

# ================= 🚀 配置区域 =================
# 1. 训练结果目录 (刚才并行训练保存的地方)
TRAIN_DIR = 'runs/ablation_retrain'

# 2. Hard Test Set 配置文件路径
DATA_YAML = '/home/lzx/car_bolt_detection/ultralytics-main/dataset/hardData/YOLODataset/hard_test_set.yaml'

# 3. 验证参数 (必须与 Windows 实验保持一致)
CONF_THRES = 0.1
IOU_THRES = 0.5
DEVICE = '5'  # 指定一张空闲显卡

# 4. 定义要验证的模型列表 (文件夹名 -> 显示名)
# 顺序很重要，要按照消融实验的逻辑顺序
models_to_eval = [
    ('1_ECA_Only',    'Baseline (ECA)'),
    ('2_ECA_Var',     '+ Variance'),
    ('3_ECA_Var_Rec', '+ Var & Rec'),
    ('4_ULECA_Full',  'Full ULECA')
]
# ==============================================

def main():
    print(f"🚀 开始最终消融实验评估 (Hard Test Set)")
    print(f"📂 数据集: {DATA_YAML}")
    print(f"⚙️  参数: conf={CONF_THRES}, iou={IOU_THRES}")
    print("=" * 80)

    results_list = []

    for folder_name, display_name in models_to_eval:
        # 构造权重路径
        weight_path = os.path.join(TRAIN_DIR, folder_name, 'weights', 'best.pt')
        
        print(f"\n>>> 正在评估: {display_name}")
        print(f"    路径: {weight_path}")

        if not os.path.exists(weight_path):
            print(f"❌ 错误: 找不到文件 {weight_path}")
            print("   请检查训练是否成功完成，或者文件夹名是否正确。")
            continue

        try:
            # 加载模型
            model = YOLO(weight_path)
            
            # 运行验证
            # 注意: 这里不需要再设置 LECA 的开关了
            # 因为这些模型是"真"训练出来的，结构参数已经固定在权重里了
            metrics = model.val(
                data=DATA_YAML,
                split='test',       # 必须是 test 集
                project='runs/ablation_eval',
                name=folder_name,
                batch=16,           # 保持一致
                device=DEVICE,
                conf=CONF_THRES,    # [核心] 对齐 Windows 参数
                iou=IOU_THRES,      # [核心] 对齐 Windows 参数
                plots=False,
                verbose=False
            )

            # 提取指标
            p = metrics.results_dict['metrics/precision(B)']
            r = metrics.results_dict['metrics/recall(B)']
            map50 = metrics.results_dict['metrics/mAP50(B)']
            
            res_entry = {
                'Method': display_name,
                'Precision': p,
                'Recall': r,
                'mAP@0.5': map50
            }
            results_list.append(res_entry)
            print(f"    ✅ 结果 -> P: {p:.4f} | R: {r:.4f} | mAP: {map50:.4f}")

        except Exception as e:
            print(f"    ❌ 运行出错: {e}")

    # ================= 打印最终表格 (直接复制到论文) =================
    print("\n\n" + "="*80)
    print("📄 FINAL ABLATION STUDY RESULTS (Copy this to Table X)")
    print("="*80)
    print(f"{'Method':<20} | {'Precision':<10} | {'Recall':<10} | {'mAP@0.5':<10}")
    print("-" * 80)
    
    for res in results_list:
        print(f"{res['Method']:<20} | {res['Precision']:.4f}     | {res['Recall']:.4f}     | {res['mAP@0.5']:.4f}")
    
    print("="*80)
    print("提示：如果 Baseline (ECA) 的 mAP 回归到了 0.95 左右，且 Full ULECA 最高，")
    print("      那么恭喜你，这组数据就是完美的！")

if __name__ == "__main__":
    main()


