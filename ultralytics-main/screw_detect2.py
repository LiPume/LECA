from ultralytics import YOLO
import os

# 把所有执行逻辑都放进一个函数，比如 main()
def run_evaluation():
    # --- 1. 定义路径 ---
    MODEL_PATH = r"E:\AIpractice\detect\car_bolt_detection\ultralytics-main\runs\detect\car_bolt_LECA7_dis\weights\best.pt"
    DATA_YAML_PATH = r"E:\AIpractice\detect\car_bolt_detection\ultralytics-main\dataset\hardData\YOLODataset\hard_test_set2.yaml"
    PROJECT_NAME = "hard_test_evaluation_2"

    # --- 2. 检查文件 ---
    if not os.path.exists(MODEL_PATH):
        print(f"错误：模型文件未找到！ {MODEL_PATH}")
        return  # 使用 return 退出函数
    if not os.path.exists(DATA_YAML_PATH):
        print(f"错误：数据集 YAML 文件未找到！ {DATA_YAML_PATH}")
        return

    print("文件路径检查通过，正在加载模型...")

    # --- 3. 加载模型 ---
    try:
        model = YOLO(MODEL_PATH)
        print("模型加载成功！")
    except Exception as e:
        print(f"模型加载失败：{e}")
        return

    # --- 4. [核心] 运行评估 ---
    print(f"正在对 {DATA_YAML_PATH} 指定的 'test' 集进行评估...")
    results = model.val(
        data=DATA_YAML_PATH,
        split='test',
        conf=0.1,
        iou=0.5,

        classes=[0],

        save_hybrid=True,  # 虽然提示过时，但暂时还能用
        save_conf=True,
        project='runs/detect',
        name=PROJECT_NAME,
        exist_ok=True,
        plots=False
    )

    # --- 5. 如何解读结果 ---
    print("\n评估完成！")
    print("--- 自动计算的指标 (mAP, Precision, Recall) ---")
    print(f"Overall mAP@0.5: {results.box.map50}")  # 打印 mAP@0.5
    print(f"Overall mAP@0.5-0.95: {results.box.map}")  # 打印 mAP@0.5-0.95
    print(f"Overall Precision (P): {results.box.mp}")  # 打印 Mean Precision
    print(f"Overall Recall (R): {results.box.mr}")  # 打印 Mean Recall



# --- 重点：主程序入口 ---
# 只有当你直接运行 'python screw_detect.py' 时，
# __name__ 才会等于 '__main__'，下面的代码才会执行。
# 当子进程导入这个文件时，__name__ != '__main__'，
# run_evaluation() 就不会被调用，从而避免了死循环。
if __name__ == '__main__':
    run_evaluation()