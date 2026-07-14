from ultralytics import YOLO
import os

# --- 1. 定义文件路径和模型路径 ---
# 注意：在 Windows 路径中使用 r"" (原始字符串) 或使用双反斜杠 \\ 避免转义错误

# 您的模型权重路径
MODEL_PATH = r"E:\AIpractice\detect\car_bolt_detection\ultralytics-main\runs\detect\car_bolt4\weights\best.pt"

# 您的测试视频路径
VIDEO_PATH = r"E:\AIpractice\detect\car_bolt_detection\ultralytics-main\tests\4.mp4"

# 结果保存目录
# 运行后，结果将保存在 runs/detect/predict/ 目录下
PROJECT_NAME = "video_validation"

# --- 2. 检查文件是否存在 (可选但推荐) ---
if not os.path.exists(MODEL_PATH):
    print(f"错误：模型文件未找到！请检查路径：{MODEL_PATH}")
    # 尝试查找默认路径下的模型（如果您没有使用 'car_bolt' 命名）
    # 备用路径示例：r"E:\AIpractice\detect\car_bolt_detection\ultralytics-main\runs\detect\car_bolt\weights\best.pt"
    # return
if not os.path.exists(VIDEO_PATH):
    print(f"错误：视频文件未找到！请检查路径：{VIDEO_PATH}")
    # return

print("文件路径检查通过，正在加载模型...")

# --- 3. 加载模型 ---
try:
    # 加载您在训练过程中保存的最佳模型权重
    model = YOLO(MODEL_PATH)
    print("模型加载成功！")
except Exception as e:
    print(f"模型加载失败：{e}")
    # return

# --- 4. 对视频进行预测 (验证) ---
print(f"正在对视频文件 {VIDEO_PATH} 执行预测...")

# model.predict() 会处理视频、图像或文件夹，并返回结果
# save=True 会将带有检测框的视频文件保存到 runs/detect/predict*/ 目录下
results = model.predict(
    source=VIDEO_PATH,
    conf=0.25,  # 目标置信度阈值 (根据您的模型性能调整，默认0.25)
    iou=0.7,  # IoU阈值用于非极大值抑制 (NMS，默认0.7)
    save=True,  # 🌟 关键：保存带有检测结果的视频
    project='runs/detect',  # 指定顶层输出目录
    name=PROJECT_NAME,  # 指定本次运行的名称
    exist_ok=True  # 如果文件夹存在则覆盖
)

# --- 5. 结果输出 ---
print("\n预测完成！")
# Ultralytics 会将处理后的视频保存在 runs/detect/video_validation/ 下
output_dir = os.path.join('runs/detect', PROJECT_NAME)

# 尝试找到保存的视频文件名
saved_files = os.listdir(output_dir)
saved_video_name = [f for f in saved_files if f.endswith('.mp4') or f.endswith('.avi')]

if saved_video_name:
    print(f"🌟 结果视频已保存到：{os.path.join(output_dir, saved_video_name[0])}")
else:
    print(f"预测结果已保存到目录：{output_dir}。请检查该目录下的文件。")