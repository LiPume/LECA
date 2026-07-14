from ultralytics import YOLO

# 加载你的 best.pt
model = YOLO(r"E:\AIpractice\detect\car_bolt_detection\ultralytics-main\runs\detect\test_run\weights\best.pt")

# 导出 ONNX，不使用 onnxslim
model.export(format="onnx", optimize=False, simplify=False, dynamic=False)
