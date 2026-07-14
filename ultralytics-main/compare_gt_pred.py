from ultralytics import YOLO
import cv2
import os

# ==========================
# 配置路径（修改为你的路径）
# ==========================
MODEL_PATH = r"E:\AIpractice\detect\car_bolt_detection\ultralytics-main\runs\detect\car_bolt_LECA7\weights\best.pt"
IMAGES_DIR = r"E:\AIpractice\detect\car_bolt_detection\ultralytics-main\dataset\hardData\YOLODataset\images\test"
LABELS_DIR = r"E:\AIpractice\detect\car_bolt_detection\ultralytics-main\dataset\hardData\YOLODataset\labels\test"
OUTPUT_DIR = r"E:\AIpractice\detect\car_bolt_detection\ultralytics-main\runs\ULECA"


# ==========================
# 帮助函数：读取标签
# ==========================
def load_label(label_path, img_w, img_h):
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, "r") as f:
        for line in f.readlines():
            cls, x, y, w, h = map(float, line.strip().split())
            boxes.append([
                int((x - w/2) * img_w),
                int((y - h/2) * img_h),
                int((x + w/2) * img_w),
                int((y + h/2) * img_h),
                int(cls)
            ])
    return boxes


# ==========================
# 主逻辑
# ==========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = YOLO(MODEL_PATH)
    print("模型加载成功！")

    img_list = sorted(os.listdir(IMAGES_DIR))

    for img_name in img_list:
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        img_path = os.path.join(IMAGES_DIR, img_name)
        label_path = os.path.join(LABELS_DIR, img_name.replace(".jpg", ".txt").replace(".png", ".txt"))

        # 读取图像
        img = cv2.imread(img_path)
        img_h, img_w = img.shape[:2]

        # ================ 1. 读取 GT 真实框（绿色） ================
        gt_boxes = load_label(label_path, img_w, img_h)
        for (x1, y1, x2, y2, cls_id) in gt_boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"GT:{cls_id}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # ================ 2. YOLO 预测框（红色） ================
        results = model.predict(img_path, conf=0.1, iou=0.5, verbose=False)
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"Pred:{cls_id} {conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # 保存最终图
        save_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(save_path, img)
        print(f"已保存：{save_path}")

    print("全部完成！结果保存在：", OUTPUT_DIR)


if __name__ == "__main__":
    main()
