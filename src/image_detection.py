import os
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO

def run_image_detection(image_path, weights_path, conf_threshold, output_dir):
    # 1. Kiểm tra sự tồn tại của file weights và file ảnh
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[!] Không tìm thấy file weights tại: {weights_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"[!] Không tìm thấy file ảnh tại: {image_path}")

    # 2. Khởi tạo mô hình với weights đã huấn luyện
    print(f"[*] Đang tải mô hình từ: {weights_path}")
    model = YOLO(weights_path)

    # 3. Thực hiện dự đoán
    print(f"[*] Đang xử lý ảnh: {image_path} (Confidence threshold: {conf_threshold})")
    results = model.predict(source=image_path, conf=conf_threshold, save=False)

    # 4. Trực quan hóa kết quả (Bounding box, Label, Confidence Score)
    result_img = results[0].plot()

    # 5. Lưu ảnh kết quả
    os.makedirs(output_dir, exist_ok=True)
    image_name = Path(image_path).name
    save_path = os.path.join(output_dir, f"detected_{image_name}")
    cv2.imwrite(save_path, result_img)

    print(f"[✓] Nhận diện hoàn tất! Kết quả đã lưu tại: {save_path}")

    # 6. In thống kê số lượng đối tượng phát hiện được
    boxes = results[0].boxes
    print(f"[*] Tổng số đối tượng phát hiện: {len(boxes)}")
    for box in boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = float(box.conf[0])
        print(f"    - {cls_name}: {conf:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Image Detection Script")
    parser.add_argument("--source", type=str, default="assets/results.png", help="Đường dẫn file ảnh đầu vào")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Đường dẫn file weights (.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng tin cậy (Confidence Threshold)")
    parser.add_argument("--output", type=str, default="runs/predict", help="Thư mục lưu ảnh kết quả")
    
    args = parser.parse_args()
    run_image_detection(args.source, args.weights, args.conf, args.output)