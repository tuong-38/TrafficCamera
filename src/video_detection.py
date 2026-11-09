import os
import cv2
import time
import argparse
from pathlib import Path
from ultralytics import YOLO

def run_video_tracking(video_path, weights_path, conf_threshold, output_dir):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[!] Không tìm thấy file weights tại: {weights_path}")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"[!] Không tìm thấy file video tại: {video_path}")

    # 1. Load mô hình
    print(f"[*] Đang tải mô hình từ: {weights_path}")
    model = YOLO(weights_path)

    # 2. Mở file video bằng OpenCV
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 30

    # 3. Khởi tạo VideoWriter để lưu kết quả
    os.makedirs(output_dir, exist_ok=True)
    video_name = Path(video_path).stem
    save_path = os.path.join(output_dir, f"tracked_{video_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    print(f"[*] Bắt đầu xử lý video: {video_path}")
    print("[*] Nhấn 'q' trên màn hình hiển thị để dừng sớm.")

    prev_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 4. Chạy Detection + ByteTrack (persist=True để duy trì ID đối tượng qua các frame)
        results = model.track(source=frame, persist=True, tracker="bytetrack.yaml", conf=conf_threshold)

        # 5. Vẽ Bounding Box & Track ID
        annotated_frame = results[0].plot()

        # 6. Tính toán & hiển thị FPS
        curr_time = time.time()
        calc_fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        cv2.putText(annotated_frame, f"FPS: {calc_fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 7. Ghi frame vào file đầu ra
        out.write(annotated_frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"[✓] Theo dõi video hoàn tất! Kết quả đã lưu tại: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO + ByteTrack Video Detection & Tracking")
    parser.add_argument("--source", type=str, required=True, help="Đường dẫn file video đầu vào (.mp4, .avi)")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Đường dẫn file weights")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng tin cậy (Confidence Threshold)")
    parser.add_argument("--output", type=str, default="runs/predict", help="Thư mục lưu video kết quả")

    args = parser.parse_args()
    run_video_tracking(args.source, args.weights, args.conf, args.output)