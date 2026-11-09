import os
import cv2
import time
import argparse
from ultralytics import YOLO

def run_realtime_webcam(weights_path, cam_index, conf_threshold, imgsz):
    # 1. Kiểm tra file weights
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[!] Không tìm thấy file weights tại: {weights_path}")

    # 2. Nạp mô hình YOLO
    print(f"[*] Đang tải mô hình từ: {weights_path}")
    model = YOLO(weights_path)

    # 3. Mở luồng camera (0 là webcam mặc định của máy tính)
    print(f"[*] Đang khởi tạo camera (index: {cam_index})...")
    cap = cv2.VideoCapture(cam_index)
    
    # Thiết lập độ phân giải camera (1280x720) để tối ưu hiển thị
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[!] Không thể mở camera tại index {cam_index}. Vui lòng kiểm tra lại thiết bị!")
        return

    print("[*] Webcam đã sẵn sàng!")
    print("[*] Nhấn phím 'q' trên cửa sổ hiển thị để thoát chương trình.\n")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Không nhận được luồng hình ảnh từ camera.")
            break

        # 4. Chạy Detection + ByteTrack
        # stream=True giúp tối ưu hóa bộ nhớ cho luồng video thời gian thực
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False
        )

        # 5. Vẽ Bounding Box & Track ID lên frame
        annotated_frame = results[0].plot()

        # 6. Tính toán và vẽ FPS thời gian thực
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        # 7. Hiển thị cửa sổ nhận diện
        cv2.imshow("Traffic Camera - Real-time Detection & ByteTrack", annotated_frame)

        # 8. Nhấn phím 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[*] Đã dừng chương trình webcam.")
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Traffic Object Detection & ByteTrack via Webcam")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Đường dẫn file weights (.pt)")
    parser.add_argument("--cam", type=int, default=0, help="Index của webcam (mặc định: 0)")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng tin cậy (Confidence Threshold)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh xử lý")

    args = parser.parse_args()
    run_realtime_webcam(args.weights, args.cam, args.conf, args.imgsz)