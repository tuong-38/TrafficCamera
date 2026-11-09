import os
import cv2
import time
import argparse
from ultralytics import YOLO

def run_realtime_webcam(weights_path, cam_input, conf_threshold, imgsz):
    # 1. Kiểm tra file weights
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"[!] Không tìm thấy file weights tại: {weights_path}")

    # 2. Xử lý đầu vào camera: Ép kiểu int nếu là Webcam index (0, 1), giữ nguyên string nếu là RTSP/URL[cite: 14]
    cam_source = int(cam_input) if str(cam_input).isdigit() else cam_input

    # 3. Nạp mô hình YOLO
    print(f"[*] Đang tải mô hình từ: {weights_path}")
    model = YOLO(weights_path)

    # 4. Mở luồng camera (Webcam USB hoặc RTSP IP Stream)[cite: 14]
    print(f"[*] Đang khởi tạo camera từ nguồn: {cam_source}...")
    cap = cv2.VideoCapture(cam_source)
    
    # Thiết lập độ phân giải cho USB webcam (nếu là RTSP stream thì OpenCV sẽ tự nhận độ phân giải của luồng)
    if isinstance(cam_source, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"[!] Không thể mở camera từ nguồn '{cam_source}'. Vui lòng kiểm tra lại thiết bị hoặc đường dẫn network!")
        return

    print("[*] Camera đã sẵn sàng!")
    print("[*] Nhấn phím 'q' trên cửa sổ hiển thị để thoát chương trình.\n")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Không nhận được luồng hình ảnh từ camera.")
            break

        # 5. Chạy Detection + ByteTrack
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False
        )

        # 6. Vẽ Bounding Box & Track ID lên frame
        annotated_frame = results[0].plot()

        # 7. Tính toán và vẽ FPS thời gian thực
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

        # 8. Hiển thị cửa sổ nhận diện
        cv2.imshow("Traffic Camera - Real-time Detection & ByteTrack", annotated_frame)

        # 9. Nhấn phím 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[*] Đã dừng chương trình camera.")
            break

    # Giải phóng tài nguyên
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Traffic Object Detection & ByteTrack via Webcam/RTSP")
    parser.add_argument("--weights", type=str, default="weights/best.pt", help="Đường dẫn file weights (.pt)")
    # Đổi type=str để chấp nhận cả số (0, 1) lẫn chuỗi RTSP URL[cite: 14]
    parser.add_argument("--cam", type=str, default="0", help="Index của webcam (0, 1) HOẶC đường dẫn RTSP Stream")
    parser.add_argument("--conf", type=float, default=0.25, help="Ngưỡng tin cậy (Confidence Threshold)")
    parser.add_argument("--imgsz", type=int, default=640, help="Kích thước ảnh xử lý")

    args = parser.parse_args()
    run_realtime_webcam(args.weights, args.cam, args.conf, args.imgsz)