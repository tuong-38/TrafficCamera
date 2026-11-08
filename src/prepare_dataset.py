import os
import shutil
import random
from pathlib import Path
from PIL import Image

# Định nghĩa các lớp đối tượng cần nhận diện trong giao thông
CLASS_MAPPING = {
    'Car': 0,
    'Van': 0,         # Gom Van vào nhóm Car/Vehicle hoặc tách riêng tùy nhu cầu
    'Truck': 0,       # Gom Truck vào nhóm Vehicle (Class 0: Vehicle)
    'Pedestrian': 1,  # Class 1: Pedestrian
    'Person_sitting': 1,
    'Cyclist': 2      # Class 2: Cyclist
}

CLASS_NAMES = ['vehicle', 'pedestrian', 'cyclist']

def convert_kitti_to_yolo_bbox(size, box):
    """
    Chuyển đổi từ bbox KITTI (xmin, ymin, xmax, ymax) 
    sang bbox YOLO (x_center, y_center, width, height) chuẩn hóa [0, 1].
    """
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    
    xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
    
    x_center = (xmin + xmax) / 2.0
    y_center = (ymin + ymax) / 2.0
    w = xmax - xmin
    h = ymax - ymin
    
    x_center *= dw
    w *= dw
    y_center *= dh
    h *= dh
    
    return (x_center, y_center, w, h)

def process_kitti_dataset(raw_dir, output_dir, train_ratio=0.8):
    """
    Xử lý dữ liệu raw KITTI và phân chia Train/Val theo chuẩn YOLO.
    """
    images_raw_dir = Path(raw_dir) / "image_2"
    labels_raw_dir = Path(raw_dir) / "label_2"

    if not images_raw_dir.exists() or not labels_raw_dir.exists():
        print(f"[!] Thư mục dữ liệu KITTI chưa đúng cấu trúc tại: {raw_dir}")
        print("Vui lòng đặt ảnh vào 'datasets/KITTI/raw/image_2' và nhãn vào 'datasets/KITTI/raw/label_2'")
        return

    # Đường dẫn thư mục đầu ra
    out_path = Path(output_dir)
    train_img_dir = out_path / "images" / "train"
    val_img_dir = out_path / "images" / "val"
    train_lbl_dir = out_path / "labels" / "train"
    val_lbl_dir = out_path / "labels" / "val"

    for p in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        p.mkdir(parents=True, exist_ok=True)

    image_files = list(images_raw_dir.glob("*.png")) + list(images_raw_dir.glob("*.jpg"))
    random.seed(42)
    random.shuffle(image_files)

    split_idx = int(len(image_files) * train_ratio)
    train_files = image_files[:split_idx]
    val_files = image_files[split_idx:]

    print(f"[*] Tổng số ảnh: {len(image_files)} | Train: {len(train_files)} | Val: {len(val_files)}")

    def convert_set(files, img_out, lbl_out):
        count = 0
        for img_path in files:
            label_path = labels_raw_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            with Image.open(img_path) as img:
                img_w, img_h = img.size

            yolo_labels = []
            with open(label_path, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    obj_type = parts[0]
                    if obj_type in CLASS_MAPPING:
                        cls_id = CLASS_MAPPING[obj_type]
                        # KITTI format: type, truncated, occluded, alpha, bbox(left, top, right, bottom), ...
                        left, top, right, bottom = map(float, parts[4:8])
                        yolo_box = convert_kitti_to_yolo_bbox((img_w, img_h), (left, top, right, bottom))
                        yolo_labels.append(f"{cls_id} {' '.join(f'{x:.6f}' for x in yolo_box)}")

            # Copy ảnh và lưu file label YOLO nếu có object hợp lệ
            if yolo_labels:
                shutil.copy(img_path, img_out / img_path.name)
                with open(lbl_out / f"{img_path.stem}.txt", 'w') as f_out:
                    f_out.write('\n'.join(yolo_labels))
                count += 1
        return count

    print("[*] Đang chuyển đổi tập Train...")
    c_train = convert_set(train_files, train_img_dir, train_lbl_dir)
    print("[*] Đang chuyển đổi tập Validation...")
    c_val = convert_set(val_files, val_img_dir, val_lbl_dir)

    print(f"[✓] Chuyển đổi thành công: {c_train} ảnh train, {c_val} ảnh val.")

def generate_data_yaml(output_yaml_path, dataset_dir):
    """
    Tự động tạo file configs/data.yaml cho YOLO training.
    """
    dataset_path = Path(dataset_dir).resolve()
    yaml_content = f"""path: {dataset_path.as_posix()}
train: images/train
val: images/val

names:
  0: vehicle
  1: pedestrian
  2: cyclist
"""
    Path(output_yaml_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_yaml_path, 'w') as f:
        f.write(yaml_content)
    print(f"[✓] Đã tạo file cấu hình: {output_yaml_path}")

if __name__ == "__main__":
    RAW_KITTI_DIR = "datasets/KITTI/raw"
    PROCESSED_DIR = "datasets/KITTI/processed"
    YAML_PATH = "configs/data.yaml"

    process_kitti_dataset(RAW_KITTI_DIR, PROCESSED_DIR)
    generate_data_yaml(YAML_PATH, PROCESSED_DIR)