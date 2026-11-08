import os
import torch
from pathlib import Path
from ultralytics import YOLO

def train_custom_yolo():
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    # Sử dụng mô hình YOLOv8n pretrained
    model = YOLO("yolov8n.pt")
    
    config_path = Path("configs/data.yaml").resolve()
    
    results = model.train(
        data=str(config_path),
        epochs=50,
        imgsz=640,
        batch=16,
        device=device,
        project="runs/detect",
        name="kitti_yolo_run",
        exist_ok=True
    )

if __name__ == "__main__":
    train_custom_yolo()