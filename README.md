# Autonomous Driving Traffic Detection & Tracking

An end-to-end computer vision project for detecting and tracking vehicles, pedestrians, and cyclists in traffic footage. It uses the KITTI Vision Benchmark Dataset, Ultralytics YOLOv8, and ByteTrack.

![Evaluation metrics](assets/BoxPR_curve.png)

## Features and architecture

- **Core technology:** Python 3.10, PyTorch, Ultralytics YOLOv8n, OpenCV, and ByteTrack
- **Dataset:** KITTI Vision Benchmark, converted to YOLO format with the classes `vehicle`, `pedestrian`, and `cyclist`
- **Evaluation (50 epochs):** mAP@50 of 87.3% overall; 96.0% for vehicles, 79.5% for pedestrians, and 86.3% for cyclists
- **Hardware profile:** Training on a Google Colab Tesla T4 GPU; local inference on an NVIDIA RTX 3050
- **Docker:** CPU inference baseline

## Project structure

```text
TrafficCamera/
├── assets/                  # Evaluation plots and sample media
├── configs/
│   └── data.yaml            # YOLO dataset configuration
├── src/
│   ├── prepare_dataset.py   # Convert KITTI annotations to YOLO format
│   ├── train.py             # Train a custom YOLO model
│   ├── image_detection.py   # Run inference on an image
│   ├── video_detection.py   # Detect and track objects in a video
│   └── realtime_webcam.py   # Detect and track objects from webcam or RTSP
├── weights/                 # Place best.pt here; model weights are Git-ignored
├── datasets/                # Local KITTI data; Git-ignored
├── runs/                    # Inference and tracking outputs; Git-ignored
├── Dockerfile               # CPU-based Docker image
├── .dockerignore            # Files excluded from the Docker build context
├── .gitignore               # Files excluded from Git
├── requirements.txt         # Python dependencies
└── README.md
```

## Weights and dataset

The trained model file `weights/best.pt` is excluded from Git. Place the file in the `weights/` directory before running inference.

Raw and processed KITTI data belong under `datasets/`. These local data files are also excluded from GitHub. Run `src/prepare_dataset.py` to convert raw KITTI data and generate the dataset configuration when needed.

## Quick start: local execution

### 1. Set up the environment

Replace `<repository-url>` with the URL of your Git repository.

```bash
git clone <repository-url>
cd TrafficCamera

conda create -n cv_traffic python=3.10 -y
conda activate cv_traffic
pip install -r requirements.txt
```

### 2. Run inference

Image detection:

```bash
python src/image_detection.py --source assets/test_image.jpg --weights weights/best.pt
```

Video tracking (press `q` in the preview window to stop early):

```bash
python src/video_detection.py --source assets/test_video.mp4 --weights weights/best.pt
```

Real-time webcam or RTSP stream:

```bash
# Webcam (index 0)
python src/realtime_webcam.py --cam 0

# RTSP stream
python src/realtime_webcam.py --cam "rtsp://admin:password@192.168.1.100:554/live"
```

## Docker

The provided Docker image runs inference on the CPU. Before running it, make sure `weights/best.pt` exists on the host. The container mounts the host `weights/` directory so it can read the model, and mounts `runs/` so output files are available on the host after the container exits.

### 1. Build the image

Run from the project root:

```bash
docker build -t traffic-camera-cv:latest .
```

### 2. Run the default image inference

The Dockerfile uses `assets/BoxPR_curve.png` as the input image and saves the annotated result under `runs/predict/`.

PowerShell (Windows):

```powershell
docker run --rm `
  -v "${PWD}/weights:/app/weights" `
  -v "${PWD}/runs:/app/runs" `
  traffic-camera-cv:latest
```

Command Prompt (Windows):

```cmd
docker run --rm -v %cd%/weights:/app/weights -v %cd%/runs:/app/runs traffic-camera-cv:latest
```

Linux or macOS (Bash):

```bash
docker run --rm \
  -v "$(pwd)/weights:/app/weights" \
  -v "$(pwd)/runs:/app/runs" \
  traffic-camera-cv:latest
```
