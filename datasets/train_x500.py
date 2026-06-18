#!/usr/bin/env python3
"""Fine-tune YOLO to detect the gz x500 quad (our real-drone detector returns 0 on it). Trains on the
auto-labeled (background-subtraction) dataset from capture_x500_dataset.py.
  .venv\\Scripts\\python.exe datasets\\train_x500.py
"""
from ultralytics import YOLO

if __name__ == "__main__":
    YOLO("yolov8s.pt").train(
        data="datasets/x500_air/data.yaml",
        epochs=60, imgsz=960, batch=8,
        name="x500_air", project="runs/train",
        patience=20, verbose=True)
