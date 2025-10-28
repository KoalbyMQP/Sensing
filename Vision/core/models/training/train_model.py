#!/usr/bin/env python3
"""
Simple YOLOv8n Training Script for DepthAI
Trains a YOLOv8n model that can be converted for use with OAK-D cameras
"""

from ultralytics import YOLO
from pathlib import Path

# =================== CONFIGURATION ===================
# Change these paths to match your dataset
DATA_PATH = "dataset.yaml"  # Path to your dataset YAML file
MODEL_SIZE = "n"  # Options: 'n' (nano), 's', 'm', 'l', 'x'
EPOCHS = 100
IMG_SIZE = 640
BATCH = 16
# ======================================================


def train_yolo_model():
    """Train YOLOv8 model with configured parameters"""
    
    # Initialize model
    print(f"Loading YOLOv8{MODEL_SIZE} model...")
    model = YOLO(f'yolov8{MODEL_SIZE}.pt')
    
    # Display configuration
    print(f"\n{'='*50}")
    print(f"Training Configuration:")
    print(f"  Model: YOLOv8{MODEL_SIZE}")
    print(f"  Dataset: {DATA_PATH}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Batch size: {BATCH}")
    print(f"  Epochs: {EPOCHS}")
    print(f"{'='*50}\n")
    
    # Train the model
    print("Starting training...\n")
    results = model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        save=True,
        val=True,
    )
    
    # Display results
    print(f"\n{'='*50}")
    print(f"Training completed!")
    print(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"  Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")
    print(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Check if dataset exists
    if not Path(DATA_PATH).exists():
        print(f"Error: Dataset file not found: {DATA_PATH}")
        print("Please update DATA_PATH in the script with the correct path to your dataset YAML file.")
        exit(1)
    
    train_yolo_model()

