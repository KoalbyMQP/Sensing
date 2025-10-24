#!/usr/bin/env python3

from ultralytics import YOLO
import torch

def train_model_improved(dataset_path, model_variant="yolov8n", epochs=200, batch_size=32):
    """
    Train a YOLOv8 model with improved parameters for better performance.
    
    Args:
        dataset_path: Path to the dataset (should contain data.yaml)
        model_variant: YOLOv8 variant (yolov8n, yolov8s, yolov8m, etc.)
        epochs: Number of training epochs (increased for better convergence)
        batch_size: Batch size for training (increased for better training)
    
    Returns:
        Path to the trained model
    """
    from pathlib import Path
    
    # Ensure dataset_path is a Path object
    dataset_path = Path(dataset_path)
    
    # Check if data.yaml exists
    data_yaml = dataset_path / "data.yaml"
    if not data_yaml.exists():
        print(f"Error: data.yaml not found in {dataset_path}")
        return None
    
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load model
    model = YOLO(f"{model_variant}.pt")
    
    # Train model with improved parameters
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=device,
        # Improved training parameters
        lr0=0.01,          # Initial learning rate
        lrf=0.01,          # Final learning rate
        momentum=0.937,    # SGD momentum
        weight_decay=0.0005,  # Optimizer weight decay
        warmup_epochs=3,   # Warmup epochs
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        # Data augmentation
        hsv_h=0.015,       # HSV hue augmentation
        hsv_s=0.7,         # HSV saturation augmentation
        hsv_v=0.4,         # HSV value augmentation
        degrees=0.0,       # Rotation degrees
        translate=0.1,     # Translation fraction
        scale=0.5,         # Scale gain
        shear=0.0,         # Shear degrees
        perspective=0.0,   # Perspective gain
        flipud=0.0,        # Flip up-down probability
        fliplr=0.5,        # Flip left-right probability
        mosaic=1.0,        # Mosaic augmentation probability
        mixup=0.0,         # Mixup augmentation probability
        copy_paste=0.0,    # Copy-paste augmentation probability
        # Validation settings
        val=True,          # Validate during training
        plots=True,        # Generate training plots
        save=True,         # Save checkpoints
        save_period=10,    # Save checkpoint every N epochs
        # Optimization
        optimizer='auto',  # Optimizer selection
        close_mosaic=10,   # Disable mosaic augmentation for final N epochs
    )
    
    # Save trained model
    model_path = f"{model_variant}_improved.pt"
    model.save(model_path)
    
    print(f"Training completed!")
    print(f"Model saved to: {model_path}")
    
    # Print training results summary
    if results:
        print("\nTraining Results Summary:")
        print(f"Best mAP50: {results.get('metrics/mAP50(B)', 'N/A')}")
        print(f"Best mAP50-95: {results.get('metrics/mAP50-95(B)', 'N/A')}")
    
    return model_path

if __name__ == "__main__":
    # Use Chess-4 dataset (150 training images, 15 classes)
    dataset_path = "/Users/azieldawit/Desktop/School/WPI/MQP/Sensing/Vision/modules/Chess/data/annotated_data/Chess-4"
    
    # Train the model with improved parameters
    model_path = train_model_improved(
        dataset_path=dataset_path,
        model_variant="yolov8n",
        epochs=50,         # Quick test - ~1-2 hours on MPS
        batch_size=16      # Smaller batch for stability
    )
    
    print(f"Trained model: {model_path}")
