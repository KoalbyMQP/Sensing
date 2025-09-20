#!/usr/bin/env python3

from ultralytics import YOLO

def train_model(dataset_path, model_variant="yolov8n", epochs=100):
    """
    Train a YOLOv8 model on the dataset.
    
    Args:
        dataset_path: Path to the dataset (should contain data.yaml)
        model_variant: YOLOv8 variant (yolov8n, yolov8s, yolov8m, etc.)
        epochs: Number of training epochs
    
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
    
    # Load model
    model = YOLO(f"{model_variant}.pt")
    
    # Train model
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=640,
        batch=16,
        device="auto"
    )
    
    # Save trained model
    model_path = f"{model_variant}_trained.pt"
    model.save(model_path)
    
    print(f"Training completed!")
    print(f"Model saved to: {model_path}")
    
    return model_path

if __name__ == "__main__":
    # Example usage - replace with your actual dataset path
    dataset_path = "Vision/modules/Chess/data"  # Your dataset path
    
    # Train the model
    model_path = train_model(
        dataset_path=dataset_path,
        model_variant="yolov8n",
        epochs=100
    )
    
    print(f"Trained model: {model_path}")
