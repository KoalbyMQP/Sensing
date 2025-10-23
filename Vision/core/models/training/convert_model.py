#!/usr/bin/env python3

from ultralytics import YOLO
import depthai as dai
import json
from pathlib import Path

def convert_to_depthai(pytorch_model_path, model_name, output_dir):
    """
    Convert PyTorch model to DepthAI format.
    
    Args:
        pytorch_model_path: Path to the trained PyTorch model (.pt file)
        model_name: Name for the DepthAI model
        output_dir: Directory to save DepthAI model files
    
    Returns:
        Path to the model directory
    """
    print(f"Converting {pytorch_model_path} to DepthAI format...")
    
    # Create output directory
    model_dir = Path(output_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    
    # Load PyTorch model
    model = YOLO(pytorch_model_path)
    
    # Get classes from the model
    classes = model.names.values() if hasattr(model, 'names') else []
    print(f"Model classes: {list(classes)}")
    
    # Export to ONNX
    onnx_path = model_dir / f"{model_name}.onnx"
    #TODO: Change size of export to image size.
    model.export(format="onnx", imgsz=640)
    
    # Move ONNX file to model directory
    pytorch_path = Path(pytorch_model_path)
    onnx_source = pytorch_path.parent / f"{pytorch_path.stem}.onnx"
    if onnx_source.exists():
        onnx_source.rename(onnx_path)
    
    # Convert ONNX to blob using DepthAI
    blob_path = model_dir / f"{model_name}.blob"
    try:
        # Create a simple pipeline to compile the model
        pipeline = dai.Pipeline()
        nn = pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(str(onnx_path))
        
        # This will compile the model to blob format
        with dai.Device(pipeline) as device:
            pass  # The blob is created during pipeline creation
        
        # Find the generated blob file
        blob_files = list(onnx_path.parent.glob("*.blob"))
        if blob_files:
            blob_files[0].rename(blob_path)
        
    except Exception as e:
        print(f"Error converting to blob: {e}")
        return None
    
    # Create JSON configuration
    config = {
        "model": {
            "model_name": model_name,
            "zoo": "custom"
        },
        "nn_config": {
            "output_format": "detection",
            "NN_family": "YOLO",
            "NN_specific_metadata": {
                "classes": len(classes),
                "coordinates": 4,
                "anchors": [],
                "anchor_masks": {},
                "iou_threshold": 0.45,
                "confidence_threshold": 0.35
            }
        },
        "mappings": {
            "labels": classes
        },
        "version": 1
    }
    
    # Save JSON configuration
    config_path = model_dir / f"{model_name}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"Conversion completed!")
    print(f"Model directory: {model_dir}")
    print(f"Files created:")
    print(f"  - {config_path}")
    print(f"  - {blob_path}")
    
    return model_dir

if __name__ == "__main__":
    # Example usage
    pytorch_model = "yolov8n_trained.pt"  # Your trained model
    model_name = "chess_detection_yolov8n"
    
    # Convert model
    model_dir = convert_to_depthai(
        pytorch_model_path=pytorch_model,
        model_name=model_name,
        output_dir="Vision/modules/Chess/models"
    )
    
    if model_dir:
        print(f"Model ready for DepthAI: {model_dir}")
