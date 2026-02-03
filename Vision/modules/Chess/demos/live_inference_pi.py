#!/usr/bin/env python3

import depthai as dai
import cv2
from ultralytics import YOLO
import os

# Get the model path relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "/home/chess/Desktop/Sensing/Vision/modules/Chess/models/yolov11m_snake_final.pt")

def main():
    # Load YOLOv8 model

    print(f"Loading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("Model loaded successfully!")
    
    # Create OAKD device
    device = dai.Device()
    
    # Create pipeline with device
    with dai.Pipeline(device) as pipeline:
        # Create camera (DepthAI v3 API)
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.RGB)
        
        # Create output queue
        outputQueue = cam.requestFullResolutionOutput().createOutputQueue()
        
        # Start pipeline
        pipeline.start()
        
        print("OAK-D Lite Live Inference")
        print("Press 'q' in the video window to quit")
        
        frame_count = 0
        while pipeline.isRunning():
            # Get frame from OAKD camera
            videoIn = outputQueue.get()
            frame = videoIn.getCvFrame()
            frame_count += 1
            
            # Run YOLOv8 inference
            results = model.predict(frame, conf=0.25, verbose=False)
            
            # Get the first result (since we're processing one frame at a time)
            result = results[0]
            
            # Draw bounding boxes and labels
            annotated_frame = result.plot()  # This automatically draws boxes
            
            # Display frame with detections
            cv2.imshow("YOLOv8 Live Inference", annotated_frame)
            
            # Check for 'q' key to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cv2.destroyAllWindows()
        print("Live inference ended!")

if __name__ == "__main__":
    main()
