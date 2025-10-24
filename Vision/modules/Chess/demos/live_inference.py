#!/usr/bin/env python3
"""
Live Chess Piece Detection Demo
Uses DepthAI OAK-D camera with trained YOLOv8 model for real-time chess piece detection
"""

import cv2
import numpy as np
import depthai as dai
from pathlib import Path
import sys
import json

# Add the core module to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "core" / "models" / "model_manager"))

from model_manager import DepthAIModelManager

class ChessDetectionDemo:
    def __init__(self, model_dir="/Users/azieldawit/Desktop/School/WPI/MQP/Sensing/Vision/modules/Chess/models/chess_detection_yolov8n_1", model_base="chess_detection_yolov8n"):
        """
        Initialize the chess detection demo
        
        Args:
            model_path: Path to the DepthAI model directory
        """
        self.model_dir = Path(model_dir)
        self.model_base = model_base
        self.model_manager = None
        self.pipeline = None
        self.device = None
        
        # Load model configuration
        self.load_model_config()
        
        # Initialize DepthAI pipeline
        self.setup_pipeline()
        
    def load_model_config(self):
        """Load model configuration from JSON file"""
        config_path = self.model_dir / f"{self.model_base}.json"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        self.class_names = self.config['mappings']['labels']
        self.num_classes = len(self.class_names)
        
        print(f"Loaded model with {self.num_classes} classes:")
        for i, class_name in enumerate(self.class_names):
            print(f"  {i}: {class_name}")
            
    def setup_pipeline(self):
        """Setup DepthAI pipeline for chess detection using v3 API"""
        # Create device first
        self.device = dai.Device()
        
        # Create pipeline with device (DepthAI v3 API)
        self.pipeline = dai.Pipeline(self.device)
        
        # Create camera node using v3 API - match working live stream pattern
        cam_rgb = self.pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.RGB)
        
        # Request full resolution output for display
        self.video_out = cam_rgb.requestFullResolutionOutput()
        self.q_rgb = self.video_out.createOutputQueue()
        
        # Create ImageManip node to resize and convert format (v3 API - CORRECT!)
        manip = self.pipeline.create(dai.node.ImageManip)
        manip.setMaxOutputFrameSize(2000000)  # 2MB to handle RGB888i 640x640
        
        # Use letterboxing to match YOLOv8 training behavior
        manip.initialConfig.setOutputSize(640, 640, dai.ImageManipConfig.ResizeMode.LETTERBOX)
        manip.initialConfig.setFrameType(dai.ImgFrame.Type.RGB888p)  # RGB888 planar (CHW format)
        
        # Create neural network node
        nn = self.pipeline.create(dai.node.NeuralNetwork)
        nn.setBlobPath(str(self.model_dir / f"{self.model_base}.blob"))
        
        # Link pipeline: camera -> manip -> nn
        self.video_out.link(manip.inputImage)
        manip.out.link(nn.input)
        
        # Create NN output queue
        self.q_nn = nn.out.createOutputQueue()
        
        print("Pipeline setup complete")
        
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except:
            pass
        
    def start_device(self):
        """Start the DepthAI device"""
        try:
            # Pipeline is already created with device, just start it
            self.pipeline.start()
            print("Device started successfully")
            return True
        except Exception as e:
            print(f"Failed to start device: {e}")
            return False
            
    def decode_yolo_output(self, output_data, input_shape=(640, 640), frame_count=0):
        """
        Decode YOLOv8 output to bounding boxes
        
        Args:
            output_data: Raw neural network output
            input_shape: Input image shape (height, width)
            
        Returns:
            List of detections with format: [x1, y1, x2, y2, confidence, class_id]
        """
        # YOLOv8 output format: [batch, 19, 8400] where 19 = 4 (bbox) + 15 (classes)
        # Remove batch dimension if present
        if output_data.ndim == 3:
            output_data = output_data[0]  # Remove batch dimension
        
        # Reshape to [19, 8400]
        output = output_data.reshape((self.num_classes + 4, -1))
        
        # Extract bounding boxes and class probabilities
        bbox_coords = output[:4, :]  # [4, 8400] - center_x, center_y, width, height
        class_probs = output[4:, :]  # [15, 8400]
        
        # Find detections above confidence threshold
        confidence_threshold = 0.00015  # Filter out most noise, keep only top detections
        detections = []
        
        # Debug: Check some confidence values
        if frame_count % 60 == 0:  # Print every 60 frames to avoid spam
            sample_confidences = []
            for i in range(0, min(100, class_probs.shape[1]), 10):  # Sample first 100 predictions
                class_scores = class_probs[:, i]
                max_confidence = np.max(class_scores)
                sample_confidences.append(max_confidence)
            if sample_confidences:
                print(f"Frame {frame_count}: Sample max confidences: {sample_confidences[:10]}")
                print(f"Frame {frame_count}: Max confidence in sample: {max(sample_confidences):.6f}")
        
        for i in range(class_probs.shape[1]):
            # Get class with highest probability
            class_scores = class_probs[:, i]
            max_class_id = np.argmax(class_scores)
            confidence = class_scores[max_class_id]
            
            if confidence > confidence_threshold:
                # Decode bounding box coordinates
                x_center, y_center, width, height = bbox_coords[:, i]
                
                # Debug: Print raw coordinates for first few detections
                if len(detections) < 3:  # Only debug first 3 detections to avoid spam
                    print(f"  Raw coords {len(detections)}: center=({x_center:.3f}, {y_center:.3f}), size=({width:.3f}, {height:.3f})")
                
                # The model outputs pixel coordinates directly (not normalized)
                img_width, img_height = input_shape[1], input_shape[0]
                
                # No scaling needed - model already outputs pixel coordinates
                # Just ensure they're within bounds
                
                # Convert from center format to corner format
                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)
                
                # Debug: Print final coordinates for first few detections
                if len(detections) < 3:  # Only debug first 3 detections to avoid spam
                    print(f"  Final coords {len(detections)}: ({x1}, {y1}, {x2}, {y2})")
                
                # Clamp coordinates to image bounds
                x1 = max(0, min(x1, img_width - 1))
                y1 = max(0, min(y1, img_height - 1))
                x2 = max(0, min(x2, img_width - 1))
                y2 = max(0, min(y2, img_height - 1))
                
                # Only add if box has reasonable size
                if (x2 - x1) > 10 and (y2 - y1) > 10:
                    detections.append([x1, y1, x2, y2, confidence, max_class_id])
        
        return detections
        
    def draw_detections(self, frame, detections):
        """
        Draw bounding boxes and labels on frame
        
        Args:
            frame: Input frame
            detections: List of detections
            
        Returns:
            Frame with drawn detections
        """
        frame_with_detections = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2, confidence, class_id = detection
            
            # Draw bounding box
            cv2.rectangle(frame_with_detections, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{self.class_names[class_id]}: {confidence:.6f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw label background
            cv2.rectangle(frame_with_detections, 
                         (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), 
                         (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(frame_with_detections, label, 
                       (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return frame_with_detections
        
    def run_demo(self):
        """Run the live detection demo"""
        # Force cleanup of any existing windows first
        cv2.destroyAllWindows()
        cv2.waitKey(100)  # Give more time for cleanup
        
        if not self.start_device():
            return
            
        # Use the pre-created output queue
        q_rgb = self.q_rgb
        
        print("Starting live detection...")
        print("Press 'q' to quit, 's' to save current frame")
        print("Waiting for camera frames... (this may take a moment if camera is reconnecting)")
        
        frame_count = 0
        
        print(f"Pipeline running: {self.pipeline.isRunning()}")
        
        while self.pipeline.isRunning():
            # Get camera frame with timeout
            try:
                in_rgb = q_rgb.tryGet()
                if in_rgb is None:
                    continue
                frame = in_rgb.getCvFrame()
            except Exception as e:
                print(f"Frame capture error: {e}")
                continue
            
            # Debug: Print frame info
            if frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                print(f"Frame {frame_count}: Shape={frame.shape}, Type={frame.dtype}")
            
            # Get neural network output (non-blocking check)
            detections = []
            if self.q_nn.has():
                in_nn = self.q_nn.get()
                print(f"Frame {frame_count}: Got NN output, type: {type(in_nn)}")
                try:
                    # v3-style NN output parsing - get tensor data by layer name
                    tensor_data = in_nn.getTensor("output0")
                    if tensor_data is not None:
                        nn_output = np.array(tensor_data, dtype=np.float32)
                        print(f"Frame {frame_count}: NN output shape: {nn_output.shape}")
                        detections = self.decode_yolo_output(nn_output, input_shape=(640, 640), frame_count=frame_count)
                        print(f"Frame {frame_count}: Decoded {len(detections)} detections")
                    else:
                        print(f"Frame {frame_count}: No tensor data found")
                except Exception as e:
                    print(f"NN output parsing error: {e}")
                    # Try alternative parsing methods
                    try:
                        # Try getting layers and accessing data differently
                        layers = in_nn.getAllLayers()
                        if layers:
                            print(f"Debug: Found {len(layers)} layers")
                            # In v3, try to get the actual tensor data
                            layer_info = layers[0]
                            print(f"Debug: Layer info type: {type(layer_info)}")
                            if hasattr(layer_info, 'getData'):
                                nn_output = np.array(layer_info.getData(), dtype=np.float32)
                                print(f"Debug: Alternative parsing - shape: {nn_output.shape}")
                                detections = self.decode_yolo_output(nn_output, frame_count=frame_count)
                    except Exception as e2:
                        print(f"Alternative parsing also failed: {e2}")
            else:
                if frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                    print(f"Frame {frame_count}: No NN output available")

            # Draw detections on frame
            frame_with_detections = self.draw_detections(frame, detections)
            
            # Print detections to terminal for debugging (reduced output)
            if detections:
                print(f"Frame {frame_count}: Found {len(detections)} detections")
                # Only print first few detections to reduce spam
                for i, det in enumerate(detections[:3]):
                    x1, y1, x2, y2, confidence, class_id = det
                    class_name = self.class_names[class_id]
                    print(f"  {class_name} (conf: {confidence:.2f}) at ({x1}, {y1}, {x2}, {y2})")
                if len(detections) > 3:
                    print(f"  ... and {len(detections) - 3} more detections")
            
            # Display frame with bounding boxes
            cv2.imshow("Chess Piece Detection", frame_with_detections)
            
            # Debug: Check if window is created
            if frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                window_exists = cv2.getWindowProperty("Chess Piece Detection", cv2.WND_PROP_VISIBLE)
                print(f"Window visible: {window_exists}")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"chess_detection_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame_with_detections)
                print(f"Saved frame: {filename}")
            
            frame_count += 1
            
        # Cleanup
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Give OpenCV time to close windows
        if hasattr(self, 'device') and self.device:
            self.device.close()
        print("Demo ended")

def main():
    """Main function"""
    print("Chess Piece Detection Demo")
    print("=" * 40)
    
    # Environment-specific model paths
    model_dir = Path("/Users/azieldawit/Desktop/School/WPI/MQP/Sensing/Vision/modules/Chess/models/chess_detection_yolov8n_1")
    model_base = "chess_detection_yolov8n"

    # Validate required files
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        return
    blob_path = model_dir / f"{model_base}.blob"
    json_path = model_dir / f"{model_base}.json"
    if not blob_path.exists():
        print(f"Error: Blob not found: {blob_path}")
        return
    if not json_path.exists():
        print(f"Error: Config not found: {json_path}")
        return

    try:
        # Create and run demo
        demo = ChessDetectionDemo(model_dir=str(model_dir), model_base=model_base)
        demo.run_demo()
        
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure cleanup happens even if there's an error
        cv2.destroyAllWindows()
        cv2.waitKey(1)

if __name__ == "__main__":
    main()
