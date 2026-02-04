import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, ImgDetectionsBridge
from ultralytics import YOLO
import time
import cv2
class Model:

    def __init__(self, model_path, device) -> None:
        # Path to the compiled DepthAI model archive
        self.model_path = model_path
        self.device = device
        self.platform = self.device.getPlatform()
        

    
    def liveInference(self):

        img_frame_type = (
            dai.ImgFrame.Type.BGR888i if self.platform.name == "RVC4" else dai.ImgFrame.Type.BGR888p
        )

        with dai.Pipeline(self.device) as pipeline:
            cam = pipeline.create(dai.node.Camera).build()
            nn_archive = dai.NNArchive(self.model_path)

            # Create the neural network node
            nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
                cam.requestOutput((640, 640), type=img_frame_type, fps=30, enableUndistortion=False),
                nn_archive,
            )

            # Create queues to get image and detections (same as predict function)
            passthrough_queue = nn_with_parser.passthrough.createOutputQueue(maxSize=4, blocking=False)
            detections_queue = nn_with_parser.out.createOutputQueue(maxSize=4, blocking=False)

            pipeline.start()

            while pipeline.isRunning():
                # Get image and detections
                if passthrough_queue.has() and detections_queue.has():
                    frame = passthrough_queue.get().getCvFrame()
                    img_detections = detections_queue.get()
                    
                    # Get image dimensions
                    img_height, img_width = frame.shape[:2]
                    
                    # Draw bounding boxes and class names for each detection
                    for detection in img_detections.detections:
                        # Convert normalized coordinates to pixel coordinates
                        x1 = int(detection.xmin * img_width)
                        y1 = int(detection.ymin * img_height)
                        x2 = int(detection.xmax * img_width)
                        y2 = int(detection.ymax * img_height)
                        
                        # Draw bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Get class name
                        class_name = detection.labelName
                        
                        # Draw class name label
                        cv2.putText(frame, class_name, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Display frame with center dots
                    cv2.imshow("Chess Detection", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            
            cv2.destroyAllWindows()
        return

    def requestImage(self, size: tuple[int, int]) -> dai.ImgFrame:

        img_frame_type = (
            dai.ImgFrame.Type.BGR888i if self.platform.name == "RVC4" 
            else dai.ImgFrame.Type.BGR888p
        )
        
        with dai.Pipeline(self.device) as pipeline:
            cam = pipeline.create(dai.node.Camera).build()
            
            # Request output and create queue
            outputQueue = cam.requestOutput(size, type=img_frame_type, fps=30).createOutputQueue()
            
            pipeline.start()
            
            # Get the ImgFrame object (not converted to CV2)
            imgFrame = outputQueue.get()
            
            return imgFrame  # Return the DepthAI ImgFrame object

    

    def get_center(self, tl, tr, bl, br) -> tuple[int, int]:
        return (((tl[0]+tr[0]+bl[0]+br[0])//4), ((tl[1]+tr[1]+bl[1]+br[1])//4))

    def get_bbox_center(self, xmin: float, ymin: float, xmax: float, ymax: float, 
                        img_width: int, img_height: int) -> tuple[int, int]:
        """
        Calculate the center pixel coordinates from normalized bounding box coordinates.
        
        Args:
            xmin, ymin, xmax, ymax: Normalized coordinates (0-1 range)
            img_width, img_height: Image dimensions in pixels
            
        Returns:
            (center_x, center_y) in pixel coordinates
        """
        center_x = int((xmin + xmax) / 2 * img_width)
        center_y = int((ymin + ymax) / 2 * img_height)
        return (center_x, center_y)
    
    def process_detections(self, img_detections: dai.ImgDetections, 
                          img_width: int, img_height: int) -> list[tuple[str, tuple[int, int]]]:
        """
        Process all detections and extract class names with center pixel coordinates.
        
        Args:
            img_detections: The ImgDetections object from the model
            img_width, img_height: Image dimensions in pixels
            
        Returns:
            List of tuples: [(class_name, (center_x, center_y)), ...]
        """

        resize_factor_x = 2104/640
        resize_factor_y = 1560/640

        results = []
        for detection in img_detections.detections:
            class_name = detection.labelName
            center_x, center_y = self.get_bbox_center(
                detection.xmin, detection.ymin, 
                detection.xmax, detection.ymax,
                img_width, img_height
            )
            center_x *= resize_factor_x
            center_y *= resize_factor_y
            results.append((class_name, (center_x, center_y)))
        return results

    
    def predict(self, distortion: bool) -> list[tuple[str, tuple[int, int]]]:
        """
        Run inference on a frame and print raw predictions.
        Creates its own pipeline to capture a frame and run inference.       
        
        """
        
        img_frame_type = (
            dai.ImgFrame.Type.BGR888i if self.platform.name == "RVC4" 
            else dai.ImgFrame.Type.BGR888p
        )
        
        # Initialize results list
        results = []
        
        # Use context manager for proper cleanup (as in Luxonis example)
        with dai.Pipeline(self.device) as pipeline:
            # Create camera node (following example - no socket specified)
            cam = pipeline.create(dai.node.Camera).build()
            
            # Load model archive
            nn_archive = dai.NNArchive(self.model_path)

            # Create the neural network node with parser (following example pattern)

            if distortion:
                nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
                    cam.requestOutput((640, 640), resizeMode=dai.ImgResizeMode.STRETCH, type=img_frame_type, fps=30, enableUndistortion=False),
                    nn_archive,
                )
            else:
                nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
                    cam.requestOutput((640, 640), resizeMode=dai.ImgResizeMode.STRETCH, type=img_frame_type, fps=30, enableUndistortion=True),
                    nn_archive,
                )
            
            # Create output queue to get raw predictions from nn_with_parser.out
            # This is the parsed detections output (as shown in Luxonis example where
            # nn_with_parser.out goes to visualizer as "Detections")
            nn_output_queue = nn_with_parser.out.createOutputQueue(maxSize=1, blocking=True)
            
            # Start the pipeline
            pipeline.start()
            
            
            # Skip first 10 frames to flush queue and ensure stable detections
            for _ in range(20):
                nn_output_queue.get()
            
            # Get the neural network output (blocking call)
            img_detections = nn_output_queue.get()
            

            
            # Process detections to get class names and center coordinates
            # Image size is 640x640 based on the requestOutput call
            img_width, img_height = 640, 640
            results = self.process_detections(img_detections, img_width, img_height)
            
            # Print processed results
            print(f"\nFound {len(results)} detections:")
            for class_name, (center_x, center_y) in results:
                print(f"  {class_name}: center at ({center_x}, {center_y})")
        
        return results

    def predict2(self, yolov8_model_path: str, distortion: bool) -> list[tuple[str, tuple[int, int]]]:
        """
        Run YOLOv8 inference on Raspberry Pi for a single frame from OAK-D camera.
        Very simple - just get one image, run YOLO, get centers with rescale.
        """
        
        # Load YOLO model
        print(f"Loading YOLO model from {yolov8_model_path}...")
        yolo_model = YOLO(yolov8_model_path)
        print("YOLO model loaded!")
        
        # Get one frame from OAK-D camera
        with dai.Pipeline(self.device) as pipeline:
            cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.RGB)
            outputQueue = cam.requestFullResolutionOutput().createOutputQueue()
            pipeline.start()
            
            # Get one frame
            videoIn = outputQueue.get()
            frame = videoIn.getCvFrame()
        
        # Run YOLOv8 inference
        results = yolo_model.predict(frame, conf=0.25, verbose=False)
        result = results[0]
        
        # Get image dimensions
        img_height, img_width = frame.shape[:2]
        
        # Same rescale factors as process_detections
        resize_factor_x = 1
        resize_factor_y = 1
        
        # Process detections (same logic as process_detections)
        detections = []
        for box in result.boxes:
            # Get class name
            class_id = int(box.cls[0])
            class_name = yolo_model.names[class_id]
            
            # Get normalized bbox coordinates
            xmin, ymin, xmax, ymax = box.xyxyn[0].tolist()
            
            # Get center (same as get_bbox_center)
            center_x, center_y = self.get_bbox_center(xmin, ymin, xmax, ymax, img_width, img_height)
            
            # Apply rescale factors (same as process_detections)
            center_x *= resize_factor_x
            center_y *= resize_factor_y
            
            detections.append((class_name, (int(center_x), int(center_y))))
        
        # Print processed results (same as predict)
        # print(f"\nFound {len(detections)} detections:")
        # for class_name, (center_x, center_y) in detections:
        #     print(f"  {class_name}: center at ({center_x}, {center_y})")
        
        return detections
