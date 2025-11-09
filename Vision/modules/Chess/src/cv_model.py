import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, ImgDetectionsBridge

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

        visualizer = dai.RemoteConnection(httpPort=8082)

        with dai.Pipeline(self.device) as pipeline:
            cam = pipeline.create(dai.node.Camera).build()
            nn_archive = dai.NNArchive(self.model_path)

            # Create the neural network node
            nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
                cam.requestOutput((640, 640), type=img_frame_type, fps=30),
                nn_archive,
            )

            # Bridge the detections to the visualizer
            label_encoding = {
                k: v for k, v in enumerate(
                    nn_archive.getConfig().model.heads[0].metadata.classes
                )
            }
            bridge = pipeline.create(ImgDetectionsBridge).build(nn_with_parser.out)
            bridge.setLabelEncoding(label_encoding)

            # Configure the visualizer node
            visualizer.addTopic("Video", nn_with_parser.passthrough, "images")
            visualizer.addTopic("Detections", bridge.out, "detections")

            pipeline.start()
            visualizer.registerPipeline(pipeline)

            while pipeline.isRunning():
                key = visualizer.waitKey(1)
                if key == ord("q"):
                    print("Got q key from the remote connection!")
                    break
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
        results = []
        for detection in img_detections.detections:
            class_name = detection.labelName
            center_x, center_y = self.get_bbox_center(
                detection.xmin, detection.ymin, 
                detection.xmax, detection.ymax,
                img_width, img_height
            )
            results.append((class_name, (center_x, center_y)))
        return results

    
    def predict(self, imgFrame: dai.ImgFrame = None) -> list[tuple[str, tuple[int, int]]]:
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
            nn_with_parser = pipeline.create(ParsingNeuralNetwork).build(
                cam.requestOutput((640, 640), type=img_frame_type, fps=30),
                nn_archive,
            )
            
            # Create output queue to get raw predictions from nn_with_parser.out
            # This is the parsed detections output (as shown in Luxonis example where
            # nn_with_parser.out goes to visualizer as "Detections")
            nn_output_queue = nn_with_parser.out.createOutputQueue(maxSize=1, blocking=True)
            
            # Start the pipeline
            pipeline.start()
            
            # Wait for and get the neural network output (blocking call)
            img_detections = nn_output_queue.get()
            
            # Print the raw predictions
            print("Raw predictions from model:")
            print(img_detections)
            print(f"Output type: {type(img_detections)}")
            
            # Process detections to get class names and center coordinates
            # Image size is 640x640 based on the requestOutput call
            img_width, img_height = 640, 640
            results = self.process_detections(img_detections, img_width, img_height)
            
            # Print processed results
            print(f"\nFound {len(results)} detections:")
            for class_name, (center_x, center_y) in results:
                print(f"  {class_name}: center at ({center_x}, {center_y})")
        
        return results
