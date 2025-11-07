import depthai as dai
from depthai_nodes.node import ParsingNeuralNetwork, ImgDetectionsBridge

class Model:

    


    def __init__(
        self, model_path, device) -> None:
    

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

    

    def get_centers(self, tl, tr, bl, br) -> tuple[int, int]:
        return (((tl[0]+tr[0]+bl[0]+br[0])//4), ((tl[1]+tr[1]+bl[1]+br[1])//4))

    
    def predict(self, imgFrame: dai.ImgFrame) -> list[tuple[str, tuple[int, int]]]:

        

        return
