import depthai as dai
import cv2

# Create pipeline
pipeline = dai.Pipeline()

# Define source and output - using ColorCamera and SPIOut (DepthAI 3.0 API)
camRgb = pipeline.create(dai.node.ColorCamera)
xoutRgb = pipeline.create(dai.node.SPIOut)

# Set properties
camRgb.setPreviewSize(640, 480)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)

# Linking
camRgb.preview.link(xoutRgb.input)
xoutRgb.setStreamName("rgb")

# Connect to device and start pipeline
with dai.Device() as device:
    device.startPipeline(pipeline)
    # Get output queue
    qRgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    
    # Take a picture
    inRgb = qRgb.get()  # blocking call, will wait until a new data has arrived
    frame = inRgb.getCvFrame()
    
    # Save the image
    cv2.imwrite("oakd_picture.jpg", frame)
    print("Picture saved as oakd_picture.jpg")