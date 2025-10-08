#!/usr/bin/env python3

import depthai as dai
import cv2

def main():

    # Create device
    device = dai.Device()

    # Create pipeline with device
    with dai.Pipeline(device) as pipeline:
        # Create camera (DepthAI v3 API)
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.RGB)

        # Create output queue
        outputQueue = cam.requestFullResolutionOutput().createOutputQueue()

        print("OAK-D Lite Live Stream")
        print("Press 'q' in the video window to quit")

        # Start pipeline
        pipeline.start()

        print("Live stream started!")

        frame_count = 0
        while pipeline.isRunning():
            # Get frame
            videoIn = outputQueue.get()
            frame = videoIn.getCvFrame()
            frame_count += 1

            # Show frame
            cv2.imshow("OAK-D Lite Live Stream", frame)

            # Check for 'q' key to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        print("Live stream ended!")

if __name__ == "__main__":
    main()
