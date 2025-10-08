#!/usr/bin/env python3

import depthai as dai
import cv2
import threading
import sys
import time

# Global variables for communication between threads
current_frame = None
save_requested = False
saved_count = 0

def input_handler():
    """Handle terminal input in a separate thread"""
    global save_requested, saved_count
    while True:
        try:
            user_input = input()
            if user_input.strip():  # If something was typed
                save_requested = True
        except EOFError:
            break

def main():
    global current_frame, save_requested, saved_count

    # Create device
    device = dai.Device()

    # Create pipeline with device
    with dai.Pipeline(device) as pipeline:
        # Create camera (DepthAI v3 API)
        cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.RGB)

        # Create output queue
        outputQueue = cam.requestFullResolutionOutput().createOutputQueue()

        # Start pipeline
        pipeline.start()

        print("OAK-D Lite Dataset Capture")
        print("Live view running... type anything and press Enter to capture an image!")
        print("Press 'q' in the video window to quit")

        # Start input handler thread
        input_thread = threading.Thread(target=input_handler, daemon=True)
        input_thread.start()

        frame_count = 0
        while pipeline.isRunning():
            # Get frame
            videoIn = outputQueue.get()
            frame = videoIn.getCvFrame()
            frame_count += 1
            current_frame = frame  # Store current frame for saving

            # Show frame (no text overlay)
            cv2.imshow("OAK-D Lite Camera", frame)

            # Check for 'q' key to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break


        cv2.destroyAllWindows()
        print("Demo ended!")

if __name__ == "__main__":
    main()
