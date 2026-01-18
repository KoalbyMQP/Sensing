#!/usr/bin/env python3

import cv2
import depthai as dai
import time
import numpy as np

with dai.Pipeline() as pipeline:
    
    hostCamera = pipeline.create(dai.node.Camera).build()
    aprilTagNode = pipeline.create(dai.node.AprilTag)
    #Max Resolution is 4208x3120
    hostCamera.requestOutput((4208, 3120), resizeMode=dai.ImgResizeMode.STRETCH).link(aprilTagNode.inputImage)
    passthroughOutputQueue = aprilTagNode.passthroughInputImage.createOutputQueue()
    outQueue = aprilTagNode.out.createOutputQueue()

    color = (0, 255, 0)
    startTime = time.monotonic()
    counter = 0
    fps = 0.0

    pipeline.start()
    while pipeline.isRunning():
        aprilTagMessage = outQueue.get()
        assert(isinstance(aprilTagMessage, dai.AprilTags))
        aprilTags = aprilTagMessage.aprilTags

        counter += 1
        currentTime = time.monotonic()
        if (currentTime - startTime) > 1:
            fps = counter / (currentTime - startTime)
            counter = 0
            startTime = currentTime

        passthroughImage: dai.ImgFrame = passthroughOutputQueue.get()
        frame = passthroughImage.getCvFrame()

        def to_int(tag):
            return (int(tag.x), int(tag.y))
        
        def line_intersection(line1, line2):
            """Find intersection point of two lines defined by two points each.
            Returns (x, y) or None if lines are parallel."""
            (x1, y1), (x2, y2) = line1
            (x3, y3), (x4, y4) = line2
            
            # Calculate denominators
            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-6:
                return None  # Lines are parallel
            
            # Calculate intersection
            t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (int(x), int(y))

        # Group detected tags by ID so we can connect matching pairs
        id_to_centers = {}

        for tag in aprilTags:
            print("Found tag: ", tag.id)
            topLeft = to_int(tag.topLeft)
            topRight = to_int(tag.topRight)
            bottomRight = to_int(tag.bottomRight)
            bottomLeft = to_int(tag.bottomLeft)

            center = (int((topLeft[0] + bottomRight[0]) / 2), int((topLeft[1] + bottomRight[1]) / 2))

            cv2.line(frame, topLeft, topRight, color, 2, cv2.LINE_AA, 0)
            cv2.line(frame, topRight,bottomRight, color, 2, cv2.LINE_AA, 0)
            cv2.line(frame, bottomRight,bottomLeft, color, 2, cv2.LINE_AA, 0)
            cv2.line(frame, bottomLeft,topLeft, color, 2, cv2.LINE_AA, 0)

            idStr = "ID: " + str(tag.id)
            cv2.putText(frame, idStr, center, cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            cv2.putText(frame, f"fps: {fps:.1f}", (200, 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

            # Collect centers per ID for drawing connection lines later
            id_to_centers.setdefault(tag.id, []).append(center)

        # Draw connection lines and separate into horizontal (rows) and vertical (columns)
        horizontal_lines = []  # List of (tag_id, line_points) for rows
        vertical_lines = []    # List of (tag_id, line_points) for columns
        
        for tag_id, centers in id_to_centers.items():
            if len(centers) >= 2:
                # If more than 2 detections with same ID, connect the two farthest points
                max_dist = -1.0
                p1, p2 = centers[0], centers[0]
                for i in range(len(centers)):
                    for j in range(i + 1, len(centers)):
                        dx = centers[i][0] - centers[j][0]
                        dy = centers[i][1] - centers[j][1]
                        dist = dx * dx + dy * dy
                        if dist > max_dist:
                            max_dist = dist
                            p1, p2 = centers[i], centers[j]
                
                # Determine if line is horizontal or vertical based on angle
                dx = abs(p2[0] - p1[0])
                dy = abs(p2[1] - p1[1])
                
                cv2.line(frame, p1, p2, (255, 0, 0), 2, cv2.LINE_AA, 0)
                
                # Classify as horizontal (row) or vertical (column)
                if dx > dy:  # More horizontal
                    horizontal_lines.append((tag_id, (p1, p2)))
                else:  # More vertical
                    vertical_lines.append((tag_id, (p1, p2)))
        
        # Find all intersections between horizontal and vertical lines
        chess_squares = {}  # Dictionary to store chess square positions: "A1" -> (x, y)
        
        if len(horizontal_lines) > 0 and len(vertical_lines) > 0:
            # Sort horizontal lines by y-coordinate (top to bottom)
            horizontal_lines.sort(key=lambda x: (x[1][0][1] + x[1][1][1]) / 2)
            # Sort vertical lines by x-coordinate (left to right)
            vertical_lines.sort(key=lambda x: (x[1][0][0] + x[1][1][0]) / 2)
            
            # Find intersections - each intersection IS the center of a chess square
            # We have 8 horizontal and 8 vertical lines, giving 8x8 = 64 intersections
            for row_idx, (row_id, row_line) in enumerate(horizontal_lines):
                if row_idx >= 8:  # Only process 8 rows
                    break
                for col_idx, (col_id, col_line) in enumerate(vertical_lines):
                    if col_idx >= 8:  # Only process 8 columns
                        break
                    
                    intersection = line_intersection(row_line, col_line)
                    if intersection is not None:
                        # Draw intersection point
                        cv2.circle(frame, intersection, 3, (0, 0, 255), -1)
                        
                        # Map to chess notation: A-H (columns), 1-8 (rows)
                        # Note: row 0 is top (row 8 in chess), row 7 is bottom (row 1 in chess)
                        chess_col = chr(ord('A') + col_idx)
                        chess_row = 8 - row_idx  # Flip row numbering
                        square_name = f"{chess_col}{chess_row}"
                        
                        chess_squares[square_name] = intersection
                        
                        # Draw square label
                        cv2.putText(frame, square_name, (intersection[0] - 10, intersection[1]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Print chess squares for debugging
            if len(chess_squares) > 0:
                print("\n=== Chess Square Positions ===")
                for square in sorted(chess_squares.keys()):
                    print(f"{square}: {chess_squares[square]}")
                print("==============================\n")

        cv2.imshow("detections", frame)
        if cv2.waitKey(1) == ord("q"):
            break