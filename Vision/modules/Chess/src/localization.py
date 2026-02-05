
from ast import Tuple
import cv2
import depthai as dai
import time
import numpy as np


class Localize:

    def __init__(self, device) -> None:
        # Path to the compiled DepthAI model archive
        self.device = device
        self.platform = self.device.getPlatform()

    def request_image(self, size: tuple[int, int]) -> dai.ImgFrame:

        img_frame_type = (
            dai.ImgFrame.Type.BGR888i if self.platform.name == "RVC4" 
            else dai.ImgFrame.Type.BGR888p
        )
        
        with dai.Pipeline() as pipeline:
            hostCamera = pipeline.create(dai.node.Camera).build()
            aprilTagNode = pipeline.create(dai.node.AprilTag)
            #Max Resolution is 4208x3120
            hostCamera.requestOutput((4208, 3120), resizeMode=dai.ImgResizeMode.STRETCH).link(aprilTagNode.inputImage)
            passthroughOutputQueue = aprilTagNode.passthroughInputImage.createOutputQueue()
            outQueue = aprilTagNode.out.createOutputQueue()

            aprilTagMessage = outQueue.get()
            
            return aprilTagMessage, pipeline  # Return the DepthAI ImgFrame object

    @staticmethod
    def to_int(tag):
        return (int(tag.x), int(tag.y))

    @staticmethod
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

    def apriltag_detection(self, distortion: bool) -> dict:
        """Detect AprilTags from camera and return centers grouped by tag ID.
        Waits until all 32 unique tags are detected (each with at least 2 detections)."""
        
        with dai.Pipeline(self.device) as pipeline:
            hostCamera = pipeline.create(dai.node.Camera).build()
            aprilTagNode = pipeline.create(dai.node.AprilTag)

            if distortion:
                # Max Resolution is 4208x3120
                hostCamera.requestOutput((2104, 1560), resizeMode=dai.ImgResizeMode.STRETCH, enableUndistortion=False).link(aprilTagNode.inputImage)
                passthroughOutputQueue = aprilTagNode.passthroughInputImage.createOutputQueue()
                outQueue = aprilTagNode.out.createOutputQueue()
            else:
                # Max Resolution is 4208x3120
                hostCamera.requestOutput((2104, 1560), resizeMode=dai.ImgResizeMode.STRETCH, enableUndistortion=True).link(aprilTagNode.inputImage)
                passthroughOutputQueue = aprilTagNode.passthroughInputImage.createOutputQueue()
                outQueue = aprilTagNode.out.createOutputQueue()

            pipeline.start()
            
            id_to_centers = {}
            image_size_printed = False
            
            # Keep looping until we have all 32 unique tags, each with at least 2 detections
            while True:

                id_to_centers = {}

                aprilTagMessage = outQueue.get()
                assert(isinstance(aprilTagMessage, dai.AprilTags))
                aprilTags = aprilTagMessage.aprilTags
                
                # Print image size once
                if not image_size_printed:
                    passthroughImage = passthroughOutputQueue.get()
                    frame = passthroughImage.getCvFrame()
                    height, width = frame.shape[:2]
                    print(f"Image resolution: {width} x {height}")
                    image_size_printed = True

                # Process tags and group centers by ID
                for tag in aprilTags:
                    topLeft = self.to_int(tag.topLeft)
                    topRight = self.to_int(tag.topRight)
                    bottomRight = self.to_int(tag.bottomRight)
                    bottomLeft = self.to_int(tag.bottomLeft)

                    center = (int((topLeft[0] + bottomRight[0]) / 2), int((topLeft[1] + bottomRight[1]) / 2))
                    id_to_centers.setdefault(tag.id, []).append(center)
                
                # Count how many unique tags have at least 2 detections
                tags_with_two_or_more = sum(1 for centers in id_to_centers.values() if len(centers) >= 2)

                # Stop when we have all 32 unique tags, each with at least 2 detections
                if len(id_to_centers) >= 16 and tags_with_two_or_more >= 16:
                    break

            return id_to_centers

    
    def extract_grid_lines(self, id_to_centers: dict) -> tuple[list, list]:

        horizontal_lines = []
        vertical_lines = []

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

                # Classify as horizontal (row) or vertical (column)
                if dx > dy:  # More horizontal
                    horizontal_lines.append((tag_id, (p1, p2)))
                else:  # More vertical
                    vertical_lines.append((tag_id, (p1, p2)))

        return (horizontal_lines, vertical_lines)

    
    def get_squares(self, horizontal_lines: list, vertical_lines: list) -> dict:

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
                    
                    intersection = self.line_intersection(row_line, col_line)
                    if intersection is not None:

                        # Map to chess notation: A-H (columns), 1-8 (rows)
                        # Note: row 0 is top (row 8 in chess), row 7 is bottom (row 1 in chess)
                        chess_col = chr(ord('A') + col_idx)
                        chess_row = 8 - row_idx  # Flip row numbering
                        square_name = f"{chess_col}{chess_row}"
                        
                        chess_squares[square_name] = intersection

        return chess_squares

    
    def calibrate(self, distortion: bool):
       id_to_centers = self.apriltag_detection(distortion)
       horizontal_lines, vertical_lines = self.extract_grid_lines(id_to_centers)
       chess_squares = self.get_squares(horizontal_lines, vertical_lines)
       return chess_squares

    def localize(self, chess_peices: list[tuple[str, tuple[int, int]]], squares: dict) -> tuple[str, str]:

        """
        Map chess pieces to their closest chess squares using Euclidean distance.
        
        Args:
            chess_peices: List of tuples (class_name, (center_x, center_y))
            squares: Dictionary mapping square names (e.g., "A1") to center coordinates (x, y)
            
        Returns:
            Dictionary mapping square names to class names (empty string if no piece on square)
        """
        # Initialize result dictionary with all squares set to empty string
        full_board = {square_name: "" for square_name in squares.keys()}
        
        # For each detected chess piece, find the closest square
        for class_name, piece_center in chess_peices:
            #print(f"\n{class_name} at ({piece_center[0]}, {piece_center[1]})")
            min_distance = float('inf')
            closest_square = None
            
            # Calculate Euclidean distance to all squares
            for square_name, square_center in squares.items():
                # Calculate squared Euclidean distance (no need for sqrt for comparison)
                dx = piece_center[0] - square_center[0]
                dy = piece_center[1] - square_center[1]
                distance_squared = dx * dx + dy * dy
                distance = (distance_squared ** 0.5)  # Calculate actual distance for printing
                
                # Print every distance calculation
                #print(f"  Distance to {square_name} ({square_center[0]}, {square_center[1]}): {distance:.2f}")
                
                # Track the closest square
                if distance_squared < min_distance:
                    min_distance = distance_squared
                    closest_square = square_name
            
            #print(f"  -> Closest square: {closest_square} (distance: {(min_distance ** 0.5):.2f})")
            
            # Assign the class name to the closest square
            if closest_square is not None:
                full_board[closest_square] = class_name
        
        return full_board 

    def display_squares_live(self, squares: dict):
        """Display live video feed with dots at the center of each chess square.
        
        Args:
            squares: Dictionary mapping square names to center coordinates (x, y)
        """

        with dai.Pipeline(self.device) as pipeline:
            hostCamera = pipeline.create(dai.node.Camera).build()
            aprilTagNode = pipeline.create(dai.node.AprilTag)
            # Max Resolution is 4208x3120
            hostCamera.requestOutput((1920, 1080), resizeMode=dai.ImgResizeMode.STRETCH).link(aprilTagNode.inputImage)
            passthroughOutputQueue = aprilTagNode.passthroughInputImage.createOutputQueue()
            
            pipeline.start()
            
            print("Displaying square centers live. Press 'q' to quit.")
            
            while pipeline.isRunning():
                passthroughImage = passthroughOutputQueue.get()
                frame = passthroughImage.getCvFrame()
                
                # Draw red dot and label at center of each square
                for square_name, (center_x, center_y) in squares.items():
                    cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)
                    # Print square name next to the dot
                    cv2.putText(frame, square_name, (center_x + 10, center_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow("Chess Board Squares", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            cv2.destroyAllWindows()

    
    def map_point_640_to_1080(self, pt_640, calib, cam=dai.CameraBoardSocket.RGB):
        """
        Map a point from 640x640 preview to 1920x1080 full-resolution RGB image
        for an OAK-D-Lite using DepthAI v3 calibration.
        """
        # --- Get camera models ---
        # Full 1080p intrinsics / distortion
        K_1080 = np.array(calib.getCameraIntrinsics(cam, 1920, 1080), dtype=np.float32)
        D_1080 = np.array(calib.getDistortionCoefficients(cam), dtype=np.float32)

        # Preview 640x640 intrinsics / distortion
        K_640 = np.array(calib.getCameraIntrinsics(cam, 640, 640), dtype=np.float32)
        D_640 = np.array(calib.getDistortionCoefficients(cam), dtype=np.float32)

        # --- Step 1: Undistort the preview pixel into normalized coordinates ---
        pts = np.array([[pt_640]], dtype=np.float32)  # shape (1,1,2)

        # undistortPoints gives normalized (x, y) on the camera's ideal pinhole plane
        norm = cv2.undistortPoints(pts, K_640, D_640)  # shape (1,1,2)
        x, y = norm[0,0]

        # --- Step 2: Convert normalized point into 3D point (z=1) ---
        pt3d = np.array([[x, y, 1.0]], dtype=np.float32)

        # --- Step 3: Re-project into 1920x1080 with that camera’s model ---
        projected, _ = cv2.projectPoints(
            pt3d,
            rvec=np.zeros(3),
            tvec=np.zeros(3),
            cameraMatrix=K_1080,
            distCoeffs=D_1080
        )
        return tuple(projected[0,0])
    
    def map_point_1080_to_640(self, pt_1080, calib, cam=dai.CameraBoardSocket.RGB):
        """
        Map a point from 1920x1080 full-resolution RGB image to 640x640 preview
        for an OAK-D-Lite using DepthAI v3 calibration.
        """
        # --- Get camera models ---
        # Full 1080p intrinsics / distortion
        K_1080 = np.array(calib.getCameraIntrinsics(cam, 1920, 1080), dtype=np.float32)
        D_1080 = np.array(calib.getDistortionCoefficients(cam), dtype=np.float32)

        # Preview 640x640 intrinsics / distortion
        K_640 = np.array(calib.getCameraIntrinsics(cam, 640, 640), dtype=np.float32)
        D_640 = np.array(calib.getDistortionCoefficients(cam), dtype=np.float32)

        # --- Step 1: Undistort the 1080p pixel into normalized coordinates ---
        pts = np.array([[pt_1080]], dtype=np.float32)  # shape (1,1,2)

        # undistortPoints gives normalized (x, y) on the camera's ideal pinhole plane
        norm = cv2.undistortPoints(pts, K_1080, D_1080)  # shape (1,1,2)
        x, y = norm[0,0]

        # --- Step 2: Convert normalized point into 3D point (z=1) ---
        pt3d = np.array([[x, y, 1.0]], dtype=np.float32)
        
        # --- Step 3: Re-project into 640x640 with that camera's model ---
        projected, _ = cv2.projectPoints(
            pt3d,
            rvec=np.zeros(3),
            tvec=np.zeros(3),
            cameraMatrix=K_640,
            distCoeffs=D_640
        )

        return tuple(projected[0,0])
    

