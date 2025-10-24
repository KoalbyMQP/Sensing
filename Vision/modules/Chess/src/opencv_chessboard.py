#!/usr/bin/env python3
"""
Chess Board Localization using OpenCV's findChessboardCorners
Tests the specialized chess board detection function
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

def detect_chess_board_opencv(image_path, output_path=None, board_size=(8, 8)):
    """
    Detect chess board using OpenCV's specialized chess board detection
    
    Args:
        image_path: Path to input chess board image
        output_path: Path to save output image (optional)
        board_size: Size of the chess board (width, height) in squares
    
    Returns:
        tuple: (original_image, processed_image, corners, success)
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Find chess board corners
    # Note: findChessboardCorners expects (width, height) = (columns, rows)
    ret, corners = cv2.findChessboardCorners(
        blurred,
        board_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + 
              cv2.CALIB_CB_NORMALIZE_IMAGE + 
              cv2.CALIB_CB_FILTER_QUADS
    )
    
    # Create output image
    result = img.copy()
    
    if ret:
        # Refine corner positions for sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(
            blurred, 
            corners, 
            (11, 11), 
            (-1, -1), 
            criteria
        )
        
        # Draw the detected corners
        cv2.drawChessboardCorners(result, board_size, corners_refined, ret)
        
        # Draw grid lines connecting the corners
        draw_chess_grid(result, corners_refined, board_size)
        
        # Print corner coordinates for analysis
        print(f"Chess board corners detected successfully!")
        print(f"Number of corners: {len(corners_refined)}")
        print(f"Expected corners: {board_size[0] * board_size[1]}")
        
        # Print first few corner coordinates
        print("First 5 corner coordinates:")
        for i in range(min(5, len(corners_refined))):
            x, y = corners_refined[i][0]
            print(f"  Corner {i+1}: ({x:.2f}, {y:.2f})")
        
    else:
        print("Failed to detect chess board corners")
        print("Trying with different parameters...")
        
        # Try with different flags
        ret2, corners2 = cv2.findChessboardCorners(
            blurred,
            board_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        )
        
        if ret2:
            print("Success with simplified flags!")
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_refined = cv2.cornerSubPix(
                blurred, 
                corners2, 
                (11, 11), 
                (-1, -1), 
                criteria
            )
            cv2.drawChessboardCorners(result, board_size, corners_refined, ret2)
            draw_chess_grid(result, corners_refined, board_size)
            ret = ret2
            corners = corners_refined
        else:
            print("Still failed. Board might not be clearly visible or properly oriented.")
    
    # Save result if output path provided
    if output_path:
        cv2.imwrite(str(output_path), result)
        print(f"Result saved to: {output_path}")
    
    return img, result, corners if ret else None, ret

def draw_chess_grid(image, corners, board_size):
    """
    Draw grid lines on the chess board
    
    Args:
        image: Image to draw on
        corners: Detected corner points
        board_size: Size of the chess board
    """
    if corners is None:
        return
    
    corners = corners.reshape(board_size[1], board_size[0], 2)  # Reshape to grid
    
    # Draw horizontal lines
    for i in range(board_size[1]):
        for j in range(board_size[0] - 1):
            pt1 = tuple(map(int, corners[i, j]))
            pt2 = tuple(map(int, corners[i, j + 1]))
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)
    
    # Draw vertical lines
    for i in range(board_size[1] - 1):
        for j in range(board_size[0]):
            pt1 = tuple(map(int, corners[i, j]))
            pt2 = tuple(map(int, corners[i + 1, j]))
            cv2.line(image, pt1, pt2, (0, 255, 0), 2)

def create_square_coordinates(corners, board_size):
    """
    Create coordinate mapping for each square on the chess board
    
    Args:
        corners: Detected corner points
        board_size: Size of the chess board
    
    Returns:
        Dictionary mapping chess notation to square coordinates
    """
    if corners is None:
        return None
    
    corners = corners.reshape(board_size[1], board_size[0], 2)
    squares = {}
    
    # Create mapping from chess notation (a1-h8) to square coordinates
    files = 'abcdefgh'  # columns
    ranks = '12345678'  # rows
    
    for i in range(board_size[1] - 1):  # rows
        for j in range(board_size[0] - 1):  # columns
            square_name = files[j] + ranks[7-i]  # Chess notation (a1, b1, etc.)
            
            # Get the four corners of the square
            top_left = corners[i, j]
            top_right = corners[i, j + 1]
            bottom_left = corners[i + 1, j]
            bottom_right = corners[i + 1, j + 1]
            
            # Calculate center of the square
            center_x = (top_left[0] + top_right[0] + bottom_left[0] + bottom_right[0]) / 4
            center_y = (top_left[1] + top_right[1] + bottom_left[1] + bottom_right[1]) / 4
            
            squares[square_name] = {
                'center': (center_x, center_y),
                'corners': [top_left, top_right, bottom_left, bottom_right]
            }
    
    return squares

def main():
    parser = argparse.ArgumentParser(description='Chess board detection using OpenCV findChessboardCorners')
    parser.add_argument('image_path', help='Path to chess board image')
    parser.add_argument('-o', '--output', help='Output path for result image')
    parser.add_argument('--board-size', nargs=2, type=int, default=[8, 8],
                       help='Chess board size (width height)')
    parser.add_argument('--show', action='store_true', help='Display result image')
    parser.add_argument('--create-mapping', action='store_true', 
                       help='Create and print square coordinate mapping')
    
    args = parser.parse_args()
    
    try:
        original, result, corners, success = detect_chess_board_opencv(
            args.image_path, args.output, tuple(args.board_size)
        )
        
        if args.create_mapping and success:
            squares = create_square_coordinates(corners, tuple(args.board_size))
            if squares:
                print("\nChess Square Coordinate Mapping:")
                for square, coords in squares.items():
                    center = coords['center']
                    print(f"{square}: center=({center[0]:.1f}, {center[1]:.1f})")
        
        if args.show:
            # Display original and result side by side
            combined = np.hstack((original, result))
            cv2.imshow('Original vs OpenCV Chess Board Detection', combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()


