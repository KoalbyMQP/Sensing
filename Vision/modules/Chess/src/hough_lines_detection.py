#!/usr/bin/env python3
"""
Chess Board Localization using Hough Line Transform
Tests the classic Hough line approach for detecting chess board grid lines
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

def detect_chess_board_hough(image_path, output_path=None):
    """
    Detect chess board using Hough Line Transform
    
    Args:
        image_path: Path to input chess board image
        output_path: Path to save output image (optional)
    
    Returns:
        tuple: (original_image, processed_image, grid_lines)
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    
    # Detect lines using Probabilistic Hough Transform
    lines = cv2.HoughLinesP(
        edges,
        rho=1,              # Distance resolution in pixels
        theta=np.pi/180,    # Angular resolution in radians
        threshold=100,       # Minimum number of intersections
        minLineLength=100,  # Minimum line length
        maxLineGap=10       # Maximum gap between line segments
    )
    
    # Create output image
    result = img.copy()
    
    # Draw detected lines
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Separate horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Classify lines as horizontal or vertical based on angle
            if abs(angle) < 15 or abs(angle - 180) < 15:
                horizontal_lines.append(line[0])
            elif abs(angle - 90) < 15 or abs(angle + 90) < 15:
                vertical_lines.append(line[0])
    
    # Draw horizontal lines in blue, vertical lines in red
    for line in horizontal_lines:
        x1, y1, x2, y2 = line
        cv2.line(result, (x1, y1), (x2, y2), (255, 0, 0), 3)  # Blue
    
    for line in vertical_lines:
        x1, y1, x2, y2 = line
        cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Red
    
    # Print statistics
    print(f"Hough Lines Detection Results:")
    print(f"Total lines detected: {len(lines) if lines is not None else 0}")
    print(f"Horizontal lines: {len(horizontal_lines)}")
    print(f"Vertical lines: {len(vertical_lines)}")
    
    # Save result if output path provided
    if output_path:
        cv2.imwrite(str(output_path), result)
        print(f"Result saved to: {output_path}")
    
    return img, result, {'horizontal': horizontal_lines, 'vertical': vertical_lines}

def main():
    parser = argparse.ArgumentParser(description='Chess board detection using Hough Line Transform')
    parser.add_argument('image_path', help='Path to chess board image')
    parser.add_argument('-o', '--output', help='Output path for result image')
    parser.add_argument('--show', action='store_true', help='Display result image')
    
    args = parser.parse_args()
    
    try:
        original, result, lines = detect_chess_board_hough(args.image_path, args.output)
        
        if args.show:
            # Display original and result side by side
            combined = np.hstack((original, result))
            cv2.imshow('Original vs Hough Lines Detection', combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()




