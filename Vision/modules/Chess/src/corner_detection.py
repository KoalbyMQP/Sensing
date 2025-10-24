#!/usr/bin/env python3
"""
Chess Board Localization using Corner Detection + Grid Fitting
Tests Harris and Shi-Tomasi corner detection for chess board grid detection
"""

import cv2
import numpy as np
import argparse
from pathlib import Path

def detect_chess_board_corners(image_path, output_path=None, corner_method='shi_tomasi'):
    """
    Detect chess board using corner detection and grid fitting
    
    Args:
        image_path: Path to input chess board image
        output_path: Path to save output image (optional)
        corner_method: 'harris' or 'shi_tomasi'
    
    Returns:
        tuple: (original_image, processed_image, corners, grid_points)
    """
    # Load image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Corner detection parameters
    max_corners = 100
    quality_level = 0.01
    min_distance = 10
    
    if corner_method == 'harris':
        # Harris corner detection
        corners = cv2.cornerHarris(blurred, 2, 3, 0.04)
        corners = cv2.dilate(corners, None)
        
        # Threshold for an optimal value
        corner_points = np.where(corners > 0.01 * corners.max())
        corner_coords = np.column_stack((corner_points[1], corner_points[0]))
        
    else:  # shi_tomasi
        # Shi-Tomasi corner detection
        corner_coords = cv2.goodFeaturesToTrack(
            blurred,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance
        )
        
        if corner_coords is not None:
            corner_coords = np.int0(corner_coords)
            corner_coords = corner_coords.reshape(-1, 2)
    
    # Create output image
    result = img.copy()
    
    # Draw detected corners
    if corner_coords is not None:
        for corner in corner_coords:
            x, y = corner
            cv2.circle(result, (x, y), 5, (0, 255, 0), -1)
    
    # Try to fit a grid to the corners
    grid_points = fit_grid_to_corners(corner_coords, img.shape)
    
    # Draw grid lines
    if grid_points is not None:
        draw_grid_lines(result, grid_points)
    
    # Print statistics
    print(f"Corner Detection Results ({corner_method}):")
    print(f"Corners detected: {len(corner_coords) if corner_coords is not None else 0}")
    print(f"Grid points: {len(grid_points) if grid_points is not None else 0}")
    
    # Save result if output path provided
    if output_path:
        cv2.imwrite(str(output_path), result)
        print(f"Result saved to: {output_path}")
    
    return img, result, corner_coords, grid_points

def fit_grid_to_corners(corners, image_shape):
    """
    Attempt to fit a 9x9 grid (8x8 squares) to detected corners
    
    Args:
        corners: Array of corner coordinates
        image_shape: Shape of the original image
    
    Returns:
        Array of grid intersection points or None if fitting fails
    """
    if corners is None or len(corners) < 20:
        return None
    
    # Sort corners by y-coordinate first, then x-coordinate
    sorted_corners = corners[np.lexsort((corners[:, 0], corners[:, 1]))]
    
    # Try to identify grid structure
    # This is a simplified approach - in practice, you'd want more sophisticated clustering
    
    # Group corners by approximate rows
    rows = []
    current_row = [sorted_corners[0]]
    row_threshold = 20  # pixels
    
    for i in range(1, len(sorted_corners)):
        if abs(sorted_corners[i][1] - sorted_corners[i-1][1]) < row_threshold:
            current_row.append(sorted_corners[i])
        else:
            if len(current_row) >= 3:  # Minimum corners per row
                rows.append(current_row)
            current_row = [sorted_corners[i]]
    
    if len(current_row) >= 3:
        rows.append(current_row)
    
    # Sort each row by x-coordinate
    for row in rows:
        row.sort(key=lambda x: x[0])
    
    # Filter rows with sufficient corners (expecting ~9 corners per row for 8x8 board)
    valid_rows = [row for row in rows if len(row) >= 5]
    
    if len(valid_rows) < 5:  # Need at least 5 rows for 8x8 board
        return None
    
    # Create grid points
    grid_points = []
    for row in valid_rows:
        grid_points.append(row)
    
    return grid_points

def draw_grid_lines(image, grid_points):
    """
    Draw grid lines based on detected grid points
    
    Args:
        image: Image to draw on
        grid_points: Array of grid intersection points
    """
    if grid_points is None:
        return
    
    # Draw horizontal lines
    for row in grid_points:
        if len(row) >= 2:
            for i in range(len(row) - 1):
                cv2.line(image, tuple(row[i]), tuple(row[i+1]), (255, 0, 0), 2)
    
    # Draw vertical lines (simplified - connect corresponding points in rows)
    if len(grid_points) >= 2:
        for col_idx in range(min(len(row) for row in grid_points)):
            for row_idx in range(len(grid_points) - 1):
                if col_idx < len(grid_points[row_idx]) and col_idx < len(grid_points[row_idx + 1]):
                    pt1 = tuple(grid_points[row_idx][col_idx])
                    pt2 = tuple(grid_points[row_idx + 1][col_idx])
                    cv2.line(image, pt1, pt2, (0, 0, 255), 2)

def main():
    parser = argparse.ArgumentParser(description='Chess board detection using corner detection')
    parser.add_argument('image_path', help='Path to chess board image')
    parser.add_argument('-o', '--output', help='Output path for result image')
    parser.add_argument('--method', choices=['harris', 'shi_tomasi'], default='shi_tomasi',
                       help='Corner detection method')
    parser.add_argument('--show', action='store_true', help='Display result image')
    
    args = parser.parse_args()
    
    try:
        original, result, corners, grid = detect_chess_board_corners(
            args.image_path, args.output, args.method
        )
        
        if args.show:
            # Display original and result side by side
            combined = np.hstack((original, result))
            cv2.imshow('Original vs Corner Detection', combined)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()




