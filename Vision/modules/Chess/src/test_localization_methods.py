#!/usr/bin/env python3
"""
Test script to compare all three chess board localization methods
Runs Hough Lines, Corner Detection, and OpenCV Chess Board detection on overhead images
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import sys
import os

# Add the src directory to the path so we can import our detection modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hough_lines_detection import detect_chess_board_hough
from corner_detection import detect_chess_board_corners
from opencv_chessboard import detect_chess_board_opencv

def test_all_methods(image_path, output_dir=None):
    """
    Test all three chess board detection methods on a single image
    
    Args:
        image_path: Path to the chess board image
        output_dir: Directory to save output images (optional)
    
    Returns:
        Dictionary with results from all methods
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    print(f"Testing chess board detection methods on: {image_path.name}")
    print("=" * 60)
    
    # Method 1: Hough Lines
    print("\n1. Testing Hough Line Transform...")
    try:
        orig1, result1, lines1 = detect_chess_board_hough(
            image_path, 
            output_dir / f"{image_path.stem}_hough.jpg" if output_dir else None
        )
        results['hough'] = {
            'success': True,
            'lines': lines1,
            'horizontal_count': len(lines1['horizontal']),
            'vertical_count': len(lines1['vertical'])
        }
        print(f"   ✓ Hough Lines: {len(lines1['horizontal'])} horizontal, {len(lines1['vertical'])} vertical")
    except Exception as e:
        print(f"   ✗ Hough Lines failed: {e}")
        results['hough'] = {'success': False, 'error': str(e)}
    
    # Method 2: Corner Detection (Shi-Tomasi)
    print("\n2. Testing Corner Detection (Shi-Tomasi)...")
    try:
        orig2, result2, corners2, grid2 = detect_chess_board_corners(
            image_path,
            output_dir / f"{image_path.stem}_corners.jpg" if output_dir else None,
            'shi_tomasi'
        )
        results['corners'] = {
            'success': True,
            'corner_count': len(corners2) if corners2 is not None else 0,
            'grid_points': len(grid2) if grid2 is not None else 0
        }
        print(f"   ✓ Corners: {len(corners2) if corners2 is not None else 0} corners detected")
    except Exception as e:
        print(f"   ✗ Corner Detection failed: {e}")
        results['corners'] = {'success': False, 'error': str(e)}
    
    # Method 3: OpenCV Chess Board Detection
    print("\n3. Testing OpenCV Chess Board Detection...")
    try:
        orig3, result3, corners3, success3 = detect_chess_board_opencv(
            image_path,
            output_dir / f"{image_path.stem}_opencv.jpg" if output_dir else None
        )
        results['opencv'] = {
            'success': success3,
            'corner_count': len(corners3) if corners3 is not None else 0
        }
        print(f"   {'✓' if success3 else '✗'} OpenCV: {'Success' if success3 else 'Failed'}")
    except Exception as e:
        print(f"   ✗ OpenCV Chess Board Detection failed: {e}")
        results['opencv'] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Hough Lines:     {'✓' if results['hough']['success'] else '✗'}")
    print(f"Corner Detection: {'✓' if results['corners']['success'] else '✗'}")
    print(f"OpenCV Method:   {'✓' if results['opencv']['success'] else '✗'}")
    
    return results

def test_multiple_images(image_dir, output_dir=None):
    """
    Test all methods on multiple images in a directory
    
    Args:
        image_dir: Directory containing chess board images
        output_dir: Directory to save output images
    """
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Directory not found: {image_dir}")
    
    # Find all image files
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in image_dir.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images to test")
    print("=" * 60)
    
    all_results = {}
    
    for i, image_file in enumerate(image_files, 1):
        print(f"\n[{i}/{len(image_files)}] Testing: {image_file.name}")
        
        try:
            results = test_all_methods(image_file, output_dir)
            all_results[image_file.name] = results
        except Exception as e:
            print(f"Error testing {image_file.name}: {e}")
            all_results[image_file.name] = {'error': str(e)}
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("OVERALL STATISTICS:")
    
    successful_images = {name: results for name, results in all_results.items() 
                       if 'error' not in results}
    
    if successful_images:
        hough_success = sum(1 for r in successful_images.values() if r['hough']['success'])
        corner_success = sum(1 for r in successful_images.values() if r['corners']['success'])
        opencv_success = sum(1 for r in successful_images.values() if r['opencv']['success'])
        
        total_images = len(successful_images)
        
        print(f"Total images tested: {total_images}")
        print(f"Hough Lines success rate: {hough_success}/{total_images} ({100*hough_success/total_images:.1f}%)")
        print(f"Corner Detection success rate: {corner_success}/{total_images} ({100*corner_success/total_images:.1f}%)")
        print(f"OpenCV method success rate: {opencv_success}/{total_images} ({100*opencv_success/total_images:.1f}%)")
    
    return all_results

def main():
    parser = argparse.ArgumentParser(description='Test all chess board detection methods')
    parser.add_argument('input', help='Path to chess board image or directory')
    parser.add_argument('-o', '--output', help='Output directory for result images')
    parser.add_argument('--single', action='store_true', 
                       help='Test single image (default: test all images in directory)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    try:
        if args.single or input_path.is_file():
            # Test single image
            test_all_methods(input_path, args.output)
        else:
            # Test all images in directory
            test_multiple_images(input_path, args.output)
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())


