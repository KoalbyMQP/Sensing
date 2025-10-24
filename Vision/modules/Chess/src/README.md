# Chess Board Localization Methods

This directory contains three different approaches for detecting and localizing chess board grids from overhead images.

## Methods Implemented

### 1. Hough Line Transform (`hough_lines_detection.py`)
- **Approach**: Detects straight lines using Hough transform
- **Pros**: Classic method, works well with clear board edges
- **Cons**: Sensitive to lighting and partial occlusions
- **Best for**: Clear, well-lit boards with distinct edges

### 2. Corner Detection (`corner_detection.py`)
- **Approach**: Uses Harris or Shi-Tomasi corner detection
- **Pros**: More robust to perspective distortion
- **Cons**: Requires sufficient corner points for grid fitting
- **Best for**: Boards with clear corner intersections

### 3. OpenCV Chess Board Detection (`opencv_chessboard.py`)
- **Approach**: Uses OpenCV's specialized `findChessboardCorners()`
- **Pros**: Specifically designed for chess boards, most reliable
- **Cons**: Requires clear board pattern
- **Best for**: Standard chess boards with good contrast

## How Raspberry Turk Does It

Based on the [Raspberry Turk project](https://www.raspberryturk.com/details/vision.html), they use:

1. **Perspective Transform**: Raw images are warped to 480x480 pixels
2. **Square Normalization**: Ensures all squares are equal size
3. **Cropping**: Removes non-board areas
4. **Piece Detection**: Uses SVM for piece/no-piece classification
5. **CNN**: For specific piece type identification (pawn promotion cases)

Their approach is essentially a **perspective-corrected grid detection** method.

## Usage

### Test Single Image
```bash
# Test all methods on one image
python3 quick_test.py ../data/raw_data/chess_data_2025_10_03\(overhead\)/image_001.jpg

# Test specific method
python3 src/opencv_chessboard.py ../data/raw_data/chess_data_2025_10_03\(overhead\)/image_001.jpg --show
```

### Test Multiple Images
```bash
# Test all images in a directory
python3 src/test_localization_methods.py ../data/raw_data/chess_data_2025_10_03\(overhead\)/
```

### Individual Method Testing
```bash
# Hough Lines
python3 src/hough_lines_detection.py image.jpg --show

# Corner Detection
python3 src/corner_detection.py image.jpg --method shi_tomasi --show

# OpenCV Chess Board
python3 src/opencv_chessboard.py image.jpg --show --create-mapping
```

## Expected Results

The scripts will:
1. **Detect grid lines** using each method
2. **Overlay colored lines** on the original image:
   - Green: Detected grid lines
   - Blue: Horizontal lines (Hough method)
   - Red: Vertical lines (Hough method)
3. **Save result images** to `test_outputs/` directory
4. **Print statistics** about detection success

## Next Steps

After testing these methods:

1. **Choose the best method** based on success rate
2. **Integrate with YOLOv8** piece detection
3. **Create coordinate mapping** from bounding boxes to chess squares
4. **Implement perspective correction** (like Raspberry Turk)
5. **Add real-time processing** for live chess games

## File Structure

```
src/
├── hough_lines_detection.py    # Hough Line Transform method
├── corner_detection.py         # Corner Detection method  
├── opencv_chessboard.py        # OpenCV Chess Board method
└── test_localization_methods.py # Test all methods
quick_test.py                   # Quick test script
```

## Dependencies

- OpenCV (`cv2`)
- NumPy (`numpy`)
- Pathlib (built-in)

Install with:
```bash
pip install opencv-python numpy
```