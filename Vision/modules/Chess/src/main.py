import cv_model
import localization
import depthai as dai
import time


def print_chess_pieces(chess_pieces):
    """Print detected chess pieces in a nice format."""
    print("\n" + "="*80)
    print("DETECTED CHESS PIECES")
    print("="*80)
    if not chess_pieces:
        print("  No pieces detected.")
    else:
        print(f"  Found {len(chess_pieces)} piece(s):\n")
        for i, (class_name, (center_x, center_y)) in enumerate(chess_pieces, 1):
            print(f"  {i}. {class_name:25s} at center ({int(center_x):4d}, {int(center_y):4d})")
    print("="*80 + "\n")


def print_squares(squares):
    """Print chess board squares in a visual 8x8 grid."""
    print("\n" + "="*80)
    print("CHESS BOARD SQUARES (Coordinates)")
    print("="*80)
    
    # Sort squares by row (8 to 1) and column (A to H)
    sorted_squares = sorted(squares.items(), 
                           key=lambda x: (-int(x[0][1]), x[0][0]))
    
    # Print header
    print("\n      ", end="")
    for col in 'ABCDEFGH':
        print(f"    {col}     ", end="")
    print()
    print("   " + "-" * 76)
    
    # Print rows 8 to 1
    current_row = None
    for square_name, (x, y) in sorted_squares:
        row = square_name[1]
        if current_row != row:
            if current_row is not None:
                print()
            print(f" {row} |", end="")
            current_row = row
        
        print(f" ({int(x):4d},{int(y):4d}) |", end="")
    
    print("\n   " + "-" * 76)
    print("="*80 + "\n")


def print_full_board(full_board):
    """Print the full chess board with pieces in a visual 8x8 grid."""
    print("\n" + "="*80)
    print("CHESS BOARD STATE")
    print("="*80)
    
    # Sort squares by row (8 to 1) and column (A to H)
    sorted_board = sorted(full_board.items(), 
                         key=lambda x: (-int(x[0][1]), x[0][0]))
    
    # Print header
    print("\n      ", end="")
    for col in 'ABCDEFGH':
        print(f"    {col}     ", end="")
    print("\n   " + "-" * 76)
    
    # Print rows 8 to 1
    current_row = None
    for square_name, piece_name in sorted_board:
        row = square_name[1]
        if current_row != row:
            if current_row is not None:
                print()
            print(f" {row} |", end="")
            current_row = row
        
        # Display piece or empty square
        if piece_name:
            # Truncate long names to fit
            display = piece_name[:8] if len(piece_name) <= 8 else piece_name[:5] + "..."
            print(f" {display:8s} |", end="")
        else:
            print(f" {'·':8s} |", end="")
    
    print("\n   " + "-" * 76)
    print("="*80 + "\n")


def main():

    MODEL_PATH = (
    "/Users/azieldawit/Desktop/School/WPI/MQP/Sensing/Vision/modules/Chess/models/snake_version_1/snake_yolov8n_1.rvc2.tar.xz"
    )
    
    YOLOV8_MODEL_PATH = (
    "/home/chess/Desktop/Sensing/Vision/modules/Chess/models/yolov11m_snake_final.pt"
    )
    
    # First device for calibration
    device1 = dai.Device()
    print("Created device1 for calibration")
    
    localizer = localization.Localize(device1)
    squares = localizer.calibrate(distortion=True)
    
    # Close first device
    device1.close()
    print("Closed device1")
    time.sleep(0.5)  # Give device time to fully release
    
    
    device3 = dai.Device()
    print("Created device3 for prediction")
    model = cv_model.Model(MODEL_PATH, device3)
    # model.liveInference()
    chess_peices = model.predict2(YOLOV8_MODEL_PATH, distortion=True)

    device3.close()
    print("Closed device3")
    time.sleep(0.5)  # Give device time to fully release

    
    # Localize doesn't need device - it's just computation
    full_board = localizer.localize(chess_peices, squares)

    # Print everything in a nice format
    print_chess_pieces(chess_peices)
    print_squares(squares)  
    print_full_board(full_board)

    print("full board: ", full_board)

if __name__ == "__main__":
    main()