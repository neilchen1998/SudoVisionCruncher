import cv2
import numpy as np

def draw_solution_overlay(solved_board, empty_positions: list[tuple[int, int]], board_size:int = 450) -> np.ndarray:
    """
    Draws the solution on the given image

    Args:
        solved_board: The solved Sudoku board
        empty_positions: A list of coordinates that represent empty cells
        board_size: The size of the board (default is 450)

    Returns:
        np.ndarray: The overlay image
    """

    overlay = np.zeros((board_size, board_size, 3), dtype=np.uint8)

    cell_size = board_size // 9

    for r, c in empty_positions:

        digit = str(solved_board[r][c])

        x = c * cell_size
        y = r * cell_size

        text_x = x + cell_size // 4
        text_y = y + int(cell_size * 0.75)

        cv2.putText(
            overlay,
            digit,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            1.2,    # the font scale factor
            (255, 0, 0),    # red
            2,  # the thickness of the lines
            cv2.LINE_AA # anti-aliased line type.
        )

    return overlay

def project_overlay_back(img: np.ndarray, overlay: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Warp the solved overlay back onto the original image

    Args:
        img: The original Sudoku board image
        overlay: The overlay image that contains the solution
        M: The perspective transformation matrix

    Returns:
        np.ndarray: The Sudoku board image with the solution
    """

    # Get the shape of the image
    h, w = img.shape[:2]

    # Inverse the perspective transform matrix
    M_inv = np.linalg.inv(M)

    # Transform the overlay image so that it will fit into the background image (the original image)
    warped_overlay = cv2.warpPerspective(overlay, M_inv, (w, h))

    # Create a mask that will cut holes in the background image for the foreground image
    grey = cv2.cvtColor(warped_overlay, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # Remove cells for the solution from the original image
    background = cv2.bitwise_and(img, img, mask=mask_inv)

    # Keep overlay
    foreground = cv2.bitwise_and(warped_overlay, warped_overlay, mask=mask)

    # Combine the background and the foreground
    result = cv2.add(background, foreground)

    return result
