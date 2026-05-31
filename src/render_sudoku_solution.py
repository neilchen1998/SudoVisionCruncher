import cv2
import numpy as np

def render_solution_overlay(solved_board: list[list[int]], empty_positions: list[tuple[int, int]],
                            board_size:int = 450, FONT_FACE:int = cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                            FONT_SCALE:float = 1.2, THICKNESS:int = 2, colour: tuple[int, int, int] = (0, 255, 255)) -> np.ndarray:
    """
    Renders the solution on the given image

    Args:
        solved_board: The solved Sudoku board
        empty_positions: A list of coordinates that represent empty cells
        board_size: The size of the board (default is 450)
        FONT_SCALE: The scale of the font (default is 1.2)
        THICKNESS: The thickness of the font (default is 2.0)
        colour: The BGR colour used to draw the digits (default is red)

    Returns:
        np.ndarray: The overlay image
    """

    overlay = np.zeros((board_size, board_size, 3), dtype=np.uint8)

    cell_size = board_size // 9

    for r, c in empty_positions:

        digit = str(solved_board[r][c])

        x = c * cell_size
        y = r * cell_size

        # Find the width and the height of the text based on the font, scale, etc.
        (text_w, text_h), _ = cv2.getTextSize(
            digit,
            FONT_FACE,
            FONT_SCALE,
            THICKNESS,
        )

        # Find the text position by centering the text within the box using half the text width and height offsets
        text_x = x + (cell_size - text_w) // 2
        text_y = y + (cell_size + text_h) // 2

        cv2.putText(
            overlay,
            digit,
            (text_x, text_y),
            FONT_FACE,
            FONT_SCALE,
            colour,    # (B, G, R)
            THICKNESS,
            cv2.LINE_AA # anti-aliased line type.
        )

    return overlay

def overlay_solution_on_board(img: np.ndarray, overlay: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Overlays the solution back onto the original image

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

    # Create a mask that for the foreground image
    grey = cv2.cvtColor(warped_overlay, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)

    # Keep the cells that contain the solution
    foreground = cv2.bitwise_and(warped_overlay, warped_overlay, mask=mask)

    # Create a mask that filters out the empty cells
    mask_inv = cv2.bitwise_not(mask)

    # Remove empty cells for the solution from the original image with the mask
    background = cv2.bitwise_and(img, img, mask=mask_inv)

    # Combine the background and the foreground
    result = cv2.add(background, foreground)

    return result
