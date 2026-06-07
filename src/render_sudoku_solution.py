from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# The paths of the fonts
FONT_PATHS = [
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Times.ttc",
]


FONT_TO_LABEL = {path : idx for idx, path in enumerate(FONT_PATHS)}

LABEL_TO_FONT = {idx : path for idx, path in enumerate(FONT_PATHS)}

def render_solution_overlay(
    solved_board: list[list[int]],
    empty_positions: list[tuple[int, int]],
    font_label: int,
    board_size: int = 450,
    FONT_SCALE: float = 1.2,
    colour: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    Renders Sudoku solution overlay using PIL fonts.

    Args:
        solved_board: The solved Sudoku board
        empty_positions: A list of coordinates that represent empty cells
        board_size: The size of the board (default is 450)
        FONT_SCALE: The scale of the font (default is 1.2)
        colour: The RGB colour used to draw the digits (default is red)

    Returns:
        np.ndarray: The overlay image
    """

    # Get the path of the font
    font_path = LABEL_TO_FONT[font_label]

    # Create blank image
    img = Image.new("RGB", (board_size, board_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    cell_size = board_size // 9

    # Load the font
    font_size = int(cell_size * 0.6 * FONT_SCALE)
    font = ImageFont.truetype(font_path, font_size)

    for r, c in empty_positions:

        digit = str(solved_board[r][c])

        x = c * cell_size
        y = r * cell_size

        # Get text bounding box (replacement for cv2.getTextSize)
        bbox = draw.textbbox((0, 0), digit, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]

        # Center text inside cell
        text_x = x + (cell_size - text_w) // 2
        text_y = y + (cell_size - text_h) // 2

        draw.text((text_x, text_y), digit, font=font, fill=colour)

    # Convert back to OpenCV format (BGR)
    return np.array(img)

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

    # Create a mask that for the foreground image from the channel maximum
    channel_max = np.max(warped_overlay, axis=2)    # takes the max. of its 3-channel value
    _, mask = cv2.threshold(channel_max, 10, 255, cv2.THRESH_BINARY)

    # Keep the cells that contain the solution
    foreground = cv2.bitwise_and(warped_overlay, warped_overlay, mask=mask)

    # Create a mask that filters out the empty cells
    mask_inv = cv2.bitwise_not(mask)

    # Remove empty cells for the solution from the original image with the mask
    background = cv2.bitwise_and(img, img, mask=mask_inv)

    # Combine the background and the foreground
    result = cv2.add(background, foreground)

    return result
