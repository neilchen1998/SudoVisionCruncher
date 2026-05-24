import cv2
import numpy as np

def draw_solution_overlay(
    solved_board,
    empty_positions,
    board_size=450
):

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
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return overlay

def project_overlay_back(
    original_img,
    overlay,
    M
):
    """
    Warp the solved overlay back onto the original image.
    """

    h, w = original_img.shape[:2]

    # Inverse perspective transform
    Minv = np.linalg.inv(M)

    warped_overlay = cv2.warpPerspective(
        overlay,
        Minv,
        (w, h)
    )

    # Create mask
    gray = cv2.cvtColor(warped_overlay, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    mask_inv = cv2.bitwise_not(mask)

    # Remove area from original
    background = cv2.bitwise_and(
        original_img,
        original_img,
        mask=mask_inv
    )

    # Keep overlay
    foreground = cv2.bitwise_and(
        warped_overlay,
        warped_overlay,
        mask=mask
    )

    # Combine
    result = cv2.add(background, foreground)

    return result
