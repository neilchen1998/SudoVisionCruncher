from collections import Counter
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sudovisioncruncher.sudoku.utils import is_valid_sudoku

def flatten_board(img: np.ndarray, N: int = 450) -> tuple[np.ndarray, np.ndarray]:

    """
    Flattens a Sudoku board

    Args:
        img: The input image that contains a Sudoku board
        N (optional): the size of the board (default to 450 pixels)

    Returns:
        tuple[np.ndarray, np.ndarray]: The flatten board and the perspective transformation matrix
    """

    # Check if the input image is non-empty and an RGB image
    if img is None or img.size == 0 or img.ndim != 3:
        raise ValueError("Input image must be a non-emtpy 3-channel RGB image.")

    # Check if the input dimension is bigger than N
    if img.shape[0] < N or img.shape[1] < N:
        raise ValueError(f"Input image is smaller than {N}x{N} ({img.shape[0]}x{img.shape[1]}).")

    # Turn into grey scale
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur effect to remove high frequency noises
    blurred = cv2.GaussianBlur(grey, (5, 5), 0)

    # Convert the blurry greyscale image into sharp black-and-white only image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=11, C=2)

    # Find all the contours of the Sudoku board
    # NOTE: the contours are unsorted
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort all of the contours in decending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Extract the outer contour (which is the largest)
    board_contour = contours[0]

    # Calculate the perimeter
    peri = cv2.arcLength(board_contour, True)

    # Apply Douglas-Peucker algorithm to approximate the contour with fewer vertices
    approx = cv2.approxPolyDP(board_contour, 0.01 * peri, True)

    # Make sure that there are 4 vertices
    if len(approx) != 4:
        raise ValueError("Could not find a 4-sided Sudoku board.")

    def sort_vertices(pts: np.ndarray) -> np.ndarray:
        """
        Sorts four vertices into top-left, top-right, bottom-right, bottom-left order

        Args:
            pts: A numpy array of shape (4, 2) containing four (x, y) vertex points

        Returns:
            np.ndarray: A numpy array of shape (4, 2) with points ordered clockwise
            starting from the top-left vertex
        """

        ret = np.empty((4, 2), dtype="float32")

        # Calculate the sums row-wise
        s = pts.sum(axis=1)

        # The top-left vertex has the smallest value
        # The bottom-right vertex has the largest value
        ret[0] = pts[np.argmin(s)]
        ret[2] = pts[np.argmax(s)]

        # Calculate the differences row-wise
        diff = np.diff(pts, axis=1)

        # The top-right vertex has the smallest value
        # The bottom-left vertex has the largest value
        ret[1] = pts[np.argmin(diff)]
        ret[3] = pts[np.argmax(diff)]

        return ret

    # Get all the vertices
    vertices = np.float32([v[0] for v in approx])
    sorted_vertices = sort_vertices(vertices)

    dst_vertices = np.array([
        [0, 0],         # Top-Left destination
        [N - 1, 0],     # Top-Right destination
        [N - 1, N - 1], # Bottom-Right destination
        [0, N - 1]      # Bottom-Left destination
    ], dtype="float32")

    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(sorted_vertices, dst_vertices)

    # Wrap the original image to get the final flat square
    flat_board = cv2.warpPerspective(grey, M, (N, N))

    return flat_board, M

def preprocess_digit(digit: np.ndarray, digit_target_size: int, canvas_size: int) -> np.ndarray:
    """
    Resizes a digit while preserving aspect ratio,
    centers it on a square canvas,
    and normalizs pixel values to [0, 1].

    Args:
        digit: Grayscale digit image
        digit_target_size: Size of the longest side after resizing
        canvas_size: Size of the square model input canvas

    Returns:
        np.ndarray: Normalized image
    """
    h_digit, w_digit = digit.shape

    # Resize the digit to DIGIT_TARGET_SIZE but also preserve the aspect
    scale = digit_target_size / max(h_digit, w_digit)
    new_w = max(1, int(w_digit * scale))
    new_h = max(1, int(h_digit * scale))

    resized_digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    x_offset = (canvas_size - new_w) // 2
    y_offset = (canvas_size - new_h) // 2

    # Put the digit in the center of a canvas
    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized_digit

    # Normalize to [0, 1]
    return canvas.astype(np.float32) / 255.0

def parse_sudoku_board(board: np.ndarray, ocr_model: tf.keras.Model, font_model: tf.keras.Model) -> tuple[list[list[int]], list[tuple[int, int, int]]]:
    """
    Parse a 9x9 grid image into a 2D list of predicted digits using OCR

    Args:
        board: A numpy array representing the full board image
        ocr_model: The OCR Keras model
        font_model: The font detection Keras model

    Returns:
        tuple:
            list[list[int]]: A 9x9 list of integers representing the detected digits
            list[tuple[int, int]]: A list of coordinates of empty positions
    """

    # Parameters
    MARGIN_RATIO = 0.2
    OCR_MODEL_INPUT_SIZE = 28
    FONT_MODEL_INPUT_SIZE = 64
    DIGIT_TARGET_SIZE = 18
    THRESH_BLOCK_SIZE = 11
    THRESH_C = 2

    # Find the size of the cell
    cell_size = board.shape[0] // 9

    # For batch prediction (OCR)
    batch_inputs_ocr = []

    # For batch prediction (font)
    batch_inputs_font = []

    # For reconstruction
    positions = []

    # Create a grid with all 0
    grid = [[0 for _ in range(9)] for _ in range(9)]

    # Empty positions
    empty_positions = []

    for r in range(9):
        for c in range(9):

            # Crop the cell
            x1, y1 = c * cell_size, r * cell_size
            x2, y2 = x1 + cell_size, y1 + cell_size
            cell = board[y1:y2, x1:x2]

            # Remove the outer margin to avoid boarders
            margin = int(cell_size * MARGIN_RATIO)
            cell = cell[
                margin:cell_size-margin,
                margin:cell_size-margin
            ]

            # Blur the image
            blur = cv2.GaussianBlur(cell, (3, 3), 0)

            # Use adaptive threshold
            # NOTE: block size should be an odd number
            thresh = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                blockSize=THRESH_BLOCK_SIZE,
                C=THRESH_C
            )

            # Remove noises by using a 2x2 kernel
            kernel = np.ones((2, 2), np.uint8)
            thresh = cv2.morphologyEx(
                thresh,
                cv2.MORPH_OPEN,
                kernel
            )

            # Find the contour of the digit
            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            # If there is no contour, then we deduce the current cell is blank
            if len(contours) == 0:
                empty_positions.append((r, c))
                continue

            # Find the largest area
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            # If the largest area is less than a threshold, then it must be a noise or an artifact
            cell_area = cell.shape[0] * cell.shape[1]
            min_area_threshold = cell_area * 0.01
            if area < min_area_threshold:
                empty_positions.append((r, c))
                continue

            # Find the bounding box that bounds the digit
            x, y, w, h = cv2.boundingRect(largest)

            # If the bounding box is small that means there is no digit to crop
            if w == 0 or h == 0:
                empty_positions.append((r, c))
                continue

            # Crop the digit
            digit = thresh[y:y+h, x:x+w]

            # Put the digit in the center of a 28x28 canvas
            cell_input = preprocess_digit(
                digit,
                DIGIT_TARGET_SIZE,
                OCR_MODEL_INPUT_SIZE
            )

            # Append the cell to the OCR list
            batch_inputs_ocr.append(cell_input)
            positions.append((r, c))

            # Font
            cell_input_font = preprocess_digit(digit, int(FONT_MODEL_INPUT_SIZE * 0.8), FONT_MODEL_INPUT_SIZE)

            # Append the cell to the font list
            batch_inputs_font.append(cell_input_font)

    # Batch prediction (OCR)
    if batch_inputs_ocr:

        # Reshape inputs into a numpy array
        batch_array_digit = np.array(batch_inputs_ocr).reshape(
            -1, # the original shape
            OCR_MODEL_INPUT_SIZE,
            OCR_MODEL_INPUT_SIZE,
            1
        )

        # Predict the digits in a single batch
        predictions = ocr_model.predict(batch_array_digit, verbose=0)
        predicted_digits = np.argmax(predictions, axis=1)

        # Put the prediction results back into grid
        for (r, c), digit in zip(positions, predicted_digits):
            grid[r][c] = int(digit)

    # Batch prediction (font)
    if batch_inputs_font:

        # Reshape inputs into a numpy array
        batch_array_font = np.array(batch_inputs_font).reshape(
            -1, # the original shape
            FONT_MODEL_INPUT_SIZE,
            FONT_MODEL_INPUT_SIZE,
            1
        )

        # Predict the digits in a single batch
        predictions = font_model.predict(batch_array_font, verbose=0)
        predicted_fonts = np.argmax(predictions, axis=1)

        # Make the final font prediction based on the most popular vote
        most_common_font = Counter(predicted_fonts).most_common(1)[0][0]

    # Validation
    if not is_valid_sudoku(grid):
        raise ValueError(f"Invalid Sudoku grid: {grid}.")

    return grid, empty_positions, most_common_font
