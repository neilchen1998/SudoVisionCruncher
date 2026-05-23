import cv2
import numpy as np

from src.digit_predict import load_model

def flatten_board(img, N=450) -> np.ndarray:

    """
    Flattens a Sudoku board

    Args:
        img: The input image that contains a Sudoku board
        N (optional): the size of the board (default to 450 pixels)

    Returns:
        np.ndarray: The flatten board
    """

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
        diff = np.diff(pts, axis=1).ravel()

        # The top-right vertex has the smallest value
        # The bottom-left vertex has the largest value
        ret[1] = pts[np.argmin(diff)]
        ret[3] = pts[np.argmax(diff)]

        return ret

    # Get all the vertices
    vertices = np.float32([vertices[0] for vertices in approx])
    sorted_vertices = sort_vertices(vertices)

    dst_vertices = np.array([
        [0, 0],                                         # Top-Left destination
        [N - 1, 0],                     # Top-Right destination
        [N - 1, N - 1], # Bottom-Right destination
        [0, N - 1]                      # Bottom-Left destination
    ], dtype="float32")

    # Calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(sorted_vertices, dst_vertices)

    # Wrap the original image to get the final flat square
    flat_board = cv2.warpPerspective(grey, M, (N, N))

    return flat_board

def is_valid_sudoku(board: list[list[int]]) -> bool:
    """
    Checks if a given Sudoku board is valid

    Args:
        board: A 9x9 list of integers representing the input Sudoku board

    Returns:
        bool: True if the Sudoku board is valid, False otherwise
    """

    def is_valid_sequence(seq: list[int]):

        """
        Checks if a given Sudoku sequence (row or column) is valid

        Args:
            seq: A list of integers representing the input Sudoku sequence

        Returns:
            bool: True if the Sudoku seqeunce board is valid, False otherwise
        """

        # Create a set
        seen = set()

        # Loop through all numbers in the list
        for num in seq:
            if num == 0:
                continue
            elif num in seen:
                return False
            else:
                seen.add(num)
        return True

    # Check rows
    for row in board:
        if not is_valid_sequence(row):
            return False

    # Check columns
    for col in range(9):
        sequence = [board[row][col] for row in range(9)]
        if not is_valid_sequence(sequence):
            return False

    # Check sub-grids
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):

            # Dissect the boad into a small grid and store all the numbers in the grid into a list
            sequence = [board[x][y] for x in range(i, i + 3) for y in range(j, j + 3)]

            if not is_valid_sequence(sequence):
                return False

    return True

def parse_sudoku_board(board: np.ndarray, model_path: str) -> list[list[int]] | None:

    """
    Parse a 9x9 grid image into a 2D list of predicted digits using OCR

    Args:
        board: A numpy array representing the full board image
        model_path: Path to the pre-trained OCR model

    Returns:
        A 9x9 list of integers representing the detected digits
    """

    # Parameters
    MARGIN_RATIO = 0.1
    MIN_DIGIT_AREA = 80
    MODEL_INPUT_SIZE = 28
    DIGIT_TARGET_SIZE = 18
    THRESH_BLOCK_SIZE = THRESH_BLOCK_SIZE
    THRESH_C = 2

    grid = []

    # Find the size of the cell
    cell_size = board.shape[0] // 9

    # Load the model for OCR
    model = load_model(model_path)

    for r in range(9):
        row = []
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

            # Use adaptive threshold
            # NOTE: block size should be an odd number
            thresh = cv2.adaptiveThreshold(
                cell,
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
                row.append(0)
                continue

            # Find the largest area
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            # If the largest area is less than a threshold, then it must be a noise or an artifact
            if area < MIN_DIGIT_AREA:
                row.append(0)
                continue

            # Crop the digit
            x, y, w, h = cv2.boundingRect(largest)
            digit = thresh[y:y+h, x:x+w]

            # Resize the digit to DIGIT_TARGET_SIZE but also preserve the aspect
            h_digit, w_digit = digit.shape
            scale = DIGIT_TARGET_SIZE / max(h_digit, w_digit)
            new_w = int(w_digit * scale)
            new_h = int(h_digit * scale)
            resized_digit = cv2.resize(digit, (new_w, new_h))

            # Put the digit in the center of a 28x28 canvas
            canvas = np.zeros((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), dtype=np.uint8)
            x_offset = (MODEL_INPUT_SIZE - new_w) // 2
            y_offset = (MODEL_INPUT_SIZE - new_h) // 2
            canvas[
                y_offset:y_offset+new_h,
                x_offset:x_offset+new_w
            ] = resized_digit

            # Normalize the input to the model
            cell_input = (
                canvas
                .reshape(1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 1)
                .astype("float32") / 255.0
            )

            # Predict the digit
            predictions = model.predict(cell_input, verbose=0)
            predicted_digit = np.argmax(predictions, axis=-1)[0]

            row.append(int(predicted_digit))

        grid.append(row)

    # Check the number of cells is correct
    for row in grid:
        if len(row) != 9:
            raise ValueError("The number of cells in #{} is incorrect")

    if len(grid) != 9:
        raise ValueError("The number of rows is incorrect")

    if not is_valid_sudoku(grid):
        raise ValueError("Invalid Sudoku grid")

    return grid
