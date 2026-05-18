import cv2
import numpy as np

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
