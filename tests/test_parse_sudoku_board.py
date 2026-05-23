import cv2
import numpy as np
import pytest
import random

from src.parse_sudoku_board import is_valid_sudoku, flatten_board

# This is a valid and completed Sudoku board
VALID_COMPLETED = [
    [5, 3, 4, 6, 7, 8, 9, 1, 2],
    [6, 7, 2, 1, 9, 5, 3, 4, 8],
    [1, 9, 8, 3, 4, 2, 5, 6, 7],
    [8, 5, 9, 7, 6, 1, 4, 2, 3],
    [4, 2, 6, 8, 5, 3, 7, 9, 1],
    [7, 1, 3, 9, 2, 4, 8, 5, 6],
    [9, 6, 1, 5, 3, 7, 2, 8, 4],
    [2, 8, 7, 4, 1, 9, 6, 3, 5],
    [3, 4, 5, 2, 8, 6, 1, 7, 9],
]

# This is a valid Sudoku in which all cells are 0
VALID_EMPTY = [[0] * 9 for _ in range(9)]

# This is a valid Sudoku
VALID_PARTIAL = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

# This is an invalid Sudoku
# There are two 5's in the first row
INVALID_ROW = [
    [5, 5, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

# This is an invalid Sudoku
# There are two 6's in the left-most column
INVALID_COL = [
    [6, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

# This is an invalid Sudoku
# There are two 1's in the top-left grid
INVALID_SUBGRID = [
    [5, 3, 1, 0, 7, 0, 0, 0, 0],
    [6, 1, 0, 0, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9],
]

@pytest.mark.parametrize(
    "board",
    [
        VALID_COMPLETED,
        VALID_EMPTY,
        VALID_PARTIAL,
    ],
    ids=[
        "valid_completed",
        "valid_empty",
        "valid_partial",
    ]
)
def test_valid_boards(board):
    """
    Verifies that all structurally correct Sudoku boards return True
    """

    assert is_valid_sudoku(board) is True

@pytest.mark.parametrize(
    "board",
    [
        INVALID_ROW,
        INVALID_COL,
        INVALID_SUBGRID
    ],
    ids=[
        "duplicate_in_row",
        "duplicate_in_col",
        "duplicate_in_subgrid"
    ]
)
def test_invalid_boards(board):
    """
    Verifies that all structurally incorrect Sudoku boards return False
    """

    assert is_valid_sudoku(board) is False

@pytest.fixture
def circle_image():
    """
    Generates a circle that doesn't have 4 vertices
    """

    # Create a blank black image
    img = np.zeros((600, 600, 3), dtype=np.uint8)

    # Draw a white circle
    cv2.circle(img, (300, 300), 150, (255, 255, 255), -1)

    return img

@pytest.fixture
def triangle_image():
    """
    Generates a triangle that doesn't have 4 vertices
    """

    # Create a blank black image
    img = np.zeros((600, 600, 3), dtype=np.uint8)

    # Define the three vertices
    pts = np.array([[300, 150], [150, 450], [450, 450]], dtype=np.int32)
    pts = pts.reshape((-1, 1, 2))

    # Draw the three lines on the canvas
    cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 255), thickness=3)

    return img


def test_flatten_board_success(synthetic_sudoku_image):
    """
    Tests if the function successfully flattens a valid board to the default size
    """

    N = 450

    # Run the function
    result = flatten_board(synthetic_sudoku_image)

    # Assert the type of the result
    assert isinstance(result, np.ndarray), "Output should be a numpy array"

    # Assert the shape of the result
    assert result.shape == (
        N,
        N,
    ), f"Output shape should be ({N}, {N})"

    # Assert the shape (it should be a single-channel image)
    assert (
        len(result.shape) == 2
    ), "Output should be single-channel greyscale (2D array)"

    # Check that it extracted the white board area (it shouldn't be completely black)
    assert np.max(result) > 0, "The result should contain the warped board data"

@pytest.mark.parametrize("test_n", [50, 100, 150, 200, 300, 400, 500])
def test_flatten_board_custom_n(synthetic_sudoku_image, test_n):
    """
    Tests if the function respects a custom N dimension parameter
    """

    result = flatten_board(synthetic_sudoku_image, N=test_n)

    assert result.shape == (test_n, test_n)


@pytest.fixture
def synthetic_sudoku_image(request) -> np.ndarray:
    """
    Generates an image from parameters passed via indirect parameterization
    """

    # Get the parameters from the request
    # The default values will be used when no request is provided
    params = getattr(request, "param", {"top_left": (100, 100), "width": 400})

    top_left = params["top_left"]
    width = params["width"]

    x, y = top_left
    bottom_right = (x + width, y + width)

    # Create a canvas and draw a rectangle with given parameters on it
    img = np.zeros((600, 600, 3), dtype=np.uint8)
    cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), -1)

    return img

@pytest.fixture
def synthetic_skewed_sudoku_image(request) -> np.ndarray:
    """
    Generates a skewed white quadrilateral on a black background.
    """

    params = getattr(
        request,
        "param",
        {
            "top_left": (100, 100),
            "width": 300,
            "skew": 30,
        },
    )

    top_left = params["top_left"]
    width = params["width"]
    skew = params["skew"]

    x, y = top_left

    canvas_size = 1000
    img = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)

    # Define square corners
    corners = np.array(
        [
            [x, y],                  # top-left
            [x + width, y],          # top-right
            [x + width, y + width],  # bottom-right
            [x, y + width],          # bottom-left
        ],
        dtype=np.float32,
    )

    # Apply random skew
    skewed = []
    for px, py in corners:
        skewed.append(
            [
                int(px - skew),
                int(py + skew),
            ]
        )

    skewed = np.array(skewed, dtype=np.int32)

    # Draw filled polygon
    cv2.fillPoly(img, [skewed], (255, 255, 255))

    return img

def test_flatten_board_success(synthetic_sudoku_image):
    """
    Standard success test (uses default fixture parameters)
    """

    result = flatten_board(synthetic_sudoku_image)

    assert result is not None

@pytest.mark.parametrize(
    "synthetic_sudoku_image",
    [
        ({"top_left": (50, 80), "width": 400}),
        ({"top_left": (100, 100), "width": 300}),
        ({"top_left": (150, 150), "width": 300}),
        ({"top_left": (440, 450), "width": 300}),
    ],
    indirect=["synthetic_sudoku_image"],  # Tells Python we want to call the fixture
)
def test_flatten_board_custom_values(synthetic_sudoku_image):
    """
    Tests custom bounding boxes
    """

    result = flatten_board(synthetic_sudoku_image)

    assert result.shape == (450, 450)

@pytest.mark.parametrize(
    "synthetic_skewed_sudoku_image",
    [
        ({"top_left": (50, 80), "width": 400, "skew": 30}),
        ({"top_left": (100, 100), "width": 300, "skew": 40}),
        ({"top_left": (150, 150), "width": 300, "skew": 50}),
        ({"top_left": (440, 450), "width": 300, "skew": 90}),
    ],
    indirect=["synthetic_skewed_sudoku_image"],  # Tells Python we want to call the fixture
)
def test_flatten_skewed_board_custom_values(synthetic_skewed_sudoku_image):
    """
    Tests custom skewed bounding boxes
    """

    result = flatten_board(synthetic_skewed_sudoku_image)

    assert result.shape == (450, 450)

@pytest.mark.parametrize(
    "image_fixture_name, expected_error, expected_error_msg",
    [
        ("circle_image", ValueError, "Could not find a 4-sided Sudoku board."),
        ("triangle_image", ValueError, "Could not find a 4-sided Sudoku board."),
    ]
)
def test_flatten_board_invalid_board_raises_error(request, image_fixture_name, expected_error, expected_error_msg):
    """
    Tests if a ValueError is raised when a 4-sided board cannot be found
    """

    # Get the image from the fixture that we define
    img = request.getfixturevalue(image_fixture_name)

    # Assert if the function raises the expected error
    with pytest.raises(expected_error) as exc_info:
        flatten_board(img)

    # Verify the expected exception message
    assert str(exc_info.value) == expected_error_msg
