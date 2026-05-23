# SudoVisionCruncher

## Summary

## Requirements

* [uv](https://docs.astral.sh/uv/)
* Python 3.11.15
* [Tensorflow](https://www.tensorflow.org/) 2.18.0
* [Metal](https://developer.apple.com/metal/tensorflow-plugin/) 1.2.0 (for Mac Silicon)

## Instructions

Download Python with **uv**:

```zsh
uv venv --python 3.11.15
```

Activate the environment:

```zsh
source .venv/bin/activate
```

Install the packages:

```zsh
uv pip install tensorflow
```

Install the packages via **requirements.txt**:

```zsh
uv pip install -r requirements.txt
```

Deactivate the environment:

```zsh
deactivate
```

Run the main script:

```zsh
uv run ./main.py
```

Run the **pytest**:

```zsh
uv run pytest -o pythonpath=.
```

## Pipeline

Those are the steps in our pipeline that we need to do in order to prepare each cell for the model to make prediction:

1. Crop cell

We assume that there are 9 cells in each row on the board.
We got the size of the board in the previous step and therefore we can just divide the width of the board by 9 to get the size of the cell.

We calculate the position of the top-left and the bottom-right corner of the cell and extract the pixels from the board.
Note since *flat_board* is an OpenCV array and the first index is the x axis and the second index is the y axis, we need to flip the indices when retrieving a segment of the original image.

```python
# Crop the cell
x1, y1 = c * cell_size, r * cell_size
x2, y2 = x1 + cell_size, y1 + cell_size
cell = flat_board[y1:y2, x1:x2]
```

We then need to crop out the outer part of the image to avoid the boarderlines of the grid.
In this case, we assume that the margin is 10% of the size of the cell.

```python
# Remove the outer margin to avoid boarders
margin = int(cell_size * 0.1)
cell = cell[
    margin:cell_size-margin,
    margin:cell_size-margin
]
```

2. Apply adaptive threshold

We need to invert the colour of each cell, i.e., turning the background from white to black and turning the digit from black to white since the digits used for training in TMNIST are in this format.

**cv2.ADAPTIVE_THRESH_GAUSSIAN_C** is the method that determines how the threshold is calcualted.
The threshold value is the weighted sum of neighbourhood values.
This method metigate different lighting conditions in different areas.

**cv2.THRESH_BINARY_INV** turns pixels above the treshold black and pixels below the threshold white.

**blockSize** is the size of neighbourhood area and it needs to be an odd value, the value mentioned in the documentation is 11 and it gives decent results.

**C** is a constant which is subtracted from the mean or weighted mean calculated, the value mentioned in the documentation is 2 and it gives decent results.

```python
# Use adaptive threshold
thresh = cv2.adaptiveThreshold(
    cell,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    blockSize=11,
    C=2
)
```

3. Remove noises

We use [morphology opening] (https://homepages.inf.ed.ac.uk/rbf/HIPR2/open.htm) to remove the noises in each cell.

We first need to create a kernel in this case the size is 2-pixel-by-2-pixel.
We then apply morhology opening opeation on the cell to remove noises.

Morhology opening is a two-step process that first erode noises away and then dilate the remaining pixels.

```python
# Remove noises by using a 2x2 kernel
kernel = np.ones((2, 2), np.uint8)
thresh = cv2.morphologyEx(
    thresh,
    cv2.MORPH_OPEN,
    kernel
)
```

4. Find digit contour

In order to differentiate an empty cell and an occupied cell, we need to find the contour of the digit (if any) in the cell.

**cv2.RETR_EXTERNAL** mode instructs OpenCV to return ONLY the extreme outer contours and ignore any nested inner contours.

[cv2.CHAIN_APPROX_SIMPLE](https://docs.opencv.org/4.4.0/d3/dc0/group__imgproc__shape.html#ga4303f45752694956374734a03c54d5ff) specifies the method. It compresses horizontal, vertical, and diagonal segments and leaves only their end points.

```python
contours, _ = cv2.findContours(
    thresh,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)
```

There are two conditions that we use to tell a blank cell from an occupied cell.
The first condition is that there is no contour at all after calling *cv2.findContours*.
The second condition is that the largest contour by area is smaller than 80 pixels.
If the largest contour is smaller than this threshold, it is very likely that it is a noise or an artifact instead of a digit.

```python
# If there is no contour, then we deduce the current cell is blank
if len(contours) == 0:
    row.append(0)
    idx += 1
    continue

# Find the largest area
largest = max(contours, key=cv2.contourArea)
area = cv2.contourArea(largest)

# If the largest area is less than a threshold, then it must be a noise or an artifact
if area < 80:
    row.append(0)
    idx += 1
    continue
```

5. Extract digit

We now can assume there is a digit in the grid and we want to extrac it.
*cv2.boundingRect* returns the bounding box that surrounds the digit and we can just dissect the digit from the cell.

```python
x, y, w, h = cv2.boundingRect(largest)
digit = thresh[y:y+h, x:x+w]
```

6. Resize digit

The backend model that we use for training our OCR is trained on [TMNIST](https://www.kaggle.com/datasets/nimishmagre/tmnist-typeface-mnist).
We want to make the input as similar as the training images that we used in order to achieve high accuracy result.
Therefore, we need to resize the digit and put it on a 28x28 canvas.
We resize the digit to 18x18 since the digits used in training take up a large chunk of the entire canvas (not the entire canvas)
and we would like to replicate that property.

```python
target_size = 18
h_digit, w_digit = digit.shape
scale = target_size / max(h_digit, w_digit)
new_w = int(w_digit * scale)
new_h = int(h_digit * scale)
resized_digit = cv2.resize(digit, (new_w, new_h))
```

7. Place digit in the center of a black canvas

We create a 28x28 canvas and put the digit at the center of the canvas.
We calcualte the x and y offset so that we know how much to offset in order to do so.

```python
canvas = np.zeros((28, 28), dtype=np.uint8)
x_offset = (28 - new_w) // 2
y_offset = (28 - new_h) // 2
canvas[
    y_offset:y_offset+new_h,
    x_offset:x_offset+new_w
] = resized_digit
```

8. Normalize input

As we did for our training dataset, we need to also normalize the inputs in order for OCR to work properly.

```python
cell_input = (
    canvas
    .reshape(1, 28, 28, 1)
    .astype("float32") / 255.0
)
```

Voilà, now we have prepared the cell for our OCR and we can use the pretrained model to figure out the digit of the cell.

## Temp Path & Temp Path Factory

*tmp_path* is a function provided by **Pytest** that generates a temporary directory unique to each test function.

In this example, we create a temporary test file called *test_sudoku.json* that is stored in a temporary location (Pytest handles the actual location) and we use it in the session.
It will be destoried once the test is completed.

```python
def test_save_sudoku_json(tmp_path, sample_sudoku_board):

    test_file = tmp_path / "test_sudoku.json"

    save_sudoku_json(sample_sudoku_board, str(test_file))

    # Assert the file exists
    assert test_file.exists()

    # Assert the integrity of the data
    with open(test_file, "r") as file:
        save_data = json.load(file)

    assert save_data == {"board": sample_sudoku_board}
```

*tmp_path_factory* is a session-scoped fixture which can be used to create arbitrary temporary directories from any other fixture or test.

For large files that will be used across multiple functions, one can take advantage of *tmp_path_factory*.
The file that is created will be available in the current session so that you do not need to create one every single time.

```python
@pytest.fixture(scope="session")
def sudoku_board_json_file(tmp_path_factory, sample_sudoku_board):
    """
    Generates a JSON file that contains a sample Sudoku board
    """

    fn = tmp_path_factory.mktemp("sudoku_data") / "board.json"

    fn.write_text(json.dumps({"board": sample_sudoku_board}))

    return fn
```

And you can just use it like a fixture as shown in the following example:

```python
def test_open_sudoku_json(sudoku_board_json_file, sample_sudoku_board):
    """
    Tests open_sudoku_json correctly loads a Sudoku board from a JSON file
    and returns the board
    """

    loaded_board = open_sudoku_json(sudoku_board_json_file)

    assert loaded_board == sample_sudoku_board
```

## Reference

* [Image Thresholding ](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html)
* [sudoku-5](https://mathsphere.co.uk/downloads/sudoku/10202-medium.pdf)
* [TMNIST](https://www.kaggle.com/datasets/nimishmagre/tmnist-typeface-mnist)
