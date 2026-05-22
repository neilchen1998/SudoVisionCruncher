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

```python
```

```python
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

6. Resize digit

7. Place digit in the center of a black canvas

8. Normalize input

## Reference

* [Image Thresholding ](https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_thresholding/py_thresholding.html)

* [sudoku-5](https://mathsphere.co.uk/downloads/sudoku/10202-medium.pdf)
