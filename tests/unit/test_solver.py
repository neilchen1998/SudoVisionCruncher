import pytest
import random
from hypothesis import given, strategies as st

from sudoku.solver import solve_sudoku
from sudoku.utils import is_valid_sudoku

EASY_PUZZLE = [
    [5,3,0,0,7,0,0,0,0],
    [6,0,0,1,9,5,0,0,0],
    [0,9,8,0,0,0,0,6,0],
    [8,0,0,0,6,0,0,0,3],
    [4,0,0,8,0,3,0,0,1],
    [7,0,0,0,2,0,0,0,6],
    [0,6,0,0,0,0,2,8,0],
    [0,0,0,4,1,9,0,0,5],
    [0,0,0,0,8,0,0,7,9],
]

SOLUTION_EASY_PUZZLE = [
    [5,3,4,6,7,8,9,1,2],
    [6,7,2,1,9,5,3,4,8],
    [1,9,8,3,4,2,5,6,7],
    [8,5,9,7,6,1,4,2,3],
    [4,2,6,8,5,3,7,9,1],
    [7,1,3,9,2,4,8,5,6],
    [9,6,1,5,3,7,2,8,4],
    [2,8,7,4,1,9,6,3,5],
    [3,4,5,2,8,6,1,7,9],
]

SEVENTH_CLUE_PUZZLE = [
    [0, 0, 0, 0, 0, 0, 0, 1, 0],
    [4, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 2, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 5, 0, 4, 0, 7],
    [0, 0, 8, 0, 0, 0, 3, 0, 0],
    [0, 0, 1, 0, 9, 0, 0, 0, 0],
    [3, 0, 0, 4, 0, 0, 2, 0, 0],
    [0, 5, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 8, 0, 6, 0, 0, 0]
]

SOLUTION_SEVENTH_CLUE_PUZZLE = [
    [6, 9, 3, 7, 8, 4, 5, 1, 2],
    [4, 8, 7, 5, 1, 2, 9, 3, 6],
    [1, 2, 5, 9, 6, 3, 8, 7, 4],
    [9, 3, 2, 6, 5, 1, 4, 8, 7],
    [5, 6, 8, 2, 4, 7, 3, 9, 1],
    [7, 4, 1, 3, 9, 8, 6, 2, 5],
    [3, 1, 9, 4, 7, 5, 2, 6, 8],
    [8, 5, 6, 1, 2, 9, 7, 4, 3],
    [2, 7, 4, 8, 3, 6, 1, 5, 9]
]

INVALID_BOARD = [
    [5,5,0,0,7,0,0,0,0],    # There are two 5's in the first row
    [6,0,0,1,9,5,0,0,0],
    [0,9,8,0,0,0,0,6,0],
    [8,0,0,0,6,0,0,0,3],
    [4,0,0,8,0,3,0,0,1],
    [7,0,0,0,2,0,0,0,6],
    [0,6,0,0,0,0,2,8,0],
    [0,0,0,4,1,9,0,0,5],
    [0,0,0,0,8,0,0,7,9],
]


def make_puzzle(solved_board: list[list[int]], n: int) -> list[list[int]]:
    """
    Generates Sudoku puzzle by blanking out n amount of random cells
    NOTE: the value of n cannot be lower than 17

    Args:
        solved_board: A solved valid Sudoku board
        n: The number of blank cells

    Returns:
        list[list[int]]: An unsolved Sudoku board
    """

    # Make a deep copy of the board
    puzzle = [row[:] for row in solved_board]

    # Generate a list of indices
    cells = [(r, c) for r in range(9) for c in range(9)]

    # Shuffle the list
    random.shuffle(cells)

    # Grab the top n elements from the list and make them blank
    for r, c in cells[:n]:
        puzzle[r][c] = 0

    return puzzle

@pytest.mark.parametrize(
    "puzzle, expected",
    [
        (EASY_PUZZLE, SOLUTION_EASY_PUZZLE),
        (SEVENTH_CLUE_PUZZLE, SOLUTION_SEVENTH_CLUE_PUZZLE),
    ]
)
def test_solve_sudoku(puzzle, expected):
    """
    Tests solve_sudoku with predefined Sudoku puzzles and known expected answers
    """

    solved = solve_sudoku(puzzle)

    assert is_valid_sudoku(solved) == True

    assert solved == expected

@given(st.integers(min_value=17, max_value=60))
def test_solve_sudoku_with_random_puzzles(blank_count):
    """
    Tests solve_sudoku with random Sudoku puzzles with Hypothesis
    """

    puzzle = make_puzzle(SOLUTION_EASY_PUZZLE, n=blank_count)
    solved = solve_sudoku(puzzle)

    assert is_valid_sudoku(solved)
