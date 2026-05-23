import json
import pytest

from src.board_io import save_sudoku_json, open_sudoku_json

@pytest.fixture(scope="session")
def sample_sudoku_board():
    """
    Generates a valid 9x9 Sudoku board
    """

    return [
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

@pytest.fixture(scope="session")
def sudoku_board_json_file(tmp_path_factory, sample_sudoku_board):
    """
    Generates a JSON file that contains a sample Sudoku board
    """

    fn = tmp_path_factory.mktemp("sudoku_data") / "board.json"

    fn.write_text(json.dumps({"board": sample_sudoku_board}))

    return fn

def test_save_sudoku_json(tmp_path, sample_sudoku_board):
    """
    Tests save_sudoku_json correctly saves a Sudoku board to a JSON file
    """

    test_file = tmp_path / "test_sudoku.json"

    save_sudoku_json(sample_sudoku_board, str(test_file))

    # Assert the file exists
    assert test_file.exists()

    # Assert the integrity of the data
    with open(test_file, "r") as file:
        save_data = json.load(file)

    assert save_data == {"board": sample_sudoku_board}

def test_open_sudoku_json(sudoku_board_json_file, sample_sudoku_board):
    """
    Tests open_sudoku_json correctly loads a Sudoku board from a JSON file
    and returns the board
    """

    loaded_board = open_sudoku_json(sudoku_board_json_file)

    assert loaded_board == sample_sudoku_board

def test_save_and_open_integration(tmp_path, sample_sudoku_board):
    """
    Integration test of the save and open function
    """

    test_file = str(tmp_path / "sudoku_integration.json")

    save_sudoku_json(sample_sudoku_board, test_file)
    loaded_board = open_sudoku_json(test_file)

    assert loaded_board == sample_sudoku_board

def test_open_sudoku_json_file_not_found():
    """
    Tests that FileNotFoundError is raised when the file does not exist
    """

    with pytest.raises(FileNotFoundError):
        open_sudoku_json("non_existent_sudoku.json")

def test_open_sudoku_json_empty_file(tmp_path):
    """
    Tests that json.JSONDecodeError is raised when the file does not exist
    """

    # Create an empty file
    empty_file = tmp_path / "empty_sudoku.json"
    empty_file.write_text("")

    loaded_board = open_sudoku_json(str(empty_file))

    assert loaded_board == []
