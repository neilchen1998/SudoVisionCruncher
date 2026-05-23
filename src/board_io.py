import json

def save_sudoku_json(board: list[list], file_path: str):
    """
    Saves a Sudoku board grid to a JSON file.

    Args:
        board: A list of lists that represents the Sudoku grid
        file_path The file path where the JSON data should be saved

    Raises:
        OSError: If the file cannot be opened or written to
    """

    # Open and write the JSON file
    with open(file_path, "w") as file:
        json.dump({"board": board}, file, indent=2)

def open_sudoku_json(file_path: str) -> list[list]:

    """
    Loads a Sudoku board from the given JSON file

    Args:
        file_path: The JSON file path

    Returns:
        list[list]: The Sudoku board

    Raises:
        FileNotFoundError: The given file does not exist
        json.JSONDecodeError: The given JSON file is empty or invalid
        KeyError: The given JSON file is missing the "board" key
    """

    try:
        # Open and read the JSON file
        with open(file_path, "r") as file:
            data = json.load(file)

        return data["board"]
    except json.JSONDecodeError:
        return []
