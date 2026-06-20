def is_valid_sequence(seq: list[int]) -> bool:
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


def is_valid_sudoku(board: list[list[int]]) -> bool:
    """
    Checks if a given Sudoku board is valid

    Args:
        board: A 9x9 list of integers representing the input Sudoku board

    Returns:
        bool: True if the Sudoku board is valid, False otherwise
    """

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


def print_board(board: list[list[int]], width: int = 3) -> None:
    """
    Prints the Sudoku board

    Args:
        board: The Sudoku board
        width: The width of each digit in character
    """

    # The top border
    print("+" + ("-" * (width + 2) + "+") * len(board[0]))

    for row in board:
        print("|" + "|".join(f" {str(cell).center(width)} " for cell in row) + "|")
        print("+" + ("-" * (width + 2) + "+") * len(board[0]))
