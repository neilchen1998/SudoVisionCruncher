def solve_sudoku(board: list[list]) -> list[list]:
    """
    Solves Sudoku

    Args:
        board: The Sudoku board

    Returns:
        list[list]: The solved Sudoku board
    """

    # Bitmasks for constraints
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    empties = []

    def box_index(r: int, c: int) -> int:
        """
        Calculates the box index based on the cell

        Args:
            r: the row index
            c: the column index

        Returns:
            int: the box index
        """

        return (r // 3) * 3 + (c // 3)

    # Initialize masks
    for r in range(9):
        for c in range(9):
            if board[r][c] == 0:
                empties.append((r, c))
            else:
                bit = 1 << board[r][c]
                rows[r] |= bit
                cols[c] |= bit
                boxes[box_index(r, c)] |= bit

    def backtrack(i: int = 0) -> bool:
        """
        Recursively solves the Sudoku puzzle using backtracking

        Args:
            i: The index of the empty list

        Returns:
            bool: True if the Sudoku is solved successfully, False if no valid number can be placed
        """

        # Check if we have finished backtracking
        if i >= len(empties):
            return True

        # Get the indices of the current cell
        r, c = empties[i]
        b = box_index(r, c)

        # Get the used numbers in a bitmask format
        used = rows[r] | cols[c] | boxes[b]

        # Loop through all the options
        for num in range(1, 10):

            # Convert the digit into bit format
            bit = 1 << num

            # Check if the digit is available
            if used & bit == 0:

                # Place the digit
                board[r][c] = num
                rows[r] |= bit
                cols[c] |= bit
                boxes[b] |= bit

                # Backtrack the next index
                if backtrack(i + 1):
                    return True

                # Undo
                board[r][c] = 0
                rows[r] ^= bit
                cols[c] ^= bit
                boxes[b] ^= bit

        # If we reach here that means we have exhausted all the possibilities and the puzzle cannot be solved
        return False

    backtrack()

    return board
