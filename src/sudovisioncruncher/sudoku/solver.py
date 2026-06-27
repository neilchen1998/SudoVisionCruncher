def solve_sudoku(board: list[list[int]]) -> list[list[int]]:
    """
    Solves Sudoku using MRV (Minimum Remaining Values heuristic) method.

    Args:
        board: The Sudoku board

    Returns:
        list[list[int]]: The solved Sudoku board
    """

    # Bitmasks for constraints
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    empties = []

    def box_index(r: int, c: int) -> int:
        """
        Calculates the box index based on the cell.

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

        best_idx = -1
        best_candidate = 0
        best_cnt = 10

        # Loop through all emtpy cells to find the cell that ahs the least amount of candidates
        for i, (r, c) in enumerate(empties):

            # Skip non-empty cells
            if board[r][c] != 0:
                continue

            b = box_index(r, c)

            # Get all the candidates in bit-format
            candidates = (~(rows[r] | cols[c] | boxes[b])) & 0x3FE
            cnt = candidates.bit_count()

            # This is a dead end since it is impossible to have a cell w/o candidate
            if cnt == 0:
                return False

            # Update the current best candidate
            if cnt < best_cnt:
                best_cnt = cnt
                best_candidate = candidates
                best_idx = i

                # Early break since this is the best candidate that we can get
                if cnt == 1:
                    break

        # Early return if there is no non-empty cell
        if best_idx == -1:
            return True

        # Get the metadata of the best cell
        r, c = empties[best_idx]
        b = box_index(r, c)
        candidates = best_candidate

        # Loop through all the possible candidates
        while candidates:

            # Get the digit from the bit-format
            bit = candidates & -candidates
            num = bit.bit_length() - 1

            # Place the digit
            board[r][c] = num
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit

            # Proceed to find next cell
            if backtrack():
                return True

            # Undo
            board[r][c] = 0
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit
            candidates &= (candidates - 1)

        return False

    backtrack()

    return board
