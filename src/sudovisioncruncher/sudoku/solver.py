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

        # Find the emtpy cell with the least amount of candidates
        for i, (r, c) in enumerate(empties):

            if board[r][c] != 0:
                continue

            b = box_index(r, c)

            candidates = (~(rows[r] | cols[c] | boxes[b])) & 0x3FE
            cnt = candidates.bit_count()

            if cnt == 0:
                return False

            if cnt < best_cnt:
                best_cnt = cnt
                best_candidate = candidates
                best_idx = i

                if cnt == 1:
                    break

        if best_idx == -1:
            return True

        r, c = empties[best_idx]
        b = box_index(r, c)

        candidates = best_candidate

        while candidates:

            bit = candidates & -candidates
            num = bit.bit_length() - 1

            # Place the digit
            board[r][c] = num
            rows[r] |= bit
            cols[c] |= bit
            boxes[b] |= bit

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
