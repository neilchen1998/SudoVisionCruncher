def solve_sudoku(board: list[list[int]]) -> list[list[int]]:
    """
    Solves Sudoku using MRV (Minimum Remaining Values heuristic) method.

    Args:
        board: The Sudoku board

    Returns:
        list[list[int]]: The solved Sudoku board
    """

    return SudokuMRVSolver(board).solve()


class SudokuMRVSolver:


    def __init__(self, board: list[list[int]]):

        self.board = board

        # Bitmasks for constraints
        self.rows = [0] * 9
        self.cols = [0] * 9
        self.boxes = [0] * 9
        self.empties: list[tuple[int, int]] = []

        # Initialize the constraints
        for r in range(9):
            for c in range(9):

                num = self.board[r][c]

                if num == 0:
                    self.empties.append((r, c))
                    continue

                # Convert the digit to bit
                bit = 1 << num

                # Store it in the constraints
                self.rows[r] |= bit
                self.cols[c] |= bit
                self.boxes[self._box_index(r, c)] |= bit

    def solve(self) -> list[list[int]]:
        """
        Solves the Sudoku given puzzle.

        Returns:
            list[list[int]]: The solved puzzle
        """

        self._backtrack()

        return self.board

    @staticmethod
    def _box_index(r: int, c: int) -> int:
        """
        Calculates the box index based on the cell.

        Args:
            r: the row index
            c: the column index

        Returns:
            int: the box index
        """

        return (r // 3) * 3 + (c // 3)

    def _find_best_cell(self) -> tuple[int, int]:
        """
        Finds the emtpy cell with the fewest candidates.

        Returns:
            tuple:
                int: index of the cell with the fewest candidates
                int: the mask of its candidates
        """

        best_idx = -1
        best_candidates = 0
        best_cnt = 10

        # Loop through all emtpy cells to find the cell that ahs the least amount of candidates
        for i, (r, c) in enumerate(self.empties):

            # Skip non-empty cells
            if self.board[r][c] != 0:
                continue

            b = self._box_index(r, c)

            # Get all the candidates in bit-format
            candidates = (~(self.rows[r] | self.cols[c] | self.boxes[b])) & 0x3FE
            cnt = candidates.bit_count()

            # This is a dead end since it is impossible to have a cell w/o candidate
            if cnt == 0:
                return -2, 0

            # Update the current best candidate
            if cnt < best_cnt:
                best_cnt = cnt
                best_candidates = candidates
                best_idx = i

                # Early break since this is the best candidate that we can get
                if cnt == 1:
                    break

        return best_idx, best_candidates

    def _backtrack(self) -> bool:
        """
        Recursively solves the Sudoku puzzle using backtracking.

        Returns:
            bool: True if the Sudoku is solved successfully, False if no valid number can be placed
        """

        idx, candidates = self._find_best_cell()

        # Early return if there is no non-empty cell
        if idx == -1:
            return True

        # Early return if there is a dead end
        elif idx == -2:
            return False

        # Get the metadata of the best cell
        r, c = self.empties[idx]
        b = self._box_index(r, c)

        # Loop through all the possible candidates
        while candidates:

            # Get the digit from the bit-format
            bit = candidates & -candidates
            num = bit.bit_length() - 1

            # Place the digit
            self.board[r][c] = num
            self.rows[r] |= bit
            self.cols[c] |= bit
            self.boxes[b] |= bit

            # Proceed to find next cell
            if self._backtrack():
                return True

            # Undo
            self.board[r][c] = 0
            self.rows[r] ^= bit
            self.cols[c] ^= bit
            self.boxes[b] ^= bit
            candidates &= (candidates - 1)

        return False
