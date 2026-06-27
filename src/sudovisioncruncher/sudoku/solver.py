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

    def _find_best_cell(self) -> tuple[int, int, int] | None:
        """
        Finds the emtpy cell with the fewest candidates using MRV heuristic

        Returns:
            tuple | None:
                int: the row index of the fewest candidates
                int: the column index of the fewest candidates
                int: the mask of its candidates
                None: if there is no empty cell on the board
        """

        best_r, best_c = -1, -1
        best_candidates = 0
        best_cnt = 10

        # Loop through all emtpy cells to find the cell that ahs the least amount of candidates
        for r, c in self.empties:

            # Skip non-empty cells
            if self.board[r][c] != 0:
                continue

            b = self._box_index(r, c)

            # Get all the candidates in bit-format
            candidates = (~(self.rows[r] | self.cols[c] | self.boxes[b])) & 0x3FE
            cnt = candidates.bit_count()

            # This is a dead end since it is impossible to have a cell w/o candidate
            if cnt == 0:
                return r, c, 0

            # Update the current best candidate
            if cnt < best_cnt:
                best_cnt = cnt
                best_candidates = candidates
                best_r, best_c = r, c

                # Early break since this is the best candidate that we can get
                if cnt == 1:
                    break

        if best_cnt == 10:
            return None

        return best_r, best_c, best_candidates

    def _backtrack(self) -> bool:
        """
        Recursively solves the Sudoku puzzle using backtracking.

        Returns:
            bool: True if the Sudoku is solved successfully, False if no valid number can be placed
        """

        result = self._find_best_cell()

        # Early return if there is no non-empty cell
        if result is None:
            return True

        # Get the metadata of the best cell
        r, c, candidates = result

        # Early return if there is a dead end
        if candidates == 0:
            return False

        b = self._box_index(r, c)

        # Cache references locally to optimize lookups
        board = self.board
        rows = self.rows
        cols = self.cols
        boxes = self.boxes

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
            if self._backtrack():
                return True

            # Undo
            board[r][c] = 0
            rows[r] ^= bit
            cols[c] ^= bit
            boxes[b] ^= bit
            candidates &= (candidates - 1)

        return False
