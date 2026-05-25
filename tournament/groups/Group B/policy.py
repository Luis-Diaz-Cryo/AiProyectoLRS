import numpy as np
from connect4.policy import Policy

ROWS = 6
COLS = 7
CENTER = COLS // 2


class MinimaxAlphaBeta(Policy):

    def __init__(self, depth: int = 6):
        self.depth = depth

    def mount(self, timeout=None) -> None:
        pass

    # ------------------------------------------------------------------ helpers

    def _my_player(self, board: np.ndarray) -> int:
        """Infer whose turn it is: -1 (Red) if counts are equal, else 1 (Yellow)."""
        return -1 if np.count_nonzero(board == -1) == np.count_nonzero(board == 1) else 1

    def _free_cols(self, board: np.ndarray) -> list[int]:
        return [c for c in range(COLS) if board[0, c] == 0]

    def _drop(self, board: np.ndarray, col: int, player: int) -> np.ndarray:
        b = board.copy()
        for r in reversed(range(ROWS)):
            if b[r, col] == 0:
                b[r, col] = player
                break
        return b

    def _winner(self, board: np.ndarray) -> int:
        for r in range(ROWS):
            for c in range(COLS):
                p = board[r, c]
                if p == 0:
                    continue
                if c + 3 < COLS and all(board[r, c + i] == p for i in range(4)):
                    return p
                if r + 3 < ROWS and all(board[r + i, c] == p for i in range(4)):
                    return p
                if r + 3 < ROWS and c + 3 < COLS and all(board[r + i, c + i] == p for i in range(4)):
                    return p
                if r + 3 < ROWS and c - 3 >= 0 and all(board[r + i, c - i] == p for i in range(4)):
                    return p
        return 0

    # ---------------------------------------------------------------- heuristic

    def _window_score(self, window: list, player: int) -> int:
        opp = -player
        score = 0
        pc = window.count(player)
        oc = window.count(opp)
        ec = window.count(0)
        if pc == 4:
            score += 100
        elif pc == 3 and ec == 1:
            score += 5
        elif pc == 2 and ec == 2:
            score += 2
        if oc == 3 and ec == 1:
            score -= 4
        return score

    def _evaluate(self, board: np.ndarray, player: int) -> int:
        score = 0
        # center preference
        score += list(board[:, CENTER]).count(player) * 3
        # horizontal
        for r in range(ROWS):
            for c in range(COLS - 3):
                score += self._window_score(list(board[r, c:c + 4]), player)
        # vertical
        for c in range(COLS):
            for r in range(ROWS - 3):
                score += self._window_score(list(board[r:r + 4, c]), player)
        # diagonal right-down
        for r in range(ROWS - 3):
            for c in range(COLS - 3):
                score += self._window_score([board[r + i, c + i] for i in range(4)], player)
        # diagonal left-down
        for r in range(ROWS - 3):
            for c in range(3, COLS):
                score += self._window_score([board[r + i, c - i] for i in range(4)], player)
        return score

    # -------------------------------------------------------------- minimax

    def _minimax(
        self,
        board: np.ndarray,
        depth: int,
        alpha: float,
        beta: float,
        maximizing: bool,
        my_player: int,
    ) -> tuple[float, int | None]:
        free = self._free_cols(board)
        w = self._winner(board)

        if w == my_player:
            return (1_000_000 + depth, None)
        if w == -my_player:
            return (-1_000_000 - depth, None)
        if not free:
            return (0, None)
        if depth == 0:
            return (self._evaluate(board, my_player), None)

        # move ordering: center columns first
        ordered = sorted(free, key=lambda c: abs(c - CENTER))
        current = my_player if maximizing else -my_player
        best_col = ordered[0]

        if maximizing:
            best = float("-inf")
            for col in ordered:
                val, _ = self._minimax(self._drop(board, col, current), depth - 1, alpha, beta, False, my_player)
                if val > best:
                    best, best_col = val, col
                alpha = max(alpha, best)
                if alpha >= beta:
                    break
            return best, best_col
        else:
            best = float("inf")
            for col in ordered:
                val, _ = self._minimax(self._drop(board, col, current), depth - 1, alpha, beta, True, my_player)
                if val < best:
                    best, best_col = val, col
                beta = min(beta, best)
                if alpha >= beta:
                    break
            return best, best_col

    # ----------------------------------------------------------------- act

    def act(self, s: np.ndarray) -> int:
        my_player = self._my_player(s)
        _, col = self._minimax(s, self.depth, float("-inf"), float("inf"), True, my_player)
        if col is None:
            free = self._free_cols(s)
            return free[len(free) // 2]
        return col
