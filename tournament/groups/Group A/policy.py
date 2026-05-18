import numpy as np
from connect4.policy import Policy


class LuisAgent(Policy):
    ROWS = 6
    COLS = 7
    WIN_SCORE = 1_000_000

    def __init__(self):
        self.depth = 4
        self.order = [3, 2, 4, 1, 5, 0, 6]

    def mount(self, *args, **kwargs) -> None:
        self.depth = 4
        self.order = [3, 2, 4, 1, 5, 0, 6]

    def act(self, s: np.ndarray) -> int:
        board = np.array(s, dtype=int)

        if not hasattr(self, "order"):
            self.order = [3, 2, 4, 1, 5, 0, 6]

        if not hasattr(self, "depth"):
            self.depth = 4

        player = self._current_player(board)
        valid = self._valid_moves(board)

        if not valid:
            return 0

        for col in valid:
            if self._winner(self._drop(board, col, player)) == player:
                return int(col)

        opponent = -player
        for col in valid:
            if self._winner(self._drop(board, col, opponent)) == opponent:
                return int(col)

        best_col = valid[0]
        best_score = -float("inf")
        alpha = -float("inf")
        beta = float("inf")

        for col in self.order:
            if col not in valid:
                continue

            new_board = self._drop(board, col, player)
            score = self._minimax(new_board, self.depth - 1, alpha, beta, False, player)

            if score > best_score:
                best_score = score
                best_col = col

            alpha = max(alpha, best_score)

        return int(best_col)

    def _current_player(self, board: np.ndarray) -> int:
        red_count = int(np.sum(board == -1))
        yellow_count = int(np.sum(board == 1))
        return -1 if red_count == yellow_count else 1

    def _valid_moves(self, board: np.ndarray) -> list[int]:
        return [c for c in self.order if board[0, c] == 0]

    def _drop(self, board: np.ndarray, col: int, player: int) -> np.ndarray:
        new_board = board.copy()
        for row in range(self.ROWS - 1, -1, -1):
            if new_board[row, col] == 0:
                new_board[row, col] = player
                return new_board
        return new_board

    def _is_terminal(self, board: np.ndarray) -> bool:
        return self._winner(board) != 0 or len(self._valid_moves(board)) == 0

    def _winner(self, board: np.ndarray) -> int:
        for r in range(self.ROWS):
            for c in range(self.COLS):
                piece = board[r, c]
                if piece == 0:
                    continue
                if c + 3 < self.COLS and all(board[r, c + i] == piece for i in range(4)):
                    return int(piece)
                if r + 3 < self.ROWS and all(board[r + i, c] == piece for i in range(4)):
                    return int(piece)
                if r + 3 < self.ROWS and c + 3 < self.COLS and all(board[r + i, c + i] == piece for i in range(4)):
                    return int(piece)
                if r + 3 < self.ROWS and c - 3 >= 0 and all(board[r + i, c - i] == piece for i in range(4)):
                    return int(piece)
        return 0

    def _minimax(self, board: np.ndarray, depth: int, alpha: float, beta: float, maximizing: bool, player: int) -> float:
        opponent = -player
        winner = self._winner(board)
        if winner == player:
            return self.WIN_SCORE + depth
        if winner == opponent:
            return -self.WIN_SCORE - depth
        if depth == 0 or len(self._valid_moves(board)) == 0:
            return self._score_board(board, player)

        if maximizing:
            value = -float("inf")
            for col in self._valid_moves(board):
                value = max(value, self._minimax(self._drop(board, col, player), depth - 1, alpha, beta, False, player))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value

        value = float("inf")
        for col in self._valid_moves(board):
            value = min(value, self._minimax(self._drop(board, col, opponent), depth - 1, alpha, beta, True, player))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    def _score_board(self, board: np.ndarray, player: int) -> float:
        score = 0.0
        center = board[:, self.COLS // 2]
        score += 6 * int(np.sum(center == player))
        for window in self._windows(board):
            score += self._score_window(window, player)
        return score

    def _windows(self, board: np.ndarray):
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                yield list(board[r, c:c + 4])
        for c in range(self.COLS):
            for r in range(self.ROWS - 3):
                yield list(board[r:r + 4, c])
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                yield [board[r + i, c + i] for i in range(4)]
        for r in range(self.ROWS - 3):
            for c in range(3, self.COLS):
                yield [board[r + i, c - i] for i in range(4)]

    def _score_window(self, window: list[int], player: int) -> float:
        opponent = -player
        own = window.count(player)
        enemy = window.count(opponent)
        empty = window.count(0)

        if own == 4:
            return 100000
        if own == 3 and empty == 1:
            return 80
        if own == 2 and empty == 2:
            return 12
        if enemy == 3 and empty == 1:
            return -100
        if enemy == 2 and empty == 2:
            return -15
        return 0
