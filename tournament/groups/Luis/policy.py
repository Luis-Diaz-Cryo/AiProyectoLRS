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
        opponent = -player
        valid = self._valid_moves(board)

        if not valid:
            return 0

        # 1. If the agent can win immediately, it must take that move.
        for col in valid:
            if self._winner(self._drop(board, col, player)) == player:
                return int(col)

        # 2. If the opponent can win immediately, the agent must block.
        for col in valid:
            if self._winner(self._drop(board, col, opponent)) == opponent:
                return int(col)

        # 3. Risk avoidance:
        # Remove moves that allow the opponent to win immediately next turn.
        safe_moves = []
        for col in valid:
            next_board = self._drop(board, col, player)
            if not self._has_immediate_win(next_board, opponent):
                safe_moves.append(col)

        candidates = safe_moves if safe_moves else valid

        # 4. Expectimax:
        # The agent maximizes its expected score.
        # The opponent is modeled as probabilistic/random instead of perfectly optimal.
        best_col = candidates[0]
        best_score = -float("inf")

        for col in self.order:
            if col not in candidates:
                continue

            new_board = self._drop(board, col, player)
            score = self._expectimax(new_board, self.depth - 1, False, player)

            if score > best_score:
                best_score = score
                best_col = col

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

    def _winner(self, board: np.ndarray) -> int:
        for r in range(self.ROWS):
            for c in range(self.COLS):
                piece = board[r, c]

                if piece == 0:
                    continue

                if c + 3 < self.COLS:
                    if all(board[r, c + i] == piece for i in range(4)):
                        return int(piece)

                if r + 3 < self.ROWS:
                    if all(board[r + i, c] == piece for i in range(4)):
                        return int(piece)

                if r + 3 < self.ROWS and c + 3 < self.COLS:
                    if all(board[r + i, c + i] == piece for i in range(4)):
                        return int(piece)

                if r + 3 < self.ROWS and c - 3 >= 0:
                    if all(board[r + i, c - i] == piece for i in range(4)):
                        return int(piece)

        return 0

    def _has_immediate_win(self, board: np.ndarray, player: int) -> bool:
        for col in self._valid_moves(board):
            if self._winner(self._drop(board, col, player)) == player:
                return True
        return False

    def _immediate_winning_moves(self, board: np.ndarray, player: int) -> list[int]:
        moves = []

        for col in self._valid_moves(board):
            if self._winner(self._drop(board, col, player)) == player:
                moves.append(col)

        return moves

    def _expectimax(
        self,
        board: np.ndarray,
        depth: int,
        maximizing: bool,
        player: int
    ) -> float:
        opponent = -player
        winner = self._winner(board)
        valid = self._valid_moves(board)

        if winner == player:
            return self.WIN_SCORE + depth

        if winner == opponent:
            return -self.WIN_SCORE - depth

        if depth == 0 or not valid:
            return self._score_board(board, player)

        if maximizing:
            best_score = -float("inf")

            for col in valid:
                new_board = self._drop(board, col, player)
                score = self._expectimax(new_board, depth - 1, False, player)

                if score > best_score:
                    best_score = score

            return best_score

        # Opponent turn:
        # Instead of assuming the opponent always plays the best move like minimax,
        # expectimax averages all possible opponent moves.
        total_score = 0.0

        for col in valid:
            new_board = self._drop(board, col, opponent)
            total_score += self._expectimax(new_board, depth - 1, True, player)

        return total_score / len(valid)

    def _score_board(self, board: np.ndarray, player: int) -> float:
        opponent = -player
        score = 0.0

        # Center control is valuable because the center column participates
        # in more possible Connect-4 lines.
        center = board[:, self.COLS // 2]
        score += 8 * int(np.sum(center == player))
        score -= 6 * int(np.sum(center == opponent))

        # Evaluate every possible group of four cells.
        for window in self._windows(board):
            score += self._score_window(window, player)

        # Extra tactical scoring:
        # Reward immediate winning threats and strongly punish opponent threats.
        own_threats = len(self._immediate_winning_moves(board, player))
        enemy_threats = len(self._immediate_winning_moves(board, opponent))

        score += 250 * own_threats
        score -= 350 * enemy_threats

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
            return 90

        if own == 2 and empty == 2:
            return 15

        if enemy == 4:
            return -100000

        if enemy == 3 and empty == 1:
            return -140

        if enemy == 2 and empty == 2:
            return -25

        return 0