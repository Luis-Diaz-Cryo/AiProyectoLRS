"""
Agente Connect-4 basado en Monte Carlo Tree Search (MCTS) con política UCB1.

Estrategia:
- Construcción incremental del árbol de búsqueda durante el tiempo de cómputo disponible.
- Selección de nodos mediante UCB1: balancea exploración vs explotación.
- Simulación (rollout) con política semi-aleatoria que prioriza movimientos ganadores/bloqueantes.
- Retropropagación de resultados hacia la raíz.
- Acción final: columna con más visitas (más robusta que la de mayor valor).

Variables configurables:
  - n_simulations: número de simulaciones MCTS por turno (default=500)
  - c_ucb: constante de exploración UCB1 (default=√2 ≈ 1.414)
  - rollout_depth: profundidad máxima del rollout (default=42, juego completo)
"""

import numpy as np
import math
import time
from typing import Optional
from connect4.policy import Policy

ROWS = 6
COLS = 7

# ── utilidades de tablero ──────────────────────────────────────────────────────

def _get_free_cols(board: np.ndarray) -> list[int]:
    return [c for c in range(COLS) if board[0, c] == 0]

def _drop(board: np.ndarray, col: int, player: int) -> np.ndarray:
    """Devuelve nuevo tablero con la ficha colocada (sin modificar el original)."""
    b = board.copy()
    for r in reversed(range(ROWS)):
        if b[r, col] == 0:
            b[r, col] = player
            break
    return b

def _check_win(board: np.ndarray, player: int) -> bool:
    """Comprueba si 'player' tiene 4 en raya."""
    b = board
    for r in range(ROWS):
        for c in range(COLS):
            if b[r, c] != player:
                continue
            # Horizontal
            if c + 3 < COLS and all(b[r, c+i] == player for i in range(4)):
                return True
            # Vertical
            if r + 3 < ROWS and all(b[r+i, c] == player for i in range(4)):
                return True
            # Diagonal ↘
            if r + 3 < ROWS and c + 3 < COLS and all(b[r+i, c+i] == player for i in range(4)):
                return True
            # Diagonal ↙
            if r + 3 < ROWS and c - 3 >= 0 and all(b[r+i, c-i] == player for i in range(4)):
                return True
    return False

def _is_terminal(board: np.ndarray) -> bool:
    return (
        _check_win(board, -1)
        or _check_win(board, 1)
        or len(_get_free_cols(board)) == 0
    )

def _get_winner(board: np.ndarray) -> int:
    if _check_win(board, -1):
        return -1
    if _check_win(board, 1):
        return 1
    return 0

# ── rollout con heurística ligera ──────────────────────────────────────────────

def _smart_move(board: np.ndarray, player: int, rng: np.random.Generator) -> int:
    """
    Selecciona movimiento para el rollout:
      1. Si hay movimiento ganador inmediato → jugarlo.
      2. Si el oponente tiene movimiento ganador → bloquearlo.
      3. Si no → columna aleatoria (favoreciendo el centro).
    """
    free = _get_free_cols(board)
    opponent = -player

    # 1. Ganar ahora
    for col in free:
        if _check_win(_drop(board, col, player), player):
            return col

    # 2. Bloquear victoria inmediata del oponente
    for col in free:
        if _check_win(_drop(board, col, opponent), opponent):
            return col

    # 3. Preferencia hacia el centro (pesos: 1,2,3,4,3,2,1)
    center_weights = np.array([1, 2, 3, 4, 3, 2, 1], dtype=float)
    weights = center_weights[free]
    weights /= weights.sum()
    return int(rng.choice(free, p=weights))

def _rollout(board: np.ndarray, player: int, rng: np.random.Generator) -> int:
    """Simula hasta el final; devuelve el ganador (-1, 0, 1)."""
    b = board.copy()
    p = player
    while not _is_terminal(b):
        col = _smart_move(b, p, rng)
        b = _drop(b, col, p)
        p = -p
    return _get_winner(b)

# ── nodo del árbol MCTS ────────────────────────────────────────────────────────

class _Node:
    __slots__ = ("board", "player", "parent", "action",
                 "children", "untried", "visits", "wins")

    def __init__(self, board: np.ndarray, player: int,
                 parent: Optional["_Node"] = None, action: Optional[int] = None):
        self.board   = board
        self.player  = player          # jugador que ACABA de mover (dueño de este nodo)
        self.parent  = parent
        self.action  = action          # columna que llevó aquí desde el padre
        self.children: list["_Node"] = []
        self.untried: list[int] = _get_free_cols(board)
        self.visits  = 0
        self.wins    = 0.0

    @property
    def is_fully_expanded(self) -> bool:
        return len(self.untried) == 0 and len(self.children) > 0

    @property
    def is_terminal(self) -> bool:
        return _is_terminal(self.board)

    def ucb1(self, c: float) -> float:
        if self.visits == 0:
            return float("inf")
        exploitation = self.wins / self.visits
        exploration  = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def best_child(self, c: float) -> "_Node":
        return max(self.children, key=lambda n: n.ucb1(c))

    def expand(self, rng: np.random.Generator) -> "_Node":
        """Expande un hijo no visitado al azar."""
        col = int(rng.choice(self.untried))
        self.untried.remove(col)
        next_player = -self.player
        child = _Node(
            board  = _drop(self.board, col, next_player),
            player = next_player,
            parent = self,
            action = col,
        )
        self.children.append(child)
        return child

    def backpropagate(self, result: int, root_next_player: int) -> None:
        """
        Sube el resultado por el árbol.
        'result' ∈ {-1, 0, 1}: jugador ganador.
        Contabiliza victoria desde la perspectiva del nodo que eligió la acción,
        es decir, desde la perspectiva del jugador que tiene el turno en el PADRE.
        """
        node = self
        while node is not None:
            node.visits += 1
            # El padre eligió la acción que lleva a este nodo;
            # el jugador activo en el padre es -node.player
            acting_player = -node.player
            if result == acting_player:
                node.wins += 1.0
            elif result == 0:
                node.wins += 0.5       # empate: medio punto
            node = node.parent

# ── política MCTS ──────────────────────────────────────────────────────────────

class MCTSAgent(Policy):
    """
    Agente Connect-4 con Monte Carlo Tree Search y selección UCB1.

    Parámetros
    ----------
    n_simulations : int
        Número de simulaciones por turno. Más simulaciones → mejor juego,
        pero mayor tiempo de cómputo.
    c_ucb : float
        Constante de exploración UCB1. Mayor c → más exploración.
        El valor teórico óptimo es √2; en la práctica 1.0–2.0 funciona bien.
    time_limit : float | None
        Si se proporciona, se usan simulaciones hasta agotar el tiempo (segundos).
        Si es None se usa n_simulations como límite.
    """

    def __init__(self,
                 n_simulations: int = 800,
                 c_ucb: float = math.sqrt(2),
                 time_limit: Optional[float] = None):
        self.n_simulations = n_simulations
        self.c_ucb         = c_ucb
        self.time_limit    = time_limit
        self._rng          = np.random.default_rng()

    def mount(self) -> None:
        """Llamado al inicio de cada partida — reinicia el generador."""
        self._rng = np.random.default_rng()

    # ── método principal ───────────────────────────────────────────────────────

    def act(self, s: np.ndarray) -> int:
        """
        Recibe el tablero actual y devuelve la columna elegida.
        s : np.ndarray de shape (6, 7); -1=Rojo, 1=Amarillo, 0=vacío.
        El turno del agente se infiere por la paridad de fichas.
        """
        board      = s.copy()
        my_player  = self._infer_player(board)

        # Si hay movimiento ganador inmediato → jugarlo sin perder tiempo
        free = _get_free_cols(board)
        for col in free:
            if _check_win(_drop(board, col, my_player), my_player):
                return col

        # Si el oponente ganaría en su turno → bloquearlo
        opponent = -my_player
        for col in free:
            if _check_win(_drop(board, col, opponent), opponent):
                return col

        # MCTS
        root = _Node(board=board, player=opponent)  # el "último" en mover fue el oponente
        root.untried = free[:]

        if self.time_limit is not None:
            deadline = time.time() + self.time_limit
            while time.time() < deadline:
                self._simulate(root)
        else:
            for _ in range(self.n_simulations):
                self._simulate(root)

        # Elegir la columna más visitada (más robusta que la de mayor winrate)
        best = max(root.children, key=lambda n: n.visits)
        return best.action

    # ── un paso de MCTS ────────────────────────────────────────────────────────

    def _simulate(self, root: _Node) -> None:
        # 1. SELECCIÓN
        node = root
        while not node.is_terminal and node.is_fully_expanded:
            node = node.best_child(self.c_ucb)

        # 2. EXPANSIÓN
        if not node.is_terminal and not node.is_fully_expanded:
            node = node.expand(self._rng)

        # 3. SIMULACIÓN (rollout)
        # El turno siguiente al nodo actual es -node.player
        result = _rollout(node.board, -node.player, self._rng)

        # 4. RETROPROPAGACIÓN
        node.backpropagate(result, -root.player)

    # ── inferencia del turno ───────────────────────────────────────────────────

    @staticmethod
    def _infer_player(board: np.ndarray) -> int:
        """
        Infiere quién debe jugar en base a la paridad de fichas.
        Rojo (−1) siempre abre: si #rojas == #amarillas → turno de Rojo.
        """
        reds    = np.sum(board == -1)
        yellows = np.sum(board ==  1)
        return -1 if reds == yellows else 1