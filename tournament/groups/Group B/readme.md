# Agente Connect-4 — MinimaxAlphaBeta
**Fundamentos de Inteligencia Artificial · Universidad de La Sabana 2026.1**

---

## Descripción

Agente de Connect-4 basado en **Minimax con poda Alpha-Beta**. Explora el árbol de juego hasta una profundidad configurable y escoge el movimiento que maximiza el beneficio propio asumiendo que el oponente juega de forma óptima.

**Diferencia clave respecto a los demás agentes del grupo:** Minimax garantiza optimalidad local hasta la profundidad buscada, sin depender de aprendizaje ni aleatoriedad.

---

## Estructura de archivos

```
Group B/
├── policy.py       # Implementación del agente MinimaxAlphaBeta
├── entrega.ipynb   # Notebook con experimentos y análisis
└── readme.md       # Este archivo
```

---

## Requisitos

```
Python >= 3.9
numpy
matplotlib
```

Instalar dependencias:
```bash
pip install numpy matplotlib
```

---

## Uso

El agente se usa a través de la clase `MinimaxAlphaBeta` definida en `policy.py`:

```python
from policy import MinimaxAlphaBeta

agente = MinimaxAlphaBeta(depth=6)   # depth: profundidad de búsqueda (default=6)
agente.mount()                        # inicialización (no-op en esta versión)
columna = agente.act(board)           # board: np.ndarray (6x7), retorna int (0–6)
```

### Parámetros configurables

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| `depth`   | int  | 6       | Profundidad de búsqueda del árbol Minimax |

---

## Ejecutar el torneo completo

Desde la carpeta `tournament/`:

```bash
cd tournament
python main.py
```

---

## Ejecutar el análisis

Abrir `entrega.ipynb` en Jupyter desde la carpeta `Group B/` o `tournament/`:

```bash
jupyter notebook entrega.ipynb
```

Ejecutar **Kernel → Restart & Run All**. Se generan 4 gráficas:
- `exp1_winrate_vs_random.png` — Win rate vs jugador aleatorio por profundidad y color
- `exp2_self_play.png` — Resultados del auto-juego
- `exp3_center_preference.png` — Impacto del bonus de columna central
- `exp4_time_vs_depth.png` — Costo computacional por profundidad

---

## Código fuente

Rama: [`group-b/policy-update`](https://github.com/Luis-Diaz-Cryo/AiProyectoLRS/blob/group-b/policy-update/tournament/groups/Group%20B/policy.py)
