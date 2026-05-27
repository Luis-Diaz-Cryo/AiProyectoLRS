# Agente Connect-4: Monte Carlo Tree Search con UCB1

**Fundamentos de Inteligencia Artificial — Universidad de La Sabana 2026.1**

---

## Idea principal

Este agente implementa **Monte Carlo Tree Search (MCTS)** con selección de nodos mediante la política **UCB1**. A diferencia de los agentes aleatorios o los que usan tablas de valores precalculadas, MCTS construye un árbol de búsqueda *durante la partida*, sin entrenamiento offline, y toma decisiones basadas en simulaciones de juego completas.

### ¿Qué lo distingue?

| Característica | Este agente | Agente aleatorio |
|---|---|---|
| Estrategia | MCTS + UCB1 | Uniforme |
| Planificación | Árbol de búsqueda online | Ninguna |
| Rollout | Semi-inteligente (gana/bloquea) | — |
| Configuración | `n_simulations`, `c_ucb` | — |

---

## Estructura del árbol MCTS

```
Repetir n_simulations veces:
  1. SELECCIÓN   → bajar por el árbol con UCB1 hasta nodo no expandido o terminal
  2. EXPANSIÓN   → añadir un hijo no visitado (acción aleatoria entre las no probadas)
  3. SIMULACIÓN  → rollout semi-inteligente hasta el final del juego
  4. RETROPROPAGACIÓN → actualizar visitas y victorias en todos los ancestros
Decisión final: columna del hijo con más visitas
```

**Fórmula UCB1:**

```
UCB1(v) = w_v/n_v + c * sqrt(ln(N) / n_v)
```

- `w_v / n_v`: tasa de victoria del nodo (explotación)
- `c * sqrt(ln(N) / n_v)`: bono de exploración para nodos poco visitados
- `c = √2` (valor por defecto, teóricamente óptimo para juegos de suma cero)

---

## Uso

### Requisitos

```bash
pip install numpy
```

### Colocar el archivo

Copia `policy.py` dentro de tu carpeta de grupo en `groups/<TuGrupo>/policy.py`.

### Parámetros configurables

```python
from policy import MCTSAgent

# Configuración por defecto (recomendada para el torneo)
agente = MCTSAgent(
    n_simulations=500,     # más simulaciones = mejor juego, pero más lento
    c_ucb=1.4142,          # constante de exploración UCB1 (√2)
    time_limit=None        # si se pasa un float (seg.), usa tiempo en lugar de conteo
)
```

### Ejecutar con límite de tiempo en lugar de conteo

```python
agente = MCTSAgent(time_limit=1.0)  # 1 segundo por turno
```

### Ejecutar el torneo

```bash
cd tournament/
python main.py
```

---

## Análisis rápido

| Escenario | Resultado típico (n=300) |
|---|---|
| MCTS (Rojo) vs Aleatorio | ~85–95% victorias |
| MCTS (Amarillo) vs Aleatorio | ~85–95% victorias |
| MCTS vs MCTS (self-play) | ~50% c/u (simétrico) |

El análisis completo con gráficas está en `entrega.ipynb`.

---

## Propuesta de mejoras (versión futura)

1. **Persistent tree:** reusar el árbol entre turnos para multiplicar simulaciones gratis.
2. **Función de evaluación heurística:** reemplazar el rollout estocástico por una evaluación rápida de la posición (reduce varianza).
3. **RAVE:** inicializar estadísticas de nodos nuevos con información global de la partida.

---

## Autor

Curso: Fundamentos de IA — Universidad de La Sabana, 2026.1
