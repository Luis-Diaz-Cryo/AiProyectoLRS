# LuisAgent - Connect-4 Agent

## Información general

Este directorio contiene la entrega individual del agente **LuisAgent** para el reto de Connect-4 de la asignatura Fundamentos de Inteligencia Artificial.

El agente implementa una política llamada **Tactical Expectimax with Risk Avoidance**. La idea principal es combinar reglas tácticas de corto plazo con una búsqueda probabilística de estados futuros.

La política del agente sigue este orden:

1. Si puede ganar inmediatamente, juega esa columna.
2. Si el oponente puede ganar en el siguiente turno, bloquea esa columna.
3. Evita movimientos peligrosos que le permitan al oponente ganar inmediatamente.
4. Si no hay una jugada forzada, usa **Expectimax** para escoger la acción con mejor valor esperado.

Esta solución es distinta a otros agentes del grupo porque no usa Minimax Alpha-Beta puro ni Monte Carlo Tree Search con UCB1. En cambio, modela al oponente como un jugador probabilístico, promediando sus posibles respuestas.

---

## Estructura de archivos

La carpeta contiene los siguientes archivos principales:

```text
Luis/
├── policy.py
├── entrega_luis_expectimax.ipynb
└── README.md
```

### `policy.py`

Contiene el código completo del agente. La clase principal es:

```python
class LuisAgent(Policy):
```

Esta clase hereda de `connect4.policy.Policy` y define el método principal:

```python
def act(self, s: np.ndarray) -> int:
```

El método `act()` recibe el estado actual del tablero y retorna una columna legal entre `0` y `6`.

### `entrega_luis_expectimax.ipynb`

Notebook usado para realizar el análisis experimental del agente. Incluye pruebas contra un jugador aleatorio, comparación por profundidad, análisis del costo computacional, comparación con/sin Risk Avoidance y auto-juego.

### Datos necesarios

El agente no requiere datos externos ni archivos de entrenamiento. Todo lo necesario para la ejecución está contenido en el código del agente y en el entorno del torneo.

---

## Requisitos

Para ejecutar el agente se necesita:

- Python 3.10 o superior
- NumPy
- Matplotlib
- Jupyter Notebook o Jupyter Lab
- Pydantic, requerido por la estructura del torneo
- Código base del torneo con la carpeta `connect4/`

Instalación recomendada de dependencias:

```bash
pip install numpy matplotlib jupyter pydantic
```

O, en Windows:

```bash
python -m pip install numpy matplotlib jupyter pydantic
```

---

## Cómo ejecutar el torneo

Desde la carpeta `tournament/`, ejecutar:

```bash
python main.py
```

Ejemplo de estructura esperada:

```text
tournament/
├── connect4/
├── groups/
├── Luis/
│   ├── policy.py
│   ├── entrega_luis_expectimax.ipynb
│   └── README.md
├── main.py
└── tournament.py
```

---

## Cómo ejecutar el notebook

El notebook debe abrirse desde la carpeta:

```text
tournament/Luis/
```

Luego ejecutar todas las celdas en orden.

Si se usa VS Code o Jupyter Lab:

1. Abrir `entrega_luis_expectimax.ipynb`.
2. Seleccionar el kernel de Python correcto.
3. Ejecutar `Run All`.

El notebook importa el agente desde:

```python
from policy import LuisAgent
```

Por eso `entrega_luis_expectimax.ipynb` y `policy.py` deben estar en la misma carpeta.

---

## Descripción técnica del agente

### Estado

El estado es una matriz de `6 x 7` que representa el tablero de Connect-4.

La codificación usada es:

```text
-1 = Rojo
 1 = Amarillo
 0 = Casilla vacía
```

### Acción

Una acción corresponde a elegir una columna válida entre `0` y `6`.

### Modelo interno

El agente usa un modelo interno del juego para simular consecuencias:

- `_drop()` simula colocar una ficha en una columna.
- `_winner()` detecta si un estado tiene ganador.
- `_valid_moves()` identifica columnas legales.
- `_current_player()` infiere el turno actual.

### Política

La política final es:

```text
Ganar → Bloquear → Evitar riesgo → Expectimax
```

### Evaluación heurística

Cuando la búsqueda llega al límite de profundidad, el agente evalúa el tablero con una función heurística que considera:

- Control del centro.
- Ventanas de cuatro posiciones.
- Posibles líneas de 2 y 3 fichas.
- Amenazas inmediatas propias.
- Amenazas inmediatas del oponente.

---

## Experimentos realizados

En el notebook se realizaron los siguientes experimentos:

1. **LuisAgent vs RandomPolicy como rojo**
2. **LuisAgent vs RandomPolicy como amarillo**
3. **Comparación de profundidad de Expectimax**
4. **Costo computacional según profundidad**
5. **Comparación con y sin Risk Avoidance**
6. **Auto-juego LuisAgent vs LuisAgent**

Los resultados principales fueron:

- Win rate de 100% contra RandomPolicy como rojo.
- Win rate de 100% contra RandomPolicy como amarillo.
- La profundidad 4 mantiene desempeño perfecto con menor costo que profundidad 5.
- El Risk Avoidance funciona como una capa defensiva, aunque contra RandomPolicy no generó diferencia visible en el win rate.
- En auto-juego, el jugador rojo ganó 100% de las partidas, lo que sugiere ventaja del primer jugador bajo una política determinística simétrica.

---

## Conclusiones

El agente cumple con los requisitos principales del reto porque siempre retorna acciones legales y supera el umbral de desempeño contra el jugador aleatorio en ambos colores.

La principal conclusión del análisis es que el desempeño contra RandomPolicy se mantiene perfecto incluso con distintas profundidades, pero el costo computacional crece de manera importante al aumentar la profundidad. Por esta razón, se selecciona profundidad 4 como una configuración balanceada entre rendimiento y tiempo de ejecución.

---

## Posibles mejoras futuras

Algunas mejoras posibles son:

- Implementar profundidad adaptativa según la fase del juego.
- Ajustar automáticamente los pesos de la heurística.
- Añadir una pequeña política de apertura para los primeros turnos.
- Incorporar aleatoriedad controlada para reducir el sesgo observado en auto-juego.
- Comparar el agente contra oponentes más fuertes además del RandomPolicy.

---

## Autor

Luis Jaime Díaz Salazar  
Agente: `LuisAgent`  
Política: Tactical Expectimax with Risk Avoidance
