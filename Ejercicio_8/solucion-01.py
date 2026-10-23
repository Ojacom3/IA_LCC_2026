#!/usr/bin/env python3
"""
Ejercicio 1 — Q-learning tabular en Cliff Walking (Sutton & Barto, 4x12).

Entorno del libro:
  - Cuadrícula 4 filas x 12 columnas. Fila 0 = arriba.
  - S = (3,0), G = (3,11), acantilado = (3,1)..(3,10).
  - Acciones N,E,S,W. N resta fila, E suma columna.
  - Paso normal / borde: r = -1, te quedas si chocas.
  - Caer al acantilado: r = -100, s' = S, el episodio CONTINÚA.
  - Entrar a G: r = -1, el episodio TERMINA.
  - gamma = 1.

Q-learning off-policy:
  Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
  con max = 0 si s' es terminal (G). Si s' = S tras caer, S es normal.

Parámetros de la corrida reportada:
  alpha=0.5, gamma=1, epsilon=0.1, episodios=500, max_steps=200,
  Q inicial = 0, semilla = 8, desempate fijo N,E,S,W (primero gana).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Entorno 4x12 del libro
# ---------------------------------------------------------------------------

ROWS = 4
COLS = 12
START = (3, 0)
GOAL = (3, 11)
CLIFF = {(3, c) for c in range(1, 11)}  # (3,1)..(3,10)
ACTIONS = ("N", "E", "S", "W")          # orden fijo para desempate
DELTA = {"N": (-1, 0), "E": (0, 1), "S": (1, 0), "W": (0, -1)}

STEP_REWARD = -1.0
CLIFF_REWARD = -100.0
GAMMA = 1.0


@dataclass(frozen=True)
class Transition:
    nxt: tuple[int, int]
    reward: float
    done: bool


def step(state: tuple[int, int], action: str) -> Transition:
    """Transición del Cliff Walking del libro."""
    dr, dc = DELTA[action]
    r, c = state
    nr, nc = r + dr, c + dc
    # Borde: te quedas en la misma casilla, r = -1.
    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        nr, nc = r, c
    nxt = (nr, nc)
    if nxt == GOAL:
        return Transition(nxt, STEP_REWARD, True)
    if nxt in CLIFF:
        # Caer: -100, vuelves a S, el episodio SIGUE.
        return Transition(START, CLIFF_REWARD, False)
    return Transition(nxt, STEP_REWARD, False)


def nonterminal_states() -> list[tuple[int, int]]:
    """Todos los estados salvo G (S y el acantilado son estados de la tabla:
    el acantilado nunca se 'está' en él porque caer teletransporta a S, pero
    S sí es no terminal y debe tener fila en Q)."""
    return [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) != GOAL]


# ---------------------------------------------------------------------------
# Q-learning (mismas funciones que qlearn/agent.py, adaptadas)
# ---------------------------------------------------------------------------

def empty_q() -> dict[tuple[int, int], dict[str, float]]:
    return {s: {a: 0.0 for a in ACTIONS} for s in nonterminal_states()}


def max_q(Q, state) -> float:
    if state not in Q:  # G no está en Q → valor futuro 0
        return 0.0
    return max(Q[state][a] for a in ACTIONS)


def argmax_action(Q, state) -> str:
    """Primero en ACTIONS gana empates (N, luego E, S, W)."""
    best = ACTIONS[0]
    best_v = Q[state][best]
    for a in ACTIONS[1:]:
        if Q[state][a] > best_v:
            best, best_v = a, Q[state][a]
    return best


def epsilon_greedy(Q, state, epsilon: float, rng: random.Random) -> str:
    if rng.random() < epsilon:
        return rng.choice(ACTIONS)
    return argmax_action(Q, state)


def td_target_q(Q, reward: float, nxt, done: bool) -> float:
    future = 0.0 if done else max_q(Q, nxt)
    return reward + GAMMA * future


def apply_update(Q, state, action: str, target: float, alpha: float) -> None:
    old = Q[state][action]
    Q[state][action] = old + alpha * (target - old)


def train(alpha: float, epsilon: float, episodes: int, max_steps: int,
          seed: int) -> tuple[dict, list[float]]:
    rng = random.Random(seed)
    Q = empty_q()
    returns: list[float] = []
    for _ep in range(episodes):
        s = START
        G = 0.0
        for _step in range(max_steps):
            a = epsilon_greedy(Q, s, epsilon, rng)
            tr = step(s, a)
            G += tr.reward
            target = td_target_q(Q, tr.reward, tr.nxt, tr.done)
            apply_update(Q, s, a, target, alpha)
            s = tr.nxt
            if tr.done:
                break
        returns.append(G)
    return Q, returns


# ---------------------------------------------------------------------------
# Evaluación greedy (epsilon = 0)
# ---------------------------------------------------------------------------

def greedy_rollout(Q, max_steps: int = 200) -> tuple[list[tuple[int, int]], float]:
    s = START
    path = [s]
    G = 0.0
    for _ in range(max_steps):
        a = argmax_action(Q, s)
        tr = step(s, a)
        G += tr.reward
        s = tr.nxt
        path.append(s)
        if tr.done:
            break
    return path, G


# ---------------------------------------------------------------------------
# Formato de salida (alineado a 03_train.py)
# ---------------------------------------------------------------------------

ARROW = {"N": "^", "E": ">", "S": "v", "W": "<"}


def format_q(Q) -> str:
    lines = []
    for s in nonterminal_states():
        vals = "  ".join(f"{a}={Q[s][a]:7.2f}" for a in ACTIONS)
        tag = "S" if s == START else ("C" if s in CLIFF else ".")
        lines.append(f"  ({s[0]},{s[1]:2d}) [{tag}]  {vals}")
    return "\n".join(lines)


def format_grid_policy(Q) -> str:
    lines = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            s = (r, c)
            if s == START:
                row.append("S")
            elif s == GOAL:
                row.append("G")
            elif s in CLIFF:
                row.append("C")
            else:
                row.append(ARROW[argmax_action(Q, s)])
        lines.append("  " + "  ".join(row))
    return "\n".join(lines)


def format_returns(returns: list[float], window: int = 10) -> str:
    lines = ["  ep   return     moving_avg(10)"]
    for i, r in enumerate(returns, start=1):
        lo = max(0, i - window)
        ma = sum(returns[lo:i]) / (i - lo)
        lines.append(f"  {i:3d}  {r:9.2f}  {ma:9.2f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    alpha = 0.5
    epsilon = 0.1
    episodes = 500
    max_steps = 200
    seed = 8

    print("Entorno: Cliff Walking 4x12 (Sutton & Barto)")
    print("-" * 72)
    for r in range(ROWS):
        cells = []
        for c in range(COLS):
            s = (r, c)
            if s == START:
                cells.append("S")
            elif s == GOAL:
                cells.append("G")
            elif s in CLIFF:
                cells.append("C")
            else:
                cells.append(".")
        print("  " + "  ".join(cells))
    print()
    print(f"alpha={alpha}  gamma={GAMMA}  epsilon={epsilon}  "
          f"episodes={episodes}  max_steps={max_steps}  seed={seed}")
    print()

    Q, returns = train(alpha, epsilon, episodes, max_steps, seed)

    print("Q")
    print("-" * 72)
    print(format_q(Q))
    print()

    print("Greedy policy (arrows)")
    print("-" * 72)
    print(format_grid_policy(Q))
    print()

    path, gret = greedy_rollout(Q, max_steps)
    print(f"Greedy path from S (steps={len(path) - 1}):")
    print("  " + " -> ".join(f"({r},{c})" for r, c in path))
    print()
    print(f"greedy return from start (epsilon = 0): {gret:g}")
    print()

    print("Returns per episode (moving average window = 10)")
    print("-" * 72)
    print(format_returns(returns))

    print()
    plot_returns(returns, window=10, seed=seed)

    # ---------------------------------------------------------------------------
    # Gráfica opcional (solo si matplotlib está instalado)
    # ---------------------------------------------------------------------------

def plot_returns(returns: list[float], window: int = 10, seed: int = 8) -> None:
    """Curva de retorno por episodio + media móvil. No-op si falta matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # backend sin ventana; escribe PNG
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib no está instalado; se omite la gráfica.")
        return

    eps = list(range(1, len(returns) + 1))
    ma = []
    for i in range(1, len(returns) + 1):
        lo = max(0, i - window)
        ma.append(sum(returns[lo:i]) / (i - lo))

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(eps, returns, color="tab:blue", alpha=0.35, linewidth=0.8,
            label="retorno por episodio")
    ax.plot(eps, ma, color="tab:orange", linewidth=2.0,
            label=f"media móvil ({window})")
    ax.axhline(-13, color="tab:green", linestyle="--", linewidth=1.2,
                label="óptimo greedy (−13)")
    ax.set_xlabel("episodio")
    ax.set_ylabel("retorno")
    ax.set_title(f"Q-learning en Cliff Walking 4×12  (α=0.5, γ=1, ε=0.1, seed={seed})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = f"cliff_returns_seed{seed}.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"[plot] gráfica guardada en {out}")

if __name__ == "__main__":
    main()