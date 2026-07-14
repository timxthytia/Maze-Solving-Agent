# Maze-Solving-Agent

Capstone project for CS2109S (Introduction to AI and Machine Learning), NUS — **Grid Adventure: Variant 1**.

## Overview

Grid Adventure V1 is a 2D grid-based maze game. An agent must navigate a grid containing walls, boxes, lava, doors/keys, coins, gems, and power-ups (speed, shield, phasing), collect all gems, and reach the exit — while maximizing score (`-3` per turn, `+5` per coin collected) and avoiding running out of HP.

The task is to implement an `Agent` class that decides an `Action` each turn, given either:

- **Task 1 — `GridState`**: a fully structured symbolic representation of the grid, requiring search/planning over game mechanics (boxes, doors/keys, hazards, power-ups).
- **Task 2 — `ImageObservation`**: a rendered image of the grid (plus limited structured fields like health/inventory), requiring visual perception (tile classification) before planning.
- **Task 3 — Boss level**: combines both observation types with multiple mechanics and larger, more complex grids.

## Repository Contents

| File / Folder | Description |
|---|---|
| `capstone-project.ipynb` | Main assignment notebook: task descriptions, `Agent` interface contract, evaluation harness, and hints for hardcoding trained ML models into a submission. |
| `tutorial.ipynb` | Walkthrough of `GridState`, entities, actions, and scoring, with hands-on tutorial levels. |
| `CNN_final.ipynb` | Trains a custom CNN tile classifier (`customCNN`) on synthetic rendered levels, used to parse `ImageObservation` tiles into entity labels. |
| `agent_final.py` | Final submitted agent: search-based planner over a compact `GridState` search state, plus an embedded (base64/zlib) TorchScript CNN for image-based tile classification. |
| `maze_solver.py` | Standalone solver built directly on the `grid_universe`/`grid_adventure` library primitives (levels, factories, moves, objectives) with an embedded classification model. |
| `template.py` | Starter template mirroring `agent_final.py`'s structure, for building/testing agent variants. |
| `utils.py` | Helpers for evaluating an agent locally (mirroring the Coursemology evaluation process) and for generating self-contained `get_model()` loader snippets (scikit-learn / PyTorch, with optional zlib/gzip/bz2/lzma compression) so trained models can be pasted directly into a submission. |
| `data/assets/` | Tile images for each entity type (`boots`, `box`, `coin`, `exit`, `floor`, `gem`, `ghost`, `human`, `key`, `lava`, `locked`, `opened`, `shield`, `wall`) used for rendering and for training the tile classifier. |
| `capstone-project_files/` | Rendered images referenced by the main notebook. |
| `environment.yml` | Conda environment spec (Python 3.13, NumPy, pandas, scikit-learn, PyTorch/torchvision, plus the `grid-adventure` and `grid-play` packages). |

## Setup

```bash
conda env create -f environment.yml
conda activate cs2109s-ay2526s2-capstone-project
```

## Agent Interface

```python
class Agent:
    def __init__(self):
        ...  # optional model loading, within memory/time limits

    def step(self, state: GridState | ImageObservation) -> Action:
        ...  # returns a valid Action each turn
```

Use `utils.py` to evaluate an agent locally, and [Grid Play](https://grid-universe.github.io/grid-play/tutorial/game/#ai-agent-mode) to visually debug agent behavior.
