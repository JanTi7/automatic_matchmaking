# Automatic Matchmaking in two-versus-two sports

This repository contains the relevant code to test the matching algorithms oneself, and to run a tool employing the matching algorithm to continually match and update players ratings.

## Installation

The requirements can be installed using [poetry](https://python-poetry.org/docs/) by running `poetry install --no-root`. Use `poetry shell` afterward to activate the environment.

Some of the matching algorithms rely on [Minizinc](https://www.minizinc.org/), which can be installed using `sudo snap install minizinc --classic` on Ubuntu systems.
Installing Minizinc is optional, it is possible to run the other algorithms without it. 

## Relevant Files

### `benchmark_matching_algos.py`

This file can be used to quantitatively benchmark multiple matching algorithms.

### `mock_run.py`

This file simulates a sports course setting by creating players with random skill levels and matching them for multiple rounds.

The resulting explanations for each round's matching can be viewed in `explanations/mock_runs/`.

### `terminal_interface.py`

This file starts the UI to use the algorithms in a sports course setting. 
New players can be added using the terminal interface, or multiple players can be imported at once using `python prep_db_from_excel.py inbox/example_init.xlsx`.


## Study Results

The real world matchings and game results from the four study sessions can be found in `databases/studX.json` where `X` identifies each study session.

The sessions `1` and `3` used a random matching approach, while sessions `2` and `4` used the ILP_Minizinc matcher.