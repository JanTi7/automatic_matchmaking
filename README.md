# Automatic Matchmaking in two-versus-two sports

This repository contains the relevant code to test the matching algorithms oneself, and to run a tool employing the matching algorithm to continually match and update players ratings.

## Installation

### Windows Step by Step

To runs this code on windows follow these steps:

- Download the files to you local machine (either using git clone, or as a zip from the [releases section](https://github.com/sruettgers/automatic_matchmaking/releases)).
- Move the `cmd_with_venv.bat` file from the helper folder to the folder above.
- Make sure you have python 3.12 installed. It can be installed from [the microsoft app store](https://apps.microsoft.com/detail/9ncvdn91xzqp).
- Double-Click the `cmd_with_venv.bat` file. This will take some time (~3 mins) on the first run. Afterward it takes no additional time.
  - This creates a virtual environment, and downloads and installs the relevant packages. It does not install minizinc, which is not required to use the program.
- You should now have a shell-window with the correct virtual environment activated.
  - This same window will now always open when you double-click `cmd_with_venv.bat`. 
    


### General install

Clone the repo and create a venv either according to the `requirements.txt`, OR using poetry. 

#### Using Poetry (optional)
The requirements can be installed using [poetry](https://python-poetry.org/docs/) by running `poetry install --no-root`. Use `poetry shell` afterward to activate the environment.

#### Installing Minizinc (optional)

Some of the matching algorithms rely on [Minizinc](https://www.minizinc.org/), which can be installed using `sudo snap install minizinc --classic` on Ubuntu systems.
Installing Minizinc is optional, it is possible to run the other (and also optimal) algorithms without it. 

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