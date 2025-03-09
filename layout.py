import pathlib

import io

from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.layout import Layout
from rich.console import Console
from rich.columns import Columns

from dao import BlockOfGames, db_name
from matching_algos.task_input_output import TaskOutput


def explanation_viz(
    task_output: TaskOutput,
    initial_game_block: BlockOfGames,
    print_viz=False,
    print_raw_to_terminal=False,
):
    kwargs = {} if print_viz else {"file": io.StringIO()}
    c = Console(record=True, width=270, height=75, **kwargs)

    layout = Layout()

    layout.split_row(
        Layout(name="left", size=35),
        Layout(name="middle"),
        Layout(name="right"),
    )

    from viz import print_table, viz_block_of_games

    layout["left"].update(
        Align.center(
            print_table(
                task_output.input.player_ids, width=35, include_rd=False, print=False
            ),
            vertical="middle",
        )
    )

    new_games = viz_block_of_games(block_of_games=initial_game_block, print=False)
    matchups = new_games[2]

    matchup_grid = Table.grid()
    matchup_grid.add_column()
    matchup_grid.add_row("\n")
    matchup_grid.add_row(Align.center(Panel("Neue Paarungen")))
    matchup_grid.add_row(matchups)

    layout["middle"].split_column(
        Layout(matchup_grid, name="matchups"),
        Layout(name="cf2"),
    )

    from explanations import generate_counterfactuals

    cfs = generate_counterfactuals(task_output, print_to_terminal=print_raw_to_terminal)

    layout["right"].update(Columns([cfs[0], "\n\n\n", cfs[1]]))

    layout["middle"]["cf2"].update(cfs[2])

    c.print(layout)

    from helper import get_timestamp

    c.save_html("explanations/latest.html", clear=False)
    filepath = pathlib.Path(
        f"explanations/{db_name}_{get_timestamp()}_explanation.html"
    )
    filepath.parent.mkdir(exist_ok=True)
    c.save_html(filepath, clear=False)


def save_table_as_html(player_ids: list[str]):
    from viz import print_table

    c = Console(record=True, width=270, height=67, file=io.StringIO())
    layout = Layout()
    layout.update(
        Align.center(
            print_table(player_ids, width=35, print=False),
            vertical="middle",
        )
    )

    c.print(layout)
    c.save_html("explanations/latest_table.html", clear=True)
