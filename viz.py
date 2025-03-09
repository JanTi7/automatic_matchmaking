import rich
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.align import Align
from rich.text import Text
from rich import box

from dao import (
    get_player_from_id,
    GameProposed,
    GameResult,
    load_from_db,
    generate_playerid_to_uniquename_map,
)


def pid2panel(pid, id2name, rating=None, emph=False):
    if rating is None:
        player = get_player_from_id(pid)
        rating = int(player.get_current_rating())

    if not emph:
        return Panel(id2name[pid], subtitle=str(rating))

    return Panel(
        Text.from_markup(id2name[pid]),
        style="on #ff7a7a",
        subtitle=str(rating),
        box=box.DOUBLE,
    )


def default_table(title, sparse=False):
    table = Table(
        title=title, title_justify="right", show_edge=False, show_header=not sparse
    )  # , style="on black"

    if not sparse:
        table.add_column("IDX", width=5, justify="center")
    table.add_column("", width=18)
    table.add_column("", width=18)
    if sparse:
        table.add_column(" ", width=2)
    else:
        table.add_column("PP", width=6, justify="center")
        table.add_column("PP", width=6, justify="center")
    table.add_column("", width=18)
    table.add_column("", width=18)

    return table


def viz_block_of_games(block_of_games, print=True) -> list[rich.table]:
    console = Console()

    tables = list()

    id2name = generate_playerid_to_uniquename_map(
        list_of_pids=list(), bold_first_name=True
    )  # this assumes all players

    ### GAMES CANCELLED ###
    table = default_table("CANCELLED")
    for local_idx, game in [
        (idx, GameProposed(**load_from_db(gpp_id)))
        for idx, gpp_id in block_of_games.cancelled.items()
    ]:
        panels = list()
        for rsnap_id in (
            game.rsnap_a_1,
            game.rsnap_a_2,
            game.rsnap_b_1,
            game.rsnap_b_2,
        ):
            rs = load_from_db(rsnap_id)
            player = get_player_from_id(rs["player_id"])
            rating = int(rs["rating"])

            # but how still use fat for first name?
            # regex or some weird split?
            panels.append(Panel(id2name[player.player_id], subtitle=str(rating)))

        panels.insert(2, "")
        panels.insert(2, "")
        panels.insert(0, Panel(str(local_idx)))

        table.add_row(*panels)

    tables.append(table)

    ### GAMES ALREADY PLAYED ###
    table = default_table("FINISHED")
    for local_idx, game in [
        (idx, GameResult(**load_from_db(gres_id)))
        for idx, gres_id in block_of_games.results.items()
    ]:
        # ratings_pre = game.ratings_pre_result()
        ratings_post = game.ratings_post_result(log=False)

        panels = list()
        for idx, rsnap_id in enumerate(
            (game.rsnap_a_1, game.rsnap_a_2, game.rsnap_b_1, game.rsnap_b_2)
        ):
            rs = load_from_db(rsnap_id)
            player = get_player_from_id(rs["player_id"])
            rating_pre = int(rs["rating"])

            # rating_pre = ratings_pre[idx]
            rating_post = ratings_post[idx]
            delta = rating_post.rating - rating_pre

            panels.append(
                Panel(
                    id2name[player.player_id],
                    subtitle=str(int(round(rating_post.rating, 0))),
                    title=str(f"{rating_pre} {int(round(delta, 0)):+g}"),
                )
            )

        panels.insert(2, Panel(str(game.points_b).rjust(2)))
        panels.insert(2, Panel(str(game.points_a).rjust(2)))
        panels.insert(0, Panel(str(local_idx)))

        table.add_row(*panels)

    tables.append(table)

    ### GAMES STILL TO BE PLAYED ###
    table = default_table("TO PLAY")
    for local_idx, game in [
        (idx, GameProposed(**load_from_db(gpp_id)))
        for idx, gpp_id in block_of_games.proposed.items()
    ]:
        panels = list()
        for rsnap_id in (
            game.rsnap_a_1,
            game.rsnap_a_2,
            game.rsnap_b_1,
            game.rsnap_b_2,
        ):
            rs = load_from_db(rsnap_id)
            player = get_player_from_id(rs["player_id"])
            rating = int(rs["rating"])

            panels.append(Panel(id2name[player.player_id], subtitle=str(rating)))

        panels.insert(2, "")
        panels.insert(2, "")
        panels.insert(0, Panel(str(local_idx)))

        table.add_row(*panels)

    tables.append(table)

    if print:
        for table in tables:
            if table.row_count > 0:
                console.print(table)

    return tables


def get_human_str_for_gamedict(gamedict: dict, idx):
    id2name = generate_playerid_to_uniquename_map(
        list_of_pids=list(), bold_first_name=False
    )
    rating_snap_ids = [
        gamedict[rsid] for rsid in ("rsnap_a_1", "rsnap_a_2", "rsnap_b_1", "rsnap_b_2")
    ]
    names = [
        id2name[load_from_db(rsnap_id)["player_id"]] for rsnap_id in rating_snap_ids
    ]
    return "{0} & {1} v. {2} & {3} <{4}>".format(*names, idx)


def viz_single_game(game, local_idx="", sparse=False, print=True) -> rich.table:
    console = Console()
    # console.print(game)
    table = default_table(None, sparse=sparse)

    id2name = generate_playerid_to_uniquename_map(
        list_of_pids=list(), bold_first_name=True
    )

    panels = list()
    if not sparse and hasattr(game, "points_a"):
        ratings_pre = game.ratings_pre_result()
        ratings_post = game.ratings_post_result()

        panels = list()
        for idx, rsnap_id in enumerate(
            (game.rsnap_a_1, game.rsnap_a_2, game.rsnap_b_1, game.rsnap_b_2)
        ):
            rs = load_from_db(rsnap_id)
            player = get_player_from_id(rs["player_id"])

            rating_pre = ratings_pre[idx]
            rating_post = ratings_post[idx]
            delta = rating_post.rating - rating_pre.rating

            panels.append(
                Panel(
                    id2name[player.player_id],
                    subtitle=str(int(round(rating_post.rating, 0))),
                    title=str(f"{rating_pre.rating} {int(round(delta, 0)):+g}"),
                )
            )

        panels.insert(2, Panel(str(game.points_b).rjust(2)))
        panels.insert(2, Panel(str(game.points_a).rjust(2)))
        panels.insert(0, Panel(str(local_idx)))

    else:
        for idx, rsnap_id in enumerate(
            (game.rsnap_a_1, game.rsnap_a_2, game.rsnap_b_1, game.rsnap_b_2)
        ):
            rs = load_from_db(rsnap_id)
            player = get_player_from_id(rs["player_id"])
            rating = player.get_current_rating()
            panels.append(Panel(id2name[player.player_id], subtitle=str(rating)))

        if not sparse:
            panels.insert(2, "")
            panels.insert(2, "")
            panels.insert(0, Panel(str(local_idx)))
        else:
            panels.insert(2, Align.center("vs", vertical="middle"))

    table.add_row(*panels)

    if print:
        console.print(table)

    return table


def viz_players_to_pause(sorted_list: list, num_to_pause):
    console = Console()
    table = Table(title="Wer muss Pause machen?")
    table.add_column("Name")
    table.add_column("#played", justify="right")
    table.add_column("#paused", justify="right")
    table.add_column("played per paused", justify="right")
    table.add_column("tie decider", justify="right")

    id2name = generate_playerid_to_uniquename_map(
        [p.player_id for p, _, _ in sorted_list]
    )

    for idx, (player, quot, tie_factor) in enumerate(sorted_list):
        if idx == num_to_pause:
            table.add_row(*["-"] * 5)

        table.add_row(
            id2name[player.player_id],
            str(player.games_played),
            str(player.games_paused),
            f"{1/quot:.2f}",
            f"{tie_factor:.0f}",
        )

    console.print(table)


def print_full_table():
    from dao import get_all_players

    print_table([p.player_id for p in get_all_players()])


def print_table(list_of_pids, include_rd=True, width=None, print=True, include_details=True, colored_ratingchange=True) -> rich.table:
    from dao import generate_playerid_to_uniquename_map

    id2name = generate_playerid_to_uniquename_map(list_of_pids)

    players = [get_player_from_id(pid) for pid in list_of_pids]
    players.sort(
        key=lambda p: (
            p.get_current_rating(),
            -p.get_rating_snapshot().rd,
            -ord(p.first_name[0]),
            -ord(p.first_name[1]),
        ),
        reverse=True,
    )

    table = Table(title="Standings", width=width)
    table.add_column("#", justify="right")
    table.add_column("Name")
    if include_rd:
        table.add_column("RD", justify="right")
    table.add_column("Rating", justify="right")
    if include_details:
        table.add_column("Games Played", justify="right")
        table.add_column("Games Won", justify="right")
        table.add_column("Total Rating Change", justify="right")
        table.add_column("Win Percentage", justify="right")

    for idx, player in enumerate(players):
        full_rating = player.get_rating_snapshot()

        if idx == 0 or players[idx - 1].get_current_rating() > full_rating.rating:
            place_str = str(idx + 1)
        else:
            place_str = ""

        data = [place_str, id2name[player.player_id], str(int(full_rating.rating))]

        if include_details:
            win_percentage = player.games_won /  player.games_played 
            total_rating_change = player.total_rating_change
            if colored_ratingchange:
                total_rating_change_str = f"[green]{total_rating_change:+d}[/green]" if total_rating_change > 0 else f"[red]{total_rating_change:+d}[/red]"
            else:
                total_rating_change_str = f"{total_rating_change:+d}"
            data.extend([str(player.games_played), str(player.games_won), total_rating_change_str, f"{win_percentage:.2%}"])
        
        if include_rd:
            data.insert(2, str(int(full_rating.rd)))

        table.add_row(*data)

    if print:
        console = Console()
        console.print(table)

    return table
