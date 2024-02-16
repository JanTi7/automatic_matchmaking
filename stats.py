from datetime import timedelta
from dao import get_all_games_in_timewindow
from viz import viz_single_game

def print_all_games_of(player_id, delta: timedelta):
    relevant_games = [g for g in get_all_games_in_timewindow(delta) if player_id in g.players()]
    relevant_games = sorted(relevant_games, key=lambda g: g.timestamp)

    for game in relevant_games:
        game.draw()


def parse_delta(delta_str: str):
    s = delta_str.strip()
    numbers = s[:-1]
    unit = s[-1]

    if unit == "d":
        return timedelta(days=(int(numbers)))
    if unit == "w":
        return timedelta(weeks=(int(numbers)))
    if unit == "m":
        return timedelta(days=31*(int(numbers)))
    if unit == "y":
        return timedelta(days=356*(int(numbers)))

