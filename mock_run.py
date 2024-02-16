import pathlib
import random
import logging
from rich import print as rprint
from rich.console import Console
from configargparse import ArgumentParser
from dao import use_database, load_from_db
from dao import PlayerPool, BlockOfGames, GameProposed
from dao import add_new_player

from matching_algos.helpers import calc_rdm_result

from parser import get_parser
parser = get_parser("databases/mock_run.conf")
parser.add_argument("--scenario", default="three-norm")
parser.add_argument("-n", "--n-rounds", default=4, type=int)
parser.add_argument("--pretty", action="store_true")

args = parser.parse_args()

assert args.db is None

parser.print_values()

console = Console(record=True)

import secrets
db_name = f"mock_runs/mock_run_{secrets.token_hex(4)}.json"
use_database(db_name, create_new=True)

logging.info(parser.format_values())

console.print(f"Using {db_name}")

player_elo_list = list()
if args.scenario == "three-norm":
    elo_centres = [1300, 1500, 1700]
    elo_var = 150

    num_ppl_per_pool = [5, 6, 5]

    from mock_run_names import name_pool

    name_pool = list(name_pool)
    random.shuffle(name_pool)

    for mu_elo, n_ppl in zip(elo_centres, num_ppl_per_pool):
        for _ in range(n_ppl):
            elo = random.gauss(mu_elo, elo_var)
            elo = int(elo//10)*10

            name = name_pool.pop()
            if not args.pretty:
                name = f"{name}_{elo}"
            player_elo_list.append((name, elo))


player_elo_list.sort(key=lambda p: p[1], reverse=True)
console.print(player_elo_list)
random.shuffle(player_elo_list)  # important!

pid2elo = dict()

pool = PlayerPool()
for player_name, player_elo in player_elo_list:
    pid = add_new_player(player_name, "-", player_elo + (random.random()-0.5) * 100, args.init_rd)
    pool.add_player(pid)
    pid2elo[pid] = player_elo



for round_idx in range(args.n_rounds):
    console.rule(f"ROUND {round_idx+1}")

    pool.draw()

    game_block, task_output = pool.start_next_round(pause_mode=args.pause_mode, num_sets=5,
                                                   higher_rating_weight=args.higher_rating_weight,
                                                   matching_algo=args.matching_algo,
                                                    return_task_output=True)

    from layout import explanation_viz, save_table_as_html
    explanation_viz(task_output, game_block)

    game_block.draw()

    game_results = list()
    for g_idx, game_id in game_block.proposed.items():
        game = GameProposed(**load_from_db(game_id))

        elo_team_a, elo_team_b = (pid2elo[game.players().a1] + pid2elo[game.players().a2]) / 2, \
                                 (pid2elo[game.players().b1] + pid2elo[game.players().b2]) / 2
        points_a, points_b = calc_rdm_result(elo_team_a, elo_team_b)

        game_results.append((g_idx, points_a, points_b))

    for game_res in game_results:
        game_block.register_result(*game_res)

    game_block.draw()
    game_block.commit()

    save_table_as_html(pool.everybody())


from viz import print_full_table
print_full_table()

