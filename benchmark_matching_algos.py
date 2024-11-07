import logging
import math
import time
import datetime
from collections import defaultdict
from random import shuffle, Random

import numpy as np

from sacred import Experiment, Ingredient
from helper.tinydb_hashfs import SuperTinyDbObserver, SuperTinyDbReader

from dao import add_new_player, use_database, PlayerPool, GameProposed, load_from_db
from find_matchups import TaskInput
from matching_algos.base_matching_algo import MatchupCostCalculator
from matching_algos.bruteforce_matching import BruteforceMatcher
from matching_algos.bruteforce_matching_minizinc import BruteforceMatcherMinizinc
from matching_algos.double_symmetric_matching import DoubleSymmetricMatcher
from matching_algos.fixed_order_matching import FixedOrderMatcher
from matching_algos.minizinc_matching import MinizincMatcher
from matching_algos.random_matching import RandomMatcher
from matching_algos.helpers import calc_rdm_result
from matching_algos.reliable_matching import ReliableMatcher


pool_generation = Ingredient("pool_gen")


@pool_generation.config
def pool_gen_config():
    mean_rating = 1500
    std_rating = 265

    mean_attendance_rate = 0.66
    # std_attendance_rate = 0.2

    # mean_number_of_players_per_date = 23
    number_of_players_per_date = 24
    poolsize = math.ceil(number_of_players_per_date / mean_attendance_rate)


@pool_generation.capture
def generate_pool(
    _rnd: np.random.Generator,
    mean_rating,
    std_rating,
    poolsize,
):
    from mock_run_names import name_pool

    name_pool = list(name_pool)
    _rnd.shuffle(name_pool)

    name_pool += ["alpha_" + n for n in name_pool]

    res = list()
    for _ in range(poolsize):
        elo = int(_rnd.normal(mean_rating, std_rating) // 10) * 10
        name = f"{name_pool.pop()}_{elo}"
        res.append((name, elo))

    return res


matching_benchmark = Experiment("matching_benchmark", ingredients=[pool_generation])
matching_benchmark.observers.append(SuperTinyDbObserver("benchmark_matching_results"))


@matching_benchmark.config
def matching_benchmark_config():
    load_last_res_dict_flag = False

    n_runs = 10
    n_rounds = 1
    games_per_round = 4

    pause_mode = "random"

    higher_rating_weight = 2.0

    initial_rating_estimate = True


@pool_generation.capture
def draw_from_pool(seed, pool, number_of_players_per_date):
    p = pool[:]
    Random(4).shuffle(p)
    return p[:number_of_players_per_date]


@matching_benchmark.command
def load_last_res_dict():
    folder = matching_benchmark.observers[0].root
    reader = SuperTinyDbReader(folder)
    for i in range(1, 64):
        try:
            info_dict = reader.fetch_metadata(indices=[-i])[0]["info"]
            return info_dict["results"]
        except:
            pass

    return dict()


@matching_benchmark.named_config
def large_run():
    load_last_res_dict_flag = False
    n_runs = 10
    # n_rounds = 6


@matching_benchmark.automain
def benchmark_main(
    _run,
    _rnd,
    n_runs,
    n_rounds,
    games_per_round,
    pause_mode,
    higher_rating_weight,
    initial_rating_estimate,
    load_last_res_dict_flag,
    solve_same_tasks=True,
):
    _run.info["results"] = dict()
    _run.info["results_detailed"] = dict()
    res_dict = _run.info["results"]
    res_dict_detailed = _run.info["results_detailed"]

    if load_last_res_dict_flag:
        res_dict.update(load_last_res_dict())

    matching_algos = dict(
        DoubleSymmetric=lambda: DoubleSymmetricMatcher(exponent=2),
        Minizinc=lambda: MinizincMatcher(
            search_duration=9.3,
            viz_weight_matrices=False,
            verbose=False,
            log_tasks=False,
        ),
        Random=lambda: RandomMatcher(tries=1),
        FixedOrder=FixedOrderMatcher,
        ILP_scipy=lambda: BruteforceMatcher(presolve=False, mip_rel_gap=0.0),
        ILP_Minizinc=lambda: BruteforceMatcherMinizinc(timeout=60),
    )

    for key in matching_algos.keys():
        if key in res_dict.keys():
            del res_dict[key]

    logging.getLogger().setLevel(logging.WARNING)

    player_pool_per_run = [generate_pool() for _ in range(n_runs)]

    data = defaultdict(list)
    detailed_data = defaultdict(list)

    if solve_same_tasks:
        algo_matcher_dicts = [matching_algos]
    else:
        # split the matching algos into seperate entries
        algo_matcher_dicts = [{k: v} for k, v in matching_algos.items()]

    for algo_matcher_dict in algo_matcher_dicts:
        for run_i in range(n_runs):
            playerpool = player_pool_per_run[run_i]
            names = list()
            name2rating = dict()

            for name, rating in playerpool:
                names.append(name)
                name2rating[name] = rating

            pids_for_each_round = [
                draw_from_pool(run_i, names) for _ in range(n_rounds)
            ]

            import secrets

            db_name = f"benchmark_runs/{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_{'_'.join(algo_matcher_dict.keys())}_{secrets.token_hex(1)}.json"
            use_database(
                db_name, create_new=True
            )  # can i parallelize with this global var? -> using processes?

            name2pid = dict()
            for name in names:
                start_rating = 1500
                if initial_rating_estimate:
                    start_rating = name2rating[name] + (_rnd.random() - 0.5) * 150
                pid = add_new_player(name, "-", start_rating, 125)
                name2pid[name] = pid

            pid2elo = {name2pid[name]: name2rating[name] for name in names}
            for round_i in range(n_rounds):
                subpool_names = pids_for_each_round[round_i]
                pool = PlayerPool()
                for name in subpool_names:
                    pool.add_player(name2pid[name])

                for game_i in range(games_per_round):
                    game_blocks = list()
                    for algo_name, Matcher in algo_matcher_dict.items():
                        game_block, task_output = pool.start_next_round(
                            pause_mode,
                            higher_rating_weight=higher_rating_weight,
                            matching_algo=Matcher,
                            return_task_output=True,
                        )

                        if solve_same_tasks:
                            print(
                                round_i,
                                game_i,
                                f"{task_output.cost_quad.total:.2f}".rjust(9),
                                algo_name.ljust(32),
                                f"{task_output.cost_time:.2f}s".rjust(7),
                                task_output.cost_quad,
                            )
                        else:
                            print(
                                algo_name,
                                round_i,
                                game_i,
                                task_output.cost_quad,
                                f"{task_output.cost_time:.2f}s",
                            )

                        data[algo_name].append(
                            dict(
                                idx=round_i * games_per_round + game_i,
                                runtime=task_output.cost_time,
                                **task_output.cost._asdict(),
                                **{
                                    f"quad_{key}": val
                                    for key, val in task_output.cost_quad._asdict().items()
                                },
                            )
                        )

                        detailed_data[algo_name].append(
                            dict(
                                idx=round_i * games_per_round + game_i,
                                ratings=task_output.input.rating_list,
                                matchups=task_output.matchups_as_idx(),
                                personal_costs=MatchupCostCalculator.from_taskinput(
                                    task_output.input
                                ).cost_per_person(task_output.matchups_as_idx()),
                            )
                        )

                        game_blocks.append((task_output.cost_quad, game_block))

                    if solve_same_tasks:
                        print()

                    game_block = sorted(game_blocks, key=lambda t: t[0])[0][1]

                    game_results = list()
                    for g_idx, game_id in game_block.proposed.items():
                        game = GameProposed(**load_from_db(game_id))

                        elo_team_a, elo_team_b = (
                            (pid2elo[game.players().a1] + pid2elo[game.players().a2])
                            / 2,
                            (pid2elo[game.players().b1] + pid2elo[game.players().b2])
                            / 2,
                        )
                        points_a, points_b = calc_rdm_result(elo_team_a, elo_team_b)

                        game_results.append((g_idx, points_a, points_b))

                    for game_res in game_results:
                        game_block.register_result(*game_res)

                    game_block.commit()

    for algo_name in matching_algos.keys():
        res_dict[algo_name] = data[algo_name]
        res_dict_detailed[algo_name] = detailed_data[algo_name]
