import time

from sacred import Experiment

from dao import add_new_player, use_database, PlayerPool, GameProposed, load_from_db
from helper.tinydb_hashfs import SuperTinyDbObserver
from matching_algos.bruteforce_matching import BruteforceMatcher
from matching_algos.bruteforce_matching_minizinc import BruteforceMatcherMinizinc
import logging

from matching_algos.helpers import calc_rdm_result

bruteforce_benchmark = Experiment("bruteforce_runtime")


# bruteforce_benchmark.observers.append(SuperTinyDbObserver("benchmark_bruteforce_runtime_results"))


@bruteforce_benchmark.config
def config():
    n_runs = 10


@bruteforce_benchmark.automain
def benchmark_main(_run, n_runs, n):
    logging.getLogger().setLevel(logging.WARNING)

    import secrets
    db_name = f"benchmark_runs/{time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())}_bruteforce_runtime_{secrets.token_hex(1)}.json"
    use_database(db_name, create_new=True, init_logging=False, print_ascii=False)

    pool = PlayerPool()

    for idx in range(n):
        pid = add_new_player(str(idx), "-", 1500, 125)
        pool.add_player(pid)

    runtimes = list()
    for run_i in range(n_runs):
        game_block, task_output = pool.start_next_round(pause_mode="random",
                                                        higher_rating_weight=2,
                                                        matching_algo=BruteforceMatcherMinizinc,
                                                        return_task_output=True)

        runtimes.append(task_output.cost_time)
        print(16 * " ", f"{task_output.cost_time:.2f}s")

        game_results = list()
        for g_idx, game_id in game_block.proposed.items():
            points_a, points_b = calc_rdm_result(1500, 1500)

            game_results.append((g_idx, points_a, points_b))  # todo: add warped timestamp

        for game_res in game_results:
            game_block.register_result(*game_res)

        game_block.commit()

    return runtimes
