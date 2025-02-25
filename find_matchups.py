import random
import time

import numpy as np
import asyncio
import logging
from minizinc import Instance, Model, Solver
from datetime import timedelta

from dao import Participants, get_participants_from_last_matches, Player, load_from_db

from matching_algos.task_input_output import TaskInput, TaskOutput

from collections import namedtuple

MinizincResult = namedtuple("MinizincResult", "matchup_table matchuo_order cost")


class WeightMatrixManager:
    def __init__(
        self,
        list_of_players: list[Player],
        decay_fac=0.5,
        WEIGHT_TEAM=64,
        WEIGHT_GAME=16,
    ):
        n_players = len(list_of_players)
        list_of_players.sort(key=lambda p: p.get_current_rating(), reverse=True)
        self.list_of_pids = [p.player_id for p in list_of_players]
        self.rating_snap_ids = [p.rating_snapshot for p in list_of_players]
        self.elo_list = [
            load_from_db(rsnap_id)["rating"] for rsnap_id in self.rating_snap_ids
        ]
        self.decay_fac = decay_fac
        self.WEIGHT_TEAM = WEIGHT_TEAM
        self.WEIGHT_GAME = WEIGHT_GAME

        self.cost_team = np.zeros(shape=(n_players, n_players))
        self.cost_same_game = np.zeros(shape=(n_players, n_players))

        self._calc_weights()

    def get_rating_snap_id_from_pid(self, player_id):
        return self.rating_snap_ids[self.list_of_pids.index(player_id)]

    def _calc_weights(self, verbose=False):
        previous_constellations = sorted(
            get_participants_from_last_matches(timedelta(days=9)),
        )
        recents = [
            p
            for timestamp, p in previous_constellations
            if time.time() - timestamp <= timedelta(days=1).total_seconds()
        ]
        older = [
            p
            for timestamp, p in previous_constellations
            if time.time() - timestamp > timedelta(days=1).total_seconds()
        ]
        for games in older:
            if verbose:
                print("OLDER", games)

            self._decay()
            for participants in games:
                self._register_game_played(participants)

        self._decay()
        self._decay()

        for games in recents:
            if verbose:
                print("RECENT", games)

            self._decay()
            for participants in games:
                self._register_game_played(participants)

    def _register_game_played(self, participants: Participants):
        self._add_team(participants.a1, participants.a2)
        self._add_team(participants.b1, participants.b2)

        self._played_together(participants.a1, participants.b1)
        self._played_together(participants.a1, participants.b2)
        self._played_together(participants.a2, participants.b1)
        self._played_together(participants.a2, participants.b2)

    def _add_team(self, p1, p2):
        if p1 not in self.list_of_pids or p2 not in self.list_of_pids:
            return

        i1 = self.list_of_pids.index(p1)
        i2 = self.list_of_pids.index(p2)

        # print("_add_team", p1, p2, i1, i2)

        self.cost_team[i1, i2] += self.WEIGHT_TEAM
        self.cost_team[i2, i1] += self.WEIGHT_TEAM

        self.cost_same_game[i1, i2] += self.WEIGHT_GAME
        self.cost_same_game[i2, i1] += self.WEIGHT_GAME

    def _played_together(self, p1, p2):
        if p1 not in self.list_of_pids or p2 not in self.list_of_pids:
            return

        i1 = self.list_of_pids.index(p1)
        i2 = self.list_of_pids.index(p2)

        self.cost_same_game[i1, i2] += self.WEIGHT_GAME
        self.cost_same_game[i2, i1] += self.WEIGHT_GAME

        self.cost_team[i1, i2] += self.WEIGHT_GAME
        self.cost_team[i2, i1] += self.WEIGHT_GAME

    def _decay(self):
        self.cost_same_game *= self.decay_fac
        self.cost_team *= self.decay_fac

    def get_task_input(self, higher_rating_weight):
        return TaskInput(
            player_ids=self.list_of_pids,
            rating_list=self.elo_list,
            weight_team=self.cost_team.astype(int).tolist(),
            weight_game=self.cost_same_game.astype(int).tolist(),
            higher_rating_weight=higher_rating_weight,
        )


def select_players_to_pause(
    list_of_players: list[Player], num_to_pause, mode="random", verbose=True
):
    """
    :param list_of_players:
    :param num_to_pause:
    :param mode:
    :param verbose:
    :return: Two arrays. First contains the players that will play, second list the ones which will pause.
    """
    if num_to_pause <= 0:
        return list_of_players, list()

    def games_paused_per_games_played(player):
        if player.games_played == 0:
            # favors players with more games paused if multiple with 0 played
            return 1000 + player.games_paused
        elif player.games_paused == 0:
            # favors players with fewer games when multiple with 0 paused
            return 1 / (player.games_played + 5)
        else:
            return player.games_paused / player.games_played

    # functions that decide the tie if multiple people have same amount of games paused per played
    # low values mean higher pause chance
    decide_ties_random = lambda p: random.random() * 100
    decide_ties_lowest_first = lambda p: p.get_current_rating()

    if mode == "random":
        decide_ties = decide_ties_random
    elif mode == "lowest_first":
        decide_ties = decide_ties_lowest_first
    else:
        logging.error(f"unknown pause mode {repr(mode)}, using random")
        decide_ties = decide_ties_random

    all_players = [
        (p, games_paused_per_games_played(p), decide_ties(p)) for p in list_of_players
    ]
    all_players_sorted = sorted(
        all_players, key=lambda t: (t[1], t[2], random.random())
    )

    if verbose:
        from viz import viz_players_to_pause

        viz_players_to_pause(all_players_sorted, num_to_pause)

    # return players to play and players to pause
    return [p for (p, _, _) in all_players_sorted[num_to_pause:]], [
        p for (p, _, _) in all_players_sorted[:num_to_pause]
    ]


def find_and_viz_solution(
    task_input: TaskInput,
    search_duration=20,
    log_tasks=True,
    viz_weight_matrices=True,
    verbose=True,
    minizinc_file="matching_algos/find_matchups.mzn",
    add_to_model=None,
    live_updates=False,
):
    # start_time = time.time()

    if log_tasks:
        from task_logger import TaskLogger

        task_logger = TaskLogger()
        task_logger.log_input(task_input)

    if viz_weight_matrices:
        print("Printing weight matricies:")
        task_input.viz_weights()

    minizinc_result = run_minizinc(
        task_input,
        search_duration,
        minizinc_file,
        verbose=verbose,
        add_to_model=add_to_model,
        live_updates=live_updates,
    )

    matchups = list()
    for idx_a1, idx_a2, idx_b1, idx_b2 in minizinc_res2participants(minizinc_result):
        pid_a1, pid_a2, pid_b1, pid_b2 = [
            task_input.player_ids[idx - 1] for idx in (idx_a1, idx_a2, idx_b1, idx_b2)
        ]
        matchups.append(
            Participants(
                pid_a1,
                pid_a2,
                pid_b1,
                pid_b2,
            )
        )

    task_output = TaskOutput(
        task_input,
        matchups,
        players_to_pause=[],
        # cost_time=time.time() - start_time
    )

    if (
        abs(task_output.cost.total - minizinc_result.cost) > 1
        and abs(task_output.cost_quad.total - minizinc_result.cost) > 1
    ):
        print(
            f"Minizinc returned cost {minizinc_result.cost}, which is not near the normal or quadratic cost ({task_output.cost.total} | {task_output.cost_quad.total})"
        )

    if log_tasks:
        task_logger.log_output(task_output)

    return task_output


def run_minizinc(
    task_input: TaskInput,
    search_duration,
    minizinc_file,
    verbose,
    live_updates=False,
    add_to_model=None,
):
    minizincf = run_minizinc_w_live_updates if live_updates else run_minizinc_not_async

    # for opt_level in range(5, -1, -1):
    for opt_level in [5, 0]:
        try:
            return minizincf(
                task_input,
                search_duration,
                minizinc_file,
                opt_level=opt_level,
                verbose=verbose,
                add_to_model=add_to_model,
            )

        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            if opt_level == 0:
                logging.error(f"Still failed with opt_level 0 with {e}")
                raise e

            logging.error(
                f"run_minizinc failed with {opt_level=}. Trying again with lower level now"
            )
            continue


def run_minizinc_not_async(
    task_input: TaskInput,
    search_duration,
    minizinc_file,
    opt_level,
    verbose,
    intermediate_solutions=True,
    add_to_model=None,
):
    instance = prepare_instance(
        task_input, mzn_file=minizinc_file, add_to_model=add_to_model
    )

    start_time = time.time()

    result = instance.solve(
        timeout=timedelta(seconds=search_duration),
        processes=8,
        optimisation_level=opt_level,
        intermediate_solutions=intermediate_solutions,
    )

    if intermediate_solutions:
        intermediate_solutions = result.solution[:]
        solution = result.solution[-1]

    if result is None or solution is None:
        print("Could not find a solution!")
        raise ValueError(f"Minizinc returned {result}.")

    if verbose:
        print(solution)

        print(f"Found after {time.time() - start_time:.2f}s")

        print("\n --------------------------------- \n")

    return MinizincResult(solution.matchup_table, solution.matchup_order, solution.COST)


def run_minizinc_w_live_updates(
    task_input: TaskInput,
    search_duration,
    minizinc_file,
    opt_level,
    verbose,
    add_to_model=None,
):
    async def generator():
        instance = prepare_instance(
            task_input, mzn_file=minizinc_file, add_to_model=add_to_model
        )

        start_time = time.time()

        result = None

        async for tmp_result in instance.solutions(
            timeout=timedelta(seconds=search_duration),
            processes=8,
            optimisation_level=opt_level,
        ):
            if tmp_result.solution is None and result is None:
                print("Could not find a solution!")
                continue

            if tmp_result.solution is None:
                continue

            result = tmp_result

            if verbose:
                print(result.solution)

                print(f"Found after {time.time() - start_time:.2f}s")

                print("\n --------------------------------- \n")

        return MinizincResult(
            result.solution.matchup_table,
            result.solution.matchup_order,
            result.solution.COST,
        )

    minizinc_result = asyncio.run(generator())
    return minizinc_result


def minizinc_res2participants(mres: MinizincResult):
    all_placements = list()
    for pos_type, four_players in zip(mres.matchuo_order, mres.matchup_table):
        four_players.insert(
            0, 0
        )  # so we can use the more intuitive indicies below (same as in minizinc
        if pos_type == 1:
            all_placements.append(
                Participants(
                    four_players[1], four_players[4], four_players[2], four_players[3]
                )
            )
        elif pos_type == 2:
            all_placements.append(
                Participants(
                    four_players[1], four_players[3], four_players[2], four_players[4]
                )
            )
        elif pos_type == 3:
            all_placements.append(
                Participants(
                    four_players[1], four_players[2], four_players[3], four_players[4]
                )
            )

    return all_placements


class MinizincNotFoundError(Exception):
    pass


def prepare_instance(
    task_input: TaskInput,
    mzn_file="matching_algos/find_matchups.mzn",
    data_file="matching_algos/find_matchups.dzn",
    add_to_model=None,
):
    n_players = len(task_input.rating_list)

    try:
        gecode = Solver.lookup("gecode")
    except AssertionError:
        raise MinizincNotFoundError()

    model = Model()
    model.add_file(mzn_file)
    if add_to_model:
        model.add_string(add_to_model)
    if data_file:
        model.add_file(data_file, parse_data=True)

    instance = Instance(gecode, model)
    assert n_players % 4 == 0
    assert type(n_players) == int

    instance["n_players"] = n_players
    instance["n_matchups"] = n_players // 4

    assert type(task_input.rating_list) == list
    assert (
        type(task_input.rating_list[0]) == int
    )  # it's actually defined as float in minizinc!
    instance["ELO"] = task_input.rating_list

    for l in (task_input.weight_game, task_input.weight_team):
        assert type(l) == list
        assert type(l[0]) == list
        assert type(l[0][0]) == int

        assert len(l) == n_players
        assert len(l[0]) == n_players

    if sum([sum(l) for l in task_input.weight_game]) == 0:
        print("Warning: 'weight_game' was all zeroes")

    if sum([sum(l) for l in task_input.weight_team]) == 0:
        print("Warning: 'weight_team' was all zeroes")

    instance["played_together_cost"] = task_input.weight_game
    instance["played_together_cost_team"] = task_input.weight_team

    instance["higher_rating_weight"] = task_input.higher_rating_weight

    return instance
