import itertools
import logging
import time
from typing import Any

import numpy as np

from abc import ABC, abstractmethod

from matching_algos.task_input_output import TaskInput, TaskOutput, MatchupCost
from rating_algos.base_rating_algo import weighted_mean_with_w
from task_logger import TaskLogger


class MatchupCostCalculator:
    def __init__(self, rating_list, played_together_game, played_together_team, player_ids: list[str],
                 higher_rating_weight,
                 constant_dict=None):
        self.ratings = rating_list
        self.player_ids = player_ids
        assert all([x >= y for x, y in zip(rating_list, rating_list[1:])])
        self.played_together_game = played_together_game
        self.played_together_team = played_together_team

        self.higher_rating_weight = higher_rating_weight

        if constant_dict is None:
            from minizinc.dzn import parse_dzn
            from pathlib import Path
            constant_dict = parse_dzn(Path("matching_algos/minizinc_data.dzn"))

        self.factor_elo_gap = constant_dict["cost_factor_1d"][0]
        self.factor_team_diff = constant_dict["cost_factor_2d"][0]
        self.factor_played_together = constant_dict["cost_factor_2d"][1]

        self.reduction_factor = 0.01
        self.reduction_threshold_elo_gap = constant_dict["reduce_thresh_1d"][0]
        self.reduction_threshold_team_diff = constant_dict["reduce_thresh_2d"][0]
        self.reduction_threshold_played_together = constant_dict["reduce_thresh_2d"][1]

    @classmethod
    def from_taskinput(cls, task_input: TaskInput):
        return cls(task_input.rating_list, task_input.weight_game, task_input.weight_team,
                   task_input.player_ids,
                   higher_rating_weight=task_input.higher_rating_weight)

    def convert_pids_to_abs(self, games: list[tuple[str]]) -> list[tuple[int]]:
        return [
            tuple(self.player_ids.index(pid) for pid in game)
            for game in games
        ]

    def abs_to_pids_dict(self, d: dict[int, Any]) -> dict[str, Any]:
        return {
            self.player_ids[i]: v
            for i, v in d.items()
        }



    def check_and_conv_inp(self, games: list[tuple[str|int]]) -> list[tuple[int]]:
        if len(games) == 0:
            return games
        if type(games[0][0]) == str:
            return self.convert_pids_to_abs(games)
        else:
            return games



    def _reduce_if_below_threshold(self, val, threshold):
        if val < threshold:
            return val * self.reduction_factor
        else:
            return threshold * self.reduction_factor + (val - threshold)

    def _reduce_elo_gap(self, val):
        return self._reduce_if_below_threshold(val, self.reduction_threshold_elo_gap)

    def _reduce_team_diff(self, val):
        return self._reduce_if_below_threshold(val, self.reduction_threshold_team_diff)

    def _reduce_played_together(self, val):
        return self._reduce_if_below_threshold(val, self.reduction_threshold_played_together)

    def elo_gap(self, games: list[tuple[int]], exponent=1):
        return self.factor_elo_gap * sum([
            self._reduce_elo_gap(self.ratings[min(g)] - self.ratings[max(g)]) ** exponent
            for g in games
        ])

    def _elo_gap_wrt_single_person(self, person, games: list[tuple[int]], exponent=1, raw=False):
        relevant_games = [g for g in games if person in g]
        results = list()
        for game in relevant_games:
            results.append(max([abs(self.ratings[p] - self.ratings[person]) for p in game]))

        if raw:
            return sum([r**exponent for r in results])

        return self.factor_elo_gap * sum([
            self._reduce_elo_gap(r) ** exponent
            for r in results
        ])

    def elo_gap_duo(self, idx1, idx2, exponent=1):
        return (self.factor_elo_gap * self._reduce_elo_gap(
            abs(self.ratings[idx1] - self.ratings[idx2]))) ** exponent

    def team_diff(self, games: list[tuple[int]], exponent=1, raw=False):
        wmean = weighted_mean_with_w(self.higher_rating_weight)
        res = 0
        for g in games:
            team_a = wmean([self.ratings[g[0]], self.ratings[g[1]]])
            team_b = wmean([self.ratings[g[2]], self.ratings[g[3]]])

            if not raw:
                res += self._reduce_team_diff(abs(team_a - team_b)) ** exponent
            else:
                res += abs(team_a - team_b) ** exponent

        if raw:
            return res

        return self.factor_team_diff * res

    def played_together(self, games: list[tuple[int]], exponent=1):
        return self.factor_played_together * sum([
            self._reduce_played_together(
                self.played_together_team[g[0]][g[1]] +
                self.played_together_team[g[2]][g[3]] +

                self.played_together_game[g[0]][g[2]] +
                self.played_together_game[g[0]][g[3]] +
                self.played_together_game[g[1]][g[2]] +
                self.played_together_game[g[1]][g[3]]
            ) ** exponent
            for g in games
        ])

    def _played_together_wrt_single_person(self, person, games: list[tuple[int]], exponent=1, raw=False):
        relevant_games = [g for g in games if person in g]
        results = list()
        for game in relevant_games:
            idx = game.index(person)
            team = (idx // 2) * 2
            teammate_idx = team + 1 - (idx % 2)
            teammate = game[teammate_idx]
            opponents = [p for p in game if p not in [person, teammate]]
            assert len(opponents) == 2, f"{person=} {teammate_idx=} {teammate=} {game=} {len(opponents)=} {opponents=}"

            results.append(
                self.played_together_team[person][teammate] +
                self.played_together_game[person][opponents[0]] +
                self.played_together_game[person][opponents[1]]
            )

        if raw:
            return sum([r**exponent for r in results])

        return self.factor_played_together * sum([
            self._reduce_played_together(r) ** exponent
            for r in results
        ])

    def played_together_team_duo(self, idx1, idx2, exponent=1):
        return (self.factor_played_together * self._reduce_played_together(
            self.played_together_team[idx1][idx2])) ** exponent

    def min_cost_for_tuple(self, tuple_of_four):
        sorted_list = sorted(tuple_of_four)
        positions = [[0, 3, 1, 2],
                     [0, 2, 1, 3],
                     [0, 1, 2, 3],
                     ]
        vars = [[sorted_list[i] for i in pos] for pos in positions]

        options = [(self.total_cost_quad([v]), v) for v in vars]
        # options = sorted(options)

        return min(options, key=lambda t: t[0].total)

    def total_cost(self, games: list[tuple[int]]) -> MatchupCost:
        return MatchupCost(
            elo_gap=(eg := self.elo_gap(games)),
            team_diff=(td := self.team_diff(games)),
            played_together=(pt := self.played_together(games)),
            total=eg + td + pt
        )

    def total_cost_quad(self, games: list[tuple[int]]) -> MatchupCost:
        return MatchupCost(
            elo_gap=(eg := self.elo_gap(games, 2)),
            team_diff=(td := self.team_diff(games, 2)),
            played_together=(pt := self.played_together(games, 2)),
            total=eg + td + pt
        )

    def cost_per_person(self, games: list[tuple[int]]) -> dict:
        games = self.check_and_conv_inp(games)
        costs = dict()
        for person in itertools.chain(*[t for t in games]):
            game = [g for g in games if person in g]  # list of size one
            assert len(game) == 1, f"problem with {game=} | {games=}, {person=}"
            costs[person] = MatchupCost(
                elo_gap=(eg := self._elo_gap_wrt_single_person(person, game, 1)),
                team_diff=(td := self.team_diff(game, 1)),
                played_together=(pt := self._played_together_wrt_single_person(person, games, 1)),
                total=eg + td + pt
            )

        return costs

    def cost_per_person_raw(self, games: list[tuple[int]]) -> dict:
        games = self.check_and_conv_inp(games)
        costs = dict()
        for person in itertools.chain(*[t for t in games]):
            game = [g for g in games if person in g]  # list of size one
            assert len(game) == 1, f"problem with {game=} | {games=}, {person=}"
            costs[person] = MatchupCost(
                elo_gap=(eg := self._elo_gap_wrt_single_person(person, game, 1, raw=True)),
                team_diff=(td := self.team_diff(game, 1, raw=True)),
                played_together=(pt := self._played_together_wrt_single_person(person, games, 1, raw=True)),
                total=eg + td + pt
            )

        return costs

    def teammate_cost_matrix(self, exponent=1) -> np.array:  # tuple[np.array, list[str]]:
        cost_matrix = np.zeros((len(self.ratings), len(self.ratings)))

        wmean = weighted_mean_with_w(self.higher_rating_weight)
        all_team_ratings = [(wmean([self.ratings[i], self.ratings[j]]), frozenset((i, j)))
                            for i in range(0, len(self.ratings))
                            for j in range(i + 1, len(self.ratings))]
        all_team_ratings = sorted(all_team_ratings, key=lambda t: t[0])

        for i in range(len(self.ratings)):
            for j in range(i + 1, len(self.ratings)):
                played_together = self.played_together_team_duo(i, j, exponent)
                elo_diff = self.elo_gap_duo(i, j, exponent)
                team_rating = wmean([self.ratings[i], self.ratings[j]])
                other_teams_ratings = [rating for rating, team in
                                       filter(lambda t: len(t[1].union(frozenset([i, j]))) == 4, all_team_ratings)]

                diff_ratings = [abs(team_rating - other_team_rating) for other_team_rating in other_teams_ratings]
                team_diff_estimate = (self.factor_team_diff * self._reduce_team_diff(min(diff_ratings))) ** exponent

                total_cost = played_together + elo_diff + team_diff_estimate
                cost_matrix[i, j] = total_cost
                cost_matrix[j, i] = total_cost

        return cost_matrix

    def print_detailed_cost_analysis(self, taskoutput: TaskOutput):
        from dao import generate_playerid_to_uniquename_map, load_default_database
        load_default_database()  # todo this is pretty hacky

        id2name = generate_playerid_to_uniquename_map(taskoutput.input.player_ids)
        idx2name = {taskoutput.input.player_ids.index(pid): name for pid, name in id2name.items()}

        ratings = taskoutput.input.rating_list

        for matchup in taskoutput.matchups_as_idx():
            single_cost = self.total_cost([matchup])
            single_cost_quad = self.total_cost_quad([matchup])

            a, b, c, d = matchup
            print("Looking at matchup", [idx2name[v] for v in (a, b, c, d)])

            print("Elo Gap:")
            diff = ratings[min(matchup)] - ratings[max(matchup)]
            reduction = self._reduce_elo_gap(diff)
            with_factor = reduction * self.factor_elo_gap
            with_exponent = self.factor_elo_gap * reduction ** 2
            print(f"Max: {idx2name[min(matchup)]} ({ratings[min(matchup)]}) - Min: {idx2name[max(matchup)]} ({ratings[max(matchup)]})")
            print(f"{diff=}, {reduction=}, {with_factor=}, {with_exponent=}")

            assert with_factor == single_cost.elo_gap
            assert with_exponent == single_cost_quad.elo_gap, f"{with_exponent=}, {single_cost_quad.elo_gap=}"

            print("Team Diff:")
            wmean = weighted_mean_with_w(self.higher_rating_weight)
            team_a = wmean([self.ratings[a], self.ratings[b]])
            team_b = wmean([self.ratings[c], self.ratings[d]])
            diff = abs(team_a-team_b)

            reduction = self._reduce_team_diff(diff)
            with_factor = reduction * self.factor_team_diff
            with_exponent = self.factor_team_diff * reduction ** 2
            print(f"Elo Team A {team_a}, Elo Team B {team_b}")
            print(f"{diff=}, {reduction=}, {with_factor=}, {with_exponent=}")

            assert with_factor == single_cost.team_diff
            assert with_exponent == single_cost_quad.team_diff, f"{with_exponent=}, {single_cost_quad.elo_gap=}"

            print("Played together")
            # pt_cost = 0
            team_a = self.played_together_team[a][b]
            team_b = self.played_together_team[c][d]
            print(f"Kosten für Team a = {team_a}, für Team b {team_b}")

            print(f"{self.played_together_game[a][c]}")
            print(f"{self.played_together_game[a][d]}")
            print(f"{self.played_together_game[b][c]}")
            print(f"{self.played_together_game[b][d]}")

            game_cost = self.played_together_game[a][c] + \
                        self.played_together_game[a][d] + \
                        self.played_together_game[b][c] + \
                        self.played_together_game[b][d]

            diff = team_a + team_b + game_cost
            reduction = self._reduce_played_together(diff)
            with_factor = reduction * self.factor_played_together
            with_exponent = self.factor_played_together * reduction ** 2
            print(f"{diff=}, {reduction=}, {with_factor=}, {with_exponent=}")

            assert with_factor == single_cost.played_together
            assert with_exponent == single_cost_quad.played_together, f"{with_exponent=}, {single_cost_quad.elo_gap=}"

            print()


class BaseMatchingAlgo(ABC):
    def __init__(self, name):
        self.name = name

    @abstractmethod
    def _find_matching(self, task_input: TaskInput, *args, **kwargs) -> TaskOutput:
        pass

    def _sort_matching(self, matching: list[list[int]]):
        logging.debug(f"Matching pre _sort_matching: {matching}")
        new_matchings = list()
        for a, b, c, d in matching:
            t1 = sorted([a, b])
            t2 = sorted([c, d])
            new_matchings.append(list(itertools.chain(*sorted([t1, t2]))))

        res = list(sorted(new_matchings, reverse=True))
        logging.debug(f"Matching post _sort_matching: {res}")

        return res

    def find_matching(self, task_input: TaskInput, log_task=False, *args, **kwargs) -> TaskOutput:
        start_time = time.time()
        output = self._find_matching(task_input, *args, **kwargs)
        output.cost_time = time.time() - start_time

        if log_task:
            task_logger = TaskLogger()
            task_logger.log_input(task_input)
            task_logger.log_output(output)

        logging.debug(f"{self.name} returned the following matchup: {output.matchups} | {output.matchups_as_idx()}")
        logging.debug(f"Input info {list(enumerate(task_input.player_ids))}")

        return output

    def _indices_to_player_ids(self, matchup_matrix, task_input: TaskInput):
        return [
            tuple(task_input.player_ids[p] for p in game)
            for game in self._sort_matching(matchup_matrix)
        ]
