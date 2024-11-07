import itertools
import logging
from dataclasses import dataclass, field

from collections import namedtuple

MatchupCost = namedtuple("MatchupCost", "total elo_gap team_diff played_together")


@dataclass
class TaskInput:
    player_ids: list
    rating_list: list
    weight_team: object
    weight_game: object
    higher_rating_weight: float

    def __post_init__(self):
        assert all(
            r1 >= r2 for r1, r2 in zip(self.rating_list, self.rating_list[1:])
        ), f"ratings weren't sorted {self.rating_list}"

    def viz_weights(self, vizf=logging.debug):
        from dao import generate_playerid_to_uniquename_map

        id2name = generate_playerid_to_uniquename_map(self.player_ids)
        for w_type, matrix in [("TEAM", self.weight_team), ("GAME", self.weight_game)]:
            vizf(f"-------------{w_type}-------------")

            from collections import defaultdict

            wd = defaultdict(list)
            for i in range(len(self.player_ids)):
                for j in range(i + 1, len(self.player_ids)):
                    weight = matrix[i][j]
                    if weight > 0:
                        # print(weight)
                        wd[weight].append((self.player_ids[i], self.player_ids[j]))

            keys = sorted(wd.keys(), reverse=True)

            for key in keys:
                vizf(f"---{key}---")
                for p1, p2 in wd[key]:
                    vizf(f"    {id2name[p1]} - {id2name[p2]}")


@dataclass
class TaskOutput:
    input: TaskInput
    matchups: list[
        tuple[str]
    ]  # List of tuples with 4 elements, representing a1, a2, b1, b2, containing player_id
    players_to_pause: list  # if/when matcher gets to decide who pauses
    cost_time: float = field(init=False)
    cost: MatchupCost = field(init=False)  # Optional[MatchupCost | float]
    cost_quad: MatchupCost = field(init=False)  # Optional[MatchupCost]

    def convert_to_proposed_games(self, matrix_manager):
        from dao import GameProposed

        return [
            GameProposed(
                *[matrix_manager.get_rating_snap_id_from_pid(pid) for pid in m]
            )
            for m in self.matchups
        ]

    def _check(self):
        players_defined_in_input = set(self.input.player_ids)
        players_defined_in_output = set(itertools.chain(*self.matchups)).union(
            set(self.players_to_pause)
        )

        assert (
            players_defined_in_input == players_defined_in_output
        ), f"{players_defined_in_input=}\n{players_defined_in_output=}"

    def __post_init__(self):
        from matching_algos.base_matching_algo import MatchupCostCalculator

        self._check()

        cost_calculator = MatchupCostCalculator.from_taskinput(self.input)

        matchups = self.matchups_as_idx()
        self.cost = cost_calculator.total_cost(matchups)
        self.cost_quad = cost_calculator.total_cost_quad(matchups)

    def matchups_as_idx(self) -> list[tuple[int]]:
        return [
            tuple(self.input.player_ids.index(player) for player in g)
            for g in self.matchups
        ]
