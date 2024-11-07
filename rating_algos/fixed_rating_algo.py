import itertools
import random
from collections import defaultdict

from algo_glicko2 import glicko2predict
from rating_algos.base_rating_algo import (
    BaseRatingAlgo,
    X_data,
    PredictionRaw,
    Prediction,
    Y_data,
    weighted_mean_with_w,
)
from dao import GameResult, RatingSnapshot


class FixedRatingAlgo(BaseRatingAlgo):
    def __init__(self, higher_rating_weight):
        super().__init__()
        self.players = defaultdict(lambda: 1500)
        # self.rng = random.Random(1)

        self.higher_rating_weight = higher_rating_weight

    def set_initial_rating_estimates(self, rating_estimates):
        self.players.update(rating_estimates)

    def fit(self, x: list[list[X_data]], y: list[list[Y_data]]):
        pass

    def predict(self, x: list[X_data]) -> list[Prediction]:
        pass

    def _predict(self, players, higher_rating_weight=None):
        higher_rating_weight = higher_rating_weight or self.higher_rating_weight
        wmean = weighted_mean_with_w(higher_rating_weight)
        team_elo_a = wmean([self.players[pid] for pid in players[0:2]])
        team_elo_b = wmean([self.players[pid] for pid in players[2:4]])

        return 1 / (1 + 10 ** ((team_elo_b - team_elo_a) / 400))

    def predict_raw(self, x: list[X_data], higher_rating_weight) -> list[PredictionRaw]:
        res = [
            PredictionRaw(
                self._who_wins(win_team_a),  # 0 means team a - 1 team b
                win_team_a,
                ratings=tuple(self.players[pid] for pid in game[:4]),
            )
            for game in x
            if (
                win_team_a := self._predict(
                    game[:4], higher_rating_weight=higher_rating_weight
                )
            )
            is not None  # just filler since the precition might be 0
        ]
        return res
