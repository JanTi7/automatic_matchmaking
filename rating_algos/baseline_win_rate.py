import copy

from collections import defaultdict

from rating_algos.base_rating_algo import (
    BaseRatingAlgo,
    X_data,
    Y_data,
    Prediction,
    PredictionRaw,
    weighted_mean_with_w,
)

import logging


class BaselineWinrate(BaseRatingAlgo):
    def __init__(self):
        super().__init__()
        self.initial_state = defaultdict(lambda: [0, 0])

    def set_initial_rating_estimates(self, rating_estimates):
        logging.debug(f"{type(self).__name__} doesn't use rating estimates.")

    def fit(self, x: list[list[X_data]], y: list[list[Y_data]]):
        self.players = copy.deepcopy(self.initial_state)

        for game_block, game_block_results in zip(x, y):
            for game, game_result in zip(game_block, game_block_results):
                win_team_a = int(game_result.points_a > game_result.points_b)
                win_team_b = 1 - win_team_a

                for p in [game.a1, game.a2]:
                    self.players[p][win_team_a] += 1

                for p in [game.b1, game.b2]:
                    self.players[p][win_team_b] += 1

    def predict(self, x: list[X_data]) -> list[Prediction]:
        pass

    def predict_raw(self, x: list[X_data], higher_rating_weight) -> list[PredictionRaw]:
        def player_wr(p):
            try:
                return self.players[p][1] / sum(self.players[p])
            except ZeroDivisionError:
                return 0.5

        def team_wr(p1, p2):
            return weighted_mean_with_w(higher_rating_weight)(
                [player_wr(p1), player_wr(p2)]
            )

        return [
            PredictionRaw(
                self._who_wins(
                    0.5 + team_wr(game.a1, game.a2) - team_wr(game.b1, game.b2)
                ),
                0.5 + 0.5 * (team_wr(game.a1, game.a2) - team_wr(game.b1, game.b2)),
                ratings=tuple(player_wr(p) for p in game[:4]),
            )
            for game in x
        ]
