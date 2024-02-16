import copy

from collections import defaultdict

from rating_algos.base_rating_algo import BaseRatingAlgo, X_data, Y_data, Prediction, PredictionRaw, weighted_mean_with_w


class Elo(BaseRatingAlgo):

    def __init__(self, start_rating, higher_rating_weight, result_softener, k=32):
        super().__init__()
        self.initial_state = defaultdict(lambda: start_rating)
        self.higher_rating_weight = higher_rating_weight
        self.k = k
        self.result_softener = result_softener

    def set_initial_rating_estimates(self, rating_estimates):
        for pid, rating in rating_estimates.items():
            self.initial_state[pid] = rating

    def fit(self, x: list[list[X_data]], y: list[list[Y_data]]):
        # TODO: idea - go through data twice - once reversed and then normal - to inflate amount of games
        self.players = copy.deepcopy(self.initial_state)

        for game_block, game_block_results in zip(x, y):
            for game, game_result in zip(game_block, game_block_results):
                expected_a = self._predict(game[:4])
                actual_team_a = self.result_softener(game_result.points_a, game_result.points_b)[0]

                rating_change_a = self.k * (actual_team_a-expected_a)

                for pid, fac in [(game.a1, 1), (game.a2, 1), (game.b1, -1), (game.b2, -1)]:
                    self.players[pid] += fac * rating_change_a


    def _predict(self, players, higher_rating_weight=None):
        higher_rating_weight = higher_rating_weight or self.higher_rating_weight
        wmean = weighted_mean_with_w(higher_rating_weight)
        team_elo_a = wmean([self.players[pid] for pid in players[0:2]])
        team_elo_b = wmean([self.players[pid] for pid in players[2:4]])

        return 1/(1+10**((team_elo_b-team_elo_a)/400))

    def predict(self, x: list[X_data]) -> list[Prediction]:
        pass

    def predict_raw(self, x: list[X_data], higher_rating_weight) -> list[PredictionRaw]:
        return [
            PredictionRaw(
                self._who_wins(win_team_a),  # 0 means team a - 1 team b
                win_team_a,
                ratings=tuple(self.players[pid] for pid in game[:4])
            )
            for game in x
            if (win_team_a := self._predict(game[:4], higher_rating_weight=higher_rating_weight))
               is not None  # just filler since the precition might be 0
        ]
