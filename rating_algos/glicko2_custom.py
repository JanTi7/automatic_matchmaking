import copy

from algo_glicko2 import update_players, glicko2predict
from collections import defaultdict

from rating_algos.base_rating_algo import (
    BaseRatingAlgo,
    X_data,
    Y_data,
    Prediction,
    PredictionRaw,
    weighted_mean_with_w,
)
from glicko2 import Glicko2Player

from dao import time_adjust_rd
from datetime import timedelta


class Glicko2Custom(BaseRatingAlgo):
    def __init__(
        self,
        start_rating,
        start_rd,
        start_vol,
        rubberband,
        mif,
        opp_rd_calc,
        inc_rd_with_time,
        higher_rating_weight,
        result_softener,
        inc_rd_maxv=155,
        inc_rd_inc_factor=25,
        inc_rd_max_inc=100,
    ):
        super().__init__()
        self.rubberband = rubberband
        self.mif = mif
        self.opp_rd_calc = opp_rd_calc
        self.inc_rd_with_time = inc_rd_with_time
        self.inc_rd_maxv = inc_rd_maxv
        self.inc_rd_inc_factor = inc_rd_inc_factor
        self.inc_rd_max_inc = inc_rd_max_inc

        self.higher_rating_weight = higher_rating_weight

        self.initial_state = defaultdict(
            lambda: Glicko2Player(start_rating, start_rd, start_vol)
        )

        self.result_softener = result_softener

    def set_initial_rating_estimates(self, rating_estimates):
        for pid, rating in rating_estimates.items():
            self.initial_state[pid].rating = rating

    def fit(self, x: list[list[X_data]], y: list[list[Y_data]]):
        self.players = copy.deepcopy(self.initial_state)
        last_game_ts = dict()

        for game_block, game_block_results in zip(x, y):
            for game, game_result in zip(game_block, game_block_results):
                if self.inc_rd_with_time:
                    for pid in game[:4]:  # just the 4 players
                        if pid in last_game_ts:
                            self.players[pid].rd = time_adjust_rd(
                                self.players[pid].rd,
                                timedelta(seconds=game.timestamp - last_game_ts[pid]),
                                max_rd=self.inc_rd_maxv,
                                inc_constant=self.inc_rd_inc_factor,
                                max_inc=self.inc_rd_max_inc,
                            )

                        last_game_ts[pid] = game.timestamp

                update_players(
                    self.players[game.a1],
                    self.players[game.a2],
                    self.players[game.b1],
                    self.players[game.b2],
                    game_result.points_a,
                    game_result.points_b,
                    use_mif=self.mif,
                    use_rubberband=self.rubberband,
                    opp_rd_calc=self.opp_rd_calc,
                    rating_func=weighted_mean_with_w(self.higher_rating_weight),
                    result_softener=self.result_softener,
                )

    def predict(self, x: list[X_data]) -> list[Prediction]:
        return [Prediction(None, 0) for game in x]

    def predict_raw(self, x: list[X_data], higher_rating_weight) -> list[PredictionRaw]:
        return [
            PredictionRaw(
                self._who_wins(win_team_a),  # 0 means team a - 1 team b
                win_team_a,
                ratings=tuple(self.players[pid].rating for pid in game[:4]),
            )
            for game in x
            if (
                win_team_a := glicko2predict(
                    self.players[game.a1],
                    self.players[game.a2],
                    self.players[game.b1],
                    self.players[game.b2],
                    rating_func=weighted_mean_with_w(higher_rating_weight),
                )
            )
            is not None  # just filler since the precition might be 0
        ]
