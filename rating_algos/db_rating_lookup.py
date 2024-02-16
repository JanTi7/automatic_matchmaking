import itertools
from algo_glicko2 import glicko2predict
from rating_algos.base_rating_algo import BaseRatingAlgo, X_data, PredictionRaw, Prediction, Y_data, \
    weighted_mean_with_w


class DbRatingLookup(BaseRatingAlgo):
    def __init__(self, full_rating_est):
        super().__init__()
        self.lookup_dict = dict()

        for game_id, *ratings in itertools.chain(*full_rating_est):
            self.lookup_dict[game_id] = [rs.to_glicko2_player() for rs in ratings]

    def set_initial_rating_estimates(self, rating_estimates):
        pass

    def fit(self, x: list[list[X_data]], y: list[list[Y_data]]):
        pass

    def predict(self, x: list[X_data]) -> list[Prediction]:
        pass

    def predict_raw(self, x: list[X_data], higher_rating_weight) -> list[PredictionRaw]:
        return [
            PredictionRaw(
                self._who_wins(win_team_a),  # 0 means team a - 1 team b
                win_team_a,
                ratings=[rs.rating for rs in self.lookup_dict[game.game_id]]
            )
            for game in x
            if (win_team_a := glicko2predict(*self.lookup_dict[game.game_id],
                                             rating_func=weighted_mean_with_w(higher_rating_weight)))
               is not None  # just filler since the precition might be 0
        ]
