import random
from abc import ABC, abstractmethod
from collections import namedtuple

# X_background = namedtuple("X_background", "games_in_training")
X_data = namedtuple("X_data", "a1 a2 b1 b2 timestamp game_id")
Y_data = namedtuple("Y_data", "points_a points_b")

PredictionRaw = namedtuple("Prediction", "winning_team percentage_win_a ratings")
Prediction = namedtuple("Prediction", "winning_team point_diff")


def weighted_mean_with_w(higher_rating_weight):
    def weighted_mean(values: list):
        low, high = sorted(values)
        high *= higher_rating_weight
        return (low + high) / (higher_rating_weight + 1)

    return weighted_mean


def get_result_softener(inp):
    from functools import partial

    if inp[0] == "binary":
        return lambda points_a, points_b: (
            int(points_a > points_b),
            int(points_b > points_a),
        )
    elif inp[0] == "linear":
        _, maxv = inp

        def lin_res_softener(pa, pb):
            minp, maxp = min([pa, pb]), max([pa, pb])
            fac = minp / (minp + maxp)
            courtesy = 2 * fac * maxv  # maxv is 0.5 at max

            percentages = [1 - courtesy, courtesy]

            if pb > pa:
                percentages.reverse()

            return percentages

        return lin_res_softener

    elif inp[0] == "sigmoid":
        from algo_glicko2 import _calc_actual_result

        _, maxv, fac, offset = inp
        return partial(_calc_actual_result, maxv=maxv, fac=fac, offset=offset)
    else:
        raise ValueError(f"No result softener with keyword {repr(inp[0])}")


class BaseRatingAlgo(ABC):
    def __init__(self):
        self.rng = random.Random(1)

    def _who_wins(self, pred_win_a):
        if pred_win_a != 0.5:
            return 0 if pred_win_a > 0.5 else 1

        # print("FixedRatingAlgo: randomly guessing!")
        return int(self.rng.randint(0, 1))

    @abstractmethod
    def set_initial_rating_estimates(self, rating_estimates):
        pass

    @abstractmethod
    def fit(self, x: list[list[X_data]], y: list[list[Y_data]]):
        pass

    @abstractmethod
    def predict(self, x: list[X_data]) -> list[Prediction]:
        pass

    @abstractmethod
    def predict_raw(self, x: list[X_data], higher_rating_weight) -> list[PredictionRaw]:
        pass
