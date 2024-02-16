import random
from scipy.special import betaincinv


def calc_rdm_result(team_a_elo, team_b_elo):
    game_win_chance_team_a = 1/(1+10**((team_b_elo-team_a_elo)/400))
    point_win_chance_team_a = betaincinv(15, 15, game_win_chance_team_a)  # based on binominal distribution
    # got the betaincinv function from calculating the inverse with wolframalpha
    # e.g. https://www.wolframalpha.com/input?i=CDF%5BBinomialDistribution%5B29%2C+p%5D%2C+14%5D+%3D+0.6

    points_a = 0
    points_b = 0
    threshold = point_win_chance_team_a
    while abs(points_a - points_b) < 2 or (points_b < 15 and points_a < 15):
        if random.random() < threshold:
            points_a += 1
        else:
            points_b += 1

    return points_a, points_b
