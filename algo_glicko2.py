import logging
import math
import statistics
import time
from collections import namedtuple

from dao import RatingSnapshot, id2name
from glicko2.glicko2 import Glicko2Player

RelativeGameData = namedtuple("RelativeGameData",
                              "teammate_rating opp1_rating opp2_rating team_points opp_points match_importance".split())


def register_game_result(game_res_id: str,
                         rating_a_1: RatingSnapshot, rating_a_2: RatingSnapshot,
                         rating_b_1: RatingSnapshot, rating_b_2: RatingSnapshot, points_a, points_b,
                         match_importance=1.5,
                         use_mif=True,  # TODO put in config
                         ):
    player_a_1 = Glicko2Player(rating=rating_a_1.rating, rd=rating_a_1.rd_time_adjusted(), pid=rating_a_1.player_id)
    player_a_2 = Glicko2Player(rating=rating_a_2.rating, rd=rating_a_2.rd_time_adjusted(), pid=rating_a_2.player_id)
    player_b_1 = Glicko2Player(rating=rating_b_1.rating, rd=rating_b_1.rd_time_adjusted(), pid=rating_b_1.player_id)
    player_b_2 = Glicko2Player(rating=rating_b_2.rating, rd=rating_b_2.rd_time_adjusted(), pid=rating_b_2.player_id)

    update_players(player_a_1, player_a_2, player_b_1, player_b_2,
                   points_a, points_b,
                   match_importance=match_importance,
                   use_mif=use_mif)

    return [
        RatingSnapshot(player_id=rating_a_1.player_id, timestamp=time.time(), rating=player_a_1.rating,
                       rd=player_a_1.rd, vol=player_a_1.vol, game_res_id=game_res_id),
        RatingSnapshot(player_id=rating_a_2.player_id, timestamp=time.time(), rating=player_a_2.rating,
                       rd=player_a_2.rd, vol=player_a_2.vol, game_res_id=game_res_id),
        RatingSnapshot(player_id=rating_b_1.player_id, timestamp=time.time(), rating=player_b_1.rating,
                       rd=player_b_1.rd, vol=player_b_1.vol, game_res_id=game_res_id),
        RatingSnapshot(player_id=rating_b_2.player_id, timestamp=time.time(), rating=player_b_2.rating,
                       rd=player_b_2.rd, vol=player_b_2.vol, game_res_id=game_res_id)
    ]


def update_players(player_a_1: Glicko2Player, player_a_2: Glicko2Player,
                   player_b_1: Glicko2Player, player_b_2: Glicko2Player,
                   points_a, points_b,
                   match_importance=1.5,

                   use_mif=False,
                   use_rubberband=True,
                   opp_rd_calc="team_mean",  # TODO: use + implement
                   # TODO add sigmoid params
                   rating_func=statistics.mean,
                   result_softener=None,
                   ):
    """
    :return: Nothing - changes players in place
    """

    result_softener = result_softener or _calc_actual_result

    def mif(elo_diff):  # match importance factor
        if not use_mif:
            return 1
        return 1 - 0.5 * sigmoid(0.02 * (elo_diff - 200))

    # TODO: Idea: combine the rd 3to1 (for each player individ) instead of 2to2
    _team_elo = lambda a, b: rating_func([a, b])
    team_a_elo = _team_elo(player_a_1.rating, player_a_2.rating)
    team_a_rd = statistics.mean([player_a_1.rd, player_a_2.rd])
    team_a_match_importance_factor = mif(abs(player_a_1.rating - player_a_2.rating))

    team_b_elo = _team_elo(player_b_1.rating, player_b_2.rating)
    team_b_rd = statistics.mean([player_b_1.rd, player_b_2.rd])
    team_b_match_importance_factor = mif(abs(player_b_1.rating - player_b_2.rating))

    logging.info(
        f"TEAM ELO A: {team_a_elo:.2f} (RD={team_a_rd:.2f}) Elo diff={abs(player_a_1.rating - player_a_2.rating):.0f}, mif={team_a_match_importance_factor:.3f}")
    logging.info(
        f"TEAM ELO B: {team_b_elo:.2f} (RD={team_b_rd:.2f}) Elo diff={abs(player_b_1.rating - player_b_2.rating):.0f}, mif={team_b_match_importance_factor:.3f}")

    result_a, result_b = result_softener(points_a, points_b)

    for player in [player_a_1, player_a_2]:
        player.setToTeamRating(team_a_elo)

    for player in [player_b_1, player_b_2]:
        player.setToTeamRating(team_b_elo)

    if opp_rd_calc == "team_mean":
        pa1_opp_rd, pa2_opp_rd = team_b_rd, team_b_rd
        pb1_opp_rd, pb2_opp_rd = team_a_rd, team_a_rd
    elif opp_rd_calc == "3v1":
        pa1_opp_rd = math.sqrt(sum([rd ** 2 for rd in (player_a_2.rd, player_b_1.rd, player_b_2.rd)]))
        pa2_opp_rd = math.sqrt(sum([rd ** 2 for rd in (player_a_1.rd, player_b_1.rd, player_b_2.rd)]))

        pb1_opp_rd = math.sqrt(sum([rd ** 2 for rd in (player_b_2.rd, player_a_1.rd, player_a_2.rd)]))
        pb2_opp_rd = math.sqrt(sum([rd ** 2 for rd in (player_b_1.rd, player_a_1.rd, player_a_2.rd)]))
    else:
        raise ValueError(f"Unknown opp_rd_calc mode {repr(opp_rd_calc)}. Allowed are 'team_mean' and '3v1'")

    player_a_1.update_player([team_b_elo], [pa1_opp_rd], [result_a],
                             match_importance=match_importance * team_a_match_importance_factor)
    player_a_2.update_player([team_b_elo], [pa2_opp_rd], [result_a],
                             match_importance=match_importance * team_a_match_importance_factor)

    player_b_1.update_player([team_a_elo], [pb1_opp_rd], [result_b],
                             match_importance=match_importance * team_b_match_importance_factor)
    player_b_2.update_player([team_a_elo], [pb2_opp_rd], [result_b],
                             match_importance=match_importance * team_b_match_importance_factor)

    for player in [player_a_1, player_a_2, player_b_1, player_b_2]:
        player.restoreSoloRating()

        if use_rubberband:
            if player.rating > 1700:
                overshoot = player.rating - 1700
                penalty = (overshoot / 100) ** 2
                logging.info(
                    f"Overshoot: {id2name.get(player.pid, player.pid)}'s rating was {player.rating:.0f}, applying penalty of {penalty:.1f}. New rating = {player.rating - penalty:.1f}")
                player.rating -= penalty

            if player.rating < 1300:
                undershoot = 1300 - player.rating
                bonus = (undershoot / 100) ** 2
                logging.info(
                    f"Undershoot: {id2name.get(player.pid, player.pid)}'s rating was {player.rating:.0f}, applying bonus of {bonus:.1f}. New rating = {player.rating + bonus:.1f}")

                player.rating += bonus


sigmoid = lambda x: 1 / (1 + math.exp(-x))


def _calc_actual_result(team_points_a, team_points_b, maxv=0.5, fac=6.5, offset=4.5):
    point_ratio = min(team_points_a, team_points_b) / max(team_points_a, team_points_b)
    courtesy = maxv * sigmoid(
        fac * point_ratio - offset)

    percentages = [1 - courtesy, courtesy]

    if team_points_b > team_points_a:
        percentages.reverse()

    return percentages


def glicko2predict(
        player_a_1: Glicko2Player, player_a_2: Glicko2Player,
        player_b_1: Glicko2Player, player_b_2: Glicko2Player,
        rating_func=statistics.mean,
        rd_func=statistics.mean,
        vol_func=statistics.mean
):
    team_a = Glicko2Player(
        rating=rating_func([player_a_1.rating, player_a_2.rating]),
        rd=rd_func([player_a_1.rd, player_a_2.rd]),
        vol=vol_func([player_a_1.vol, player_a_2.vol]),
    )

    team_b = Glicko2Player(
        rating=rating_func([player_b_1.rating, player_b_2.rating]),
        rd=rd_func([player_b_1.rd, player_b_2.rd]),
        vol=vol_func([player_b_1.vol, player_b_2.vol]),
    )

    return team_a.predict_outcome(team_b.rating, team_b.rd)
