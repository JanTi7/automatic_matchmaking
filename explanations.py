import itertools
import logging

from rich.panel import Panel

from matching_algos.base_matching_algo import MatchupCostCalculator
from matching_algos.task_input_output import TaskOutput, MatchupCost

from rich.table import Table
from rich.columns import Columns
from rich.align import Align
from rich.text import Text
from rich.style import Style

from viz import pid2panel


def default_tableX(highlight_idx, show_header=True):
    table = Table(show_edge=False, show_header=show_header)  # , style="on black"

    table.add_column("", width=18)
    table.add_column("", width=18)
    table.add_column(" ", width=2)
    table.add_column("", width=18)
    table.add_column("", width=18)

    # headers = ["Rating Unter schied", "Team Unter schied", "Bereits zsm gesplt", "Summe"]
    headers = ["Skill Gap", "Balance Diff", "Played to gether", "Sum"]
    to_highlight = headers.pop(highlight_idx)
    headers.insert(highlight_idx, Text(to_highlight, style=Style(bgcolor="#ff7a7a")))

    for header in headers:
        table.add_column(header, width=6)

    return table


def to1d(l: list[list]) -> list:
    return list(itertools.chain(*l))


def to2d(l: list) -> list[list]:
    return [l[i : i + 4] for i in range(0, len(l), 4)]


def generate_counterfactuals(
    task_output: TaskOutput, print_to_terminal=True, optimize_swaps=True
) -> list[Columns]:
    from dao import generate_playerid_to_uniquename_map

    if not print_to_terminal:
        print = lambda *args, **kwargs: None

    entries = list()

    id2name = generate_playerid_to_uniquename_map(
        task_output.input.player_ids, bold_first_name=True
    )
    idx2name = {
        idx: id2name[pid] for idx, pid in enumerate(task_output.input.player_ids)
    }

    idx2pid = {idx: pid for idx, pid in enumerate(task_output.input.player_ids)}

    matchup2name = lambda m: [idx2name[p] for p in m]

    cost_calc = MatchupCostCalculator.from_taskinput(task_output.input)

    # field_dict = dict(
    #     elo_gap="Gesamte Rating-Spanne",
    #     team_diff="Unterschied in Teamstärke",
    #     played_together="Bereits Zusammengespielt"
    # )
    #
    # kosten_dict = dict(
    #     elo_gap="Die Paarung mit den am weitesten auseinanderliegenden Ratings ist",
    #     team_diff="Die Paarung mit dem größten Unterschied in Teamstärke ist",
    #     played_together="Die Paarung in der die meisten Leute schonmal zusammengespielt haben ist"
    # )
    field_dict = dict(
        elo_gap="Total Skill Gap",
        team_diff="Team Balance Difference",
        played_together="Played Together Already",
    )

    kosten_dict = dict(
        elo_gap="The pairing with the highest difference in skill rating is",
        team_diff="The pairing with the highest imbalance in team strength is",
        played_together="The pairing with the most players haveing already played together is",
    )

    matchups = task_output.matchups_as_idx()
    matchups1d = to1d(matchups)

    total_cost_original = cost_calc.total_cost_quad(matchups).total
    pairings_with_cost = [
        (idx, cost_calc.total_cost_quad([pairing]))
        for idx, pairing in enumerate(matchups)
    ]

    # print(f"{pairings_with_cost=}")
    costs = [cost for p, cost in pairings_with_cost]

    import statistics

    cost_norm_fac = statistics.mean([c.total for c in costs]) / 3
    if cost_norm_fac == 0:
        cost_norm_fac = 1
    print("cost norm fac", cost_norm_fac)

    norm_cost = lambda c: MatchupCost(*[round(v / cost_norm_fac, 2) for v in c])

    logging.debug(f"[explanation_gen] Un-normed costs: {pairings_with_cost}")
    logging.debug(
        f"[explanation_gen] Normed costs: {[(p, norm_cost(c)) for p, c in pairings_with_cost]}"
    )

    for field_idx, (field, title) in enumerate(field_dict.items()):
        col = Columns()
        col.add_renderable(
            Panel(
                Columns([Align.center(Text.from_markup(f"[b]{title}[/b]"))]),
                height=3,
                expand=False,
            )
        )

        max_val_for_field = max([getattr(cost, field) for cost in costs])
        if max_val_for_field == 0:
            print(f"Kein Matchup hat {title}-Kosten.")
            # col.add_renderable(Panel(f"Kein Pairing hat {repr(title)}-Kosten."))
            col.add_renderable(Panel(f"No pairing has {repr(title)}-costs."))
            entries.append(col)
            continue
        if max_val_for_field / cost_norm_fac <= 0.6:
            print(
                f"Vernachlässigbare {field}-Kosten. ({max_val_for_field/cost_norm_fac})"
            )
            # col.add_renderable(Panel(f"Vernachlässigbare {repr(title)}-Kosten."))
            col.add_renderable(Panel(f"Pairings have negligible {repr(title)}-costs."))
            entries.append(col)
            continue

        candidates = [
            p
            for p, cost in pairings_with_cost
            if getattr(cost, field) == max_val_for_field
        ]

        candidate = candidates[0]

        possible_counterfactuals = list()
        for i in range(len(matchups1d)):
            for j in range(i + 1, len(matchups1d)):
                mcopy = matchups1d.copy()
                mcopy[i], mcopy[j] = mcopy[j], mcopy[i]
                swap_idxs = (mcopy[i], mcopy[j])

                mcopy2d = to2d(mcopy)

                if optimize_swaps:
                    m1idx = i // 4
                    m2idx = j // 4
                    for m_idx in [m1idx, m2idx]:
                        mcopy2d[m_idx] = cost_calc.min_cost_for_tuple(mcopy2d[m_idx])[1]

                cost_candidate = cost_calc.total_cost_quad([mcopy2d[candidate]])
                new_cost = getattr(cost_candidate, field)
                if new_cost >= max_val_for_field - max(
                    0.2 * max_val_for_field, 0.33 * max_val_for_field / cost_norm_fac
                ):
                    # print(f"{i, j} zu tauschen hatte keine oder negative Auswirkungen. {max_val_for_field=} {new_cost=}")
                    continue

                # print(f"{i, j} zu tauschen hatte positive Auswirkungen. {max_val_for_field=} {new_cost=}")
                # total_cost_of_variation = cost_calc.total_cost_quad(mcopy2d).total
                total_cost_of_variation = cost_calc.total_cost_quad(mcopy2d).total

                if total_cost_of_variation < total_cost_original:
                    # we can't be having counterfactuals that are better than the original
                    continue

                possible_counterfactuals.append(
                    (total_cost_of_variation, new_cost, mcopy2d, swap_idxs)
                )

        if len(possible_counterfactuals) == 0:
            # col.add_renderable(Panel(f"Kein Verbesserungsbeispiel für {repr(title)} gefunden."))
            col.add_renderable(
                Panel(f"No example of improvement found for {repr(title)}.")
            )
            entries.append(col)

            print(f"Kein Counterfactual für {field} gefunden!")
            continue

        possible_counterfactuals = list(
            sorted(possible_counterfactuals, key=lambda t: t[:2])
        )
        # pprint(possible_counterfactuals)
        best_counterfactual = possible_counterfactuals[0][2]
        best_counterfactual_cost = possible_counterfactuals[0][0]
        best_counterfactual_swap_idxs = sorted(
            possible_counterfactuals[0][3],
            key=lambda idx: idx not in matchups[candidate],
        )

        altered_original_matchup = best_counterfactual[candidate]
        other_changed_matchup = [
            idx
            for idx, m in enumerate(best_counterfactual)
            if idx != candidate and tuple(m) != matchups[idx]
        ]

        assert len(other_changed_matchup) <= 1, (
            f"Zu viele veränderte Matchups, "
            f"{matchups=}, {best_counterfactual=} {other_changed_matchup=}"
        )

        names_and_cost = (
            lambda pairing: str(matchup2name(pairing))
            + " -> "
            + str(norm_cost(cost_calc.total_cost_quad([pairing])))
        )

        def format_initial_cost(pairing):
            new_cost = norm_cost(cost_calc.total_cost_quad([pairing]))
            return [
                Align.center(str(round(v, 1)), vertical="middle")
                for v in [
                    new_cost.elo_gap,
                    new_cost.team_diff,
                    new_cost.played_together,
                    new_cost.total,
                ]
            ]

        def matchup_to_tab(
            pairing,
            show_header=False,
            highlight_idxs=(),
            highlight_idx=None,
            highlight_players=(),
            comparative_costs=None,
        ):
            highlight_idxs = list(highlight_idxs) + [highlight_idx]

            tab = default_tableX(field_idx, show_header=show_header)
            row_data = [
                pid2panel(
                    idx2pid[idx],
                    id2name,
                    emph=count in highlight_idxs or idx in highlight_players,
                )
                for count, idx in enumerate(pairing)
            ]
            row_data.insert(2, Align.center("v", vertical="middle"))
            row_data.extend(format_initial_cost(pairing))

            tab.add_row(*row_data)
            return tab

        def find_diff(p1, p2):
            for i in range(len(p1)):
                if p1[i] != p2[i]:
                    return i

        def find_both_diffs(p1, p2):
            res = list()
            for i in range(len(p1)):
                if p1[i] != p2[i]:
                    res.append(i)
                    if len(res) == 2:
                        return res

        def non_zero_round(f):
            def roundf(f, digits):
                if digits == 0:
                    return round(f)
                return round(f, digits)

            digits = 0
            while (r := roundf(f, digits)) == 0:
                digits += 1
            return r

        matchup_as_names = matchup2name(matchups[candidate])
        man = matchup_as_names

        if len(other_changed_matchup) == 0:
            print(title, "Tausch war matchup intern")
            print(title, "Ausgangspairing:", names_and_cost(matchups[candidate]))
            print(
                title, "Verändertes Pairing:", names_and_cost(altered_original_matchup)
            )

            # diff_idxs = find_both_diffs(matchups[candidate], altered_original_matchup)
            # print("diff_idxs", diff_idxs)
            persons_who_switched = [
                idx2name[idx] for idx in best_counterfactual_swap_idxs
            ]

            # col.add_renderable(
            #     f"  {kosten_dict[field]} {man[0]} & {man[1]} vs {man[2]} & {man[3]}.\n" + \
            #     f"  man könnte diesen speziellen Wert ändern indem man {' und '.join(persons_who_switched)} tauscht.")
            col.add_renderable(
                f"  {kosten_dict[field]} {man[0]} & {man[1]} vs {man[2]} & {man[3]}.\n"
                + f"  This specific cost could be lowered by swapping {' and '.join(persons_who_switched)}."
            )

            col.add_renderable(
                matchup_to_tab(
                    matchups[candidate],
                    True,
                    highlight_players=best_counterfactual_swap_idxs,
                )
            )
            # col.add_renderable(matchup_to_tab(matchups[candidate], True,
            #                                   highlight_idxs=diff_idxs))
            # col.add_renderable(Align.center(Panel("Wird zu ⇩")))
            # col.add_renderable(matchup_to_tab(altered_original_matchup,
            #                                   highlight_idxs=diff_idxs))
            col.add_renderable(
                matchup_to_tab(
                    altered_original_matchup,
                    highlight_players=best_counterfactual_swap_idxs,
                )
            )

            # col.add_renderable(Panel("Der beste Tausch war unter den Spieler:innen selbst und hat Gesamtsumme erhöht."))

        else:
            other_changed_matchup = other_changed_matchup.pop()
            print(title, "Ausgangspairing:", names_and_cost(matchups[candidate]))
            print(
                title, "Verändertes Pairing:", names_and_cost(altered_original_matchup)
            )
            print(
                title,
                "Tausch-Pairing (original):",
                names_and_cost(matchups[other_changed_matchup]),
            )
            print(
                title,
                "Tausch-Pairing (verändert):",
                names_and_cost(best_counterfactual[other_changed_matchup]),
            )

            # diff_idx1 = find_diff(matchups[candidate], altered_original_matchup)
            # diff_idx2 = find_diff(
            #     matchups[other_changed_matchup],
            #     best_counterfactual[other_changed_matchup],
            # )

            persons_who_switched = [
                idx2name[idx] for idx in best_counterfactual_swap_idxs
            ]

            col.add_renderable(
                f"  {kosten_dict[field]} {man[0]} & {man[1]} vs {man[2]} & {man[3]}.\n"
                +
                # f"  Man könnte diesen speziellen Wert senken indem man {' und '.join(persons_who_switched)} tauscht.")
                f"  This specific cost could be lowered by swapping {' and '.join(persons_who_switched)}."
            )

            # col.add_renderable(matchup_to_tab(matchups[candidate], True, highlight_idx=diff_idx1))
            col.add_renderable(
                matchup_to_tab(
                    matchups[candidate],
                    True,
                    highlight_players=best_counterfactual_swap_idxs,
                )
            )
            # col.add_renderable(Align.center(Panel("Wird zu ⇩")))
            col.add_renderable(Align.center(Panel("becomes ⇩")))
            # col.add_renderable(matchup_to_tab(altered_original_matchup, highlight_idx=diff_idx1))
            col.add_renderable(
                matchup_to_tab(
                    altered_original_matchup,
                    highlight_players=best_counterfactual_swap_idxs,
                )
            )
            # col.add_renderable(Panel(f"So würde sich die Paarung von {persons_who_switched[1]} verändern:"))
            col.add_renderable(
                Panel(
                    f"{persons_who_switched[1]}s previous pairing would then look like this:"
                )
            )

            # col.add_renderable(matchup_to_tab(matchups[other_changed_matchup], highlight_idx=diff_idx2))
            col.add_renderable(
                matchup_to_tab(
                    matchups[other_changed_matchup],
                    highlight_players=best_counterfactual_swap_idxs,
                )
            )
            # col.add_renderable(Align.center(Panel("Wird zu ⇩")))
            col.add_renderable(Align.center(Panel("becomes ⇩")))
            # col.add_renderable(matchup_to_tab(best_counterfactual[other_changed_matchup], highlight_idx=diff_idx2))
            col.add_renderable(
                matchup_to_tab(
                    best_counterfactual[other_changed_matchup],
                    highlight_players=best_counterfactual_swap_idxs,
                )
            )

            # row_data = [pid2panel(idx2pid[idx], id2name) for idx in matchups[candidate]]
            # row_data.insert(2, "")
            # row_data.extend(format_initial_cost(matchups[candidate]))
            #
            # tab.add_row(*row_data)
            # col.add_renderable(tab)

        print(
            f"Die Kosten insgesamt steigen um {non_zero_round(100*best_counterfactual_cost/total_cost_original-100)}%"
        )
        # col.add_renderable(Panel(f"Das Tauschen von {' und '.join(persons_who_switched)} würde also die Gesamt-Kosten um {non_zero_round(100*best_counterfactual_cost/total_cost_original-100)}% erhöhen"))
        col.add_renderable(
            Panel(
                f"Swapping {' and '.join(persons_who_switched)} thus would increase the total cost by {non_zero_round(100*best_counterfactual_cost/total_cost_original-100)}%."
            )
        )
        # print(f"{best_counterfactual_cost=} {total_cost_original=}")
        print()

        entries.append(col)

    return entries
