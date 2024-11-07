import concurrent
import time

import itertools

from find_matchups import TaskInput, TaskOutput
from matching_algos.base_matching_algo import BaseMatchingAlgo, MatchupCostCalculator
from scipy.optimize import milp, LinearConstraint, Bounds

import numpy as np


def get_result(c, constraints, integrality, presolve, mip_rel_gap, time_limit):
    start_time = time.time()

    tl_dict = dict(time_limit=time_limit) if time_limit else dict()

    return milp(
        c=c,
        bounds=Bounds(0, 1),
        constraints=constraints,
        integrality=integrality,
        options=dict(presolve=presolve, mip_rel_gap=mip_rel_gap, **tl_dict),
    ), time.time() - start_time


def n_combinations(N):
    return len(list(itertools.combinations(range(N), 4)))


class BruteforceMatcher(BaseMatchingAlgo):
    def __init__(self, presolve=False, mip_rel_gap=0.1):
        super().__init__("BruteforceMatcher")
        self.presolve = presolve
        self.mip_rel_gap = mip_rel_gap

    def _find_matching(
        self, task_input: TaskInput, run_parallel=True, time_limit=None
    ) -> TaskOutput:
        # run parallel seems to be important for bug reasons
        start_time = time.time()
        cost_calc = MatchupCostCalculator.from_taskinput(task_input)

        N = len(task_input.rating_list)

        tuple_calc_start_time = time.time()
        all_parts = list()
        for tuple_of_four in itertools.combinations(range(N), 4):
            all_parts.append(cost_calc.min_cost_for_tuple(tuple_of_four))

        # print(f"[BruteforceMatcher] Calculating all possible base tuples took {time.time()-tuple_calc_start_time:.2f}s")

        all_parts = sorted(all_parts)
        # pprint(all_parts[:10])

        # x has 10626 elements

        c = np.array([cost.total for cost, matchup in all_parts])

        # if i don't sort all_parts, then A is constant
        A = np.array(
            [[int(pidx in matchup) for _, matchup in all_parts] for pidx in range(N)]
        )
        b_u = np.ones(N)
        b_l = np.ones(N)

        constraints = LinearConstraint(A, b_l, b_u)
        integrality = np.ones_like(c)

        if not run_parallel:
            tl_dict = dict(time_limit=float(time_limit)) if time_limit else dict()
            # print(f"{tl_dict=}")

            # print("started scipy")
            res = milp(
                c=c,
                bounds=Bounds(0, 1),
                constraints=constraints,
                integrality=integrality,
                options=dict(
                    presolve=self.presolve, mip_rel_gap=self.mip_rel_gap, **tl_dict
                ),
            )
            # print("scipy exited")

        else:
            from concurrent.futures import (
                ThreadPoolExecutor,
                as_completed,
                ProcessPoolExecutor,
            )

            combs = [
                (self.presolve, self.mip_rel_gap),
                # (False, 0),
                # (False, 0.1),
                # (True, 0),
                # (True, 0.1),
            ]

            # executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(combs))
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=len(combs))
            futures = {
                executor.submit(
                    get_result,
                    c,
                    constraints,
                    integrality,
                    presolve,
                    mip_rel_gap,
                    time_limit,
                ): (presolve, mip_rel_gap)
                for presolve, mip_rel_gap in combs
            }

            # for future in as_completed(futures):
            #     res, time_taken = future.result()
            #     print(f"conf: {futures[future]}", f"{time.time() - start_time:.2f}s, {time_taken=}")

            results = concurrent.futures.wait(
                futures, return_when=concurrent.futures.FIRST_COMPLETED
            )
            future = list(results.done)[0]
            res, time_taken = future.result()
            # print("Best conf:", futures[future], f"{time.time() - start_time:.2f}s, {time_taken=}")
            executor.shutdown(wait=False, cancel_futures=True)

        selected_matchups = np.array(np.rint(res.x), dtype=int)

        matchup_table = [
            matchup
            for idx, (_, matchup) in enumerate(all_parts)
            if selected_matchups[idx] == 1
        ]

        return TaskOutput(
            input=task_input,
            matchups=self._indices_to_player_ids(matchup_table, task_input),
            players_to_pause=[],
            # cost_time=time.time() - start_time
        )
