from datetime import timedelta

import numpy as np
import itertools
import time

from minizinc import Instance, Model, Solver

from find_matchups import MinizincNotFoundError
from matching_algos.base_matching_algo import BaseMatchingAlgo, MatchupCostCalculator
from matching_algos.task_input_output import TaskOutput, TaskInput


class BruteforceMatcherMinizinc(BaseMatchingAlgo):
    def __init__(self, solver="coin-bc", timeout=10):
        super().__init__("BruteforceMatcherMinizinc")
        self.solver = solver
        self.timeout = timeout

    def _find_matching(self, task_input: TaskInput, *args, **kwargs) -> TaskOutput:
        start_time = time.time()
        cost_calc = MatchupCostCalculator.from_taskinput(task_input)

        N = len(task_input.rating_list)

        tuple_calc_start_time = time.time()
        all_parts = list()
        for tuple_of_four in itertools.combinations(range(N), 4):
            all_parts.append(cost_calc.min_cost_for_tuple(tuple_of_four))

        # print(
        #     f"[BruteforceMatcherMinizinc] Calculating all possible base tuples took {time.time() - tuple_calc_start_time:.2f}s")

        all_parts = sorted(all_parts)

        # x has 10626 elements

        c = np.array([cost.total for cost, matchup in all_parts])

        # if i don't sort all_parts, then A is constant
        A = np.array([[int(pidx in matchup) for _, matchup in all_parts] for pidx in range(N)])

        try:
            solver = Solver.lookup(self.solver)
        except AssertionError:
            raise MinizincNotFoundError()

        model = Model()
        model.add_file("matching_algos/minizinc_ilp.mzn")

        instance = Instance(solver, model)
        instance["num_players"] = N
        instance["num_tuples"] = len(all_parts)
        instance["mat"] = A.T
        instance["cost"] = c

        result = instance.solve(
            timeout=timedelta(seconds=max(10.0,
                self.timeout-(time.time()-start_time))),
        )

        x = result.solution.x

        selected_matchups = np.array(np.rint(result.solution.x), dtype=int)


        matchup_table = [matchup for idx, (_, matchup) in enumerate(all_parts) if selected_matchups[idx] == 1]


        return TaskOutput(
            input=task_input,
            matchups=self._indices_to_player_ids(matchup_table, task_input),
            players_to_pause=[],
        )



