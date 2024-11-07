import time
from random import shuffle

from find_matchups import TaskInput, TaskOutput
from matching_algos.base_matching_algo import BaseMatchingAlgo, MatchupCostCalculator


class RandomMatcher(BaseMatchingAlgo):
    def __init__(self, tries=None, runtime=None, optimize_sets=False):
        super().__init__("RandomMatcher")
        if tries is None and runtime is None:
            raise ValueError("Please set either the tries or runtime argument")
        if tries is not None and runtime is not None:
            raise ValueError("Please set only one of the tries or runtime arguments")

        self.tries = tries
        self.runtime = runtime
        self.optimize_sets = optimize_sets
        self.cost_calculator = None

    def _find_matching(self, task_input) -> TaskOutput:
        if self.tries:
            return self._find_matching_tries(task_input)
        if self.runtime:
            return self._find_matching_runtime(task_input)

        raise ValueError("Please set either the tries or runtime argument")

    def _opt(self, task_input, tup) -> list:
        if not self.optimize_sets:
            return tup

        if self.cost_calculator is None:
            self.cost_calculator = MatchupCostCalculator.from_taskinput(task_input)

        return self.cost_calculator.min_cost_for_tuple(tup)[1]

    def _find_matching_tries(self, task_input: TaskInput) -> TaskOutput:
        start_time = time.time()
        opt = lambda t: self._opt(task_input, t)

        min = float("inf")
        best_output = None
        for _ in range(self.tries):
            a = list(range(len(task_input.rating_list)))
            shuffle(a)

            output = TaskOutput(
                input=task_input,
                matchups=self._indices_to_player_ids(
                    [opt(a[i : i + 4]) for i in range(0, len(a), 4)], task_input
                ),
                players_to_pause=[],
                # cost_time=None
            )

            cost = lambda output: output.cost_quad.total

            if cost(output) < min:
                best_output = output
                min = cost(best_output)

        # best_output.cost_time = time.time() - start_time
        self.cost_calculator = None
        return best_output

    def _find_matching_runtime(self, task_input):
        start_time = time.time()
        opt = lambda t: self._opt(task_input, t)

        min = float("inf")
        best_output = None
        while time.time() - start_time < self.runtime or best_output is None:
            a = list(range(len(task_input.rating_list)))
            shuffle(a)

            output = TaskOutput(
                input=task_input,
                matchups=self._indices_to_player_ids(
                    [opt(a[i : i + 4]) for i in range(0, len(a), 4)], task_input
                ),
                players_to_pause=[],
                # cost_time=None
            )

            cost = lambda output: output.cost_quad.total

            if cost(output) < min:
                best_output = output
                min = cost(best_output)

        self.cost_calculator = None
        return best_output
