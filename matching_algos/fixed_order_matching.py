import time

from find_matchups import TaskInput, TaskOutput
from matching_algos.base_matching_algo import BaseMatchingAlgo


class FixedOrderMatcher(BaseMatchingAlgo):
    def __init__(self):
        super().__init__("FixedOrderMatcher")

    def _find_matching(self, task_input: TaskInput) -> TaskOutput:
        start_time = time.time()
        a = list(range(len(task_input.rating_list)))
        return TaskOutput(
            input=task_input,
            matchups=self._indices_to_player_ids(
                [(x, x+3, x+1, x+2) for x in range(0, len(a), 4)], task_input),
            players_to_pause=[],
            # cost_time=time.time()-start_time
        )