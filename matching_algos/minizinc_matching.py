import logging

from find_matchups import TaskInput, TaskOutput, find_and_viz_solution
from matching_algos.base_matching_algo import BaseMatchingAlgo
from matching_algos.random_matching import RandomMatcher


class MinizincMatcher(BaseMatchingAlgo):
    def __init__(
        self,
        minizinc_file="matching_algos/minizinc_model.mzn",
        search_duration=20,
        log_tasks=True,
        viz_weight_matrices=True,
        verbose=True,
        add_to_model=None,
    ):
        super().__init__("MinizincMatcher")
        self.search_duration = search_duration
        self.viz_weight_matrices = viz_weight_matrices
        self.verbose = verbose
        self.minizinc_file = minizinc_file
        self.log_tasks = log_tasks
        self.add_to_model = add_to_model

    def _find_matching(self, task_input: TaskInput, raise_errors=False) -> TaskOutput:
        error = None
        for _ in range(2):
            try:
                if error:
                    logging.error(f"Trying taskinput: {task_input}")
                return find_and_viz_solution(
                    task_input=task_input,
                    search_duration=self.search_duration,
                    viz_weight_matrices=self.viz_weight_matrices,
                    verbose=self.verbose,
                    minizinc_file=self.minizinc_file,
                    log_tasks=self.log_tasks,
                    add_to_model=self.add_to_model,
                )
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                import traceback

                logging.error(
                    "Caught Minizinc Error: " + "".join(traceback.format_exception(e))
                )
                # logging.error(f"Task Input: {task_input}")
                error = e

        if raise_errors:
            raise error
        else:
            logging.error("Failed multiple times. Using RandomMatcher now.")
            return RandomMatcher(runtime=5, optimize_sets=True).find_matching(
                task_input
            )
