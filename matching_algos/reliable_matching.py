import concurrent.futures
import time
from concurrent.futures import (
    as_completed,
    ProcessPoolExecutor,
)

from matching_algos.base_matching_algo import BaseMatchingAlgo

from matching_algos.minizinc_matching import MinizincMatcher
from matching_algos.bruteforce_matching import BruteforceMatcher
from matching_algos.task_input_output import TaskInput, TaskOutput


class ReliableMatcher(BaseMatchingAlgo):
    def __init__(self):
        super().__init__("ReliableMatcher")
        self.bruteforce_first_window = 4.5
        self.minizinc_duration = 9.2

    def _find_matching(self, task_input: TaskInput) -> TaskOutput:
        import secrets

        secret = secrets.token_hex(4)

        start_time = time.time()

        bruteforcer = BruteforceMatcher()

        pool = ProcessPoolExecutor(max_workers=2)
        # pool = ThreadPoolExecutor(max_workers=2)

        future2name = dict()

        bruteforce_future = pool.submit(
            bruteforcer.find_matching,
            task_input,
            time_limit=self.bruteforce_first_window + self.minizinc_duration + 0.2,
            # time_limit=0.8
        )
        future2name[bruteforce_future] = "bruteforce"

        try:
            res = bruteforce_future.result(self.bruteforce_first_window)
            print(
                secret,
                "Bruteforce got it by itself",
                f"({time.time() - start_time:.2f}s)",
            )
            pool.shutdown(wait=False)
            return res
        except concurrent.futures.TimeoutError:
            pass

        # no result from bruteforcer after ~5 sec
        # -> start minizinc

        minizincer = MinizincMatcher(
            search_duration=self.minizinc_duration,
            viz_weight_matrices=False,
            verbose=False,
            log_tasks=False,
        )

        minizinc_future = pool.submit(minizincer.find_matching, task_input)
        print(secret, "Started Minizinc", f"({time.time()-start_time:.2f}s)")

        future2name[minizinc_future] = "minizinc"

        for res in as_completed([bruteforce_future, minizinc_future]):
            print(
                secret,
                f"Future '{future2name[res]}' finished:",
                res,
                f"({time.time()-start_time:.2f}s)",
            )
            if res.exception() is None:
                pool.shutdown(wait=False)
                return res.result()
