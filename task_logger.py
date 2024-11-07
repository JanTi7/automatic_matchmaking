import time
import json
import socket

import pathlib
import dataclasses

from helper import get_timestamp as _get_timestamp

LOG_FOLDER = pathlib.Path("task_log")
LOG_FOLDER.mkdir(exist_ok=True)


class TaskLogger:
    def __init__(self):
        self.ts = _get_timestamp()

    def log_input(self, task_input):
        import dao

        filename = LOG_FOLDER / f"{dao.db_name.split('.')[0]}_{self.ts}.taskinput.json"
        filename.parent.mkdir(exist_ok=True)
        filename.write_text(
            json.dumps(dataclasses.asdict(task_input), indent=2, separators=(",", ": "))
        )

    def log_output(self, task_output):
        import dao

        # TODO: maybe add metadata like how long minizinc ran
        filename = (
            LOG_FOLDER
            / f"{dao.db_name.split('.')[0]}_{self.ts}_{socket.gethostname()}.taskres.json"
        )
        filename.write_text(
            json.dumps(
                dataclasses.asdict(task_output), indent=2, separators=(",", ": ")
            )
        )


def taskoutput_from_file(taskres_file):
    taskoutput_json = pathlib.Path(taskres_file).read_text()
    taskoutput_dict = json.loads(taskoutput_json)
    task_output = taskoutput_from_dict(taskoutput_dict)
    return task_output


def taskoutput_from_dict(taskoutput_dict):
    from matching_algos.task_input_output import TaskOutput, TaskInput

    taskinput_dict = taskoutput_dict.pop("input")
    taskoutput_dict["input"] = TaskInput(**taskinput_dict)

    del taskoutput_dict["cost"]
    del taskoutput_dict["cost_quad"]
    cost_time = taskoutput_dict.pop("cost_time")
    task_output = TaskOutput(**taskoutput_dict)
    task_output.cost_time = cost_time

    return task_output


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()

    parser.add_argument("task_file")
    args = parser.parse_args()

    if "taskinput" in args.task_file:
        print(f"Viewing Taskinput {args.task_file}:")
        taskinput_json = pathlib.Path(args.task_file).read_text()
        taskinput_dict = json.loads(taskinput_json)
        from rich import print_json

        print_json(taskinput_json)

    elif "taskres" in args.task_file:
        from matching_algos.base_matching_algo import MatchupCostCalculator

        print(f"Viewing TaskResult {args.task_file}:")
        taskoutput_json = pathlib.Path(args.task_file).read_text()
        taskoutput_dict = json.loads(taskoutput_json)

        task_output = taskoutput_from_dict(taskoutput_dict)

        cost_calc = MatchupCostCalculator.from_taskinput(task_output.input)
        cost_calc.print_detailed_cost_analysis(task_output)
