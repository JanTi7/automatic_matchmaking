from matching_algos.base_matching_algo import BaseMatchingAlgo
from matching_algos.task_input_output import TaskInput, TaskOutput

from pprint import pprint

class InteractiveMatcher(BaseMatchingAlgo):
    def __init__(self):
        super().__init__("InteractiveMatcher")

    def _find_matching(self, task_input: TaskInput, *args, **kwargs) -> TaskOutput:
        from dao import generate_playerid_to_uniquename_map

        id2name = generate_playerid_to_uniquename_map(task_input.player_ids)
        idx2name = {idx: id2name[pid] for idx, pid in enumerate(task_input.player_ids)}

        teamless_players = idx2name.copy()

        matchups = list()

        def print_matchups():
            for matchup in matchups:
                a, b, c, d = matchup
                print(f"{idx2name[a]} & {idx2name[b]} v {idx2name[c]} & {idx2name[d]}")


        current_matchup = list()
        while len(teamless_players) > 0:
            if len(teamless_players) == 2:
                current_matchup.extend(teamless_players.keys())
                matchups.append(current_matchup)
                current_matchup = list()
                break

            print_matchups()
            print()
            print("Currenty Teamless:")
            pprint(teamless_players)

            while True:
                next_person = int(input(f"Who is next in this matchup? {[idx2name[idx] for idx in current_matchup]}"))
                if next_person in teamless_players.keys():
                    teamless_players.pop(next_person)
                    current_matchup.append(next_person)

                    if len(current_matchup) == 4:
                        matchups.append(current_matchup)
                        current_matchup = list()

                    break

        return TaskOutput(
            input=task_input,
            matchups=self._indices_to_player_ids(
                matchups, task_input),
            players_to_pause=[],
        )

