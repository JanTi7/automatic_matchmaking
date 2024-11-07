import time
from dataclasses import asdict
from prompt_toolkit import prompt
from prompt_toolkit.completion import NestedCompleter, FuzzyWordCompleter
from prompt_toolkit.shortcuts import confirm
from prompt_toolkit.key_binding import KeyBindings

from enum import Enum, auto

from dao import (
    save_to_db,
    load_from_db,
    get_player_from_id,
    GameProposed,
    GameResult,
    BlockOfGames,
    load_unfinished_blocks_of_games,
    remove_all_unfinished_blocks_of_games,
)


class ReturnCode(Enum):
    ERROR = (auto(),)
    FINISHED = (auto(),)
    EXIT = (auto(),)


bindings = KeyBindings()


@bindings.add("c-d")
def _(event):
    event.app.exit()


def load_unfinished_block_interactive():
    unfinished_blocks = load_unfinished_blocks_of_games()
    if len(unfinished_blocks) == 0:
        print("No unfinished blocks found!")
        return None

    from datetime import timedelta

    if len(unfinished_blocks) == 1:
        block = unfinished_blocks[0]

        time_passed = timedelta(seconds=time.time() - block.timestamp)
        if time_passed <= timedelta(minutes=30):
            return block

    print(f"Found {len(unfinished_blocks)} unfinished blocks of games:")
    for idx, block in enumerate(unfinished_blocks):
        time_passed = timedelta(seconds=time.time() - block.timestamp)

        print(f" {idx} ".center(60, "-"))
        print(f"Started {time_passed} ago.")
        block.draw()

    while True:
        choice = input("Do you wish to load any of these blocks? [idx/n/remove_all] ")
        if choice == "n":
            return None
        if choice == "remove_all":
            remove_all_unfinished_blocks_of_games()
            return None
        try:
            idx = int(choice)
            return unfinished_blocks[idx]
        except:
            print(
                f"Could not deal with input {repr(choice)}. Try 'n' or the block index"
            )


def make_completer(game_block: BlockOfGames):
    return NestedCompleter.from_nested_dict(
        {
            "enter_result": get_fuzzy_match_completer(game_block.proposed.items()),
            "correct_result": get_fuzzy_match_completer(game_block.results.items()),
            "cancel_single_game": get_fuzzy_match_completer(
                game_block.proposed.items()
            ),
            "cancel_unfinished_games": None,
            "uncancel_single_game": get_fuzzy_match_completer(
                game_block.cancelled.items()
            ),
            "uncancel_all_games": None,
            "commit": None,
            "exit": None,
        }
    )


def get_fuzzy_match_completer(dict_items):
    from viz import get_human_str_for_gamedict

    str2game_idx = dict()
    for idx, game_id in dict_items:
        game_dict = load_from_db(game_id)
        hstr = get_human_str_for_gamedict(game_dict, idx)
        str2game_idx[hstr] = idx

    return FuzzyWordCompleter(words=list(str2game_idx.keys()), meta_dict=str2game_idx)


def manage_the_block(game_block: BlockOfGames):
    while True:
        game_block.draw()
        text = prompt("# ", completer=make_completer(game_block), key_bindings=bindings)
        if text is None or text == "exit":
            return ReturnCode.EXIT

        return_code = process_command(game_block, text)
        if return_code is not None:
            return return_code


def process_command(game_block: BlockOfGames, command: str):
    cmd_parts = command.split(" ", maxsplit=1)
    if len(cmd_parts) > 1:
        cmd, args = cmd_parts
    else:
        cmd = command
        args = None

    if cmd == "cancel_unfinished_games":
        game_block.cancelled.update(game_block.proposed)
        game_block.proposed.clear()
        save_to_db(game_block, update_if_exists=True)

    elif cmd == "cancel_single_game":
        if (local_idx := extract_local_game_idx(args)) is None:
            return

        if local_idx not in game_block.proposed.keys():
            print("Keys:", game_block.proposed.keys())
            print(f"There is no running game with the idx {local_idx}")
            return

        game = game_block.proposed.pop(local_idx)
        game_block.cancelled[local_idx] = game
        save_to_db(game_block, update_if_exists=True)

    elif cmd == "uncancel_single_game":
        if (local_idx := extract_local_game_idx(args)) is None:
            return

        if local_idx not in game_block.cancelled.keys():
            print("Keys:", game_block.cancelled.keys())
            print(f"There is no cancelled game with the idx {local_idx}")
            return

        game = game_block.cancelled.pop(local_idx)
        game_block.proposed[local_idx] = game
        save_to_db(game_block, update_if_exists=True)

    elif cmd == "enter_result":
        if (local_idx := extract_local_game_idx(args)) is None:
            return

        if local_idx not in game_block.proposed.keys():
            print("Keys:", game_block.proposed.keys())
            print(f"There is no running game with the idx {local_idx}")
            return

        game = GameProposed(**load_from_db(game_block.proposed[local_idx]))
        try:
            game.draw(local_idx)
            scores = input("Please enter the result (separated by a space): ")

            try:
                points_a, points_b = scores.split(" ")

                p_a = int(points_a)
                p_b = int(points_b)

            except:
                print("Error parsing your input.")
                return

            if max(p_a, p_b) < 15:
                if not confirm(
                    "No team got 15 or more points. Sure these are the right results?"
                ):
                    return

            if max(p_a, p_b) > 21 and abs(p_a - p_b) > 2:
                if not confirm(
                    "One team got alot more points than the other. Sure these are the right results?"
                ):
                    return

            game_block.register_result(local_idx, points_a, points_b)

        except KeyboardInterrupt:
            print("Aborting..")
            return

    elif cmd == "correct_result":
        if (local_idx := extract_local_game_idx(args)) is None:
            return

        if local_idx not in game_block.results.keys():
            print("Keys:", game_block.results.keys())
            print(f"There is no game with results with the idx {local_idx}")
            return

        game_block.unregister_result(local_idx)

        process_command(game_block, f"enter_result {local_idx}")

    elif cmd == "commit":
        if len(game_block.proposed) > 0:
            print("All games have to be finished or cancelled before committing!")
            return

        game_block.commit()

        ## ARE WE DONE??

        return ReturnCode.FINISHED

    else:
        print("Unknown command.")


def extract_local_game_idx(args):
    try:
        if "<" in args:
            i1, i2 = args.find("<"), args.find(">")
            local_idx = int(args[i1 + 1 : i2])
        else:
            local_idx = int(args)
    except:
        print(f"Error parsing the arg {repr(args)}")
        return None

    return str(local_idx)  # probably a db thing
