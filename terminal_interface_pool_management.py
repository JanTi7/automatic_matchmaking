from prompt_toolkit import prompt
from prompt_toolkit.completion import NestedCompleter
from prompt_toolkit.shortcuts import confirm
from prompt_toolkit.key_binding import KeyBindings

from dao import (
    get_all_names,
    PlayerPool,
    get_player_from_id,
    add_new_player,
    load_player_pool_interactive,
)

import stats

bindings = KeyBindings()


@bindings.add("c-d")
def _(event):
    event.app.exit()


# TODO: Add switch_db argument


def make_completer(list_of_player_ids_in_pool, list_of_players_voluntarily_pausing):
    all_names, _ = get_all_names()
    all_names = set(all_names)

    already_in_pool = {
        f"{e.first_name} {e.last_name}"
        for e in [get_player_from_id(p_id) for p_id in list_of_player_ids_in_pool]
    }
    currently_paused = {
        f"{e.first_name} {e.last_name}"
        for e in [
            get_player_from_id(p_id) for p_id in list_of_players_voluntarily_pausing
        ]
    }
    return NestedCompleter.from_nested_dict(
        {
            "add": {name: None for name in all_names.difference(already_in_pool)},
            "add_all_players": None,
            "add_new": None,
            "remove": {name: None for name in already_in_pool},
            "remove_all": None,
            "start_next_round": None,
            "pause": {name: None for name in already_in_pool},
            "unpause": {name: None for name in currently_paused},
            "stats": {name: None for name in all_names},
            "preview_pause": None,
            "table": None,
            "table_full": None,
            "exit": None,
        }
    )


def manage_the_pool(conf):
    pool = load_player_pool_interactive()
    if pool is None:
        pool = PlayerPool()

    names, ids = get_all_names()
    name2id = dict(zip(names, ids))

    from layout import save_table_as_html

    save_table_as_html(pool.everybody())

    while True:
        pool.draw()

        text = prompt(
            "# ",
            completer=make_completer(
                pool.list_of_player_ids, pool.players_voluntarily_pausing
            ),
            key_bindings=bindings,
        )

        if text is None or text == "exit":
            break

        cmd_parts = text.split(" ", maxsplit=1)
        if len(cmd_parts) > 1:
            cmd, args = cmd_parts
        else:
            cmd = text
            args = None

        if cmd == "table":
            from viz import print_table

            print_table(pool.everybody())
        elif cmd == "table_full":
            from viz import print_full_table

            print_full_table()

        elif cmd == "add":
            if not args:
                continue

            if args in names:
                pool.add_player(name2id[args])

            else:
                print(
                    f"Unknown Player {repr(args)}. Use the add_new command to add new players."
                )
                continue

        elif cmd == "add_all_players":
            for pid in ids:
                if pid not in pool.everybody():
                    pool.add_player(pid)

        elif cmd == "add_new":
            add = confirm(
                f"Do you wish to add a new player with the name {repr(args)}?"
            )

            if not add:
                continue

            initial_rating = int(
                prompt("Initial Rating: ", default=str(conf.init_rating))
            )

            parts = args.split(" ")
            if len(parts) == 2:
                new_id = add_new_player(
                    parts[0], parts[1], init_rating=initial_rating, init_rd=conf.init_rd
                )

            else:
                name_with_comma = prompt(
                    "Please add a comma between first and last name: ", default=args
                )  # could add validator checks for exactly one comma
                first, last = name_with_comma.split(",", 1)
                new_id = add_new_player(
                    first, last, init_rating=initial_rating, init_rd=conf.init_rd
                )

            names.append(args)
            name2id[args] = new_id

            pool.add_player(new_id)

        elif cmd == "pause":
            if args in names:
                pool.pause_player(name2id[args])
            else:
                print(f"Unknown player {repr(args)}")
                continue

        elif cmd == "unpause":
            if args in names:
                pool.unpause_player(name2id[args])
            else:
                print(f"Unknown player {repr(args)}")
                continue

        elif cmd == "remove":
            if args not in names:
                print("Unknown name")
                continue

            p_id_to_remove = name2id[args]
            pool.remove_player(p_id_to_remove)

        elif cmd == "remove_all":
            if confirm("Do you really want to remove all players from the pool? "):
                pool.remove_all()

        elif cmd == "preview_pause":
            pool.preview_pause(conf.pause_mode)

        elif cmd == "start_next_round":
            gameblock, task_output = pool.start_next_round(
                conf.pause_mode,
                num_sets="interactive",
                higher_rating_weight=conf.higher_rating_weight,
                matching_algo=conf.matching_algo,
                return_task_output=True,
                log_task=True,
            )
            from layout import explanation_viz, save_table_as_html

            explanation_viz(task_output, gameblock)
            save_table_as_html(task_output.input.player_ids)

            return gameblock

        elif cmd == "stats":
            if "," in args:
                args, delta = args.split(",")
            else:
                delta = "10y"
            delta = stats.parse_delta(delta)

            if args in names:
                stats.print_all_games_of(name2id[args], delta)
            else:
                print(f"Unknown player {repr(args)}")
                continue

        else:
            print(f"Unknown command {repr(cmd)} .")

    return None  # This signals the mail loop that the user wants to exit
