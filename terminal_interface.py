from terminal_interface_gameblock_management import (
    load_unfinished_block_interactive,
    manage_the_block,
    ReturnCode,
)
from terminal_interface_pool_management import manage_the_pool

import logging

if __name__ == "__main__":
    from parser import get_parser

    # the parser for all the internal stuff
    parser = get_parser()

    # the args only relevant for this use case
    # parser.add_argument("kursfile", is_config_file=True)
    # parser.add_argument("--c2", is_config_file=True)
    parser.add_argument("--copy-db", action="store_true")
    parser.add_argument("--copy-db-no-del", action="store_true")
    # parser.add_argument("--start-from-excel")
    args = parser.parse_args()

    parser.print_values()

    try:
        import pyfiglet

        pyfiglet.print_figlet(f"Using matching algo {repr(args.matching_algo)}")
        print()
    except:
        print(f"Using matching algo {args.matching_algo}")
        print()

    if args.copy_db or args.copy_db_no_del:
        import shutil
        import pathlib
        import secrets

        DB_DIR = pathlib.Path("databases")
        if args.db is None:
            ORIG_DEFAULT_CONTENT = (DB_DIR / ".default").read_text().strip()
            FILE_TO_LOAD = DB_DIR / ORIG_DEFAULT_CONTENT
        else:
            FILE_TO_LOAD = DB_DIR / args.db

        TMP_FILE = DB_DIR / f"{secrets.token_hex(8)}.json"

        shutil.copy(FILE_TO_LOAD, TMP_FILE)

        from dao import use_database

        use_database(TMP_FILE.name)

        if not args.copy_db_no_del:
            # Delete the copy when we are finished
            def restore():
                TMP_FILE.unlink()

            import atexit

            atexit.register(restore)
    else:
        if args.db is None:
            from dao import load_default_database

            load_default_database(create_new=True)
        else:
            from dao import use_database

            use_database(args.db)

    # All the way down here, bc the logger gets initialised in the use_database function
    logging.info(parser.format_values())

    unfinished_block = load_unfinished_block_interactive()
    if unfinished_block is not None:
        return_code = manage_the_block(unfinished_block)
        if return_code == ReturnCode.EXIT:
            import sys

            sys.exit()

    while True:
        block_of_games = manage_the_pool(conf=args)
        if block_of_games is None:
            break

        return_code = manage_the_block(block_of_games)

        if return_code == ReturnCode.EXIT:
            break  # or sys.exit()
