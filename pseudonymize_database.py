from argparse import ArgumentParser

from dao import use_database

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "db", help="filename of the database without the database/ path"
    )

    args = parser.parse_args()

    use_database(args.db)

    from dao import get_all_players, save_to_db

    players = get_all_players()

    for player in players:
        player.last_name = ""
        player.first_name = player.player_id

        save_to_db(player, update_if_exists=True)
