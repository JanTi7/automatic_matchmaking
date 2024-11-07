from configargparse import ArgumentParser


def get_parser(*default_config_files):
    parser = ArgumentParser(default_config_files=default_config_files)
    parser.add_argument("-c", "--config-file", is_config_file=True)
    parser.add_argument("--db", help="Path to the database file", default=None)
    parser.add_argument("--init-rating", type=int, default=1500)
    parser.add_argument("--init-rd", type=int, default=125)
    parser.add_argument(
        "--pause-mode", choices=["random", "lowest_first"], default="random"
    )
    parser.add_argument("--rubberband", action="store_true")
    parser.add_argument(
        "--rubberband-deviation",
        type=int,
        default=200,
        help="The rubberband effect (slowly!) starts kicking in at 1500 +- (value)",
    )

    parser.add_argument("--higher_rating_weight", type=float, default=2.0)

    parser.add_argument(
        "-m",
        "--matching-algo",
        choices=[
            "default",
            "random",
            "interactive",
            "scipy",
            "studientermin1",
            "studientermin2",
            "studientermin3",
            "studientermin4",
        ],
        default="default",
        # required=True
    )

    return parser
