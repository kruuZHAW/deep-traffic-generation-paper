# fmt: off
from argparse import ArgumentParser
from pathlib import Path

from traffic.core import Traffic
from traffic.core.projection import EuroPP
from traffic.data.datasets import landing_zurich_2019

# fmt: on
# n_samples


def cli_main() -> None:
    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument(
        "--samples",
        dest="n_samples",
        type=int,
        default=200,
    )
    parser.add_argument(
        "--jobs",
        dest="n_jobs",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--path",
        dest="path",
        type=str,
        default="./data/traffic.pkl",
    )
    parser.add_argument(
        "--dp",
        dest="douglas_peucker_coeff",
        type=float,
        default=None,
    )
    args = parser.parse_args()

    # ------------
    # preprocessing
    # ------------
    t: Traffic = (
        landing_zurich_2019.query("track==track")
        .assign_id()
        .resample(args.n_samples)
        .unwrap()
        .eval(max_workers=args.n_jobs, desc="")
    )

    t = t.compute_xy(projection=EuroPP())

    if args.douglas_peucker_coeff is not None:
        print("Simplification...")
        t = t.simplify(tolerance=1e3).eval(desc="")

    t = Traffic.from_flights(
        flight.assign(
            timedelta=lambda r: (r.timestamp - flight.start).apply(
                lambda t: t.total_seconds()
            )
        )
        for flight in t
    )

    t.to_pickle(Path(args.path))


if __name__ == "__main__":
    cli_main()
