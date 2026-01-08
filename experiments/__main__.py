"""Client for running experiments."""

import argparse as ap
from dataclasses import asdict, dataclass
from typing import Literal, Never, assert_never, cast

from .mnist_experiment import MnistExperiment


@dataclass
class MnistArgs:
    num_patterns: int
    train: bool
    min: int
    max: int
    batch_size: int


def mnist_subparser_defaults(args: ap.Namespace) -> MnistArgs:
    """Wrapped constructor to be used in the argument parser."""
    return MnistArgs(args.num_patterns, args.train, args.min, args.max, args.batch_size)


@dataclass
class RandomArgs:
    dist: Literal["uniform"] | Literal["skew"] | Literal["corr"]
    value: float | None
    num_patterns: int
    pattern_dim: int
    dtype: str
    min: int
    max: int


def random_subparser_defaults(args: ap.Namespace) -> RandomArgs:
    """Wrapper constructor to be used in the argument parser."""
    return RandomArgs(
        args.distribution,
        args.value,
        args.num_patterns,
        args.pattern_dim,
        args.dtype,
        args.min,
        args.max,
    )


def ExperimentParser() -> ap.ArgumentParser:
    """Thin wrapper around `argparse.ArgumentParser`."""

    argparser = ap.ArgumentParser(
        prog="experiments",
        description="Simple terminal client for running experiments",
        epilog="🐈",
    )
    _ = argparser.add_argument(
        "-o",
        "--output",
        nargs=1,
        help="Default: stdout. Path of file to output results",
    )
    subparsers = argparser.add_subparsers(help="Differing experimental conditions")

    random_subparser = subparsers.add_parser(
        "rand", help="Generate and test random data"
    )
    _ = random_subparser.add_argument(
        "distribution",
        choices=["uniform", "skew", "corr"],
        help="Kind of distribution to generate",
    )
    _ = random_subparser.add_argument(
        "value",
        type=float,
        nargs="?",
        help="Optional value used as kurtosis or skew.",
    )
    _ = random_subparser.add_argument(
        "num_patterns", type=int, nargs=1, help="Number of patterns to generate."
    )
    _ = random_subparser.add_argument(
        "pattern_dim", type=int, nargs=1, help="Dimension of patterns to generate."
    )
    _ = random_subparser.add_argument(
        "-d",
        "--dtype",
        default="float32",
        type=str,
        help="Data type to use for the arrays.",
    )
    _ = random_subparser.add_argument(
        "--min",
        type=int,
        nargs=1,
        default=2,
        help="Default: 2. Initial number of patterns to store.",
    )
    _ = random_subparser.add_argument(
        "--max",
        type=int,
        nargs=1,
        default=100,
        help="Default: 100. Final number of patterns to store.",
    )
    random_subparser.set_defaults(func=random_subparser_defaults)

    mnist_subparser = subparsers.add_parser(
        "mnist", help="Run experiments on MNIST dataset. Mainly used for testing."
    )
    _ = mnist_subparser.add_argument(
        "num_patterns", type=int, help="The number of patterns to memorize."
    )
    _ = mnist_subparser.add_argument(
        "-t",
        "--train",
        default=True,
        action=ap.BooleanOptionalAction,
        help="Whether or not to sample from the training set.",
    )
    _ = mnist_subparser.add_argument(
        "--min",
        type=int,
        nargs=1,
        default=2,
        help="Default: 2. Initial number of patterns to store.",
    )
    _ = mnist_subparser.add_argument(
        "--max",
        type=int,
        nargs=1,
        default=100,
        help="Default: 100. Final number of patterns to store.",
    )
    _ = mnist_subparser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        nargs=1,
        default=123,
        help="Batch size to take from dataloader.",
    )
    mnist_subparser.set_defaults(func=mnist_subparser_defaults)

    return argparser


def mnist_experiment(args: MnistArgs) -> None:
    MnistExperiment(*tuple(asdict(args).values())).run()


def random_experiment(args: RandomArgs) -> None:
    raise NotImplementedError("Haven't implemented random yet")


def main() -> None:
    client = ExperimentParser()
    args = client.parse_args()
    wrapped_args = cast(MnistArgs | RandomArgs, args.func(args))
    if type(wrapped_args) == MnistArgs:
        mnist_experiment(wrapped_args)
    elif type(wrapped_args) == RandomArgs:
        random_experiment(wrapped_args)
    else:
        assert_never(cast(Never, wrapped_args))


if __name__ == "__main__":
    main()
