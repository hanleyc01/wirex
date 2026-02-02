"""Client for running experiments."""

import argparse as ap
from dataclasses import dataclass
from typing import Literal, cast

from .mnist_experiment import MnistExperiment


@dataclass
class MnistArgs:
    train: bool
    min: int
    max: int
    batch_size: int
    output: str | None
    num_models: int


def mnist_subparser_defaults(args: ap.Namespace) -> MnistArgs:
    """Wrapped constructor to be used in the argument parser."""
    output = args.output[0] if type(args.output) == list else None
    return MnistArgs(
        args.train,
        args.min[0],
        args.max[0],
        args.batch_size[0],
        output,
        args.num_models[0],
    )


@dataclass
class RandomArgs:
    dist: Literal["uniform"] | Literal["skew"] | Literal["corr"]
    value: float | None
    pattern_dim: int
    dtype: str
    min: int
    max: int
    output: str | None
    num_models: int


def random_subparser_defaults(args: ap.Namespace) -> RandomArgs:
    """Wrapper constructor to be used in the argument parser."""
    output = args.output[0] if type(args.output) == list else None
    return RandomArgs(
        args.distribution,
        args.value,
        args.pattern_dim,
        args.dtype,
        args.min,
        args.max,
        output,
        args.num_models[0],
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
    _ = argparser.add_argument(
        "-n",
        "--num_models",
        nargs=1,
        default=[10],
        help="The number of models to generate for the given task.",
    )
    argparser.set_defaults(func=lambda _: None)
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
        default=[2],
        help="Default: 2. Initial number of patterns to store.",
    )
    _ = random_subparser.add_argument(
        "--max",
        type=int,
        nargs=1,
        default=[100],
        help="Default: 100. Final number of patterns to store.",
    )
    random_subparser.set_defaults(func=random_subparser_defaults)

    mnist_subparser = subparsers.add_parser(
        "mnist", help="Run experiments on MNIST dataset. Mainly used for testing."
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
        default=[2],
        help="Default: 2. Initial number of patterns to store.",
    )
    _ = mnist_subparser.add_argument(
        "--max",
        type=int,
        nargs=1,
        default=[100],
        help="Default: 100. Final number of patterns to store.",
    )
    _ = mnist_subparser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        nargs=1,
        default=[123],
        help="Batch size to take from dataloader.",
    )
    mnist_subparser.set_defaults(func=mnist_subparser_defaults)

    return argparser


def mnist_experiment(args: MnistArgs) -> None:
    experiment = MnistExperiment(
        train=args.train,
        min=args.min,
        max=args.max,
        output=args.output,
        batch_size=args.batch_size,
        num_models=args.num_models,
    )
    experiment.run()


def random_experiment(args: RandomArgs) -> None:
    raise NotImplementedError("Haven't implemented random yet")


def main() -> None:
    parser = ExperimentParser()
    args = parser.parse_args()
    wrapped_args = cast(MnistArgs | RandomArgs, args.func(args))
    if type(wrapped_args) == MnistArgs:
        mnist_experiment(wrapped_args)
    elif type(wrapped_args) == RandomArgs:
        random_experiment(wrapped_args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
