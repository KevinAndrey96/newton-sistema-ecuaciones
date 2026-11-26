"""Command line interface: list the bundled systems and solve them step by step."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import TextIO

from newton_systems.examples import EXAMPLES
from newton_systems.solver import (
    DimensionError,
    Options,
    Result,
    SingularJacobianError,
    broyden,
    newton,
)

METHODS = {"newton": newton, "broyden": broyden}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="newton-systems", description="Newton y Broyden para sistemas no lineales."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="listar los sistemas incluidos")
    solve = commands.add_parser("solve", help="resolver un sistema mostrando cada iteración")
    solve.add_argument("name", choices=sorted(EXAMPLES))
    solve.add_argument("--method", "-m", choices=sorted(METHODS), default="newton")
    solve.add_argument("--x0", help='punto inicial, por ejemplo "1 3"')
    solve.add_argument("--no-line-search", action="store_true", help="dar siempre el paso completo")
    solve.add_argument(
        "--numeric-jacobian", action="store_true", help="aproximar el jacobiano por diferencias"
    )

    return parser


def _report(result: Result, target: TextIO) -> None:
    target.write(f"{'k':>3}  {'x':<48}  {'||F||inf':>10}  {'paso':>10}  {'t':>6}\n")
    for state in result.history:
        point = " ".join(f"{value:.12g}" for value in state.x)
        target.write(
            f"{state.iteration:>3}  {point:<48}  {state.residual_norm:10.3e}  "
            f"{state.step_norm:10.3e}  {state.damping:6.4g}\n"
        )
    verdict = "convergió" if result.converged else "no convergió"
    target.write(f"{verdict} en {result.iterations} iteraciones\n")


def main(argv: Sequence[str] | None = None, stdout: TextIO | None = None) -> int:
    arguments = _parser().parse_args(argv)
    target = stdout or sys.stdout
    if arguments.command == "list":
        for example in EXAMPLES.values():
            target.write(f"{example.name:<16} x0={example.x0}  {example.description}\n")

        return 0
    example = EXAMPLES[arguments.name]
    try:
        x0 = [float(value) for value in arguments.x0.split()] if arguments.x0 else example.x0
        result = METHODS[arguments.method](
            example.system,
            x0,
            None if arguments.numeric_jacobian else example.jacobian,
            Options(line_search=not arguments.no_line_search),
        )
    except (SingularJacobianError, DimensionError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)

        return 2
    _report(result, target)

    return 0 if result.converged else 1
