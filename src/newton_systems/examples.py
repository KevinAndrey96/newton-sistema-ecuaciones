"""The two systems of the original 2020 coursework, plus classic test cases."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from newton_systems.solver import Jacobian, System, Vector

C = 24.0


@dataclass(frozen=True, slots=True)
class Example:
    name: str
    description: str
    system: System
    jacobian: Jacobian
    x0: tuple[float, ...]


def _three_unknowns(x: Vector) -> list[float]:
    x1, x2, x3 = x
    return [x1**2 + x2 * x3 - 16, x1 * x2 + x3 - 10, x1 + x2**2 + C * x3 - 15]


def _three_unknowns_jacobian(x: Vector) -> list[list[float]]:
    x1, x2, x3 = x
    return [[2 * x1, x3, x2], [x2, x1, 1.0], [1.0, 2 * x2, C]]


def _sine_curve(x: Vector) -> list[float]:
    x1, x2 = x
    return [x1**2 - x2 + 1.69, 20 * x1 * np.sin(x1) - x2 - 16.7]


def _sine_curve_jacobian(x: Vector) -> list[list[float]]:
    x1, _ = x
    return [[2 * x1, -1.0], [20 * np.sin(x1) + 20 * x1 * np.cos(x1), -1.0]]


def _circle_line(x: Vector) -> list[float]:
    x1, x2 = x
    return [x1**2 + x2**2 - 4, x1 - x2]


def _circle_line_jacobian(x: Vector) -> list[list[float]]:
    x1, x2 = x
    return [[2 * x1, 2 * x2], [1.0, -1.0]]


def _rosenbrock(x: Vector) -> list[float]:
    x1, x2 = x
    return [10 * (x2 - x1**2), 1 - x1]


def _rosenbrock_jacobian(x: Vector) -> list[list[float]]:
    x1, _ = x
    return [[-20 * x1, 10.0], [-1.0, 0.0]]


def _arctangent(x: Vector) -> list[float]:
    x1, x2 = x
    return [np.arctan(x1), np.arctan(x2) + x1 / 2]


def _arctangent_jacobian(x: Vector) -> list[list[float]]:
    x1, x2 = x
    return [[1 / (1 + x1**2), 0.0], [0.5, 1 / (1 + x2**2)]]


EXAMPLES = {
    example.name: example
    for example in (
        Example(
            "tres-incognitas",
            "x1^2 + x2 x3 = 16, x1 x2 + x3 = 10, x1 + x2^2 + 24 x3 = 15",
            _three_unknowns,
            _three_unknowns_jacobian,
            (1.8, 3.2, 4.2),
        ),
        Example(
            "parabola-seno",
            "x1^2 - x2 + 1.69 = 0, 20 x1 sin(x1) - x2 - 16.7 = 0",
            _sine_curve,
            _sine_curve_jacobian,
            (1.0, 3.0),
        ),
        Example(
            "circulo-recta",
            "x1^2 + x2^2 = 4, x1 = x2",
            _circle_line,
            _circle_line_jacobian,
            (1.0, 2.0),
        ),
        Example(
            "rosenbrock",
            "10 (x2 - x1^2) = 0, 1 - x1 = 0",
            _rosenbrock,
            _rosenbrock_jacobian,
            (-1.2, 1.0),
        ),
        Example(
            "arcotangente",
            "arctan(x1) = 0, arctan(x2) + x1 / 2 = 0",
            _arctangent,
            _arctangent_jacobian,
            (2.0, -3.0),
        ),
    )
}
