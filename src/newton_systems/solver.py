"""Newton and Broyden iterations for F(x) = 0 with F: R^n -> R^n."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

type Vector = NDArray[np.float64]
type Matrix = NDArray[np.float64]
type System = Callable[[Vector], ArrayLike]
type Jacobian = Callable[[Vector], ArrayLike]

ARMIJO_SLOPE = 1e-4
MIN_STEP_FRACTION = 1.0 / 1024


class SingularJacobianError(ArithmeticError):
    """Raised when the linear system for the Newton step cannot be solved."""


class DimensionError(ValueError):
    """Raised when F, its Jacobian and the starting point disagree in size."""


@dataclass(frozen=True, slots=True)
class Options:
    tolerance: float = 1e-12
    step_tolerance: float = 1e-14
    max_iterations: int = 50
    line_search: bool = True


@dataclass(frozen=True, slots=True)
class Iterate:
    iteration: int
    x: Vector
    residual_norm: float
    step_norm: float
    damping: float


@dataclass(frozen=True, slots=True)
class Result:
    x: Vector
    converged: bool
    iterations: int
    residual_norm: float
    history: list[Iterate] = field(repr=False)


def _evaluate(system: System, x: Vector) -> Vector:
    value = np.asarray(system(x), dtype=np.float64).reshape(-1)
    if value.shape != x.shape:
        raise DimensionError(f"F returned {value.size} values for {x.size} unknowns")

    return value


def numerical_jacobian(system: System, x: ArrayLike) -> Matrix:
    """Central differences with h_j = eps^(1/3) max(1, |x_j|), error O(h^2)."""
    point = np.asarray(x, dtype=np.float64).reshape(-1)
    columns = []
    for index in range(point.size):
        h = np.finfo(np.float64).eps ** (1 / 3) * max(1.0, abs(point[index]))
        offset = np.zeros_like(point)
        offset[index] = h
        columns.append(
            (_evaluate(system, point + offset) - _evaluate(system, point - offset)) / (2 * h)
        )

    return np.column_stack(columns)


def _jacobian_at(system: System, jacobian: Jacobian | None, x: Vector) -> Matrix:
    matrix = (
        np.asarray(jacobian(x), dtype=np.float64)
        if jacobian is not None
        else numerical_jacobian(system, x)
    )
    if matrix.shape != (x.size, x.size):
        raise DimensionError(f"Jacobian has shape {matrix.shape}, expected {(x.size, x.size)}")

    return matrix


def _solve_step(matrix: Matrix, residual: Vector) -> Vector:
    try:
        step = np.linalg.solve(matrix, -residual)
    except np.linalg.LinAlgError as error:
        raise SingularJacobianError("singular Jacobian, try another starting point") from error
    if not np.all(np.isfinite(step)):
        raise SingularJacobianError("Newton step is not finite")

    return step


def _damped(system: System, x: Vector, step: Vector, norm: float, options: Options) -> float:
    """Backtracking on phi(x) = ||F(x)||^2 / 2 with the Armijo condition."""
    if not options.line_search:
        return 1.0
    fraction = 1.0
    while fraction > MIN_STEP_FRACTION:
        trial = _evaluate(system, x + fraction * step)
        if (
            np.all(np.isfinite(trial))
            and np.linalg.norm(trial) ** 2 <= (1 - 2 * ARMIJO_SLOPE * fraction) * norm**2
        ):
            return fraction
        fraction /= 2

    return fraction


def _iterate(
    system: System,
    x0: ArrayLike,
    options: Options,
    direction: Callable[[Vector, Vector], Vector],
    update: Callable[[Vector, Vector], None],
) -> Result:
    x = np.asarray(x0, dtype=np.float64).reshape(-1).copy()
    residual = _evaluate(system, x)
    norm = float(np.linalg.norm(residual, np.inf))
    history = [Iterate(0, x.copy(), norm, 0.0, 1.0)]
    for iteration in range(1, options.max_iterations + 1):
        if norm <= options.tolerance:
            return Result(x, True, iteration - 1, norm, history)
        step = direction(x, residual)
        damping = _damped(system, x, step, float(np.linalg.norm(residual)), options)
        x_new = x + damping * step
        residual_new = _evaluate(system, x_new)
        update(x_new - x, residual_new - residual)
        x, residual = x_new, residual_new
        norm = float(np.linalg.norm(residual, np.inf))
        step_norm = float(np.linalg.norm(damping * step, np.inf))
        history.append(Iterate(iteration, x.copy(), norm, step_norm, damping))
        if step_norm <= options.step_tolerance * max(1.0, float(np.linalg.norm(x, np.inf))):
            return Result(x, norm <= options.tolerance, iteration, norm, history)

    return Result(x, norm <= options.tolerance, options.max_iterations, norm, history)


def newton(
    system: System,
    x0: ArrayLike,
    jacobian: Jacobian | None = None,
    options: Options | None = None,
) -> Result:
    """Solves J(x_k) s_k = -F(x_k), x_(k+1) = x_k + t_k s_k.

    Without a Jacobian, central differences approximate it. Convergence is
    quadratic near a simple root; the line search widens the basin.
    """

    def direction(x: Vector, residual: Vector) -> Vector:
        return _solve_step(_jacobian_at(system, jacobian, x), residual)

    def no_update(_step: Vector, _change: Vector) -> None:
        return None

    return _iterate(system, x0, options or Options(), direction, no_update)


def broyden(
    system: System,
    x0: ArrayLike,
    jacobian: Jacobian | None = None,
    options: Options | None = None,
) -> Result:
    """Good Broyden method: one Jacobian at the start, then rank one updates

    B_(k+1) = B_k + (y_k - B_k s_k) s_k^T / (s_k^T s_k),

    so each iteration costs a single evaluation of F. Convergence is
    superlinear instead of quadratic.
    """
    start = np.asarray(x0, dtype=np.float64).reshape(-1)
    approximation = [_jacobian_at(system, jacobian, start)]

    def direction(_x: Vector, residual: Vector) -> Vector:
        return _solve_step(approximation[0], residual)

    def update(step: Vector, change: Vector) -> None:
        denominator = float(step @ step)
        if denominator > 0:
            current = approximation[0]
            approximation[0] = current + np.outer(change - current @ step, step) / denominator

    return _iterate(system, start, options or Options(), direction, update)
