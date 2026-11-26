from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from newton_systems import (
    EXAMPLES,
    DimensionError,
    Options,
    Result,
    SingularJacobianError,
    broyden,
    newton,
    numerical_jacobian,
)
from newton_systems.solver import Jacobian, System, Vector

THREE = EXAMPLES["tres-incognitas"]
SINE = EXAMPLES["parabola-seno"]


def test_solves_the_original_three_unknown_system() -> None:
    result = newton(THREE.system, THREE.x0, THREE.jacobian)

    assert result.converged
    np.testing.assert_allclose(result.x, [3.936375765277, 2.488867635853, 0.202881755247])
    assert np.max(np.abs(np.asarray(THREE.system(result.x)))) < 1e-12


def test_solves_the_original_sine_system_in_three_steps() -> None:
    result = newton(SINE.system, SINE.x0, SINE.jacobian)

    assert result.converged
    assert result.iterations == 3
    np.testing.assert_allclose(result.x, [1.099743712038, 2.899436232167])


def test_quadratic_convergence_near_the_root() -> None:
    circle = EXAMPLES["circulo-recta"]
    errors = [
        abs(state.x[0] - np.sqrt(2))
        for state in newton(circle.system, circle.x0, circle.jacobian).history
    ]
    ratios = [errors[k + 1] / errors[k] ** 2 for k in (2, 3)]

    np.testing.assert_allclose(ratios, 1 / (2 * np.sqrt(2)), rtol=0.01)


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_every_example_converges_with_both_methods(name: str) -> None:
    example = EXAMPLES[name]
    for method in (newton, broyden):
        result = method(example.system, example.x0, example.jacobian)
        assert result.converged, method.__name__
        assert result.residual_norm <= 1e-12


@pytest.mark.parametrize("name", sorted(EXAMPLES))
def test_numerical_jacobian_matches_the_analytic_one(name: str) -> None:
    example = EXAMPLES[name]
    point = np.asarray(example.x0) + 0.1

    np.testing.assert_allclose(
        numerical_jacobian(example.system, point),
        np.asarray(example.jacobian(point)),
        rtol=1e-8,
        atol=1e-8,
    )


def test_line_search_rescues_a_divergent_start() -> None:
    arctan = EXAMPLES["arcotangente"]

    assert newton(arctan.system, arctan.x0, arctan.jacobian).converged
    with pytest.raises(SingularJacobianError):
        newton(arctan.system, arctan.x0, arctan.jacobian, Options(line_search=False))


def test_full_steps_can_be_faster_on_rosenbrock() -> None:
    rosenbrock = EXAMPLES["rosenbrock"]
    damped = newton(rosenbrock.system, rosenbrock.x0, rosenbrock.jacobian)
    full = newton(rosenbrock.system, rosenbrock.x0, rosenbrock.jacobian, Options(line_search=False))

    assert full.iterations == 2 < damped.iterations


@settings(deadline=None)
@given(
    arrays(np.float64, (3, 3), elements=st.floats(-5, 5)),
    arrays(np.float64, 3, elements=st.floats(-5, 5)),
)
def test_affine_systems_converge_in_one_step(matrix: Vector, rhs: Vector) -> None:
    well_conditioned = matrix + 20 * np.eye(3)

    def system(x: Vector) -> Vector:
        return well_conditioned @ x - rhs

    result = newton(system, np.zeros(3), lambda _x: well_conditioned)

    assert result.iterations <= 1
    np.testing.assert_allclose(result.x, np.linalg.solve(well_conditioned, rhs), atol=1e-12)


def test_reports_non_convergence_within_the_budget() -> None:
    result = newton(THREE.system, THREE.x0, THREE.jacobian, Options(max_iterations=2))

    assert not result.converged
    assert result.iterations == 2
    assert len(result.history) == 3


@pytest.mark.parametrize("method", [newton, broyden])
def test_stops_when_steps_fall_below_rounding(method: Callable[..., Result]) -> None:
    circle = EXAMPLES["circulo-recta"]
    result = method(circle.system, (1e6, 1e6 + 1), circle.jacobian, Options(tolerance=0.0))

    assert not result.converged
    assert result.iterations < 50
    assert result.residual_norm < 1e-15


def test_gives_up_on_systems_without_real_roots() -> None:
    result = newton(lambda x: [x[0] ** 2 + 1], [0.5], options=Options(max_iterations=200))

    assert not result.converged


def test_singular_jacobian() -> None:
    with pytest.raises(SingularJacobianError, match="singular Jacobian"):
        newton(lambda x: [x[0] + x[1], 2 * x[0] + 2 * x[1] - 1], [0.0, 0.0])


def test_non_finite_step() -> None:
    with pytest.raises(SingularJacobianError, match="not finite"):
        newton(lambda x: [x[0]], [1.0], lambda _x: [[np.nan]])


@pytest.mark.parametrize(
    ("system", "jacobian", "message"),
    [
        (lambda x: [x[0]], None, "returned 1 values for 2 unknowns"),
        (lambda x: [x[0], x[1]], lambda _x: [[1.0]], r"expected \(2, 2\)"),
    ],
)
def test_dimension_mismatches(system: System, jacobian: Jacobian | None, message: str) -> None:
    with pytest.raises(DimensionError, match=message):
        newton(system, [1.0, 2.0], jacobian)


def test_broyden_skips_zero_steps() -> None:
    result = broyden(lambda x: [x[0] - 1], [1.0])

    assert result.converged
    assert result.iterations == 0
