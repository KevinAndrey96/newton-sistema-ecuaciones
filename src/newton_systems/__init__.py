"""Newton and Broyden methods for systems of nonlinear equations."""

from newton_systems.examples import EXAMPLES, Example
from newton_systems.solver import (
    DimensionError,
    Iterate,
    Options,
    Result,
    SingularJacobianError,
    broyden,
    newton,
    numerical_jacobian,
)

__all__ = [
    "EXAMPLES",
    "DimensionError",
    "Example",
    "Iterate",
    "Options",
    "Result",
    "SingularJacobianError",
    "broyden",
    "newton",
    "numerical_jacobian",
]
