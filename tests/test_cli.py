import io
import subprocess
import sys

import pytest

from newton_systems.cli import main


def run(argv: list[str]) -> tuple[int, str]:
    output = io.StringIO()

    return main(argv, stdout=output), output.getvalue()


def test_lists_examples() -> None:
    status, output = run(["list"])

    assert status == 0
    assert "tres-incognitas" in output
    assert "x0=(1.0, 3.0)" in output


def test_solves_with_an_iteration_table() -> None:
    status, output = run(["solve", "parabola-seno"])
    lines = output.splitlines()

    assert status == 0
    assert lines[1].startswith("  0  1 3")
    assert lines[-1] == "convergió en 3 iteraciones"


def test_method_start_and_options() -> None:
    status, output = run(
        ["solve", "circulo-recta", "-m", "broyden", "--x0", "3 1", "--numeric-jacobian"]
    )

    assert status == 0
    assert "convergió" in output.splitlines()[-1]


def test_exit_code_one_when_not_converging() -> None:
    status, output = run(["solve", "arcotangente", "-m", "broyden", "--no-line-search"])

    assert status == 1
    assert output.splitlines()[-1].startswith("no convergió")


@pytest.mark.parametrize(
    "argv",
    [
        ["solve", "arcotangente", "--no-line-search"],
        ["solve", "parabola-seno", "--x0", "1 2 3"],
        ["solve", "parabola-seno", "--x0", "uno dos"],
    ],
)
def test_errors_exit_with_code_two(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    assert run(argv) == (2, "")
    assert capsys.readouterr().err.startswith("error:")


def test_module_entry_point() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "newton_systems", "solve", "tres-incognitas"],
        capture_output=True,
        text=True,
        check=True,
    )

    assert completed.stdout.splitlines()[-1] == "convergió en 8 iteraciones"
