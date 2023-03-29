"""Test that CLI args do not affect script startup."""

import subprocess


def test_start_with_args():
    result = subprocess.run(
        "python3 tests/dummy_viz.py asdf",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert "error: unrecognized arguments" not in result.stderr.decode()
    assert "Opening trace in browser" in result.stdout.decode()
