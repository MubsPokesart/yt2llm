"""Pre-commit hook: run fast unit tests; tolerate empty test tree during bootstrap."""

from __future__ import annotations

import pathlib
import subprocess
import sys

NO_TESTS_COLLECTED = 5


def main() -> int:
    if not pathlib.Path("tests/unit").is_dir():
        return 0
    rc = subprocess.call(["uv", "run", "pytest", "tests/unit", "-q"])  # noqa: S607
    if rc == NO_TESTS_COLLECTED:
        return 0
    return rc


if __name__ == "__main__":
    sys.exit(main())
