"""Command-line entry point.

Kept intentionally thin: wire real commands as the project grows. Excluded
from coverage in pyproject — logic belongs in importable, testable modules,
not in the CLI shell.
"""

from __future__ import annotations

import sys

from make_char_dataset import __version__
from make_char_dataset.config import get_settings
from make_char_dataset.observability import init_sentry


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    args = sys.argv[1:] if argv is None else argv

    if args and args[0] in {"-v", "--version"}:
        print(f"make_char_dataset {__version__}")
        return 0

    settings = get_settings()
    init_sentry()
    print(f"make_char_dataset {__version__} — environment: {settings.environment}")
    print("Replace this entry point with your real commands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
