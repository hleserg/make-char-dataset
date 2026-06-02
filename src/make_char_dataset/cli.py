"""Command-line entry point: drive the make-char-dataset pipeline stages.

Kept thin (excluded from coverage in pyproject) — the dispatch/idempotency logic
lives in the importable, tested :mod:`make_char_dataset.orchestrate`. CLI flags
override ``.env``/environment settings (CLI > env) before the typed settings are
resolved.
"""

from __future__ import annotations

import argparse
import os
import sys

from make_char_dataset import __version__
from make_char_dataset.config import get_settings
from make_char_dataset.observability import init_sentry
from make_char_dataset.orchestrate import run_all, run_stage

_STAGE_COMMANDS = ("import", "generate", "clean", "caption", "run-all")
_NEEDS_EXPORT = {"import", "run-all"}


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser (import/run-all take an export_dir; others don't).

    Shared flags live on each subcommand (git-style: ``run-all <export> --force``).
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--workspace", default=None, help="Override APP_WORKSPACE.")
    common.add_argument("--trigger", default=None, help="Override APP_TRIGGER_TOKEN.")
    common.add_argument("--repeats", type=int, default=None, help="Override APP_DATASET_REPEATS.")
    common.add_argument("--force", action="store_true", help="Re-run stages even if complete.")

    parser = argparse.ArgumentParser(prog="make-char-dataset", description=__doc__)
    parser.add_argument(
        "-v", "--version", action="version", version=f"make_char_dataset {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for command in _STAGE_COMMANDS:
        stage_parser = sub.add_parser(command, parents=[common], help=f"Run the {command} stage.")
        if command in _NEEDS_EXPORT:
            stage_parser.add_argument("export_dir", help="Path to the create-char-passport export.")
    return parser


def _apply_overrides(args: argparse.Namespace) -> None:
    """Apply CLI overrides as env vars (CLI > env), then reset the settings cache."""
    overrides = {
        "APP_WORKSPACE": args.workspace,
        "APP_TRIGGER_TOKEN": args.trigger,
        "APP_DATASET_REPEATS": None if args.repeats is None else str(args.repeats),
    }
    for key, value in overrides.items():
        if value is not None:
            os.environ[key] = value
    get_settings.cache_clear()


def main(argv: list[str] | None = None) -> int:
    """Run the CLI. Returns a process exit code."""
    args = _build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    _apply_overrides(args)
    init_sentry()

    if args.command == "run-all":
        results = run_all(args.export_dir, force=args.force)
        print(f"run-all complete — stages run: {', '.join(results) or '(none enabled)'}")
    else:
        run_stage(args.command, export_dir=getattr(args, "export_dir", None), force=args.force)
        print(f"stage '{args.command}' complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
