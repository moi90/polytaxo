"""CLI helpers for the PolyTaxo GUI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

from nicegui import ui

from .app import initialize_app_single


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="polytaxo-gui",
        description="PolyTaxo Annotation GUI",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Enable NiceGUI development behavior (e.g. auto-reload).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the application window.",
    )
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--dataset",
        type=Path,
        help="Path to a dataset folder for single-dataset mode.",
    )
    return parser


def validate_dataset_path(dataset: Path | None) -> Path | None:
    if dataset is None:
        return None

    normalized = dataset.expanduser()
    if not normalized.exists():
        raise ValueError(f"dataset path does not exist: {normalized}")
    if not normalized.is_dir():
        raise ValueError(f"dataset path is not a directory: {normalized}")

    return normalized.resolve()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # NiceGUI reload workers expect startup to be reachable from a module main.
    # Console-script entrypoints call this function directly, so we trampoline once.
    if args.dev and __name__ not in {"__main__", "__mp_main__"}:
        parser.error(
            "NiceGUI development mode requires running as a module, e.g. `python -m polytaxo_gui.cli`"
        )

    try:
        dataset = validate_dataset_path(args.dataset)
    except ValueError as exc:
        parser.error(str(exc))

    if args.dataset is not None:
        # Single-dataset mode
        initialize_app_single(dataset=dataset)

    ui.run(
        reload=args.dev,
        show=args.show,
    )
    return 0


if __name__ in {"__main__", "__mp_main__"}:
    # Wrapping in sys.exit() or raising SystemExit does not work.
    main()
