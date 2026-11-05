from __future__ import annotations

from pathlib import Path
from nicegui import ui


def initialize_app_single(*, dataset: Path | None) -> None:
    """Render a minimal first-screen shell for single-dataset mode."""

    ui.label("PolyTaxo GUI")
    ui.label("Single-dataset annotation mode")
    if dataset is None:
        ui.label("Dataset: not provided")
    else:
        ui.label(f"Dataset: {dataset}")
