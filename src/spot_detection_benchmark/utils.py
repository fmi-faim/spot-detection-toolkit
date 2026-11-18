from __future__ import annotations

import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

MODES = ["2d", "2d_stack", "3d"]
IMAGE_EXTS = {".tif", ".tiff"}


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(level=level, format="%(message)s")


def get_device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def set_piscis_dir(piscis_dir: str | Path | None = None) -> tuple[Path, Path]:
    """Point Piscis to a folder holding its models/ and checkpoints/ folders.

    Piscis reads these folders from module constants. They are overwritten in
    every Piscis module that uses them. Both folders are created if needed.

    Args:
        piscis_dir: Piscis folder. Default: $PISCIS_DIR, else Piscis's own
            folder (~/.piscis).

    Returns:
        (models folder, checkpoints folder).
    """
    import piscis.convert
    import piscis.core
    import piscis.downloads
    import piscis.paths
    import piscis.training

    piscis_dir = Path(
        piscis_dir or os.environ.get("PISCIS_DIR", Path.home() / ".piscis")
    )
    models_dir = piscis_dir / "models"
    checkpoints_dir = piscis_dir / "checkpoints"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    piscis.paths.PISCIS_DIR = piscis_dir
    for module in (
        piscis.paths,
        piscis.core,
        piscis.training,
        piscis.downloads,
        piscis.convert,
    ):
        module.MODELS_DIR = models_dir
    for module in (piscis.paths, piscis.training):
        module.CHECKPOINTS_DIR = checkpoints_dir
    return models_dir, checkpoints_dir


def given_options(**options: Any) -> dict[str, Any]:
    """Keep only the options that were given.

    Args:
        **options: option values, None meaning "not given".

    Returns:
        The options whose value is not None.
    """
    kept = {}
    for name, value in options.items():
        if value is not None:
            kept[name] = value
    return kept


def list_images(folder: str | Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)


def subset_of(stem: str) -> str:
    return stem.split("_")[0]


def load_coords(csv: str | Path) -> np.ndarray:
    """Read the spot coordinates of a CSV.

    Args:
        csv: CSV with one axis-k column per axis.

    Returns:
        Coordinates, shape (N, D), columns in axis order.

    """
    df = pd.read_csv(csv, float_precision="round_trip")
    cols = sorted(
        (c for c in df.columns if c.startswith("axis-")),
        key=lambda c: int(c.split("-")[1]),
    )
    if not cols:
        raise ValueError(f"{csv}: no axis-* column")
    return df[cols].to_numpy(float) if len(df) else np.empty((0, len(cols)))


def write_run_info(output_dir: str | Path, **info: Any) -> Path:
    """Write the settings of a run, with the date, to output_dir/run_info.json.

    Args:
        output_dir: folder of the run's results.
        **info: settings to record.

    Returns:
        Path of the JSON file.
    """
    info = {
        **info,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    path = Path(output_dir) / "run_info.json"
    path.write_text(json.dumps(info, indent=2, default=str), encoding="utf-8")
    return path
