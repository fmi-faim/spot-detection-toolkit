from __future__ import annotations

import argparse
import logging
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile

from spot_detection_benchmark.detectors import (
    TOOLS,
    Detector,
    detect,
    make_detector,
    resolve_model,
    results_dirname,
    run_name,
)
from spot_detection_benchmark.evaluate import MIN_TRUE, evaluate_folder, summarize
from spot_detection_benchmark.run_inference import (
    add_detection_args,
    detector_kwargs_from_args,
    predict_folder,
)
from spot_detection_benchmark.utils import (
    list_images,
    setup_logging,
    subset_of,
    write_run_info,
)

log = logging.getLogger(__name__)

THRESHOLD_GRIDS = {
    "spotiflow": np.round(np.arange(0.1, 1.05, 0.05), 3).tolist(),
    "piscis": np.round(np.arange(0.1, 1.25, 0.05), 3).tolist(),
    "ufish": np.round(np.arange(0.1, 1.05, 0.05), 3).tolist(),
    "log": np.unique(np.geomspace(1, 3000, 30).astype(int)).tolist(),
}

COLUMNS = [
    "threshold",
    "subset",
    "image",
    "TP",
    "FP",
    "FN",
    "n_true",
    "n_pred",
    "precision",
    "recall",
    "f1",
    "f1_integral",
    "distance_error",
]


def optimal_per_subset(sweep: pd.DataFrame) -> pd.DataFrame:
    """Threshold maximising mean F1-integral, for each image type.

    Args:
        sweep: per-image metrics with a threshold column (one row per
            threshold x image).

    Returns:
        One row per image type: subset, optimal_threshold, f1_integral_mean,
        precision_mean, recall_mean, n_images.
    """
    summary = summarize(sweep, by=["threshold", "subset"], add_all=False)
    best_idx = summary.groupby("subset")["f1_integral_mean"].idxmax()
    return summary.loc[
        best_idx,
        [
            "subset",
            "threshold",
            "f1_integral_mean",
            "precision_mean",
            "recall_mean",
            "n_images",
        ],
    ].rename(columns={"threshold": "optimal_threshold"})


def can_filter(detector: Detector, mode: str) -> bool:
    """Whether the sweep can predict once and filter the spots by score.

    True when keeping the spots with score > t of a prediction at a lower
    threshold gives the same spots as a prediction at t (see docs/grid_search.md).

    Args:
        detector: detector built by make_detector.
        mode: "2d", "2d_stack" or "3d".

    Returns:
        True if mode is in the detector's filter_modes.
    """
    return mode in getattr(detector, "filter_modes", ())


def predict_all_spots(
    detector: Detector,
    images: list[Path],
    mode: str,
    threshold: float,
    output_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """Predict every image once, at the lowest threshold of the sweep.

    Args:
        detector: detector built by make_detector.
        images: TIFF images to predict.
        mode: "2d", "2d_stack" or "3d".
        threshold: lowest threshold of the sweep.
        output_dir: where the <image stem>.csv files are written (all_spots/).

    Returns:
        Spots of each image, by image stem.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_spots = {}
    for tif in images:
        spots = detect(detector, tifffile.imread(tif), mode, threshold)
        spots.to_csv(output_dir / f"{tif.stem}.csv", index=False)
        all_spots[tif.stem] = spots
    return all_spots


def write_filtered(
    all_spots: dict[str, pd.DataFrame],
    score_name: str,
    threshold: float,
    output_dir: str | Path,
) -> None:
    """Write, for each image, the spots whose score is above threshold.

    Spots without score (NaN) are kept at every threshold.

    Args:
        all_spots: output of predict_all_spots.
        score_name: score column used to filter.
        threshold: the spots with score > threshold are kept.
        output_dir: where the <image stem>.csv files are written.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stem, spots in all_spots.items():
        score = spots[score_name]
        kept = spots[score.isna() | (score > threshold)]
        kept.to_csv(output_dir / f"{stem}.csv", index=False)


def grid_search(
    tool: str,
    val_dir: str | Path,
    gt_dir: str | Path,
    output_root: str | Path,
    mode: str,
    model_path: str | Path | None = None,
    thresholds: Sequence[float] | None = None,
    voxel_size: Sequence[float] | None = None,
    detector_kwargs: dict[str, Any] | None = None,
) -> tuple[dict[str, float], pd.DataFrame] | None:
    """Sweep a tool's threshold and keep the best one per image type.

    Writes to <output_root>/<run_name(tool, model_path)>/ the sweep, its summary, the
    optimal thresholds, the predictions at the optimal threshold of each image
    type (optimal/) and run_info.json (see docs/grid_search.md).

    Args:
        tool: tool name.
        val_dir: folder of .tif images.
        gt_dir: folder of ground-truth CSVs.
        output_root: root folder of the results.
        mode: "2d", "2d_stack" or "3d".
        model_path: model to use (file, folder or name). Default: the tool's
            default model, downloaded on first use.
        thresholds: thresholds to try. Default: THRESHOLD_GRIDS[tool].
        voxel_size: (z, y, x) voxel size, used in 3D to scale z when scoring.
        detector_kwargs: settings of the detector class (see make_detector).

    Returns:
        (optimal threshold by image type, per-image sweep), or None if no
        threshold gave results.

    Raises:
        FileNotFoundError: no .tif image in val_dir.
    """
    val_dir = Path(val_dir).resolve()
    gt_dir = Path(gt_dir).resolve()
    output_root = Path(output_root).resolve()
    tag = run_name(tool, model_path)
    subdir = results_dirname(tool, model_path)
    eval_kw = dict(
        voxel_size=voxel_size if mode == "3d" else None, stack=mode == "2d_stack"
    )

    model_path = resolve_model(tool, mode, model_path)
    detector = make_detector(
        tool,
        mode,
        model_path=model_path,
        **(detector_kwargs or {}),
    )
    images = list_images(val_dir)
    if not images:
        raise FileNotFoundError(f"No .tif images found in {val_dir}")
    thresholds = list(thresholds) if thresholds else THRESHOLD_GRIDS[tool]
    tool_out = output_root / tag
    tmp_dir = output_root / f"_tmp_{tag}"
    sweep_method = "filter" if can_filter(detector, mode) else "predict"

    log.info(
        f"[{tag}] {len(images)} images, mode {mode}, model {model_path or 'none'}, "
        f"sweeping {len(thresholds)} thresholds ({thresholds[0]} -> {thresholds[-1]}) "
        f"by {sweep_method}"
    )
    if sweep_method == "filter":
        lowest = min(thresholds)
        all_spots_dir = tool_out / "all_spots" / subdir
        if all_spots_dir.exists():
            shutil.rmtree(all_spots_dir, ignore_errors=True)
        log.info(f"[{tag}] predicting every image once at threshold={lowest}")
        all_spots = predict_all_spots(detector, images, mode, lowest, all_spots_dir)

    def predict_at(threshold: float) -> Path:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if sweep_method == "filter":
            write_filtered(all_spots, detector.score_name, threshold, tmp_dir / subdir)
        else:
            predict_folder(
                detector,
                images,
                mode,
                output_dir=tmp_dir / subdir,
                threshold=threshold,
            )
        return tmp_dir / subdir

    sweep_rows = []
    for threshold in thresholds:
        try:
            pred_dir = predict_at(threshold)
        except Exception:
            log.exception(f"  threshold={threshold} FAILED")  # message + traceback
            continue
        df = evaluate_folder(gt_dir, pred_dir, **eval_kw)
        if df.empty:
            log.info(f"  threshold={threshold} no scored images")
            continue
        df.insert(0, "threshold", threshold)
        sweep_rows.append(df)
        scored = df[df["n_true"] >= MIN_TRUE]
        log.info(
            f"  threshold={threshold} F1i={scored['f1_integral'].mean():.4f} "
            f"P={scored['precision'].mean():.4f} R={scored['recall'].mean():.4f}"
        )

    if not sweep_rows:
        log.error(f"[{tag}] no valid results - check errors above")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    tool_out.mkdir(parents=True, exist_ok=True)

    sweep = pd.concat(sweep_rows, ignore_index=True)
    sweep = sweep[COLUMNS].sort_values(["subset", "threshold", "image"])
    sweep.to_csv(tool_out / f"{tag}_threshold_sweep.csv", index=False)
    summarize(sweep, by=["threshold", "subset"]).to_csv(
        tool_out / f"{tag}_threshold_summary.csv", index=False
    )
    log.info(f"[{tag}] sweep saved to {tool_out}")
    log.info(
        f"[{tag}] {sweep['image'].nunique()} images x "
        f"{sweep['threshold'].nunique()} thresholds = {len(sweep)} rows "
        f"({sweep['subset'].nunique()} image types)"
    )

    best = optimal_per_subset(sweep)
    best.to_csv(tool_out / f"{tag}_optimal_thresholds.csv", index=False)
    log.info(f"[{tag}] optimal threshold per image type:")
    log.info(best.round(4).to_string(index=False))
    best_by_subset = dict(zip(best["subset"], best["optimal_threshold"]))

    optimal_out = tool_out / "optimal"
    if optimal_out.exists():
        shutil.rmtree(optimal_out, ignore_errors=True)
    final_dir = optimal_out / subdir
    final_dir.mkdir(parents=True, exist_ok=True)

    needed = sorted(set(best_by_subset.values()))
    log.info(
        f"[{tag}] writing the predictions at {len(needed)} distinct "
        f"optimal threshold(s): {needed}"
    )
    for threshold in needed:
        try:
            pred_dir = predict_at(threshold)
        except Exception:
            log.exception(f"  threshold={threshold} FAILED")
            continue
        wanted = {s for s, t in best_by_subset.items() if t == threshold}
        n_copied = 0
        for pcsv in sorted(pred_dir.glob("*.csv")):
            if subset_of(pcsv.stem) in wanted:
                shutil.copy2(pcsv, final_dir / pcsv.name)
                n_copied += 1
        log.info(f"  threshold={threshold} {n_copied} CSVs for {sorted(wanted)}")

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    provenance = pd.DataFrame(
        [
            {
                "image": p.stem,
                "subset": subset_of(p.stem),
                "threshold": best_by_subset.get(subset_of(p.stem), np.nan),
            }
            for p in sorted(final_dir.glob("*.csv"))
        ]
    )
    provenance.to_csv(optimal_out / f"{tag}_threshold_per_image.csv", index=False)
    write_run_info(
        tool_out,
        script="grid_search",
        tool=tool,
        mode=mode,
        model=model_path,
        thresholds=thresholds,
        sweep_method=sweep_method,
        all_spots_threshold=min(thresholds) if sweep_method == "filter" else None,
        optimal_thresholds=best_by_subset,
        settings=detector_kwargs or {},
        voxel_size=voxel_size,
        images_dir=val_dir,
        gt_dir=gt_dir,
        n_images=len(images),
    )
    return best_by_subset, sweep


def main() -> None:
    """Command-line entry point (see docs/arguments.md)."""
    parser = argparse.ArgumentParser(
        description="Grid search over detection thresholds on images with ground truth."
    )
    parser.add_argument(
        "--tool",
        required=True,
        choices=TOOLS,
    )
    parser.add_argument("--val_dir", required=True, help="Images folder")
    parser.add_argument("--gt_dir", required=True, help="Ground-truth CSVs folder")
    parser.add_argument(
        "--output_root", required=True, help="Root folder for sweep results"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model file, folder or name. Omit to use the tool's default model, "
        "downloaded on first use.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help="Thresholds to try (default: THRESHOLD_GRIDS[tool])",
    )
    add_detection_args(parser)
    args = parser.parse_args()
    setup_logging()

    grid_search(
        args.tool,
        args.val_dir,
        args.gt_dir,
        args.output_root,
        mode=args.mode,
        model_path=args.model,
        thresholds=args.thresholds,
        voxel_size=args.voxel_size,
        detector_kwargs=detector_kwargs_from_args(args.tool, args),
    )


if __name__ == "__main__":
    main()
