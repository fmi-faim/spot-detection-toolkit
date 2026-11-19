from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import tifffile

from spot_detection_toolkit.detectors import (
    TOOLS,
    Detector,
    detect,
    make_detector,
    resolve_model,
    results_dirname,
)
from spot_detection_toolkit.utils import (
    MODES,
    given_options,
    list_images,
    setup_logging,
    write_run_info,
)

log = logging.getLogger(__name__)


def predict_folder(
    detector: Detector,
    images: list[Path],
    mode: str,
    threshold: float | None,
    output_dir: str | Path,
) -> None:
    """Detect the spots of each image and write one CSV per image.

    Args:
        detector: detector built by make_detector.
        images: TIFF images to predict.
        mode: "2d", "2d_stack" or "3d".
        threshold: detection threshold. None: the detector's default.
        output_dir: where <image stem>.csv files are written (created if needed).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for tif in images:
        spots = detect(detector, tifffile.imread(tif), mode, threshold)
        spots.to_csv(output_dir / f"{tif.stem}.csv", index=False)


def run_inference(
    tool: str,
    input_dir: str | Path,
    output_root: str | Path,
    mode: str,
    model_path: str | Path | None = None,
    threshold: float | None = None,
    detector_kwargs: dict[str, Any] | None = None,
) -> Path:
    """Run a tool on a folder of images and write one CSV per image, plus run_info.json.

    Args:
        tool: tool name.
        input_dir: folder of .tif images.
        output_root: the CSVs go to <output_root>/<results_dirname(tool, model_path)>.
        mode: "2d", "2d_stack" or "3d".
        model_path: model to use (file, folder or name). Default: the tool's
            default model, downloaded on first use.
        threshold: detection threshold. Default: the tool's default threshold.
        detector_kwargs: settings of the detector class (see make_detector).

    Returns:
        Folder of the CSVs.
    """
    output_dir = Path(output_root) / results_dirname(tool, model_path)
    if any(output_dir.glob("*.csv")):
        raise FileExistsError(
            f"{output_dir} already holds results: choose another --output_root, "
            "or delete that folder to run again"
        )
    model_path = resolve_model(tool, mode, model_path)
    detector = make_detector(
        tool,
        mode,
        model_path=model_path,
        **(detector_kwargs or {}),
    )
    images = list_images(input_dir)
    if not images:
        raise FileNotFoundError(f"No .tif images found in {input_dir}")
    used_threshold = threshold if threshold is not None else detector.default_threshold
    log.info(
        f"Running {tool} on {len(images)} images in {input_dir} -> {output_dir}. "
        f"Mode: {mode}, model: {model_path or 'none'}, "
        f"threshold: {used_threshold:.3g}, settings: {detector_kwargs or 'defaults'}."
    )
    predict_folder(detector, images, mode, threshold, output_dir)
    write_run_info(
        output_dir,
        script="run_inference",
        tool=tool,
        mode=mode,
        model=model_path,
        threshold=used_threshold,
        settings=detector_kwargs or {},
        input_dir=Path(input_dir).resolve(),
        n_images=len(images),
    )
    return output_dir


def add_detection_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    g = parser.add_argument_group("detection mode and 3D parameters")
    g.add_argument(
        "--mode",
        choices=MODES,
        default="2d",
        help="2d: (y, x) images; 2d_stack: (n, y, x) independent planes. "
        "3d: (z, y, x) volumes",
    )
    g.add_argument(
        "--voxel_size",
        type=float,
        nargs=3,
        default=None,
        metavar=("Z", "Y", "X"),
        help="3D: voxel size (any unit, only ratios matter), used by LoG and, in "
        "grid_search.py, to scale z when scoring. Default: 1 1 1",
    )
    g.add_argument(
        "--lateral_sigma",
        type=float,
        default=None,
        help="LoG: lateral spot sigma in px. Default: 1.5",
    )
    g.add_argument(
        "--axial_sigma",
        type=float,
        default=None,
        help="LoG 3D: axial spot sigma in planes. Default: lateral_sigma * X / Z",
    )
    g.add_argument(
        "--no_ufish_blend_3d",
        action="store_true",
        help="U-FISH 3D: do not blend the z/y/x enhancements",
    )
    return parser


def detector_kwargs_from_args(tool: str, args: argparse.Namespace) -> dict[str, Any]:
    """Detector settings given on the command line, for make_detector.

    Args:
        tool: tool name. Only this tool's settings are kept.
        args: parsed arguments of add_detection_args.

    Returns:
        Keyword arguments of the tool's detector class.
    """
    kwargs = {}
    if tool == "log":
        kwargs = given_options(
            lateral_sigma=args.lateral_sigma,
            axial_sigma=args.axial_sigma,
            voxel_size=args.voxel_size,
        )
    elif tool == "ufish" and args.no_ufish_blend_3d:
        kwargs["blend_3d"] = False
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run inference using a specified tool on input images."
    )
    parser.add_argument("--tool", required=True, choices=TOOLS)
    parser.add_argument("--input_dir", required=True, help="Folder of .tif images")
    parser.add_argument(
        "--output_root",
        required=True,
        help="Root folder; CSVs go to <output_root>/results_<tool>_default "
        "(default model), results_<tool>_<model name> (--model) or results_log",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model file, folder or name. Omit to use the tool's default model, "
        "downloaded on first use.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Detection threshold. Omit to use the tool's default.",
    )
    add_detection_args(parser)

    args = parser.parse_args()
    setup_logging()
    run_inference(
        args.tool,
        args.input_dir,
        args.output_root,
        args.mode,
        model_path=args.model,
        threshold=args.threshold,
        detector_kwargs=detector_kwargs_from_args(args.tool, args),
    )


if __name__ == "__main__":
    main()
