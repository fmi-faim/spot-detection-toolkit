from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path
from typing import Any

from spot_detection_benchmark.detectors import (
    load_spotiflow,
    load_ufish_weights,
)
from spot_detection_benchmark.utils import (
    get_device,
    given_options,
    set_piscis_dir,
    setup_logging,
)

log = logging.getLogger(__name__)

DEFAULT_PISCIS_MODEL_NAME = "piscis_customed_512"
DEFAULT_PISCIS_INPUT_SIZE = (512, 512)
DEFAULT_UFISH_HPARAMS = {"lr": 1e-4, "batch_size": 16, "num_epochs": 200}


def _log_gpu(tool: str) -> None:
    import torch

    on_gpu = torch.cuda.is_available()
    log.info(f"[{tool}] training on {'GPU' if on_gpu else 'CPU'}")


def train_spotiflow(
    input_dir: str | Path | None,
    output_dir: str | Path | None,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    base_model: str | Path | None = None,
) -> None:
    """Train a Spotiflow model.

    Args:
        input_dir: folder with train/ and val/ (images and CSVs).
        output_dir: model folder.
        epochs: number of epochs. Default: Spotiflow's.
        lr: learning rate. Default: Spotiflow's.
        batch_size: batch size. Default: Spotiflow's.
        base_model: Spotiflow model folder or pretrained model name to start
            from for fine-tuning (see load_spotiflow).
    """
    if input_dir is None or output_dir is None:
        raise ValueError("spotiflow requires --input_dir and --output_dir")

    from spotiflow.model import Spotiflow
    from spotiflow.utils import get_data

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_imgs, train_spots, val_imgs, val_spots = get_data(input_dir)

    train_config = given_options(num_epochs=epochs, lr=lr, batch_size=batch_size)
    log.info(f"[spotiflow] train_config overrides: {train_config or '<none>'}")

    if base_model is None:
        model = Spotiflow()
    else:
        # inference_mode=False: the loaded model is left trainable
        model = load_spotiflow(
            base_model, inference_mode=False, map_location=get_device()
        )
        train_config["finetuned_from"] = str(base_model)  # recorded with the model
    model.fit(
        train_imgs,
        train_spots,
        val_imgs,
        val_spots,
        save_dir=output_dir,
        train_config=train_config or None,
    )


def train_piscis(
    input_dir: str | Path | None,
    output_dir: str | Path | None = None,
    model_name: str | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    base_model: str | None = None,
) -> None:
    """Train a Piscis model.

    Args:
        input_dir: folder with train/, val/ and test/, each with x/ and y/
            (made by make_datasets.py piscis).
        output_dir: Piscis folder holding models/ and checkpoints/. Default:
            $PISCIS_DIR, else Piscis's own folder (~/.piscis), the folder
            inference loads Piscis models from by name. The model is saved to
            <output_dir>/models/<model_name>.pt.
        model_name: name of the trained model. Default: DEFAULT_PISCIS_MODEL_NAME.
        epochs: number of epochs. Default: Piscis's.
        lr: learning rate. Default: Piscis's.
        batch_size: batch size. Default: Piscis's.
        base_model: name of the Piscis model to start from for fine-tuning.
    """
    if input_dir is None:
        raise ValueError("piscis requires --input_dir")
    model_name = model_name or DEFAULT_PISCIS_MODEL_NAME
    if base_model is not None and model_name == str(base_model):
        raise ValueError(
            f"piscis: --model_name {model_name} would overwrite the base model; "
            "give the fine-tuned model another name"
        )

    from piscis.training import train_model

    default_models_dir, _ = set_piscis_dir()
    models_dir, checkpoints_dir = set_piscis_dir(output_dir)
    if base_model is not None:
        base_file = f"{base_model}.pt"
        if (
            not (models_dir / base_file).exists()
            and (default_models_dir / base_file).exists()
        ):
            shutil.copy2(default_models_dir / base_file, models_dir / base_file)

    hparams = given_options(epochs=epochs, learning_rate=lr, batch_size=batch_size)
    log.info(f"[piscis] model: {models_dir / (model_name + '.pt')}")
    log.info(f"[piscis] checkpoints: {checkpoints_dir}")
    log.info(f"[piscis] hyperparameter overrides: {hparams or '<none>'}")

    train_model(
        model_name=model_name,
        dataset_path=input_dir,
        initial_model_name=base_model,
        input_size=DEFAULT_PISCIS_INPUT_SIZE,
        device=get_device(),
        **hparams,
    )


def train_ufish(
    input_dir: str | Path | None,
    val_dir: str | Path | None,
    output_dir: str | Path | None,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    base_model: str | Path | None = None,
) -> None:
    """Train a U-FISH model.

    Args:
        input_dir: training folder (images and CSVs).
        val_dir: validation folder (images and CSVs).
        output_dir: folder where the .pth weights and the TensorBoard logs
            are saved.
        epochs: number of epochs. Default: DEFAULT_UFISH_HPARAMS.
        lr: learning rate. Default: DEFAULT_UFISH_HPARAMS.
        batch_size: batch size. Default: DEFAULT_UFISH_HPARAMS.
        base_model: .pth weights, or name of U-FISH online weights (see
            load_ufish_weights), to start from for fine-tuning.
    """
    if input_dir is None or val_dir is None or output_dir is None:
        raise ValueError("ufish requires --input_dir, --val_dir and --output_dir")
    if base_model is not None and not str(base_model).endswith(".pth"):
        raise ValueError(
            f"ufish: {base_model} cannot be fine-tuned: it is not a PyTorch .pth model"
        )

    from ufish.api import UFish

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    hparams = {
        **DEFAULT_UFISH_HPARAMS,
        **given_options(num_epochs=epochs, lr=lr, batch_size=batch_size),
    }
    log.info(f"[ufish] validation: {val_dir}")
    log.info(f"[ufish] hyperparameters: {hparams}")

    ufish = UFish()
    if base_model is not None:
        load_ufish_weights(ufish, base_model)
    ufish.train(
        input_dir,
        val_dir,
        model_save_dir=output_dir,
        summary_dir=str(Path(output_dir) / "runs"),
        **hparams,
    )


TRAINERS = {
    "spotiflow": train_spotiflow,
    "piscis": train_piscis,
    "ufish": train_ufish,
}

TOOLS = list(TRAINERS)


def train(
    tool: str,
    input_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    base_model: str | Path | None = None,
    **tool_kwargs: Any,
) -> None:
    """Train a tool's model, from scratch or from base_model.

    Args:
        tool: tool name, a key of TRAINERS.
        input_dir: training data (see the tool's train_* function).
        output_dir: where the model is saved (see the tool's train_* function).
        epochs: number of epochs. Default: the tool's.
        lr: learning rate. Default: the tool's.
        batch_size: batch size. Default: the tool's.
        base_model: model to start from for fine-tuning.
        **tool_kwargs: tool-specific arguments: model_name (piscis), val_dir
            (ufish).
    """
    if tool not in TRAINERS:
        raise ValueError(f"Unknown tool: {tool}. Available tools: {TOOLS}")

    _log_gpu(tool)
    log.info(f"[{tool}] input: {input_dir}")
    log.info(f"[{tool}] output: {output_dir or '<tool default>'}")
    log.info(f"[{tool}] start: {base_model or 'from scratch'}")

    TRAINERS[tool](
        input_dir=input_dir,
        output_dir=output_dir,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        base_model=base_model,
        **tool_kwargs,
    )

    log.info(f"[{tool}] training complete")
    if tool == "piscis":
        log.info(
            "[piscis] to use this model, give its name with --model (run_inference.py, "
            "grid_search.py) or --base_model (finetuning.py), with PISCIS_DIR set to "
            "--output_dir if it is not the default folder"
        )
    else:
        log.info(
            f"[{tool}] to use this model, give its path with --model (run_inference.py, "
            "grid_search.py) or --base_model (finetuning.py)"
        )


def tool_kwargs_from_args(tool: str, args: argparse.Namespace) -> dict[str, Any]:
    """Tool-specific training arguments given on the command line.

    Args:
        tool: tool name.
        args: parsed arguments of add_training_args.

    Returns:
        model_name for piscis, val_dir for ufish, else {}.
    """
    if tool == "piscis":
        return {"model_name": args.model_name}
    if tool == "ufish":
        return {"val_dir": args.val_dir}
    return {}


def add_training_args(
    parser: argparse.ArgumentParser, require_dirs: bool = False
) -> argparse.ArgumentParser:
    parser.add_argument("--tool", required=True, choices=TOOLS)
    parser.add_argument(
        "--input_dir",
        required=require_dirs,
        default=None,
        help=(
            "Training data: spotiflow=folder with train/ and val/, piscis=folder with train/, val/ "
            "and test/, each with x/ and y/, ufish=training folder."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=require_dirs,
        default=None,
        help=(
            "Where the model is saved. spotiflow, ufish: the model folder. "
            "piscis: the Piscis folder "
            "holding models/ and checkpoints/ (default $PISCIS_DIR, else ~/.piscis); "
            "the model is saved to <output_dir>/models/<model_name>.pt. Inference "
            "loads Piscis models by name from $PISCIS_DIR (else ~/.piscis) only."
        ),
    )
    parser.add_argument(
        "--val_dir", default=None, help="Validation data folder (ufish only)"
    )
    parser.add_argument(
        "--model_name",
        default=None,
        help=f"Name of the trained model (piscis only, default: {DEFAULT_PISCIS_MODEL_NAME})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of epochs (tool default if omitted)",
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Learning rate (tool default if omitted)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (tool default if omitted)",
    )
    return parser


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a spot detection model from scratch with the specified tool."
    )
    add_training_args(parser)
    args = parser.parse_args()
    setup_logging()

    train(
        args.tool,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        **tool_kwargs_from_args(args.tool, args),
    )


if __name__ == "__main__":
    main()
