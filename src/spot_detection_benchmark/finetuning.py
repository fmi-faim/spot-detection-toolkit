from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from spot_detection_benchmark.detectors import default_model
from spot_detection_benchmark.train import (
    add_training_args,
    tool_kwargs_from_args,
    train,
)
from spot_detection_benchmark.utils import setup_logging


def _check_not_overwritten(base_model: str | Path, output_dir: str | Path) -> None:
    base = Path(str(base_model))
    if not base.exists():
        return
    base = base.resolve()
    if Path(output_dir).resolve() in (base, base.parent):
        raise ValueError(
            f"--output_dir {output_dir} would overwrite the base model {base_model}: "
            "save the fine-tuned model in another folder"
        )


def finetune(
    tool: str,
    input_dir: str | Path | None,
    output_dir: str | Path | None,
    base_model: str | Path | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    **tool_kwargs: Any,
) -> None:
    """Fine-tune an existing model on new data.

    Args:
        tool: tool name.
        input_dir: training data (see train).
        output_dir: where the fine-tuned model is saved. Must not be the base
            model's folder.
        base_model: model to fine-tune (file, folder or name). Default: the
            tool's default 2D model, downloaded on first use.
        epochs: number of epochs. Default: the tool's.
        lr: learning rate. Default: the tool's.
        batch_size: batch size. Default: the tool's.
        **tool_kwargs: tool-specific arguments (see train). For piscis, model_name
            defaults to <base model>_finetuned.
    """
    if input_dir is None or output_dir is None:
        raise ValueError("fine-tuning requires --input_dir and --output_dir")
    if base_model is None:
        base_model = default_model(tool, "2d")
    _check_not_overwritten(base_model, output_dir)

    if tool == "piscis" and tool_kwargs.get("model_name") is None:
        tool_kwargs["model_name"] = f"{Path(str(base_model)).stem}_finetuned"

    train(
        tool,
        input_dir=input_dir,
        output_dir=output_dir,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        base_model=base_model,
        **tool_kwargs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune an existing spot detection model on new data."
    )
    add_training_args(parser, require_dirs=True)
    parser.add_argument(
        "--base_model",
        default=None,
        help=(
            "Model to fine-tune: spotiflow folder, piscis name, "
            "ufish .pth, or a model name of the tool. Omit to use the tool's "
            "default 2D model, downloaded on first use."
        ),
    )
    args = parser.parse_args()
    setup_logging()

    finetune(
        args.tool,
        args.input_dir,
        args.output_dir,
        base_model=args.base_model,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        **tool_kwargs_from_args(args.tool, args),
    )


if __name__ == "__main__":
    main()
