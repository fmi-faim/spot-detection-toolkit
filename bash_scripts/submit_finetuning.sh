#!/bin/bash
#SBATCH --account=faim-dev
#SBATCH --job-name=finetune
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --partition=main,several
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=56:00:00
#SBATCH --output=logs/finetune-%j.out
#SBATCH --error=logs/finetune-%j.err

# Fine-tuning SLURM Batch Script
# Usage:
# sbatch bash_scripts/submit_finetuning.sh <tool> --input_dir <dir> --output_dir <dir> [finetuning.py arguments...]
#
# Starts from the tool's default 2D model (downloaded on first use), or from
# --base_model <model>. --input_dir and --output_dir are required.
#
# Examples:
#   sbatch bash_scripts/submit_finetuning.sh spotiflow --input_dir <data> --output_dir models/spotiflow/general_finetuned --lr 3e-5 --epochs 50
#   sbatch bash_scripts/submit_finetuning.sh ufish --input_dir <train> --val_dir <val> --output_dir models/ufish/finetuned --base_model models/ufish/best_model.pth --lr 1e-5
#   PISCIS_DIR=models/piscis sbatch bash_scripts/submit_finetuning.sh piscis --input_dir <piscis dataset> --output_dir models/piscis --base_model piscis_customed_512 --model_name my_finetuned
# Optional for every tool: --epochs, --lr, --batch_size (tool default if omitted)

set -eu

if [ $# -lt 1 ]; then
    echo "Usage: sbatch bash_scripts/submit_finetuning.sh <tool> --input_dir <dir> --output_dir <dir> [finetuning.py arguments...]"
    exit 1
fi

# Activate environment
WD="$(pwd)"
export PATH="$PATH:$WD/infrastructure/apps/pixi/bin"
export PIXI_CACHE_DIR="$WD/infrastructure/apps/pixi/.pixi_cache"
export TMPDIR="$WD/infrastructure/.tmp_$USER"
mkdir -p "$TMPDIR"

PIXIBIN="$WD/infrastructure/apps/pixi/bin/pixi"
if [[ ! -x "$PIXIBIN" ]]; then
    echo "[INFO] Pixi binary not found; running install.sh"
    bash "$WD/bash_scripts/install.sh"
fi

TOOL=$1
shift

if [ "$TOOL" = "ufish" ]; then
    PIXI_ENV="ufish-gpu"
else
    PIXI_ENV="$TOOL"
fi

echo "[INFO] Job ID: $SLURM_JOB_ID"
echo "[INFO] Node: $SLURMD_NODENAME"
echo "[INFO] Starting fine-tuning job $SLURM_JOB_ID"
echo "[INFO] Tool: $TOOL (pixi environment: $PIXI_ENV)"
echo "[INFO] Arguments: $*"

pixi run -e "$PIXI_ENV" python src/spot_detection_benchmark/finetuning.py \
    --tool "$TOOL" \
    "$@"

echo "[INFO] Fine-tuning complete"
