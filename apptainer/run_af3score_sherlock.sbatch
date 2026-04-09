#!/bin/bash
#SBATCH --job-name=af3score
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: sbatch $0 IMAGE.sif INPUT_DIR OUTPUT_DIR WEIGHTS_BIN [DB_DIR]" >&2
  exit 1
fi

IMAGE=$1
INPUT_DIR=$(realpath "$2")
OUTPUT_DIR=$(realpath "$3")
WEIGHTS_BIN=$(realpath "$4")
DB_DIR=${5:-}

mkdir -p "$OUTPUT_DIR"

BIND_ARGS=(
  --bind "$INPUT_DIR":"$INPUT_DIR"
  --bind "$OUTPUT_DIR":"$OUTPUT_DIR"
  --bind "$(dirname "$WEIGHTS_BIN")":"$(dirname "$WEIGHTS_BIN")"
)

if [[ -n "$DB_DIR" ]]; then
  DB_DIR=$(realpath "$DB_DIR")
  BIND_ARGS+=(--bind "$DB_DIR":"$DB_DIR")
  DB_FLAG=(--db_dir "$DB_DIR")
else
  DB_FLAG=()
fi

apptainer exec --nv "${BIND_ARGS[@]}" "$IMAGE" \
  python /opt/af3score/af3score_pipeline.py \
    --input "$INPUT_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --weights "$WEIGHTS_BIN" \
    --run_data_pipeline=false \
    --run_inference=true \
    "${DB_FLAG[@]}"
