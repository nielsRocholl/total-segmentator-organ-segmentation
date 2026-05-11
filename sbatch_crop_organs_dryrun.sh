#!/bin/bash
#SBATCH --qos=low
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=07-00:00:00
#SBATCH --job-name=totalseg-organ-crops-dry
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/totalseg-organ-crops-dry.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/totalseg-organ-crops-dry.err
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

set -euo pipefail

export PIP_CACHE_DIR=/root/.pip-cache
mkdir -p "$PIP_CACHE_DIR"

REPO_DIR=/root/total-segmentator-organ-segmentation
GIT_URL="${TOTALSEG_GIT_URL:-git@github.com:nielsRocholl/total-segmentator-organ-segmentation.git}"

if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone --depth 1 --branch main "$GIT_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --depth 1 origin main
git -C "$REPO_DIR" reset --hard origin/main

pip3 install -r "${REPO_DIR}/requirements.txt"

# Predictions: export TOTALSEG_PRED_ROOT=... if not default below.
TOTALSEG_PRED_ROOT="${TOTALSEG_PRED_ROOT:-/nnunet_data/unprocessed-universal-lesion-segmentation/totalseg_segmentations}"

cd "$REPO_DIR"
python3 crops/batch_crop_organs.py \
  --nnunet-raw /nnunet_data/nnUNet_raw \
  --totalseg-root "$TOTALSEG_PRED_ROOT" \
  --out-root /nnunet_data/unprocessed-universal-lesion-segmentation/totalseg_crops \
  --margin 10 \
  --dry-run
