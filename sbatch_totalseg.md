# SLURM: batch TotalSegmentator on nnUNet_raw

Host data is mounted at **`/nnunet_data`** (see `--container-mounts`). Inside the container:

| Path | Role |
|------|------|
| `/nnunet_data/nnUNet_raw` | Input: nnU-Net raw datasets (`Dataset*`, `imagesTr`, `imagesTs`, …) |
| `/nnunet_data/nnUNet_totalseg` | Output: mirrored layout with `labelsTr` / `labelsTs` multilabel NIfTIs |

Place this repo (or at least `batch_totalseg_nnunet_raw.py` and `requirements.txt`) somewhere under the mounted tree, e.g. `/nnunet_data/total-segmentator/`, and point `SCRIPT_DIR` below to that directory.

Weights: first run needs Hub access, or pre-populate `~/.totalsegmentator` / set `TOTALSEG_HOME_DIR` to a path on the mount.

Default job script:

```bash
#!/bin/bash
#SBATCH --qos=vram
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=07-00:00:00
#SBATCH --job-name=totalseg-nnunet-raw
#SBATCH --output=/data/oncology/experiments/universal-lesion-segmentation/logs/totalseg-nnunet-raw.out
#SBATCH --error=/data/oncology/experiments/universal-lesion-segmentation/logs/totalseg-nnunet-raw.err
#SBATCH --container-mounts=/data/oncology/experiments/universal-lesion-segmentation:/nnunet_data
#SBATCH --container-image="dockerdex.umcn.nl:5005/nielsrocholl/nnunet-v2-pro-sol-docker:latest"

set -euo pipefail

export PIP_CACHE_DIR=/root/.pip-cache
mkdir -p "$PIP_CACHE_DIR"

# Directory containing batch_totalseg_nnunet_raw.py and requirements.txt (on the host, under the mount).
SCRIPT_DIR=/nnunet_data/total-segmentator

pip3 install -r "${SCRIPT_DIR}/requirements.txt"

python3 "${SCRIPT_DIR}/batch_totalseg_nnunet_raw.py" \
  --nnunet-raw /nnunet_data/nnUNet_raw \
  --out-root /nnunet_data/nnUNet_totalseg \
  --device gpu
```

Optional flags (append as needed):

- `--fast` — lower-resolution model (less VRAM / faster).
- `--task total_mr` — MR volumes instead of default CT `total`.
- `--fail-log /nnunet_data/nnUNet_totalseg/failures.jsonl` — append JSON lines on per-case errors.
- `--verbose` — forward TotalSegmentator logging (noisy; progress bar stays on stderr).

Resume: re-submit the same job; completed cases with non-empty label files are skipped unless you pass `--force`.
