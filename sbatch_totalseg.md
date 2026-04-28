# SLURM: batch TotalSegmentator on nnUNet_raw

Host data is mounted at **`/nnunet_data`** (see `--container-mounts`). Inside the container:

| Path | Role |
|------|------|
| `/nnunet_data/nnUNet_raw` | Input: nnU-Net raw datasets (`Dataset*`, `imagesTr`, `imagesTs`, …) |
| `/nnunet_data/nnUNet_totalseg` | Output: mirrored layout with `labelsTr` / `labelsTs` multilabel NIfTIs |

The job script **clones this repository into `/root`** on the compute node, installs `requirements.txt` from there, and runs the batch script. Data stays on **`/nnunet_data`** (mount only). For a **private** repo, use an SSH URL plus agent forwarding / deploy key in the container, or set `TOTALSEG_GIT_URL` to an HTTPS URL with a token.

Weights: first run needs Hub access, or pre-populate `~/.totalsegmentator` / set `TOTALSEG_HOME_DIR` to a path on the mount (e.g. under `/nnunet_data`).

**`No module named 'pydicom.pixels'`:** the base image often has pydicom 2.x; TotalSegmentator needs **pydicom ≥ 3.0** (see `requirements.txt`). Ensure `pip install -r requirements.txt` runs on every job so the pin upgrades the env.

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

REPO_DIR=/root/total-segmentator-organ-segmentation
GIT_URL="${TOTALSEG_GIT_URL:-git@github.com:nielsRocholl/total-segmentator-organ-segmentation.git}"

if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone --depth 1 --branch main "$GIT_URL" "$REPO_DIR"
fi
git -C "$REPO_DIR" fetch --depth 1 origin main
git -C "$REPO_DIR" reset --hard origin/main

pip3 install -r "${REPO_DIR}/requirements.txt"

python3 "${REPO_DIR}/batch_totalseg_nnunet_raw.py" \
  --nnunet-raw /nnunet_data/nnUNet_raw \
  --out-root /nnunet_data/nnUNet_totalseg \
  --device gpu
```

Optional flags (append as needed):

- **`--fast`** — 3 mm model (much faster than default, modest quality drop). **`--fastest`** — 6 mm, fastest coarsest.
- **`--qc-first`** — log HU-like percentiles on the first pending volume; if values look nothing like CT (not ~ −1000…+1000), fix intensity/slope/intercept upstream before trusting masks.
- **`--body-seg`** — crop to body first (often less VRAM, sometimes better field-of-view).
- **`--nr-thr-resamp 8`** — more CPU threads for resampling (match `--cpus-per-task` roughly).
- **`--only-dataset Dataset027_MCT_LTDiag`** — repeat to limit which folders are processed.
- **`--shard-index K --shard-total N`** — parallel jobs: stable hash splits cases across *N* jobs (each needs its own `$SLURM_ARRAY_TASK_ID` or manual `K`).
- `--task total_mr` — MR volumes instead of default CT **`total`**.
- `--fail-log /nnunet_data/nnUNet_totalseg/failures.jsonl` — append JSON lines on per-case errors.
- `--verbose` — forward TotalSegmentator logging (noisy; progress bar stays on stderr).

Resume: re-submit the same job; completed cases with non-empty label files are skipped unless you pass `--force`.

### Slow runs

Default task **`total`** chains **several** nn-U-Net refinements at **1.5 mm** — rough **many minutes per case** is normal on one GPU. Use **`--fast`** / **`--fastest`** when throughput matters more than full-res detail, or spawn **`--shard-total`** parallel GPU jobs across the scheduler.

### Bad-looking masks

TotalSegmentator assumes **CT Hounsfield scale** on the **`CT`** channel in `dataset.json`. If tensors are **[0,1] floats**, PET/SUV wrongly labeled **`CT`**, or **MR** routed through **`total`**, overlay will look patchy/wrong — fix preprocessing or **`--task total_mr`** for MR.

Parallel multi-GPU bash sketch (array size = shard count):

```bash
python3 "${REPO_DIR}/batch_totalseg_nnunet_raw.py" \
  --nnunet-raw /nnunet_data/nnUNet_raw \
  --out-root /nnunet_data/nnUNet_totalseg \
  --device gpu \
  --fast \
  --qc-first \
  --shard-index "${SLURM_ARRAY_TASK_ID}" \
  --shard-total 4
```
