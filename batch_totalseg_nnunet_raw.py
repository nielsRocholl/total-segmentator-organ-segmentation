#!/usr/bin/env python3
"""
Batch TotalSegmentator over nnU-Net raw datasets.

Deps: pip install TotalSegmentator rich  (+ PyTorch per TotalSegmentator docs)
Offline weights: totalseg_download_weights -t total  or copy ~/.totalsegmentator;
override cache dir with TOTALSEG_HOME_DIR.

Speed: task "total" (default) runs multiple chained nnU-Net steps at 1.5 mm — expect
minutes per volume. Use --fast (3 mm) or --fastest (6 mm) for large speedups; quality drops.

Quality: models assume CT-like Hounsfield units (roughly air ~-1000, water ~0, soft tissue
~40–80). If volumes are rescaled (0–1), wrong slope/intercept, or MR with --task total,
masks will look fragmented or wrong. Use --qc-first to log intensity percentiles on one case.

Example:
  python batch_totalseg_nnunet_raw.py \\
    --nnunet-raw /data/nnUNet_raw --out-root /data/nnUNet_totalseg --device gpu
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import traceback
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

CASE_RE = re.compile(r"^(.+)_(\d{4})$")


@dataclass(frozen=True)
class Job:
    dataset_name: str
    split: str
    case_id: str
    input_path: Path
    output_path: Path


def pick_ct_channel(channel_names: dict) -> int:
    for k, v in channel_names.items():
        if str(v).upper() == "CT":
            return int(k)
    return 0


def stem_without_ending(name: str, file_ending: str) -> str:
    if not name.endswith(file_ending):
        return name
    return name[: -len(file_ending)]


def iter_cases_in_dir(
    images_dir: Path, file_ending: str, channel: int
) -> Iterator[tuple[str, Path]]:
    if not images_dir.is_dir():
        return
    ch = f"{channel:04d}"
    for p in sorted(images_dir.iterdir()):
        if not p.is_file():
            continue
        if p.name.endswith(".json"):
            continue
        if not p.name.endswith(file_ending):
            continue
        stem = stem_without_ending(p.name, file_ending)
        m = CASE_RE.match(stem)
        if not m or m.group(2) != ch:
            continue
        yield m.group(1), p


def collect_jobs(
    nnunet_raw: Path,
    out_root: Path,
    only_datasets: set[str] | None,
) -> list[Job]:
    jobs: list[Job] = []
    for ds in sorted(nnunet_raw.iterdir()):
        if not ds.is_dir() or not ds.name.startswith("Dataset"):
            continue
        if ds.name.startswith("Dataset999_"):
            continue
        if only_datasets is not None and ds.name not in only_datasets:
            continue
        dj = ds / "dataset.json"
        if not dj.is_file():
            continue
        meta = json.loads(dj.read_text())
        file_ending = meta["file_ending"]
        channel = pick_ct_channel(meta["channel_names"])
        out_ds = out_root / ds.name
        seen_tr: set[str] = set()
        for case_id, inp in iter_cases_in_dir(ds / "imagesTr", file_ending, channel):
            seen_tr.add(case_id)
            jobs.append(
                Job(
                    ds.name,
                    "Tr",
                    case_id,
                    inp,
                    out_ds / "labelsTr" / f"{case_id}{file_ending}",
                )
            )
        for case_id, inp in iter_cases_in_dir(ds / "imagesTs", file_ending, channel):
            if case_id in seen_tr:
                continue
            jobs.append(
                Job(
                    ds.name,
                    "Ts",
                    case_id,
                    inp,
                    out_ds / "labelsTs" / f"{case_id}{file_ending}",
                )
            )
    return jobs


def qc_ct_slice_line(path: Path) -> str:
    import numpy as np
    import nibabel as nib

    img = nib.load(str(path))
    d = np.asanyarray(img.dataobj)
    if d.ndim == 4 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim < 3:
        return "non-3D volume — check input"
    z = int(d.shape[2] // 2)
    sl = np.asarray(np.squeeze(d[..., z]), dtype=np.float64).ravel()
    sl = sl[np.isfinite(sl)]
    if sl.size == 0:
        return "no finite voxels on mid slice"
    p1, p50, p99 = np.percentile(sl, [1.0, 50.0, 99.0])
    ad = np.asarray(img.affine)[(0, 1, 2), (0, 1, 2)]
    return (
        f"mid-slice HU-like p1/p50/p99 = {p1:.1f} / {p50:.1f} / {p99:.1f}; "
        f"affine voxel mm (diag≈){np.round(ad, 4)!s} "
        f"(expect CT: air~-1000, soft~40–80; wrong scale ⇒ bad masks)"
    )


def shard_hit(job: Job, shard_index: int, shard_total: int) -> bool:
    if shard_total <= 1:
        return True
    h = zlib.crc32(f"{job.dataset_name}/{job.case_id}".encode()) % shard_total
    return h == shard_index


def is_done(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def run_one(
    job: Job,
    *,
    device: str,
    task: str,
    fast: bool,
    fastest: bool,
    verbose: bool,
    body_seg: bool,
    nr_thr_resamp: int,
    robust_crop: bool,
) -> None:
    from totalsegmentator.python_api import totalsegmentator

    job.output_path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    fe = job.output_path.name[len(job.case_id) :]
    tmp = job.output_path.with_name(f"{job.case_id}.part.{pid}{fe}")
    if tmp.exists():
        tmp.unlink()
    try:
        totalsegmentator(
            job.input_path,
            tmp,
            ml=True,
            task=task,
            device=device,
            fast=fast,
            fastest=fastest,
            quiet=not verbose,
            verbose=verbose,
            nr_thr_resamp=nr_thr_resamp,
            nr_thr_saving=1,
            body_seg=body_seg,
            robust_crop=robust_crop,
        )
        os.replace(tmp, job.output_path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise


def sync_dataset_json(nnunet_raw: Path, out_root: Path, dataset_name: str) -> None:
    src = nnunet_raw / dataset_name / "dataset.json"
    dst = out_root / dataset_name / "dataset.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="TotalSegmentator batch for nnUNet_raw layout.")
    ap.add_argument("--nnunet-raw", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--task", default="total")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fast", action="store_true")
    g.add_argument("--fastest", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--fail-log", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument(
        "--body-seg",
        action="store_true",
        help="Crop to rough body mask first (often faster / less VRAM; can help cropping).",
    )
    ap.add_argument(
        "--robust-crop",
        action="store_true",
        help="Better ROI crops for roi_subset/subtasks; slower.",
    )
    ap.add_argument(
        "--nr-thr-resamp",
        type=int,
        default=6,
        help="CPU threads for resampling (TotalSegmentator default in API was 1; higher uses more CPU).",
    )
    ap.add_argument(
        "--only-dataset",
        action="append",
        dest="only_datasets",
        metavar="DatasetXXX_Name",
        help="Repeat to restrict to dataset folder names.",
    )
    ap.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Parallel jobs: shard 0 .. shard-total-1",
    )
    ap.add_argument(
        "--shard-total",
        type=int,
        default=1,
        help="Number of parallel job shards (hash-split by dataset/case)",
    )
    ap.add_argument(
        "--qc-first",
        action="store_true",
        help="Log mid-slice intensity stats for first pending job (HU sanity)",
    )
    ap.add_argument("--no-qc-first", dest="qc_first", action="store_false")
    ap.set_defaults(qc_first=False)
    args = ap.parse_args()

    nn_raw = args.nnunet_raw.resolve()
    out_root = args.out_root.resolve()
    if not nn_raw.is_dir():
        raise SystemExit(f"not a directory: {nn_raw}")
    od = args.only_datasets
    only: set[str] | None = set(od) if od else None

    all_jobs = collect_jobs(nn_raw, out_root, only)
    console = Console(stderr=True)
    for ds_name in sorted({j.dataset_name for j in all_jobs}):
        sync_dataset_json(nn_raw, out_root, ds_name)

    pending_all = [j for j in all_jobs if args.force or not is_done(j.output_path)]
    pending = [j for j in pending_all if shard_hit(j, args.shard_index, args.shard_total)]
    n_shard_skip = len(pending_all) - len(pending)
    if n_shard_skip and args.shard_total > 1:
        console.print(f"[dim]{n_shard_skip} case(s) assigned to other shards[/dim]")

    n_skip = len(all_jobs) - len(pending_all)
    if n_skip:
        console.print(f"[dim]{n_skip} case(s) already complete (skipped)[/dim]")
    if not pending:
        console.print("[dim]Nothing to do.[/dim]")
        return

    if args.qc_first:
        console.print("[yellow]QC first volume[/yellow] " + qc_ct_slice_line(pending[0].input_path))

    fail_f = args.fail_log.open("a", encoding="utf-8") if args.fail_log else None

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("totalseg", total=len(pending))
            for job in pending:
                progress.update(
                    task_id,
                    description=f"{job.dataset_name} | {job.case_id} | {job.split}",
                )
                try:
                    run_one(
                        job,
                        device=args.device,
                        task=args.task,
                        fast=args.fast,
                        fastest=args.fastest,
                        verbose=args.verbose,
                        body_seg=args.body_seg,
                        nr_thr_resamp=args.nr_thr_resamp,
                        robust_crop=args.robust_crop,
                    )
                except BaseException as e:
                    rec = {
                        "dataset": job.dataset_name,
                        "split": job.split,
                        "case_id": job.case_id,
                        "input": str(job.input_path),
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }
                    if fail_f:
                        fail_f.write(json.dumps(rec) + "\n")
                        fail_f.flush()
                    console.print(f"[red]FAIL[/red] {job.dataset_name} {job.case_id}: {e}")
                    if args.fail_fast:
                        raise
                progress.advance(task_id)
    finally:
        if fail_f:
            fail_f.close()


if __name__ == "__main__":
    main()
