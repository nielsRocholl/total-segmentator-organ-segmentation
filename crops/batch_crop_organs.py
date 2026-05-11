#!/usr/bin/env python3
"""Batch crop liver/lung from TotalSegmentator outputs + nnUNet_raw CT."""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

import SimpleITK as sitk
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

from crops.crop_geom import crop_case
from crops.nnunet_walk import CropJob, collect_crop_jobs, shard_hit


def one_case_per_dataset(jobs: list[CropJob]) -> list[CropJob]:
    seen: set[str] = set()
    out: list[CropJob] = []
    for j in jobs:
        if j.dataset_name in seen:
            continue
        seen.add(j.dataset_name)
        out.append(j)
    return out


def task_label_sets(task: str) -> tuple[list[int], list[int]]:
    from totalsegmentator.map_to_binary import class_map

    cm = class_map[task]
    liver = [int(k) for k, v in cm.items() if "liver" in str(v).lower()]
    lung = [int(k) for k, v in cm.items() if "lung" in str(v).lower()]
    return liver, lung


def run_case(job: CropJob, out_root: Path, liver_ids: list[int], lung_ids: list[int], margin: int, force: bool) -> None:
    if not job.mask_path.is_file():
        raise FileNotFoundError(
            f"{job.mask_path}\n(mask missing — check --totalseg-root, "
            "e.g. .../totalseg_segmentations)"
        )
    ct = sitk.ReadImage(str(job.image_path))
    seg = sitk.ReadImage(str(job.mask_path))
    crop_case(ct, seg, liver_ids, lung_ids, out_root / job.dataset_name / job.case_id, margin, force)


def main() -> None:
    ap = argparse.ArgumentParser(description="Crop liver/lung from TotalSegmentator nnUNet outputs.")
    ap.add_argument("--nnunet-raw", type=Path, required=True)
    ap.add_argument("--totalseg-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--margin", type=int, default=10)
    ap.add_argument("--task", default="total")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--only-dataset", action="append", dest="only_datasets", metavar="DatasetXXX_Name")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-total", type=int, default=1)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Process only the first case per dataset (check outputs before full run).",
    )
    args = ap.parse_args()

    nn_raw = args.nnunet_raw.resolve()
    ts_root = args.totalseg_root.resolve()
    out_root = args.out_root.resolve()
    if not nn_raw.is_dir():
        raise SystemExit(f"not a directory: {nn_raw}")
    only: set[str] | None = set(args.only_datasets) if args.only_datasets else None

    liver_ids, lung_ids = task_label_sets(args.task)
    jobs = collect_crop_jobs(nn_raw, ts_root, only)
    if args.dry_run:
        jobs = one_case_per_dataset(jobs)
    console = Console(stderr=True)
    if args.dry_run:
        console.print(
            f"[yellow]Dry-run:[/yellow] {len(jobs)} case(s), one per dataset — outputs still written under --out-root."
        )
        if args.shard_index != 0 or args.shard_total != 1:
            console.print("[dim]Dry-run ignores --shard-index / --shard-total.[/dim]")
        pending = jobs
    else:
        pending = [j for j in jobs if shard_hit(j, args.shard_index, args.shard_total)]
    if not args.dry_run and args.shard_total > 1:
        console.print(f"[dim]{len(jobs) - len(pending)} case(s) on other shards[/dim]")
    if not pending:
        console.print("[dim]Nothing to do.[/dim]")
        return

    errs = 0
    cols = (
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    )
    with Progress(*cols, console=console) as prog:
        tid = prog.add_task("crop", total=len(pending))
        for job in pending:
            prog.update(tid, description=f"{job.dataset_name} | {job.case_id} | {job.split}")
            try:
                run_case(job, out_root, liver_ids, lung_ids, args.margin, args.force)
            except BaseException as e:
                errs += 1
                console.print(f"[red]FAIL[/red] {job.dataset_name} {job.case_id}: {e}")
                if args.verbose:
                    console.print(traceback.format_exc())
                if args.fail_fast:
                    raise
            prog.advance(tid)
    if errs:
        raise SystemExit(errs)


if __name__ == "__main__":
    main()
