#!/usr/bin/env python3
"""
Batch TotalSegmentator over nnU-Net raw datasets.

Deps: pip install TotalSegmentator rich  (+ PyTorch per TotalSegmentator docs)
Offline weights: totalseg_download_weights -t total  or copy ~/.totalsegmentator;
override cache dir with TOTALSEG_HOME_DIR.

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


def collect_jobs(nnunet_raw: Path, out_root: Path) -> list[Job]:
    jobs: list[Job] = []
    for ds in sorted(nnunet_raw.iterdir()):
        if not ds.is_dir() or not ds.name.startswith("Dataset"):
            continue
        if ds.name.startswith("Dataset999_"):
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


def is_done(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def run_one(job: Job, *, device: str, task: str, fast: bool, verbose: bool) -> None:
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
            quiet=not verbose,
            verbose=verbose,
            nr_thr_saving=1,
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
    # copyfile only (no copystat): NFS/shares often deny utime on destination.
    shutil.copyfile(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser(description="TotalSegmentator batch for nnUNet_raw layout.")
    ap.add_argument("--nnunet-raw", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--device", default="gpu")
    ap.add_argument("--task", default="total")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--fail-log", type=Path, default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    nn_raw = args.nnunet_raw.resolve()
    out_root = args.out_root.resolve()
    if not nn_raw.is_dir():
        raise SystemExit(f"not a directory: {nn_raw}")

    all_jobs = collect_jobs(nn_raw, out_root)
    console = Console(stderr=True)
    for ds_name in sorted({j.dataset_name for j in all_jobs}):
        sync_dataset_json(nn_raw, out_root, ds_name)

    pending = [j for j in all_jobs if args.force or not is_done(j.output_path)]
    n_skip = len(all_jobs) - len(pending)
    if n_skip:
        console.print(f"[dim]{n_skip} case(s) already complete (skipped)[/dim]")
    if not pending:
        console.print("[dim]Nothing to do.[/dim]")
        return

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
                        verbose=args.verbose,
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
