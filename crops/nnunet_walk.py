"""nnUNet_raw iteration helpers."""
from __future__ import annotations

import json
import re
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

CASE_RE = re.compile(r"^(.+)_(\d{4})$")


@dataclass(frozen=True)
class CropJob:
    dataset_name: str
    split: str
    case_id: str
    image_path: Path
    lesion_label_path: Path
    totalseg_path: Path


def pick_ct_channel(channel_names: dict) -> int:
    for k, v in channel_names.items():
        if str(v).upper() == "CT":
            return int(k)
    return 0


def stem_without_ending(name: str, file_ending: str) -> str:
    return name[: -len(file_ending)] if name.endswith(file_ending) else name


def iter_cases_in_dir(
    images_dir: Path, file_ending: str, channel: int
) -> Iterator[tuple[str, Path]]:
    if not images_dir.is_dir():
        return
    ch = f"{channel:04d}"
    for p in sorted(images_dir.iterdir()):
        if not p.is_file() or p.name.endswith(".json"):
            continue
        if not p.name.endswith(file_ending):
            continue
        stem = stem_without_ending(p.name, file_ending)
        m = CASE_RE.match(stem)
        if not m or m.group(2) != ch:
            continue
        yield m.group(1), p


def collect_crop_jobs(
    nnunet_raw: Path,
    totalseg_root: Path,
    only_datasets: set[str] | None,
) -> list[CropJob]:
    jobs: list[CropJob] = []
    for ds in sorted(nnunet_raw.iterdir()):
        if not ds.is_dir() or not ds.name.startswith("Dataset"):
            continue
        if ds.name.startswith("Dataset999_") or ds.name.startswith("Dataset010"):
            continue
        if only_datasets is not None and ds.name not in only_datasets:
            continue
        dj = ds / "dataset.json"
        if not dj.is_file():
            continue
        meta = json.loads(dj.read_text())
        fe, ch = meta["file_ending"], pick_ct_channel(meta["channel_names"])
        seg_ds = totalseg_root / ds.name
        seen_tr: set[str] = set()
        for cid, inp in iter_cases_in_dir(ds / "imagesTr", fe, ch):
            seen_tr.add(cid)
            jobs.append(
                CropJob(
                    ds.name,
                    "Tr",
                    cid,
                    inp,
                    ds / "labelsTr" / f"{cid}{fe}",
                    seg_ds / "labelsTr" / f"{cid}{fe}",
                )
            )
        for cid, inp in iter_cases_in_dir(ds / "imagesTs", fe, ch):
            if cid not in seen_tr:
                jobs.append(
                    CropJob(
                        ds.name,
                        "Ts",
                        cid,
                        inp,
                        ds / "labelsTs" / f"{cid}{fe}",
                        seg_ds / "labelsTs" / f"{cid}{fe}",
                    )
                )
    return jobs


def shard_hit(job: CropJob, shard_index: int, shard_total: int) -> bool:
    if shard_total <= 1:
        return True
    return zlib.crc32(f"{job.dataset_name}/{job.case_id}".encode()) % shard_total == shard_index
