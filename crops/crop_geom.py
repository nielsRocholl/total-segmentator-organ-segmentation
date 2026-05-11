"""BBox crop + pad with consistent LPS/metadata."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import SimpleITK as sitk


def assert_same_grid(ct: sitk.Image, m: sitk.Image) -> None:
    if ct.GetSize() != m.GetSize():
        raise RuntimeError(f"size mismatch ct{ct.GetSize()} mask{m.GetSize()}")
    if np.max(np.abs(np.asarray(ct.GetSpacing()) - np.asarray(m.GetSpacing()))) > 1e-4:
        raise RuntimeError("spacing mismatch")
    if np.max(np.abs(np.asarray(ct.GetDirection()) - np.asarray(m.GetDirection()))) > 1e-4:
        raise RuntimeError("direction cosine mismatch")


def binarize_labels(full: sitk.Image, label_ids: list[int]) -> sitk.Image:
    arr = sitk.GetArrayFromImage(full)
    out = (
        np.zeros_like(arr, dtype=np.uint8)
        if not label_ids
        else np.isin(arr, np.asarray(label_ids, dtype=arr.dtype)).astype(np.uint8)
    )
    m = sitk.GetImageFromArray(out)
    m.CopyInformation(full)
    return sitk.Cast(m, sitk.sitkUInt8)


def bbox_with_margin(
    binary: sitk.Image, margin: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    b = sitk.Cast(binary > 0, sitk.sitkUInt8)
    stats = sitk.LabelShapeStatisticsImageFilter()
    stats.Execute(b)
    labs = [x for x in stats.GetLabels() if x != 0]
    if not labs:
        return None
    x0 = y0 = z0 = 10**9
    x1 = y1 = z1 = -1
    for lab in labs:
        bb = stats.GetBoundingBox(lab)
        xs, ys, zs, xl, yl, zl = bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]
        x0, y0, z0 = min(x0, xs), min(y0, ys), min(z0, zs)
        x1 = max(x1, xs + xl - 1)
        y1 = max(y1, ys + yl - 1)
        z1 = max(z1, zs + zl - 1)
    start = (x0 - margin, y0 - margin, z0 - margin)
    end = (x1 + margin, y1 + margin, z1 + margin)
    crop_size = tuple(int(end[i] - start[i] + 1) for i in range(3))
    if any(c <= 0 for c in crop_size):
        return None
    return start, crop_size


def extract_crop(image: sitk.Image, region: tuple[tuple[int, int, int], tuple[int, int, int]], *, is_mask: bool):
    start_index, crop_size = region
    full_start = tuple(int(start_index[i]) for i in range(3))
    image_size = image.GetSize()
    pad_lower = [max(0, -s) for s in full_start]
    pad_upper = [max(0, full_start[i] + crop_size[i] - image_size[i]) for i in range(3)]
    adjusted_start = [max(0, s) for s in full_start]
    extract_size = []
    for i in range(3):
        avail = image_size[i] - adjusted_start[i]
        need = crop_size[i] - pad_lower[i] - pad_upper[i]
        extract_size.append(max(0, min(avail, need)))
    if image.GetNumberOfComponentsPerPixel() > 1:
        cf = sitk.VectorIndexSelectionCastImageFilter()
        cf.SetIndex(0)
        image = cf.Execute(image)
    if all(s > 0 for s in extract_size):
        ex = sitk.ExtractImageFilter()
        ex.SetSize(extract_size)
        ex.SetIndex(adjusted_start)
        cropped = ex.Execute(image)
    else:
        cropped = sitk.Image(extract_size, image.GetPixelID())
        cropped.SetSpacing(image.GetSpacing())
        cropped.SetDirection(image.GetDirection())
    if any(p > 0 for p in pad_lower + pad_upper):
        pf = sitk.ConstantPadImageFilter()
        pf.SetPadLowerBound(pad_lower)
        pf.SetPadUpperBound(pad_upper)
        pf.SetConstant(0.0 if is_mask else -1000.0)
        cropped = pf.Execute(cropped)
    orig = np.asarray(image.GetOrigin(), dtype=np.float64)
    direction = np.asarray(image.GetDirection(), dtype=np.float64).reshape(3, 3)
    spacing = np.asarray(image.GetSpacing(), dtype=np.float64)
    new_orig = orig + direction @ (np.asarray(full_start, dtype=np.float64) * spacing)
    cropped.SetOrigin(tuple(float(x) for x in new_orig))
    cropped.SetSpacing(image.GetSpacing())
    cropped.SetDirection(image.GetDirection())
    return cropped


def organ_complete(out_dir: Path) -> bool:
    i, m = out_dir / "image.nii.gz", out_dir / "mask.nii.gz"
    return i.is_file() and i.stat().st_size > 0 and m.is_file() and m.stat().st_size > 0


def crop_one_organ(ct: sitk.Image, organ_bin: sitk.Image, out_dir: Path, margin: int, force: bool) -> None:
    if not np.any(sitk.GetArrayFromImage(organ_bin)):
        return
    if not force and organ_complete(out_dir):
        return
    region = bbox_with_margin(organ_bin, margin)
    if region is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(extract_crop(ct, region, is_mask=False), str(out_dir / "image.nii.gz"), useCompression=True)
    sitk.WriteImage(extract_crop(organ_bin, region, is_mask=True), str(out_dir / "mask.nii.gz"), useCompression=True)


def crop_case(ct: sitk.Image, seg: sitk.Image, liver_ids: list[int], lung_ids: list[int], case_root: Path, margin: int, force: bool) -> None:
    assert_same_grid(ct, seg)
    crop_one_organ(ct, binarize_labels(seg, liver_ids), case_root / "liver", margin, force)
    crop_one_organ(ct, binarize_labels(seg, lung_ids), case_root / "lung", margin, force)
