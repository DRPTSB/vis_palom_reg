"""Shared IO helpers for the HiRes<->CytAssist registration pipeline."""
from pathlib import Path

import cv2
import dask.array as da
import numpy as np
import tifffile
import zarr

UNIT_TO_UM = {2: 25400.0, 3: 10000.0}  # TIFF ResolutionUnit: 2=inch, 3=cm


def um_per_pixel(path):
    """Physical pixel size (um/px), read from a TIFF's XResolution/ResolutionUnit tags."""
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        xres = page.tags.get("XResolution")
        if xres is None:
            return None
        num, den = xres.value
        runit = page.tags.get("ResolutionUnit")
        unit = runit.value if runit is not None else 3
        if unit not in UNIT_TO_UM:
            return None
        return UNIT_TO_UM[unit] / (num / den)


def read_series0_pyramid(path):
    """The pyramid levels of a TIFF's first series, as dask arrays."""
    with tifffile.TiffFile(path) as tf:
        pyramid = []
        for level in tf.series[0].levels:
            z = zarr.open(level.aszarr(), "r")
            arr = z[0] if isinstance(z, zarr.hierarchy.Group) else z
            pyramid.append(da.from_array(arr, name=False))
    return pyramid


def downsample_to_physical_scale(path, target_um_per_px):
    """Downsample an image so its pixel size matches target_um_per_px.

    HiRes and CytAssist do not cover the same physical field of view, so
    downsampling by matching pixel-array shape assumes equal FOV and
    silently produces the wrong scale. This uses the ratio of *physical*
    pixel sizes instead.
    """
    src_um_per_px = um_per_pixel(path)
    if src_um_per_px is None:
        raise ValueError(f"{path}: no XResolution/ResolutionUnit tags found")
    pyramid = read_series0_pyramid(path)
    base_shape = pyramid[0].shape[:2]
    scale_factor = target_um_per_px / src_um_per_px
    print(f"{Path(path).name}: {src_um_per_px:.4f} um/px -> target {target_um_per_px:.4f} "
          f"um/px (downsample factor {scale_factor:.3f})")

    level_downsamples = [base_shape[0] / lvl.shape[0] for lvl in pyramid]
    best_level = max(i for i, ds in enumerate(level_downsamples) if ds <= scale_factor)
    remaining_factor = scale_factor / level_downsamples[best_level]

    base = pyramid[best_level]
    stride = max(1, round(remaining_factor))
    strided = base[::stride, ::stride, :].compute()
    target_h = round(base.shape[0] / remaining_factor)
    target_w = round(base.shape[1] / remaining_factor)
    resized = cv2.resize(strided, (target_w, target_h), interpolation=cv2.INTER_AREA)
    print(f"  level {best_level} ({base.shape[0]}x{base.shape[1]}) -> final {resized.shape[:2]}")
    return resized
