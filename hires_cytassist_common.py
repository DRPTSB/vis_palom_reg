"""Shared IO helpers for the HiRes<->CytAssist registration pipeline."""
import datetime
import os
import sys
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



class _Tee:
    """Duplicates writes to both an underlying stream and a shared log
    file, so console output is captured to a persistent per-sample log
    without changing any individual print()/logger call. Wrapping
    sys.stdout/sys.stderr with this (see setup_pipeline_log below) means a
    crash's traceback, a warning emitted by a third-party library (palom,
    tifffile, dask), and every existing print()/logger.info() call all end
    up in the log -- not just messages this codebase explicitly writes to
    it."""
    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file

    def write(self, data):
        self._stream.write(data)
        self._log_file.write(data)
        self._log_file.flush()

    def flush(self):
        self._stream.flush()
        self._log_file.flush()

    def isatty(self):
        return False


def setup_pipeline_log(log_dir, sample, script_name):
    """Append this process's entire stdout/stderr -- every print(),
    logger call, warning, and crash traceback -- to a single, persistent
    <log_dir>/<sample>_pipeline.log, shared across every stage of the
    pipeline (register.py, tissue_detection_pipeline.py,
    build_full_res_deliverables.py, merge_loupe_alignment.py, report.py,
    run_pipeline.py). Call this once, as early as possible, at the very
    start of each script's entry point (before other top-level code that
    might itself configure logging -- see tissue_detection_pipeline.py,
    which moves its logging.basicConfig() call to after this one so its
    handler binds to the already-wrapped stderr).

    Why this exists: build_full_res_deliverables.py's --checkpoint runs
    typically span many separate process invocations (one per device-shell
    call, each capped at ~180s on this project's remote-device bridge), so
    console output from any one invocation was otherwise lost the moment
    that shell call ended -- including, in one real 2026-09-27 incident, a
    crash traceback that would have made a checkpoint-corruption bug much
    faster to diagnose. Appending to one shared file instead means the
    full history of a sample's run survives past any single invocation.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{sample}_pipeline.log"
    log_file = open(log_path, "a", buffering=1)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_file.write(f"\n=== {ts} -- {script_name} started (pid {os.getpid()}) ===\n")
    log_file.flush()
    sys.stdout = _Tee(sys.stdout, log_file)
    sys.stderr = _Tee(sys.stderr, log_file)
    return log_path
