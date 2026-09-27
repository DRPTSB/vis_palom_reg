"""
Encode a tissue mask (in CytAssist pixel space) into the `oligo` bitfield of a
real 10x `*-fiducials-image-registration.json`, so our own tissue calls (e.g.
from `tissue_detection_pipeline.py`) can replace Space Ranger/Loupe's own
tissue detection in a file it will otherwise trust as-is.

Background (see the project doc "tissue_oligo_encoding" for the full writeup):

`oligo` is a base64-encoded, `numpy.packbits`-packed, row-major, top-left-
origin bitmask over an NxN grid (N = sqrt(spot_count)), where each cell is
one bin at `spot_metadata.bin_level`. Its row/col indices are the slide
design's fixed ARRAY coordinate space (the same array_row/array_col space
used in tissue_positions.csv) -- not raw image pixels, and not dependent on
any one image's alignment.

What ties that fixed array space to a SPECIFIC CytAssist photograph is the
top-level `transform` field. It expects coordinates pre-scaled by
`bin_level` (i.e. at native bin_level=1 resolution), not raw array-grid
indices -- confirmed both by a physical-constant cross-check (transform's
scale x bin_level matches the expected um-per-bin / um-per-px ratio to
within 0.6%) and by a forward-project/invert self-consistency test (IoU
0.99+ between a real oligo grid and its own reconstruction after being
projected into pixel space and back). This module only encodes NEW tissue
calls into `oligo` -- it never touches `transform` or `cytAssistInfo`,
which are assumed to already be correct (sourced from Space Ranger's own
automatic fiducial detection, or a trusted Loupe export) for this exact
slide+area.

This module deliberately hard-fails rather than guessing when its
assumptions don't hold for a given base file (see `validate_base_json`) --
per an explicit instruction from the user of this pipeline to prefer an
error over silently doing something unverified.
"""
import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np

# The x-bin_level transform convention below has only been empirically
# verified for bin_level=8 (Visium HD). Don't silently assume it generalizes.
VERIFIED_BIN_LEVELS = (8,)


def validate_base_json(base):
    """Hard-fail (not warn) if this file doesn't match the assumptions this
    module's transform convention was verified against."""
    if "spot_metadata" not in base or "bin_level" not in base["spot_metadata"]:
        raise ValueError("base JSON has no spot_metadata.bin_level -- required to know the "
                          "oligo grid's bin size; refusing to guess.")
    bin_level = base["spot_metadata"]["bin_level"]
    if bin_level not in VERIFIED_BIN_LEVELS:
        raise ValueError(f"spot_metadata.bin_level={bin_level!r} has not been verified for this "
                          f"encoder (only {VERIFIED_BIN_LEVELS} tested) -- refusing to guess "
                          "whether the same '(bin_level) transform convention applies. Re-derive "
                          "and verify the convention for this bin_level before proceeding.")
    if "transform" not in base:
        raise ValueError("base JSON has no top-level 'transform' field -- required to map the "
                          "oligo array grid to CytAssist pixel space.")
    transform = np.array(base["transform"], dtype=float)
    if transform.shape != (3, 3):
        raise ValueError(f"'transform' is not a 3x3 matrix (got shape {transform.shape})")
    if "spot_count" not in base:
        raise ValueError("base JSON has no 'spot_count' -- required to know the oligo grid size.")
    n = base["spot_count"]
    grid_size = int(round(n ** 0.5))
    if grid_size * grid_size != n:
        raise ValueError(f"spot_count={n} is not a perfect square -- the NxN grid assumption "
                          "this encoder relies on doesn't hold for this file.")
    return bin_level, transform, grid_size


def decode_oligo(base):
    """Decode the current oligo grid from a base JSON. Returns (grid, bin_level,
    transform, grid_size). `grid` is bool, shape (grid_size, grid_size), indexed
    [array_row, array_col]."""
    bin_level, transform, grid_size = validate_base_json(base)
    n = grid_size * grid_size
    raw = base64.b64decode(base["oligo"])
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:n]
    grid = bits.reshape(grid_size, grid_size).astype(bool)
    return grid, bin_level, transform, grid_size


def encode_oligo(grid):
    """Inverse of decode_oligo's unpacking: bool grid -> base64 str, matching
    the schema's packbits/row-major/top-left convention exactly (verified
    byte-for-byte round-trip against a real 10x file)."""
    bits = grid.astype(np.uint8).flatten()
    packed = np.packbits(bits)
    return base64.b64encode(packed.tobytes()).decode("ascii")


def bin_to_pixel_matrix(transform, bin_level):
    """3x3 matrix mapping array-grid bin index (col, row) directly to
    CytAssist pixel (x, y): transform expects native (bin_level=1) resolution,
    so pre-scale the linear part by bin_level."""
    T = np.array(transform, dtype=float).copy()
    T[:, 0] *= bin_level
    T[:, 1] *= bin_level
    return T


def tissue_mask_to_oligo_grid(tissue_mask, transform, bin_level, grid_size,
                               supersample=4, aggregation="majority"):
    """Project a boolean tissue mask in CytAssist pixel space into the NxN
    oligo array-space grid.

    aggregation:
      "majority" (default) -- a bin is tissue if >=50% of the pixel area it
        covers is tissue. Verified best match (IoU 0.996) in a forward-
        project/invert self-consistency test against a real oligo grid.
      "any" -- a bin is tissue if ANY covered pixel is tissue (more
        permissive at tissue edges; IoU 0.991 in the same test).
    """
    if aggregation not in ("any", "majority"):
        raise ValueError(f"unknown aggregation {aggregation!r}, must be 'any' or 'majority'")
    T_bin = bin_to_pixel_matrix(transform, bin_level)
    S = int(supersample)
    T_super = T_bin.copy()
    T_super[:, 0] /= S
    T_super[:, 1] /= S
    M = T_super[:2, :].astype(np.float64)
    mask_u8 = (np.asarray(tissue_mask).astype(np.uint8) * 255)
    super_grid = cv2.warpAffine(
        mask_u8, M, (grid_size * S, grid_size * S),
        flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )
    super_grid = super_grid.reshape(grid_size, S, grid_size, S)
    frac = super_grid.mean(axis=(1, 3)) / 255.0
    if aggregation == "any":
        grid = frac > 0
    else:
        grid = frac >= 0.5
    return grid.astype(bool)


def build_oligo_from_tissue_mask(base_json_path, tissue_mask, aggregation="majority", supersample=4):
    """Load base_json_path, validate it, and return a new oligo base64 string
    built from tissue_mask (a boolean/0-1 array in CytAssist pixel space,
    matching the CytAssist image this base file's transform/cytAssistInfo
    were computed against). Raises rather than guessing if the base file's
    schema doesn't match this module's verified assumptions."""
    base = json.loads(Path(base_json_path).read_text())
    bin_level, transform, grid_size = validate_base_json(base)
    grid = tissue_mask_to_oligo_grid(tissue_mask, transform, bin_level, grid_size,
                                      supersample=supersample, aggregation=aggregation)
    return encode_oligo(grid), grid


def _self_check(base_json_path):
    """Round-trip + forward/inverse self-consistency check against a real
    base file's own oligo grid, with no external tissue mask needed -- run
    this against any new base JSON before trusting it for real encoding."""
    base = json.loads(Path(base_json_path).read_text())
    grid, bin_level, transform, grid_size = decode_oligo(base)
    print(f"grid_size={grid_size} bin_level={bin_level} tissue_fraction={grid.mean():.4f}")

    re_b64 = encode_oligo(grid)
    ok = re_b64 == base["oligo"]
    print(f"decode/encode byte-exact round-trip: {'PASS' if ok else 'FAIL'}")
    if not ok:
        raise SystemExit("round-trip check failed -- do not trust this encoder on this file")

    T_bin = bin_to_pixel_matrix(transform, bin_level)
    M_fwd = T_bin[:2, :].astype(np.float64)
    # canvas size doesn't need to be exact -- just large enough to hold the
    # real grid's projected extent; this is a self-consistency check, not a
    # real image.
    canvas_w = canvas_h = grid_size * 6
    synth_mask = cv2.warpAffine(
        (grid.astype(np.uint8) * 255), M_fwd, (canvas_w, canvas_h),
        flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ) > 0
    recon, _ = None, None
    recon_grid = tissue_mask_to_oligo_grid(synth_mask, transform, bin_level, grid_size,
                                            aggregation="majority")
    inter = (grid & recon_grid).sum()
    union = (grid | recon_grid).sum()
    iou = inter / union if union else 1.0
    print(f"forward-project / invert self-consistency IoU: {iou:.4f}")
    if iou < 0.95:
        raise SystemExit(f"self-consistency IoU {iou:.4f} is below the 0.95 threshold seen on "
                          "the verified reference file -- something about this file's transform "
                          "convention may differ; do not trust this encoder on it without "
                          "investigating further.")
    print("self-check PASSED")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("self-check", help="verify this module's transform convention "
                                                 "against a base file's own existing oligo grid")
    p_check.add_argument("base_json")

    p_build = sub.add_parser("encode", help="build a new oligo field from a tissue mask image")
    p_build.add_argument("base_json")
    p_build.add_argument("tissue_mask_png", help="a 0/255 (or 0/1) single-channel PNG in "
                                                  "CytAssist pixel space, e.g. from "
                                                  "tissue_detection_pipeline.py's *_tissue_mask.png")
    p_build.add_argument("--aggregation", choices=["any", "majority"], default="majority")
    p_build.add_argument("--supersample", type=int, default=4)
    p_build.add_argument("--output", default=None, help="write the full base JSON with oligo "
                                                          "replaced to this path; if omitted, "
                                                          "just prints the tissue-fraction summary")

    args = p.parse_args()
    if args.cmd == "self-check":
        _self_check(args.base_json)
    elif args.cmd == "encode":
        mask_img = cv2.imread(args.tissue_mask_png, cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise SystemExit(f"could not read {args.tissue_mask_png}")
        mask = mask_img > 0
        new_b64, grid = build_oligo_from_tissue_mask(
            args.base_json, mask, aggregation=args.aggregation, supersample=args.supersample)
        print(f"new oligo tissue fraction: {grid.mean():.4f}")
        if args.output:
            base = json.loads(Path(args.base_json).read_text())
            base["oligo"] = new_b64
            Path(args.output).write_text(json.dumps(base))
            print(f"wrote {args.output}")
        else:
            print("(pass --output to write a full base JSON with oligo replaced)")
