"""
Whole-image (no fragments) registration of VisiumHD HiRes microscopy to the
CytAssist reference image, built directly on palom's primitives
(https://github.com/labsyspharm/palom) -- not on the xenium_he_registration
package (that package assumes same-channel, same-FOV inputs like Xenium
DAPI/H&E, which doesn't hold here -- see README.md).

Pipeline, in order:
1. Downsample HiRes so its pixel size matches CytAssist's (physical um/px,
   not pixel-array shape -- the two images cover different fields of view).
2. Resolve the flip/rotation ambiguity (HiRes and CytAssist image the slide
   from different sides) via palom.register.match_test_flip_rotate. Its
   returned coordinate matrix is saved as-is (see "orientation_matrix"
   below) rather than assumed -- this is what lets
   build_full_res_deliverables.py handle ANY sample's orientation
   generically, with no hardcoded flip/rotate case.
3. Optionally apply one extra manual horizontal mirror (--extra-mirror-x),
   for samples where match_test_flip_rotate's own search picks the wrong
   orientation (it only tests half of the 16 possible flip x rotation
   combinations -- see the docstring on that flag below).
4. Fit a coarse affine: normally palom's ORB+RANSAC feature match
   (coarse_register_affine); optionally seeded manually (--manual-translation)
   or, if the ORB fit's scale is implausible, a translation-only whole-image
   template-matching fallback (phase_correlation_affine below).
5. Fit a per-block local refinement (palom's compute_shifts /
   constrain_shifts), with one safety net: blocks that have no raw local
   signal (typically tissue-free background) are NOT trusted to
   constrain_shifts()'s regression-based extrapolation -- which can be
   wildly wrong far outside its support region -- and are overwritten with
   the coarse-only matrix instead.

Output: `<sample>_transform.npz` (coarse affine + per-block refinement +
the orientation matrix needed to reconstruct native-resolution coordinates)
plus a CytAssist-resolution warped preview, for a fast visual sanity check.
Feed the npz into build_full_res_deliverables.py for the full-resolution
result and a separate exportable affine, and into report.py for a full
per-run QC report.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import dask.array as da
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hires_cytassist_common import downsample_to_physical_scale, um_per_pixel


def load_cytassist_rgb(path):
    rgb = tifffile.imread(path)
    if rgb.ndim == 3 and rgb.shape[0] in (3, 4) and rgb.shape[-1] not in (3, 4):
        rgb = np.moveaxis(rgb, 0, -1)
    return rgb


def phase_correlation_affine(ref_thumb, moving_thumb, n_peaks=5, peak_excl=100):
    """Translation-only coarse affine via whole-image template matching.

    Fallback for when palom's ORB+RANSAC coarse fit fails (scale far from
    1.0) -- typically repetitive/self-similar tissue where a self-consistent
    -but-wrong keypoint match set can outnumber correct matches and win
    RANSAC. Assumes scale=1, rotation=0 beyond the flip/rotate already
    resolved by match_test_flip_rotate, which is reasonable once that
    ambiguity is out of the way.

    ref_thumb and moving_thumb are frequently different shapes (HiRes and
    CytAssist commonly cover different physical extents, not a shared-corner
    crop of each other), so this treats the smaller image as a template and
    searches for its best-correlating position within the larger one via
    normalized cross-correlation (cv2.matchTemplate) -- unlike a naive
    same-shape phase_cross_correlation, this handles arbitrary relative
    offsets correctly.

    Also returns the top `n_peaks` distinct local maxima (masking a
    `peak_excl`-pixel box around each one found so far), so the caller can
    tell whether the winning peak is well-separated or effectively tied
    with a competing hypothesis (a real risk on repetitive tissue).
    """
    import cv2

    ref_sig = (255.0 - ref_thumb).astype("float32")
    mov_sig = (255.0 - moving_thumb).astype("float32")

    ref_fits_in_moving = (ref_sig.shape[0] <= mov_sig.shape[0]) and (ref_sig.shape[1] <= mov_sig.shape[1])
    moving_fits_in_ref = (mov_sig.shape[0] <= ref_sig.shape[0]) and (mov_sig.shape[1] <= ref_sig.shape[1])
    if ref_fits_in_moving:
        template, search, template_is_ref = ref_sig, mov_sig, True
    elif moving_fits_in_ref:
        template, search, template_is_ref = mov_sig, ref_sig, False
    else:
        # Neither fully contains the other (e.g. one taller but narrower) --
        # fall back to the largest common top-left crop as a last resort.
        h, w = min(ref_sig.shape[0], mov_sig.shape[0]), min(ref_sig.shape[1], mov_sig.shape[1])
        template, search, template_is_ref = ref_sig[:h, :w], mov_sig[:h, :w], True

    scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
    peaks = []
    remaining = scores.copy()
    for _ in range(n_peaks):
        _, score, _, (x, y) = cv2.minMaxLoc(remaining)
        peaks.append((float(score), y, x))
        remaining[max(0, y - peak_excl):y + peak_excl, max(0, x - peak_excl):x + peak_excl] = -2

    _, best_y, best_x = peaks[0]
    if template_is_ref:
        # ref's top-left lands at (best_y,best_x) within moving -> moving-space
        # point (best_x,best_y) is ref-space (0,0).
        dx, dy = -best_x, -best_y
    else:
        # moving's top-left found at (best_y,best_x) within ref -> moving-space
        # (0,0) is ref-space (best_x,best_y).
        dx, dy = best_x, best_y

    affine_matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
    return affine_matrix, peaks


def override_tissue_free_blocks(matrices_np, raw_valid, grid_shape, coarse_matrix):
    """Blocks with no raw local-refinement signal (typically tissue-free
    background) are, by default, filled in by palom's constrain_shifts()
    via regression-based extrapolation from the blocks that do have signal
    -- which can be wildly wrong far outside its support region. Overwrite
    every such block's matrix with the coarse-only affine instead, which is
    always a safe fallback: a block with no real local signal gets no
    correction rather than an extrapolated, possibly bogus one.
    """
    gr, gc = grid_shape
    raw_valid_grid = raw_valid.reshape(gr, gc)
    n_overridden = 0
    for i in range(gr):
        for j in range(gc):
            if not raw_valid_grid[i, j]:
                matrices_np[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3] = coarse_matrix
                n_overridden += 1
    return n_overridden


def main(hires_path, cytassist_path, output_dir, sample="sample",
         block_step=128, block_size=None, n_keypoints=12000, thumbnail_level=2,
         scale_tol=0.15, extra_mirror_x=False, manual_translation=None):
    import palom  # only needed here; keep it optional at import time for common.py users
    import palom.register_util as register_util

    logging.getLogger("palom").setLevel(logging.WARNING)
    block_size = block_size or 2 * block_step
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cyt_um_per_px = um_per_pixel(cytassist_path)
    if cyt_um_per_px is None:
        raise ValueError(f"{cytassist_path}: no resolution tags found")
    print(f"CytAssist pixel size: {cyt_um_per_px:.4f} um/px")

    # Hd: HiRes downsampled to match CytAssist's pixel size, but NOT YET
    # flipped/rotated/mirrored into CytAssist's orientation.
    hires_rgb = downsample_to_physical_scale(hires_path, cyt_um_per_px)
    hd_shape = hires_rgb.shape[:2]
    cyt_rgb = load_cytassist_rgb(cytassist_path)
    hires_gray = hires_rgb.astype("float32").mean(axis=-1)
    cyt_gray = cyt_rgb.astype("float32").mean(axis=-1)

    warnings = []  # collected here, saved to <sample>_run_stats.json for report.py

    print("Resolving flip/rotation ambiguity...")
    flip_rotate_func, orientation_matrix = palom.register.match_test_flip_rotate(cyt_gray, hires_gray)
    hires_gray = flip_rotate_func(hires_gray)
    hires_rgb = np.stack([flip_rotate_func(hires_rgb[..., c]) for c in range(3)], axis=-1)
    print(f"Oriented HiRes shape: {hires_rgb.shape}")

    if extra_mirror_x:
        # match_test_flip_rotate does its own ORB-based orientation search,
        # and on highly repetitive/self-similar tissue it can be fooled the
        # same way the coarse ORB+RANSAC fit can: it only tests half of the
        # 16 possible flip x rotation combinations (an optimization that
        # assumes rotational equivalence, which doesn't hold for non-square
        # images), so it can miss an orientation that needed a horizontal
        # mirror it never tried. Found on NMR3_Colon via an independent
        # check (ring-centroid pattern voting against a rough user-supplied
        # location hint): with this extra mirror, all 6 CytAssist tissue
        # rings simultaneously matched HiRes rings under one consistent
        # translation (vs. at most 2/6 without it).
        print("Applying additional user-specified horizontal (x) mirror on top of "
              "match_test_flip_rotate's own choice.")
        extra_mirror_matrix = register_util.get_flip_mx(hires_gray.shape, 1)
        orientation_matrix = extra_mirror_matrix @ orientation_matrix
        hires_gray = hires_gray[:, ::-1].copy()
        hires_rgb = hires_rgb[:, ::-1, :].copy()

    chunk = (block_step, block_step)
    hires_gray_da = da.from_array(hires_gray, chunks=chunk)
    cyt_gray_da = da.from_array(cyt_gray, chunks=chunk)
    factor = 2 ** thumbnail_level
    ref_thumbnail = cyt_gray_da[::factor, ::factor].compute()
    moving_thumbnail = hires_gray_da[::factor, ::factor].compute()
    aligner = palom.align.Aligner(
        ref_img=cyt_gray_da, moving_img=hires_gray_da,
        ref_thumbnail=ref_thumbnail,
        moving_thumbnail=moving_thumbnail,
        ref_thumbnail_down_factor=factor, moving_thumbnail_down_factor=factor,
    )

    if manual_translation is not None:
        # Skip both the ORB fit and the correlation fallback entirely -- use
        # a translation-only coarse affine supplied directly (in full
        # CytAssist-scale pixel units; converted to thumbnail-scale here
        # since aligner.affine_matrix rescales coarse_affine_matrix back up
        # automatically via ref/moving_thumbnail_down_factor). For samples
        # where automatic coarse registration can't be trusted at all (e.g.
        # repetitive tissue + limited FOV overlap -- see NMR3_Colon), this
        # lets a manually-derived translation (from a rough location hint
        # plus e.g. ring-centroid matching) be used directly.
        man_dx, man_dy = manual_translation
        print(f"Using manually-supplied translation-only coarse affine: "
              f"dx={man_dx:.1f} dy={man_dy:.1f} (full/CytAssist-scale units) -- "
              "skipping ORB fit and correlation fallback entirely.")
        thumb_matrix = np.array([[1.0, 0.0, man_dx / factor], [0.0, 1.0, man_dy / factor]])
        aligner.coarse_affine_matrix = np.vstack([thumb_matrix, [0, 0, 1]])
        used_phase_fallback = False
    else:
        aligner.coarse_register_affine(n_keypoints=n_keypoints)
        used_phase_fallback = False

    m = aligner.affine_matrix
    scale_x, scale_y = np.hypot(m[0, 0], m[1, 0]), np.hypot(m[0, 1], m[1, 1])
    angle = np.degrees(np.arctan2(m[1, 0], m[0, 0]))
    print(f"Coarse affine: scale=({scale_x:.4f},{scale_y:.4f}) angle={angle:.2f} deg "
          "-- scale should be close to 1.0")

    if manual_translation is None and (abs(scale_x - 1) > scale_tol or abs(scale_y - 1) > scale_tol):
        msg = ("coarse scale is far from 1.0 -- ORB-based fit likely failed (common on "
               "repetitive/self-similar tissue). Falling back to phase-cross-correlation "
               "translation estimate...")
        print(f"WARNING: {msg}")
        warnings.append(msg)
        try:
            fallback_matrix, peaks = phase_correlation_affine(ref_thumbnail, moving_thumbnail)
            print("Top correlation peaks (score, y, x) at thumbnail scale -- check these "
                  "aren't near-tied (a sign of repetitive-tissue ambiguity):")
            for score, py, px in peaks:
                print(f"    {score:.4f}  y={py} x={px}")
            # fallback_matrix's translation is thumbnail-scale (same convention
            # as coarse_register_affine's own output) -- don't rescale by
            # `factor` here, aligner.affine_matrix does that automatically.
            aligner.coarse_affine_matrix = np.vstack([fallback_matrix, [0, 0, 1]])
            m = aligner.affine_matrix
            scale_x, scale_y = np.hypot(m[0, 0], m[1, 0]), np.hypot(m[0, 1], m[1, 1])
            angle = np.degrees(np.arctan2(m[1, 0], m[0, 0]))
            print(f"Correlation-search fallback affine: scale=({scale_x:.4f},{scale_y:.4f}) "
                  f"angle={angle:.2f} deg, translation=({m[0,2]:.1f},{m[1,2]:.1f})")
            if len(peaks) > 1 and peaks[0][0] - peaks[1][0] < 0.05:
                msg = ("top two correlation peaks are within 0.05 of each other -- this "
                       "translation may be ambiguous (repetitive tissue). Inspect the QC "
                       "report carefully before trusting this run.")
                print(f"WARNING: {msg}")
                warnings.append(msg)
            used_phase_fallback = True
        except Exception as e:
            msg = (f"correlation-search fallback failed too ({e}) -- kept the ORB-based fit "
                   "despite the scale warning; inspect the result carefully.")
            print(f"WARNING: {msg}")
            warnings.append(msg)

    aligner.block_size, aligner.block_step = block_size, block_step
    print("grid_shape:", aligner.ref_img.numblocks)
    aligner.compute_shifts()
    raw_valid = np.isfinite(np.linalg.norm(aligner.shifts, axis=1))
    print(f"Blocks with valid local signal: {raw_valid.sum()}/{len(aligner.shifts)}")
    if np.prod(aligner.grid_shape) >= 4:
        try:
            aligner.constrain_shifts()
        except Exception as e:
            print(f"constrain_shifts failed ({e}), using raw shifts")

    matrices = aligner.block_affine_matrices_da
    matrices_np = matrices.compute() if hasattr(matrices, "compute") else np.asarray(matrices)
    n_overridden = override_tissue_free_blocks(matrices_np, raw_valid, aligner.grid_shape, m)
    if n_overridden:
        gr, gc = aligner.grid_shape
        print(f"Overrode {n_overridden}/{gr * gc} blocks with no raw local signal "
              "-> coarse-only matrix (instead of extrapolated regression).")

    npz_path = output_dir / f"{sample}_transform.npz"
    np.savez_compressed(
        npz_path,
        block_affine_matrices=matrices_np,
        coarse_affine_matrix=m,
        grid_shape=np.asarray(aligner.grid_shape),
        block_step=block_step,
        block_size=block_size,
        ref_shape=np.asarray(cyt_rgb.shape[:2]),
        # Ho: fully-oriented (flip/rotate + optional extra mirror) HiRes shape,
        # at CytAssist pixel scale -- what compute_shifts/block matrices are in.
        moving_shape_oriented=np.asarray(hires_rgb.shape[:2]),
        # Hd: same HiRes image, same pixel scale, but BEFORE any
        # flip/rotate/mirror -- together with orientation_matrix this is
        # enough to map any Ho/C-space point back to native-resolution,
        # original-orientation HiRes coordinates, generically (no hardcoded
        # flip/rotate assumption needed downstream).
        moving_shape_pre_orientation=np.asarray(hd_shape),
        orientation_matrix=orientation_matrix,
        used_phase_fallback=used_phase_fallback,
        n_blocks_overridden=n_overridden,
    )
    print(f"Wrote {npz_path}")

    # Quick CytAssist-resolution preview, for a fast visual/coverage sanity
    # check. Uses the CORRECTED matrices (with the tissue-free-block
    # override applied above), not palom's original per-block matrices.
    matrices_corrected = da.from_array(matrices_np, chunks=3)
    hires_color_chw = da.from_array(np.moveaxis(hires_rgb, -1, 0), chunks=(1,) + chunk)
    warped = palom.align.block_affine_transformed_moving_img(
        ref_img=cyt_gray_da, moving_img=hires_color_chw, mxs=matrices_corrected).compute()
    coverage = (warped.sum(axis=0) > 0).mean()
    print(f"Preview coverage: {coverage:.4f}")
    if coverage < 0.9:
        msg = "less than 90% coverage -- inspect the preview before trusting this run."
        print(f"WARNING: {msg}")
        warnings.append(msg)
    preview_path = output_dir / f"{sample}_preview_warped_to_cytassist.ome.tif"
    palom.pyramid.write_pyramid(mosaics=[da.from_array(warped, chunks=(1,) + chunk)],
                                 output_path=str(preview_path), pixel_size=cyt_um_per_px)
    print(f"Wrote {preview_path}")

    gr, gc = aligner.grid_shape
    stats = {
        "sample": sample, "hires_path": str(hires_path), "cytassist_path": str(cytassist_path),
        "cyt_um_per_px": cyt_um_per_px, "block_step": block_step, "block_size": block_size,
        "n_keypoints": n_keypoints, "extra_mirror_x": extra_mirror_x,
        "manual_translation": list(manual_translation) if manual_translation else None,
        "used_phase_fallback": used_phase_fallback,
        "coarse_scale": [float(scale_x), float(scale_y)], "coarse_angle_deg": float(angle),
        "coarse_translation": [float(m[0, 2]), float(m[1, 2])],
        "grid_shape": [int(gr), int(gc)],
        "n_blocks_total": int(gr * gc), "n_blocks_valid_raw": int(raw_valid.sum()),
        "n_blocks_overridden": n_overridden, "coverage": float(coverage), "warnings": warnings,
    }
    stats_path = output_dir / f"{sample}_run_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {stats_path}")
    return {"transform": npz_path, "preview": preview_path, "stats": stats_path, **stats}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", default=None, help="prepend to sys.path if `palom` isn't importable")
    p.add_argument("--hires", required=True)
    p.add_argument("--cytassist", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sample", default="sample")
    p.add_argument("--block-step", type=int, default=128, help="block-wise refinement grid step, in pixels at CytAssist scale")
    p.add_argument("--block-size", type=int, default=None, help="default: 2x --block-step")
    p.add_argument("--n-keypoints", type=int, default=12000)
    p.add_argument("--thumbnail-level", type=int, default=2)
    p.add_argument("--scale-tol", type=float, default=0.15, help="trigger phase-correlation fallback if |scale-1| exceeds this")
    p.add_argument("--extra-mirror-x", action="store_true",
                    help="apply an additional horizontal mirror on top of match_test_flip_rotate's "
                         "own orientation choice -- needed on samples where that function itself gets "
                         "fooled by repetitive/self-similar tissue (see NMR3_Colon in the project notes)")
    p.add_argument("--manual-translation", type=float, nargs=2, default=None, metavar=("DX", "DY"),
                    help="skip the ORB fit and correlation fallback entirely and use this translation-only "
                         "coarse affine directly (full/CytAssist-scale pixel units, applied AFTER "
                         "--extra-mirror-x if both are given). For samples where automatic coarse "
                         "registration can't be trusted (e.g. repetitive tissue + limited FOV overlap).")
    args = p.parse_args()
    if args.source_dir:
        sys.path.insert(0, args.source_dir)
    main(args.hires, args.cytassist, args.output_dir, sample=args.sample,
         block_step=args.block_step, block_size=args.block_size,
         n_keypoints=args.n_keypoints, thumbnail_level=args.thumbnail_level,
         scale_tol=args.scale_tol, extra_mirror_x=args.extra_mirror_x,
         manual_translation=tuple(args.manual_translation) if args.manual_translation else None)
