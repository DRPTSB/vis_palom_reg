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
   (coarse_register_affine); or, if the ORB fit's scale/angle looks
   implausible, a rotation-aware whole-image template-matching fallback
   (search_rotation_translation_affine); or, when the automatic fit can't be
   trusted at all (e.g. repetitive tissue + limited FOV overlap), a full
   affine (rotation + translation) refined near a manually-supplied seed
   translation (--manual-translation, see
   refine_rotation_and_translation_near_seed) -- never a bare identity-
   rotation assumption, which was found to silently miss a real ~0.5deg
   rotation on NMR3_DRG (2026-09-27).
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
from hires_cytassist_common import downsample_to_physical_scale, um_per_pixel, setup_pipeline_log


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


def search_rotation_translation_affine(ref_thumb, moving_thumb, angle_range=(-10, 10),
                                        angle_step=0.5, n_peaks=3, peak_excl=50):
    """Combined rotation+translation coarse-fit fallback, more robust than
    ORB+RANSAC on repetitive/self-similar tissue (colon rings, scattered DRG
    explants, etc. -- see project notes finding 6). For each candidate angle,
    rotates moving_thumb about its center, then finds the best-correlating
    TRANSLATION against ref_thumb via a real sliding-window search
    (cv2.matchTemplate, same "smaller image is the template" convention as
    phase_correlation_affine below) rather than assuming zero translation --
    a plain top-left-crop correlation at each angle is meaningless before any
    translation is known and gives noise-level scores on real data.

    Returns (affine_2x3, best_angle_deg, best_score, per_angle_scores) all in
    THUMBNAIL-scale units (same convention as coarse_register_affine's own
    output -- aligner.affine_matrix rescales automatically).
    """
    import cv2

    ref_sig = (255.0 - ref_thumb).astype("float32")
    h, w = moving_thumb.shape[:2]
    center = (w / 2.0, h / 2.0)

    best = None  # (score, angle, dx, dy, peaks)
    per_angle_scores = []
    angles = np.arange(angle_range[0], angle_range[1] + angle_step, angle_step)
    for angle in angles:
        rot_2x3 = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(moving_thumb, rot_2x3, (w, h))
        mov_sig = (255.0 - rotated).astype("float32")

        ref_fits = ref_sig.shape[0] <= mov_sig.shape[0] and ref_sig.shape[1] <= mov_sig.shape[1]
        mov_fits = mov_sig.shape[0] <= ref_sig.shape[0] and mov_sig.shape[1] <= ref_sig.shape[1]
        if ref_fits:
            template, search, template_is_ref = ref_sig, mov_sig, True
        elif mov_fits:
            template, search, template_is_ref = mov_sig, ref_sig, False
        else:
            hh, ww = min(ref_sig.shape[0], mov_sig.shape[0]), min(ref_sig.shape[1], mov_sig.shape[1])
            template, search, template_is_ref = ref_sig[:hh, :ww], mov_sig[:hh, :ww], True

        scores = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        peaks = []
        remaining = scores.copy()
        for _ in range(n_peaks):
            _, score, _, (x, y) = cv2.minMaxLoc(remaining)
            peaks.append((float(score), int(y), int(x)))
            remaining[max(0, y - peak_excl):y + peak_excl, max(0, x - peak_excl):x + peak_excl] = -2

        top_score, top_y, top_x = peaks[0]
        if template_is_ref:
            dx, dy = -top_x, -top_y
        else:
            dx, dy = top_x, top_y

        per_angle_scores.append((float(angle), top_score))
        if best is None or top_score > best[0]:
            best = (top_score, float(angle), float(dx), float(dy), peaks)

    best_score, best_angle, dx, dy, best_peaks = best
    rot_2x3 = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    # Translate AFTER rotate (dx,dy found on the already-rotated thumbnail) --
    # for an affine applied as M @ [x,y,1], adding to the translation column
    # is correct since it's the last transform applied.
    rot_2x3 = rot_2x3.copy()
    rot_2x3[0, 2] += dx
    rot_2x3[1, 2] += dy
    return rot_2x3, best_angle, best_score, per_angle_scores, best_peaks


def refine_rotation_and_translation_near_seed(ref_thumb, moving_thumb, seed_dx, seed_dy,
                                               angle_range=(-10, 10), angle_step=0.5,
                                               search_margin=40.0):
    """Full affine (rotation + translation) refinement seeded by an already-
    known-good translation (e.g. --manual-translation), for samples where an
    open-ended rotation+translation search (search_rotation_translation_affine
    above) converges to a wrong, far-away local optimum on repetitive/
    self-similar tissue -- which is exactly why a manual seed was needed in
    the first place. Unlike that function, the translation search here is
    restricted to a small window (+/- search_margin px, THUMBNAIL-scale)
    around the known-good seed, at every candidate angle, so it can refine
    rotation and make a small translation correction without ever being able
    to jump back to that wrong distant optimum.

    seed_dx, seed_dy: THUMBNAIL-scale, same sign convention as
    phase_correlation_affine's/search_rotation_translation_affine's own
    output (i.e. what you'd pass straight into the coarse-affine translation
    column after dividing the full-scale manual translation by the
    thumbnail downsample factor).

    Returns (affine_2x3, best_angle_deg, best_score, per_angle_scores,
    refined_dx, refined_dy) -- all thumbnail-scale, same convention as
    search_rotation_translation_affine.
    """
    import cv2

    ref_sig = (255.0 - ref_thumb).astype("float32")
    h, w = moving_thumb.shape[:2]
    center = (w / 2.0, h / 2.0)
    # moving's own canvas shape is unchanged by warpAffine's rotation, so
    # which image is the "template" vs. the "search" area doesn't depend on
    # angle -- resolve it once, matching the other two functions' convention.
    ref_fits_in_moving = ref_sig.shape[0] <= h and ref_sig.shape[1] <= w

    best = None  # (score, angle, dx, dy)
    per_angle_scores = []
    angles = np.arange(angle_range[0], angle_range[1] + angle_step, angle_step)
    for angle in angles:
        rot_2x3 = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        rotated = cv2.warpAffine(moving_thumb, rot_2x3, (w, h))
        mov_sig = (255.0 - rotated).astype("float32")

        if ref_fits_in_moving:
            template, search, template_is_ref = ref_sig, mov_sig, True
        elif mov_sig.shape[0] <= ref_sig.shape[0] and mov_sig.shape[1] <= ref_sig.shape[1]:
            template, search, template_is_ref = mov_sig, ref_sig, False
        else:
            hh, ww = min(ref_sig.shape[0], mov_sig.shape[0]), min(ref_sig.shape[1], mov_sig.shape[1])
            template, search, template_is_ref = ref_sig[:hh, :ww], mov_sig[:hh, :ww], True

        th, tw = template.shape[:2]
        sh, sw = search.shape[:2]
        # Expected top-left placement of `template` within `search`, per the
        # seed translation and this module's sign convention.
        if template_is_ref:
            exp_x, exp_y = -seed_dx, -seed_dy
        else:
            exp_x, exp_y = seed_dx, seed_dy

        x0 = int(np.clip(np.floor(exp_x - search_margin), 0, max(sw - 1, 0)))
        y0 = int(np.clip(np.floor(exp_y - search_margin), 0, max(sh - 1, 0)))
        x1 = int(np.clip(np.ceil(exp_x + tw + search_margin), x0 + 1, sw))
        y1 = int(np.clip(np.ceil(exp_y + th + search_margin), y0 + 1, sh))
        if (y1 - y0) < th or (x1 - x0) < tw:
            # Seed placed the template too close to search's edge for the
            # crop to fully contain it -- widen to the template's own size,
            # clipped to the search image's bounds, rather than crash.
            y1 = min(sh, y0 + th)
            x1 = min(sw, x0 + tw)
            y0 = max(0, y1 - th)
            x0 = max(0, x1 - tw)
        crop = search[y0:y1, x0:x1]

        scores = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, (local_x, local_y) = cv2.minMaxLoc(scores)
        top_x, top_y = x0 + local_x, y0 + local_y

        if template_is_ref:
            dx, dy = -top_x, -top_y
        else:
            dx, dy = top_x, top_y

        per_angle_scores.append((float(angle), float(score)))
        if best is None or score > best[0]:
            best = (float(score), float(angle), float(dx), float(dy))

    best_score, best_angle, dx, dy = best
    rot_2x3 = cv2.getRotationMatrix2D(center, best_angle, 1.0)
    rot_2x3 = rot_2x3.copy()
    rot_2x3[0, 2] += dx
    rot_2x3[1, 2] += dy
    return rot_2x3, best_angle, best_score, per_angle_scores, dx, dy


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
         scale_tol=0.15, extra_mirror_x=False, manual_translation=None,
         rotation_search_deg=10.0, rotation_search_step=0.5,
         manual_translation_search_margin=150.0):
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

    rotation_correction_deg = None
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
        # Skip the ORB fit and the OPEN-ENDED correlation fallback entirely
        # (both already known to fail for this sample -- typically repetitive
        # tissue + limited FOV overlap, e.g. NMR3_Colon/NMR3_DRG), but DO NOT
        # assume rotation=0: a manually-seeded translation only tells us
        # roughly where the image sits, not that HiRes and CytAssist are
        # perfectly axis-aligned. Bug found 2026-09-27 (NMR3_DRG): assuming
        # rotation=0 here left a real ~0.5deg rotation completely out of the
        # exported coarse affine (cytAssistInfo.transformImages) -- the
        # per-block local refinement still ends up geometrically correct (a
        # small rotation shows up as a smooth per-block translation gradient
        # it can absorb), but the exported single global affine was then
        # wrong by ~30px at the image edges, and the local refinement was
        # carrying a full-frame rotation it wasn't designed for. Fix: refine
        # a FULL affine (rotation sweep, +/- --rotation-search-deg, PLUS a
        # translation search) around the manual seed, with the translation
        # search restricted to a small window (+/-
        # --manual-translation-search-margin px, full-scale) around that
        # seed -- narrow enough that it can't re-converge to the same wrong,
        # far-away optimum that made the manual override necessary in the
        # first place, but wide enough to find the true rotation and any
        # small accompanying translation correction.
        man_dx, man_dy = manual_translation
        print(f"Manually-supplied seed translation: dx={man_dx:.1f} dy={man_dy:.1f} "
              "(full/CytAssist-scale units). Refining a full affine (rotation "
              f"+/-{rotation_search_deg:.1f} deg, translation search restricted to "
              f"+/-{manual_translation_search_margin:.0f}px of the seed, full-scale) "
              "around it, rather than assuming rotation=0 -- skipping ORB and the "
              "open-ended correlation fallback (both already ruled out for this sample).")
        seed_dx_thumb, seed_dy_thumb = man_dx / factor, man_dy / factor
        margin_thumb = manual_translation_search_margin / factor
        refined_matrix, best_angle, best_score, per_angle_scores, ref_dx_thumb, ref_dy_thumb = (
            refine_rotation_and_translation_near_seed(
                ref_thumbnail, moving_thumbnail, seed_dx_thumb, seed_dy_thumb,
                angle_range=(-rotation_search_deg, rotation_search_deg),
                angle_step=rotation_search_step, search_margin=margin_thumb,
            )
        )
        rotation_correction_deg = best_angle
        ref_dx_full, ref_dy_full = ref_dx_thumb * factor, ref_dy_thumb * factor
        print(f"Manual-translation-seeded affine refinement: best_angle={best_angle:+.2f} deg "
              f"(score={best_score:.4f}); refined translation=({ref_dx_full:.1f},{ref_dy_full:.1f}) "
              f"full-scale (seed was ({man_dx:.1f},{man_dy:.1f}))")
        if abs(best_angle) >= (rotation_search_deg - rotation_search_step):
            msg = (f"winning angle ({best_angle:+.1f} deg) is at the edge of the searched "
                   f"+/-{rotation_search_deg:.1f} deg range -- the true angle may lie outside "
                   "it; widen --rotation-search-deg and re-run if the QC report looks wrong.")
            print(f"WARNING: {msg}")
            warnings.append(msg)
        if (abs(ref_dx_full - man_dx) > manual_translation_search_margin * 0.9
                or abs(ref_dy_full - man_dy) > manual_translation_search_margin * 0.9):
            msg = (f"refined translation ({ref_dx_full:.1f},{ref_dy_full:.1f}) landed close to "
                   f"the edge of the +/-{manual_translation_search_margin:.0f}px search window "
                   "around the manual seed -- consider widening "
                   "--manual-translation-search-margin and re-running.")
            print(f"WARNING: {msg}")
            warnings.append(msg)
        aligner.coarse_affine_matrix = np.vstack([refined_matrix, [0, 0, 1]])
        used_phase_fallback = False
    else:
        aligner.coarse_register_affine(n_keypoints=n_keypoints)
        used_phase_fallback = False

    m = aligner.affine_matrix
    scale_x, scale_y = np.hypot(m[0, 0], m[1, 0]), np.hypot(m[0, 1], m[1, 1])
    angle = np.degrees(np.arctan2(m[1, 0], m[0, 0]))
    print(f"Coarse affine: scale=({scale_x:.4f},{scale_y:.4f}) angle={angle:.2f} deg "
          "-- scale should be close to 1.0")

    # A valid ORB fit can legitimately land near ANY multiple of 90 degrees, not just
    # near 0: match_test_flip_rotate's own orientation search doesn't always fully
    # resolve a 90-degree ambiguity, and ORB can validly refine on top of that with an
    # extra ~90/180/270 degrees. Bug found 2026-09-27: flagging any |angle| > 15 as
    # implausible wrongly overrode a real, spaceranger-verified-correct fit on NMR2_DRG
    # (angle=90.67, exactly reproducing the archived good transform.npz) with a worse
    # fallback fit. Check distance to the NEAREST multiple of 90 instead.
    angle_residual = angle % 90.0
    angle_residual = min(angle_residual, 90.0 - angle_residual)
    angle_implausible = angle_residual > 15.0
    if manual_translation is None and (abs(scale_x - 1) > scale_tol or abs(scale_y - 1) > scale_tol
                                        or angle_implausible):
        reason = "scale far from 1.0" if not angle_implausible else "implausible rotation angle"
        if angle_implausible and (abs(scale_x - 1) > scale_tol or abs(scale_y - 1) > scale_tol):
            reason = "scale far from 1.0 and implausible rotation angle"
        msg = (f"coarse ORB-based fit looks wrong ({reason}) -- likely failed (common on "
               "repetitive/self-similar tissue). Falling back to a rotation-aware "
               f"correlation search (+/-{rotation_search_deg:.1f} deg)...")
        print(f"WARNING: {msg}")
        warnings.append(msg)
        try:
            fallback_matrix, best_angle, best_score, per_angle_scores, peaks = (
                search_rotation_translation_affine(
                    ref_thumbnail, moving_thumbnail,
                    angle_range=(-rotation_search_deg, rotation_search_deg),
                    angle_step=rotation_search_step,
                )
            )
            rotation_correction_deg = best_angle
            print(f"Rotation+translation search: best_angle={best_angle:+.2f} deg "
                  f"(score={best_score:.4f})")
            print("Top correlation peaks at the winning angle (score, y, x) at thumbnail "
                  "scale -- check these aren't near-tied (a sign of repetitive-tissue "
                  "ambiguity):")
            for score, py, px in peaks:
                print(f"    {score:.4f}  y={py} x={px}")
            if abs(best_angle) >= (rotation_search_deg - rotation_search_step):
                msg = (f"winning angle ({best_angle:+.1f} deg) is at the edge of the searched "
                       f"+/-{rotation_search_deg:.1f} deg range -- the true angle may lie "
                       "outside it; widen --rotation-search-deg and re-run if the QC report "
                       "looks wrong.")
                print(f"WARNING: {msg}")
                warnings.append(msg)
            # fallback_matrix's translation is thumbnail-scale (same convention
            # as coarse_register_affine's own output) -- don't rescale by
            # `factor` here, aligner.affine_matrix does that automatically.
            aligner.coarse_affine_matrix = np.vstack([fallback_matrix, [0, 0, 1]])
            m = aligner.affine_matrix
            scale_x, scale_y = np.hypot(m[0, 0], m[1, 0]), np.hypot(m[0, 1], m[1, 1])
            angle = np.degrees(np.arctan2(m[1, 0], m[0, 0]))
            print(f"Rotation-aware fallback affine: scale=({scale_x:.4f},{scale_y:.4f}) "
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
        rotation_search_correction_deg=np.nan if rotation_correction_deg is None else rotation_correction_deg,
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
        "rotation_search_correction_deg": rotation_correction_deg,
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
                    help="skip the ORB fit and the open-ended correlation fallback, seeding a full "
                         "affine (rotation + translation) refinement at this translation instead of "
                         "assuming rotation=0 (full/CytAssist-scale pixel units, applied AFTER "
                         "--extra-mirror-x if both are given). See --rotation-search-deg/-step for "
                         "the rotation search range and --manual-translation-search-margin for how "
                         "far the translation may move from this seed. For samples where automatic "
                         "coarse registration can't be trusted at all (e.g. repetitive tissue + "
                         "limited FOV overlap).")
    p.add_argument("--manual-translation-search-margin", type=float, default=150.0,
                    help="with --manual-translation: how far (px, full/CytAssist-scale) the "
                         "translation search is allowed to move away from the manual seed while "
                         "refining rotation. Wide enough to correct a real small offset error in "
                         "the seed, narrow enough to never re-converge to the wrong, far-away "
                         "optimum that made the manual override necessary in the first place. "
                         "Default 150.0.")
    p.add_argument("--rotation-search-deg", type=float, default=10.0,
                    help="explicit small-angle rotation search range in degrees (+/-), applied before the "
                         "ORB coarse fit to catch residual rotations ORB under-detects on repetitive/"
                         "low-texture tissue. 0 disables it. Default 10.0.")
    p.add_argument("--rotation-search-step", type=float, default=0.5,
                    help="rotation search angle step in degrees. Default 0.5.")
    p.add_argument("--log-dir", default=None,
                    help="where <sample>_pipeline.log lives; default: --output-dir. Set "
                         "explicitly when a stage's own --output-dir isn't the shared "
                         "per-sample directory (run_pipeline.py always sets this).")
    args = p.parse_args()
    setup_pipeline_log(args.log_dir or args.output_dir, args.sample, "register.py")
    if args.source_dir:
        sys.path.insert(0, args.source_dir)
    main(args.hires, args.cytassist, args.output_dir, sample=args.sample,
         block_step=args.block_step, block_size=args.block_size,
         n_keypoints=args.n_keypoints, thumbnail_level=args.thumbnail_level,
         scale_tol=args.scale_tol, extra_mirror_x=args.extra_mirror_x,
         manual_translation=tuple(args.manual_translation) if args.manual_translation else None,
         rotation_search_deg=args.rotation_search_deg, rotation_search_step=args.rotation_search_step,
         manual_translation_search_margin=args.manual_translation_search_margin)
