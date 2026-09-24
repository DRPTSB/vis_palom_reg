# visium-hires-cytassist-registration

Registers a VisiumHD/CytAssist HiRes microscopy image to its CytAssist reference
image, producing the inputs Space Ranger needs for manual alignment: a
corrected image with the local (block-wise) warp baked in, and a global affine
JSON in the same schema as the slide's own auto-generated fiducial-alignment
file.

Built directly on [`palom`](https://github.com/labsyspharm/palom)'s
primitives (`Aligner`, `coarse_register_affine`, `compute_shifts`,
`block_affine_matrices_da`) — not on `xenium_he_registration`, whose
same-channel/same-FOV assumptions don't hold for HiRes/CytAssist pairs.

## Why this exists

A single global affine can't represent the small (a few px at CytAssist
scale, up to ~100+px at native HiRes scale) but spatially coherent local
distortion between a HiRes scan and its CytAssist capture. `palom`'s
block-wise refinement finds that distortion, but Space Ranger's manual
alignment input wants one matrix, not a per-block warp field. This pipeline
splits the result into the two pieces Space Ranger actually accepts:

1. **`<sample>_local_warp_only.ome.tif`** — the original HiRes image, full
   native resolution, original orientation, with *only* the local
   (block-wise) correction baked into the pixels.
2. **`<sample>_global_affine.json`** — the coarse affine (flip/rotation +
   scale + translation) as a `cytAssistInfo.transformImages`-schema JSON,
   the same schema the slide's own auto-generated fiducial file uses.

Feed both into `spaceranger count`: the image via `--image`, the JSON via
`--loupe-alignment`. See "Space Ranger integration" below — **this requires
a full rerun of `spaceranger count`**, there is no partial-alignment-only
mode.

## Install

```bash
pip install -r requirements.txt
```

## Pipeline

```
register.py  -->  <sample>_transform.npz, _run_stats.json, _preview_warped_to_cytassist.ome.tif
     |
     +--> report.py                     -->  <sample>_report.html + QC images
     +--> build_full_res_deliverables.py -->  <sample>_local_warp_only.ome.tif, _global_affine.json
```

### 1. `register.py` — coarse + block-wise registration

```bash
python register.py \
  --hires HiRes.tif --cytassist CytAssist.tif \
  --output-dir outputs --sample SAMPLE_NAME \
  --block-step 128 --n-keypoints 12000
```

Steps, in order:

1. Downsamples HiRes to CytAssist's physical pixel size (TIFF resolution
   tags — the two images don't cover the same field of view, so matching
   pixel-array shape instead would be wrong).
2. Resolves the flip/rotation ambiguity via `palom.register.match_test_flip_rotate`,
   saving its result (`orientation_matrix`) directly rather than assuming a
   fixed case — nothing downstream is hardcoded to one orientation.
3. Fits a coarse affine (ORB+RANSAC by default; see "When automatic
   registration fails" below for when it isn't trustworthy).
4. Fits per-block local refinement, with a safety net: blocks with no raw
   local signal (typically tissue-free background) are overwritten with the
   coarse-only matrix instead of `palom`'s regression-extrapolated one,
   which can be wildly wrong far outside its support region.

Outputs `<sample>_transform.npz` (coarse affine + per-block matrices +
orientation matrix + shapes needed to reconstruct native-resolution
coordinates), `<sample>_run_stats.json` (metrics + any warnings), and a
quick CytAssist-scale preview warp for a fast sanity check.

Useful flags:

- `--n-keypoints` (default 12000) — texture/contrast-dependent; too few can
  silently under-correct a low-texture region (see "Gotchas" below).
- `--block-step` (default 128) — local-refinement grid spacing, in
  CytAssist-scale pixels.
- `--scale-tol` (default 0.15) — triggers a phase-correlation fallback if
  the coarse fit's scale is more than this far from 1.0.
- `--extra-mirror-x` / `--manual-translation DX DY` — for samples where
  automatic orientation/coarse-fit can't be trusted (see below).

### 2. `report.py` — per-sample QC report

```bash
python report.py --sample SAMPLE_NAME --output-dir outputs --hires HiRes.tif
```

Builds one self-contained `<sample>_report.html`: run metrics and warnings,
a CytAssist-scale red(HiRes)/green(CytAssist)/yellow(aligned) overlay, a
color-coded (hue = direction) local-warp quiver plot, a before/after
(coarse-affine-only vs affine+local-warp) comparison over the whole region,
and a full-tissue-extent, native-resolution-detail overlay JPG (capped via
`--cap-long-side`, default 12000px long side — big enough to zoom into any
part of the tissue, small enough to stay a manageable file).

### 3. `build_full_res_deliverables.py` — full-resolution Space Ranger inputs

```bash
python build_full_res_deliverables.py \
  --hires HiRes.tif --transform-npz outputs/SAMPLE_NAME_transform.npz \
  --output-dir . --sample SAMPLE_NAME --serial-number SERIAL --area AREA
```

Streams the output to disk in row-bands (never holds the full ~20GB image
in memory); pass `--checkpoint` on a RAM/time-constrained machine to cache
progress in a resumable zarr store.

## Coordinate spaces

- **`H0`** — native resolution, original orientation (the raw HiRes file).
- **`Hd`** — `H0` downsampled to CytAssist's pixel size, still original
  orientation.
- **`Ho`** — `Hd` with the flip/rotate/mirror applied so it matches
  CytAssist's orientation. `register.py`'s block matrices are in this space.
- **`C`** — CytAssist pixel coordinates.

`register.py` saves the `H0`→`Hd` scale (via each shape) and the `Hd`→`Ho`
transform (`orientation_matrix`) directly, whatever flip/rotate/mirror
combination it used — nothing is hardcoded to one sample's orientation.

## When automatic registration fails

On tissue with many near-identical fragments (repetitive ring-shaped cross
sections, many similar-looking small explants, etc.) with limited
HiRes/CytAssist field-of-view overlap, both `match_test_flip_rotate`'s own
orientation search and the ORB+RANSAC coarse fit can lock onto a
plausible-looking but wrong answer — a self-consistent-but-wrong keypoint
match set can outnumber correct matches and win RANSAC.

Symptoms: a "coarse scale far from 1.0" warning, then a phase-correlation
fallback whose top candidate peaks are near-tied, then very few blocks
(e.g. 30/600) ending up with real raw local signal, and — most tellingly —
the QC overlay showing almost no real tissue overlap (the fallback often
locks onto the repetitive fiducial-frame border pattern instead of tissue).

**Fix:** detect tissue blobs in both images (HSV saturation threshold, e.g.
Otsu) and vote for a consistent translation across every blob-centroid pair,
with and without an extra horizontal mirror. A correct
orientation+translation gets simultaneous support from dozens of blob
pairs with a tight residual spread (tens of px); a wrong one doesn't. Feed
the winning translation in via `--extra-mirror-x --manual-translation DX DY`
and confirm: valid-local-signal block count should jump close to 100%, and
tissue-mask IoU between the warped HiRes and CytAssist should land
comfortably above 0.5.

## Other gotchas

- A single raw RGB channel doesn't have enough contrast for ORB matching on
  brightfield images — use RGB luminance.
- `chunks="auto"` on the dask arrays passed to `palom.align.Aligner` can
  collapse to a single chunk, silently disabling block-wise refinement.
- Device/RAM constraints: on a memory-limited machine (e.g. ~4GB), keep
  overlay/compositing math in `float32` and process in row *and* column
  bands — `cv2.remap`/`cv2.warpAffine` also require both dimensions under
  32767px (`SHRT_MAX`).
- Two genuinely well-aligned images can still render red- or
  green-dominated rather than yellow in the QC overlay if one scan simply
  has lower stain contrast than the other — that's a per-image percentile
  normalization difference, not necessarily misalignment. Check tissue-mask
  IoU or valid-block-count if the overlay's color balance looks off.

## Space Ranger integration

`--image` takes `<sample>_local_warp_only.ome.tif` in place of the original
HiRes TIFF; `--loupe-alignment` takes `<sample>_global_affine.json` in place
of Space Ranger's automatic registration. Everything else (`--cytaimage`,
`--fastqs`, `--transcriptome`/`--probe-set`, `--slide`, `--area`) stays the
same, under a fresh `--id`.

10x Genomics' own docs are explicit that there's no partial/incremental
mode for this: *"If the pipeline is run using the alignment file containing
only the image registration information and the results are not
satisfactory, both manual workflows must be repeated."* `spaceranger count`
redoes fiducial/tissue detection, spot-to-image assignment and the count
matrix in one pass, since spot calling depends on the image + alignment.

`_global_affine.json` matches the schema of the CytAssist-to-microscope
*image-registration* portion specifically (`cytAssistInfo.transformImages`),
not a full fiducial/tissue-detection Loupe alignment — if Space Ranger's own
fiducial detection on the CytAssist image was already fine, this should
slot in as-is.
