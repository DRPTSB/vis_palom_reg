"""
Merge our corrected CytAssist<->HiRes image-registration affine into a
COMPLETE Loupe/Space-Ranger alignment JSON, instead of feeding Space Ranger
a bare `cytAssistInfo`-only file on its own.

Why this exists: `spaceranger count`'s LOUPE_ALIGNMENT_READER stage always
reads the file as a *combined* manual-alignment export -- it unconditionally
touches keys like `oligo`, `spot_metadata`, `transform` (the separate
spot/fiducial-grid transform), etc., even when you only ever wanted to
correct the CytAssist-to-microscope image registration
(`cytAssistInfo.transformImages`). A file containing only `serialNumber`,
`area`, `cytAssistInfo` -- which is what `build_full_res_deliverables.py`
writes on its own -- crashes that stage with a bare
`KeyError: 'oligo'` (seen in production on spaceranger-4.1.0):

    File ".../loupe_alignment_reader/__init__.py", line 111, in parse_loupe_json
      if loupe_data.contain_spots_info():
    File ".../cellranger/spatial/loupe_util.py", line 421, in contain_spots_info
      return len(self._data_dict["oligo"]) > 0
             ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
    KeyError: 'oligo'

The fix is to take a REAL, complete alignment JSON for the same
slide+area -- the one Space Ranger's own automatic fiducial/spot detection
produced for this slide (or a Loupe-exported one), which already has
correct, complete `oligo` / `spot_metadata` / `metadata` / `slide_layout_file`
/ `spot_count` / `transform` / `checksum` sections -- and only overwrite the
piece(s) our pipeline actually corrects:

  - `cytAssistInfo.transformImages` (the CytAssist<->HiRes image
    registration) -- always replaced.
  - `oligo` (the tissue-detection bitmask) -- replaced only when
    `--tissue-mask` is given, using our own tissue calls (e.g. from
    `tissue_detection_pipeline.py`) instead of the base file's own tissue
    detection. See `oligo_tissue.py` for how this is encoded, and the
    project doc "tissue_oligo_encoding" for why the transform convention
    it uses is trustworthy. `transform`, `spot_metadata`, `metadata`,
    `slide_layout_file`, `spot_count` are NEVER touched -- they describe
    the base file's own (trusted) fiducial calibration, which this
    pipeline never re-derives.

`cytAssistInfo.checksumHiRes` looks like an MD5 of the HiRes image file the
alignment was computed against (32 hex chars); since we're now pairing this
alignment with a DIFFERENT image (`<sample>_local_warp_only.ome.tif`, not
the original raw HiRes TIFF), this script recomputes it against that file
so it's consistent, in case Space Ranger (or Loupe) validates it -- cheap
insurance, since we don't know for certain whether it's enforced.

The output filename follows 10x's own naming convention for these files
(`<serialNumber>-<area>-fiducials-image-registration.json`, matching what
Space Ranger's automatic detection / Loupe's export already name it) --
pass --output to override, but the default is deliberately the same name a
real 10x-produced alignment file for this slide+area would have.

Usage:
    python merge_loupe_alignment.py \\
      --base-json H1-RCH86GZ-D1-fiducials-image-registration.json \\
      --our-json NMR2_DRG_global_affine.json \\
      --image NMR2_DRG_local_warp_only.ome.tif \\
      --output-dir .
    # writes ./H1-RCH86GZ-D1-fiducials-image-registration.json (serial/area
    # taken from the base file), overwriting nothing outside --output-dir

    # also replace tissue (oligo) with our own tissue mask:
    python merge_loupe_alignment.py \\
      --base-json H1-RCH86GZ-D1-fiducials-image-registration.json \\
      --our-json NMR2_DRG_global_affine.json \\
      --image NMR2_DRG_local_warp_only.ome.tif \\
      --tissue-mask NMR2_DRG_tissue_mask.png \\
      --output-dir .
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2

import oligo_tissue
from hires_cytassist_common import setup_pipeline_log


def md5_of_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def default_output_name(serial_number, area):
    """10x's own naming convention for these files, e.g.
    'H1-RCH86GZ-D1-fiducials-image-registration.json' -- matches what Space
    Ranger's auto-detection or a Loupe export would already call it."""
    return f"{serial_number}-{area}-fiducials-image-registration.json"


def main(base_json, our_json, image_path, output_path=None, output_dir=".",
         tissue_mask_path=None, tissue_aggregation="majority", tissue_supersample=4):
    base = json.loads(Path(base_json).read_text())
    ours = json.loads(Path(our_json).read_text())

    if base.get("serialNumber") != ours.get("serialNumber") or base.get("area") != ours.get("area"):
        raise SystemExit(
            f"serialNumber/area mismatch: base={base.get('serialNumber')}/{base.get('area')} "
            f"vs ours={ours.get('serialNumber')}/{ours.get('area')} -- wrong base file for this sample?"
        )

    # Sanity check: our independently-computed affine should be in the same
    # ballpark as the base file's own (both describe the same physical
    # CytAssist<->HiRes registration) -- large disagreement means either the
    # wrong base file was picked, or one of the two registrations is wrong.
    base_m = base["cytAssistInfo"]["transformImages"]
    ours_m = ours["cytAssistInfo"]["transformImages"]
    base_scale = (base_m[0][0] ** 2 + base_m[0][1] ** 2) ** 0.5
    ours_scale = (ours_m[0][0] ** 2 + ours_m[0][1] ** 2) ** 0.5
    scale_pct_diff = abs(base_scale - ours_scale) / base_scale * 100
    dx = abs(base_m[0][2] - ours_m[0][2])
    dy = abs(base_m[1][2] - ours_m[1][2])
    print(f"cross-check vs base file's own registration: scale differs by {scale_pct_diff:.2f}%, "
          f"translation differs by ({dx:.1f}, {dy:.1f})px")
    if scale_pct_diff > 2 or dx > 200 or dy > 200:
        print("WARNING: this is a much larger disagreement than seen on other samples "
              "(<1% scale, <~90px translation) -- double check this is the right base file "
              "before trusting the merged output.")

    merged = dict(base)  # shallow copy is fine -- we only replace top-level keys
    print(f"computing MD5 of {image_path} for cytAssistInfo.checksumHiRes ...")
    checksum = md5_of_file(image_path)
    merged["cytAssistInfo"] = {
        "checksumHiRes": checksum,
        "transformImages": ours_m,
    }

    if tissue_mask_path is not None:
        print(f"encoding tissue mask {tissue_mask_path} into oligo "
              f"(aggregation={tissue_aggregation}, supersample={tissue_supersample}) ...")
        mask_img = cv2.imread(str(tissue_mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_img is None:
            raise SystemExit(f"could not read tissue mask image {tissue_mask_path}")
        mask = mask_img > 0
        # validates base's bin_level/transform/spot_count itself -- raises
        # rather than silently proceeding if this base file doesn't match
        # the verified convention (see oligo_tissue.py).
        new_oligo, grid = oligo_tissue.build_oligo_from_tissue_mask(
            base_json, mask, aggregation=tissue_aggregation, supersample=tissue_supersample)
        base_tissue_frac = oligo_tissue.decode_oligo(base)[0].mean()
        print(f"base file's own oligo tissue fraction: {base_tissue_frac:.4f}; "
              f"our tissue mask's encoded fraction: {grid.mean():.4f}")
        merged["oligo"] = new_oligo
    else:
        print("no --tissue-mask given: oligo copied unchanged from the base file "
              "(base file's own tissue detection is kept as-is)")

    if output_path is None:
        output_path = Path(output_dir) / default_output_name(base["serialNumber"], base["area"])
    output_path = Path(output_path)
    if output_path.resolve() == Path(base_json).resolve():
        raise SystemExit(
            f"refusing to overwrite the base file itself ({base_json}) -- "
            "pass a different --output-dir or --output"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(merged))
    print(f"wrote {output_path} (checksumHiRes={checksum})")
    unchanged_keys = "spot_metadata, metadata, slide_layout_file, spot_count, transform, " \
                     "serialNumber, area, checksum, removeImagePages"
    if tissue_mask_path is None:
        unchanged_keys = "oligo, " + unchanged_keys
    print(f"unchanged from {base_json}: {unchanged_keys}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-json", required=True,
                    help="Space Ranger / Loupe's own complete alignment export for this slide+area")
    p.add_argument("--our-json", required=True, help="our cytAssistInfo-only *_global_affine.json")
    p.add_argument("--image", required=True, help="the corrected *_local_warp_only.ome.tif")
    p.add_argument("--tissue-mask", default=None,
                    help="optional: a 0/255 single-channel tissue mask PNG in CytAssist pixel "
                         "space (e.g. tissue_detection_pipeline.py's *_tissue_mask.png) to encode "
                         "into oligo, replacing the base file's own tissue detection. If omitted, "
                         "oligo is copied unchanged from --base-json.")
    p.add_argument("--tissue-aggregation", choices=["any", "majority"], default="majority",
                    help="only used with --tissue-mask; see oligo_tissue.py for details")
    p.add_argument("--tissue-supersample", type=int, default=4,
                    help="only used with --tissue-mask; see oligo_tissue.py for details")
    p.add_argument("--output", default=None,
                    help="explicit output path; default is 10x's own naming convention "
                         "(<serialNumber>-<area>-fiducials-image-registration.json) under --output-dir")
    p.add_argument("--output-dir", default=".", help="used only when --output is not given")
    p.add_argument("--sample", default=None,
                    help="sample name, for logging only (matches the other stages' own "
                         "--sample so they share one <sample>_pipeline.log). Default: derived "
                         "from the base JSON's serialNumber-area, for standalone use.")
    p.add_argument("--log-dir", default=None,
                    help="where <sample>_pipeline.log lives; default: --output-dir. Set "
                         "explicitly since this stage's own --output-dir is normally a "
                         "spaceranger_alignment/ subfolder, not the shared per-sample "
                         "directory (run_pipeline.py always sets this).")
    args = p.parse_args()
    _log_sample = args.sample
    if _log_sample is None:
        _base_for_log = json.loads(Path(args.base_json).read_text())
        _log_sample = f"{_base_for_log.get('serialNumber', 'unknown')}-{_base_for_log.get('area', 'unknown')}"
    setup_pipeline_log(args.log_dir or args.output_dir, _log_sample, "merge_loupe_alignment.py")
    main(args.base_json, args.our_json, args.image, args.output, args.output_dir,
         tissue_mask_path=args.tissue_mask, tissue_aggregation=args.tissue_aggregation,
         tissue_supersample=args.tissue_supersample)
