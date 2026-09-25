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
one piece our pipeline actually corrects: `cytAssistInfo.transformImages`
(the CytAssist<->HiRes image registration). Everything else about spot/
fiducial calibration on the CytAssist image itself is untouched, since our
pipeline never touches that.

`cytAssistInfo.checksumHiRes` looks like an MD5 of the HiRes image file the
alignment was computed against (32 hex chars); since we're now pairing this
alignment with a DIFFERENT image (`<sample>_local_warp_only.ome.tif`, not
the original raw HiRes TIFF), this script recomputes it against that file
so it's consistent, in case Space Ranger (or Loupe) validates it -- cheap
insurance, since we don't know for certain whether it's enforced.

Usage:
    python merge_loupe_alignment.py \\
      --base-json H1-RCH86GZ-D1-fiducials-image-registration.json \\
      --our-json NMR2_DRG_global_affine.json \\
      --image NMR2_DRG_local_warp_only.ome.tif \\
      --output NMR2_DRG_merged_alignment.json
"""
import argparse
import hashlib
import json
from pathlib import Path


def md5_of_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main(base_json, our_json, image_path, output_path):
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

    merged = dict(base)  # shallow copy is fine -- we only replace top-level cytAssistInfo
    print(f"computing MD5 of {image_path} for cytAssistInfo.checksumHiRes ...")
    checksum = md5_of_file(image_path)
    merged["cytAssistInfo"] = {
        "checksumHiRes": checksum,
        "transformImages": ours_m,
    }
    Path(output_path).write_text(json.dumps(merged))
    print(f"wrote {output_path} (checksumHiRes={checksum})")
    print("all other keys (oligo, spot_metadata, metadata, slide_layout_file, spot_count, "
          "transform, serialNumber, area, checksum, removeImagePages) copied unchanged from "
          f"{base_json}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-json", required=True,
                    help="Space Ranger / Loupe's own complete alignment export for this slide+area")
    p.add_argument("--our-json", required=True, help="our cytAssistInfo-only *_global_affine.json")
    p.add_argument("--image", required=True, help="the corrected *_local_warp_only.ome.tif")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    main(args.base_json, args.our_json, args.image, args.output)
