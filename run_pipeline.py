"""
End-to-end orchestrator: HiRes H&E + CytAssist image in, warped image + a
10x-named alignment JSON (with both tissue positions and the image
registration matrices) out -- bypassing 10x's manual Loupe alignment steps
entirely.

This wires together the individually-verified pieces of this pipeline in
one fixed order and stops at the first failure -- it does NOT retry with
different flags, does NOT silently fall back to a worse method, and does
NOT guess a missing input. Each step below is exactly one call to an
existing, already-tested script in this repo; this file adds no new
registration/tissue-detection/encoding logic of its own.

Requires a `--base-json`: a real, complete Space Ranger / Loupe alignment
export for the EXACT slide+area (i.e. produced by Space Ranger's own
automatic fiducial detection, or a Loupe export) -- NOT one this pipeline
invents. Its `transform` / `spot_metadata` / fiducial-derived fields are
trusted as-is and never touched; this pipeline only ever replaces
`cytAssistInfo` (image registration) and `oligo` (tissue calls). See the
project's "tissue_oligo_encoding" doc for why that split is safe, and
merge_loupe_alignment.py / oligo_tissue.py for the mechanism.

Steps:
  1. Validate --base-json against --serial-number/--area up front (fail
     before doing any expensive work on a mismatched base file).
  2. register.py       -- HiRes<->CytAssist coarse + block-wise registration
  3. tissue_detection_pipeline.py -- tissue mask on the CytAssist image
  4. build_full_res_deliverables.py -- full-res warped image + affine JSON
  5. merge_loupe_alignment.py -- merge affine + tissue into the base JSON
  6. report.py (optional, default on) -- QC HTML report

Usage:
    python run_pipeline.py \\
      --hires HiRes.tif --cytassist CytAssist.tif \\
      --sample NMR2_Colon --serial-number H1-RCH86GZ --area A1 \\
      --base-json H1-RCH86GZ-A1-fiducials-image-registration.json \\
      --output-dir outputs
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import oligo_tissue
from hires_cytassist_common import setup_pipeline_log

HERE = Path(__file__).resolve().parent


def run_step(name, cmd):
    print(f"\n=== {name} ===")
    print(" ".join(str(c) for c in cmd))
    result = subprocess.run(cmd, cwd=HERE)
    if result.returncode != 0:
        raise SystemExit(f"step '{name}' failed (exit code {result.returncode}) -- stopping. "
                          "No fallback is attempted; fix the underlying issue and re-run.")


def require_file(path, what):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"expected {what} at {path} but it doesn't exist -- a prior step "
                          "did not produce what this orchestrator assumed it would.")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hires", required=True)
    p.add_argument("--cytassist", required=True)
    p.add_argument("--sample", required=True)
    p.add_argument("--serial-number", required=True)
    p.add_argument("--area", required=True)
    p.add_argument("--base-json", required=True,
                    help="a real Space Ranger / Loupe alignment export for this exact slide+area "
                         "-- its transform/spot_metadata/fiducial fields are trusted and never "
                         "modified; only cytAssistInfo and oligo are replaced.")
    p.add_argument("--output-dir", required=True)

    # register.py passthrough
    p.add_argument("--block-step", type=int, default=128)
    p.add_argument("--n-keypoints", type=int, default=12000)
    p.add_argument("--extra-mirror-x", action="store_true")
    p.add_argument("--manual-translation", type=float, nargs=2, default=None, metavar=("DX", "DY"))
    p.add_argument("--rotation-search-deg", type=float, default=10.0)
    p.add_argument("--rotation-search-step", type=float, default=0.5)

    # tissue_detection_pipeline.py passthrough
    p.add_argument("--tissue-model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"])
    p.add_argument("--tissue-device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--tissue-saturation-threshold", type=int, default=30)
    p.add_argument("--tissue-min-blob-area", type=int, default=500)
    p.add_argument("--tissue-min-mask-area", type=int, default=100)
    p.add_argument("--force-morphological-tissue", action="store_true",
                    help="skip SAM even if installed, use HSV+morphology only")

    # oligo encoding
    p.add_argument("--tissue-aggregation", choices=["any", "majority"], default="majority")
    p.add_argument("--tissue-supersample", type=int, default=4)

    # build_full_res_deliverables.py passthrough
    p.add_argument("--checkpoint", action="store_true")

    p.add_argument("--skip-report", action="store_true")

    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tissue_dir = output_dir / "tissue_detection"
    alignment_dir = output_dir / "spaceranger_alignment"

    # Every stage below is passed --log-dir pointing at this same output_dir,
    # so all 6 scripts (this one included) append to one shared
    # <sample>_pipeline.log, regardless of which one's own --output-dir is a
    # subfolder (tissue_dir, alignment_dir). See hires_cytassist_common.py.
    setup_pipeline_log(output_dir, args.sample, "run_pipeline.py")

    hires = require_file(args.hires, "HiRes image")
    cytassist = require_file(args.cytassist, "CytAssist image")
    base_json_path = require_file(args.base_json, "base alignment JSON")

    # --- Step 1: validate the base JSON up front, before any expensive work ---
    base = json.loads(base_json_path.read_text())
    if base.get("serialNumber") != args.serial_number or base.get("area") != args.area:
        raise SystemExit(
            f"--base-json's serialNumber/area ({base.get('serialNumber')}/{base.get('area')}) "
            f"does not match --serial-number/--area ({args.serial_number}/{args.area}) -- "
            "wrong base file for this sample. Refusing to proceed."
        )
    # raises if bin_level/transform/spot_count don't match this encoder's verified assumptions
    oligo_tissue.validate_base_json(base)
    print(f"base JSON validated: serialNumber={base['serialNumber']} area={base['area']} "
          f"bin_level={base['spot_metadata']['bin_level']}")

    # --- Step 2: registration ---
    register_cmd = [
        sys.executable, "register.py",
        "--hires", str(hires), "--cytassist", str(cytassist),
        "--output-dir", str(output_dir), "--sample", args.sample,
        "--block-step", str(args.block_step), "--n-keypoints", str(args.n_keypoints),
        "--rotation-search-deg", str(args.rotation_search_deg),
        "--rotation-search-step", str(args.rotation_search_step),
    ]
    register_cmd += ["--log-dir", str(output_dir)]
    if args.extra_mirror_x:
        register_cmd.append("--extra-mirror-x")
    if args.manual_translation is not None:
        register_cmd += ["--manual-translation", str(args.manual_translation[0]), str(args.manual_translation[1])]
    run_step("register.py", register_cmd)
    transform_npz = require_file(output_dir / f"{args.sample}_transform.npz", "register.py's transform npz")

    # --- Step 3: tissue detection (independent of registration; CytAssist-pixel-space mask) ---
    tissue_cmd = [
        sys.executable, "tissue_detection_pipeline.py", str(cytassist),
        "--sample", args.sample, "--output-dir", str(tissue_dir),
        "--model-type", args.tissue_model_type, "--device", args.tissue_device,
        "--saturation-threshold", str(args.tissue_saturation_threshold),
        "--min-blob-area", str(args.tissue_min_blob_area),
        "--min-mask-area", str(args.tissue_min_mask_area),
        "--log-dir", str(output_dir),
    ]
    if args.force_morphological_tissue:
        tissue_cmd.append("--force-morphological")
    run_step("tissue_detection_pipeline.py", tissue_cmd)
    tissue_mask = require_file(tissue_dir / f"{args.sample}_tissue_mask.png", "tissue mask PNG")

    # --- Step 4: full-resolution deliverables (warped image + affine JSON) ---
    build_cmd = [
        sys.executable, "build_full_res_deliverables.py",
        "--hires", str(hires), "--transform-npz", str(transform_npz),
        "--output-dir", str(output_dir), "--sample", args.sample,
        "--serial-number", args.serial_number, "--area", args.area,
        "--log-dir", str(output_dir),
    ]
    if args.checkpoint:
        build_cmd.append("--checkpoint")
    run_step("build_full_res_deliverables.py", build_cmd)
    warped_image = require_file(output_dir / f"{args.sample}_local_warp_only.ome.tif", "warped OME-TIFF")
    global_affine_json = require_file(output_dir / f"{args.sample}_global_affine.json", "global affine JSON")

    # --- Step 5: merge into the final 10x-named alignment JSON ---
    merge_cmd = [
        sys.executable, "merge_loupe_alignment.py",
        "--base-json", str(base_json_path), "--our-json", str(global_affine_json),
        "--image", str(warped_image), "--tissue-mask", str(tissue_mask),
        "--tissue-aggregation", args.tissue_aggregation,
        "--tissue-supersample", str(args.tissue_supersample),
        "--output-dir", str(alignment_dir),
        "--sample", args.sample, "--log-dir", str(output_dir),
    ]
    run_step("merge_loupe_alignment.py", merge_cmd)
    final_json = require_file(
        alignment_dir / f"{args.serial_number}-{args.area}-fiducials-image-registration.json",
        "final merged alignment JSON")

    # --- Step 6: QC report (optional) ---
    qc_report = None
    if not args.skip_report:
        report_cmd = [sys.executable, "report.py", "--sample", args.sample,
                       "--output-dir", str(output_dir), "--hires", str(hires),
                       "--log-dir", str(output_dir)]
        run_step("report.py", report_cmd)
        qc_report = require_file(output_dir / f"{args.sample}_report.html", "QC HTML report")

    print("\n=== DONE ===")
    print(f"warped HiRes image (feed to Space Ranger alongside the JSON below):\n  {warped_image}")
    print(f"final alignment JSON (--loupe-alignment):\n  {final_json}")
    print(f"tissue mask / overlay:\n  {tissue_mask}\n  {tissue_dir / (args.sample + '_tissue_overlay.png')}")
    if qc_report:
        print(f"QC report:\n  {qc_report}")


if __name__ == "__main__":
    main()
