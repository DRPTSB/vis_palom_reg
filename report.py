"""
Generate a single self-contained HTML QC report for one register.py run:
metrics table, the CytAssist-scale red/green "yellow" overlay, a
color-coded local-warp direction/magnitude quiver plot, a before/after
(coarse-affine-only vs affine+local-warp) comparison, and a link to a
much higher-detail whole-tissue-region overlay JPEG (built from real
native-resolution HiRes pixels, capped to a manageable file size so it's
still easy to open and zoom into).

Usage:
    python3 report.py --sample NMR3_Colon --output-dir outputs \\
        --hires "$HOME/mnt/Downloads/NMR_3_Colon.tif"

Expects `<sample>_transform.npz`, `<sample>_run_stats.json` and
`<sample>_preview_warped_to_cytassist.ome.tif` (all written by register.py)
to already exist in --output-dir.
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_full_res_deliverables import fit_local_correction, global_affine, get_block_matrix
from hires_cytassist_common import read_series0_pyramid


# ---------------------------------------------------------------- overlay --
#
# All of this works in float32, and (for the whole-region overlay) band by
# band, on purpose: at the ~10000px-square sizes the whole-region overlay
# needs, a naive single-shot float64 version materializes several
# multi-gigabyte arrays at once (red + green + RGB output, each width x
# height x 8 bytes) and reliably OOMs a memory-constrained machine. Bounding
# the per-call array size is what makes this safe regardless of how large
# the final image is.

def signal_percentiles(gray, no_data=None, lo=1, hi=99.5):
    """The (p_lo, p_hi) brightfield->signal normalization range, estimated
    once from a whole (small or subsampled) image so every band/tile of a
    larger image is normalized consistently -- computing percentiles
    per-tile instead would make brightness/contrast jump at tile edges."""
    sig = (255.0 - gray).astype(np.float32)
    valid = sig if no_data is None else sig[~no_data]
    p_lo, p_hi = np.percentile(valid, [lo, hi])
    return float(p_lo), float(p_hi)


def norm_signal(gray, p_lo, p_hi, no_data=None):
    """Brightfield -> normalized signal (0=background, 1=strongest tissue),
    using pre-computed global percentiles (see signal_percentiles). `no_data`
    marks pixels with no source image at all (e.g. outside the warped
    region), forced to 0 rather than treated as signal -- palom's block warp
    fills those with black, which would otherwise invert to "maximum
    signal" and show up as a false corruption-looking patch.
    """
    sig = (255.0 - gray).astype(np.float32)
    out = np.clip((sig - p_lo) / max(p_hi - p_lo, 1e-6), 0, 1).astype(np.float32)
    if no_data is not None:
        out[no_data] = 0
    return out


def yellow_overlay_uint8(hires_rgb, cyt_rgb, hires_percentiles, cyt_percentiles):
    """Red = HiRes, green = CytAssist, yellow = the two agree. Returns uint8
    directly (no separate float clip/cast pass) using percentiles computed
    up front so tiles/bands stay visually consistent with each other."""
    hires_no_data = hires_rgb.sum(-1) == 0
    cyt_no_data = cyt_rgb.sum(-1) == 0
    red = norm_signal(hires_rgb.mean(-1), *hires_percentiles, no_data=hires_no_data)
    green = norm_signal(cyt_rgb.mean(-1), *cyt_percentiles, no_data=cyt_no_data)
    out = np.empty((*red.shape, 3), dtype=np.float32)
    out[..., 0], out[..., 1] = red, green
    out[..., 2] = np.minimum(red, green) * 0.4
    return (out * 255).astype(np.uint8)


def save_rgb(img_uint8, path, jpeg_quality=None):
    bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if jpeg_quality else []
    cv2.imwrite(str(path), bgr, params)


# ------------------------------------------------------- CytAssist overlay --

def build_qc_overlay(sample, output_dir, cyt_rgb, out_path):
    """The original, small (CytAssist-resolution) sanity-check overlay:
    fast to build, good for judging overall coverage/shape at a glance.
    Built from register.py's own preview, which already has the per-block
    local-warp correction baked in (not affine-only) -- see the
    before/after comparison below for an explicit affine-only vs
    affine+local-warp view."""
    preview_path = output_dir / f"{sample}_preview_warped_to_cytassist.ome.tif"
    warped = tifffile.imread(preview_path)
    if warped.ndim == 3 and warped.shape[0] in (3, 4):
        warped = np.moveaxis(warped, 0, -1)
    hires_p = signal_percentiles(warped.mean(-1), no_data=warped.sum(-1) == 0)
    cyt_p = signal_percentiles(cyt_rgb.mean(-1), no_data=cyt_rgb.sum(-1) == 0)
    save_rgb(yellow_overlay_uint8(warped, cyt_rgb, hires_p, cyt_p), out_path)
    return out_path


# ------------------------------------------------------- direction quiver --

def build_warp_quiver(sample, transform, cyt_rgb, out_path):
    """Local (block-wise) correction as a vector field: one arrow per
    block, direction color-coded (hue = angle, via a standard HSV color
    wheel), length = magnitude. Blocks the tissue-free-region safety net
    overrode (finding 8a) have zero correction by construction and show as
    a dot, not an arrow -- a real signal that no local signal existed there,
    not a bug.
    """
    matrices = transform["block_affine_matrices"]
    m = transform["coarse_affine_matrix"]
    gr, gc = transform["grid_shape"]
    block_step = int(transform["block_step"])

    cx = (np.arange(gc) + 0.5) * block_step
    cy = (np.arange(gr) + 0.5) * block_step
    grid_x, grid_y = np.meshgrid(cx, cy)
    dx = np.array([[get_block_matrix(matrices, i, j)[0, 2] - m[0, 2] for j in range(gc)] for i in range(gr)])
    dy = np.array([[get_block_matrix(matrices, i, j)[1, 2] - m[1, 2] for j in range(gc)] for i in range(gr)])
    angle_deg = (np.degrees(np.arctan2(dy, dx)) + 360) % 360
    magnitude = np.hypot(dx, dy)

    fig, (ax, wheel_ax) = plt.subplots(
        1, 2, figsize=(11, 9), dpi=140, gridspec_kw={"width_ratios": [8, 1]})
    ax.imshow(cyt_rgb.mean(-1), cmap="gray", vmin=0, vmax=255)
    colors = plt.cm.hsv(angle_deg.ravel() / 360)
    ax.quiver(grid_x, grid_y, dx, dy, color=colors, angles="xy", scale_units="xy",
              scale=1 / 8, width=0.0035)
    ax.set_title(f"{sample}: local-warp direction (color) and magnitude (arrow length)\n"
                 f"max |correction| = {magnitude.max():.1f}px at CytAssist scale")
    ax.set_xlim(0, cyt_rgb.shape[1]); ax.set_ylim(cyt_rgb.shape[0], 0)
    ax.axis("off")

    # Color-wheel legend for direction.
    theta = np.linspace(0, 2 * np.pi, 256)
    radius = np.linspace(0, 1, 32)
    T, R = np.meshgrid(theta, radius)
    wheel_ax.remove()
    wheel_ax = fig.add_subplot(1, 2, 2, projection="polar")
    wheel_ax.pcolormesh(T, R, T, cmap="hsv", shading="auto")
    wheel_ax.set_yticklabels([]); wheel_ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    wheel_ax.set_xticklabels(["→", "↑", "←", "↓"], fontsize=13)
    wheel_ax.set_title("direction", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    return out_path, {"max_correction_px": float(magnitude.max()),
                       "median_correction_px": float(np.median(magnitude))}


# ------------------------------------------------ whole-region native-res --

def local_delta_at(h0x, h0y, index_to_h0, index_to_h0_offset, ddx_grid, ddy_grid, grid_shape):
    """Local (block-wise) correction, in H0 pixel units, at arbitrary
    (possibly non-integer) H0 positions -- the same lookup
    build_full_res_deliverables.make_corrector does, just for an arbitrary
    point array instead of one raw H0 pixel band."""
    from scipy.ndimage import map_coordinates
    gr, gc = grid_shape
    h0_to_index = np.linalg.inv(index_to_h0)
    pts = np.stack([h0x.ravel(), h0y.ravel()], axis=1) - index_to_h0_offset
    ji = pts @ h0_to_index.T
    jf = np.clip(ji[:, 0], 0, gc - 1)
    iif = np.clip(ji[:, 1], 0, gr - 1)
    dx = map_coordinates(ddx_grid, [iif, jf], order=1, mode="nearest").reshape(h0x.shape)
    dy = map_coordinates(ddy_grid, [iif, jf], order=1, mode="nearest").reshape(h0x.shape)
    return dx.astype(np.float32), dy.astype(np.float32)


def _render_hires_canvas(transform, h0, h0_shape, cyt_shape, cap_long_side,
                          apply_correction, band_rows=300):
    """Remap native H0 pixels onto a CytAssist-aligned canvas capped at
    `cap_long_side`, in row/column bands so memory stays bounded regardless
    of image size. `apply_correction=True` adds the fitted per-block local
    warp on top of the coarse global affine (the real, final result);
    `apply_correction=False` uses the coarse affine alone, for an explicit
    before/after comparison of what the local warp actually changes.

    Shared by build_whole_region_overlay (the full-detail, apply_correction
    =True-only deliverable) and build_before_after_comparison (both modes,
    at a smaller size).
    """
    h0_h, h0_w = h0_shape
    ref_h, ref_w = cyt_shape
    os_factor = max(1, round(cap_long_side / max(ref_h, ref_w)))
    out_h, out_w = ref_h * os_factor, ref_w * os_factor

    index_to_h0, index_to_h0_offset, ddx_grid, ddy_grid = fit_local_correction(transform, h0_shape)
    total_affine = global_affine(transform, h0_shape)
    total_inv = np.linalg.inv(total_affine)
    grid_shape = transform["grid_shape"]

    # Tile in both row and column bands: cv2.remap requires every dimension
    # (source AND destination) to stay under 32767px, and because of the
    # ~18x HiRes->CytAssist downsample factor, even a modest-height row band
    # can need a native-resolution source crop spanning tens of thousands of
    # columns if it isn't also split column-wise.
    band_cols = band_rows * 4
    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    n_tiles = ((out_h + band_rows - 1) // band_rows) * ((out_w + band_cols - 1) // band_cols)
    tile_i = 0
    for r0 in range(0, out_h, band_rows):
        r1 = min(r0 + band_rows, out_h)
        for c0 in range(0, out_w, band_cols):
            c1 = min(c0 + band_cols, out_w)
            tile_i += 1
            band_y, band_x = np.mgrid[r0:r1, c0:c1].astype(np.float32)
            cx, cy = band_x / os_factor, band_y / os_factor
            v = np.stack([cx.ravel(), cy.ravel(), np.ones(cx.size)])
            h0_naive = total_inv @ v
            h0x = (h0_naive[0] / h0_naive[2]).reshape(cx.shape)
            h0y = (h0_naive[1] / h0_naive[2]).reshape(cx.shape)
            if apply_correction:
                dx, dy = local_delta_at(h0x, h0y, index_to_h0, index_to_h0_offset, ddx_grid, ddy_grid, grid_shape)
                h0x, h0y = h0x + dx, h0y + dy

            bx0, bx1 = max(0, int(np.floor(h0x.min())) - 4), min(h0_w, int(np.ceil(h0x.max())) + 4)
            by0, by1 = max(0, int(np.floor(h0y.min())) - 4), min(h0_h, int(np.ceil(h0y.max())) + 4)
            if bx1 <= bx0 or by1 <= by0 or (bx1 - bx0) >= 32767 or (by1 - by0) >= 32767:
                continue  # outside the HiRes image, or (rare) still too wide -- leave black
            src = h0[by0:by1, bx0:bx1, :].compute()
            canvas[r0:r1, c0:c1, :] = cv2.remap(
                src, (h0x - bx0).astype(np.float32), (h0y - by0).astype(np.float32),
                interpolation=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
            del src
            if tile_i % 20 == 0:
                print(f"  tile {tile_i}/{n_tiles} (apply_correction={apply_correction})", flush=True)
    return canvas, out_h, out_w, os_factor


def build_whole_region_overlay(sample, transform, hires_path, cyt_rgb, out_path,
                                cap_long_side=12000, band_rows=300):
    """Whole-tissue-extent yellow overlay, built from real native-resolution
    HiRes pixels (not just CytAssist's own coarse grid), upsampled as far as
    `cap_long_side` allows. This is the final, fully-corrected (affine +
    local warp) result -- see build_before_after_comparison for a smaller
    side-by-side showing what the local warp changed.
    """
    h0 = read_series0_pyramid(hires_path)[0]
    h0_shape = h0.shape[:2]
    print(f"Whole-region overlay: cap_long_side={cap_long_side}", flush=True)
    hires_canvas, out_h, out_w, os_factor = _render_hires_canvas(
        transform, h0, h0_shape, cyt_rgb.shape[:2], cap_long_side,
        apply_correction=True, band_rows=band_rows)
    print(f"  canvas {out_w}x{out_h} ({os_factor}x CytAssist's own resolution)", flush=True)

    # Upsample CytAssist to the same canvas size once (uint8, cheap) -- but
    # do the actual overlay math (percentile normalization, RGB compose) in
    # row-bands rather than on the full array: at these pixel counts the
    # naive single-shot version needs several float arrays the size of the
    # whole canvas at once and can double or triple peak memory for no
    # visual benefit. Percentiles themselves are still computed globally
    # (from a strided subsample) so bands stay visually consistent.
    print("Compositing final overlay...", flush=True)
    cyt_up = cv2.resize(cyt_rgb, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    hires_p = signal_percentiles(hires_canvas[::4, ::4].mean(-1), no_data=hires_canvas[::4, ::4].sum(-1) == 0)
    cyt_p = signal_percentiles(cyt_up[::4, ::4].mean(-1), no_data=cyt_up[::4, ::4].sum(-1) == 0)

    final = np.empty((out_h, out_w, 3), dtype=np.uint8)
    for r0 in range(0, out_h, band_rows):
        r1 = min(r0 + band_rows, out_h)
        final[r0:r1] = yellow_overlay_uint8(hires_canvas[r0:r1], cyt_up[r0:r1], hires_p, cyt_p)
    del hires_canvas, cyt_up

    save_rgb(final, out_path, jpeg_quality=92)
    print(f"Wrote {out_path}", flush=True)
    return out_path, {"width": out_w, "height": out_h, "oversample_vs_cytassist": os_factor}


def build_before_after_comparison(sample, transform, hires_path, cyt_rgb, out_path,
                                   cap_long_side=3000):
    """Side-by-side: coarse-affine-only (BEFORE) vs affine+local-warp
    (AFTER), over the whole tissue region, at a modest resolution -- just to
    make the local correction's visual effect obvious at a glance. Use
    build_whole_region_overlay's full-resolution output to actually zoom
    into a specific spot.
    """
    h0 = read_series0_pyramid(hires_path)[0]
    h0_shape = h0.shape[:2]
    cyt_shape = cyt_rgb.shape[:2]

    print(f"Before/after comparison: cap_long_side={cap_long_side}", flush=True)
    before_canvas, out_h, out_w, _ = _render_hires_canvas(
        transform, h0, h0_shape, cyt_shape, cap_long_side, apply_correction=False)
    after_canvas, _, _, _ = _render_hires_canvas(
        transform, h0, h0_shape, cyt_shape, cap_long_side, apply_correction=True)

    cyt_up = cv2.resize(cyt_rgb, (out_w, out_h), interpolation=cv2.INTER_CUBIC)
    # Shared percentiles (from the AFTER canvas) so BEFORE and AFTER are
    # normalized identically -- any visual difference is the local warp,
    # not a brightness-normalization artifact.
    hires_p = signal_percentiles(after_canvas.mean(-1), no_data=after_canvas.sum(-1) == 0)
    cyt_p = signal_percentiles(cyt_up.mean(-1), no_data=cyt_up.sum(-1) == 0)

    before_ov = yellow_overlay_uint8(before_canvas, cyt_up, hires_p, cyt_p)
    after_ov = yellow_overlay_uint8(after_canvas, cyt_up, hires_p, cyt_p)

    gap, label_h = 12, 40
    panel = np.zeros((out_h + label_h, out_w * 2 + gap, 3), dtype=np.uint8)
    panel[label_h:, :out_w] = before_ov
    panel[label_h:, out_w + gap:] = after_ov
    cv2.putText(panel, "BEFORE: coarse affine only", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(panel, "AFTER: + per-block local warp", (out_w + gap + 10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    save_rgb(panel, out_path, jpeg_quality=90)
    print(f"Wrote {out_path}", flush=True)
    return out_path


# --------------------------------------------------------------- report --

def _b64_img(path):
    return base64.b64encode(Path(path).read_bytes()).decode()


def _b64_thumbnail(path, max_width=1400):
    """A small, embeddable preview of a (possibly very large) image, so the
    HTML report itself stays a normal-sized file. The full-resolution image
    stays on disk as its own file for actual zooming."""
    img = cv2.imread(str(path))
    scale = min(1.0, max_width / img.shape[1])
    if scale < 1.0:
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()


def write_html_report(sample, stats, qc_path, quiver_path, quiver_stats,
                       before_after_path, whole_region_path, whole_region_stats, out_path):
    warn_html = "".join(f"<li>{w}</li>" for w in stats["warnings"]) or "<li>none</li>"
    rows = {
        "Coarse scale (x, y)": f"{stats['coarse_scale'][0]:.4f}, {stats['coarse_scale'][1]:.4f}",
        "Coarse rotation": f"{stats['coarse_angle_deg']:.2f} deg",
        "Coarse translation": f"{stats['coarse_translation'][0]:.1f}, {stats['coarse_translation'][1]:.1f} px",
        "Used phase-correlation fallback": stats["used_phase_fallback"],
        "Extra mirror-x applied": stats["extra_mirror_x"],
        "Manual translation seed": stats["manual_translation"],
        "Block grid": f"{stats['grid_shape'][0]} x {stats['grid_shape'][1]} "
                       f"({stats['n_blocks_total']} blocks, step={stats['block_step']}px)",
        "Blocks with raw local signal": f"{stats['n_blocks_valid_raw']}/{stats['n_blocks_total']}",
        "Blocks overridden (tissue-free)": stats["n_blocks_overridden"],
        "Local-warp correction (median / max)":
            f"{quiver_stats['median_correction_px']:.1f} / {quiver_stats['max_correction_px']:.1f} px",
        "Preview coverage": f"{stats['coverage']:.4f}",
    }
    row_html = "".join(f"<tr><th>{k}</th><td>{v}</td></tr>" for k, v in rows.items())

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{sample} registration QC report</title>
<style>
body {{ font-family: -apple-system, Helvetica, Arial, sans-serif; max-width: 1100px; margin: 2em auto; color: #222; }}
h1 {{ font-size: 1.4em; }} h2 {{ font-size: 1.1em; margin-top: 2em; border-bottom: 1px solid #ccc; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ text-align: left; padding: 4px 12px 4px 0; border-bottom: 1px solid #eee; }}
th {{ font-weight: 600; color: #555; }}
img {{ max-width: 100%; border: 1px solid #ddd; }}
.warn {{ background: #fff6e5; border: 1px solid #f0c36d; padding: 0.6em 1em; border-radius: 6px; }}
</style></head>
<body>
<h1>{sample} -- registration QC report</h1>
<p>HiRes: <code>{stats['hires_path']}</code><br>CytAssist: <code>{stats['cytassist_path']}</code></p>

<h2>Run summary</h2>
<table>{row_html}</table>

<div class="warn"><b>Warnings raised during this run:</b><ul>{warn_html}</ul></div>

<h2>CytAssist-scale overlay (red=HiRes, green=CytAssist, yellow=aligned)</h2>
<p>Fast sanity-check preview at CytAssist's own resolution. Already includes the
per-block local-warp correction (not affine-only) -- see the before/after
comparison below to see that correction's effect explicitly.</p>
<img src="data:image/jpeg;base64,{_b64_img(qc_path)}">

<h2>Local-warp direction, color-coded</h2>
<img src="data:image/png;base64,{_b64_img(quiver_path)}">

<h2>Before vs after: what did the local warp actually change?</h2>
<p>Same whole-tissue region: coarse affine only (left) vs affine + per-block
local warp (right), same brightness normalization on both sides so any
visible difference is the correction itself, not a rendering artifact.</p>
<img src="data:image/jpeg;base64,{_b64_img(before_after_path)}">

<h2>Whole-region overlay at native-resolution detail</h2>
<p>{whole_region_stats['width']}x{whole_region_stats['height']}px
({whole_region_stats['oversample_vs_cytassist']}x CytAssist's own pixel grid). This preview is
downscaled to keep the report itself small -- open <code>{Path(whole_region_path).name}</code>
(next to this report) directly in an image viewer to zoom into full detail anywhere in the tissue.</p>
<a href="{Path(whole_region_path).name}">
<img src="data:image/jpeg;base64,{_b64_thumbnail(whole_region_path)}"></a>
</body></html>"""
    Path(out_path).write_text(html)
    return out_path


def main(sample, output_dir, hires_path, cytassist_rgb_path=None, cap_long_side=12000,
         before_after_cap=3000):
    output_dir = Path(output_dir)
    transform = np.load(output_dir / f"{sample}_transform.npz", allow_pickle=True)
    stats = json.loads((output_dir / f"{sample}_run_stats.json").read_text())
    cytassist_rgb_path = cytassist_rgb_path or stats["cytassist_path"]

    cyt_rgb = tifffile.imread(cytassist_rgb_path)
    if cyt_rgb.ndim == 3 and cyt_rgb.shape[0] in (3, 4) and cyt_rgb.shape[-1] not in (3, 4):
        cyt_rgb = np.moveaxis(cyt_rgb, 0, -1)

    qc_path = build_qc_overlay(sample, output_dir, cyt_rgb, output_dir / f"{sample}_qc_overlay.jpg")
    quiver_path, quiver_stats = build_warp_quiver(sample, transform, cyt_rgb, output_dir / f"{sample}_warp_quiver.png")
    before_after_path = build_before_after_comparison(
        sample, transform, hires_path, cyt_rgb, output_dir / f"{sample}_before_after.jpg",
        cap_long_side=before_after_cap)
    whole_region_path, whole_region_stats = build_whole_region_overlay(
        sample, transform, hires_path, cyt_rgb, output_dir / f"{sample}_whole_region_overlay.jpg",
        cap_long_side=cap_long_side)

    report_path = write_html_report(sample, stats, qc_path, quiver_path, quiver_stats,
                                     before_after_path, whole_region_path, whole_region_stats,
                                     output_dir / f"{sample}_report.html")
    print(f"Wrote {report_path}")
    return report_path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sample", required=True)
    p.add_argument("--output-dir", required=True, help="directory containing register.py's outputs")
    p.add_argument("--hires", required=True, help="path to the native-resolution HiRes TIFF")
    p.add_argument("--cytassist", default=None, help="default: read from <sample>_run_stats.json")
    p.add_argument("--cap-long-side", type=int, default=12000,
                    help="max long-side pixels for the whole-region overlay JPEG")
    p.add_argument("--before-after-cap", type=int, default=3000,
                    help="max long-side pixels (per panel) for the before/after comparison JPEG")
    args = p.parse_args()
    main(args.sample, args.output_dir, args.hires, args.cytassist, args.cap_long_side, args.before_after_cap)
