"""
Turn a register.py transform (coarse affine + block-wise refinement) into
two deliverables, both at the HiRes image's own native resolution and
original orientation, suitable for feeding Space Ranger the way manual
Loupe alignment would:

  1. `<sample>_local_warp_only.ome.tif` -- the original HiRes image with
     ONLY the local (block-wise) correction baked in. No flip, rotation,
     scale or coarse translation.
  2. `<sample>_global_affine.json` -- the separate global affine (mirror/
     flip + rotation + scale + coarse translation), mapping full-resolution
     ORIGINAL-orientation HiRes pixel coordinates -> CytAssist pixel
     coordinates, forward direction. Same `cytAssistInfo.transformImages`
     3x3-matrix schema as the slide's own auto-generated
     fiducial-image-registration.json.

Feed the two together: the image supplies the local correction, the JSON
supplies the affine -- don't apply another non-affine warp on top of the
JSON's matrix.

Why split at all: a single affine can't represent the small but spatially
coherent local distortion between HiRes and CytAssist that palom's
block-wise refinement finds, but Space Ranger's manual-alignment-style
input expects one matrix, not a per-block warp field. See README.md for
the derivation.

Coordinate spaces, in the order the image passes through them:
  H0 -- native resolution, original orientation (the raw HiRes file).
  Hd -- H0 downsampled to CytAssist's pixel size, still original orientation.
  Ho -- Hd with the sample's flip/rotate/mirror applied (matches CytAssist's
        orientation). This is the space register.py's block matrices are in.
  C  -- CytAssist pixel coordinates.
register.py saves the H0->Hd scale (via each shape) and the Hd->Ho
transform (`orientation_matrix`) directly, whatever flip/rotate/mirror
combination it ended up using -- so nothing here is hardcoded to one
sample's orientation.

Streams the output to disk in row-bands (never holds the full ~20GB image
in memory). On a memory- or time-constrained machine, pass --checkpoint to
cache progress in a resumable zarr store next to the output, so an
interrupted run can just be re-launched with the same arguments.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hires_cytassist_common import read_series0_pyramid, um_per_pixel

WRITE_TILE = 512
HALO = 512          # margin around each band/chunk, must exceed the largest local correction
COMPUTE_W = 8192     # cv2.remap requires both dims < 32767 (SHRT_MAX)


def get_block_matrix(matrices, i, j):
    return matrices[i * 3:(i + 1) * 3, j * 3:(j + 1) * 3]


def apply_affine_pt(M, pt):
    """Apply a 3x3 homogeneous matrix to a single (x, y) point."""
    v = M @ np.array([pt[0], pt[1], 1.0])
    return v[:2] / v[2]


def h0_to_hd_matrix(hd_shape, h0_shape):
    """Pure-scale H0 -> Hd matrix (downsampling only, no reorientation)."""
    hd_h, hd_w = hd_shape
    h0_h, h0_w = h0_shape
    return np.diag([hd_w / h0_w, hd_h / h0_h, 1.0])


def global_affine(transform, h0_shape):
    """Full H0 (native, original orientation) -> C (CytAssist) affine."""
    m = transform["coarse_affine_matrix"]
    orientation_matrix = transform["orientation_matrix"]  # Hd -> Ho
    t1 = h0_to_hd_matrix(transform["moving_shape_pre_orientation"], h0_shape)  # H0 -> Hd
    return m @ orientation_matrix @ t1


def fit_local_correction(transform, h0_shape):
    """Local-only per-block correction, expressed as a function of native
    (H0-space) HiRes pixel position.

    Each block's own matrix and the coarse-only matrix both map Ho -> C,
    forward direction (confirmed from palom.block_affine's source). For
    each block center c, the difference between M_block^-1(c) and m^-1(c)
    is the local-only correction, in Ho-space. We convert that into an
    H0-space vector via the (linear part of the) Ho -> H0 transform, then
    fit a smooth function of (block row, block col) -> both position and
    correction, since the block grid is only regular in *index* space, not
    in raw pixel space (the coarse affine's rotation tilts it).
    """
    matrices = transform["block_affine_matrices"]
    m = transform["coarse_affine_matrix"]
    gr, gc = transform["grid_shape"]
    block_step = int(transform["block_step"])
    orientation_matrix = transform["orientation_matrix"]  # Hd -> Ho
    t1 = h0_to_hd_matrix(transform["moving_shape_pre_orientation"], h0_shape)  # H0 -> Hd

    # Ho -> H0, as a single 3x3 matrix. Composing then inverting (rather than
    # inverting each piece separately) keeps this correct for ANY
    # combination of flip/rotate/mirror/scale -- no per-orientation cases.
    ho_to_h0 = np.linalg.inv(orientation_matrix @ t1)
    ho_to_h0_linear = ho_to_h0[:2, :2]  # for transforming *vectors* (deltas)
    m_inv = np.linalg.inv(m)

    p_global_h0 = np.zeros((gr, gc, 2))
    delta_h0 = np.zeros((gr, gc, 2))
    for i in range(gr):
        for j in range(gc):
            c = ((j + 0.5) * block_step, (i + 0.5) * block_step)
            p_local_ho = apply_affine_pt(np.linalg.inv(get_block_matrix(matrices, i, j)), c)
            p_global_ho = apply_affine_pt(m_inv, c)
            p_global_h0[i, j] = apply_affine_pt(ho_to_h0, p_global_ho)
            delta_h0[i, j] = ho_to_h0_linear @ (p_local_ho - p_global_ho)

    # Block centers form a regular grid in (row, col) index space, not in
    # raw H0 pixel space (the coarse affine's rotation/scale tilts it) --
    # fit a plane through (j, i) -> H0 position to get a smooth mapping.
    ii, jj = np.meshgrid(np.arange(gr), np.arange(gc), indexing="ij")
    design = np.stack([jj.ravel(), ii.ravel(), np.ones(gr * gc)], axis=1).astype(float)
    sol_x, *_ = np.linalg.lstsq(design, p_global_h0[..., 0].ravel(), rcond=None)
    sol_y, *_ = np.linalg.lstsq(design, p_global_h0[..., 1].ravel(), rcond=None)
    index_to_h0 = np.array([[sol_x[0], sol_x[1]], [sol_y[0], sol_y[1]]])   # (j,i) -> (x,y) in H0
    index_to_h0_offset = np.array([sol_x[2], sol_y[2]])
    return (index_to_h0, index_to_h0_offset,
            delta_h0[..., 0].astype(np.float32), delta_h0[..., 1].astype(np.float32))


def make_corrector(index_to_h0, index_to_h0_offset, delta_dx_grid, delta_dy_grid, grid_shape):
    gr, gc = grid_shape
    h0_to_index = np.linalg.inv(index_to_h0)

    def correction(r0, r1, c0, c1):
        yy, xx = np.mgrid[r0:r1, c0:c1].astype(np.float32)
        ji = (np.stack([xx.ravel(), yy.ravel()], axis=1) - index_to_h0_offset) @ h0_to_index.T
        ji[:, 0] = np.clip(ji[:, 0], 0, gc - 1)
        ji[:, 1] = np.clip(ji[:, 1], 0, gr - 1)
        map_j, map_i = (ji[:, k].reshape(xx.shape).astype(np.float32) for k in (0, 1))
        dx = cv2.remap(delta_dx_grid, map_j, map_i, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        dy = cv2.remap(delta_dy_grid, map_j, map_i, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return xx, yy, dx, dy
    return correction


def corrected_band(h0, correction, r0, r1):
    """One row-band of the locally-corrected image, computed in column
    chunks (cv2.remap needs both dims < 32767)."""
    H0_w = h0.shape[1]
    src_r0, src_r1 = max(0, r0 - HALO), min(h0.shape[0], r1 + HALO)
    src_full = h0[src_r0:src_r1, :, :].compute()
    out = np.empty((r1 - r0, H0_w, 3), dtype=np.uint8)
    for cw0 in range(0, H0_w, COMPUTE_W):
        cw1 = min(cw0 + COMPUTE_W, H0_w)
        src_c0, src_c1 = max(0, cw0 - HALO), min(H0_w, cw1 + HALO)
        xx, yy, dx, dy = correction(r0, r1, cw0, cw1)
        src_x = np.ascontiguousarray((xx + dx) - src_c0, dtype=np.float32)
        src_y = np.ascontiguousarray((yy + dy) - src_r0, dtype=np.float32)
        for c in range(3):
            out[:, cw0:cw1, c] = cv2.remap(src_full[:, src_c0:src_c1, c], src_x, src_y,
                                            interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return out


def write_local_warp_tiff(h0, correction, out_path, um_per_px, checkpoint_dir=None):
    H0_h, H0_w = h0.shape[:2]
    n_bands = (H0_h + WRITE_TILE - 1) // WRITE_TILE

    cache = None
    if checkpoint_dir is not None:
        root = zarr.open(zarr.DirectoryStore(str(checkpoint_dir)), mode="a")
        if "corrected" not in root:
            root.create_dataset("corrected", shape=(H0_h, H0_w, 3), chunks=(WRITE_TILE, 4096, 3), dtype="uint8")
            root.create_dataset("done", shape=(n_bands,), dtype="bool", fill_value=False)
        cache = root

    def bands():
        for bi in range(n_bands):
            r0, r1 = bi * WRITE_TILE, min(bi * WRITE_TILE + WRITE_TILE, H0_h)
            if cache is not None and bool(cache["done"][bi]):
                band = cache["corrected"][r0:r1, :, :]
            else:
                band = corrected_band(h0, correction, r0, r1)
                if cache is not None:
                    cache["corrected"][r0:r1, :, :] = band
                    cache["done"][bi] = True
            if bi % 20 == 0:
                print(f"band {bi + 1}/{n_bands}", flush=True)
            for c0 in range(0, H0_w, WRITE_TILE):
                c1 = min(c0 + WRITE_TILE, H0_w)
                yield band[:, c0:c1, :]

    res = (1e4 / um_per_px, 1e4 / um_per_px)
    with tifffile.TiffWriter(out_path, bigtiff=True) as tw:
        tw.write(bands(), shape=(H0_h, H0_w, 3), dtype=np.uint8, tile=(WRITE_TILE, WRITE_TILE),
                  photometric="rgb", compression="jpeg", resolution=res, resolutionunit=tifffile.RESUNIT.CENTIMETER)


def main(hires_path, transform_npz, output_dir, sample="sample", serial_number="", area="", checkpoint=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    transform = np.load(transform_npz, allow_pickle=True)

    pyramid = read_series0_pyramid(hires_path)
    h0 = pyramid[0]
    h0_shape = h0.shape[:2]
    index_to_h0, index_to_h0_offset, ddx, ddy = fit_local_correction(transform, h0_shape)

    total_affine = global_affine(transform, h0_shape)
    affine_path = output_dir / f"{sample}_global_affine.json"
    affine_path.write_text(json.dumps({
        "_generated_by": "build_full_res_deliverables.py",
        "_note": ("Global affine only (no local warp): maps full-resolution, ORIGINAL-"
                  "orientation HiRes pixel coordinates to CytAssist pixel coordinates, "
                  "forward direction. The local (block-wise) warp is already baked into "
                  f"{sample}_local_warp_only.ome.tif -- don't apply another non-affine "
                  "correction on top of this matrix."),
        "serialNumber": serial_number, "area": area,
        "cytAssistInfo": {"transformImages": total_affine.tolist()},
    }, indent=2))
    print(f"Wrote {affine_path}\n{total_affine}")

    correction = make_corrector(index_to_h0, index_to_h0_offset, ddx, ddy, transform["grid_shape"])
    checkpoint_dir = (output_dir / f".{sample}_checkpoint.zarr") if checkpoint else None
    out_tiff = output_dir / f"{sample}_local_warp_only.ome.tif"
    write_local_warp_tiff(h0, correction, str(out_tiff), um_per_pixel(hires_path), checkpoint_dir)
    print(f"Wrote {out_tiff}")
    if checkpoint_dir is not None:
        print(f"Done -- you can delete the checkpoint cache: {checkpoint_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--hires", required=True)
    p.add_argument("--transform-npz", required=True, help="output of register.py")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--sample", default="sample")
    p.add_argument("--serial-number", default="")
    p.add_argument("--area", default="")
    p.add_argument("--checkpoint", action="store_true",
                    help="cache progress to disk so an interrupted run can resume "
                         "(useful on RAM- or time-constrained machines)")
    args = p.parse_args()
    main(args.hires, args.transform_npz, args.output_dir, sample=args.sample,
         serial_number=args.serial_number, area=args.area, checkpoint=args.checkpoint)
