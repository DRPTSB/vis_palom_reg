#!/usr/bin/env python3
"""
Tissue detection for Visium alignment pipeline.
Uses SAM for zero-shot tissue segmentation, with morphological fallback.
Optional SAM installation - morphological approximations work without it.
"""
import cv2
import numpy as np
from pathlib import Path
import argparse
import json
import logging
from typing import Tuple, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def find_tissue_seed_points(image: np.ndarray,
                           n_points: int = 5,
                           min_area: int = 500,
                           saturation_threshold: int = 30) -> List[Tuple[int, int]]:
    """Find obvious tissue regions as SAM prompts using HSV saturation."""
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = img_hsv[:, :, 1]
    tissue_mask = saturation > saturation_threshold
    tissue_uint8 = (tissue_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(tissue_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_by_area = sorted(
        [(cv2.contourArea(c), c) for c in contours],
        key=lambda x: x[0],
        reverse=True
    )
    seed_points = []
    for area, contour in contours_by_area:
        if area < min_area:
            continue
        M = cv2.moments(contour)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            seed_points.append((cx, cy))
            logger.info(f"  Tissue blob: area={area:.0f}, centroid=({cx}, {cy})")
        if len(seed_points) >= n_points:
            break
    logger.info(f"Found {len(seed_points)} seed points")
    return seed_points

def segment_tissue_with_sam(image_path: str, tissue_points: List[Tuple[int, int]],
                           model_type: str = "vit_b", device_str: str = "cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Run SAM segmentation. Returns (image, mask) or (image, None) if SAM unavailable."""
    try:
        import torch
        from segment_anything import sam_model_registry, SamPredictor
    except ImportError:
        logger.warning("SAM not installed. Use: pip install segment-anything torch")
        img = cv2.imread(image_path)
        if img is None:
            from PIL import Image
            img = np.array(Image.open(image_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
        return img_rgb, None
    
    logger.info(f"Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        from PIL import Image
        img = np.array(Image.open(image_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img
    
    device = torch.device(device_str)
    logger.info(f"Loading SAM {model_type}...")
    sam = sam_model_registry[model_type](checkpoint=f"sam_{model_type}_01ec64.pth")
    sam = sam.to(device)
    predictor = SamPredictor(sam)
    predictor.set_image(img)
    
    point_coords = np.array(tissue_points)
    point_labels = np.ones(len(tissue_points), dtype=int)
    masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, multimask_output=False)
    logger.info(f"SAM score: {scores[0]:.3f}")
    return img, masks[0]

def morphological_tissue_mask(image: np.ndarray, saturation_threshold: int = 30, min_area: int = 100) -> np.ndarray:
    """Fallback: Create tissue mask using morphological operations."""
    img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = img_hsv[:, :, 1]
    tissue_uint8 = ((saturation > saturation_threshold).astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tissue_uint8 = cv2.morphologyEx(tissue_uint8, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(tissue_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(tissue_uint8)
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(clean_mask, [contour], 0, 255, -1)
    return clean_mask.astype(bool)

def clean_mask(mask: np.ndarray, min_area: int = 100, morphology_kernel_size: int = 5) -> np.ndarray:
    """Post-process mask to remove small artifacts."""
    mask_uint8 = (mask.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morphology_kernel_size, morphology_kernel_size))
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(mask_uint8)
    kept = 0
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            cv2.drawContours(clean_mask, [contour], 0, 255, -1)
            kept += 1
    logger.info(f"Cleaned: kept {kept}/{len(contours)} components")
    return clean_mask.astype(bool)

def extract_tissue_bounds(mask: np.ndarray) -> dict:
    """Extract bounding box and statistics from tissue mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    tissue_pixels = np.sum(mask)
    total_pixels = mask.size
    coverage = 100 * tissue_pixels / total_pixels
    bounds = {
        "bbox": [int(x_min), int(y_min), int(x_max), int(y_max)],
        "center": [int((x_min + x_max) / 2), int((y_min + y_max) / 2)],
        "tissue_pixels": int(tissue_pixels),
        "total_pixels": int(total_pixels),
        "coverage_percent": float(coverage),
        "shape": list(mask.shape)
    }
    logger.info(f"Tissue bounds: ({x_min}, {y_min}) to ({x_max}, {y_max}), coverage: {coverage:.1f}%")
    return bounds

def save_tissue_outputs(image: np.ndarray, mask: np.ndarray, bounds: dict,
                       output_dir: Path, sample_name: str = "sample"):
    """Save tissue mask, overlay, and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    mask_path = output_dir / f"{sample_name}_tissue_mask.png"
    cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))
    logger.info(f"Saved mask: {mask_path}")
    
    overlay = image.copy().astype(float)
    overlay[mask == 0] *= 0.3
    overlay[mask == 1] = overlay[mask == 1] * 0.6 + np.array([0, 150, 0]) * 0.4
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    overlay_path = output_dir / f"{sample_name}_tissue_overlay.png"
    cv2.imwrite(str(overlay_path), overlay_bgr)
    logger.info(f"Saved overlay: {overlay_path}")
    
    npz_path = output_dir / f"{sample_name}_tissue_mask.npz"
    np.savez_compressed(npz_path, mask=mask, bounds=bounds)
    logger.info(f"Saved NPZ: {npz_path}")
    
    metadata_path = output_dir / f"{sample_name}_tissue_metadata.json"
    metadata = {"sample": sample_name, "bounds": bounds, "mask_files": {"png": str(mask_path.name), "npz": str(npz_path.name)}}
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {metadata_path}")
    
    return mask_path, overlay_path, npz_path, metadata_path

def main():
    parser = argparse.ArgumentParser(description="Tissue detection for Visium pipeline")
    parser.add_argument("cytassist_image", help="Path to CytAssist image")
    parser.add_argument("--sample", required=True, help="Sample name")
    parser.add_argument("--output-dir", default="./tissue_detection_output", help="Output directory")
    parser.add_argument("--model-type", default="vit_b", choices=["vit_b", "vit_l", "vit_h"], help="SAM model")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--n-seed-points", type=int, default=5, help="Number of seed points")
    parser.add_argument("--min-blob-area", type=int, default=500, help="Min blob area")
    parser.add_argument("--saturation-threshold", type=int, default=30, help="Saturation threshold")
    parser.add_argument("--min-mask-area", type=int, default=100, help="Min mask area")
    parser.add_argument("--no-clean", action="store_true", help="Skip mask cleaning")
    parser.add_argument("--force-morphological", action="store_true", help="Use morphological only")
    args = parser.parse_args()

    logger.info(f"=== Tissue Detection ===")
    logger.info(f"Sample: {args.sample}, Image: {args.cytassist_image}")
    
    logger.info("Step 1: Finding tissue seed points...")
    img_cv = cv2.imread(args.cytassist_image)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    seed_points = find_tissue_seed_points(img_rgb, args.n_seed_points, args.min_blob_area, args.saturation_threshold)
    
    if not seed_points:
        logger.error("No tissue seed points found!")
        return 1

    logger.info("Step 2: Running tissue segmentation...")
    if args.force_morphological:
        image, tissue_mask = img_rgb, morphological_tissue_mask(img_rgb, args.saturation_threshold, args.min_mask_area)
    else:
        image, tissue_mask = segment_tissue_with_sam(args.cytassist_image, seed_points, args.model_type, args.device)
        if tissue_mask is None:
            logger.info("Using morphological fallback...")
            tissue_mask = morphological_tissue_mask(img_rgb, args.saturation_threshold, args.min_mask_area)

    if not args.no_clean:
        logger.info("Step 3: Cleaning mask...")
        tissue_mask = clean_mask(tissue_mask, args.min_mask_area)

    logger.info("Step 4: Extracting tissue bounds...")
    bounds = extract_tissue_bounds(tissue_mask)
    if bounds is None:
        logger.error("No tissue found!")
        return 1

    logger.info("Step 5: Saving outputs...")
    mask_path, overlay_path, npz_path, metadata_path = save_tissue_outputs(image, tissue_mask, bounds, args.output_dir, args.sample)
    logger.info(f"\n=== Complete ===\nOutputs: {Path(args.output_dir).absolute()}")
    return 0

if __name__ == "__main__":
    exit(main())
