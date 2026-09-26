# Tissue Detection Pipeline (Optional SAM)

AI-powered tissue segmentation for CytAssist images using Segment Anything Model (SAM).

**Key Feature:** Works with or without SAM. Without SAM, uses morphological approximations. With SAM, gets true learned tissue boundaries.

## Quick Start (No Installation)

Works immediately with morphological approximations:

```bash
python tissue_detection_pipeline.py \
  /path/to/CytAssist.tif \
  --sample NMR2_DRG \
  --output-dir ./tissue_detection_output \
  --force-morphological
```

Output:
- `NMR2_DRG_tissue_mask.png` – Binary tissue mask
- `NMR2_DRG_tissue_overlay.png` – Visualization
- `NMR2_DRG_tissue_mask.npz` – Compressed mask + bounds
- `NMR2_DRG_tissue_metadata.json` – Bounds and statistics

## Optional: Install SAM (Recommended)

For true learned tissue boundaries:

```bash
pip install segment-anything torch torchvision opencv-python
```

Then run without `--force-morphological`:

```bash
python tissue_detection_pipeline.py \
  /path/to/CytAssist.tif \
  --sample NMR2_DRG \
  --output-dir ./tissue_detection_output
```

**Note:** First run downloads SAM model (~375MB). Uses cached model after that.

## Integrate with Space Ranger

Bypass Space Ranger's tissue detection using AI-detected masks:

```bash
python integrate_tissue_with_spaceranger.py \
  --spaceranger-json /path/to/outs/alignment.json \
  --tissue-metadata ./tissue_detection_output/NMR2_DRG_tissue_metadata.json \
  --output ./aligned_with_tissue.json
```

This adds tissue bounds and coverage to Space Ranger's output.

## Pipeline Steps

1. **Seed Point Detection**: HSV saturation analysis → ~5 tissue centroids
2. **Segmentation**: SAM (if available) or morphological closing
3. **Cleaning**: Remove small artifacts, extract bounds
4. **Output**: Mask PNG, overlay, NPZ, JSON metadata

## Troubleshooting

### "No tissue found" or bad seed points
```bash
python tissue_detection_pipeline.py ... --saturation-threshold 20
```

### Memory issues
```bash
python tissue_detection_pipeline.py ... --model-type vit_b --force-morphological
```

### First run slow
SAM downloads model (~375MB) on first use. Subsequent runs are fast.

## Output Files

- **tissue_mask.png**: Binary mask (white=tissue, black=background)
- **tissue_overlay.png**: Original with green tissue overlay
- **tissue_mask.npz**: Compressed mask + bounds
- **tissue_metadata.json**: Bounds, center, coverage %, shape

```json
{
  "sample": "NMR2_DRG",
  "bounds": {
    "bbox": [x_min, y_min, x_max, y_max],
    "center": [cx, cy],
    "coverage_percent": 15.2,
    ...
  }
}
```

## Space Ranger Integration

Adds this to alignment.json:

```json
{
  "tissue_detection": {
    "method": "SAM with HSV seed detection",
    "sample": "NMR2_DRG",
    "bounds": {...},
    "coverage_percent": 15.2,
    "bypass_spaceranger_detection": true
  }
}
```

## FAQ

**Q: Do I need SAM?**  
A: No. Morphological approximations work without it.

**Q: Can I use this independently?**  
A: Yes. Tissue masks are standalone.

**Q: What if saturation is wrong?**  
A: Adjust `--saturation-threshold` (default 30, try 15-25).

**Q: Deterministic?**  
A: Yes. Same input → same output.

**Q: GPU acceleration?**  
A: `--device cuda` for NVIDIA GPU. macOS uses Metal automatically.

## Files

- `tissue_detection_pipeline.py` – Main detection script
- `integrate_tissue_with_spaceranger.py` – Space Ranger JSON integration
- `TISSUE_DETECTION_README.md` – This file

## References

- SAM: [Kirillov et al., 2023](https://arxiv.org/abs/2304.02643)
- Code: [facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
