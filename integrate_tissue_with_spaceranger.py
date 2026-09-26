#!/usr/bin/env python3
"""
Integrate tissue detection masks into Space Ranger alignment JSON.
Allows bypassing Space Ranger's tissue detection using AI-detected masks.

Usage:
  python integrate_tissue_with_spaceranger.py \
    --spaceranger-json /path/to/alignment.json \
    --tissue-metadata tissue_metadata.json \
    --output modified_alignment.json
"""
import json
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def load_tissue_metadata(metadata_path: str) -> dict:
    """Load tissue detection metadata."""
    with open(metadata_path, 'r') as f:
        return json.load(f)

def load_spaceranger_json(spaceranger_json: str) -> dict:
    """Load Space Ranger alignment JSON."""
    with open(spaceranger_json, 'r') as f:
        return json.load(f)

def integrate_tissue_bounds(spaceranger_data: dict, tissue_metadata: dict) -> dict:
    """Add tissue bounds to Space Ranger JSON."""
    bounds = tissue_metadata.get('bounds', {})
    sample = tissue_metadata.get('sample', 'unknown')
    
    if 'tissue_detection' not in spaceranger_data:
        spaceranger_data['tissue_detection'] = {}
    
    spaceranger_data['tissue_detection'].update({
        'method': 'SAM (Segment Anything Model) with HSV seed detection',
        'sample': sample,
        'bounds': bounds,
        'coverage_percent': bounds.get('coverage_percent', 0),
        'bypass_spaceranger_detection': True,
    })
    
    logger.info(f"Added tissue detection for {sample}")
    logger.info(f"  Bounds: {bounds.get('bbox', 'N/A')}")
    logger.info(f"  Coverage: {bounds.get('coverage_percent', 0):.1f}%")
    return spaceranger_data

def main():
    parser = argparse.ArgumentParser(description="Integrate tissue detection into Space Ranger JSON")
    parser.add_argument("--spaceranger-json", required=True, help="Path to Space Ranger alignment.json")
    parser.add_argument("--tissue-metadata", required=True, help="Path to tissue_metadata.json")
    parser.add_argument("--output", required=True, help="Output path")
    parser.add_argument("--inplace", action="store_true", help="Modify in place (creates backup)")
    args = parser.parse_args()
    
    logger.info(f"Loading Space Ranger JSON: {args.spaceranger_json}")
    spaceranger_data = load_spaceranger_json(args.spaceranger_json)
    
    logger.info(f"Loading tissue metadata: {args.tissue_metadata}")
    tissue_metadata = load_tissue_metadata(args.tissue_metadata)
    
    logger.info("Integrating tissue detection...")
    spaceranger_data = integrate_tissue_bounds(spaceranger_data, tissue_metadata)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(spaceranger_data, f, indent=2)
    
    logger.info(f"✓ Saved: {output_path}")
    
    if args.inplace:
        backup_path = Path(args.spaceranger_json).with_suffix('.json.backup')
        import shutil
        shutil.copy(args.spaceranger_json, backup_path)
        shutil.copy(output_path, args.spaceranger_json)
        logger.info(f"✓ Backup: {backup_path}")
        logger.info(f"✓ Updated: {args.spaceranger_json}")
    
    return 0

if __name__ == "__main__":
    exit(main())
