#!/usr/bin/env python3
"""
Main entry point for the Basketball Player Mapping Tool.
This script processes all images in the data/images directory and saves outputs to outputs/.
"""

import sys
import os
import glob

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import main function from map_players
from map_players import main

def process_all_images():
    """Process all images in the data/images directory."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths
    images_dir = os.path.join(script_dir, 'data', 'images')
    outputs_dir = os.path.join(script_dir, 'outputs')
    
    # Create outputs directory if it doesn't exist
    os.makedirs(outputs_dir, exist_ok=True)
    
    # Find all image files (png, jpg, jpeg)
    # Use only lowercase extensions since Windows filesystem is case-insensitive
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(images_dir, ext)))
    
    # Remove duplicates (in case of case-insensitive filesystem issues)
    # Convert to set of normalized paths, then back to list
    image_files = list(set(os.path.normpath(f) for f in image_files))
    
    # Sort image files for consistent processing order
    image_files.sort()
    
    if not image_files:
        print(f"No images found in {images_dir}")
        return
    
    print(f"=" * 70)
    print(f"BATCH PROCESSING: {len(image_files)} IMAGES")
    print(f"=" * 70)
    print(f"Images directory: {images_dir}")
    print(f"Outputs directory: {outputs_dir}")
    print(f"\nFound {len(image_files)} image(s):")
    for i, img_path in enumerate(image_files, 1):
        print(f"  {i}. {os.path.basename(img_path)}")
    print()
    
    # Process each image
    for i, image_path in enumerate(image_files, 1):
        image_name = os.path.basename(image_path)
        print(f"\n{'='*70}")
        print(f"PROCESSING IMAGE {i}/{len(image_files)}: {image_name}")
        print(f"{'='*70}\n")
        
        try:
            # Process the image
            main(image_path=image_path, output_dir=outputs_dir)
            print(f"\n✓ Successfully processed: {image_name}")
        except Exception as e:
            print(f"\n✗ Error processing {image_name}: {e}")
            print(f"Continuing with next image...\n")
            continue
    
    print(f"\n{'='*70}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Processed {len(image_files)} image(s)")
    print(f"Outputs saved to: {outputs_dir}")


if __name__ == '__main__':
    process_all_images()
