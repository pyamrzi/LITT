# Practical DAB-H Cell Location Extractor
# This file provides optimized approaches for extracting DAB-H positive cell locations

"""
COMPUTATIONAL EFFICIENCY RECOMMENDATIONS FOR DAB-H POSITIVE CELL EXTRACTION
==========================================================================

Based on your existing code analysis, here are the most efficient approaches:

1. **TILE-BASED PROCESSING WITH COORDINATE TRACKING** (Recommended)
   - Builds on your existing efficient tiling strategy
   - Memory efficient for large whole-slide images
   - Parallel processing friendly

2. **SPARSE REPRESENTATION** 
   - Only store coordinates, not full images
   - Use compressed formats for very large datasets

3. **MULTI-LEVEL APPROACH**
   - Quick detection at low resolution
   - Detailed analysis only in positive regions

4. **OPTIMIZED DATA STRUCTURES**
   - NumPy arrays for coordinates
   - HDF5 for very large datasets
   - Spatial indexing for fast queries

KEY EFFICIENCY IMPROVEMENTS TO YOUR CURRENT CODE:
===============================================

A. Pre-compute expensive operations (✓ you already do this with butterworth filter)
B. Skip empty tiles early (✓ you already do this with mask checking)
C. Store locations instead of just counts (✗ missing - this is what you need)
D. Use vectorized operations where possible
E. Consider parallel processing for independent tiles

MEMORY USAGE COMPARISON:
======================
- Centroids only: ~16 bytes per cell (x, y coordinates as float64)
- Bounding boxes: ~32 bytes per cell (4 coordinates)
- Full contours: 100-1000+ bytes per cell (depends on cell complexity)

For a typical WSI with 100,000 DAB+ cells:
- Centroids: ~1.6 MB
- Bounding boxes: ~3.2 MB  
- Full contours: 10-100+ MB

RECOMMENDATION FOR YOUR USE CASE:
===============================
"""

import numpy as np
from typing import List, Tuple, Dict, Union, Iterator
import time

def extract_dabh_centroids_optimized(large_recon_pv, mask=None, down_factor=10000):
    """
    MOST EFFICIENT approach for extracting DAB-H positive cell locations.
    
    This function modifies your existing DABLR function to:
    1. Keep all your validated image processing logic
    2. Add coordinate extraction with minimal overhead
    3. Return centroids in global coordinates
    
    Returns:
    --------
    centroids : np.ndarray
        Array of shape (N, 2) containing (x, y) coordinates of cell centers
    tile_summary : dict
        Processing statistics for validation
    """
    # Import your existing functions
    from DABLR import filter_shapes_by_overlap  # Assuming this exists
    from get_LR_tissue_mask import get_LR_tissue_mask
    from get_tile_list import get_tile_list
    from skimage.filters import threshold_multiotsu, butterworth
    from skimage.morphology import remove_small_objects, binary_erosion, binary_dilation
    from skimage.color import rgb2hsv, rgb2hed
    from skimage.measure import label, regionprops
    import tqdm
    
    # Your existing preprocessing (keep identical for validation)
    large_recon = large_recon_pv.numpy()
    tiles = get_tile_list(large_recon, down_factor)
    
    if mask is None:
        mask_LR, _ = get_LR_tissue_mask(large_recon_pv)
        mask_LR = mask_LR.numpy()
    else:
        mask_LR = mask
    
    dabfull = rgb2hed(large_recon)
    highpassdabfull = butterworth(dabfull[..., 2], cutoff_frequency_ratio=.02)
    
    # Collect centroids instead of just counts
    centroids = []
    tile_summary = {'processed_tiles': 0, 'skipped_tiles': 0, 'total_cells': 0}
    
    for ((start_x, start_y, end_x, end_y), (x, y)) in tqdm.tqdm(tiles, desc="Extracting DAB-H cells"):
        mask_tile = mask_LR[start_x:end_x, start_y:end_y]
        if not np.any(mask_tile):
            tile_summary['skipped_tiles'] += 1
            continue
        
        tile_summary['processed_tiles'] += 1
        
        # YOUR EXISTING PROCESSING LOGIC (KEEP IDENTICAL)
        tile = large_recon[start_x:end_x, start_y:end_y, :]
        dab = highpassdabfull[start_x:end_x, start_y:end_y]
        
        dabbin = dab > .08
        dabbinb = remove_small_objects(dabbin, min_size=4)
        hsvt = rgb2hsv(tile)
        dif = hsvt[..., 1] - np.sum(tile, axis=2)
        
        try:
            _, dif_thr = threshold_multiotsu(dif)
        except:
            dif_thr = 10000
            
        difbin = dif > dif_thr
        difbin_e = binary_erosion(difbin)
        difbin_be = remove_small_objects(difbin_e, min_size=680, connectivity=1)
        difbin_bed = binary_dilation(difbin_be)
        dont_analyze = filter_shapes_by_overlap(difbin, difbin_bed).astype(bool)
        dabbinb = dabbinb * np.invert(dont_analyze)
        
        hem = dabfull[start_x:end_x, start_y:end_y, 0]
        hemthresh = .015
        hembin = (hem > hemthresh)
        hembin = hembin * np.invert(dont_analyze)
        hembinb = remove_small_objects(hembin, min_size=4)
        lobjshem = remove_small_objects(hembinb, min_size=60)
        hembinm = hembinb ^ lobjshem
        
        lobjsdab = remove_small_objects(dabbinb, min_size=70)
        dabbinm = dabbinb ^ lobjsdab
        clumpycellsdab = remove_small_objects(dabbinm, min_size=35)
        smallcellsdab = dabbinm ^ clumpycellsdab
        clumpycellsdabero = binary_erosion(clumpycellsdab)
        dabbinmselero = smallcellsdab + clumpycellsdabero
        
        # EXTRACT COORDINATES (NEW ADDITION)
        labeled_dab = label(dabbinmselero, connectivity=1)
        
        # Get centroids in global coordinates
        for region in regionprops(labeled_dab):
            centroid_y, centroid_x = region.centroid
            global_x = start_x + centroid_x
            global_y = start_y + centroid_y
            centroids.append([global_x, global_y])
    
    tile_summary['total_cells'] = len(centroids)
    
    return np.array(centroids), tile_summary

def save_cell_locations(centroids, filename, format='npy'):
    """
    Efficiently save cell locations in various formats.
    
    Parameters:
    -----------
    centroids : np.ndarray
        Array of cell coordinates
    filename : str
        Output filename (extension will be added automatically)
    format : str
        'npy' - NumPy binary (fastest)
        'csv' - Comma-separated values (human readable)
        'hdf5' - HDF5 format (best for very large datasets)
    """
    if format == 'npy':
        np.save(f"{filename}.npy", centroids)
        print(f"Saved {len(centroids)} cell locations to {filename}.npy")
        
    elif format == 'csv':
        np.savetxt(f"{filename}.csv", centroids, delimiter=',', 
                   header='x,y', comments='', fmt='%.2f')
        print(f"Saved {len(centroids)} cell locations to {filename}.csv")
        
    elif format == 'hdf5':
        import h5py
        with h5py.File(f"{filename}.h5", 'w') as f:
            f.create_dataset('cell_centroids', data=centroids, compression='gzip')
            f.attrs['num_cells'] = len(centroids)
            f.attrs['format'] = 'x,y coordinates'
        print(f"Saved {len(centroids)} cell locations to {filename}.h5")

def benchmark_approaches(large_recon_pv, down_factor=10000):
    """
    Compare computational efficiency of different approaches.
    """
    print("BENCHMARKING DAB-H CELL EXTRACTION APPROACHES")
    print("=" * 50)
    
    # Approach 1: Your current method (counts only)
    start_time = time.time()
    # Would call your existing DABLR function here
    # cell_counts, dabh_counts, ratios, mask = DABLR(large_recon_pv, down_factor=down_factor)
    # approach1_time = time.time() - start_time
    # print(f"Approach 1 (counts only): {approach1_time:.2f} seconds")
    
    # Approach 2: Modified method with coordinates
    start_time = time.time()
    centroids, summary = extract_dabh_centroids_optimized(large_recon_pv, down_factor=down_factor)
    approach2_time = time.time() - start_time
    print(f"Approach 2 (with coordinates): {approach2_time:.2f} seconds")
    print(f"Found {len(centroids)} DAB-H positive cells")
    print(f"Memory usage: ~{len(centroids) * 16 / 1024:.1f} KB for centroids")
    
    return centroids, summary

# USAGE EXAMPLES
def usage_examples():
    """
    Examples of how to use the efficient extraction functions.
    """
    print("\nUSAGE EXAMPLES:")
    print("-" * 30)
    
    print("1. Basic extraction:")
    print("   centroids, summary = extract_dabh_centroids_optimized(large_recon_pv)")
    print("   print(f'Found {len(centroids)} cells')")
    
    print("\n2. Save results:")
    print("   save_cell_locations(centroids, 'my_slide_dabh_cells', format='csv')")
    
    print("\n3. Process multiple slides:")
    print("   for slide_path in slide_paths:")
    print("       centroids, _ = extract_dabh_centroids_optimized(load_slide(slide_path))")
    print("       save_cell_locations(centroids, f'{slide_path}_dabh_cells')")
    
    print("\n4. Memory-efficient processing:")
    print("   # Process smaller tiles for very large images")
    print("   centroids, _ = extract_dabh_centroids_optimized(image, down_factor=5000)")

if __name__ == "__main__":
    usage_examples()
