#!/usr/bin/env python3
"""
Spatial Analysis Script for Cell-Vessel Distance Calculations

This script processes pickle files containing vessel and LITT margin polygons along with 
CSV files containing cell centroids to generate a comprehensive analysis of DAB+ and DAB- 
cellularity as a function of distance from vessel margins.

The script calculates:
1. Distance from each vessel centroid to nearest LITT margin
2. Distance from each cell to nearest vessel margin 
3. Bins cells by distance from vessels (configurable bin sizes)
4. Generates summary statistics and ratios

Output CSV format:
- vessel_id: Unique identifier for each vessel
- vessel_distance_to_litt: Distance from vessel centroid to nearest LITT margin
- bin_X_Y_dab_pos/neg/total/ratio: Cell counts and ratios for each distance bin

Created by M. Pouya Mirzaei for the LaViolette Lab
Medical College of Wisconsin
"""

import numpy as np
import pandas as pd
import pickle
import argparse
import os
import sys
from pathlib import Path
from tqdm import tqdm
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points
from shapely.strtree import STRtree
import warnings

# Suppress shapely warnings for cleaner output
warnings.filterwarnings('ignore', category=RuntimeWarning)


def load_spatial_data(vessel_pkl_path, litt_pkl_path, pos_csv_path, neg_csv_path):
    """
    Load spatial data from pickle and CSV files.
    
    Args:
        vessel_pkl_path (str): Path to vessel polygons pickle file
        litt_pkl_path (str): Path to LITT polygons pickle file  
        pos_csv_path (str): Path to DAB+ cell centroids CSV
        neg_csv_path (str): Path to DAB- cell centroids CSV
        
    Returns:
        tuple: (vessel_polygons, litt_polygons, pos_cells_df, neg_cells_df)
    """
    print("Loading spatial data...")
    
    # Load vessel polygons
    try:
        with open(vessel_pkl_path, 'rb') as f:
            vessel_data = pickle.load(f)
        print(f"Loaded {len(vessel_data)} vessel polygons from {vessel_pkl_path}")
    except Exception as e:
        raise FileNotFoundError(f"Error loading vessel polygons: {e}")
    
    # Load LITT polygons
    try:
        with open(litt_pkl_path, 'rb') as f:
            litt_data = pickle.load(f)
        print(f"Loaded {len(litt_data)} LITT polygons from {litt_pkl_path}")
    except Exception as e:
        raise FileNotFoundError(f"Error loading LITT polygons: {e}")
    
    # Convert polygon data to Shapely polygons if needed
    vessel_polygons = []
    for i, vessel in enumerate(vessel_data):
        if isinstance(vessel, np.ndarray):
            # Convert numpy array of points to Shapely Polygon
            if len(vessel) >= 3:  # Need at least 3 points for a polygon
                vessel_polygons.append(Polygon(vessel))
            else:
                print(f"Warning: Vessel {i} has insufficient points ({len(vessel)}), skipping")
        elif hasattr(vessel, 'exterior'):  # Already a Shapely polygon
            vessel_polygons.append(vessel)
        else:
            print(f"Warning: Unknown vessel data type for vessel {i}: {type(vessel)}")
    
    litt_polygons = []
    for i, litt in enumerate(litt_data):
        if isinstance(litt, np.ndarray):
            # Convert numpy array of points to Shapely Polygon
            if len(litt) >= 3:  # Need at least 3 points for a polygon
                litt_polygons.append(Polygon(litt))
            else:
                print(f"Warning: LITT {i} has insufficient points ({len(litt)}), skipping")
        elif hasattr(litt, 'exterior'):  # Already a Shapely polygon
            litt_polygons.append(litt)
        else:
            print(f"Warning: Unknown LITT data type for LITT {i}: {type(litt)}")
    
    print(f"Converted to {len(vessel_polygons)} valid vessel polygons")
    print(f"Converted to {len(litt_polygons)} valid LITT polygons")
    
    # Load cell centroids
    try:
        pos_cells_df = pd.read_csv(pos_csv_path)
        print(f"Loaded {len(pos_cells_df)} DAB+ cells from {pos_csv_path}")
        
        # Validate required columns (try common column name variations)
        x_col, y_col = None, None
        for x_name in ['x', 'X', 'centroid_x', 'Centroid X', 'X_centroid']:
            if x_name in pos_cells_df.columns:
                x_col = x_name
                break
        for y_name in ['y', 'Y', 'centroid_y', 'Centroid Y', 'Y_centroid']:
            if y_name in pos_cells_df.columns:
                y_col = y_name
                break
                
        if x_col is None or y_col is None:
            raise ValueError(f"Could not find x,y coordinate columns in {pos_csv_path}. Available columns: {list(pos_cells_df.columns)}")
        
        # Standardize column names
        pos_cells_df = pos_cells_df.rename(columns={x_col: 'x', y_col: 'y'})
        
    except Exception as e:
        raise FileNotFoundError(f"Error loading DAB+ cells: {e}")
    
    try:
        neg_cells_df = pd.read_csv(neg_csv_path)
        print(f"Loaded {len(neg_cells_df)} DAB- cells from {neg_csv_path}")
        
        # Find and standardize column names
        x_col, y_col = None, None
        for x_name in ['x', 'X', 'centroid_x', 'Centroid X', 'X_centroid']:
            if x_name in neg_cells_df.columns:
                x_col = x_name
                break
        for y_name in ['y', 'Y', 'centroid_y', 'Centroid Y', 'Y_centroid']:
            if y_name in neg_cells_df.columns:
                y_col = y_name
                break
                
        if x_col is None or y_col is None:
            raise ValueError(f"Could not find x,y coordinate columns in {neg_csv_path}. Available columns: {list(neg_cells_df.columns)}")
        
        # Standardize column names
        neg_cells_df = neg_cells_df.rename(columns={x_col: 'x', y_col: 'y'})
        
    except Exception as e:
        raise FileNotFoundError(f"Error loading DAB- cells: {e}")
    
    return vessel_polygons, litt_polygons, pos_cells_df, neg_cells_df


def calculate_vessel_to_litt_distances(vessel_polygons, litt_polygons):
    """
    Calculate distance from each vessel centroid to nearest LITT margin.
    
    Args:
        vessel_polygons (list): List of Shapely Polygon objects for vessels
        litt_polygons (list): List of Shapely Polygon objects for LITT margins
        
    Returns:
        list: Distances from each vessel to nearest LITT margin
    """
    print("Calculating vessel-to-LITT distances...")
    
    if not litt_polygons:
        print("Warning: No LITT polygons found, setting all distances to NaN")
        return [np.nan] * len(vessel_polygons)
    
    vessel_to_litt_distances = []
    
    for i, vessel in enumerate(tqdm(vessel_polygons, desc="Computing vessel-LITT distances")):
        try:
            vessel_centroid = vessel.centroid
            
            # Find minimum distance to any LITT polygon boundary
            min_distance = float('inf')
            for litt_polygon in litt_polygons:
                distance = vessel_centroid.distance(litt_polygon.boundary)
                min_distance = min(min_distance, distance)
            
            vessel_to_litt_distances.append(min_distance if min_distance != float('inf') else np.nan)
            
        except Exception as e:
            print(f"Warning: Error calculating distance for vessel {i}: {e}")
            vessel_to_litt_distances.append(np.nan)
    
    return vessel_to_litt_distances


def calculate_cell_to_vessel_distances(cells_df, vessel_polygons):
    """
    Calculate minimum distance from each cell centroid to nearest vessel margin.
    
    Args:
        cells_df (pd.DataFrame): DataFrame with cell coordinates (columns: 'x', 'y')
        vessel_polygons (list): List of Shapely Polygon objects for vessels
        
    Returns:
        pd.DataFrame: Input DataFrame with added 'distance_to_vessel' and 'nearest_vessel_id' columns
    """
    print(f"Calculating cell-to-vessel distances for {len(cells_df)} cells...")
    
    if not vessel_polygons:
        print("Warning: No vessel polygons found, setting all distances to NaN")
        cells_df['distance_to_vessel'] = np.nan
        cells_df['nearest_vessel_id'] = -1
        return cells_df
    
    distances = []
    nearest_vessel_ids = []
    
    # Create vessel boundaries for distance calculations
    vessel_boundaries = [vessel.boundary for vessel in vessel_polygons]
    
    # For large datasets, use optimized distance calculation
    if len(cells_df) > 1000:
        print("Using optimized calculation for large dataset...")
        # Create spatial index for vessels if we have many vessels
        if len(vessel_polygons) > 10:
            vessel_tree = STRtree(vessel_polygons)
            
            for idx, row in tqdm(cells_df.iterrows(), total=len(cells_df), desc="Computing cell-vessel distances"):
                try:
                    cell_point = Point(row['x'], row['y'])
                    
                    # Use spatial index to find candidate vessels
                    candidates = vessel_tree.query(cell_point.buffer(100))  # Look within 100 units
                    if not candidates:
                        # If no candidates found nearby, check all vessels
                        candidates = range(len(vessel_polygons))
                    
                    min_distance = float('inf')
                    nearest_vessel_id = -1
                    
                    for vessel_id in candidates:
                        if vessel_id < len(vessel_polygons):  # Safety check
                            distance = cell_point.distance(vessel_boundaries[vessel_id])
                            if distance < min_distance:
                                min_distance = distance
                                nearest_vessel_id = vessel_id
                    
                    distances.append(min_distance if min_distance != float('inf') else np.nan)
                    nearest_vessel_ids.append(nearest_vessel_id)
                    
                except Exception as e:
                    print(f"Warning: Error calculating distance for cell at row {idx}: {e}")
                    distances.append(np.nan)
                    nearest_vessel_ids.append(-1)
        else:
            # For smaller number of vessels, direct computation is fine
            for idx, row in tqdm(cells_df.iterrows(), total=len(cells_df), desc="Computing cell-vessel distances"):
                try:
                    cell_point = Point(row['x'], row['y'])
                    
                    min_distance = float('inf')
                    nearest_vessel_id = -1
                    
                    for vessel_id, boundary in enumerate(vessel_boundaries):
                        distance = cell_point.distance(boundary)
                        if distance < min_distance:
                            min_distance = distance
                            nearest_vessel_id = vessel_id
                    
                    distances.append(min_distance if min_distance != float('inf') else np.nan)
                    nearest_vessel_ids.append(nearest_vessel_id)
                    
                except Exception as e:
                    print(f"Warning: Error calculating distance for cell at row {idx}: {e}")
                    distances.append(np.nan)
                    nearest_vessel_ids.append(-1)
    else:
        # For smaller datasets, use simpler approach
        for idx, row in tqdm(cells_df.iterrows(), total=len(cells_df), desc="Computing cell-vessel distances"):
            try:
                cell_point = Point(row['x'], row['y'])
                
                min_distance = float('inf')
                nearest_vessel_id = -1
                
                for vessel_id, boundary in enumerate(vessel_boundaries):
                    distance = cell_point.distance(boundary)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_vessel_id = vessel_id
                
                distances.append(min_distance if min_distance != float('inf') else np.nan)
                nearest_vessel_ids.append(nearest_vessel_id)
                
            except Exception as e:
                print(f"Warning: Error calculating distance for cell at row {idx}: {e}")
                distances.append(np.nan)
                nearest_vessel_ids.append(-1)
    
    cells_df['distance_to_vessel'] = distances
    cells_df['nearest_vessel_id'] = nearest_vessel_ids
    
    return cells_df


def create_distance_bins(max_distance=200, bin_size=10):
    """
    Create distance bins for spatial analysis.
    
    Args:
        max_distance (float): Maximum distance to analyze (in μm)
        bin_size (float): Size of each distance bin (in μm)
        
    Returns:
        list: List of (bin_start, bin_end) tuples
    """
    bins = []
    current = 0
    while current < max_distance:
        bins.append((current, current + bin_size))
        current += bin_size
    
    print(f"Created {len(bins)} distance bins from 0 to {max_distance}μm with {bin_size}μm intervals")
    return bins


def bin_cells_by_distance(pos_cells_df, neg_cells_df, distance_bins, vessel_polygons):
    """
    Bin cells by distance from vessel margins for each vessel.
    
    Args:
        pos_cells_df (pd.DataFrame): DAB+ cells with distance calculations
        neg_cells_df (pd.DataFrame): DAB- cells with distance calculations  
        distance_bins (list): List of (bin_start, bin_end) tuples
        vessel_polygons (list): List of vessel polygons
        
    Returns:
        pd.DataFrame: Summary data with vessel info and binned cell counts
    """
    print("Binning cells by distance for each vessel...")
    
    # Initialize results dataframe
    results = []
    
    for vessel_id in range(len(vessel_polygons)):
        row_data = {'vessel_id': vessel_id}
        
        # Filter cells belonging to this vessel
        pos_vessel_cells = pos_cells_df[pos_cells_df['nearest_vessel_id'] == vessel_id]
        neg_vessel_cells = neg_cells_df[neg_cells_df['nearest_vessel_id'] == vessel_id]
        
        # Bin cells by distance
        for bin_start, bin_end in distance_bins:
            # Count cells in this distance bin
            pos_in_bin = pos_vessel_cells[
                (pos_vessel_cells['distance_to_vessel'] >= bin_start) & 
                (pos_vessel_cells['distance_to_vessel'] < bin_end)
            ]
            neg_in_bin = neg_vessel_cells[
                (neg_vessel_cells['distance_to_vessel'] >= bin_start) & 
                (neg_vessel_cells['distance_to_vessel'] < bin_end)
            ]
            
            pos_count = len(pos_in_bin)
            neg_count = len(neg_in_bin)
            total_count = pos_count + neg_count
            
            # Calculate ratio (DAB+ / total)
            ratio = pos_count / total_count if total_count > 0 else 0
            
            # Store in results
            bin_name = f"bin_{int(bin_start)}_{int(bin_end)}"
            row_data[f"{bin_name}_dab_pos"] = pos_count
            row_data[f"{bin_name}_dab_neg"] = neg_count
            row_data[f"{bin_name}_total"] = total_count
            row_data[f"{bin_name}_ratio"] = ratio
        
        results.append(row_data)
    
    results_df = pd.DataFrame(results)
    print(f"Generated binned results for {len(results_df)} vessels")
    
    return results_df


def generate_output_csv(vessel_results_df, vessel_to_litt_distances, output_path):
    """
    Generate final output CSV with vessel metadata and binned cell data.
    
    Args:
        vessel_results_df (pd.DataFrame): Binned cell count data by vessel
        vessel_to_litt_distances (list): Distances from vessels to LITT margins
        output_path (str): Path for output CSV file
    """
    print("Generating output CSV...")
    
    # Add vessel-to-LITT distances
    vessel_results_df['vessel_distance_to_litt'] = vessel_to_litt_distances
    
    # Reorder columns to put metadata first
    metadata_cols = ['vessel_id', 'vessel_distance_to_litt']
    other_cols = [col for col in vessel_results_df.columns if col not in metadata_cols]
    final_cols = metadata_cols + sorted(other_cols)  # Sort distance bin columns
    
    final_df = vessel_results_df[final_cols]
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    final_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    print(f"Output contains {len(final_df)} vessels with {len(final_cols)} columns")
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Vessel-to-LITT distance range: {final_df['vessel_distance_to_litt'].min():.2f} - {final_df['vessel_distance_to_litt'].max():.2f}")
    
    # Calculate total cell counts across all bins
    total_pos_cols = [col for col in final_df.columns if col.endswith('_dab_pos')]
    total_neg_cols = [col for col in final_df.columns if col.endswith('_dab_neg')]
    
    total_pos = final_df[total_pos_cols].sum().sum()
    total_neg = final_df[total_neg_cols].sum().sum()
    total_cells = total_pos + total_neg
    
    print(f"Total DAB+ cells: {total_pos}")
    print(f"Total DAB- cells: {total_neg}")
    print(f"Total cells analyzed: {total_cells}")
    print(f"Overall DAB+ ratio: {total_pos/total_cells:.3f}" if total_cells > 0 else "Overall DAB+ ratio: N/A")


def main():
    """Main function to run the spatial analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Spatial Analysis Script for Cell-Vessel Distance Calculations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python 2-spatial_analysis.py \\
    --vessel_pkl data/vessel_masks.pkl \\
    --litt_pkl data/litt_masks.pkl \\
    --pos_csv data/positive_cells.csv \\
    --neg_csv data/negative_cells.csv \\
    --output results/vessel_cellularity_analysis.csv
        """
    )
    
    parser.add_argument('--vessel_pkl', required=True, 
                       help='Path to vessel polygons pickle file')
    parser.add_argument('--litt_pkl', required=True,
                       help='Path to LITT polygons pickle file')
    parser.add_argument('--pos_csv', required=True,
                       help='Path to DAB+ cell centroids CSV file')
    parser.add_argument('--neg_csv', required=True,
                       help='Path to DAB- cell centroids CSV file')
    parser.add_argument('--output', required=True,
                       help='Path for output CSV file')
    parser.add_argument('--max_distance', type=float, default=200,
                       help='Maximum analysis distance in μm (default: 200)')
    parser.add_argument('--bin_size', type=float, default=10,
                       help='Distance bin size in μm (default: 10)')
    
    args = parser.parse_args()
    
    # Validate input files exist
    for file_path, name in [(args.vessel_pkl, 'vessel_pkl'), 
                           (args.litt_pkl, 'litt_pkl'),
                           (args.pos_csv, 'pos_csv'), 
                           (args.neg_csv, 'neg_csv')]:
        if not os.path.exists(file_path):
            print(f"Error: {name} file not found: {file_path}")
            sys.exit(1)
    
    try:
        # Load spatial data
        vessel_polygons, litt_polygons, pos_cells_df, neg_cells_df = load_spatial_data(
            args.vessel_pkl, args.litt_pkl, args.pos_csv, args.neg_csv
        )
        
        # Calculate vessel-to-LITT distances
        vessel_to_litt_distances = calculate_vessel_to_litt_distances(
            vessel_polygons, litt_polygons
        )
        
        # Calculate cell-to-vessel distances
        pos_cells_df = calculate_cell_to_vessel_distances(pos_cells_df, vessel_polygons)
        neg_cells_df = calculate_cell_to_vessel_distances(neg_cells_df, vessel_polygons)
        
        # Create distance bins
        distance_bins = create_distance_bins(args.max_distance, args.bin_size)
        
        # Bin cells by distance
        vessel_results_df = bin_cells_by_distance(
            pos_cells_df, neg_cells_df, distance_bins, vessel_polygons
        )
        
        # Generate output CSV
        generate_output_csv(vessel_results_df, vessel_to_litt_distances, args.output)
        
        print("\nSpatial analysis completed successfully!")
        
    except Exception as e:
        print(f"Error during spatial analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()