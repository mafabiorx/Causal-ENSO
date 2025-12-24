"""
Perform Rotated EOF (REOF) for SST

This script:
1. Loads 'SST' using the load_era_field function
2. Uses the entire time series (not limited to specific seasons)
3. Performs REOF using an irregular region defined by vertices
4. Computes and saves 4 modes across the entire time series
5. Saves the NetCDF files with the REOF patterns and time series
6. Plots the REOF patterns for each mode
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional
from sklearn.decomposition import PCA
from statsmodels.multivariate.factor_rotation import rotate_factors
import xarray as xr
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import warnings
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(filename)s - %(message)s')

# Ensure the project root is in sys.path so that the module can be found
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.TBI_functions import (
    compute_seasonal_anomalies,
    standardize_seasonal,
    lat_lon_weighting,
    get_symmetric_levels,
    load_era_field,
    select_irregular_region
)
from utils.paths import get_data_path, get_results_path

# Filter out the specific RuntimeWarning about degrees of freedom
warnings.filterwarnings(
    "ignore",
    message="Degrees of freedom <= 0 for slice",
    category=RuntimeWarning
)

# Get environment variables for file paths
target_descriptor = os.environ.get("TARGET_DESCRIPTOR")
generation = os.environ.get("GA_GENERATION")
individual = os.environ.get("GA_INDIVIDUAL")
trial = os.environ.get("TRIAL")

# Ensure environment variables are set
if not all([target_descriptor, generation, individual, trial]):
    print("Error: Required environment variables not set")
    print(f"TARGET_DESCRIPTOR: {target_descriptor}")
    print(f"GA_GENERATION: {generation}")
    print(f"GA_INDIVIDUAL: {individual}")
    print(f"TRIAL: {trial}")
    sys.exit(1)

VERSION = os.environ.get("VERSION", 'v0')
logging.info(f"REOF analysis using VERSION: {VERSION}")

# Set the paths
DATA_DIR = get_data_path('1_deg_seasonal/', data_type="interim")
RESULTS_DIR = get_results_path('REOF_SST/')
# Construct target_dir using trial, target_descriptor, and VERSION
target_dir = os.path.join(RESULTS_DIR, trial, target_descriptor, VERSION)

# Save directory using target directory
SAVE_DIR = os.path.join(target_dir, 'NetCDFs_REOF_modes')
os.makedirs(SAVE_DIR, exist_ok=True)

# Plot directory using target directory
PLOT_DIR = os.path.join(target_dir, 'Plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Define default irregular region vertices
DEFAULT_VERTICES = [
    (-50, -10),
    (5, -10),
    (5, -25),
    (-40, -25)
]

# Build vertices from env-vars if they exist, else fall back to default box
lat_min = os.getenv("RECT_LAT_MIN")
lat_max = os.getenv("RECT_LAT_MAX")
lon_min = os.getenv("RECT_LON_MIN")
lon_max = os.getenv("RECT_LON_MAX")

if all([lat_min, lat_max, lon_min, lon_max]):
    try:
        lat_min, lat_max = int(lat_min), int(lat_max)
        lon_min, lon_max = int(lon_min), int(lon_max)
        vertices = [
            (lon_min, lat_max),  # NW
            (lon_max, lat_max),  # NE
            (lon_max, lat_min),  # SE
            (lon_min, lat_min),  # SW
        ]
        logging.info(f"Using vertices from env-vars: {vertices}")
    except ValueError:
        logging.error("RECT_* env-vars are not integers; falling back to default vertices.")
        vertices = DEFAULT_VERTICES
else:
    vertices = DEFAULT_VERTICES

####
# REOF Function
####

def perform_reof(field_data: xr.DataArray,
                 region_vertices: List[Tuple[float, float]],
                 n_components: int = 7,
                 n_modes_to_return: int = 4) -> Optional[List[Tuple[xr.DataArray, xr.DataArray, float]]]:
    """
    Perform Rotated EOF (REOF) analysis on a given field over a specified domain.

    Args:
        field_data (xr.DataArray): Input climate field [time, lat, lon].
        region_vertices (list): List of (lon, lat) tuples defining polygon vertices.
        n_components (int, optional): Number of components for initial PCA. Defaults to 7.
                                      Should be >= n_modes_to_return for stable rotation.
        n_modes_to_return (int, optional): Number of rotated modes to return. Defaults to 4.

    Returns:
        Optional[List[Tuple[xr.DataArray, xr.DataArray, float]]]:
            A list where each tuple contains:
            - Rotated EOF spatial pattern (xr.DataArray, physical units)
            - Standardized Rotated PC time series (xr.DataArray)
            - Explained variance percentage for the mode (float)
            Returns None if the analysis fails.
    """
    try:
        # 1) Subset to the specified irregular region
        field_dom = select_irregular_region(field_data, region_vertices)
        print(f"Field domain shape after region selection: {field_dom.shape}")

        # 2) Apply latitude weighting
        field_weighted, weights = lat_lon_weighting(field_dom)
        print(f"Field shape after weighting: {field_weighted.shape}")

        # 3) Reshape to (time, n_points) and handle NaNs
        ntime = field_weighted.time.size
        nlat = field_weighted.latitude.size
        nlon = field_weighted.longitude.size
        original_shape_2d = (nlat, nlon)

        field_flat = field_weighted.values.reshape(ntime, -1)

        # Create a mask for valid (non-NaN) grid points based on the first time step
        # Assuming NaN mask is consistent across time after region selection
        nan_mask_1d = np.isnan(field_flat[0, :])
        valid_cols = ~nan_mask_1d
        field_clean = field_flat[:, valid_cols]

        print(f"Cleaned data shape (time, valid_points): {field_clean.shape}")
        if field_clean.shape[1] == 0:
            print("Error: No valid data points found after masking.")
            return None

        # 4) Perform PCA
        pca = PCA(n_components=n_components)
        # pcs_raw shape: (ntime, n_components)
        # eofs_raw shape: (n_components, n_valid_points)
        pcs_raw = pca.fit_transform(field_clean)
        eofs_raw = pca.components_
        # explained_variance_pca = pca.explained_variance_ratio_ * 100
        # eigenvalues_pca = pca.explained_variance_

        print(f"PCA completed. Raw PCs shape: {pcs_raw.shape}, Raw EOFs shape: {eofs_raw.shape}")

        # 5) Perform Varimax Rotation
        # Rotate the loading matrix (transpose EOFs to shape (n_valid_points, n_components))
        loadings = eofs_raw.T
        # rotated_loadings shape: (n_valid_points, n_components)
        # rotation_matrix shape: (n_components, n_components)
        rotated_loadings, rotation_matrix = rotate_factors(loadings, method='varimax')

        # Get rotated EOFs: transpose back to (n_components, n_valid_points)
        rotated_eofs = rotated_loadings.T
        # Rotate the PC time series using the rotation matrix
        # rotated_pcs shape: (ntime, n_components)
        rotated_pcs = np.dot(pcs_raw, rotation_matrix)

        print(f"Varimax rotation completed. Rotated PCs shape: {rotated_pcs.shape}, Rotated EOFs shape: {rotated_eofs.shape}")

        # 6) Calculate Rotated Explained Variance
        # Variance of each rotated PC time series
        rotated_pcs_variance = np.var(rotated_pcs, axis=0, ddof=1) # Use ddof=1 for sample variance
        total_variance = np.sum(rotated_pcs_variance)
        rotated_explained_variance = (rotated_pcs_variance / total_variance) * 100.0

        # Print explained variance for all rotated components
        for i in range(n_components):
             print(f"Explained variance by rotated mode {i+1}: {rotated_explained_variance[i]:.2f}%")

        # Validate sufficient modes for requested output
        if n_components < n_modes_to_return:
            print(f"Warning: Requested {n_modes_to_return} modes, but only {n_components} were computed.")
            n_modes_to_return = n_components

        # 7) Reconstruct spatial EOFs, prepare PC time series, and collect results
        results = []
        for mode_idx in range(n_modes_to_return):
            # --- Reconstruct Spatial EOF Pattern ---
            # Get the 1D rotated EOF for this mode
            eof_1d = rotated_eofs[mode_idx, :] # shape: (n_valid_points,)

            # Create a full spatial grid initialized with NaNs
            eof_spatial_full = np.full(nlat * nlon, np.nan)
            # Fill in the valid grid points with the EOF values
            eof_spatial_full[valid_cols] = eof_1d
            # Reshape back to 2D (lat, lon)
            eof_spatial_2d = eof_spatial_full.reshape(original_shape_2d)

            # Get the corresponding rotated PC time series
            pc_ts = rotated_pcs[:, mode_idx] # shape: (ntime,)

            # Un-weight the spatial pattern by dividing by latitude weights
            # Need weights reshaped to 2D to match eof_spatial_2d
            weights_2d = weights.values.reshape(original_shape_2d)
            eof_unweighted = eof_spatial_2d / weights_2d

            # Scale the pattern by the standard deviation of the corresponding PC time series
            # This converts EOFs to physical units (units of original data * std dev of PC)
            # Use unstandardized PC standard deviation to preserve physical amplitude scaling
            pc_std_dev = np.std(pc_ts, ddof=1)
            eof_physical_units = eof_unweighted * pc_std_dev

            # Convert the final spatial pattern to an xarray DataArray
            eof_pattern_da = xr.DataArray(
                eof_physical_units,
                coords=[field_weighted.latitude, field_weighted.longitude],
                dims=['latitude', 'longitude'],
                name=f'eof_mode_{mode_idx+1}'
            )

            # --- Prepare PC Time Series ---
            # Convert the rotated PC time series to an xarray DataArray
            pc_da = xr.DataArray(
                pc_ts,
                coords={'time': field_weighted.time},
                dims=['time'],
                name=f'pc_mode_{mode_idx+1}'
            )

            # Standardize the PC time series (mean=0, std=1)
            pc_std_da = standardize_seasonal(pc_da)

            # --- Store Results ---
            results.append((eof_pattern_da, pc_std_da, rotated_explained_variance[mode_idx]))

        return results

    except Exception as e:
        print(f"Error during REOF analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

####
# Plotting Function
####

def plot_reof_pattern(eof_pattern: xr.DataArray,
                      explained_var: float,
                      mode_num: int,
                      title: str,
                      save_path: str,
                      variable_units: str = 'SST Anomaly [K * std dev]'):
    """
    Plot a single REOF spatial pattern.

    Args:
        eof_pattern (xr.DataArray): The spatial pattern to plot.
        explained_var (float): Explained variance percentage for this mode.
        mode_num (int): The mode number (e.g., 1, 2, 3, 4).
        title (str): Base title for the plot.
        save_path (str): Full path to save the output image.
        variable_units (str): Units label for the colorbar.
    """
    fig = plt.figure(figsize=(12, 6)) # Adjusted size for single plot
    proj = ccrs.PlateCarree()
    ax = plt.subplot(1, 1, 1, projection=proj)

    # Determine plot extent from data
    lon_min, lon_max = eof_pattern.longitude.min(), eof_pattern.longitude.max()
    lat_min, lat_max = eof_pattern.latitude.min(), eof_pattern.latitude.max()
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # Add map features
    ax.coastlines(linewidth=0.8, color='black')
    ax.add_feature(cartopy.feature.BORDERS, linewidth=0.6, linestyle=':')
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}

    # Use get_symmetric_levels for contour levels
    # Adjust spacing based on typical SST anomaly magnitudes if needed
    spacing = np.nanmax(np.abs(eof_pattern.values)) / 10.0 # Heuristic for spacing
    if spacing == 0: spacing = 0.1 # Avoid division by zero if pattern is flat
    levels, contour_levels = get_symmetric_levels(eof_pattern, spacing=spacing)
    cmap = 'coolwarm'

    # Plot filled contours
    cf = ax.contourf(
        eof_pattern.longitude, eof_pattern.latitude, eof_pattern.values,
        transform=ccrs.PlateCarree(),
        levels=levels, cmap=cmap,
        extend='both'
    )

    # Add contour lines for detail
    try:
        cs = ax.contour(
            eof_pattern.longitude, eof_pattern.latitude, eof_pattern.values,
            transform=ccrs.PlateCarree(), levels=contour_levels,
            colors='darkgray', linewidths=0.7
        )
        # Add labels to contour lines
        plt.clabel(
            cs, contour_levels, fmt='%1.2f', # Format for potentially smaller values
            fontsize='x-small', colors='dimgrey',
            inline=True, inline_spacing=4
        )
    except ValueError as e:
        print(f"Could not draw contours for mode {mode_num}: {e}")

    # Add colorbar
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', shrink=0.7, pad=0.1, label=variable_units)
    cbar.ax.tick_params(labelsize=10)

    # Set title
    plot_title = f'{title} - Mode {mode_num}\nExplained Variance: {explained_var:.1f}%'
    ax.set_title(plot_title, fontsize=14, pad=15)

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    print(f"Plot saved to: {save_path}")
    plt.close(fig) # Close the figure to free memory

####
# Load Data
####

print("Loading SST data...")
SST = load_era_field(
    filepath=os.path.join(DATA_DIR, 'SST_seas_1deg.nc'),
    var_name='sst',
)
print(f"SST data loaded. Shape: {SST.shape}, Units: K")

# Compute seasonal anomalies using the specified date range from original script
print("Computing seasonal anomalies...")
SST_anom = compute_seasonal_anomalies(SST, '1940-09-01', '2024-02-01') # Use YYYY-MM-DD for clarity
SST_data = SST_anom
print(f"Anomaly calculation complete. Shape: {SST_data.shape}")

####
# Perform REOF Analysis
####

print("\n==== Performing REOF analysis for SST ====")
print(f"Using irregular region with {len(vertices)} vertices: {vertices}")
# Perform REOF analysis, requesting 7 components for stability, returning top 4
reof_results = perform_reof(
    field_data=SST_data,
    region_vertices=vertices,
    n_components=7,       # Number of components for initial PCA
    n_modes_to_return=4   # Number of rotated modes to save and plot
)

# Check if REOF was successful
if reof_results is None:
    print("Error performing REOF analysis. Exiting.")
    sys.exit(1)

####
# Save Results and Plot Patterns
####

print("\n==== Saving REOF results and plotting patterns ====")
# Loop through the returned modes
for mode_idx, (eof_pattern, pc_std_series, explained_var) in enumerate(reof_results):
    mode_num = mode_idx + 1 # Mode number (1-based)

    # --- Save EOF Pattern ---
    # Create dataset for the EOF pattern
    eof_ds = xr.Dataset({
        'eof_pattern': eof_pattern,
        'explained_variance': xr.DataArray(explained_var, dims=[], attrs={'units': '%'})
    })
    eof_ds['eof_pattern'].attrs['long_name'] = f'Rotated EOF pattern mode {mode_num} for SST'
    eof_ds['eof_pattern'].attrs['units'] = 'K * std dev' # Reflects scaling

    # Define EOF output filename
    eof_filename = os.path.join(SAVE_DIR, f'REOF_SST_pattern_mode{mode_num}_gen{generation}_ind{individual}.nc')
    eof_ds.to_netcdf(eof_filename)
    print(f"Saved EOF pattern for mode {mode_num} to: {eof_filename}")

    # --- Save PC Time Series ---
    # Create dataset for the standardized PC time series
    pc_var_name = f'PC{mode_num}_SST'
    pc_ds = xr.Dataset({
        pc_var_name: pc_std_series,
        'explained_variance': xr.DataArray(explained_var, dims=[], attrs={'units': '%'})
    })
    pc_ds[pc_var_name].attrs['long_name'] = f'Standardized Rotated PC time series mode {mode_num} for SST'
    pc_ds[pc_var_name].attrs['units'] = 'unitless (standardized)'

    # Define PC output filename
    pc_filename = os.path.join(SAVE_DIR, f'REOF_SST_timeseries_mode{mode_num}_gen{generation}_ind{individual}.nc')
    pc_ds.to_netcdf(pc_filename)
    print(f"Saved PC time series for mode {mode_num} to: {pc_filename}")

    # --- Plot EOF Pattern ---
    # Define plot filename
    plot_filename = os.path.join(PLOT_DIR, f'REOF_SST_pattern_mode{mode_num}_gen{generation}_ind{individual}.png')

    # Call the plotting function
    plot_reof_pattern(
        eof_pattern=eof_pattern,
        explained_var=explained_var,
        mode_num=mode_num,
        title='REOF SST Pattern',
        save_path=plot_filename,
        variable_units='SST Anomaly [K * std dev]' # Units after scaling
    )

print(f"\nREOF analysis completed successfully for {len(reof_results)} modes.")
print(f"Output NetCDF files saved in: {SAVE_DIR}")
print(f"Output plots saved in: {PLOT_DIR}")

# Indicate successful completion for the workflow
sys.exit(0)