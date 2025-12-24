"""
Perform Maximum Covariance Analysis (MCA) / Singular Value Decomposition (MCA)
between precipitation and RWS at 200 hPa (RWS_200).

This script:
1. Loads 'prec' and 'RWS_200' using the load_era_field function
2. Uses the entire time series (not limited to specific seasons)
3. Performs MCA using an irregular region defined by vertices
4. Computes and saves 4 modes across the entire time series
5. Saves the NetCDF files with the MCA patterns and time series
6. Plots the MCA patterns for each mode
"""

import numpy as np
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
if not all([target_descriptor, generation, individual]):
    print("Error: Required environment variables not set")
    print(f"TARGET_DESCRIPTOR: {target_descriptor}")
    print(f"GA_GENERATION: {generation}")
    print(f"GA_INDIVIDUAL: {individual}")
    print(f"TRIAL: {trial}")
    sys.exit(1)

VERSION = os.environ.get("VERSION", 'v0')
logging.info(f"MCA analysis using VERSION: {VERSION}")

# Set the paths
DATA_DIR = get_data_path('1_deg_seasonal/', data_type="interim")
RESULTS_DIR = get_results_path('MCA_prec_RWS_200/')
# Construct target_dir using trial, target_descriptor, and VERSION
target_dir = os.path.join(RESULTS_DIR, trial, target_descriptor, VERSION)

# Save directory using target directory
SAVE_DIR = os.path.join(target_dir, 'NetCDFs_MCA_modes')
os.makedirs(SAVE_DIR, exist_ok=True)

# Plot directory using target directory
PLOT_DIR = os.path.join(target_dir, 'Plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Define default irregular region vertices
DEFAULT_VERTICES = [
    (-55, -15),
    (-35, -15),
    (-35, -30),
    (-55, -30)
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

####################################
# Load prec and RWS
####################################

prec = load_era_field(
    filepath=os.path.join(DATA_DIR, 'prec_seas_1deg.nc'),
    var_name='tp'
)

RWS_200 = load_era_field(
    filepath=os.path.join(DATA_DIR, 'RWS_200_seas_1deg.nc'),
    var_name='__xarray_dataarray_variable__',
    lat_slice=(-50, 50)
)

# Compute seasonal anomalies
prec_anom = compute_seasonal_anomalies(prec, '1945-06-01', '2024-02-01')
RWS_200_anom = compute_seasonal_anomalies(RWS_200, '1945-06-01', '2024-02-01')

# Use all data (no seasonal filtering)
prec_data = prec_anom
RWS_200_data = RWS_200_anom

print(f"Full time series data shape - prec: {prec_data.shape}, RWS: {RWS_200_data.shape}")

####################################
# MCA Function
####################################

def perform_mca(field1, field2, region_vertices, n_modes=4):
    """
    Perform MCA on two fields over the given domain.

    Args:
        field1 (xarray.DataArray): First field [time, lat, lon]
        field2 (xarray.DataArray): Second field [time, lat, lon]
        region_vertices (list): List of (lon, lat) tuples defining polygon vertices
        n_modes (int, optional): Number of modes to compute and return

    Returns:
        list of tuples: Each tuple contains (field1_pattern, field2_pattern, pc_field1, pc_field2, explained_var)
                       for each mode requested
    """
    # 1) Subset to the specified irregular region
    field1_dom = select_irregular_region(field1, region_vertices)
    field2_dom = select_irregular_region(field2, region_vertices)

    # Print shapes for debugging
    print(f"Field1 domain shape: {field1_dom.shape}")
    print(f"Field2 domain shape: {field2_dom.shape}")
    
    # 2) Standardize data
    field1_dom_std = standardize_seasonal(field1_dom)
    field2_dom_std = standardize_seasonal(field2_dom)

    # 3) Apply latitude weighting
    field1_weighted, weights1 = lat_lon_weighting(field1_dom_std)
    field2_weighted, weights2 = lat_lon_weighting(field2_dom_std)

    # 4) Flatten each field to (time, n_points)
    ntime = field1_weighted.time.size
    nlat1 = field1_weighted.latitude.size
    nlon1 = field1_weighted.longitude.size
    
    nlat2 = field2_weighted.latitude.size
    nlon2 = field2_weighted.longitude.size

    field1_flat = field1_weighted.values.reshape(ntime, nlat1*nlon1)
    field2_flat = field2_weighted.values.reshape(ntime, nlat2*nlon2)

    # 5) Create valid masks for each field.
    #    We'll keep only columns with no NaNs for the entire time dimension.
    valid_field1 = ~np.isnan(field1_flat).any(axis=0)
    valid_field2 = ~np.isnan(field2_flat).any(axis=0)

    # Print number of valid points for debugging
    print(f"Number of valid points in field1: {np.sum(valid_field1)}")
    print(f"Number of valid points in field2: {np.sum(valid_field2)}")
    
    # 6) Subset the arrays
    field1_clean = field1_flat[:, valid_field1]
    field2_clean = field2_flat[:, valid_field2]

    # 7) Compute cross-covariance, then MCA
    # cross-cov matrix (X^T Y) shape => (n_field1_points, n_field2_points)
    C = np.dot(field1_clean.T, field2_clean) / (ntime - 1)
    
    # Handle any NaN values in the covariance matrix
    if np.any(np.isnan(C)):
        print("WARNING: NaN values found in cross-covariance matrix, replacing with zeros")
        C = np.nan_to_num(C)

    try:
        U_s, singular_values, V_s = np.linalg.svd(C, full_matrices=False)
        print(f"Number of singular values: {len(singular_values)}")
        
        # Validate sufficient singular values for requested modes
        if len(singular_values) < n_modes:
            print(f"WARNING: Requested {n_modes} modes, but only {len(singular_values)} are available")
            n_modes = len(singular_values)
            
        explained_variance = (singular_values**2) / np.sum(singular_values**2) * 100
        for i in range(n_modes):
            print(f"Explained variance by mode {i+1}: {explained_variance[i]:.2f}%")

    except Exception as e:
        print(f"MCA ERROR: {str(e)}")
        # Create dummy results
        return None
    
    # 8) Extract results for each mode
    results = []
    
    for mode in range(n_modes):
        # Extract singular vectors for this mode
        field1_pattern_1d = U_s[:, mode]  # shape (#valid_field1_points,)
        field2_pattern_1d = V_s[mode, :]  # shape (#valid_field2_points,)

        # Obtain time series by projecting original data onto these vectors
        pc_field1_arr = np.dot(field1_clean, field1_pattern_1d)
        pc_field2_arr = np.dot(field2_clean, field2_pattern_1d)

        # Reshape the patterns back into spatial fields and unweight
        field1_pattern_full = np.full(nlat1*nlon1, np.nan)
        field1_pattern_full[valid_field1] = field1_pattern_1d
        field1_pattern_2d = field1_pattern_full.reshape(nlat1, nlon1)

        field1_pattern_da = xr.DataArray(
            (field1_pattern_2d / weights1.data) * pc_field1_arr.std(),
            dims=['latitude','longitude'],
            coords={'latitude': field1_weighted.latitude, 'longitude': field1_weighted.longitude}
        )

        field2_pattern_full = np.full(nlat2*nlon2, np.nan)
        field2_pattern_full[valid_field2] = field2_pattern_1d
        field2_pattern_2d = field2_pattern_full.reshape(nlat2, nlon2)

        field2_pattern_da = xr.DataArray(
            (field2_pattern_2d / weights2.data) * pc_field2_arr.std(),
            dims=['latitude','longitude'],
            coords={'latitude': field2_weighted.latitude, 'longitude': field2_weighted.longitude}
        )

        # Convert PCs to DataArray, standardize them
        pc_field1_da = xr.DataArray(pc_field1_arr, dims=['time'], coords={'time': field1_weighted.time})
        pc_field2_da = xr.DataArray(pc_field2_arr, dims=['time'], coords={'time': field2_weighted.time})

        pc_field1_std = standardize_seasonal(pc_field1_da)
        pc_field2_std = standardize_seasonal(pc_field2_da)
        
        # Add to results
        results.append((field1_pattern_da, field2_pattern_da, pc_field1_std, pc_field2_std, explained_variance[mode]))
        
    return results

####################################
# Plotting Function
####################################

def plot_mca_patterns(prec_pattern, RWS_pattern, title, save_path, explained_var, mode_num):
    """
    Plot the MCA patterns for precipitation and RWS
    
    Uses standardized plotting formats from 20_Multivar_regrs_plots_E_v_C_40_24.py
    """
    fig = plt.figure(figsize=(15, 18))
    proj = ccrs.PlateCarree()
    
    # Plot prec pattern
    ax1 = plt.subplot(2, 1, 1, projection=proj)
    ax1.set_extent([prec_pattern.longitude.min(), prec_pattern.longitude.max(),
                         prec_pattern.latitude.min(), prec_pattern.latitude.max()],
                        crs=ccrs.PlateCarree())
    ax1.coastlines(linewidth=0.6)
    ax1.add_feature(cartopy.feature.BORDERS, linewidth=0.6)
    gl1 = ax1.gridlines(draw_labels=True)
    gl1.top_labels = False
    gl1.right_labels = False
    
    # Use get_symmetric_levels
    prec_levels, prec_contour_levels = get_symmetric_levels(prec_pattern, 0.05)
    prec_cmap = 'BrBG'
    
    c1 = prec_pattern.plot.contourf(
        ax=ax1, transform=ccrs.PlateCarree(),
        levels=prec_levels, cmap=prec_cmap,
        add_colorbar=False, extend='both'
    )
    
    # Add contour lines
    try:
        cs1 = ax1.contour(
            prec_pattern.longitude, prec_pattern.latitude, prec_pattern.values,
            transform=ccrs.PlateCarree(), levels=prec_contour_levels,
            colors='darkgray', linewidths=0.5
        )
        plt.clabel(
            cs1, prec_contour_levels, fmt='%1.1f', 
            fontsize='x-small', colors='dimgrey', 
            inline=True, inline_spacing=4
        )
    except ValueError as e:
        print(f"Could not draw contours for prec: {e}")
    
    # Add compact colorbar
    cax1 = fig.add_axes([0.3, 0.51, 0.4, 0.01])  # [left, bottom, width, height]
    cb1 = plt.colorbar(c1, cax=cax1, orientation='horizontal', label='prec Pattern [Covariation Units]')
    
    ax1.set_title(f'prec Pattern (Mode {mode_num})', fontsize=16)
    
    # Plot RWS pattern
    ax2 = plt.subplot(2, 1, 2, projection=proj)
    ax2.set_extent([RWS_pattern.longitude.min(), RWS_pattern.longitude.max(),
                         RWS_pattern.latitude.min(), RWS_pattern.latitude.max()],
                        crs=ccrs.PlateCarree())
    ax2.coastlines(linewidth=0.6)
    ax2.add_feature(cartopy.feature.BORDERS, linewidth=0.6)
    gl2 = ax2.gridlines(draw_labels=True)
    gl2.top_labels = False
    gl2.right_labels = False
    
    # Use get_symmetric_levels with conversion factor
    RWS_levels, RWS_contour_levels = get_symmetric_levels(RWS_pattern, 0.05)
    RWS_cmap = 'coolwarm'
    
    c2 = RWS_pattern.plot.contourf(
        ax=ax2, transform=ccrs.PlateCarree(),
        levels=RWS_levels, cmap=RWS_cmap,
        add_colorbar=False, extend='both'
    )
    
    # Add contour lines
    try:
        cs2 = ax2.contour(
            RWS_pattern.longitude, RWS_pattern.latitude, RWS_pattern.values,
            transform=ccrs.PlateCarree(), levels=RWS_contour_levels,
            colors='darkgray', linewidths=0.5
        )
        plt.clabel(
            cs2, RWS_contour_levels, fmt='%1.1f', 
            fontsize='x-small', colors='dimgrey', 
            inline=True, inline_spacing=4
        )
    except ValueError as e:
        print(f"Could not draw contours for RWS: {e}")
    
    # Add compact colorbar
    cax2 = fig.add_axes([0.3, 0.07, 0.4, 0.01])  # [left, bottom, width, height]
    cb2 = plt.colorbar(c2, cax=cax2, orientation='horizontal', label='RWS Pattern [Covariation Units]')
    
    ax2.set_title(f'RWS Pattern at 200 hPa (Mode {mode_num})', fontsize=16)
    
    # Add super title with explained variance
    plt.suptitle(f"{title}\nExplained Variance: {explained_var:.1f}%", fontsize=18, y=0.98)
    
    plt.subplots_adjust(left=0.04, bottom=0.1, right=0.96, top=0.92, hspace=0.4)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

####################################
# Perform MCA for full time series
####################################

# Perform MCA for full time series
print("\n====== Performing MCA for full time series ======")
print(f"Using irregular region with {len(vertices)} vertices")
mca_results = perform_mca(
    prec_data, RWS_200_data, 
    region_vertices=vertices, 
    n_modes=2
)

# Check if MCA was successful
if mca_results is None:
    print("Error performing MCA. Exiting.")
    sys.exit(1)

####################################
# Save Results
####################################

# Create datasets and save results for each mode
for mode_idx, (prec_pattern, RWS_pattern, pc_prec, pc_RWS_200, explained_var) in enumerate(mca_results):
    mode_num = mode_idx + 1
    
    # Create a dataset for the patterns
    patterns_ds = xr.Dataset({
        'prec_pattern': prec_pattern,
        'RWS_pattern': RWS_pattern
    })

    # Create separate datasets for the time series of each variable
    ts_prec_ds = xr.Dataset({
        'pc_prec': pc_prec,
        'explained_variance': xr.DataArray(explained_var, dims=[])
    })
    
    ts_RWS_200_ds = xr.Dataset({
        'pc_RWS_200': pc_RWS_200,
        'explained_variance': xr.DataArray(explained_var, dims=[])
    })

    # Save results to NetCDF with generation and individual info
    patterns_ds.to_netcdf(os.path.join(SAVE_DIR, f'MCA_prec_RWS_200_patterns_mode{mode_num}_gen{generation}_ind{individual}.nc'))
    ts_prec_ds.to_netcdf(os.path.join(SAVE_DIR, f'MCA_prec_RWS_200_timeseries_prec_mode{mode_num}_gen{generation}_ind{individual}.nc'))
    ts_RWS_200_ds.to_netcdf(os.path.join(SAVE_DIR, f'MCA_prec_RWS_200_timeseries_RWS_200_mode{mode_num}_gen{generation}_ind{individual}.nc'))

    # Plot the patterns
    plot_mca_patterns(
        prec_pattern, RWS_pattern, 
        f'MCA between prec and RWS (200 hPa)', 
        os.path.join(PLOT_DIR, f'MCA_prec_RWS_200_patterns_mode{mode_num}_gen{generation}_ind{individual}.png'),
        explained_var,
        mode_num
    )

print(f"\nMCA analysis completed successfully:")
for mode_idx, (_, _, _, _, explained_var) in enumerate(mca_results):
    print(f"Mode {mode_idx+1} explained variance: {explained_var:.1f}%")
print(f"Results saved in {SAVE_DIR}")
print(f"Plots saved in {PLOT_DIR}")