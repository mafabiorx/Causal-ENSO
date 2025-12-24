"""
Perform Maximum Covariance Analysis (MCA) / Singular Value Decomposition (SVD)
between Rossby Wave Source (RWS) RWS and Wave Activity Flux (WAF) vectors at 200 hPa (WAFx_200, WAFy_200).

This script performs the following steps:
1. Loads 'RWS', 'WAFx_200', 'WAFy_200' using load_era_field
2. Uses the entire time series (no specific season filtering here)
3. Performs SVD using an irregular region defined by vertices (if specified via env vars)
4. Computes and saves 4 modes across the entire time series
5. Saves the NetCDF files with the MCA patterns and time series
6. Plots the MCA patterns for each mode (RWS shading, WAF vectors)
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
RESULTS_DIR = get_results_path('MCA_RWS_WAF/')
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
    (-40, -15),
    (-25, -15),
    (-25, -35),
    (-40, -35)
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
# Load RWS and WAF
####################################

RWS = load_era_field(
    filepath=os.path.join(DATA_DIR, 'RWS_200_seas_1deg.nc'),
    var_name='__xarray_dataarray_variable__'
)

WAFx_200 = load_era_field(
    filepath=os.path.join(DATA_DIR, 'WAF_200_components_1deg.nc'),
    var_name='WAFx',
    lat_slice=(-50, 50)
)

WAFy_200 = load_era_field(
    filepath=os.path.join(DATA_DIR, 'WAF_200_components_1deg.nc'),
    var_name='WAFy',
    lat_slice=(-50, 50)
)

# Compute seasonal anomalies
RWS_anom = compute_seasonal_anomalies(RWS, '1945-06-01', '2024-02-01')
WAFx_200_anom = compute_seasonal_anomalies(WAFx_200, '1945-06-01', '2024-02-01')
WAFy_200_anom = compute_seasonal_anomalies(WAFy_200, '1945-06-01', '2024-02-01')

# Use all data (no seasonal filtering)
RWS_data = RWS_anom
WAFx_data = WAFx_200_anom
WAFy_data = WAFy_200_anom

print(f"Full time series data shape - RWS: {RWS_data.shape}, WAFx: {WAFx_data.shape}, WAFy: {WAFy_data.shape}")

####################################
# MCA Function
####################################

def perform_mca_scalar_vector(field1, field2x, field2y, region_vertices, n_modes=4):
    """
    Perform MCA between a scalar field (field1) and a vector field (field2x, field2y).
    Handles the vector field by concatenating its components.

    Args:
        field1 (xarray.DataArray): First scalar field [time, lat, lon]
        field2x (xarray.DataArray): X-component of the second vector field [time, lat, lon]
        field2y (xarray.DataArray): Y-component of the second vector field [time, lat, lon]
        region_vertices (list): List of (lon, lat) tuples defining polygon vertices
        n_modes (int, optional): Number of modes to compute and return

    Returns:
        list of tuples: Each tuple contains (pattern_field1, pattern_field2x, pattern_field2y, pc_field1, pc_field2, explained_var)
                       for each mode requested
    """
    # 1) Subset fields to the specified irregular region
    field1_dom = select_irregular_region(field1, region_vertices)
    field2x_dom = select_irregular_region(field2x, region_vertices)
    field2y_dom = select_irregular_region(field2y, region_vertices)

    # Align again after region selection, just in case grid points differ slightly
    field1_dom, field2x_dom, field2y_dom = xr.align(field1_dom, field2x_dom, field2y_dom, join='inner')

    print(f"Field1 domain shape: {field1_dom.shape}")
    print(f"Field2x domain shape: {field2x_dom.shape}")
    print(f"Field2y domain shape: {field2y_dom.shape}")

    # 2) Standardize data (seasonally)
    field1_dom_std = standardize_seasonal(field1_dom)
    field2x_dom_std = standardize_seasonal(field2x_dom)
    field2y_dom_std = standardize_seasonal(field2y_dom)

    # 3) Apply latitude weighting
    field1_weighted, weights1 = lat_lon_weighting(field1_dom_std)
    field2x_weighted, weights2x = lat_lon_weighting(field2x_dom_std)
    field2y_weighted, weights2y = lat_lon_weighting(field2y_dom_std) # Assumes weights are same for x/y

    # 4) Flatten fields to (time, n_points)
    ntime = field1_weighted.time.size
    nlat1 = field1_weighted.latitude.size
    nlon1 = field1_weighted.longitude.size
    npoints1 = nlat1 * nlon1

    nlat2 = field2x_weighted.latitude.size # Assume same grid for x/y
    nlon2 = field2x_weighted.longitude.size
    npoints2 = nlat2 * nlon2

    field1_flat = field1_weighted.values.reshape(ntime, npoints1)
    field2x_flat = field2x_weighted.values.reshape(ntime, npoints2)
    field2y_flat = field2y_weighted.values.reshape(ntime, npoints2)

    # Concatenate vector components
    field2_vector_flat = np.concatenate((field2x_flat, field2y_flat), axis=1) # Shape: (ntime, 2 * npoints2)

    # 5) Create valid masks
    valid_field1 = ~np.isnan(field1_flat).any(axis=0)
    # Mask for vector field considers points valid only if *both* components are valid
    valid_field2x = ~np.isnan(field2x_flat).any(axis=0)
    valid_field2y = ~np.isnan(field2y_flat).any(axis=0)
    valid_field2_combined = np.concatenate((valid_field2x, valid_field2y)) # Mask for the concatenated vector

    print(f"Number of valid points in field1: {np.sum(valid_field1)}")
    print(f"Number of valid points in field2 (vector): {np.sum(valid_field2x & valid_field2y)} (considering pairs)")

    # 6) Subset the arrays
    field1_clean = field1_flat[:, valid_field1]
    field2_vector_clean = field2_vector_flat[:, valid_field2_combined]

    # 7) Compute cross-covariance, then SVD
    C = np.dot(field1_clean.T, field2_vector_clean) / (ntime - 1)

    if np.any(np.isnan(C)):
        print("WARNING: NaN values found in cross-covariance matrix, replacing with zeros")
        C = np.nan_to_num(C)

    try:
        U_s, singular_values, V_s = np.linalg.svd(C, full_matrices=False)
        print(f"Number of singular values: {len(singular_values)}")

        if len(singular_values) < n_modes:
            print(f"WARNING: Requested {n_modes} modes, but only {len(singular_values)} are available.")
            n_modes = len(singular_values)

        explained_variance = (singular_values**2) / np.sum(singular_values**2) * 100
        for i in range(n_modes):
            print(f"Explained variance by mode {i+1}: {explained_variance[i]:.2f}%")

    except np.linalg.LinAlgError as e:
        print(f"SVD ERROR: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected ERROR during SVD: {str(e)}")
        return None

    # 8) Extract results for each mode
    results = []
    n_valid_points2 = field2_vector_clean.shape[1] # Number of valid points in concatenated vector

    for mode in range(n_modes):
        # Extract singular vectors
        field1_pattern_1d = U_s[:, mode] # shape (#valid_field1_points,)
        field2_vector_pattern_1d = V_s[mode, :] # shape (#valid_combined_field2_points,)

        # Obtain time series by projection
        pc_field1_arr = np.dot(field1_clean, field1_pattern_1d)
        pc_field2_arr = np.dot(field2_vector_clean, field2_vector_pattern_1d)

        # --- Reshape pattern for field 1 (scalar) ---
        field1_pattern_full = np.full(npoints1, np.nan)
        field1_pattern_full[valid_field1] = field1_pattern_1d
        field1_pattern_2d = field1_pattern_full.reshape(nlat1, nlon1)
        # Unweight and scale
        pattern_field1_da = xr.DataArray(
            (field1_pattern_2d / weights1.data) * pc_field1_arr.std(),
            dims=['latitude', 'longitude'],
            coords={'latitude': field1_weighted.latitude, 'longitude': field1_weighted.longitude}
        )

        # --- Reshape pattern for field 2 (vector) ---
        # Determine number of valid points for X and Y components
        n_valid_points2x = np.sum(valid_field2x)
        n_valid_points2y = np.sum(valid_field2y) # Should be same as n_valid_points2x if grid is identical

        # Check consistency
        if len(field2_vector_pattern_1d) != (n_valid_points2x + n_valid_points2y):
             print(f"WARNING: Length mismatch in singular vector splitting. "
                   f"Vector length: {len(field2_vector_pattern_1d)}, "
                   f"Valid X points: {n_valid_points2x}, Valid Y points: {n_valid_points2y}")
             # Handle error or return None, depending on desired robustness
             # Proceed with analysis despite potential pattern length mismatch

        # Split the combined pattern vector based on the number of valid points
        pattern_field2x_1d_valid = field2_vector_pattern_1d[0 : n_valid_points2x]
        pattern_field2y_1d_valid = field2_vector_pattern_1d[n_valid_points2x : n_valid_points2x + n_valid_points2y] # Ensure correct slicing end

        # Reshape X component onto the original grid using the valid_field2x mask
        pattern_field2x_full = np.full(npoints2, np.nan) # npoints2 = nlat2 * nlon2
        pattern_field2x_full[valid_field2x] = pattern_field2x_1d_valid
        pattern_field2x_2d = pattern_field2x_full.reshape(nlat2, nlon2)

        # Unweight and scale X component
        pattern_field2x_da = xr.DataArray(
            (pattern_field2x_2d / weights2x.data) * pc_field2_arr.std(), # Scale by combined PC std dev
            dims=['latitude', 'longitude'],
            coords={'latitude': field2x_weighted.latitude, 'longitude': field2x_weighted.longitude}
        )

        # Reshape Y component onto the original grid using the valid_field2y mask
        pattern_field2y_full = np.full(npoints2, np.nan)
        pattern_field2y_full[valid_field2y] = pattern_field2y_1d_valid
        pattern_field2y_2d = pattern_field2y_full.reshape(nlat2, nlon2)

        # Unweight and scale Y component
        pattern_field2y_da = xr.DataArray(
            (pattern_field2y_2d / weights2y.data) * pc_field2_arr.std(), # Scale by combined PC std dev
            dims=['latitude', 'longitude'],
            coords={'latitude': field2y_weighted.latitude, 'longitude': field2y_weighted.longitude}
        )
        # --- End of Reshape pattern for field 2 ---

        # Convert PCs to DataArray and standardize seasonally
        pc_field1_da = xr.DataArray(pc_field1_arr, dims=['time'], coords={'time': field1_weighted.time})
        # PC for field 2 represents the combined vector projection
        pc_field2_da = xr.DataArray(pc_field2_arr, dims=['time'], coords={'time': field1_weighted.time}) # Use field1 time coord

        pc_field1_std = standardize_seasonal(pc_field1_da)
        pc_field2_std = standardize_seasonal(pc_field2_da) # PC of the vector field

        results.append((pattern_field1_da, pattern_field2x_da, pattern_field2y_da, pc_field1_std, pc_field2_std, explained_variance[mode]))

    return results

####################################
# Plotting Function
####################################

def plot_mca_patterns_RWS_waf(WAF_pattern, WAFx_pattern, WAFy_pattern, title, save_path, explained_var, mode_num):
    """
    Plot the MCA patterns for RWS (shading) and WAF vectors (quiver)
    with corrected labels for physical units.
    """
    fig = plt.figure(figsize=(15, 11)) # Adjusted size for single panel focus
    proj = ccrs.PlateCarree()

    ax = plt.subplot(1, 1, 1, projection=proj)

    # Determine extent from WAF patterns (usually larger extent)
    min_lon = WAFx_pattern.longitude.min().item()
    max_lon = WAFx_pattern.longitude.max().item()
    min_lat = WAFx_pattern.latitude.min().item()
    max_lat = WAFx_pattern.latitude.max().item()
    ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

    ax.coastlines(linewidth=0.6)
    ax.add_feature(cartopy.feature.BORDERS, linewidth=0.6)
    gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # Use get_symmetric_levels
    RWS_levels, RWS_contour_levels = get_symmetric_levels(WAF_pattern, 0.05)
    RWS_cmap = 'coolwarm'  # Color map for RWS pattern
    
    c1 = WAF_pattern.plot.contourf(
        ax=ax, transform=ccrs.PlateCarree(),
        levels=RWS_levels, cmap=RWS_cmap,
        add_colorbar=False, extend='both'
    )
    
    # Add contour lines
    try:
        cs1 = ax.contour(
            WAF_pattern.longitude, WAF_pattern.latitude, WAF_pattern.values,
            transform=ccrs.PlateCarree(), levels=RWS_contour_levels,
            colors='darkgray', linewidths=0.5
        )
        plt.clabel(
            cs1, RWS_contour_levels, fmt='%1.1f', 
            fontsize='x-small', colors='dimgrey', 
            inline=True, inline_spacing=4
        )
    except ValueError as e:
        print(f"Could not draw contours for RWS: {e}")
    
    # Add compact colorbar
    cax = fig.add_axes([0.3, 0.05, 0.4, 0.01])  # [left, bottom, width, height]
    cb1 = plt.colorbar(c1, cax=cax, orientation='horizontal', label='RWS Pattern [mm/day per PC std dev]')
    
    # Plot WAF vectors (quiver)
    subsample = 3
    lon = WAFx_pattern['longitude'][::subsample].values
    lat = WAFx_pattern['latitude'][::subsample].values
    u = WAFx_pattern.sel(longitude=lon, latitude=lat).values
    v = WAFy_pattern.sel(longitude=lon, latitude=lat).values

    # Normalize vectors for consistent arrow length based on magnitude
    magnitude = np.sqrt(u**2 + v**2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Use nanquantile to find percentiles, handling NaN values appropriately
        m05 = np.nanquantile(magnitude, 0.05) if not np.all(np.isnan(magnitude)) else 0
        m95 = np.nanquantile(magnitude, 0.95) if not np.all(np.isnan(magnitude)) else 1
    
    # Define scaling parameters
    min_len = 0.6
    max_len = 1.4
    if m95 > m05:  # Avoid division by zero
        slope = (max_len - min_len) / (m95 - m05)
        intercept = max_len - slope * m95
              
        # Calculate new magnitude
        new_magn = magnitude * slope + intercept
        
        # Calculate ratio for scaling
        ratio = new_magn / magnitude
        ratio = np.nan_to_num(ratio, nan=0.0)
        
        # Apply scaling
        u = u * ratio
        v = v * ratio

    ref_magnitude = np.nanpercentile(magnitude, 95) # Use 95th percentile as reference scale
    if ref_magnitude == 0: ref_magnitude = 1.0 # Avoid division by zero
    scale_factor = 30 * ref_magnitude

    q = ax.quiver(lon, lat, u, v, transform=ccrs.PlateCarree(),
                  scale= scale_factor,
                  color='black', width=0.004, headwidth=4, headlength=6)

    # Add quiver key with corrected label and format
    ref_val_for_key = ref_magnitude
    ax.quiverkey(q, X=0.85, Y=1.02, U=ref_val_for_key,
                 label=f'{ref_val_for_key:.1e} [m²/s²]', labelpos='E',
                 coordinates='axes')

    plt.suptitle(f'{title}\nMode {mode_num} - Explained Variance: {explained_var:.1f}%', fontsize=16)

    # Adjust layout
    plt.subplots_adjust(left=0.05, bottom=0.09, right=0.97, top=0.93, wspace=0.1, hspace=0.1)

    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

####################################
# Perform MCA for full time series
####################################

# Perform MCA for full time series
print("\n====== Performing MCA for full time series ======")
print(f"Using irregular region with {len(vertices)} vertices")
mca_results = perform_mca_scalar_vector(
    RWS_data, WAFx_data, WAFy_data,
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
for mode_idx, (pattern_RWS, pattern_WAFx, pattern_WAFy, pc_RWS, pc_WAF, explained_var) in enumerate(mca_results):
    mode_num = mode_idx + 1
    
    # Create a dataset for the patterns
    patterns_ds = xr.Dataset({
        'RWS_pattern': pattern_RWS,
        'WAFx_pattern': pattern_WAFx,
        'WAFy_pattern': pattern_WAFy,
    })

    # Create separate datasets for the time series of each variable
    ts_RWS_ds = xr.Dataset({
        'pc_RWS': pc_RWS,
        'explained_variance': xr.DataArray(explained_var, dims=[])
    })
    
    ts_WAF_ds = xr.Dataset({
        'pc_WAF': pc_WAF,
        'explained_variance': xr.DataArray(explained_var, dims=[])
    })

    # Save results to NetCDF with generation and individual info
    patterns_ds.to_netcdf(os.path.join(SAVE_DIR, f'MCA_RWS_WAF_patterns_mode{mode_num}_gen{generation}_ind{individual}.nc'))
    ts_RWS_ds.to_netcdf(os.path.join(SAVE_DIR, f'MCA_RWS_WAF_timeseries_RWS_mode{mode_num}_gen{generation}_ind{individual}.nc'))
    ts_WAF_ds.to_netcdf(os.path.join(SAVE_DIR, f'MCA_RWS_WAF_timeseries_WAF_mode{mode_num}_gen{generation}_ind{individual}.nc'))

    # Plot the patterns
    plot_mca_patterns_RWS_waf(
        pattern_RWS, pattern_WAFx, pattern_WAFy,
        f'MCA between RWS and WAF Vectors (200hPa)',
        os.path.join(PLOT_DIR, f'MCA_RWS_WAF_patterns_mode{mode_num}_gen{generation}_ind{individual}.png'),
        explained_var,
        mode_num
    )

print(f"\nMCA analysis completed successfully:")
for mode_idx, (*_, explained_var) in enumerate(mca_results):
    print(f"Mode {mode_idx+1} explained variance: {explained_var:.1f}%")
print(f"Results saved in {SAVE_DIR}")
print(f"Plots saved in {PLOT_DIR}")