"""
Perform Rotated EOF (REOF) for SST

This script:
1. Loads SST, computes seasonal anomalies
2. Performs REOF on an irregular region, selects a single mode (MODE)
3. Computes regression of global SST anomalies onto the selected PC
4. Masks regression coefficients that are not significant at 95% confidence
5. Saves regression map to NetCDF, plots only significant coefficients
6. Uses a tightened layout and places the colorbar close to the map
"""

import os
import sys
from pathlib import Path
import warnings
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Polygon
from scipy.stats import t
from sklearn.decomposition import PCA
from statsmodels.multivariate.factor_rotation import rotate_factors

# Ensure the project root is in sys.path so utils can be imported
sys.path.append(str(Path(__file__).parent.parent.parent))

from utils.paths import get_data_path, get_results_path
from utils.TBI_functions import (
    compute_seasonal_anomalies,
    standardize_seasonal,
    lat_lon_weighting,
    load_era_field,
    select_irregular_region,
    get_symmetric_levels_fixed_spacing
)
from utils.plotting_optimization import (
    save_figure_optimized,
    add_plotting_arguments,
    setup_cartopy_warnings,
    apply_rasterization_settings
)

# Suppress a known RuntimeWarning from degrees of freedom <= 0
warnings.filterwarnings(
    "ignore",
    message="Degrees of freedom <= 0 for slice",
    category=RuntimeWarning
)

# -----------------------------------------------------------------------------
# ARGUMENT PARSER
# -----------------------------------------------------------------------------
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Perform Rotated EOF (REOF) analysis for SST with optimized plotting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Analysis parameters
    parser.add_argument(
        '--mode',
        type=int,
        default=4,
        help='EOF mode to analyze (1-based indexing)'
    )
    parser.add_argument(
        '--reverse-pattern',
        action='store_true',
        help='Reverse the pattern signs (multiply by -1)'
    )
    parser.add_argument(
        '--plot-extent',
        nargs=4,
        type=float,
        default=[-70, 30, -50, 10],
        metavar=('LON_MIN', 'LON_MAX', 'LAT_MIN', 'LAT_MAX'),
        help='Plot extent: [lon_min, lon_max, lat_min, lat_max]'
    )
    
    # Add standard plotting arguments
    add_plotting_arguments(parser)
    
    return parser.parse_args()

# -----------------------------------------------------------------------------
# PARAMETERS
# -----------------------------------------------------------------------------
# Parse arguments
args = parse_arguments()

# Setup warning suppression for Cartopy
setup_cartopy_warnings()

MODE = args.mode
REVERSE_PATTERN = args.reverse_pattern
PLOT_EXTENT = args.plot_extent

DATA_DIR = get_data_path('1_deg_seasonal/', data_type="interim")
RESULTS_DIR = get_results_path('Best_modes/E_6/')

# Rectangular region for REOF
vertices = [(-35, -22), (0, -22), (0, -39), (-35, -39)]


# -----------------------------------------------------------------------------
# REOF ANALYSIS FUNCTION
# -----------------------------------------------------------------------------
def perform_reof(field_data: xr.DataArray,
                 region_vertices: list,
                 n_components: int = 7,
                 n_modes_to_return: int = 4
                 ) -> list:
    """
    Perform Rotated EOF (REOF) analysis on a field over an irregular domain.
    Returns a list of tuples (eof_pattern, standardized_pc, explained_variance).
    """
    try:
        # 1. Subset to irregular region
        field_dom = select_irregular_region(field_data, region_vertices)

        # 2. Apply latitude weighting
        field_weighted, weights = lat_lon_weighting(field_dom)

        # 3. Reshape to 2D (time, points) and mask invalid
        ntime = field_weighted.time.size
        nlat = field_weighted.latitude.size
        nlon = field_weighted.longitude.size
        flat = field_weighted.values.reshape(ntime, -1)
        nan_mask = np.isnan(flat[0, :])
        valid = ~nan_mask
        data_clean = flat[:, valid]
        if data_clean.shape[1] == 0:
            raise ValueError("No valid data points after masking.")

        # 4. PCA for leading components
        pca = PCA(n_components=n_components)
        pcs_raw = pca.fit_transform(data_clean)
        eofs_raw = pca.components_

        # 5. Varimax rotation
        loadings = eofs_raw.T
        rot_loadings, rot_matrix = rotate_factors(loadings, method='varimax')
        rot_eofs = rot_loadings.T
        rot_pcs = pcs_raw.dot(rot_matrix)

        # 6. Compute rotated explained variance (percent)
        var_pcs = np.var(rot_pcs, axis=0, ddof=1)
        rot_explained = 100.0 * var_pcs / var_pcs.sum()

        # 7. Build results for requested modes
        n_modes = min(n_components, n_modes_to_return)
        results = []
        for m in range(n_modes):
            # reconstruct EOF pattern on original grid
            eof_vec = rot_eofs[m, :]
            full = np.full(nlat * nlon, np.nan)
            full[valid] = eof_vec
            eof2d = full.reshape(nlat, nlon)
            w2d = weights.values.reshape(nlat, nlon)
            eof_unw = eof2d / w2d

            # standardized PC time series
            pc_ts = rot_pcs[:, m]
            pc_std = standardize_seasonal(
                xr.DataArray(pc_ts,
                             coords={'time': field_weighted.time},
                             dims=['time'],
                             name=f'pc_mode_{m+1}')
            )

            # physical EOF (scaled by PC std dev)
            phys = eof_unw * np.std(pc_ts, ddof=1)
            eof_da = xr.DataArray(
                phys,
                coords=[field_weighted.latitude, field_weighted.longitude],
                dims=['latitude', 'longitude'],
                name=f'eof_mode_{m+1}'
            )

            results.append((eof_da, pc_std, rot_explained[m]))

        return results

    except Exception:
        import traceback
        traceback.print_exc()
        return None

# -----------------------------------------------------------------------------
# LOAD AND PREPROCESS DATA
# -----------------------------------------------------------------------------
print("Loading SST data...")
SST = load_era_field(
    filepath=os.path.join(DATA_DIR, 'SST_seas_1deg.nc'),
    var_name='sst',
)
print(f"SST loaded: {SST.shape}")

print("Computing seasonal anomalies...")
SST_anom = compute_seasonal_anomalies(SST, '1940-09-01', '2024-02-01')
SST_data = SST_anom
print(f"Anomalies computed: {SST_data.shape}")

# -----------------------------------------------------------------------------
# RUN REOF AND EXTRACT SELECTED MODE
# -----------------------------------------------------------------------------
print(f"Performing REOF, selecting mode {MODE}...")
reof_results = perform_reof(
    field_data=SST_data,
    region_vertices=vertices,
    n_components=7,
    n_modes_to_return=MODE
)
if reof_results is None or MODE > len(reof_results):
    sys.exit("Error: REOF failed or MODE out of range.")

eof_sel, pc_sel_std, explained_sel = reof_results[MODE-1]

# -----------------------------------------------------------------------------
# REGRESSION OF GLOBAL SST ANOMALIES ONTO SELECTED PC
# -----------------------------------------------------------------------------
ntime = SST_data.time.size
nlat = SST_data.latitude.size
nlon = SST_data.longitude.size

# Flatten spatial dimensions for regression
sst_flat = SST_data.values.reshape(ntime, -1)

# Identify grid cells with all-NaN (e.g., land) to avoid empty-slice warnings
all_nan = np.all(np.isnan(sst_flat), axis=0)
# Compute mean only over cells that have at least one valid value
sst_mean_valid = np.nanmean(sst_flat[:, ~all_nan], axis=0)
# Reconstruct full mean array with NaNs in land cells
sst_mean = np.full(sst_flat.shape[1], np.nan)
sst_mean[~all_nan] = sst_mean_valid

# Center both SST and PC
pc = pc_sel_std.values
pc_c = pc - np.nanmean(pc)   # PC anomalies

sst_c = sst_flat - sst_mean[None, :]  # SST anomalies centered by time mean

# Compute covariance and regression slope
cov = np.nansum(pc_c[:, None] * sst_c, axis=0) / (ntime - 1)
var_pc = np.nansum(pc_c**2) / (ntime - 1)
slope = cov / var_pc

# Compute residuals for standard error of slope
resid = sst_c - pc_c[:, None] * slope[None, :]
df = ntime - 2
SSR = np.nansum(resid**2, axis=0)
# Standard error of slope
se_slope = np.sqrt(SSR / df) / np.sqrt((ntime - 1) * var_pc)

# Avoid division-by-zero: compute t-stats only where se_slope != 0
t_stats = np.zeros_like(slope)
valid_se = se_slope != 0
t_stats[valid_se] = slope[valid_se] / se_slope[valid_se]

# 95% confidence threshold for two-tailed t-test
tcrit = t.ppf(1 - 0.025, df)
signif = np.abs(t_stats) >= tcrit

# Reshape back to spatial grid
slope2d = slope.reshape((nlat, nlon))
# Mask out land points (where all SST values were NaN)
land_mask = all_nan.reshape((nlat, nlon))
slope2d[land_mask] = np.nan
# Mask out non-significant values
slope_sig2d = np.where(signif.reshape((nlat, nlon)), slope2d, np.nan)

# Build DataArrays for saving and plotting
regression_map = xr.DataArray(
    slope2d,
    coords=[SST_data.latitude, SST_data.longitude],
    dims=['latitude', 'longitude'],
    name=f'regression_map_mode{MODE}'
)
regression_map.attrs['long_name'] = f'Regression coefficient of SST onto PC mode {MODE}'
regression_map.attrs['units'] = 'K per standard deviation of PC'

regression_map_sig = xr.DataArray(
    slope_sig2d,
    coords=[SST_data.latitude, SST_data.longitude],
    dims=['latitude', 'longitude'],
    name=f'regression_map_sig_mode{MODE}'
)
regression_map_sig.attrs = regression_map.attrs

# -----------------------------------------------------------------------------
# SAVE FULL REGRESSION MAP TO NETCDF
# -----------------------------------------------------------------------------
ds = xr.Dataset({
    f'regression_map_mode{MODE}': regression_map,
    'explained_variance': xr.DataArray(explained_sel,
                                       dims=[],
                                       attrs={'units': '%'})
})
nc_file = os.path.join(RESULTS_DIR, f'REOF_SST_regression_mode{MODE}.nc')
ds.to_netcdf(nc_file)
print(f"NetCDF saved: {nc_file}")

# -----------------------------------------------------------------------------
# PLOT WITH TRANSPARENCY FOR NON-SIGNIFICANT AREAS
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(15, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
ax.coastlines(linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.6, linestyle=':')
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                  alpha=0.5, linestyle='--')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}

# Apply pattern reversal if requested
plot_data = regression_map.copy()
if REVERSE_PATTERN:
    plot_data = -plot_data

# Generate symmetric contour levels with fixed spacing using full data
levels, contour_lvls = get_symmetric_levels_fixed_spacing(plot_data, spacing=0.1)
cmap = 'coolwarm'

# Build kwargs for contourf with conditional rasterization
contourf_kwargs_transparent = {
    'levels': levels,
    'cmap': cmap,
    'extend': 'both',
    'alpha': 0.4,  # Transparent for non-significant
    'transform': ccrs.PlateCarree()
}
apply_rasterization_settings(contourf_kwargs_transparent, args.output_format)

contourf_kwargs_opaque = {
    'levels': levels,
    'cmap': cmap,
    'extend': 'both',
    'transform': ccrs.PlateCarree()
}
apply_rasterization_settings(contourf_kwargs_opaque, args.output_format)

# Layer 1: All values with transparency
cf1 = ax.contourf(
    plot_data.longitude,
    plot_data.latitude,
    plot_data.values,
    **contourf_kwargs_transparent
)

# Layer 2: Significant values only (opaque)
cf2 = ax.contourf(
    regression_map_sig.longitude,
    regression_map_sig.latitude,
    regression_map_sig.values if not REVERSE_PATTERN else -regression_map_sig.values,
    **contourf_kwargs_opaque
)

# Optional contour lines on all data (thin, transparent)
try:
    cs1 = ax.contour(
        plot_data.longitude,
        plot_data.latitude,
        plot_data.values,
        levels=contour_lvls,
        colors='gray',
        linewidths=0.5,
        alpha=0.5,
        transform=ccrs.PlateCarree()
    )
except ValueError:
    pass

# Contour lines on significant areas (thick, opaque)
try:
    cs2 = ax.contour(
        regression_map_sig.longitude,
        regression_map_sig.latitude,
        regression_map_sig.values if not REVERSE_PATTERN else -regression_map_sig.values,
        levels=contour_lvls,
        colors='darkgray',
        linewidths=1.2,
        transform=ccrs.PlateCarree()
    )
    plt.clabel(cs2, contour_lvls, fmt='%1.1f',
               fontsize='small', colors='black',
               inline=True, inline_spacing=4)
except ValueError:
    pass

# Draw analysis region on top of everything else for better visibility
region_poly = Polygon(vertices, fill=False, edgecolor='black', 
                      linewidth=3, linestyle='-', transform=ccrs.PlateCarree(),
                      zorder=10)  # High zorder to ensure it's on top
ax.add_patch(region_poly)

# Use cf2 for colorbar (the opaque layer)
cbar = plt.colorbar(cf2, ax=ax,
                    orientation='horizontal',
                    shrink=0.5,
                    pad=0.05)
cbar.set_label('SST anomaly [K per std dev PC]', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# ax.set_title(
#     f'Regression of SST onto REOF {MODE}\n'
#     f'Explained Variance: {explained_sel:.1f}%',
#     fontsize=18, pad=15
# )

plt.tight_layout()

# Save figure using optimized function
base_filepath = os.path.join(RESULTS_DIR, f'REOF_SST_regression_mode{MODE}')
save_figure_optimized(
    fig,
    base_filepath,
    args.output_format,
    args.raster_dpi,
    args.vector_dpi
)
print(f"Plot saved: {base_filepath}.{args.output_format if args.output_format != 'both' else 'pdf and png'}")
plt.close(fig)

sys.exit(0)
