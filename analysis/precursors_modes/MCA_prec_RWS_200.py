"""
Perform MCA/SVD between precipitation (prec) and RWS at 200 hPa (RWS_200).

 1. Load prec & RWS & compute anomalies globally.
 2. Perform scalar–scalar MCA on a specific region, pick MODE.
 3. Compute regression of prec & RWS anomalies onto the selected MCA time series (PC1), restricted to PLOT_EXTENT.
 4. Mask regression coefficients not significant at 95% confidence.
 5. Save regression maps (prec & RWS) to NetCDF.
 6. Plot significant prec regression (filled) + significant RWS regression (contour lines) on one map in the SVD_SST style.
"""
import os
import sys
from pathlib import Path
import warnings
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.stats import t

# Ensure project root for utils
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

# Suppress known RuntimeWarnings
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for category=RuntimeWarning")
warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
warnings.filterwarnings("ignore", message="invalid value encountered in scalar divide", category=RuntimeWarning)

# ---- PRECURSOR CONFIGURATIONS ----
PRECURSOR_CONFIGS = {
    'C-5': {
        'results_dir': 'Best_modes/C_5/',
        'pc_source': 'prec',
        'pc_mode': 2,
        'vertices': [(-39, -8), (-29, -8), (-29, -19), (-39, -19)],
        'plot_extent': [-64, 0, -38, 10]
    },
    'C-4': {
        'results_dir': 'Best_modes/C_4/',
        'pc_source': 'RWS',
        'pc_mode': 2,
        'vertices': [(-40, -23), (-29, -23), (-29, -33), (-40, -33)],
        'plot_extent': [-66, 0, -44, 10]
    },
    'C-3': {
        'results_dir': 'Best_modes/C_3/',
        'pc_source': 'RWS',
        'pc_mode': 2,
        'vertices': [(-63, 2), (-29, 2), (-29, -22), (-63, -22)],
        'plot_extent': [-66, -2, -40, 10]
    },
    'E-3': {
        'results_dir': 'Best_modes/E_3/',
        'pc_source': 'RWS',
        'pc_mode': 2,
        'vertices': [(-59, 8), (-25, 8), (-25, -29), (-59, -29)],
        'plot_extent': [-64, -6, -40, 10]
    }
}

# ---- ARGUMENT PARSER ----
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Perform MCA/SVD between precipitation and RWS at 200 hPa with optimized plotting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Analysis parameters
    parser.add_argument(
        '--precursor',
        choices=['C-5', 'C-4', 'C-3', 'E-3'],
        required=True,
        help='Select which precursor configuration to use'
    )
    parser.add_argument(
        '--reverse-pattern',
        action='store_true',
        help='Reverse the pattern signs (multiply by -1)'
    )
    parser.add_argument(
        '--significance-level',
        type=float,
        default=0.05,
        help='Significance level for masking'
    )
    
    # Add standard plotting arguments
    add_plotting_arguments(parser)
    
    return parser.parse_args()

# -- PARAMETERS ----
# Parse arguments
args = parse_arguments()

# Setup warning suppression for Cartopy
setup_cartopy_warnings()

# Get configuration for selected precursor
config = PRECURSOR_CONFIGS[args.precursor]
PC_SOURCE = config['pc_source']
PC_MODE = config['pc_mode']
vertices = config['vertices']
PLOT_EXTENT = config['plot_extent']
SIGNIFICANCE_LEVEL = args.significance_level
REVERSE_PATTERN = args.reverse_pattern

DATA_DIR = get_data_path('1_deg_seasonal/', data_type="interim")
DATA_DIR_TS = get_data_path('time_series/', data_type="processed")

RESULTS_DIR = get_results_path(config['results_dir'])
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---- MCA ANALYSIS FUNCTION ----
def perform_mca(field1, field2, region_vertices, n_modes=4):
    """
    Scalar–scalar MCA, returns list of:
      (pattern1, pattern2, pc1_std, pc2_std, explained_cov_fraction)
    """
    # region subset
    d1 = select_irregular_region(field1, region_vertices)
    d2 = select_irregular_region(field2, region_vertices)

    # seasonal standardization
    s1 = standardize_seasonal(d1)
    s2 = standardize_seasonal(d2)

    # lat–lon weighting
    w1, wt1 = lat_lon_weighting(s1)
    w2, wt2 = lat_lon_weighting(s2)

    # reshape
    nt = w1.time.size
    n1 = w1.latitude.size * w1.longitude.size
    n2 = w2.latitude.size * w2.longitude.size

    f1f = w1.values.reshape(nt, n1)
    f2f = w2.values.reshape(nt, n2)

    # mask invalid
    v1 = ~np.isnan(f1f).any(axis=0)
    v2 = ~np.isnan(f2f).any(axis=0)
    f1c = f1f[:, v1]
    f2c = f2f[:, v2]

    # covariance & SVD
    C = np.dot(f1c.T, f2c) / (nt - 1)
    if np.any(np.isnan(C)):
        C = np.nan_to_num(C)
    U, svals, Vt = np.linalg.svd(C, full_matrices=False)
    explained = (svals**2) / np.sum(svals**2) * 100
    nm = min(n_modes, len(svals))

    results = []
    for m in range(nm):
        um = U[:, m]
        vm = Vt[m, :]

        # raw PCs
        pc1_raw = np.dot(f1c, um)
        pc2_raw = np.dot(f2c, vm)

        # back to grid
        full1 = np.full(n1, np.nan); full1[v1] = um
        full2 = np.full(n2, np.nan); full2[v2] = vm

        p1 = full1.reshape(w1.latitude.size, w1.longitude.size)
        p2 = full2.reshape(w2.latitude.size, w2.longitude.size)

        da1 = xr.DataArray((p1 / wt1.data) * pc1_raw.std(ddof=1),
                           dims=['latitude','longitude'],
                           coords={'latitude': w1.latitude, 'longitude': w1.longitude},
                           name=f'prec_pattern_mode{m+1}')
        da2 = xr.DataArray((p2 / wt2.data) * pc2_raw.std(ddof=1),
                           dims=['latitude','longitude'],
                           coords={'latitude': w2.latitude, 'longitude': w2.longitude},
                           name=f'RWS_pattern_mode{m+1}')

        pc1_da = xr.DataArray(pc1_raw, dims=['time'], coords={'time': w1.time})
        pc2_da = xr.DataArray(pc2_raw, dims=['time'], coords={'time': w1.time})
        pc1_std = standardize_seasonal(pc1_da)
        pc2_std = standardize_seasonal(pc2_da)

        results.append((da1, da2, pc1_std, pc2_std, explained[m]))
    return results


# ---- REGRESSION AND SIGNIFICANCE FUNCTION ----
def perform_regression(field_data: xr.DataArray,
                       pc_series: xr.DataArray,
                       significance_level: float = 0.05
                       ) -> tuple[xr.DataArray, xr.DataArray]:
    """
    Regress a field onto PC, within PLOT_EXTENT.
    Returns (slope_map, slope_map_masked_significant).
    """
    lon_min, lon_max, lat_min, lat_max = PLOT_EXTENT
    fd = field_data.sel(longitude=slice(lon_min, lon_max),
                        latitude =slice(lat_min, lat_max))
    fd_al, pc_al = xr.align(fd, pc_series, join='inner')
    ntime = fd_al.time.size

    flat = fd_al.values.reshape(ntime, -1)
    valid = ~np.all(np.isnan(flat), axis=0)
    fv = flat[:, valid]
    pcv = pc_al.values - np.nanmean(pc_al.values)

    # center field
    mv = np.nanmean(fv, axis=0)
    fv_c = fv - mv[None, :]

    # regression
    cov = np.nansum(pcv[:, None] * fv_c, axis=0) / (ntime - 1)
    var_pc = np.nansum(pcv ** 2) / (ntime - 1)
    slope = cov / var_pc

    # significance
    df = ntime - 2
    if df > 0:
        resid = fv_c - pcv[:, None] * slope[None, :]
        SSR = np.nansum(resid**2, axis=0)
        se = np.sqrt(SSR / df) / np.sqrt((ntime - 1) * var_pc)
        t_stats = slope / se
        tcrit = t.ppf(1 - significance_level/2, df)
        signif = np.abs(t_stats) >= tcrit
    else:
        signif = np.zeros_like(slope, dtype=bool)

    # reconstruct
    full_s = np.full(flat.shape[1], np.nan); full_s[valid] = slope
    full_sig = np.full(flat.shape[1], False); full_sig[valid] = signif

    slope2d = full_s.reshape((fd_al.latitude.size, fd_al.longitude.size))
    mask2d  = np.where(full_sig, full_s, np.nan).reshape((fd_al.latitude.size, fd_al.longitude.size))

    da_slope = xr.DataArray(
        slope2d,
        coords=[fd_al.latitude, fd_al.longitude],
        dims=['latitude','longitude'],
        name=f'regression_{fd.name}_on_{pc_series.name}'
    )
    da_sig = xr.DataArray(
        mask2d,
        coords=[fd_al.latitude, fd_al.longitude],
        dims=['latitude','longitude'],
        name=f'regression_sig_{fd.name}_on_{pc_series.name}'
    )
    da_slope.attrs['units'] = f'{fd.attrs.get("units","")} per std dev of PC'
    da_slope.attrs['long_name'] = f'Regression of {fd.name} on {pc_series.name}'
    da_sig.attrs = da_slope.attrs.copy()
    da_sig.attrs['long_name'] += f' (p < {significance_level})'

    return da_slope, da_sig


# ---- LOAD & PREPROCESS DATA ----
print("Loading precipitation data...")
prec = load_era_field(
    filepath=os.path.join(DATA_DIR, 'prec_seas_1deg.nc'),
    var_name='tp'
)
prec.attrs['units'] = 'mm/day'
print(f"prec loaded: {prec.shape}")

print("Loading RWS data...")
RWS = load_era_field(
    filepath=os.path.join(DATA_DIR, 'RWS_200_seas_1deg.nc'),
    var_name='RWS'
)
RWS.attrs['units'] = 'm^2/s'
print(f"RWS loaded: {RWS.shape}")

print("Computing seasonal anomalies...")
prec_an = compute_seasonal_anomalies(prec, '1945-06-01', '2024-02-01')
rws_an  = compute_seasonal_anomalies(RWS, '1945-06-01', '2024-02-01')
prec_an.name = 'prec_anom'
rws_an .name = 'RWS_anom'
print(f"Anomalies shapes — prec:{prec_an.shape}, RWS:{rws_an.shape}")

# ---- RUN MCA & SELECT MODE ----
print(f"Performing MCA, computing {PC_MODE} mode(s)…")
mca_res = perform_mca(prec_an, rws_an, vertices, n_modes=PC_MODE)
if mca_res is None or PC_MODE > len(mca_res):
    sys.exit(f"Error: MCA failed or PC_MODE {PC_MODE} out of range")
# unpack patterns and both PCs
prec_pat, rws_pat, pc1_std, pc2_std, explained_cov = mca_res[PC_MODE-1]

# pick the desired PC‐series
if PC_SOURCE.upper() == 'PREC':
    pc_std_sel = pc1_std
elif PC_SOURCE.upper() == 'RWS':
    pc_std_sel = pc2_std
else:
    sys.exit("Error: PC_SOURCE must be 'prec' or 'RWS'")

print(f"Selected PC series: {PC_SOURCE}, Mode: {PC_MODE}, "
      f"Explained Covariance: {explained_cov:.1f}%")

# ---- REGRESSIONS (within PLOT_EXTENT) ----
print(f"Regressing prec onto {PC_SOURCE} PC Mode {PC_MODE}...")
prec_regr_map, prec_regr_map_sig = perform_regression(prec_an, pc_std_sel, SIGNIFICANCE_LEVEL)
print(f"Regressing RWS onto {PC_SOURCE} PC Mode {PC_MODE}...")
rws_regr_map, rws_regr_map_sig = perform_regression(rws_an, pc_std_sel, SIGNIFICANCE_LEVEL)

# ---- SAVE TO NETCDF ----
# Dynamically create variable names for the dataset
prec_regr_var_name = f'prec_regression_{PC_SOURCE}_mode{PC_MODE}'
prec_regr_sig_var_name = f'prec_regression_sig_{PC_SOURCE}_mode{PC_MODE}'
rws_regr_var_name = f'rws_regression_{PC_SOURCE}_mode{PC_MODE}'
rws_regr_sig_var_name = f'rws_regression_sig_{PC_SOURCE}_mode{PC_MODE}'

ds = xr.Dataset({
    prec_regr_var_name: prec_regr_map,
    prec_regr_sig_var_name: prec_regr_map_sig,
    rws_regr_var_name: rws_regr_map,
    rws_regr_sig_var_name: rws_regr_map_sig,
    'explained_covariance_fraction': xr.DataArray(
        explained_cov,
        dims=[],
        attrs={'units':'%', 'long_name': f'Squared Covariance Fraction for MCA {PC_SOURCE} Mode {PC_MODE}'}
    )
})
nc_file = os.path.join(RESULTS_DIR, f'MCA_prec_RWS_regression_{PC_SOURCE}_mode{PC_MODE}.nc')
ds.to_netcdf(nc_file)
print(f"Saved NetCDF: {nc_file}")

# ---- PLOT ----
print("Generating plot…")
fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
ax.coastlines(linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.6, linestyle=':')
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                  alpha=0.5, linestyle='-')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 12}
gl.ylabel_style = {'size': 12}

# Draw the analysis region polygon after gridlines
poly = Polygon(vertices, transform=ccrs.PlateCarree(), 
               fill=False, edgecolor='black', linewidth=2, linestyle='-')
ax.add_patch(poly)

# Apply pattern reversal if requested
if REVERSE_PATTERN:
    prec_plot_data = -prec_regr_map
    prec_plot_data_sig = -prec_regr_map_sig
    rws_plot_data = -rws_regr_map
    rws_plot_data_sig = -rws_regr_map_sig
else:
    prec_plot_data = prec_regr_map
    prec_plot_data_sig = prec_regr_map_sig
    rws_plot_data = rws_regr_map
    rws_plot_data_sig = rws_regr_map_sig

# Layer 1: prec regression (all values, transparent)
prec_lvls, _ = get_symmetric_levels_fixed_spacing(prec_plot_data_sig, spacing=0.05)
cf1 = ax.contourf(
    prec_plot_data.longitude,
    prec_plot_data.latitude,
    prec_plot_data.values,
    levels=prec_lvls,
    cmap='BrBG',
    extend='both',
    transform=ccrs.PlateCarree(),
    alpha=0.5
)

# Layer 2: prec regression (significant values only, opaque)
cf2 = ax.contourf(
    prec_plot_data_sig.longitude,
    prec_plot_data_sig.latitude,
    prec_plot_data_sig.values,
    levels=prec_lvls,
    cmap='BrBG',
    extend='both',
    transform=ccrs.PlateCarree()
)

# RWS regression - prepare levels
_, rws_conts = get_symmetric_levels_fixed_spacing(rws_plot_data_sig, spacing=4e-12)

# Layer 1: RWS regression (all values, thin transparent lines)
cs1 = ax.contour(
    rws_plot_data.longitude,
    rws_plot_data.latitude,
    rws_plot_data.values,
    levels=rws_conts,
    cmap=mcolors.LinearSegmentedColormap.from_list("", ["darkblue", "blue", "white", "red", "darkred"]),
    linewidths=0.8,
    alpha=0.5,
    transform=ccrs.PlateCarree()
)

# Layer 2: RWS regression (significant values only, thick opaque lines)
cs2 = ax.contour(
    rws_plot_data_sig.longitude,
    rws_plot_data_sig.latitude,
    rws_plot_data_sig.values,
    levels=rws_conts,
    cmap=mcolors.LinearSegmentedColormap.from_list("", ["darkblue", "blue", "white", "red", "darkred"]),
    linewidths=1.6,
    transform=ccrs.PlateCarree()
)
ax.clabel(cs2, rws_conts, fmt='%.1e', fontsize='small')

# colorbar for prec (use cf2 for the significant values)
cbar = plt.colorbar(cf2, ax=ax, orientation='horizontal',
                    shrink=0.3, pad=0.05, aspect=30)
cbar.set_label('Prec anomaly [mm/day per std dev]', fontsize=14)
cbar.ax.tick_params(labelsize=12)

# RWS legend lines (use cs2 for significant values)
handles, labels = [], []
for lvl, coll in zip(rws_conts, cs2.collections):
    color = coll.get_edgecolor()[0]
    lw = coll.get_linewidths()[0]
    handles.append(Line2D([0],[0], color=color, linewidth=lw))
    labels.append(f"{lvl:.1e}")
ax.legend(handles, labels,
          title='RWS Anomaly \n[s$^{-2}$ per std dev]',
          loc='upper right', bbox_to_anchor=(1.24,1),
          fontsize=12, title_fontsize=14, frameon=True)

# title & layout
title_suffix = " (patterns reversed)" if REVERSE_PATTERN else ""
ax.set_title(
    f'Regression onto {PC_SOURCE} PC Mode {PC_MODE} (prec & RWS){title_suffix}\n'
    f'Squared Covariance Fraction: {explained_cov:.1f}%',
    fontsize=16, pad=15
)
plt.tight_layout(rect=[0,0.05,1,0.95])

# Save figure using optimized function
base_filepath = os.path.join(RESULTS_DIR, f'MCA_prec_RWS_regression_{PC_SOURCE}_mode{PC_MODE}')
save_figure_optimized(
    fig,
    base_filepath,
    args.output_format,
    args.raster_dpi,
    args.vector_dpi
)
print(f"Saved plot: {base_filepath}.{args.output_format if args.output_format != 'both' else 'pdf and png'}")

plt.close(fig)
print("Script finished.")
sys.exit(0)