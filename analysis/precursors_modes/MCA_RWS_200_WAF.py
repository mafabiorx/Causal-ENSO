"""
Perform MCA/SVD between RWS at 200 hPa and WAF vectors at 200 hPa.

 1. Load RWS, WAFx, WAFy & compute anomalies globally.
 2. Perform scalar–vector MCA on a specific region, pick MODE.
 3. Compute regression of RWS, WAFx & WAFy anomalies onto the selected MCA time series (PC1),
    restricted to PLOT_EXTENT.
 4. Mask regression coefficients not significant at 95% confidence.
 5. Save regression maps (RWS & WAF components) to NetCDF.
 6. Plot significant RWS regression (filled) + significant WAF regression (vectors) on one map
    in the SVD_SST style.
"""
import os
import sys
from pathlib import Path
import warnings
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    'E-5': {
        'results_dir': 'Best_modes/E_5/',
        'pc_source': 'WAF',
        'pc_mode': 1,
        'vertices': [(-42, -3), (-19, -3), (-19, -23), (-42, -23)],
        'plot_extent': [-63, 10, -44, 20],
        'scaling': 2e8
    },
    'E-4': {
        'results_dir': 'Best_modes/E_4/',
        'pc_source': 'RWS',
        'pc_mode': 2,
        'vertices': [(-42, -16), (-28, -16), (-28, -34), (-42, -34)],
        'plot_extent': [-66, 15, -45, 0],
        'scaling': 5e8
    }
}

# ---- ARGUMENT PARSER ----
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Perform MCA/SVD between RWS and WAF vectors at 200 hPa with optimized plotting',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Analysis parameters
    parser.add_argument(
        '--precursor',
        choices=['E-5', 'E-4'],
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

# ---- PARAMETERS ----
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
scaling = config['scaling']
SIGNIFICANCE_LEVEL = args.significance_level
REVERSE_PATTERN = args.reverse_pattern

DATA_DIR = get_data_path('1_deg_seasonal/', data_type="interim")

RESULTS_DIR = get_results_path(config['results_dir'])
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---- MCA ANALYSIS FUNCTION ----
def perform_mca(f1, f2x, f2y, region_vertices, n_modes=4):
    """
    Scalar–vector MCA between f1 and (f2x, f2y). Returns list of
    (pattern1, pattern2x, pattern2y, pc1_std, pc2_std, explained_cov_fraction).
    """
    # 1) subset to region
    d1 = select_irregular_region(f1, region_vertices)
    dx = select_irregular_region(f2x, region_vertices)
    dy = select_irregular_region(f2y, region_vertices)
    d1, dx, dy = xr.align(d1, dx, dy, join='inner')

    # 2) seasonal standardization
    s1 = standardize_seasonal(d1)
    sx = standardize_seasonal(dx)
    sy = standardize_seasonal(dy)

    # 3) latitude weighting
    w1, wt1 = lat_lon_weighting(s1)
    wx, wtx = lat_lon_weighting(sx)
    wy, wty = lat_lon_weighting(sy)

    # 4) reshape to (time,points)
    nt = w1.time.size
    n1 = w1.latitude.size * w1.longitude.size
    n2 = wx.latitude.size   * wx.longitude.size

    f1f = w1.values.reshape(nt, n1)
    f2xf = wx.values.reshape(nt, n2)
    f2yf = wy.values.reshape(nt, n2)

    # 5) stack vector
    f2v = np.concatenate([f2xf, f2yf], axis=1)

    # 6) mask invalid columns
    v1 = ~np.isnan(f1f).any(axis=0)
    v2x= ~np.isnan(f2xf).any(axis=0)
    v2y= ~np.isnan(f2yf).any(axis=0)
    v2 = np.concatenate([v2x, v2y])

    f1c = f1f[:, v1]
    f2c = f2v[:, v2]

    # 7) covariance and SVD
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
        full2 = np.full(2*n2, np.nan); full2[v2] = vm

        p1 = full1.reshape(w1.latitude.size, w1.longitude.size)
        p2x = full2[:n2].reshape(wx.latitude.size, wx.longitude.size)
        p2y = full2[n2:].reshape(wy.latitude.size, wy.longitude.size)

        # unweight & scale by PC std
        da1 = xr.DataArray((p1 / wt1.data) * pc1_raw.std(ddof=1),
                           dims=['latitude','longitude'],
                           coords={'latitude': w1.latitude, 'longitude': w1.longitude},
                           name=f'RWS_pattern_mode{m+1}')
        da2x = xr.DataArray((p2x / wtx.data) * pc2_raw.std(ddof=1),
                            dims=['latitude','longitude'],
                            coords={'latitude': wx.latitude, 'longitude': wx.longitude},
                            name=f'WAFx_pattern_mode{m+1}')
        da2y = xr.DataArray((p2y / wty.data) * pc2_raw.std(ddof=1),
                            dims=['latitude','longitude'],
                            coords={'latitude': wy.latitude, 'longitude': wy.longitude},
                            name=f'WAFy_pattern_mode{m+1}')

        # standardized seasonal PCs
        pc1_da = xr.DataArray(pc1_raw, dims=['time'], coords={'time': w1.time})
        pc2_da = xr.DataArray(pc2_raw, dims=['time'], coords={'time': w1.time})
        pc1_std = standardize_seasonal(pc1_da)
        pc2_std = standardize_seasonal(pc2_da)

        results.append((da1, da2x, da2y, pc1_std, pc2_std, explained[m]))
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

    # reconstruct full grids
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
    da_slope.attrs['units'] = f'{fd.attrs.get("units","")}per std dev of PC'
    da_slope.attrs['long_name'] = f'Regression of {fd.name} on {pc_series.name}'
    da_sig.attrs = da_slope.attrs.copy()
    da_sig.attrs['long_name'] += f' (p < {significance_level})'

    return da_slope, da_sig


# ---- LOAD & PREPROCESS DATA ----
print("Loading RWS data...")
RWS = load_era_field(
    filepath=os.path.join(DATA_DIR, 'RWS_200_seas_1deg.nc'),
    var_name='RWS'
)
RWS.attrs['units'] = 'm^2/s'
print(f"RWS loaded: {RWS.shape}")

print("Loading WAFx data...")
WAFx = load_era_field(
    filepath=os.path.join(DATA_DIR, 'WAF_200_components_1deg.nc'),
    var_name='WAFx'
)
WAFx.attrs['units'] = 'm^2/s^2'
print(f"WAFx loaded: {WAFx.shape}")

print("Loading WAFy data...")
WAFy = load_era_field(
    filepath=os.path.join(DATA_DIR, 'WAF_200_components_1deg.nc'),
    var_name='WAFy'
)
WAFy.attrs['units'] = 'm^2/s^2'
print(f"WAFy loaded: {WAFy.shape}")

print("Computing seasonal anomalies...")
rws_an = compute_seasonal_anomalies(RWS, '1945-06-01', '2024-02-01')
wx_an  = compute_seasonal_anomalies(WAFx, '1945-06-01', '2024-02-01')
wy_an  = compute_seasonal_anomalies(WAFy, '1945-06-01', '2024-02-01')
rws_an.name = 'RWS_anom'
wx_an.name  = 'WAFx_anom'
wy_an.name  = 'WAFy_anom'
print(f"Anomalies shapes — RWS: {rws_an.shape}, WAFx: {wx_an.shape}, WAFy: {wy_an.shape}")

# ---- RUN MCA & SELECT MODE ----
print(f"Performing MCA, computing first {PC_MODE} mode(s)…")
mca_res = perform_mca(rws_an, wx_an, wy_an, vertices, n_modes=PC_MODE)
if mca_res is None or PC_MODE > len(mca_res):
    sys.exit(f"Error: MCA failed or PC_MODE {PC_MODE} out of range")
# unpack patterns and both PCs
rws_pat, wafx_pat, wafy_pat, pc1_std, pc2_std, explained_cov = mca_res[PC_MODE-1]

# pick the desired PC‐series
if PC_SOURCE.upper() == 'RWS':
    pc_std_sel = pc1_std
elif PC_SOURCE.upper() == 'WAF':
    pc_std_sel = pc2_std
else:
    sys.exit("Error: PC_SOURCE must be 'RWS' or 'WAF'")

print(f"Selected PC series: {PC_SOURCE}, Mode: {PC_MODE}, "
      f"Explained Covariance: {explained_cov:.1f}%")

# ---- REGRESSIONS (within PLOT_EXTENT) ----
print(f"Regressing RWS onto {PC_SOURCE} PC Mode {PC_MODE}...")
rws_regr_map, rws_regr_map_sig = perform_regression(rws_an, pc_std_sel, SIGNIFICANCE_LEVEL)
print(f"Regressing WAFx onto {PC_SOURCE} PC Mode {PC_MODE}...")
wafx_regr_map, wafx_regr_map_sig = perform_regression(wx_an, pc_std_sel, SIGNIFICANCE_LEVEL)
print(f"Regressing WAFy onto {PC_SOURCE} PC Mode {PC_MODE}...")
wafy_regr_map, wafy_regr_map_sig = perform_regression(wy_an, pc_std_sel, SIGNIFICANCE_LEVEL)

# ---- SAVE TO NETCDF ----
# Dynamically create variable names for the dataset
rws_regr_var_name = f'rws_regression_{PC_SOURCE}_mode{PC_MODE}'
rws_regr_sig_var_name = f'rws_regression_sig_{PC_SOURCE}_mode{PC_MODE}'
wafx_regr_var_name = f'wafx_regression_{PC_SOURCE}_mode{PC_MODE}'
wafx_regr_sig_var_name = f'wafx_regression_sig_{PC_SOURCE}_mode{PC_MODE}'
wafy_regr_var_name = f'wafy_regression_{PC_SOURCE}_mode{PC_MODE}'
wafy_regr_sig_var_name = f'wafy_regression_sig_{PC_SOURCE}_mode{PC_MODE}'

ds = xr.Dataset({
    rws_regr_var_name: rws_regr_map,
    rws_regr_sig_var_name: rws_regr_map_sig,
    wafx_regr_var_name: wafx_regr_map,
    wafx_regr_sig_var_name: wafx_regr_map_sig,
    wafy_regr_var_name: wafy_regr_map,
    wafy_regr_sig_var_name: wafy_regr_map_sig,
    'explained_covariance_fraction': xr.DataArray(
        explained_cov,  # Use the correct explained_cov variable
        dims=[],
        attrs={
            'units': '%',
            'long_name': f'Squared Covariance Fraction for MCA {PC_SOURCE} Mode {PC_MODE}'
        }
    )
})
nc_file = os.path.join(
    RESULTS_DIR,
    f'MCA_RWS_WAF_regression_{PC_SOURCE}_mode{PC_MODE}.nc'
)
ds.to_netcdf(nc_file)
print(f"Saved NetCDF: {nc_file}")

# ---- PLOT ----
print("Generating plot…")

# Apply pattern reversal if requested
if REVERSE_PATTERN:
    print("Reversing pattern (multiplying by -1)...")
    rws_regr_map = rws_regr_map * -1
    rws_regr_map_sig = rws_regr_map_sig * -1
    wafx_regr_map = wafx_regr_map * -1
    wafx_regr_map_sig = wafx_regr_map_sig * -1
    wafy_regr_map = wafy_regr_map * -1
    wafy_regr_map_sig = wafy_regr_map_sig * -1

fig = plt.figure(figsize=(18, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent(PLOT_EXTENT, crs=ccrs.PlateCarree())
ax.coastlines(linewidth=0.8)
ax.add_feature(cfeature.BORDERS, linewidth=0.6, linestyle=':')
gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                  alpha=0.5, linestyle='-')
gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {'size': 14}
gl.ylabel_style = {'size': 14}

# Draw the analysis region polygon
polygon = Polygon(vertices, closed=True, edgecolor='black', 
                  facecolor='none', linewidth=2, linestyle='-',
                  transform=ccrs.PlateCarree())
ax.add_patch(polygon)

# RWS regression - TWO LAYERS
# Build kwargs for contourf with conditional rasterization
rws_levels, _ = get_symmetric_levels_fixed_spacing(rws_regr_map, spacing=5e-12)
rws_cmap = mcolors.LinearSegmentedColormap.from_list("", ["darkblue", "blue", "white", "red", "darkred"])

contourf_kwargs_transparent = {
    'levels': rws_levels,
    'cmap': rws_cmap,
    'extend': 'both',
    'alpha': 0.4,
    'transform': ccrs.PlateCarree()
}
apply_rasterization_settings(contourf_kwargs_transparent, args.output_format)

contourf_kwargs_opaque = {
    'levels': rws_levels,
    'cmap': rws_cmap,
    'extend': 'both',
    'alpha': 1.0,
    'transform': ccrs.PlateCarree()
}
apply_rasterization_settings(contourf_kwargs_opaque, args.output_format)

# Layer 1: All regression values (semi-transparent)
cf1 = ax.contourf(
    rws_regr_map.longitude,
    rws_regr_map.latitude,
    rws_regr_map.values,
    **contourf_kwargs_transparent
)

# Layer 2: Only significant values (fully opaque)
cf2 = ax.contourf(
    rws_regr_map_sig.longitude,
    rws_regr_map_sig.latitude,
    rws_regr_map_sig.values,
    **contourf_kwargs_opaque
)

# WAF vectors - TWO LAYERS
subsample = 3
lon = wafx_regr_map['longitude'][::subsample].values
lat = wafx_regr_map['latitude'][::subsample].values

# Get all vectors and significant vectors
u_all = wafx_regr_map.sel(longitude=lon, latitude=lat).values
v_all = wafy_regr_map.sel(longitude=lon, latitude=lat).values
u_sig = wafx_regr_map_sig.sel(longitude=lon, latitude=lat).values
v_sig = wafy_regr_map_sig.sel(longitude=lon, latitude=lat).values

# Filter out equatorial vectors (5°N to 5°S) to prevent scale distortion
lat_mask = (lat < -5) | (lat > 5)
u_all_filtered = u_all.copy()
v_all_filtered = v_all.copy()
u_sig_filtered = u_sig.copy()
v_sig_filtered = v_sig.copy()

# Set equatorial vectors to NaN for scaling calculations
u_all_filtered[~lat_mask, :] = np.nan
v_all_filtered[~lat_mask, :] = np.nan
u_sig_filtered[~lat_mask, :] = np.nan
v_sig_filtered[~lat_mask, :] = np.nan

# Calculate magnitudes for scaling using filtered vectors
mag_all = np.sqrt(u_all_filtered**2 + v_all_filtered**2)
mag_sig = np.sqrt(u_sig_filtered**2 + v_sig_filtered**2)

# Use filtered vectors for scaling calculations (excluding equatorial region)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    m05 = np.nanquantile(mag_all, 0.05) if not np.all(np.isnan(mag_all)) else 0.0
    m95 = np.nanquantile(mag_all, 0.95) if not np.all(np.isnan(mag_all)) else 1.0

min_len, max_len = 0.6, 1.4
if m95 > m05:
    slope = (max_len - min_len) / (m95 - m05)
    intercept = max_len - slope * m95
    
    # Scale all vectors (using filtered magnitudes for scaling but original vectors for output)
    mag_all_orig = np.sqrt(u_all**2 + v_all**2)
    new_mag_all = mag_all * slope + intercept
    ratio_all = np.zeros_like(mag_all_orig)
    non_zero_mask_all = mag_all_orig != 0
    # Use filtered magnitudes for scaling calculation but apply to original vectors
    ratio_all[non_zero_mask_all] = new_mag_all[non_zero_mask_all] / mag_all_orig[non_zero_mask_all]
    ratio_all[np.isnan(mag_all) | (np.abs(lat[:, None]) <= 3)] = 0.0  # Zero out equatorial vectors
    u_all_scaled = u_all * ratio_all
    v_all_scaled = v_all * ratio_all
    
    # Scale significant vectors
    mag_sig_orig = np.sqrt(u_sig**2 + v_sig**2)
    new_mag_sig = mag_sig * slope + intercept
    ratio_sig = np.zeros_like(mag_sig_orig)
    non_zero_mask_sig = mag_sig_orig != 0
    ratio_sig[non_zero_mask_sig] = (new_mag_sig[non_zero_mask_sig] / 
                                     mag_sig_orig[non_zero_mask_sig])
    ratio_sig[np.isnan(mag_sig) | (np.abs(lat[:, None]) <= 3)] = 0.0  # Zero out equatorial vectors
    u_sig_scaled = u_sig * ratio_sig
    v_sig_scaled = v_sig * ratio_sig
else:
    # Even when no scaling, still filter out equatorial vectors
    u_all_scaled = u_all.copy()
    v_all_scaled = v_all.copy()
    u_sig_scaled = u_sig.copy()
    v_sig_scaled = v_sig.copy()
    # Zero out equatorial vectors
    equatorial_mask = np.abs(lat[:, None]) <= 3
    u_all_scaled[equatorial_mask] = 0.0
    v_all_scaled[equatorial_mask] = 0.0
    u_sig_scaled[equatorial_mask] = 0.0
    v_sig_scaled[equatorial_mask] = 0.0

ref_mag = np.nanpercentile(mag_all, 95) if not np.all(np.isnan(mag_all)) else 1.0
if ref_mag == 0: ref_mag = 1.0
scale_factor = scaling * ref_mag

# Layer 1: All vectors (gray, thin, semi-transparent)
q1 = ax.quiver(
    lon, lat, u_all_scaled, v_all_scaled,
    transform=ccrs.PlateCarree(),
    scale=scale_factor,
    color='gray', width=0.002, headwidth=3, headlength=4,
    alpha=0.5
)

# Layer 2: Significant vectors (black, thick, fully opaque)
q2 = ax.quiver(
    lon, lat, u_sig_scaled, v_sig_scaled,
    transform=ccrs.PlateCarree(),
    scale=scale_factor,
    color='black', width=0.004, headwidth=4, headlength=6,
    alpha=1.0
)

"""
# Quiverkey for reference
ax.quiverkey(q2, X=0.85, Y=1.02, U=ref_mag,
             label=f'{ref_mag:.1e} [m²/s²]', labelpos='E',
             coordinates='axes')
"""

# Colorbar for RWS
cbar = plt.colorbar(cf2, ax=ax, orientation='horizontal',
                    shrink=0.4, pad=0.06, aspect=30)
cbar.set_label(f'RWS regression [{rws_regr_map.attrs.get("units", "s^-2")}]', fontsize=16)
cbar.ax.tick_params(labelsize=14)

# ax.set_title(
#     f'Regression onto {PC_SOURCE} PC Mode {PC_MODE} (RWS & WAF)\n'
#     f'Squared Covariance Fraction: {explained_cov:.1f}%',
#     fontsize=16, pad=15
# )
plt.tight_layout(rect=[0, 0.05, 1, 0.95])

# Save figure using optimized function
base_filepath = os.path.join(
    RESULTS_DIR,
    f'MCA_RWS_WAF_regression_{PC_SOURCE}_mode{PC_MODE}'
)
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
