"""
Pacific Meridional Modes: NPMM and SPMM Computation via MCA

Core Functionality:
- Compute North Pacific Meridional Mode (NPMM) via Maximum Covariance Analysis of SST and surface winds
- Calculate South Pacific Meridional Mode (SPMM) using SST-wind coupling analysis
- Remove ENSO influence through Cold Tongue Index regression before MCA computation
- Apply seasonal standardization and latitude weighting for robust mode extraction

Output: NPMM and SPMM time series and spatial pattern visualizations for meridional mode analysis
"""

import numpy as np
import xarray as xr
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))
from utils.TBI_functions import total_average, compute_seasonal_anomalies, standardize_seasonal, lat_lon_weighting, get_symmetric_levels, add_cyclic_point_xr
from utils.paths import get_data_path, get_results_path

# Set the paths
DATA_DIR = get_data_path('seasonal/', data_type="raw")
SAVE_DIR = get_data_path('time_series/')
PLOT_DIR = get_results_path('main_climate_modes/')


def compute_cti(sst):
    """
    Compute the Cold Tongue Index (CTI) defined as the average SSTA over (6S-6N, 180-90W).
    Note: 90W = 270E if longitudes are 0-360.

    Args:
        sst (xarray.DataArray): Sea surface temperature anomalies

    Returns:
        cti (xarray.DataArray): The CTI time series
    """
    # Select the region (6S-6N, 180-90W)
    cti_region = sst.sel(latitude=slice(-6, 6), longitude=slice(180, 270))
    cti = total_average(cti_region, 'latitude')
    cti = standardize_seasonal(cti)
    return cti


def remove_cti_influence(data, cti):
    """Remove CTI influence using linear regression"""
    # Reshape data for regression
    data_2d = data.stack(space=['latitude', 'longitude'])

    def regress_out_cti(x):
        if np.any(np.isnan(x)) or np.any(np.isnan(cti)):
            return x
        slope, intercept = np.polyfit(cti, x, 1)
        return x - (slope * cti + intercept)

    # Apply regression
    data_corrected = xr.apply_ufunc(
        regress_out_cti,
        data_2d,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        vectorize=True
    ).unstack('space')

    return data_corrected


def normalize_wind_field(u, v):
    """
    Normalize U and V wind components jointly but separately for each month to maintain
    consistency with SST standardization approach.

    Args:
        u (xarray.DataArray): Zonal wind component
        v (xarray.DataArray): Meridional wind component

    Returns:
        tuple: (u_norm, v_norm) Normalized wind components
    """
    def standardize_winds_monthly(month_u, month_v):
        # Combine fields for joint standardization
        combined = np.concatenate([month_u.values.flatten(),
                                 month_v.values.flatten()])
        # Calculate joint standard deviation
        std = np.std(combined)
        # Normalize both components with the same std
        return month_u / std, month_v / std

    # Group by months
    month_labels_u = u.time.dt.month
    month_labels_v = v.time.dt.month

    # Initialize output arrays
    u_norm = u.copy()
    v_norm = v.copy()

    # Process each month separately
    for month in np.unique(month_labels_u):
        # Select data for current month
        u_month = u.where(month_labels_u == month, drop=True)
        v_month = v.where(month_labels_v == month, drop=True)

        # Normalize jointly
        u_norm_month, v_norm_month = standardize_winds_monthly(u_month, v_month)

        # Assign back to output arrays
        u_norm.loc[{'time': u_month.time}] = u_norm_month
        v_norm.loc[{'time': v_month.time}] = v_norm_month

    return u_norm, v_norm


def prepare_mca_data(sst, u, v, lat_range, lon_range, cti):
    """
    Prepare data for MCA analysis: subset, remove CTI, standardize, weight, and flatten.

    Args:
        sst (xarray.DataArray): Sea surface temperature anomalies [time, lat, lon]
        u   (xarray.DataArray): Zonal wind anomalies [time, lat, lon]
        v   (xarray.DataArray): Meridional wind anomalies [time, lat, lon]
        lat_range (tuple): (lat_min, lat_max)
        lon_range (tuple): (lon_min, lon_max)
        cti (xarray.DataArray): Cold Tongue Index time series [time]

    Returns:
        dict: Dictionary containing:
            - sst_clean: Cleaned SST data (time, valid_points)
            - wind_matrix: Combined U,V wind data (time, 2*valid_wind_points)
            - valid_sst: Boolean mask for valid SST points
            - valid_wind: Boolean mask for valid wind points
            - weights: Latitude weights
            - coords: Dictionary with latitude, longitude, time coordinates
            - dims: Dictionary with ntime, nlat, nlon dimensions
    """
    # 1) Subset to the specified lat-lon range
    sst_dom = sst.sel(latitude=slice(lat_range[0], lat_range[1]),
                      longitude=slice(lon_range[0], lon_range[1]))
    u_dom = u.sel(latitude=slice(lat_range[0], lat_range[1]),
                  longitude=slice(lon_range[0], lon_range[1]))
    v_dom = v.sel(latitude=slice(lat_range[0], lat_range[1]),
                  longitude=slice(lon_range[0], lon_range[1]))

    # 2) Regress out CTI from each field
    sst_dom_cti = remove_cti_influence(sst_dom, cti)
    u_dom_cti   = remove_cti_influence(u_dom,   cti)
    v_dom_cti   = remove_cti_influence(v_dom,   cti)

    # 3) Standardize data
    sst_dom_cti = standardize_seasonal(sst_dom_cti)
    u_dom_cti   = standardize_seasonal(u_dom_cti)
    v_dom_cti   = standardize_seasonal(v_dom_cti)

    # 4) Apply latitude weighting
    sst_weighted, weights = lat_lon_weighting(sst_dom_cti)
    u_weighted,   _       = lat_lon_weighting(u_dom_cti)
    v_weighted,   _       = lat_lon_weighting(v_dom_cti)

    # 5) Flatten each field to (time, n_points)
    ntime = sst_weighted.time.size
    nlat  = sst_weighted.latitude.size
    nlon  = sst_weighted.longitude.size

    sst_flat = sst_weighted.values.reshape(ntime, nlat*nlon)
    u_flat   = u_weighted.values.reshape(ntime, nlat*nlon)
    v_flat   = v_weighted.values.reshape(ntime, nlat*nlon)

    # 6) Create valid masks for each field
    valid_sst  = ~np.isnan(sst_flat).any(axis=0)
    valid_u    = ~np.isnan(u_flat).any(axis=0)
    valid_v    = ~np.isnan(v_flat).any(axis=0)
    valid_wind = valid_u & valid_v

    # 7) Subset the arrays
    sst_clean = sst_flat[:, valid_sst]
    u_clean   = u_flat[:, valid_wind]
    v_clean   = v_flat[:, valid_wind]

    # Combine U and V into one matrix for cross-covariance
    wind_matrix = np.concatenate([u_clean, v_clean], axis=1)

    return {
        'sst_clean': sst_clean,
        'wind_matrix': wind_matrix,
        'valid_sst': valid_sst,
        'valid_wind': valid_wind,
        'weights': weights,
        'coords': {
            'latitude': sst_weighted.latitude,
            'longitude': sst_weighted.longitude,
            'time': sst_weighted.time,
            'u_latitude': u_dom_cti.latitude,
            'u_longitude': u_dom_cti.longitude
        },
        'dims': {
            'ntime': ntime,
            'nlat': nlat,
            'nlon': nlon
        }
    }


def compute_mca_decomposition(sst_clean, wind_matrix, ntime):
    """
    Compute MCA decomposition using SVD on cross-covariance matrix.

    Args:
        sst_clean (numpy.ndarray): Cleaned SST data (time, valid_sst_points)
        wind_matrix (numpy.ndarray): Combined wind data (time, 2*valid_wind_points)
        ntime (int): Number of time steps

    Returns:
        dict: Dictionary containing:
            - sst_pattern_1d: First singular vector for SST
            - wind_pattern_1d: First singular vector for wind
            - pc_sst_arr: SST principal component time series
            - pc_wind_arr: Wind principal component time series
            - explained_var: Explained variance (%) by first mode
    """
    # Compute cross-covariance matrix, then SVD
    C = np.dot(sst_clean.T, wind_matrix) / (ntime - 1)

    U_s, singular_values, V_s = np.linalg.svd(C, full_matrices=False)
    explained_variance = (singular_values**2) / np.sum(singular_values**2) * 100
    explained_var = explained_variance[0]

    # Extract first singular vectors (MCA-1 patterns)
    sst_pattern_1d = U_s[:, 0]
    wind_pattern_1d = V_s[0, :]

    # Obtain time series by projecting original data onto these vectors
    pc_sst_arr = np.dot(sst_clean, sst_pattern_1d)
    pc_wind_arr = np.dot(wind_matrix, wind_pattern_1d)

    return {
        'sst_pattern_1d': sst_pattern_1d,
        'wind_pattern_1d': wind_pattern_1d,
        'pc_sst_arr': pc_sst_arr,
        'pc_wind_arr': pc_wind_arr,
        'explained_var': explained_var
    }


def reconstruct_mca_patterns(mca_results, data_info):
    """
    Reconstruct spatial patterns from MCA results.

    Args:
        mca_results (dict): Output from compute_mca_decomposition
        data_info (dict): Output from prepare_mca_data

    Returns:
        tuple: (sst_pattern_da, wind_u_pattern_da, wind_v_pattern_da, pc_sst_std, pc_winds_std)
    """
    # Extract needed values
    sst_pattern_1d = mca_results['sst_pattern_1d']
    wind_pattern_1d = mca_results['wind_pattern_1d']
    pc_sst_arr = mca_results['pc_sst_arr']
    pc_wind_arr = mca_results['pc_wind_arr']

    valid_sst = data_info['valid_sst']
    valid_wind = data_info['valid_wind']
    weights = data_info['weights']
    coords = data_info['coords']
    dims = data_info['dims']

    nlat, nlon = dims['nlat'], dims['nlon']

    # Reshape SST pattern back to spatial field
    sst_pattern_full = np.full(nlat*nlon, np.nan)
    sst_pattern_full[valid_sst] = sst_pattern_1d
    sst_pattern_2d = sst_pattern_full.reshape(nlat, nlon)

    # Create SST DataArray with unweighting and PC std scaling
    sst_pattern_da = xr.DataArray(
        (sst_pattern_2d / weights.data) * pc_sst_arr.std(),
        dims=['latitude','longitude'],
        coords={'latitude': coords['latitude'], 'longitude': coords['longitude']}
    )

    # Separate combined wind pattern into U and V components
    n_wind = np.count_nonzero(valid_wind)
    wind_u_1d = wind_pattern_1d[:n_wind]
    wind_v_1d = wind_pattern_1d[n_wind:]

    # Fill to 2D
    u_full = np.full(nlat*nlon, np.nan)
    v_full = np.full(nlat*nlon, np.nan)
    u_full[valid_wind] = wind_u_1d
    v_full[valid_wind] = wind_v_1d

    wind_u_2d = u_full.reshape(nlat, nlon)
    wind_v_2d = v_full.reshape(nlat, nlon)

    wind_u_pattern_da = xr.DataArray(
        (wind_u_2d / weights.data) * pc_wind_arr.std(),
        dims=['latitude','longitude'],
        coords={'latitude': coords['u_latitude'], 'longitude': coords['u_longitude']}
    )
    wind_v_pattern_da = xr.DataArray(
        (wind_v_2d / weights.data) * pc_wind_arr.std(),
        dims=['latitude','longitude'],
        coords={'latitude': coords['u_latitude'], 'longitude': coords['u_longitude']}
    )

    # Convert PCs to DataArray and standardize
    pc_sst_da = xr.DataArray(pc_sst_arr, dims=['time'], coords={'time': coords['time']})
    pc_winds_da = xr.DataArray(pc_wind_arr, dims=['time'], coords={'time': coords['time']})

    pc_sst_std   = standardize_seasonal(pc_sst_da)
    pc_winds_std = standardize_seasonal(pc_winds_da)

    return sst_pattern_da, wind_u_pattern_da, wind_v_pattern_da, pc_sst_std, pc_winds_std


def perform_mca(sst, u, v, lat_range, lon_range, cti):
    """
    Perform MCA on SST and combined wind (U,V) fields over the given domain,
    after subtracting CTI from each field.

    This is the main orchestration function that calls helper functions for:
    1. Data preparation (subsetting, CTI removal, standardization, weighting)
    2. MCA decomposition (SVD on cross-covariance matrix)
    3. Pattern reconstruction (reshape to spatial fields)

    Args:
        sst (xarray.DataArray): Sea surface temperature anomalies [time, lat, lon]
        u   (xarray.DataArray): Zonal wind anomalies [time, lat, lon]
        v   (xarray.DataArray): Meridional wind anomalies [time, lat, lon]
        lat_range (tuple): (lat_min, lat_max)
        lon_range (tuple): (lon_min, lon_max)
        cti (xarray.DataArray): Cold Tongue Index time series [time]

    Returns:
        sst_pattern_da  (xarray.DataArray): SST spatial pattern of MCA-1
        wind_u_pattern_da (xarray.DataArray): U wind spatial pattern of MCA-1
        wind_v_pattern_da (xarray.DataArray): V wind spatial pattern of MCA-1
        pc_sst_std      (xarray.DataArray): Time series (PC) for SST
        pc_winds_std    (xarray.DataArray): Time series (PC) for winds
        explained_var   (float): % variance explained by MCA-1
    """
    # Step 1: Prepare data
    data_info = prepare_mca_data(sst, u, v, lat_range, lon_range, cti)

    # Step 2: Compute MCA decomposition
    mca_results = compute_mca_decomposition(
        data_info['sst_clean'],
        data_info['wind_matrix'],
        data_info['dims']['ntime']
    )

    # Step 3: Reconstruct spatial patterns
    sst_pattern_da, wind_u_pattern_da, wind_v_pattern_da, pc_sst_std, pc_winds_std = \
        reconstruct_mca_patterns(mca_results, data_info)

    return (sst_pattern_da, wind_u_pattern_da, wind_v_pattern_da,
            pc_sst_std, pc_winds_std, mca_results['explained_var'])


def plot_mca_patterns(sst_pattern, u_pattern, v_pattern, title, save_path, explained_var):
    """Plot MCA spatial patterns with SST contours and wind vectors."""
    fig = plt.figure(figsize=(15,9))
    proj = ccrs.PlateCarree()
    proj_mod = ccrs.PlateCarree(central_longitude=180)

    ax = plt.subplot(1,1,1, projection=proj_mod)
    ax.set_extent([sst_pattern.longitude.min(), sst_pattern.longitude.max(),
                   sst_pattern.latitude.min(), sst_pattern.latitude.max()], crs=proj)
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Add cyclic point to eliminate dateline gap
    sst_pattern_cyclic = add_cyclic_point_xr(sst_pattern)

    # SST pattern
    clevs, _ = get_symmetric_levels(sst_pattern, spacing=0.1)
    c = sst_pattern_cyclic.plot.contourf(
        ax=ax, transform=proj,
        levels=clevs, cmap='RdBu_r',
        add_colorbar=False
    )
    cb = plt.colorbar(c, ax=ax, orientation='horizontal', label='SST pattern [C]', shrink=0.4)

    # Wind vectors - add cyclic points to vectors as well
    u_pattern_cyclic = add_cyclic_point_xr(u_pattern)
    v_pattern_cyclic = add_cyclic_point_xr(v_pattern)
    skip = 2
    ax.quiver(
        u_pattern_cyclic.longitude.values[::skip], u_pattern_cyclic.latitude.values[::skip],
        u_pattern_cyclic.values[::skip, ::skip], v_pattern_cyclic.values[::skip, ::skip],
        transform=proj, scale=50
    )

    # Include explained variance in the title
    ax.set_title(f"{title}\nExplained Variance: {explained_var:.1f}%")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function for NPMM and SPMM computation."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    ####################################
    # NPMM and SPMM Computation
    ####################################

    SST = xr.open_dataset(os.path.join(DATA_DIR, 'SST_seas_1deg.nc')).sst.sortby('latitude', ascending=True)
    U_10m = xr.open_dataset(os.path.join(DATA_DIR, 'U_10m_seas.nc')).u10.sortby('latitude', ascending=True)
    V_10m = xr.open_dataset(os.path.join(DATA_DIR, 'V_10m_seas.nc')).v10.sortby('latitude', ascending=True)

    SST = SST.assign_coords(longitude=((SST.longitude + 360) % 360)).sortby('longitude')
    U_10m = U_10m.assign_coords(longitude=((U_10m.longitude + 360) % 360)).sortby('longitude')
    V_10m = V_10m.assign_coords(longitude=((V_10m.longitude + 360) % 360)).sortby('longitude')

    # Compute seasonal anomalies
    SST_anom = compute_seasonal_anomalies(SST, '1945-06-01', '2024-02-01')
    U_10m_anom = compute_seasonal_anomalies(U_10m, '1945-06-01', '2024-02-01')
    V_10m_anom = compute_seasonal_anomalies(V_10m, '1945-06-01', '2024-02-01')

    # Compute CTI
    cti = compute_cti(SST)

    # Define domains
    # NPMM: 21S-32N, 175E-95W
    npmm_lat_range = (-21, 32)
    npmm_lon_range = (175, 265)  # 95W = 265E

    # SPMM: 35-10S, 180-70W
    spmm_lat_range = (-35, -10)
    spmm_lon_range = (180, 290)

    # Perform MCA for NPMM
    npmm_sst_pat, npmm_u_pat, npmm_v_pat, npmm_pc_sst, npmm_pc_winds, npmm_expl_var = perform_mca(
        SST_anom, U_10m_anom, V_10m_anom, npmm_lat_range, npmm_lon_range, cti
    )

    # Perform MCA for SPMM
    spmm_sst_pat, spmm_u_pat, spmm_v_pat, spmm_pc_sst, spmm_pc_winds, spmm_expl_var = perform_mca(
        SST_anom, U_10m_anom, V_10m_anom, spmm_lat_range, spmm_lon_range, cti
    )

    # Save PCs
    npmm_pc_sst.to_netcdf(os.path.join(SAVE_DIR, 'NPMM_SST_st_ts.nc'))
    npmm_pc_winds.to_netcdf(os.path.join(SAVE_DIR, 'NPMM_wind_st_ts.nc'))

    spmm_pc_sst.to_netcdf(os.path.join(SAVE_DIR, 'SPMM_SST_st_ts.nc'))
    spmm_pc_winds.to_netcdf(os.path.join(SAVE_DIR, 'SPMM_wind_st_ts.nc'))

    # Plot spatial patterns
    plot_mca_patterns(npmm_sst_pat, npmm_u_pat, npmm_v_pat, 'NPMM Spatial Pattern', os.path.join(PLOT_DIR, 'NPMM_pattern.png'), npmm_expl_var)
    plot_mca_patterns(spmm_sst_pat, spmm_u_pat, spmm_v_pat, 'SPMM Spatial Pattern', os.path.join(PLOT_DIR, 'SPMM_pattern.png'), spmm_expl_var)

    print("NPMM and SPMM processing completed successfully!")


if __name__ == "__main__":
    main()
