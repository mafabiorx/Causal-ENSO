"""
ENSO Diversity Index Computation: Eastern Pacific (E) and Central Pacific (C) Indices

Core Functionality:
- Apply PCA to tropical Pacific SST anomalies to extract first two modes
- Construct E and C indices using mathematical combination (PC1±PC2)/√2
- Generate spatial patterns for Eastern Pacific and Central Pacific ENSO types
- Perform seasonal standardization for subsequent causal analysis

Output: Standardized E/C indices and spatial patterns for ENSO diversity analysis
"""

# Import the tools
import numpy as np
import xarray as xr
import pandas as pd
import cartopy.crs as ccrs
from shapely.geometry import Point, Polygon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
plt.rcParams['figure.figsize'] = (15, 10)
import warnings
import os
import sys
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
plt.rcParams['font.size'] = 14
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))
from utils.TBI_functions import compute_seasonal_anomalies, lat_lon_weighting, standardize_seasonal, get_symmetric_levels, add_cyclic_point_xr
from utils.paths import get_data_path, get_results_path

# Set up directories
DATA_DIR = get_data_path('seasonal/', data_type="raw")
SAVE_DIR = get_data_path('time_series/', data_type="processed")
PLOT_DIR = get_results_path('main_climate_modes/', result_type="figures")


def main():
    """Main function for E and C indices computation."""
    ## Equatorial SST
    SST = xr.open_dataset(os.path.join(DATA_DIR, 'SST_seas.nc')).sst.sortby('latitude', ascending=True)
    SST_seas = compute_seasonal_anomalies(SST, '1945-06', '2024-02').drop_vars('season')

    # Select the region in the tropical Pacific
    sst_region1 = SST_seas.sel(latitude=slice(-5, 5), longitude=slice(170, 180))
    sst_region2 = SST_seas.sel(latitude=slice(-5, 5), longitude=slice(-180, -80))
    SST_eq = xr.concat([sst_region1, sst_region2], dim='longitude')
    SST_eq, weights = lat_lon_weighting(SST_eq, 'latitude')

    # Reshape the data into a 2D array (time, space)
    n_time, n_lat, n_lon = SST_eq.shape
    SST_eq_2d = SST_eq.values.reshape((n_time, -1))

    # Build a "valid_columns" mask: keep only columns with no NaNs
    valid_mask = ~np.isnan(SST_eq_2d).any(axis=0)   # shape: (n_points,)

    # Subset the data to those valid columns
    SST_eq_2d_no_nans = SST_eq_2d[:, valid_mask]

    ## Perform PCA on only the valid ocean points ##
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(SST_eq_2d_no_nans)
    PC1, PC2 = pcs[:, 0], pcs[:, 1]
    PC1_std = np.std(PC1)
    PC2_std = np.std(PC2)
    explained_variance = pca.explained_variance_ratio_  * 100
    EOFs = pca.components_

    print(' The mean of PC1 is ', np.mean(PC1), ' and the mean of PC2 is ', np.mean(PC2))
    print(' The standard deviation of PC1 is ', PC1_std, ' and the standard deviation of PC2 is ', PC2_std)
    print('The variance explained by PC1 is ', explained_variance[0], ' and the variance explained by PC2 is ', explained_variance[1])

    ## Compute the C-index and E-index ##
    C_index = (PC1 - PC2) / np.sqrt(2)
    E_index = (PC1 + PC2) / np.sqrt(2)

    # Converting to xarray DataArray:
    C_index = xr.DataArray(C_index, dims=('time'), coords={'time': SST_eq.time})
    E_index = xr.DataArray(E_index, dims=('time'), coords={'time': SST_eq.time})

    C_index_st = standardize_seasonal(C_index)
    E_index_st = standardize_seasonal(E_index)


    # Assign names to the DataArray objects
    C_index_st.name = 'C_index'
    E_index_st.name = 'E_index'

    C_index_st.to_netcdf(os.path.join(SAVE_DIR, 'C_index_st_ts.nc'))
    E_index_st.to_netcdf(os.path.join(SAVE_DIR, 'E_index_st_ts.nc'))

    # Create a mask from the original SST anomalies data
    nan_mask = np.isnan(SST_eq)

    # Reshape the EOFs to the shape of the SST anomalies in the selected region
    EOF1_reshaped = np.full((SST_eq.shape[1], SST_eq.shape[2]), np.nan)
    EOF2_reshaped = np.full((SST_eq.shape[1], SST_eq.shape[2]), np.nan)

    EOF1_reshaped[~nan_mask.isel(time=0)] = EOFs[0]
    EOF2_reshaped[~nan_mask.isel(time=0)] = EOFs[1]

    # Convert reshaped EOFs to xarray DataArray
    EOF1 = xr.DataArray(
        (EOF1_reshaped/ weights.data)*PC1_std,
        coords=[SST_eq.latitude, SST_eq.longitude],
        dims=["latitude", "longitude"],
    )

    EOF1 = EOF1.where((SST_eq.isel(time=0).notnull()).drop_vars('time'))

    EOF2 = xr.DataArray(
        (EOF2_reshaped/ weights.data)*PC2_std,
        coords=[SST_eq.latitude, SST_eq.longitude],
        dims=["latitude", "longitude"],
    )

    EOF2 = EOF2.where((SST_eq.isel(time=0).notnull()).drop_vars('time'))

    ## Plot the spatial patterns of EOF1 and EOF2 ##
    proj = ccrs.PlateCarree()
    proj_mod = ccrs.PlateCarree(central_longitude=180)
    fig, (ax1, ax2) = plt.subplots(2, 1, subplot_kw={'projection': proj_mod}, figsize=(17, 5))

    # Add cyclic points to eliminate dateline gap
    EOF1_cyclic = add_cyclic_point_xr(EOF1)
    EOF2_cyclic = add_cyclic_point_xr(EOF2)

    clevs_1, levs_cont_1 = get_symmetric_levels(EOF1, 0.1)
    ax1.set_title(f'EOF1 ({explained_variance[0]:.1f}% variance explained)')
    im1 = ax1.contourf(EOF1_cyclic.longitude, EOF1_cyclic.latitude, EOF1_cyclic, transform=proj, levels = clevs_1, cmap='RdBu_r')
    cs1 = ax1.contour(EOF1_cyclic.longitude, EOF1_cyclic.latitude, EOF1_cyclic.values, transform=proj, levels=levs_cont_1, colors='darkgray', linewidths=0.5)
    plt.clabel(cs1, levels=levs_cont_1, fmt='%1.1f', fontsize='small', colors='dimgrey', inline=True, inline_spacing=4)
    ax1.coastlines()
    ax1.gridlines(draw_labels=True)
    fig.colorbar(im1, ax=ax1, shrink=0.6, label='[K]', pad=0.02)
    ax1.set_extent([170, 280, -5, 5], crs=proj)

    clevs_2, levs_cont_2 = get_symmetric_levels(EOF2, 0.1)
    ax2.set_title(f'EOF2 ({explained_variance[1]:.1f}% variance explained)')
    im2 = ax2.contourf(EOF2_cyclic.longitude, EOF2_cyclic.latitude, EOF2_cyclic, transform=proj, levels = clevs_2, cmap='RdBu_r')
    cs2 = ax2.contour(EOF2_cyclic.longitude, EOF2_cyclic.latitude, EOF2_cyclic.values, transform=proj, levels=levs_cont_2, colors='darkgray', linewidths=0.5)
    plt.clabel(cs2, levels=levs_cont_2, fmt='%1.1f', fontsize='small', colors='dimgrey', inline=True, inline_spacing=4)
    ax2.coastlines()
    ax2.gridlines(draw_labels=True)
    fig.colorbar(im2, ax=ax2, shrink=0.6, label='[K]', pad=0.02)
    ax2.set_extent([170, 280, -5, 5], crs=proj)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'EOFs_SST_eq_40_24.png'), bbox_inches='tight', pad_inches = 0.1, dpi = 250)

    ## Spatial patterns associated with E and C indices ##
    Y_valid = SST_eq_2d_no_nans   # (time, valid_points)

    # Fit linear regression for E_index
    regressor = LinearRegression()
    regressor.fit(E_index_st.values.reshape(-1, 1), Y_valid)

    E_pattern_valid = regressor.coef_
    E_pattern_valid = E_pattern_valid.ravel()
    E_pattern_full = np.full(n_lat*n_lon, np.nan)
    E_pattern_full[valid_mask] = E_pattern_valid
    E_pattern_2d = E_pattern_full.reshape(n_lat, n_lon)

    E_pattern = xr.DataArray(
        E_pattern_2d,
        coords=[SST_eq.latitude, SST_eq.longitude],
        dims=["latitude", "longitude"]
    )
    E_pattern = E_pattern.where((SST_eq.isel(time=0).notnull()).drop_vars('time'))

    # Fit linear regression for C_index
    regressor.fit(C_index_st.values.reshape(-1, 1), Y_valid)

    C_pattern_valid = regressor.coef_
    C_pattern_valid = C_pattern_valid.ravel()
    C_pattern_full = np.full(n_lat * n_lon, np.nan)
    C_pattern_full[valid_mask] = C_pattern_valid
    C_pattern_2d = C_pattern_full.reshape(n_lat, n_lon)

    C_pattern = xr.DataArray(
        C_pattern_2d,
        coords=[SST_eq.latitude, SST_eq.longitude],
        dims=["latitude", "longitude"],
    )

    C_pattern = C_pattern.where((SST_eq.isel(time=0).notnull()).drop_vars('time'))

    # Plot the spatial patterns of E-index and C-index
    proj = ccrs.PlateCarree()
    proj_mod = ccrs.PlateCarree(central_longitude=180)

    fig, (ax1, ax2) = plt.subplots(2, 1, subplot_kw={'projection': proj_mod}, figsize=(17, 5))

    # Add cyclic points to eliminate dateline gap
    E_pattern_cyclic = add_cyclic_point_xr(E_pattern)
    C_pattern_cyclic = add_cyclic_point_xr(C_pattern)

    # Generate levels for E-index
    clevs_E, levs_cont_E = get_symmetric_levels(E_pattern, spacing=0.1)

    ax1.set_title('E-index', fontsize=20)
    im1 = ax1.contourf(E_pattern_cyclic.longitude, E_pattern_cyclic.latitude, E_pattern_cyclic, transform=proj, levels=clevs_E, cmap='RdBu_r')
    cs1 = ax1.contour(E_pattern_cyclic.longitude, E_pattern_cyclic.latitude, E_pattern_cyclic.values, transform=proj, levels=levs_cont_E, colors='darkgray', linewidths=0.5)
    plt.clabel(cs1, levels=levs_cont_E, fmt='%1.1f', fontsize='small', colors='dimgrey', inline=True, inline_spacing=4)
    ax1.coastlines()
    ax1.gridlines(draw_labels=True)
    fig.colorbar(im1, ax=ax1, shrink=0.6, label='[K]', pad=0.02)
    ax1.set_extent([170, 280, -5, 5], crs=proj)

    # Generate levels for C-index
    clevs_C, levs_cont_C = get_symmetric_levels(C_pattern, spacing=0.1)

    ax2.set_title('C-index', fontsize=20)
    im2 = ax2.contourf(C_pattern_cyclic.longitude, C_pattern_cyclic.latitude, C_pattern_cyclic, transform=proj, levels=clevs_C, cmap='RdBu_r')
    cs2 = ax2.contour(C_pattern_cyclic.longitude, C_pattern_cyclic.latitude, C_pattern_cyclic.values, transform=proj, levels=levs_cont_C, colors='darkgray', linewidths=0.5)
    plt.clabel(cs2, levels=levs_cont_C, fmt='%1.1f', fontsize='small', colors='dimgrey', inline=True, inline_spacing=4)
    ax2.coastlines()
    ax2.gridlines(draw_labels=True)
    fig.colorbar(im2, ax=ax2, shrink=0.6, label='[K]', pad=0.02)
    ax2.set_extent([170, 280, -5, 5], crs=proj)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, 'C_E_ind_40_24.png'), bbox_inches='tight', pad_inches=0.1, dpi=250)


if __name__ == "__main__":
    main()
