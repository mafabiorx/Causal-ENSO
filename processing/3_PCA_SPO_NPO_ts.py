"""
Pacific Oscillation Indices: SPO and NPO Mode Computation via PCA

Core Functionality:
- Extract South Pacific Oscillation (SPO) from surface pressure EOF analysis
- Compute North Pacific Oscillation (NPO) as 2nd EOF of North Pacific sea level pressure
- Apply latitude weighting and seasonal standardization for climate index construction
- Generate spatial pattern visualizations and standardized time series

Output: SPO and NPO standardized time series and spatial pattern plots for climate analysis
"""

import numpy as np
import xarray as xr
import os
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from sklearn.decomposition import PCA
# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))
from utils.TBI_functions import compute_seasonal_anomalies, standardize_seasonal, lat_lon_weighting, get_symmetric_levels, add_cyclic_point_xr
from utils.paths import get_data_path, get_results_path

# Set the paths
DATA_DIR = get_data_path('seasonal/', data_type="raw")
SAVE_DIR = get_data_path('time_series/')
PLOT_DIR = get_results_path('main_climate_modes/')


def compute_spo_mode(sp_anom, weights_array):
    """
    Compute SPO mode using EOF analysis of surface pressure anomalies.

    Args:
        sp_anom (xarray.DataArray): Surface pressure anomalies
        weights_array (xarray.DataArray): Latitude weights

    Returns:
        eof_da (xarray.DataArray): EOF pattern [lat, lon]
        pc_da (xarray.DataArray): Principal Component time series [time]
        explained_variance_ratio (float): Fraction of variance explained
    """
    # Reshape data for PCA
    nlat, nlon = len(sp_anom.latitude), len(sp_anom.longitude)
    ntime = len(sp_anom.time)
    X = sp_anom.values.reshape(ntime, nlat * nlon)

    # Perform PCA
    pca = PCA(n_components=1)
    pcs = pca.fit_transform(X)

    # Get EOF (spatial pattern)
    eof = pca.components_.reshape(nlat, nlon)

    # Apply sign convention
    if eof.mean() < 0:
        eof *= -1
        pcs *= -1

    # Unweight EOF and multiply by the standard deviation of the PC
    eof = (eof / weights_array.data) * pcs.std()

    # Create xarray DataArrays for EOF and PC
    eof_da = xr.DataArray(
        eof,
        dims=['latitude', 'longitude'],
        coords={
            'latitude': sp_anom.latitude,
            'longitude': sp_anom.longitude
        }
    )

    pc_da = xr.DataArray(
        pcs.flatten(),
        dims=['time'],
        coords={'time': sp_anom.time}
    )

    # Standardize the PC time series
    pc_st = standardize_seasonal(pc_da)

    # Save SPO time series
    pc_st.to_netcdf(os.path.join(SAVE_DIR, 'SPO_st_ts.nc'))

    return eof_da, pc_da, pca.explained_variance_ratio_[0]


def plot_spo_pattern(eof_da, explained_variance, save_dir):
    """
    Plot SPO spatial pattern.

    Args:
        eof_da (xarray.DataArray): EOF pattern
        explained_variance (float): Explained variance ratio
        save_dir (str): Directory to save plot
    """
    fig = plt.figure(figsize=(10, 6))
    proj = ccrs.PlateCarree()

    ax = plt.subplot(1, 1, 1, projection=proj)
    ax.set_extent([-160, -70, -45, -10], crs=proj)
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    clevs, _ = get_symmetric_levels(eof_da, spacing=0.1)
    c = eof_da.plot.contourf(
        ax=ax, transform=proj,
        levels=clevs, cmap='RdBu_r',
        add_colorbar=False
    )
    cb = plt.colorbar(c, ax=ax, orientation='horizontal',
                     label='Surface Pressure pattern [hPa]', shrink=0.5)

    eof_da.plot.contour(
        ax=ax, transform=proj,
        levels=clevs[::2], colors='k',
        linewidths=0.5, alpha=0.3
    )

    ax.set_title(f"SPO Spatial Pattern\nExplained Variance: {explained_variance*100:.1f}%")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'SPO_pattern.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_spo_timeseries(pc_da, save_dir):
    """
    Plot SPO time series.

    Args:
        pc_da (xarray.DataArray): Principal Component time series
        save_dir (str): Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(pc_da.time, pc_da, color='blue')
    plt.xlabel('Time')
    plt.ylabel('SPO')
    plt.title('Standardized Principal Component of Surface Pressure')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'SPO_time_series.png'), dpi=300, bbox_inches='tight')
    plt.close()


def compute_npo_mode(slp_anom, weights_array):
    """
    Compute NPO mode using EOF analysis of sea level pressure anomalies.
    NPO is defined as the 2nd EOF of SLP in the North Pacific region.

    Args:
        slp_anom (xarray.DataArray): Sea level pressure anomalies
        weights_array (xarray.DataArray): Latitude weights

    Returns:
        eof_da (xarray.DataArray): EOF pattern [lat, lon]
        pc_da (xarray.DataArray): Principal Component time series [time]
        explained_variance_ratio (float): Fraction of variance explained by 2nd EOF
    """
    # Reshape data for PCA
    nlat, nlon = len(slp_anom.latitude), len(slp_anom.longitude)
    ntime = len(slp_anom.time)
    X = slp_anom.values.reshape(ntime, nlat * nlon)

    # Perform PCA
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X)

    # Get 2nd EOF (NPO spatial pattern)
    eof = pca.components_[1].reshape(nlat, nlon)
    pc = pcs[:, 1]

    # Apply sign convention
    if eof.mean() < 0:
        eof *= -1
        pc *= -1

    # Unweight EOF and multiply by the standard deviation of the PC
    eof = (eof / weights_array.data) * pc.std()

    # Create xarray DataArrays for EOF and PC
    eof_da = xr.DataArray(
        eof,
        dims=['latitude', 'longitude'],
        coords={
            'latitude': slp_anom.latitude,
            'longitude': slp_anom.longitude
        }
    )

    pc_da = xr.DataArray(
        pc,
        dims=['time'],
        coords={'time': slp_anom.time}
    )

    pc_st = standardize_seasonal(pc_da)
    pc_st.to_netcdf(os.path.join(SAVE_DIR, 'NPO_st_ts.nc'))

    return eof_da, pc_da, pca.explained_variance_ratio_[1]  # 2nd mode variance


def plot_npo_pattern(eof_da, explained_variance, save_dir):
    """
    Plot NPO spatial pattern.

    Args:
        eof_da (xarray.DataArray): EOF pattern
        explained_variance (float): Explained variance ratio
        save_dir (str): Directory to save plot
    """
    proj = ccrs.PlateCarree()
    proj_mod = ccrs.PlateCarree(central_longitude=180)

    fig = plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1, projection=proj_mod)
    ax.set_extent([120, 240, 20, 85], crs=proj)
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Add cyclic point to eliminate dateline gap
    eof_da_cyclic = add_cyclic_point_xr(eof_da)

    clevs, levs_cont = get_symmetric_levels(eof_da, spacing=0.1)
    c = eof_da_cyclic.plot.contourf(
        ax=ax, transform=proj,
        levels=clevs, cmap='RdBu_r',
        add_colorbar=False
    )
    plt.colorbar(c, ax=ax, orientation='horizontal',
                 label='Sea Level Pressure pattern [hPa]', shrink=0.5)

    eof_da_cyclic.plot.contour(
        ax=ax, transform=proj,
        levels=levs_cont, colors='darkgray',
        linewidths=0.5, alpha=0.7
    )

    ax.set_title(f"NPO Spatial Pattern\nExplained Variance: {explained_variance*100:.1f}%")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'NPO_pattern.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_npo_timeseries(pc_da, save_dir):
    """
    Plot NPO time series.

    Args:
        pc_da (xarray.DataArray): Principal Component time series
        save_dir (str): Directory to save plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(pc_da.time, pc_da, color='red')
    plt.xlabel('Time')
    plt.ylabel('NPO')
    plt.title('North Pacific Oscillation')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'NPO_time_series.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function for SPO and NPO mode computation."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    ####################################
    # SPO Mode Computation
    ####################################

    # Load surface pressure data with proper unit conversion
    sp_data = xr.open_dataset(os.path.join(DATA_DIR, 'surf_pres_seas.nc')).sp.sortby('latitude', ascending=True)
    sp_data = sp_data * 0.01  # Convert to hPa

    # Select South Pacific region
    sp_data = sp_data.sel(latitude=slice(-45, -10), longitude=slice(-160, -70))

    # Compute seasonal anomalies
    sp_anom = compute_seasonal_anomalies(sp_data, '1945-06-01', '2024-02-01')
    sp_anom, weights_array = lat_lon_weighting(sp_anom, 'latitude')

    # Compute SPO mode
    eof_pattern, pc_timeseries, explained_variance = compute_spo_mode(sp_anom, weights_array)

    # Print variance explained
    print(f"\nSPO Mode - Variance explained: {explained_variance*100:.1f}%")

    # Plot SPO pattern
    plot_spo_pattern(eof_pattern, explained_variance, PLOT_DIR)

    # Plot SPO standardized time series
    spo_st = xr.open_dataarray(os.path.join(SAVE_DIR, 'SPO_st_ts.nc'))
    plot_spo_timeseries(spo_st, PLOT_DIR)

    print("SPO processing completed successfully!")

    ####################################
    # NPO Mode Computation
    ####################################

    # Load surface pressure data for North Pacific region
    slp_data = xr.open_dataset(os.path.join(DATA_DIR, 'surf_pres_seas.nc')).sp.sortby('latitude', ascending=True)
    slp_data = slp_data * 0.01  # Convert to hPa

    # Select North Pacific region (20-85N, 120E-120W) - handles dateline crossing
    slp_data_west = slp_data.sel(latitude=slice(20, 85), longitude=slice(120, 180))
    slp_data_east = slp_data.sel(latitude=slice(20, 85), longitude=slice(-180, -120))
    slp_data = xr.concat([slp_data_east, slp_data_west], dim='longitude')

    # Compute seasonal anomalies
    slp_anom = compute_seasonal_anomalies(slp_data, '1945-06-01', '2024-02-01')
    slp_anom, slp_weights_array = lat_lon_weighting(slp_anom, 'latitude')

    # Compute NPO mode
    npo_eof_pattern, npo_pc_timeseries, npo_explained_variance = compute_npo_mode(slp_anom, slp_weights_array)

    # Print variance explained
    print(f"\nNPO Mode - Variance explained: {npo_explained_variance*100:.1f}%")

    # Plot NPO pattern
    plot_npo_pattern(npo_eof_pattern, npo_explained_variance, PLOT_DIR)

    # Plot NPO time series (standardized version)
    npo_st = xr.open_dataarray(os.path.join(SAVE_DIR, 'NPO_st_ts.nc'))
    plot_npo_timeseries(npo_st, PLOT_DIR)

    print("NPO processing completed successfully!")
    print("All SPO and NPO processing completed!")


if __name__ == "__main__":
    main()
