"""
Pacific-South American (PSA) Mode Computation via EOF Analysis

Core Functionality:
- Compute PSA1 and PSA2 modes using EOF analysis of 500 hPa geopotential height
- PSA1 corresponds to EOF2, PSA2 corresponds to EOF3 of Southern Hemisphere Z500
- Apply latitude weighting and seasonal standardization for climate index construction
- Generate spatial pattern visualizations for each mode

Output: PSA1 and PSA2 standardized time series and spatial pattern plots
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
from utils.TBI_functions import total_average, compute_seasonal_anomalies, standardize_seasonal, lat_lon_weighting, get_symmetric_levels, add_cyclic_point_xr
from utils.paths import get_data_path, get_results_path

# Set the paths
DATA_DIR = get_data_path('1_deg_seasonal/', data_type="interim")
SAVE_DIR = get_data_path('time_series/')
PLOT_DIR = get_results_path('main_climate_modes/')

# Physical constants
g = 9.81  # Acceleration due to gravity (m/s²)


def compute_psa_modes(Z_500_anom, weights_array):
    """
    Compute PSA modes using EOF analysis of Z500 anomalies.

    Args:
        Z_500_anom (xarray.DataArray): Geopotential height anomalies at 500 hPa
        weights_array (xarray.DataArray): Latitude weights

    Returns:
        eof_da (xarray.DataArray): EOF patterns [3, lat, lon]
        pc_da (xarray.DataArray): Principal Component time series [time, 3]
        explained_variance_ratio (numpy.ndarray): Fraction of variance explained by each mode
    """
    # Reshape data for PCA
    nlat, nlon = len(Z_500_anom.latitude), len(Z_500_anom.longitude)
    ntime = len(Z_500_anom.time)
    X = Z_500_anom.values.reshape(ntime, nlat * nlon)

    # Perform PCA
    pca = PCA(n_components=3)  # We only need first 3 EOFs
    pcs = pca.fit_transform(X)

    # Get EOFs (spatial patterns)
    eofs = pca.components_.reshape(3, nlat, nlon)

    for i in range(3):
        if eofs[i].mean() < 0:
            eofs[i] *= -1
            pcs[:, i] *= -1

    # Unweight EOFs for each mode and multiply by the standard deviation of the corresponding PC
    for i in range(3):
        eofs[i] = (eofs[i] / weights_array.data) * pcs[:, i].std()

    # Create xarray DataArrays for EOFs and PCs
    eof_da = xr.DataArray(
        eofs,
        dims=['mode', 'latitude', 'longitude'],
        coords={
            'mode': ['EOF1', 'EOF2', 'EOF3'],
            'latitude': Z_500_anom.latitude,
            'longitude': Z_500_anom.longitude
        }
    )

    pc_da = xr.DataArray(
        pcs,
        dims=['time', 'mode'],
        coords={
            'time': Z_500_anom.time,
            'mode': ['PC1', 'PC2', 'PC3']
        }
    )

    #standardize the PC time series using the standardize_seasonal function
    pc_st = standardize_seasonal(pc_da)


    # Extract PSA1 (PC2) and save
    psa1 = pc_st.sel(mode='PC2').drop('mode')
    psa1.to_netcdf(os.path.join(SAVE_DIR, 'PSA1_st_ts.nc'))

    # Extract PSA2 (PC3) and save
    psa2 = pc_st.sel(mode='PC3').drop('mode')
    psa2.to_netcdf(os.path.join(SAVE_DIR, 'PSA2_st_ts.nc'))

    return eof_da, pc_da, pca.explained_variance_ratio_


def plot_psa_patterns(eof_da, explained_variance, save_dir):
    """
    Plot PSA spatial patterns.

    Args:
        eof_da (xarray.DataArray): EOF patterns
        explained_variance (numpy.ndarray): Explained variance ratios
        save_dir (str): Directory to save plots
    """
    # PSA modes correspond to EOF2 and EOF3
    modes = ['PSA1', 'PSA2']
    patterns = [eof_da[1], eof_da[2]]  # EOF2 and EOF3
    var_explained = explained_variance[[1, 2]]  # Variance for EOF2 and EOF3

    for idx, (mode, pattern, var) in enumerate(zip(modes, patterns, var_explained)):
        # Create individual figure for each mode to match NPMM/SPMM style
        fig = plt.figure(figsize=(15, 9))
        proj = ccrs.PlateCarree()
        proj_mod = ccrs.PlateCarree(central_longitude=180)

        ax = plt.subplot(1, 1, 1, projection=proj_mod)
        ax.set_extent([-180, 180, -70, -10], crs=proj)
        ax.coastlines()
        ax.gridlines(draw_labels=True)

        # Add cyclic point to eliminate dateline gap
        pattern_cyclic = add_cyclic_point_xr(pattern)

        # Plot data
        clevs, _ = get_symmetric_levels(pattern, spacing=5)
        c = pattern_cyclic.plot.contourf(
            ax=ax, transform=proj,
            levels=clevs, cmap='RdBu_r',
            add_colorbar=False
        )
        cb = plt.colorbar(c, ax=ax, orientation='horizontal',
                         label='Z500 pattern [m]', shrink=0.4)

        # Add contour lines
        pattern_cyclic.plot.contour(
            ax=ax, transform=proj,
            levels=clevs[::2], colors='k',
            linewidths=0.5, alpha=0.3
        )

        # Add title with explained variance
        ax.set_title(f"{mode} Spatial Pattern\nExplained Variance: {var*100:.1f}%")

        # Save individual plots
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{mode}_pattern.png'), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main function for PSA mode computation."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    ####################################
    # PSA Modes Computation
    ####################################

    Z_500 = xr.open_dataset(os.path.join(DATA_DIR, 'Z_500_seas_1deg.nc')).z.sortby('latitude', ascending=True).isel(pressure_level=0).drop_vars('pressure_level')/g  # converted to height [m]

    # Convert longitude to 0-360 range to match repository standard
    Z_500 = Z_500.assign_coords(longitude=((Z_500.longitude + 360) % 360)).sortby('longitude')

    # Compute seasonal anomalies with proper date range
    Z_500_anom = compute_seasonal_anomalies(Z_500, '1945-06-01', '2024-02-01')
    Z_500_anom, weights_array = lat_lon_weighting(Z_500_anom, 'latitude')

    # Compute PSA modes
    eof_patterns, pc_timeseries, explained_variance = compute_psa_modes(Z_500_anom, weights_array)

    # Print variance explained
    print("\nVariance explained by each mode:")
    print(f"EOF1: {explained_variance[0]*100:.1f}%")
    print(f"PSA1 (EOF2): {explained_variance[1]*100:.1f}%")
    print(f"PSA2 (EOF3): {explained_variance[2]*100:.1f}%")

    # Plot PSA patterns
    plot_psa_patterns(eof_patterns, explained_variance, PLOT_DIR)

    print("PSA processing completed successfully!")


if __name__ == "__main__":
    main()
