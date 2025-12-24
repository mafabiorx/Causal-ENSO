"""
Creates and plots composite spatial anomaly fields based on extreme phases
(top/bottom N years or terciles) of selected precursor indices or sets of indices.

The script performs:
1.  For individual precursors:
    a. Identifies top N and bottom N years based on their specific active season.
    b. Converts these years to an "ENSO event reference year" (Y_event_ref), defined as
       the calendar year of the MAM(0) data point and the Jan/Feb data point of the DJF(0)
       precursor/ENSO peak period (e.g., for a DJF 1996/1997 peak, Y_event_ref is 1997).
    c. Composites target spatial anomaly fields for these Y_event_ref years using a
       standard sequence of lags relative to Y_event_ref (JJA(-1) to DJF(+1)).
    d. Performs t-tests for significance.
2.  For EP and CP precursor sets (stringent criterion):
    a. Identifies Y_event_ref years where all precursors in a set meet their
       (ENSO-sign-adjusted) tercile conditions for the seasons JJA(-1), SON(-1), DJF(0), MAM(0)
       relative to Y_event_ref.
    b. Composites target spatial fields using the same standard sequence of lags relative to Y_event_ref.
    c. Tests significance.
3.  For EP and CP precursor sets (3 of 4 criterion):
    a. Identifies Y_event_ref years where at least 3 out of 4 precursors in a set meet their
       (ENSO-sign-adjusted) tercile conditions and the remaining precursor is not in the 
       opposite tercile, for the seasons JJA(-1), SON(-1), DJF(0), MAM(0) relative to Y_event_ref.
    b. Composites target spatial fields using the same standard sequence of lags relative to Y_event_ref.
    c. Tests significance.
4.  Plots multi-panel composite maps (one figure per primary field & composite type)
5.  Saves consolidated NetCDF files corresponding to each multi-panel figure.
"""

import sys
import os
import logging
import argparse
from pathlib import Path
import functools
from typing import List, Tuple, Dict, Optional, Any
import concurrent.futures
import psutil
import numpy as np
import xarray as xr
import pandas as pd
from scipy.stats import ttest_ind, ttest_1samp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker
import matplotlib.collections
import matplotlib.contour
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import csv
import warnings
import gc

# --- Add project root to Python path ---
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.plotting_optimization import (
    setup_cartopy_warnings, save_figure_optimized, 
    add_plotting_arguments, apply_rasterization_settings, 
    clean_data_for_pdf, filter_seasons_to_plot,
    setup_single_panel_figure, create_clean_panel_title,
    create_descriptive_filename, get_contourf_kwargs, get_pcolormesh_kwargs
)

# Add immediately after imports
setup_cartopy_warnings()

try:
    from utils.paths import get_data_path, get_results_path
    from utils.TBI_functions import (
        load_era_field,
        compute_seasonal_anomalies,
        get_symmetric_levels_fixed_spacing,
        add_cyclic_point_xr,
        add_cyclic_point_to_vectors
    )
    from utils.composite_utils import (
        # 10 identical functions
        setup_logging,
        load_all_precursor_timeseries,
        get_active_season_precursor_series,
        select_extreme_years_for_precursor,
        select_target_data_for_compositing,
        get_precursor_ref_calendar_year_and_season,
        calculate_fieldwise_ttest_significance,
        get_processed_composite_winds,
        save_contributing_years_csv,
        available_workers,
        # Unified functions
        get_spatial_anomalies_field,
        save_consolidated_composites_netcdf,
        # New season selection function
        get_target_season_specs_for_precursor,
        # Shared constants
        SEASON_TO_MIDDLE_MONTH,
        PRECURSOR_TO_DEFINITION_SEASON_KEY_MAP,
        PRECURSOR_ENSO_SIGN_MAP,
        STANDARD_ENSO_CYCLE_PLOT_SEQUENCE
    )
except ImportError as e:
    print(f"Error importing from utils: {e}. Ensure TBI-codebase/src is in PYTHONPATH or script is run from root.")
    sys.exit(1)

# --- 0. CONFIGURATION ---
DEFAULT_N_YEARS_COMPOSITE = 10
DEFAULT_SIGNIFICANCE_ALPHA = 0.05
DEFAULT_CLIMATOLOGY_START_YEAR = '1945'
DEFAULT_CLIMATOLOGY_END_YEAR = '2024'
DEFAULT_ANALYSIS_START_DATE_STR = '1945-06-01'
DEFAULT_ANALYSIS_END_DATE_STR = '2024-02-29'

# Define latitude bands for vertical section averaging (matches 18_vertical_streamfunction_regrs_plots.py)
VERTICAL_SECTION_LAT_BANDS = [
    (-5, 5),      # Equatorial band
    (-15, -5),    # Northern tropics
    (-25, -15),   # Northern subtropics  
    (-35, -25)    # Northern extratropics
]

# Band names for directory structure
LAT_BAND_NAMES = {
    (-5, 5): "equatorial_5S-5N",
    (-15, -5): "northern_tropics_15S-5S",
    (-25, -15): "northern_subtropics_25S-15S",
    (-35, -25): "northern_extratropics_35S-25S"
}

DATA_DIR_TS = Path(get_data_path('time_series', data_type="processed"))
PRECURSOR_TS_FILEPATH = DATA_DIR_TS / 'PCMCI_data_ts_st.nc'
DATA_DIR_GRIDDED = Path(get_data_path('1_deg_seasonal/', data_type="interim"))
PLOT_DIR = Path(get_results_path('composites/'))
SAVE_DATA_DIR = Path(get_data_path('composites/'))

INDIVIDUAL_PRECURSOR_NAMES_PART1 = [
    'REOF SST JJA', 'MCA WAF-RWS SON', 'MCA RWS-WAF DJF', 'MCA RWS-prec MAM(E)',
    'MCA prec-RWS SON', 'MCA RWS-prec DJF', 'MCA RWS-prec MAM(C)'
]

EP_PRECURSORS_3OF4 = ['REOF SST JJA', 'MCA WAF-RWS SON', 'MCA RWS-WAF DJF', 'MCA RWS-prec MAM(E)']
CP_PRECURSORS_3OF4 = ['REOF SST JJA', 'MCA prec-RWS SON', 'MCA RWS-prec DJF', 'MCA RWS-prec MAM(C)']
PRECURSOR_SETS_3OF4_DICT = {'EP_set': EP_PRECURSORS_3OF4, 'CP_set': CP_PRECURSORS_3OF4}

TARGET_SPATIAL_FIELDS_CONFIG: Dict[str, Dict[str, Any]] = {
    'vertical_streamfunction': {
        'filename': 'vertical_streamfunction_seas_1deg.nc',
        'var_name_in_file': 'psi',
        'units': '$10^{11}$ kg s$^{-1}$',
        'scale': 1e-11,  # Divide by 1e11 to convert kg/s to 10^11 kg/s units
        'cmap': 'RdBu_r',  # Red positive (enhanced eastward flow aloft)
        'levels_spacing': 1,
        'contour_spacing_factor': 5,  # Contour lines every 5th level
        'primary_plot_field': True,
        'is_vertical_section': True,  # Flag for specialized handling
        'plot_significant_composite': True,
        'default_wind_overlays': [],
        'pressure_levels': [1000, 850, 700, 500, 300, 200],
        'log_scale_pressure': True,
        'include_topography': True,
        'topography_features': {  # Equatorial topographic features to annotate
            'Andes': {'lon': -75, 'search_range': 10, 'text_y': 400},
            'E.Africa': {'lon': 35, 'search_range': 15, 'text_y': 500},
            'Maritime\nContinent': {'lon': 120, 'search_range': 20, 'text_y': 600},
        }
    },
    'RWS_multi': {
        'filename': 'RWS_seas_1deg.nc',
        'var_name_in_file': 'RWS',
        'units': '$10^{-11}$ s$^{-2}$',
        'scale': 1e11,  # Convert to display units (multiply by 1e11)
        'cmap': 'custom_rws',  # Will create custom colormap
        'levels': np.arange(-20, 22, 2),  # Fixed levels from -20 to +20
        'contour_lines': False,  # No contour lines for RWS
        'primary_plot_field': True,
        'is_vertical_section': True,
        'plot_significant_composite': True,
        'pressure_levels': [500, 400, 300, 250, 200],  # RWS-specific levels
        'pressure_range': (500, 200),  # For axis limits
        'log_scale_pressure': True,
        'include_topography': False,  # No topography for RWS
        'walker_references': True
    },
}

for k, v_dict in TARGET_SPATIAL_FIELDS_CONFIG.items():
    v_dict['raw_data_path'] = DATA_DIR_GRIDDED / v_dict['filename']

# Surface elevation configuration
SURFACE_ELEV_FILE = get_data_path("additional/Surface_elevation_as_geopot.nc", data_type="raw")

# Physical constants for elevation conversion
G = 9.80665  # Standard gravity (m/s²)
P0 = 1013.25  # Sea level pressure (hPa)
L = 0.0065    # Temperature lapse rate (K/m)
T0 = 288.15   # Sea level temperature (K)
M = 0.0289644  # Molar mass of air (kg/mol)
R = 8.31447   # Universal gas constant J/(mol·K)

# --- 1. UTILITY FUNCTIONS ---

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Turn down third-party verbosity
for noisy_mod in ('matplotlib', 'matplotlib.font_manager', 'PIL', 'fontTools', 'cartopy', 'pyproj'):
    logging.getLogger(noisy_mod).setLevel(logging.WARNING)


def get_variable_colormap(field_config: Dict[str, Any]):
    """Return appropriate colormap for variable."""
    cmap_spec = field_config.get('cmap', 'RdBu_r')
    if cmap_spec == 'custom_rws':
        return mcolors.LinearSegmentedColormap.from_list("", 
            ["darkblue", "blue", "white", "red", "darkred"])
    return cmap_spec


def get_variable_specific_levels(field_config: Dict[str, Any], composite_list: List = None):
    """Return appropriate levels based on variable configuration."""
    if 'levels' in field_config:
        return field_config['levels'], None
    elif 'custom_levels' in field_config:
        return field_config['custom_levels'], field_config.get('custom_clevels_cont')
    else:
        # Fallback to dynamic level calculation
        if composite_list and len(composite_list) > 0:
            valid_composites = [c for c in composite_list if not np.all(np.isnan(c.values))]
            if valid_composites:
                combined_da = xr.concat(valid_composites, dim='temp_concat_dim')
                levels_spacing = field_config.get('levels_spacing', 1.0)
                levels, contour_levels = get_symmetric_levels_fixed_spacing(combined_da, spacing=levels_spacing)
                del combined_da
                return levels, contour_levels
        
        # Final fallback
        max_abs_val = 10.0
        levels = np.linspace(-max_abs_val, max_abs_val, 21)
        contour_levels = levels[::4]
        return levels, contour_levels


# Custom formatter for pressure axis (from regression script)
class PlainLogFormatter(matplotlib.ticker.LogFormatterMathtext):
    """Log formatter that always returns plain integers for pressure values."""
    def __call__(self, x, pos=None):
        if 100 <= x <= 1000:
            return f'{int(x)}'
        return ''

def extract_latitude_band(data_array: xr.DataArray, lat_band: Tuple[float, float] = (-5, 5)) -> xr.DataArray:
    """
    Extract and average data over specified latitude band with cosine weighting.
    
    Parameters:
    -----------
    data_array : xr.DataArray
        Input data with latitude dimension
    lat_band : tuple
        Latitude bounds for averaging (default: 5°S to 5°N for equatorial band)
    
    Returns:
    --------
    xr.DataArray : Averaged over specified latitude band
    """
    lat_min, lat_max = lat_band
    
    # Select latitude band
    band_data = data_array.sel(latitude=slice(lat_min, lat_max))
    
    # Weight by cosine of latitude for proper spherical averaging
    weights = np.cos(np.deg2rad(band_data.latitude))
    weighted_data = band_data.weighted(weights)
    
    # Average over latitude dimension
    result = weighted_data.mean(dim='latitude')
    result.attrs['latitude_band'] = f'{lat_min}°S to {lat_max}°N'
    
    return result

def process_surface_elevation_for_vertical_plot(lat_band: Tuple[float, float] = (-5, 5)) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
    """
    Process high-resolution surface elevation data for pressure-longitude plots.
    Returns both 1-degree and high-resolution versions.
    """
    try:
        # Load surface elevation as geopotential (0.1° resolution)
        surface_geopot = load_era_field(
            SURFACE_ELEV_FILE,
            var_name='z',  # Geopotential
            lat_slice=(lat_band[0]-1, lat_band[1]+1)  # Slightly wider for averaging
        )
        
        # Convert geopotential (m²/s²) to geopotential height (m)
        surface_height = surface_geopot / G
        
        # Convert height to pressure using standard atmosphere
        exponent = (G * M) / (R * L)
        surface_pressure = P0 * (1 - L * surface_height / T0) ** exponent
        
        # Step 1: Latitude band averaging at high resolution
        weights = np.cos(np.deg2rad(surface_pressure.latitude))
        surf_press_weighted = surface_pressure.weighted(weights)
        surf_press_lat_avg = surf_press_weighted.mean(dim='latitude')
        
        # Step 2: Regrid to 1° longitude resolution
        lon_1deg = np.arange(-180, 180, 1.0)
        surf_press_1deg = surf_press_lat_avg.interp(
            longitude=lon_1deg,
            method='linear'
        )
        
        logging.info(f"Processed surface elevation for equatorial band {lat_band}")
        return surf_press_1deg, surf_press_lat_avg
        
    except Exception as e:
        logging.warning(f"Could not load surface elevation data: {e}")
        return None, None



def plot_vertical_section_composite_figure(
    collected_seasonal_data: List[Dict],
    composite_case: str,
    primary_field_name: str,
    primary_field_config: Dict[str, Any],
    target_spatial_fields_config_global: Dict[str, Dict[str, Any]],
    composite_type_tag: str,
    output_dir_plots: Path,
    alpha: float,
    n_input_years1: int,
    n_input_years2: Optional[int] = None
) -> None:
    """Plot multi-panel composite maps for all seasons."""
    n_seasons = len(collected_seasonal_data)
    if n_seasons == 0:
        logging.warning(f"No seasonal data to plot for {primary_field_name}, case {composite_case}.")
        return

    fig, axs = plt.subplots(n_seasons, 1, figsize=(21, 8 * n_seasons),
                            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    if n_seasons == 1:
        axs = [axs]

    var_units = primary_field_config.get('units', '')
    cmap = primary_field_config.get('cmap', 'RdBu_r')
    levels_spacing = primary_field_config.get('levels_spacing', 0.1)

    # Collect all composites for common level calculation
    all_primary_composites_for_case = []
    for season_data in collected_seasonal_data:
        comp = season_data.get(f'comp_{composite_case.lower()}_primary')
        if comp is not None:
            all_primary_composites_for_case.append(comp)
    
    common_levels, common_clevels_cont = None, None
    contourf_mappable = None

    # Check if field has custom levels defined
    if 'custom_levels' in primary_field_config:
        common_levels = primary_field_config['custom_levels']
        common_clevels_cont = primary_field_config.get('custom_clevels_cont', common_levels[::4])
        logging.info(f"Using custom levels for {primary_field_name}: {common_levels[0]} to {common_levels[-1]}")
    elif all_primary_composites_for_case:
        valid_composites = [c for c in all_primary_composites_for_case if not np.all(np.isnan(c.data))]
        if valid_composites:
            combined_da_for_levels = xr.concat(valid_composites, dim='temp_concat_dim')
            common_levels, common_clevels_cont = get_symmetric_levels_fixed_spacing(combined_da_for_levels, spacing=levels_spacing)
            del combined_da_for_levels

    if common_levels is None or len(common_levels) < 2:
        max_abs_val = 1.0
        common_levels = np.linspace(-max_abs_val, max_abs_val, 21)
        common_clevels_cont = common_levels[::2]

    for idx, season_data_dict in enumerate(collected_seasonal_data):
        ax = axs[idx]
        display_suffix = season_data_dict['display_suffix']
        
        primary_comp = season_data_dict.get(f'comp_{composite_case.lower()}_primary')
        primary_sig = season_data_dict.get(f'sig_{composite_case.lower()}_primary')

        n_years_info_str = ""
        if composite_case == 'High':
            n_contrib = len(season_data_dict.get('actual_high_contrib_years_primary', []))
            n_years_info_str = f"N={n_contrib} years"
        elif composite_case == 'Low':
            n_contrib = len(season_data_dict.get('actual_low_contrib_years_primary', []))
            n_years_info_str = f"N={n_contrib} years"
        elif composite_case == 'Diff':
            n_h_contrib = len(season_data_dict.get('actual_high_contrib_years_primary', []))
            n_l_contrib = len(season_data_dict.get('actual_low_contrib_years_primary', []))
            n_years_info_str = f"N_H={n_h_contrib} years, N_L={n_l_contrib} years"

        ax_title = f"{display_suffix}"

        if primary_comp is None or np.all(np.isnan(primary_comp.data)):
            ax.text(0.5, 0.5, "No Composite Data", transform=ax.transAxes, ha="center", va="center", fontsize=28, color='gray')
        else:
            # Add cyclic point to eliminate dateline gap
            primary_comp_cyclic = add_cyclic_point_xr(primary_comp)
            
            mesh = primary_comp_cyclic.plot.contourf(ax=ax, transform=ccrs.PlateCarree(),
                                              levels=common_levels, cmap=cmap,
                                              add_colorbar=False, extend='both')
            if contourf_mappable is None:
                contourf_mappable = mesh

            if common_clevels_cont is not None and len(common_clevels_cont) > 1:
                try:
                    cs = ax.contour(
                        primary_comp_cyclic.longitude, primary_comp_cyclic.latitude, primary_comp_cyclic.data,
                        transform=ccrs.PlateCarree(),
                        levels=common_clevels_cont,
                        colors='darkgray', linewidths=0.5
                    )
                    plt.clabel(cs, fmt='%1.1f', fontsize=14, colors='dimgrey', inline=True, inline_spacing=4)
                except ValueError as ve:
                    logging.warning(f"Could not draw/label contours for {ax_title}: {ve}")
            
            # Plot significance dots only if enabled
            if primary_field_config.get('plot_significant_composite', True) and \
               primary_sig is not None and not np.all(np.isnan(primary_sig.data)) and np.any(primary_sig.data):
                # Add cyclic point to significance field as well
                primary_sig_cyclic = add_cyclic_point_xr(primary_sig)
                lon_m, lat_m = np.meshgrid(primary_sig_cyclic.longitude.data, primary_sig_cyclic.latitude.data)
                ax.scatter(lon_m[primary_sig_cyclic.data], lat_m[primary_sig_cyclic.data],
                           s=0.3, color='black', alpha=0.6, transform=ccrs.PlateCarree(), marker='o', linewidths=0)

        # Add wind overlays
        for wind_overlay_data in season_data_dict.get('wind_overlays_data', []):
            u_comp_wind = wind_overlay_data.get(f'comp_{composite_case.lower()}_u')
            v_comp_wind = wind_overlay_data.get(f'comp_{composite_case.lower()}_v')
            sig_mask_wind = wind_overlay_data.get(f'sig_mask_{composite_case.lower()}_wind')
            plot_type = wind_overlay_data['vector_plot_type']
            special_scaling = wind_overlay_data.get('special_scaling')

            if u_comp_wind is not None and v_comp_wind is not None:
                if not primary_field_config.get('plot_significant_composite', True):
                    sig_mask_wind = xr.full_like(u_comp_wind, True, dtype=bool)
                elif sig_mask_wind is None:
                    continue
                
                u_plot, v_plot, lw_plot = get_processed_composite_winds(
                    u_comp_wind, v_comp_wind, sig_mask_wind, plot_type, special_scaling
                )
                
                if u_plot is not None and v_plot is not None:
                    if not (np.all(np.isnan(u_plot.data)) and np.all(np.isnan(v_plot.data))):
                        if plot_type == 'quiver':
                            # Add cyclic points to vector components
                            u_plot_cyclic, v_plot_cyclic = add_cyclic_point_to_vectors(u_plot, v_plot)
                            
                            max_abs_val_wind = np.nanmax(np.sqrt(u_plot_cyclic.data**2 + v_plot_cyclic.data**2)) if not np.all(np.isnan(u_plot_cyclic.data)) else 0
                            q_scale = max_abs_val_wind * 30 if max_abs_val_wind > 0 else 40
                            if special_scaling == 'waf':
                                q_scale = 37

                            ax.quiver(u_plot_cyclic.longitude.data, u_plot_cyclic.latitude.data,
                                      u_plot_cyclic.data, v_plot_cyclic.data,
                                      transform=ccrs.PlateCarree(), color='black',
                                      scale=q_scale, width=0.003, headwidth=3, headlength=5, zorder=5)
                        elif plot_type == 'streamline' and lw_plot is not None:
                            # Add cyclic points to vector components and linewidth
                            u_plot_cyclic, v_plot_cyclic = add_cyclic_point_to_vectors(u_plot, v_plot)
                            lw_plot_cyclic = add_cyclic_point_xr(lw_plot)
                            
                            ax.streamplot(u_plot_cyclic.longitude.data, u_plot_cyclic.latitude.data,
                                          u_plot_cyclic.data, v_plot_cyclic.data,
                                          density=1.1, color='forestgreen', maxlength=1.5,
                                          arrowsize=1.7, linewidth=lw_plot_cyclic.data,
                                          transform=ccrs.PlateCarree(), zorder=4)
        
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
        ax.set_extent([0, 359.99, -50, 50], crs=ccrs.PlateCarree())
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5, color='gray', linewidth=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 22}
        gl.ylabel_style = {'size': 22}
        ax.set_title(ax_title, fontsize=32)

    if contourf_mappable:
        fig.subplots_adjust(bottom=0.065 if n_seasons > 1 else 0.15)
        cbar_ax = fig.add_axes([0.3, 0.02, 0.4, 0.015])
        cb = fig.colorbar(contourf_mappable, cax=cbar_ax, orientation='horizontal')
        cb.ax.tick_params(labelsize=22)
        cb.set_label(f"{primary_field_name} Anomaly ({var_units})", size=24)

    # Collect year information for suptitle from first season's data
    n_years_suptitle_str = ""
    if collected_seasonal_data:
        first_season = collected_seasonal_data[0]
        if composite_case == 'High':
            n_contrib = len(first_season.get('actual_high_contrib_years_primary', []))
            n_years_suptitle_str = f"N={n_contrib} years"
        elif composite_case == 'Low':
            n_contrib = len(first_season.get('actual_low_contrib_years_primary', []))
            n_years_suptitle_str = f"N={n_contrib} years"
        elif composite_case == 'Diff':
            n_h_contrib = len(first_season.get('actual_high_contrib_years_primary', []))
            n_l_contrib = len(first_season.get('actual_low_contrib_years_primary', []))
            n_years_suptitle_str = f"N_H={n_h_contrib} years, N_L={n_l_contrib} years"
    
    # fig.suptitle(f"{composite_case.upper()} ({composite_type_tag}) - {primary_field_name} ({n_years_suptitle_str})\n(Significance p<{alpha})", 
    #              fontsize=18, y=0.99 if n_seasons > 1 else 1.05)
    plt.subplots_adjust(left=0.04, right=0.96, top=0.95, hspace=0.25)

    filename_plot = f"composite_multiseason_{composite_case.lower()}_{composite_type_tag}_{primary_field_name}.png"
    save_filepath = output_dir_plots / filename_plot
    output_dir_plots.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_filepath, bbox_inches='tight', pad_inches=0.1, dpi=300)
    logging.info(f"  Saved multi-season plot: {save_filepath}")
    plt.close(fig)


def plot_vertical_section_composite(
    ax, composite_data: xr.DataArray, significance_mask: Optional[xr.DataArray],
    levels: np.ndarray, cmap, field_config: Dict[str, Any],
    output_format: str = 'png'
) -> matplotlib.contour.QuadContourSet:
    """
    Plot pressure-longitude section for vertical streamfunction composites.
    
    Parameters:
    -----------
    ax : matplotlib axis
        Axis to plot on
    composite_data : xr.DataArray
        Composite data with dimensions (pressure_level, longitude)
    significance_mask : xr.DataArray or None
        Boolean mask indicating significant grid points
    levels : array-like
        Contour levels for filled contours
    cmap : colormap
        Colormap for contourf
    field_config : dict
        Field configuration dictionary
    """
    # Ensure we have valid data
    if np.all(np.isnan(composite_data.values)):
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                ha='center', va='center', fontsize=28, color='gray')
        return None
    
    # Main contourf plot with conditional rasterization
    contourf_kwargs = {
        'levels': levels,
        'cmap': cmap,
        'extend': 'both'
    }
    apply_rasterization_settings(contourf_kwargs, output_format)
    cf = ax.contourf(
        composite_data.longitude.values,
        composite_data.pressure_level.values,
        composite_data.values,
        **contourf_kwargs
    )
    
    # Add significance hatching (diagonal lines for significant areas)
    if significance_mask is not None and np.any(significance_mask.values):
        # Create mask where data IS significant (True values)
        sig_data = significance_mask.values.astype(float)
        
        # Add hatching using contourf
        ax.contourf(
            composite_data.longitude.values,
            composite_data.pressure_level.values,
            sig_data,
            levels=[0.5, 1.5],  # Captures all values of 1 (significant)
            colors='none',
            hatches=['///'],  # Diagonal lines
            alpha=0
        )
    
    # Add contour lines (conditional based on variable type)
    if field_config.get('contour_lines', True):
        contour_spacing = field_config.get('contour_spacing_factor', 4)
        contour_levels = levels[::contour_spacing]
        if len(contour_levels) > 1:
            try:
                cs = ax.contour(
                    composite_data.longitude.values,
                    composite_data.pressure_level.values,
                    composite_data.values,
                    levels=contour_levels,
                    colors='black',
                    linewidths=0.8,
                    alpha=0.7
                )
                # Label contours
                ax.clabel(cs, contour_levels[::2], fmt='%1.1f',
                         fontsize=18, colors='black',
                         inline=True, inline_spacing=4)
            except ValueError:
                pass
    
    # Configure axes
    ax.set_yscale('log')
    ax.invert_yaxis()
    
    # Variable-specific pressure range and ticks
    pressure_range = field_config.get('pressure_range', (1000, 200))
    ax.set_ylim(*pressure_range)
    ax.set_xlim(-180, 180)
    
    # Variable-specific pressure ticks
    pressure_levels = field_config.get('pressure_levels', [1000, 850, 700, 500, 300, 200])
    if len(pressure_levels) >= 5:  # RWS case
        major_ticks = pressure_levels
        minor_ticks = pressure_levels[:]  # Copy
        # Add intermediate ticks for RWS range
        if pressure_range == (500, 200):
            minor_ticks.extend([450, 350])
        minor_ticks.sort(reverse=True)  # Descending order
    else:  # Streamfunction case
        major_ticks = [1000, 850, 700, 500, 300, 200]
        minor_ticks = [1000, 900, 850, 800, 700, 600, 500, 400, 300, 200]
    
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.yaxis.set_major_formatter(PlainLogFormatter())
    ax.yaxis.set_minor_formatter(PlainLogFormatter())
    ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numticks=12))
    
    # Labels
    ax.set_xlabel('Longitude', fontsize=28)
    ax.set_ylabel('Pressure (hPa)', fontsize=28)
    ax.tick_params(labelsize=22)
    
    # Variable-specific reference lines
    if pressure_range == (500, 200):  # RWS
        ax.axhline(y=300, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
    else:  # Streamfunction
        ax.axhline(y=500, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    return cf

def add_vertical_section_annotations(ax, field_config: Dict[str, Any]):
    """
    Add Walker circulation references and surface topography to vertical section plot.
    """
    # Walker circulation centers
    walker_refs = [
        (-160, 'Pacific'), (160, 'Pacific'),
        (-30, 'Atlantic'), (90, 'Indian')
    ]
    
    # Variable-specific y-position for Walker circulation labels
    pressure_range = field_config.get('pressure_range', (1000, 200))
    if pressure_range == (500, 200):  # RWS case
        label_y_pos = 350  # At 350 hPa level
    else:  # Streamfunction case
        label_y_pos = 850  # At 850 hPa level

    for lon, label in walker_refs:
        ax.axvline(x=lon, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.text(lon, label_y_pos, label, ha='center', va='bottom', fontsize=28, fontweight='bold')
    
    # Add surface topography if available and enabled for this variable
    if field_config.get('include_topography', False) and '_cached_topography' in field_config:
        surf_press_1deg, surf_press_highres = field_config['_cached_topography']
        if surf_press_1deg is not None:
            # Main topography fill
            ax.fill_between(
                surf_press_1deg.longitude,
                surf_press_1deg.values,
                1050,  # Below plot bottom
                where=(surf_press_1deg.values < 1000),
                color='black',
                alpha=0.4,
                step='mid',
                label='Surface elevation'
            )
            
            # Add topographic feature annotations
            features = field_config.get('topography_features', {})
            for name, info in features.items():
                try:
                    lon_vals = surf_press_1deg.longitude.values
                    lon_mask = (lon_vals >= info['lon']-info['search_range']) & \
                              (lon_vals <= info['lon']+info['search_range'])
                    
                    if np.any(lon_mask):
                        local_data = surf_press_1deg.values[lon_mask]
                        local_lons = lon_vals[lon_mask]
                        
                        if len(local_data) > 0 and not np.all(np.isnan(local_data)):
                            local_min = np.nanmin(local_data)
                            if local_min < 1000:
                                min_idx = np.nanargmin(local_data)
                                min_lon = local_lons[min_idx]
                                
                                ax.annotate(name,
                                           xy=(min_lon, max(local_min, 950)),
                                           xytext=(info['lon'], info['text_y']),
                                           fontsize=20,
                                           ha='center',
                                           va='center',
                                           fontweight='bold',
                                           color='darkred',
                                           bbox=dict(boxstyle='round,pad=0.4',
                                                   facecolor='wheat',
                                                   edgecolor='darkred',
                                                   alpha=0.9,
                                                   linewidth=1.5),
                                           arrowprops=dict(arrowstyle='->',
                                                         connectionstyle='arc3,rad=0.2',
                                                         color='darkred',
                                                         alpha=0.9,
                                                         linewidth=2.5,
                                                         shrinkA=5, shrinkB=5),
                                           zorder=1000)
                except Exception as e:
                    logging.warning(f"Could not annotate {name}: {e}")

def plot_vertical_section_composite_figure(
    collected_seasonal_data: List[Dict],
    composite_case: str,
    primary_field_name: str,
    primary_field_config: Dict[str, Any],
    composite_type_tag: str,
    output_dir_plots: Path,
    alpha: float,
    n_input_years1: int,
    n_input_years2: Optional[int] = None,
    output_format: str = 'png',
    raster_dpi: int = 150,
    vector_dpi: int = 300,
    single_panel: bool = False,
    panel_suffix: str = ''
) -> None:
    """
    Plot multi-panel or single-panel vertical section composites for all seasons.
    
    Similar to plot_vertical_section_composite_figure but specialized for
    pressure-longitude sections.
    """
    n_seasons = len(collected_seasonal_data)
    if n_seasons == 0:
        logging.warning(f"No seasonal data to plot for {primary_field_name}, case {composite_case}.")
        return

    var_units = primary_field_config.get('units', '')
    cmap = get_variable_colormap(primary_field_config)

    # Collect all composites for common level calculation
    all_primary_composites_for_case = []
    for season_data in collected_seasonal_data:
        comp = season_data.get(f'comp_{composite_case.lower()}_primary')
        if comp is not None:
            all_primary_composites_for_case.append(comp)
    
    # Use variable-specific level calculation
    common_levels, common_clevels_cont = get_variable_specific_levels(
        primary_field_config, all_primary_composites_for_case
    )

    # Get latitude band info for titles/filenames
    lat_band_info = ""
    lat_band_str = ""
    if '_current_lat_band' in primary_field_config:
        lat_band = primary_field_config['_current_lat_band']
        lat_str = f"{abs(lat_band[0])}°{'S' if lat_band[0] < 0 else 'N'}-{abs(lat_band[1])}°{'S' if lat_band[1] < 0 else 'N'}"
        lat_band_info = f" ({lat_str} average)"
        lat_band_str = f"_{lat_str.replace('°', '').replace('-', '_')}"
    
    if single_panel:
        # Generate individual files for each season
        for idx, season_data_dict in enumerate(collected_seasonal_data):
            display_suffix = season_data_dict['display_suffix']
            
            # Create single panel figure
            fig, ax = setup_single_panel_figure(figsize=(21, 6))
            
            primary_comp = season_data_dict.get(f'comp_{composite_case.lower()}_primary')
            primary_sig = season_data_dict.get(f'sig_{composite_case.lower()}_primary')
            
            # Simple title for single panels
            ax.set_title(create_clean_panel_title(display_suffix), fontsize=32, pad=10)

            if primary_comp is None or np.all(np.isnan(primary_comp.values)):
                ax.text(0.5, 0.5, "No Composite Data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=28, color='gray')
            else:
                # Plot vertical section with optimization
                cf = plot_vertical_section_composite(
                    ax, primary_comp, primary_sig,
                    common_levels, cmap, primary_field_config,
                    output_format
                )
                
                # Add annotations (Walker circulation, topography)
                add_vertical_section_annotations(ax, primary_field_config)
                
                # Add colorbar for single panel
                if cf is not None:
                    cax = fig.add_axes([0.3, 0.02, 0.4, 0.015])
                    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
                    cbar.set_label(var_units, fontsize=24)
                    cbar.ax.tick_params(labelsize=22)
            
            # Descriptive filename for single panel
            season_code = display_suffix.replace('(', '_').replace(')', '').replace(' ', '_')
            filename = create_descriptive_filename(
                base_name='vertical_section',
                method=composite_case.lower(),
                var_name=primary_field_name,
                predictor=composite_type_tag,
                pathway=lat_band_str,
                lag='',
                season=season_code,
                suffix=panel_suffix
            )
            
            filepath = output_dir_plots / filename
            
            try:
                save_figure_optimized(fig, str(filepath), output_format, raster_dpi, vector_dpi)
                logging.info(f"Saved single panel vertical section: {filepath}")
            except Exception as e:
                logging.error(f"Failed to save single panel {filepath}: {e}")
            finally:
                plt.close(fig)
                gc.collect()
    
    else:
        # Original multi-panel code
        fig, axs = plt.subplots(n_seasons, 1, figsize=(21, 8 * n_seasons),
                                subplot_kw={'aspect': 'auto'})
        if n_seasons == 1:
            axs = [axs]
        
        contourf_mappable = None

        for idx, season_data_dict in enumerate(collected_seasonal_data):
            ax = axs[idx]
            display_suffix = season_data_dict['display_suffix']
            
            primary_comp = season_data_dict.get(f'comp_{composite_case.lower()}_primary')
            primary_sig = season_data_dict.get(f'sig_{composite_case.lower()}_primary')

            # Build title with year information
            n_years_info_str = ""
            if composite_case == 'High':
                n_contrib = len(season_data_dict.get('actual_high_contrib_years_primary', []))
                n_years_info_str = f"N={n_contrib} years"
            elif composite_case == 'Low':
                n_contrib = len(season_data_dict.get('actual_low_contrib_years_primary', []))
                n_years_info_str = f"N={n_contrib} years"
            elif composite_case == 'Diff':
                n_h_contrib = len(season_data_dict.get('actual_high_contrib_years_primary', []))
                n_l_contrib = len(season_data_dict.get('actual_low_contrib_years_primary', []))
                n_years_info_str = f"N_H={n_h_contrib} years, N_L={n_l_contrib} years"

            ax_title = f"{display_suffix}"
            ax.set_title(ax_title, fontsize=32, pad=10)

            if primary_comp is None or np.all(np.isnan(primary_comp.values)):
                ax.text(0.5, 0.5, "No Composite Data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=28, color='gray')
            else:
                # Plot vertical section with optimization
                cf = plot_vertical_section_composite(
                    ax, primary_comp, primary_sig,
                    common_levels, cmap, primary_field_config,
                    output_format
                )
                
                if cf is not None:
                    contourf_mappable = cf
                
                # Add annotations (Walker circulation, topography)
                add_vertical_section_annotations(ax, primary_field_config)
        
        # Add single colorbar at bottom
        if contourf_mappable is not None:
            cax = fig.add_axes([0.3, 0.02, 0.4, 0.015])
            cbar = plt.colorbar(contourf_mappable, cax=cax, orientation='horizontal')
            cbar.set_label(var_units, fontsize=24)
            cbar.ax.tick_params(labelsize=22)

        # Collect year information for suptitle from first season's data
        n_years_suptitle_str = ""
        if collected_seasonal_data:
            first_season = collected_seasonal_data[0]
            if composite_case == 'High':
                n_contrib = len(first_season.get('actual_high_contrib_years_primary', []))
                n_years_suptitle_str = f"N={n_contrib} years"
            elif composite_case == 'Low':
                n_contrib = len(first_season.get('actual_low_contrib_years_primary', []))
                n_years_suptitle_str = f"N={n_contrib} years"
            elif composite_case == 'Diff':
                n_h_contrib = len(first_season.get('actual_high_contrib_years_primary', []))
                n_l_contrib = len(first_season.get('actual_low_contrib_years_primary', []))
                n_years_suptitle_str = f"N_H={n_h_contrib} years, N_L={n_l_contrib} years"
        
        # plt.suptitle(f"{composite_case.upper()} ({composite_type_tag}) - {primary_field_name}{lat_band_info} ({n_years_suptitle_str})",
        #              fontsize=18, y=0.99)
        plt.subplots_adjust(left=0.06, bottom=0.065, right=0.94, top=0.95, hspace=0.28)

        # Save figure with optimization
        base_filename = f"{primary_field_name}_{composite_case}_{composite_type_tag.replace(' ', '_')}_vertical_section"
        filepath = output_dir_plots / base_filename
        
        try:
            save_figure_optimized(fig, str(filepath), output_format, raster_dpi, vector_dpi)
            logging.info(f"Saved vertical section composite plot: {filepath}")
        except Exception as e:
            logging.error(f"Failed to save vertical section plot {filepath}: {e}")
        finally:
            plt.close(fig)
            gc.collect()


def process_vertical_section_composites(
    primary_field_name: str,
    event_years_high: List[int],
    event_years_low: List[int],
    target_season_specs: List[Tuple[str, int, str]],
    composite_type_tag: str,
    output_dir_plots: Path,
    output_dir_nc: Path,
    alpha: float,
    n_high_input: int,
    n_low_input: int,
    precursor_name: str = "",
    output_format: str = 'png',
    raster_dpi: int = 150,
    vector_dpi: int = 300,
    single_panel: bool = False,
    panel_suffix: str = '',
    use_precursor_seasons: bool = False,
    seasons_filter: Optional[List[str]] = None
) -> None:
    """
    Process vertical section composites for all latitude bands.
    """
    primary_field_config = TARGET_SPATIAL_FIELDS_CONFIG[primary_field_name]
    
    
    # Load primary field data once
    analysis_slice = slice(None)  # Use all available data
    primary_anomalies_dask = get_spatial_anomalies_field(
        primary_field_name, primary_field_config,
        DEFAULT_CLIMATOLOGY_START_YEAR, DEFAULT_CLIMATOLOGY_END_YEAR,
        analysis_slice
    )
    
    if primary_anomalies_dask is None:
        logging.error(f"Failed to load primary field data for {primary_field_name}. Skipping multi-band processing.")
        return
    
    primary_anomalies_loaded = primary_anomalies_dask.load()
    del primary_anomalies_dask
    
    # Define the standard ENSO cycle plot sequence
    target_season_specs = STANDARD_ENSO_CYCLE_PLOT_SEQUENCE
    
    # Process each latitude band
    for lat_band in VERTICAL_SECTION_LAT_BANDS:
        band_name = LAT_BAND_NAMES.get(lat_band, f"lat{lat_band[0]}-{lat_band[1]}")
        
        # Create band-specific output directories
        band_plots_dir = output_dir_plots / "vertical_sections" / band_name
        band_nc_dir = output_dir_nc / "vertical_sections" / band_name
        band_plots_dir.mkdir(parents=True, exist_ok=True)
        band_nc_dir.mkdir(parents=True, exist_ok=True)
        
        # Pre-load surface elevation data for this band
        surf_press_1deg, surf_press_highres = process_surface_elevation_for_vertical_plot(lat_band)
        modified_field_config = primary_field_config.copy()
        modified_field_config['_cached_topography'] = (surf_press_1deg, surf_press_highres)
        modified_field_config['_current_lat_band'] = lat_band
        
        # Process composites for this specific band with optimization
        generate_seasonal_composites_and_outputs_for_band(
            primary_field_name=primary_field_name,
            primary_field_config=modified_field_config,
            primary_anomalies_loaded=primary_anomalies_loaded,
            event_years_high=event_years_high,
            event_years_low=event_years_low,
            target_season_specs=target_season_specs,
            composite_type_tag=composite_type_tag,
            output_dir_plots=band_plots_dir,
            output_dir_nc=band_nc_dir,
            alpha=alpha,
            n_high_input=n_high_input,
            n_low_input=n_low_input,
            precursor_name=precursor_name,
            lat_band=lat_band,
            output_format=output_format,
            raster_dpi=raster_dpi,
            vector_dpi=vector_dpi,
            single_panel=single_panel,
            panel_suffix=panel_suffix,
            use_precursor_seasons=use_precursor_seasons,
            seasons_filter=seasons_filter
        )

def generate_seasonal_composites_and_outputs_for_band(
    primary_field_name: str,
    primary_field_config: Dict[str, Any],
    primary_anomalies_loaded: xr.DataArray,
    event_years_high: List[int],
    event_years_low: List[int],
    target_season_specs: List[Tuple[str, int, str]],
    composite_type_tag: str,
    output_dir_plots: Path,
    output_dir_nc: Path,
    alpha: float,
    n_high_input: int,
    n_low_input: int,
    precursor_name: str = "",
    lat_band: Tuple[float, float] = (-5, 5),
    output_format: str = 'png',
    raster_dpi: int = 150,
    vector_dpi: int = 300,
    single_panel: bool = False,
    panel_suffix: str = '',
    use_precursor_seasons: bool = False,
    seasons_filter: Optional[List[str]] = None
) -> None:
    """
    Generate composites for a specific latitude band (used for vertical sections).
    """
    
    # Determine target season specifications dynamically
    if seasons_filter or (use_precursor_seasons and precursor_name):
        target_season_specs = get_target_season_specs_for_precursor(
            precursor_name if use_precursor_seasons else "",
            STANDARD_ENSO_CYCLE_PLOT_SEQUENCE,
            seasons_filter
        )
        logging.info(f"Using dynamic season selection for vertical sections: {len(target_season_specs)} seasons")
    else:
        # Use the provided target_season_specs parameter (backward compatibility)
        logging.info(f"Using provided season specifications for vertical sections: {len(target_season_specs)} seasons")
    
    collected_seasonal_data_for_plotting = []
    
    for target_season_code, year_offset, display_suffix in target_season_specs:
        current_season_results = {'display_suffix': display_suffix}

        # Initialize variables for later use in DIFF significance
        slices_high_eq = None
        slices_low_eq = None

        # Process HIGH composite
        comp_high_primary, sig_high_primary, actual_high_contrib_years_primary = None, None, []
        if event_years_high:
            slices_high_primary_local, actual_high_contrib_years_primary = select_target_data_for_compositing(
                primary_anomalies_loaded, event_years_high, target_season_code, year_offset
            )
            if slices_high_primary_local is not None and slices_high_primary_local.time.size >= max(1, n_high_input // 2):
                # Apply latitude band averaging before computing composite
                slices_high_eq = extract_latitude_band(slices_high_primary_local, lat_band)
                comp_high_primary = slices_high_eq.mean(dim='time')
                
                # For significance, test on latitude-averaged time series
                if primary_field_config.get('plot_significant_composite', True):
                    # Compute significance on the latitude-averaged data (still has time dimension)
                    sig_high_primary = calculate_fieldwise_ttest_significance(
                        slices_high_eq, None, popmean=0.0, alpha=alpha
                    )
        # Process LOW composite
        comp_low_primary, sig_low_primary, actual_low_contrib_years_primary = None, None, []
        if event_years_low:
            slices_low_primary_local, actual_low_contrib_years_primary = select_target_data_for_compositing(
                primary_anomalies_loaded, event_years_low, target_season_code, year_offset
            )
            if slices_low_primary_local is not None and slices_low_primary_local.time.size >= max(1, n_low_input // 2):
                # Apply latitude band averaging before computing composite
                slices_low_eq = extract_latitude_band(slices_low_primary_local, lat_band)
                comp_low_primary = slices_low_eq.mean(dim='time')
                
                # For significance, test on latitude-averaged time series
                if primary_field_config.get('plot_significant_composite', True):
                    # Compute significance on the latitude-averaged data (still has time dimension)
                    sig_low_primary = calculate_fieldwise_ttest_significance(
                        slices_low_eq, None, popmean=0.0, alpha=alpha
                    )
        # Compute difference composite
        comp_diff_primary, sig_diff_primary = None, None
        if comp_high_primary is not None and comp_low_primary is not None:
            comp_diff_primary = comp_high_primary - comp_low_primary
            # For difference significance, use two-sample t-test on latitude-averaged data
            if primary_field_config.get('plot_significant_composite', True) and \
               slices_high_eq is not None and slices_low_eq is not None:
                sig_diff_primary = calculate_fieldwise_ttest_significance(
                    slices_high_eq, slices_low_eq, alpha=alpha
                )

        # Store results for this season
        current_season_results.update({
            'comp_high_primary': comp_high_primary,
            'comp_low_primary': comp_low_primary,
            'comp_diff_primary': comp_diff_primary,
            'sig_high_primary': sig_high_primary,
            'sig_low_primary': sig_low_primary,
            'sig_diff_primary': sig_diff_primary,
            'actual_high_contrib_years_primary': actual_high_contrib_years_primary,
            'actual_low_contrib_years_primary': actual_low_contrib_years_primary,
        })
        
        collected_seasonal_data_for_plotting.append(current_season_results)

    # Generate plots
    if collected_seasonal_data_for_plotting:
        lat_str = f"{abs(lat_band[0])}°{'S' if lat_band[0] < 0 else 'N'}-{abs(lat_band[1])}°{'S' if lat_band[1] < 0 else 'N'}"
        band_name = LAT_BAND_NAMES.get(lat_band, f"lat{lat_band[0]}-{lat_band[1]}")
        band_suffix = f"_{band_name}"
        
        # Plot HIGH, LOW, and DIFF composites with optimization
        if any(d.get('comp_high_primary') is not None for d in collected_seasonal_data_for_plotting):
            plot_vertical_section_composite_figure(
                collected_seasonal_data_for_plotting,
                'High', primary_field_name, primary_field_config,
                composite_type_tag + band_suffix, output_dir_plots, alpha,
                n_high_input, n_low_input,
                output_format, raster_dpi, vector_dpi, single_panel, panel_suffix
            )
        
        if any(d.get('comp_low_primary') is not None for d in collected_seasonal_data_for_plotting):
            plot_vertical_section_composite_figure(
                collected_seasonal_data_for_plotting,
                'Low', primary_field_name, primary_field_config,
                composite_type_tag + band_suffix, output_dir_plots, alpha,
                n_high_input, n_low_input,
                output_format, raster_dpi, vector_dpi, single_panel, panel_suffix
            )
        
        if any(d.get('comp_diff_primary') is not None for d in collected_seasonal_data_for_plotting):
            plot_vertical_section_composite_figure(
                collected_seasonal_data_for_plotting,
                'Diff', primary_field_name, primary_field_config,
                composite_type_tag + band_suffix, output_dir_plots, alpha,
                n_high_input, n_low_input,
                output_format, raster_dpi, vector_dpi, single_panel, panel_suffix
            )
        
        # Save NetCDF for this band
        save_consolidated_composites_netcdf(
            collected_seasonal_data_for_plotting,
            primary_field_name,
            primary_field_config,
            TARGET_SPATIAL_FIELDS_CONFIG,
            composite_type_tag + band_suffix,
            output_dir_nc,
            alpha,
            DEFAULT_CLIMATOLOGY_START_YEAR,
            DEFAULT_CLIMATOLOGY_END_YEAR,
            slice(None)
        )

def main(args: argparse.Namespace) -> None:
    """Main function for sequential processing."""
    N_YEARS = args.n_years
    ALPHA = args.alpha
    CLIM_START_STR = f"{args.clim_start_year}-01-01"
    CLIM_END_STR = f"{args.clim_end_year}-12-31"
    ANALYSIS_SLICE = slice(DEFAULT_ANALYSIS_START_DATE_STR, DEFAULT_ANALYSIS_END_DATE_STR)

    run_specific_plot_dir = PLOT_DIR / f"N{N_YEARS}_alpha{str(ALPHA).replace('.', 'p')}"
    setup_logging(run_specific_plot_dir)
    run_specific_save_dir = SAVE_DATA_DIR / f"N{N_YEARS}_alpha{str(ALPHA).replace('.', 'p')}"
    
    logging.info(f"--- Composite Analysis ---")
    logging.info(f"N for Top/Bottom Composites: {N_YEARS}")
    logging.info(f"Significance Alpha: {ALPHA}")
    logging.info(f"Climatology Period for Anomalies: {args.clim_start_year} to {args.clim_end_year}")
    logging.info(f"Analysis Time Slice for Anomalies: {DEFAULT_ANALYSIS_START_DATE_STR} to {DEFAULT_ANALYSIS_END_DATE_STR}")

    all_precursors_ds = load_all_precursor_timeseries(PRECURSOR_TS_FILEPATH)
    if all_precursors_ds is None:
        return

    primary_fields_to_process = [
        key for key, conf in TARGET_SPATIAL_FIELDS_CONFIG.items() 
        if conf.get('primary_plot_field', False) and conf.get('is_vertical_section', False)
    ]
    if args.field_key:
        if args.field_key in primary_fields_to_process:
            primary_fields_to_process = [args.field_key]
            logging.info(f"Processing only specified primary field: {args.field_key}")
        else:
            logging.error(f"Specified field_key '{args.field_key}' is not a configured primary plot field. Exiting.")
            return
    else:
        logging.info(f"Processing all primary plot fields: {primary_fields_to_process}")

    if args.job_type in ['all', 'individual']:
        logging.info("\n==== PART 1: Individual Precursor Composites ====")
        logging.info("SKIPPING PART 1 - Already completed")

        for precursor_name in INDIVIDUAL_PRECURSOR_NAMES_PART1:
            if precursor_name not in all_precursors_ds:
                logging.warning(f"Skipping individual precursor {precursor_name}: Not found in loaded dataset.")
                continue
            
            logging.info(f"\n--- Processing Individual Precursor: {precursor_name} ---")
            
            active_ts = get_active_season_precursor_series(all_precursors_ds, precursor_name)
            
            # Select extreme years based on the precursor's defining season
            top_years, bottom_years = select_extreme_years_for_precursor(
                active_ts, precursor_name, N_YEARS,
                PRECURSOR_TO_DEFINITION_SEASON_KEY_MAP,
                SEASON_TO_MIDDLE_MONTH
            )
            
            if not top_years or not bottom_years:
                logging.warning(f"  Insufficient data for {precursor_name}. Skipping.")
                continue
            
            precursor_def_season_key = PRECURSOR_TO_DEFINITION_SEASON_KEY_MAP.get(precursor_name)
            if not precursor_def_season_key:
                logging.warning(f"  Definition season key not found for {precursor_name}. Skipping.")
                continue
            
            # Convert to Y_event_ref
            year_adjustment_to_get_y_event_ref = 0
            if precursor_def_season_key in ['JJA_-1', 'SON_-1']:
                year_adjustment_to_get_y_event_ref = 1
    
            event_y_event_ref_high = [y + year_adjustment_to_get_y_event_ref for y in top_years]
            event_y_event_ref_low = [y + year_adjustment_to_get_y_event_ref for y in bottom_years]
            
            logging.info(f"  Event reference years HIGH: {sorted(event_y_event_ref_high)}")
            logging.info(f"  Event reference years LOW: {sorted(event_y_event_ref_low)}")
    
            target_season_specs_list = STANDARD_ENSO_CYCLE_PLOT_SEQUENCE
            
            output_plots_dir_part1 = run_specific_plot_dir / "Individual" / precursor_name.replace(' ', '_').replace('(', '').replace(')', '')
            output_nc_dir_part1 = run_specific_save_dir / "Individual_NetCDF" / precursor_name.replace(' ', '_').replace('(', '').replace(')', '')
    
            collected_data_for_csv_part1: Optional[List[Dict]] = None
    
            for prim_field_name in primary_fields_to_process:
                # Use new multi-band vertical section processing with optimization
                process_vertical_section_composites(
                    primary_field_name=prim_field_name,
                    event_years_high=event_y_event_ref_high,
                    event_years_low=event_y_event_ref_low,
                    target_season_specs=target_season_specs_list,  # Will be overridden if use_precursor_seasons=True
                    composite_type_tag=f"TopBottom{N_YEARS}_{precursor_name.replace(' ', '_').replace('(', '').replace(')', '')}",
                    output_dir_plots=output_plots_dir_part1,
                    output_dir_nc=output_nc_dir_part1,
                    alpha=ALPHA,
                    n_high_input=len(event_y_event_ref_high),
                    n_low_input=len(event_y_event_ref_low),
                    precursor_name=precursor_name,
                    output_format=args.output_format,
                    raster_dpi=args.raster_dpi,
                    vector_dpi=args.vector_dpi,
                    single_panel=args.single_panel,
                    panel_suffix=args.panel_suffix,
                    use_precursor_seasons=getattr(args, 'precursor_based_seasons', False),
                    seasons_filter=getattr(args, 'seasons', None)
                )
                temp_collected_data = None  # No single collected data for multi-band processing
    

    if args.job_type in ['all', 'set', 'set_3of4']:
        logging.info("\n==== PART 2: Precursor Set Composites (Terciles - 3 of 4 Criterion) ====")
        MIN_YEARS_FOR_TERCILE_COMPOSITE = args.min_years_tercile

        # Calculate potential Y_event_ref range
        analysis_start_year = pd.to_datetime(DEFAULT_ANALYSIS_START_DATE_STR).year
        start_potential_Y_event_ref = analysis_start_year + 1
    
        analysis_end_dt = pd.to_datetime(DEFAULT_ANALYSIS_END_DATE_STR)
        max_data_stamp_year_for_djf = analysis_end_dt.year if analysis_end_dt.month >= 1 else analysis_end_dt.year - 1
        end_potential_Y_event_ref = max_data_stamp_year_for_djf - 1
    
        potential_event_ref_years = pd.RangeIndex(start=start_potential_Y_event_ref,
                                                 stop=end_potential_Y_event_ref + 1)
    
        for set_name, set_precursor_names_list in PRECURSOR_SETS_3OF4_DICT.items():
            logging.info(f"\n--- Processing Precursor Set: {set_name} (3 of 4 criterion) ---")
    
            # Pre-calculate terciles and collect data for each precursor
            precursor_data = {}
            is_set_valid = True
    
            for precursor_name in set_precursor_names_list:
                if precursor_name not in all_precursors_ds:
                    logging.warning(f"  Precursor '{precursor_name}' not found in dataset. Skipping set {set_name}.")
                    is_set_valid = False
                    break
    
                active_ts = get_active_season_precursor_series(all_precursors_ds, precursor_name)
                if active_ts.time.size < 3:
                    logging.warning(f"  Insufficient data for {precursor_name}. Skipping set {set_name}.")
                    is_set_valid = False
                    break
    
                def_season_key = PRECURSOR_TO_DEFINITION_SEASON_KEY_MAP.get(precursor_name)
                if not def_season_key:
                    logging.warning(f"  No definition season key for {precursor_name}. Skipping set {set_name}.")
                    is_set_valid = False
                    break
    
                # Get the specific season for this precursor
                season_code = def_season_key.split('_')[0]
                target_month = SEASON_TO_MIDDLE_MONTH.get(season_code)
                
                # Filter to only the relevant season
                season_filtered_ts = active_ts.sel(time=active_ts.time.dt.month == target_month)
                
                # Calculate terciles on the filtered data
                low_terc, high_terc = np.nanpercentile(season_filtered_ts.data, [100./3., 200./3.])
                
                precursor_data[precursor_name] = {
                    'ts': season_filtered_ts,  # Store filtered time series
                    'low_terc': low_terc,
                    'high_terc': high_terc,
                    'def_season_key': def_season_key,
                    'enso_sign': PRECURSOR_ENSO_SIGN_MAP.get(precursor_name, 1)
                }
    
            if not is_set_valid:
                continue
    
            # Process each potential ENSO event reference year
            upper_phase_event_ref_years = []
            lower_phase_event_ref_years = []
    
            for y_event_ref in potential_event_ref_years:
                # Track tercile positions for each precursor
                upper_tercile_count = 0  # Count of precursors in their upper tercile
                lower_tercile_count = 0  # Count of precursors in their lower tercile
                has_opposite_upper = False  # Any precursor in opposite tercile for upper phase
                has_opposite_lower = False  # Any precursor in opposite tercile for lower phase
                has_nan = False
    
                for precursor_name, p_data in precursor_data.items():
                    try:
                        cal_year, season = get_precursor_ref_calendar_year_and_season(
                            y_event_ref, p_data['def_season_key']
                        )
                        month = SEASON_TO_MIDDLE_MONTH[season]
    
                        # Find the precursor value for this specific time point
                        precursor_value = p_data['ts'].sel(
                            time=f"{cal_year}-{month:02d}-15", method="nearest"
                        ).item()
    
                        if np.isnan(precursor_value):
                            has_nan = True
                            break
    
                        # Determine tercile position considering ENSO sign
                        if p_data['enso_sign'] == 1:
                            if precursor_value >= p_data['high_terc']:
                                upper_tercile_count += 1
                                has_opposite_lower = True
                            elif precursor_value <= p_data['low_terc']:
                                lower_tercile_count += 1
                                has_opposite_upper = True
                            # else: precursor is in middle tercile, no action needed
                        else:  # enso_sign == -1
                            if precursor_value <= p_data['low_terc']:
                                upper_tercile_count += 1
                                has_opposite_lower = True
                            elif precursor_value >= p_data['high_terc']:
                                lower_tercile_count += 1
                                has_opposite_upper = True
                            # else: precursor is in middle tercile, no action needed
    
                    except Exception as e:
                        logging.warning(f"Error processing Y_event_ref {y_event_ref} for {precursor_name}: {e}")
                        has_nan = True
                        break
    
                # Apply the "3 of 4, none opposite" criterion
                if not has_nan:
                    # For upper phase: need at least 3 in upper tercile and none in opposite
                    if upper_tercile_count >= 3 and not has_opposite_upper:
                        upper_phase_event_ref_years.append(y_event_ref)
                    
                    # For lower phase: need at least 3 in lower tercile and none in opposite
                    if lower_tercile_count >= 3 and not has_opposite_lower:
                        lower_phase_event_ref_years.append(y_event_ref)
    
            n_upper = len(upper_phase_event_ref_years)
            n_lower = len(lower_phase_event_ref_years)
            logging.info(f"  Set {set_name}: Found {n_upper} years for UPPER phase (3 of 4 criterion)")
            logging.info(f"  Set {set_name}: Found {n_lower} years for LOWER phase (3 of 4 criterion)")
    
            target_season_specs_list_for_set = STANDARD_ENSO_CYCLE_PLOT_SEQUENCE
    
            output_plots_dir_part3 = run_specific_plot_dir / "Sets_3of4" / set_name
            output_nc_dir_part3 = run_specific_save_dir / "Sets_NetCDF_3of4" / set_name
    
            event_years_high_for_set = upper_phase_event_ref_years if n_upper >= MIN_YEARS_FOR_TERCILE_COMPOSITE else []
            event_years_low_for_set = lower_phase_event_ref_years if n_lower >= MIN_YEARS_FOR_TERCILE_COMPOSITE else []
    
            if not event_years_high_for_set and not event_years_low_for_set:
                logging.warning(f"  Not enough years for either phase in set {set_name}. Min required: {MIN_YEARS_FOR_TERCILE_COMPOSITE}")
                continue
    
            collected_data_for_csv_part3: Optional[List[Dict]] = None
    
            for prim_field_name in primary_fields_to_process:
                # Use new multi-band vertical section processing with optimization
                process_vertical_section_composites(
                    primary_field_name=prim_field_name,
                    event_years_high=event_years_high_for_set,
                    event_years_low=event_years_low_for_set,
                    target_season_specs=target_season_specs_list_for_set,
                    composite_type_tag=f"Tercile_3of4_{set_name}",
                    output_dir_plots=output_plots_dir_part3,
                    output_dir_nc=output_nc_dir_part3,
                    alpha=ALPHA,
                    n_high_input=n_upper,
                    n_low_input=n_lower,
                    precursor_name=f"Set_{set_name}_3of4",
                    output_format=args.output_format,
                    raster_dpi=args.raster_dpi,
                    vector_dpi=args.vector_dpi,
                    single_panel=args.single_panel,
                    panel_suffix=args.panel_suffix
                )
                temp_collected_data = None  # No single collected data for multi-band processing

    logging.info("--- Composite Analysis Finished ---")


def run_composite_job(job_spec, args):
    """Execute a single composite analysis job."""
    job_type, entity_name, primary_field_name = job_spec

    N_YEARS = args.n_years
    ALPHA = args.alpha
    CLIM_START_STR = f"{args.clim_start_year}-01-01"
    CLIM_END_STR = f"{args.clim_end_year}-12-31"
    ANALYSIS_SLICE = slice(DEFAULT_ANALYSIS_START_DATE_STR, DEFAULT_ANALYSIS_END_DATE_STR)

    run_specific_plot_dir = PLOT_DIR / f"N{N_YEARS}_alpha{str(ALPHA).replace('.', 'p')}"
    run_specific_save_dir = SAVE_DATA_DIR / f"N{N_YEARS}_alpha{str(ALPHA).replace('.', 'p')}"

    all_precursors_ds = load_all_precursor_timeseries(PRECURSOR_TS_FILEPATH)
    if all_precursors_ds is None:
        return

    if job_type == 'Individual':
        precursor_name = entity_name
        if precursor_name not in all_precursors_ds:
            logging.warning(f"Skipping individual precursor {precursor_name}: Not found in loaded dataset.")
            return

        active_ts = get_active_season_precursor_series(all_precursors_ds, precursor_name)
        
        # Use the new season-aware selection
        top_years, bottom_years = select_extreme_years_for_precursor(
            active_ts, precursor_name, N_YEARS,
            PRECURSOR_TO_DEFINITION_SEASON_KEY_MAP,
            SEASON_TO_MIDDLE_MONTH
        )
        
        if not top_years or not bottom_years:
            logging.warning(f"Insufficient data for {precursor_name}. Skipping.")
            return

        precursor_def_season_key = PRECURSOR_TO_DEFINITION_SEASON_KEY_MAP.get(precursor_name)
        if not precursor_def_season_key:
            logging.warning(f"Definition season key not found for {precursor_name}. Skipping.")
            return

        year_adjustment_to_get_y_event_ref = 0
        if precursor_def_season_key in ['JJA_-1', 'SON_-1']:
            year_adjustment_to_get_y_event_ref = 1

        event_y_event_ref_high = [y + year_adjustment_to_get_y_event_ref for y in top_years]
        event_y_event_ref_low = [y + year_adjustment_to_get_y_event_ref for y in bottom_years]

        output_plots_dir = run_specific_plot_dir / "Individual" / precursor_name.replace(' ', '_').replace('(', '').replace(')', '')
        output_nc_dir = run_specific_save_dir / "Individual_NetCDF" / precursor_name.replace(' ', '_').replace('(', '').replace(')', '')

        # Use new multi-band vertical section processing with optimization
        process_vertical_section_composites(
            primary_field_name=primary_field_name,
            event_years_high=event_y_event_ref_high,
            event_years_low=event_y_event_ref_low,
            target_season_specs=STANDARD_ENSO_CYCLE_PLOT_SEQUENCE,  # Will be overridden if use_precursor_seasons=True
            composite_type_tag=f"TopBottom{N_YEARS}_{precursor_name.replace(' ', '_').replace('(', '').replace(')', '')}",
            output_dir_plots=output_plots_dir,
            output_dir_nc=output_nc_dir,
            alpha=ALPHA,
            n_high_input=len(event_y_event_ref_high),
            n_low_input=len(event_y_event_ref_low),
            precursor_name=precursor_name,
            output_format=args.output_format,
            raster_dpi=args.raster_dpi,
            vector_dpi=args.vector_dpi,
            single_panel=args.single_panel,
            panel_suffix=args.panel_suffix,
            use_precursor_seasons=getattr(args, 'precursor_based_seasons', False),
            seasons_filter=getattr(args, 'seasons', None)
        )
        collected_data = None  # No single collected data for multi-band processing
        if collected_data:
            composite_type_tag = f"TopBottom{N_YEARS}_{precursor_name.replace(' ', '_').replace('(', '').replace(')', '')}"
            save_contributing_years_csv(collected_data, composite_type_tag, output_plots_dir)

    elif job_type == 'Set_3of4':
        set_name = entity_name
        MIN_YEARS_FOR_TERCILE_COMPOSITE = args.min_years_tercile
        
        # Calculate potential Y_event_ref range
        analysis_start_year = pd.to_datetime(DEFAULT_ANALYSIS_START_DATE_STR).year
        start_potential_Y_event_ref = analysis_start_year + 1
        
        analysis_end_dt = pd.to_datetime(DEFAULT_ANALYSIS_END_DATE_STR)
        max_data_stamp_year_for_djf = analysis_end_dt.year if analysis_end_dt.month >= 1 else analysis_end_dt.year - 1
        end_potential_Y_event_ref = max_data_stamp_year_for_djf - 1
        
        potential_event_ref_years = pd.RangeIndex(start=start_potential_Y_event_ref,
                                                 stop=end_potential_Y_event_ref + 1)
        
        set_precursor_names_list = PRECURSOR_SETS_3OF4_DICT[set_name]
        
        # Pre-calculate terciles and collect data for each precursor
        precursor_data = {}
        is_set_valid = True
        
        for precursor_name in set_precursor_names_list:
            if precursor_name not in all_precursors_ds:
                logging.warning(f"  Precursor '{precursor_name}' not found in dataset. Skipping set {set_name}.")
                is_set_valid = False
                break
            
            active_ts = get_active_season_precursor_series(all_precursors_ds, precursor_name)
            if active_ts.time.size < 3:
                logging.warning(f"  Insufficient data for {precursor_name}. Skipping set {set_name}.")
                is_set_valid = False
                break
            
            def_season_key = PRECURSOR_TO_DEFINITION_SEASON_KEY_MAP.get(precursor_name)
            if not def_season_key:
                logging.warning(f"  No definition season key for {precursor_name}. Skipping set {set_name}.")
                is_set_valid = False
                break
            
            # Get the specific season for this precursor
            season_code = def_season_key.split('_')[0]
            target_month = SEASON_TO_MIDDLE_MONTH.get(season_code)
            
            # Filter to only the relevant season
            season_filtered_ts = active_ts.sel(time=active_ts.time.dt.month == target_month)
            
            # Calculate terciles on the filtered data
            low_terc, high_terc = np.nanpercentile(season_filtered_ts.data, [100./3., 200./3.])
            
            precursor_data[precursor_name] = {
                'ts': season_filtered_ts,  # Store filtered time series
                'low_terc': low_terc,
                'high_terc': high_terc,
                'def_season_key': def_season_key,
                'enso_sign': PRECURSOR_ENSO_SIGN_MAP.get(precursor_name, 1)
            }
        
        if not is_set_valid:
            return
        
        # Process each potential ENSO event reference year
        upper_phase_event_ref_years = []
        lower_phase_event_ref_years = []
        
        for y_event_ref in potential_event_ref_years:
            # Track tercile positions for each precursor
            upper_tercile_count = 0  # Count of precursors in their upper tercile
            lower_tercile_count = 0  # Count of precursors in their lower tercile
            has_opposite_upper = False  # Any precursor in opposite tercile for upper phase
            has_opposite_lower = False  # Any precursor in opposite tercile for lower phase
            has_nan = False
            
            for precursor_name, p_data in precursor_data.items():
                try:
                    cal_year, season = get_precursor_ref_calendar_year_and_season(
                        y_event_ref, p_data['def_season_key']
                    )
                    month = SEASON_TO_MIDDLE_MONTH[season]
                    
                    # Find the precursor value for this specific time point
                    precursor_value = p_data['ts'].sel(
                        time=f"{cal_year}-{month:02d}-15", method="nearest"
                    ).item()
                    
                    if np.isnan(precursor_value):
                        has_nan = True
                        break
                    
                    # Determine tercile position considering ENSO sign
                    if p_data['enso_sign'] == 1:
                        if precursor_value >= p_data['high_terc']:
                            upper_tercile_count += 1
                            has_opposite_lower = True
                        elif precursor_value <= p_data['low_terc']:
                            lower_tercile_count += 1
                            has_opposite_upper = True
                        # else: precursor is in middle tercile, no action needed
                    else:  # enso_sign == -1
                        if precursor_value <= p_data['low_terc']:
                            upper_tercile_count += 1
                            has_opposite_lower = True
                        elif precursor_value >= p_data['high_terc']:
                            lower_tercile_count += 1
                            has_opposite_upper = True
                        # else: precursor is in middle tercile, no action needed
                    
                except Exception as e:
                    logging.warning(f"Error processing Y_event_ref {y_event_ref} for {precursor_name}: {e}")
                    has_nan = True
                    break
            
            # Apply the "3 of 4, none opposite" criterion
            if not has_nan:
                # For upper phase: need at least 3 in upper tercile and none in opposite
                if upper_tercile_count >= 3 and not has_opposite_upper:
                    upper_phase_event_ref_years.append(y_event_ref)
                
                # For lower phase: need at least 3 in lower tercile and none in opposite
                if lower_tercile_count >= 3 and not has_opposite_lower:
                    lower_phase_event_ref_years.append(y_event_ref)
        
        n_upper = len(upper_phase_event_ref_years)
        n_lower = len(lower_phase_event_ref_years)
        logging.info(f"  Set {set_name}: Found {n_upper} years for UPPER phase (3 of 4 criterion)")
        logging.info(f"  Set {set_name}: Found {n_lower} years for LOWER phase (3 of 4 criterion)")
        
        output_plots_dir = run_specific_plot_dir / "Sets_3of4" / set_name
        output_nc_dir = run_specific_save_dir / "Sets_NetCDF_3of4" / set_name
        
        event_years_high_for_set = upper_phase_event_ref_years if n_upper >= MIN_YEARS_FOR_TERCILE_COMPOSITE else []
        event_years_low_for_set = lower_phase_event_ref_years if n_lower >= MIN_YEARS_FOR_TERCILE_COMPOSITE else []
        
        if not event_years_high_for_set and not event_years_low_for_set:
            logging.warning(f"  Not enough years for either phase in set {set_name}. Min required: {MIN_YEARS_FOR_TERCILE_COMPOSITE}")
            return
        
        # Use vertical section processing for vertical fields with optimization
        process_vertical_section_composites(
            primary_field_name=primary_field_name,
            event_years_high=event_years_high_for_set,
            event_years_low=event_years_low_for_set,
            target_season_specs=STANDARD_ENSO_CYCLE_PLOT_SEQUENCE,
            composite_type_tag=f"Tercile_3of4_{set_name}",
            output_dir_plots=output_plots_dir,
            output_dir_nc=output_nc_dir,
            alpha=ALPHA,
            n_high_input=n_upper,
            n_low_input=n_lower,
            precursor_name=f"Set_{set_name}",
            output_format=args.output_format,
            raster_dpi=args.raster_dpi,
            vector_dpi=args.vector_dpi,
            single_panel=args.single_panel,
            panel_suffix=args.panel_suffix
        )
        collected_data = None  # No single collected data for multi-band processing
        
        if collected_data:
            composite_type_tag = f"Tercile_3of4_{set_name}"
            save_contributing_years_csv(collected_data, composite_type_tag, output_plots_dir)


def launch_driver(args):
    """Build the list of jobs and dispatch them in parallel."""
    # Setup logging
    run_specific_plot_dir = PLOT_DIR / f"N{args.n_years}_alpha{str(args.alpha).replace('.', 'p')}"
    setup_logging(run_specific_plot_dir)

    # Build job list
    jobs = []

    # Individual precursor jobs
    if args.job_type in ['all', 'individual']:
        for precursor_name in INDIVIDUAL_PRECURSOR_NAMES_PART1:
            for field_name in [args.field_key] if args.field_key else [
                key for key, conf in TARGET_SPATIAL_FIELDS_CONFIG.items()
                if conf.get('primary_plot_field', False) and conf.get('is_vertical_section', False)
            ]:
                jobs.append(('Individual', precursor_name, field_name))

    # Set-based jobs (3 of 4 criterion)
    if args.job_type in ['all', 'set', 'set_3of4']:
        for set_name in PRECURSOR_SETS_3OF4_DICT.keys():
            for field_name in [args.field_key] if args.field_key else [
                key for key, conf in TARGET_SPATIAL_FIELDS_CONFIG.items()
                if conf.get('primary_plot_field', False) and conf.get('is_vertical_section', False)
            ]:
                jobs.append(('Set_3of4', set_name, field_name))

    # Execute jobs in parallel
    n_workers = min(args.workers, len(jobs)) if args.workers else min(available_workers(), len(jobs))
    logging.info(f"Launching {len(jobs)} jobs using {n_workers} workers")

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(run_composite_job, job, args) for job in jobs]
        concurrent.futures.wait(futures)

    logging.info("All jobs completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run composite analysis based on precursor indices."
    )
    parser.add_argument('--n_years', type=int, default=DEFAULT_N_YEARS_COMPOSITE,
                        help=f"Number of top/bottom years for individual precursor composites (default: {DEFAULT_N_YEARS_COMPOSITE}).")
    parser.add_argument('--alpha', type=float, default=DEFAULT_SIGNIFICANCE_ALPHA,
                        help=f"Significance level for t-tests (default: {DEFAULT_SIGNIFICANCE_ALPHA}).")
    parser.add_argument('--clim_start_year', type=str, default=DEFAULT_CLIMATOLOGY_START_YEAR,
                        help=f"Start year for climatology calculation (default: {DEFAULT_CLIMATOLOGY_START_YEAR}).")
    parser.add_argument('--clim_end_year', type=str, default=DEFAULT_CLIMATOLOGY_END_YEAR,
                        help=f"End year for climatology calculation (default: {DEFAULT_CLIMATOLOGY_END_YEAR}).")
    parser.add_argument('--field_key', type=str, default=None,
                        help=f"Specific primary spatial field key to process from TARGET_SPATIAL_FIELDS_CONFIG (e.g., 'SST'). Processes all primary fields by default.")
    parser.add_argument('--min_years_tercile', type=int, default=3,
                        help="Minimum number of years required to make a tercile-based set composite (default: 3).")
    parser.add_argument('--workers', type=int, default=None,
                        help="Number of parallel workers. Defaults to system-determined safe value.")
    parser.add_argument('--job_type', type=str, default='all',
                        choices=['all', 'individual', 'set', 'set_3of4'],
                        help="Type of jobs to run: 'all' (default), 'individual' (Part 1 only), or 'set'/'set_3of4' (Part 2 only)")
    parser.add_argument('--precursor_based_seasons', action='store_true', default=False,
                        help='Start season sequence from precursor definition season for individual precursors (default: start from JJA(-1))')
    parser.add_argument('--seasons', nargs='+', 
                        choices=['JJA(-1)', 'SON(-1)', 'DJF(0)', 'MAM(0)', 'JJA(0)', 'SON(0)', 'DJF(+1)'],
                        default=None,
                        help='Generate only specific seasons (overrides precursor-based selection)')
    
    # Add standard plotting arguments (skip seasons since we define our own)
    add_plotting_arguments(parser, skip_seasons=True)

    args = parser.parse_args()
    launch_driver(args)
