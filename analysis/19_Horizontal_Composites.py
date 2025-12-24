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

try:
    from utils.paths import get_data_path, get_results_path
    from utils.plotting_optimization import (
        setup_cartopy_warnings, save_figure_optimized, 
        add_plotting_arguments, apply_rasterization_settings, 
        clean_data_for_pdf, filter_seasons_to_plot,
        setup_single_panel_figure, create_clean_panel_title,
        create_descriptive_filename, get_contourf_kwargs, get_pcolormesh_kwargs
    )
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


DATA_DIR_TS = Path(get_data_path('time_series', data_type="processed"))
PRECURSOR_TS_FILEPATH = DATA_DIR_TS / 'PCMCI_data_ts_st.nc'
DATA_DIR_GRIDDED = Path(get_data_path('1_deg_seasonal/', data_type="interim"))
PLOT_DIR = Path(get_results_path('composites/'))
SAVE_DATA_DIR = Path(get_data_path('composites/'))

INDIVIDUAL_PRECURSOR_NAMES_PART1 = [
'REOF SST JJA'
]
"""
    'REOF SST JJA', 'MCA WAF-RWS SON', 'MCA RWS-WAF DJF', 'MCA RWS-prec MAM(E)',
    'MCA prec-RWS SON', 'MCA RWS-prec DJF', 
    'MCA RWS-prec MAM(C)'
"""


EP_PRECURSORS_3OF4 = ['REOF SST JJA', 'MCA WAF-RWS SON', 'MCA RWS-WAF DJF', 'MCA RWS-prec MAM(E)']
CP_PRECURSORS_3OF4 = ['REOF SST JJA', 'MCA prec-RWS SON', 'MCA RWS-prec DJF', 'MCA RWS-prec MAM(C)']
PRECURSOR_SETS_3OF4_DICT = {'EP_set': EP_PRECURSORS_3OF4, 'CP_set': CP_PRECURSORS_3OF4}

TARGET_SPATIAL_FIELDS_CONFIG: Dict[str, Dict[str, Any]] = {
    'SST': {
        'filename': 'SST_seas_1deg.nc', 'var_name_in_file': 'sst', 'units': 'K',
        'cmap': 'coolwarm', 'levels_spacing': 0.1,
        'primary_plot_field': True,
        'default_wind_overlays': ['U10m', 'U_200'],
        'plot_significant_composite': True
    },
    'RWS_200': {
        'filename': 'RWS_200_seas_1deg.nc', 'var_name_in_file': 'RWS',
        'units': '$10^{-11}$ s$^{-2}$', 'scale': 1e11,
        'cmap': mcolors.LinearSegmentedColormap.from_list("", ["darkblue", "blue", "white", "red", "darkred"]),
        'primary_plot_field': True,
        'default_wind_overlays': ['WAFx_200', 'U_200'],
        'plot_significant_composite': True,
        'custom_levels': np.arange(-100, 105, 5),  # Custom levels for RWS_200 due to large value range
        'custom_clevels_cont': np.arange(-100, 120, 20)  # Contour levels (every 4th level)
    },
    'prec': {
        'filename': 'prec_seas_1deg.nc', 'var_name_in_file': 'tp', 'units': 'mm day$^{-1}$',
        'cmap': mcolors.LinearSegmentedColormap.from_list("", ["maroon", "saddlebrown", "white", "limegreen", "darkgreen"]),
        'levels_spacing': 0.2,
        'primary_plot_field': True,
        'default_wind_overlays': ['U_850', 'U_200'],
        'plot_significant_composite': True
    },
    'surf_pres': {
        'filename': 'surf_pres_seas_1deg.nc', 'var_name_in_file': 'sp', 'units': 'hPa', 'lat_slice': (-50, 50),
        'cmap': mcolors.LinearSegmentedColormap.from_list("", ["teal", "turquoise", "white", "hotpink", "purple"]),
        'levels_spacing': 0.25,
        'primary_plot_field': True,
        'default_wind_overlays': ['U_850', 'U_200'],
        'plot_significant_composite': True
    },
    'sf_200': {
        'filename': 'sf_200_seas_1deg.nc', 'var_name_in_file': 'streamfunction',
        'units': '$10^5$ m$^2$ s$^{-1}$', 'scale': 1e-5,
        'cmap': 'coolwarm', 'levels_spacing': 10,
        'primary_plot_field': True,
        'default_wind_overlays': ['U_850'],
        'plot_significant_composite': True
    },
    'vp_200': {
        'filename': 'vp_200_seas_1deg.nc', 'var_name_in_file': 'velocity_potential',
        'units': '$10^5$ m$^2$ s$^{-1}$', 'scale': 1e-5,
        'cmap': 'BrBG_r', 'levels_spacing': 2,
        'primary_plot_field': True,
        'default_wind_overlays': ['U_850', 'U_200'],
        'plot_significant_composite': True
    },
    'low_clouds': {
        'filename': 'low_cloud_cover_seas_1deg.nc', 'var_name_in_file': 'lcc',
        'units': '-', 'cmap': mcolors.LinearSegmentedColormap.from_list("", ["blue", "deepskyblue", "white", "darkgrey", "dimgray"]),
        'levels_spacing': 0.02,
        'primary_plot_field': True,
        'default_wind_overlays': ['U10m', 'U_200'],
        'plot_significant_composite': True
    },
    'U10m': {
        'filename': 'U_10m_seas_1deg.nc', 'var_name_in_file': 'u10', 'units': 'm s$^{-1}$',
        'primary_plot_field': False, 'is_vector_u_component': True,
        'v_component_key': 'V10m', 'vector_plot_type': 'quiver',
        'plot_significant_composite': False
    },
    'V10m': {
        'filename': 'V_10m_seas_1deg.nc', 'var_name_in_file': 'v10', 'units': 'm s$^{-1}$',
        'primary_plot_field': False, 'is_vector_v_component': True,
        'plot_significant_composite': False
    },
    'U_850': {
        'filename': 'U_850_seas_1deg.nc', 'var_name_in_file': 'u', 'units': 'm s$^{-1}$',
        'primary_plot_field': False, 'is_vector_u_component': True,
        'v_component_key': 'V_850', 'vector_plot_type': 'quiver',
        'plot_significant_composite': False
    },
    'V_850': {
        'filename': 'V_850_seas_1deg.nc', 'var_name_in_file': 'v', 'units': 'm s$^{-1}$',
        'primary_plot_field': False, 'is_vector_v_component': True,
        'plot_significant_composite': False
    },
    'WAFx_200': {
        'filename': 'WAF_200_components_1deg.nc', 'var_name_in_file': 'WAFx', 'units': 'm$^2$ s$^{-2}$',
        'primary_plot_field': False, 'is_vector_u_component': True,
        'v_component_key': 'WAFy_200', 'vector_plot_type': 'quiver', 'special_scaling': 'waf',
        'plot_significant_composite': False
    },
    'WAFy_200': {
        'filename': 'WAF_200_components_1deg.nc', 'var_name_in_file': 'WAFy', 'units': 'm$^2$ s$^{-2}$',
        'primary_plot_field': False, 'is_vector_v_component': True,
        'plot_significant_composite': False
    },
    'U_200': {
        'filename': 'U_200_seas_1deg.nc', 'var_name_in_file': 'u', 'units': 'm s$^{-1}$',
        'primary_plot_field': False, 'is_vector_u_component': True,
        'v_component_key': 'V_200', 'vector_plot_type': 'streamline',
        'plot_significant_composite': False
    },
    'V_200': {
        'filename': 'V_200_seas_1deg.nc', 'var_name_in_file': 'v', 'units': 'm s$^{-1}$',
        'primary_plot_field': False, 'is_vector_v_component': True,
        'plot_significant_composite': False
    },
}

for k, v_dict in TARGET_SPATIAL_FIELDS_CONFIG.items():
    v_dict['raw_data_path'] = DATA_DIR_GRIDDED / v_dict['filename']



# --- 1. UTILITY FUNCTIONS ---

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Turn down third-party verbosity
for noisy_mod in ('matplotlib', 'matplotlib.font_manager', 'PIL', 'fontTools', 'cartopy', 'pyproj'):
    logging.getLogger(noisy_mod).setLevel(logging.WARNING)

# Setup Cartopy warning suppression immediately
setup_cartopy_warnings()

# Custom formatter for pressure axis (from regression script)
class PlainLogFormatter(matplotlib.ticker.LogFormatterMathtext):
    """Log formatter that always returns plain integers for pressure values."""
    def __call__(self, x, pos=None):
        if 100 <= x <= 1000:
            return f'{int(x)}'
        return ''

def plot_horizontal_composite_figure(
    collected_seasonal_data: List[Dict],
    composite_case: str,
    primary_field_name: str,
    primary_field_config: Dict[str, Any],
    target_spatial_fields_config_global: Dict[str, Dict[str, Any]],
    composite_type_tag: str,
    output_dir_plots: Path,
    alpha: float,
    n_input_years1: int,
    n_input_years2: Optional[int] = None,
    show_all_winds: bool = False,
    args: Optional[argparse.Namespace] = None
) -> None:
    """Plot multi-panel composite maps for all seasons."""
    n_seasons = len(collected_seasonal_data)
    if n_seasons == 0:
        logging.warning(f"No seasonal data to plot for {primary_field_name}, case {composite_case}.")
        return

    fig, axs = plt.subplots(n_seasons, 1, figsize=(21, 6 * n_seasons),
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
    
    # Collect year count information for suptitle
    figure_level_n_years_str = ""
    if composite_case == 'High' and n_input_years1 is not None:
        figure_level_n_years_str = f"N={n_input_years1} years"
    elif composite_case == 'Low' and n_input_years1 is not None:
        figure_level_n_years_str = f"N={n_input_years1} years"
    elif composite_case == 'Diff' and n_input_years1 is not None and n_input_years2 is not None:
        figure_level_n_years_str = f"N_H={n_input_years1} years, N_L={n_input_years2} years"
    
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

        # Simplified title - only show the season
        ax_title = f"{display_suffix}"

        if primary_comp is None or np.all(np.isnan(primary_comp.data)):
            ax.text(0.5, 0.5, "No Composite Data", transform=ax.transAxes, ha="center", va="center", fontsize=18, color='gray')
        else:
            # Add cyclic point to eliminate dateline gap
            primary_comp_cyclic = add_cyclic_point_xr(primary_comp)
            
            # Use optimized contourf with conditional rasterization
            output_format = getattr(args, 'output_format', 'png') if args else 'png'
            contourf_kwargs = get_contourf_kwargs(common_levels, cmap, ccrs.PlateCarree(), output_format)
            contourf_kwargs['add_colorbar'] = False
            contourf_kwargs['extend'] = 'both'
            mesh = primary_comp_cyclic.plot.contourf(ax=ax, **contourf_kwargs)
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
                    if not (args and getattr(args, 'no_contour_labels', False)):
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
                           s=2, color='black', alpha=0.6, transform=ccrs.PlateCarree(), marker='o', linewidths=0)

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
                
                # Two-pass approach when show_all_winds is True
                if show_all_winds and sig_mask_wind is not None:
                    # First pass: Plot all winds with lighter colors
                    u_all, v_all, lw_all = get_processed_composite_winds(
                        u_comp_wind, v_comp_wind, sig_mask_wind, plot_type, special_scaling, 
                        apply_significance_mask=False,
                        equatorial_mask_deg=10.0 if special_scaling == 'waf' else None
                    )
                    
                    if u_all is not None and v_all is not None:
                        # Check if we have valid data to plot
                        u_valid = ~np.isnan(u_all.data)
                        v_valid = ~np.isnan(v_all.data)
                        has_valid_data = np.any(u_valid & v_valid)
                        
                        if has_valid_data:
                            if plot_type == 'quiver':
                                # Add cyclic points to vector components
                                u_all_cyclic, v_all_cyclic = add_cyclic_point_to_vectors(u_all, v_all)
                                
                                max_abs_val_wind = np.nanmax(np.sqrt(u_all_cyclic.data**2 + v_all_cyclic.data**2)) if not np.all(np.isnan(u_all_cyclic.data)) else 0
                                q_scale = max_abs_val_wind * 30 if max_abs_val_wind > 0 else 40
                                if special_scaling == 'waf':
                                    q_scale = 37

                                # Preserve NaNs by masking invalids so quiver skips non-significant points
                                u_all_masked = np.ma.masked_invalid(u_all_cyclic.data)
                                v_all_masked = np.ma.masked_invalid(v_all_cyclic.data)
                                ax.quiver(u_all_cyclic.longitude.data, u_all_cyclic.latitude.data,
                                          u_all_masked, v_all_masked,
                                          transform=ccrs.PlateCarree(), color='gray', alpha=0.4,
                                          scale=q_scale, width=0.003, headwidth=3, headlength=5, zorder=3)
                            elif plot_type == 'streamline' and lw_all is not None:
                                try:
                                    # Add cyclic points to vector components and linewidth
                                    u_all_cyclic, v_all_cyclic = add_cyclic_point_to_vectors(u_all, v_all)
                                    lw_all_cyclic = add_cyclic_point_xr(lw_all)
                                    
                                    # Clean data for PDF compatibility
                                    u_filled, v_filled, lw_filled = clean_data_for_pdf(
                                        u_all_cyclic.data, v_all_cyclic.data, lw_all_cyclic.data
                                    )
                                    
                                    ax.streamplot(u_all_cyclic.longitude.data, u_all_cyclic.latitude.data,
                                                  u_filled, v_filled,
                                                  density=1.1, color='lightgreen', maxlength=1.5,
                                                  arrowsize=1.7, linewidth=lw_filled * 0.5,
                                                  transform=ccrs.PlateCarree(), zorder=2)
                                except Exception as e:
                                    logging.warning(f"Failed to plot all winds streamlines: {e}")
                
                # Second pass (or only pass if show_all_winds is False): Plot significant winds
                u_plot, v_plot, lw_plot = get_processed_composite_winds(
                    u_comp_wind, v_comp_wind, sig_mask_wind, plot_type, special_scaling,
                    apply_significance_mask=True,
                    equatorial_mask_deg=10.0 if special_scaling == 'waf' else None
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

                            # Preserve NaNs by masking invalids so quiver skips non-significant points
                            u_plot_masked = np.ma.masked_invalid(u_plot_cyclic.data)
                            v_plot_masked = np.ma.masked_invalid(v_plot_cyclic.data)
                            ax.quiver(u_plot_cyclic.longitude.data, u_plot_cyclic.latitude.data,
                                      u_plot_masked, v_plot_masked,
                                      transform=ccrs.PlateCarree(), color='black',
                                      scale=q_scale, width=0.003, headwidth=3, headlength=5, zorder=5)
                        elif plot_type == 'streamline' and lw_plot is not None:
                            try:
                                # Add cyclic points to vector components and linewidth
                                u_plot_cyclic, v_plot_cyclic = add_cyclic_point_to_vectors(u_plot, v_plot)
                                lw_plot_cyclic = add_cyclic_point_xr(lw_plot)
                                
                                # Clean data for PDF compatibility
                                u_plot_clean, v_plot_clean, lw_plot_clean = clean_data_for_pdf(
                                    u_plot_cyclic.data, v_plot_cyclic.data, lw_plot_cyclic.data
                                )
                                ax.streamplot(u_plot_cyclic.longitude.data, u_plot_cyclic.latitude.data,
                                              u_plot_clean, v_plot_clean,
                                              density=1.1, color='forestgreen', maxlength=1.5,
                                              arrowsize=1.7, linewidth=lw_plot_clean,
                                              transform=ccrs.PlateCarree(), zorder=4)
                            except Exception as e:
                                logging.warning(f"Failed to plot significant winds streamlines: {e}")
        
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=':')
        
        # Determine map extent based on field type
        if primary_field_name in ['RWS_200', 'sf_200']:
            # Extended southern boundary for Rossby wave and streamfunction fields
            ax.set_extent([0, 359.99, -70, 30], crs=ccrs.PlateCarree())
            logging.debug(f"Using extended extent (-70°S to 30°N) for {primary_field_name}")
        else:
            # Standard tropical/subtropical extent for other fields
            ax.set_extent([0, 359.99, -50, 50], crs=ccrs.PlateCarree())
            logging.debug(f"Using standard extent (-50°S to 50°N) for {primary_field_name}")
        gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5, color='gray', linewidth=0.5)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 22}
        gl.ylabel_style = {'size': 22}
        ax.set_title(ax_title, fontsize=32)

        # Add equatorial boundary indicators if WAF vectors were plotted
        has_waf = any(w.get('special_scaling') == 'waf' 
                      for w in season_data_dict.get('wind_overlays_data', []))
        if has_waf:
            ax.axhline(y=10, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
            ax.axhline(y=-10, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

    if contourf_mappable:
        # Dynamic adjustment based on number of seasons
        if n_seasons <= 2:
            cbar_height = 0.045
            bottom_adjust = 0.12
            cbar_bottom = 0.02
        elif n_seasons <= 4:
            cbar_height = 0.02
            bottom_adjust = 0.02
            cbar_bottom = 0.015
        else:  # 5+ seasons
            cbar_height = 0.015
            bottom_adjust = 0.04
            cbar_bottom = 0.01

        fig.subplots_adjust(bottom=bottom_adjust)
        cbar_ax = fig.add_axes([0.3, cbar_bottom, 0.4, cbar_height])
        cb = fig.colorbar(contourf_mappable, cax=cbar_ax, orientation='horizontal')
        cb.ax.tick_params(labelsize=22)
        cb.set_label(f"{primary_field_name} Anomaly ({var_units})", size=24)

    plt.subplots_adjust(left=0.04, right=0.96, top=0.95, hspace=0.25)

    # Handle single panel mode
    if args and getattr(args, 'single_panel', False) and len(collected_seasonal_data) == 1:
        # For single panel, use simple title without suptitle
        display_suffix = collected_seasonal_data[0]['display_suffix']
        axs[0].set_title(create_clean_panel_title(display_suffix), fontsize=32)
        # Remove suptitle for single panels
        # fig.suptitle('')
    
    # Create filename based on mode
    base_filename = create_descriptive_filename(
        base_name='composite',
        method='composite',
        var_name=primary_field_name,
        predictor=composite_type_tag,
        pathway=composite_case.lower(),
        lag='multiseason',
        season='_'.join([d['display_suffix'].replace('(', '').replace(')', '') for d in collected_seasonal_data[:3]]) if len(collected_seasonal_data) > 3 else '_'.join([d['display_suffix'].replace('(', '').replace(')', '') for d in collected_seasonal_data]),
        wind_level='all_winds' if show_all_winds else None,
        suffix=getattr(args, 'panel_suffix', '') if args else ''
    )
    
    save_filepath = output_dir_plots / base_filename
    output_dir_plots.mkdir(parents=True, exist_ok=True)
    
    # Use optimized save function
    output_format = getattr(args, 'output_format', 'png') if args else 'png'
    raster_dpi = getattr(args, 'raster_dpi', 150) if args else 150
    vector_dpi = getattr(args, 'vector_dpi', 300) if args else 300
    
    try:
        save_figure_optimized(fig, str(save_filepath), output_format, raster_dpi, vector_dpi)
    except Exception as e:
        logging.error(f"  Failed to save plot {save_filepath}: {e}", exc_info=True)
    finally:
        plt.close(fig)


def generate_seasonal_composites_and_outputs(
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
    target_spatial_fields_config_global: Dict[str, Dict[str, Any]],
    clim_start_str: str,
    clim_end_str: str,
    analysis_slice: slice,
    show_all_winds: bool = False,
    args: Optional[argparse.Namespace] = None,
    precursor_name: str = "",
    use_precursor_seasons: bool = False,
    seasons_filter: Optional[List[str]] = None
) -> Optional[List[Dict]]:
    """Generate composites for all seasons and create outputs."""
    logging.info(f"Processing multi-season composites for primary field: {primary_field_name} ({composite_type_tag})")
    
    # Determine target season specifications dynamically
    if seasons_filter or (use_precursor_seasons and precursor_name):
        target_season_specs = get_target_season_specs_for_precursor(
            precursor_name if use_precursor_seasons else "",
            STANDARD_ENSO_CYCLE_PLOT_SEQUENCE,
            seasons_filter
        )
        logging.info(f"Using dynamic season selection: {len(target_season_specs)} seasons")
    else:
        # Use the provided target_season_specs parameter (backward compatibility)
        logging.info(f"Using provided season specifications: {len(target_season_specs)} seasons")
    
    primary_field_config_local = target_spatial_fields_config_global[primary_field_name]
    primary_anomalies_dask = get_spatial_anomalies_field(
        primary_field_name, primary_field_config_local,
        clim_start_str, clim_end_str, analysis_slice
    )
    if primary_anomalies_dask is None:
        logging.error(f"Primary spatial anomaly field '{primary_field_name}' could not be prepared. Skipping.")
        return None
    
    logging.info(f"  Loading primary field '{primary_field_name}' into memory...")
    primary_anomalies_loaded = primary_anomalies_dask.load()
    del primary_anomalies_dask
    

    collected_seasonal_data_for_plotting = []
    wind_overlay_u_component_keys = primary_field_config_local.get('default_wind_overlays', [])

    for target_season_code, year_offset, display_suffix in target_season_specs:
        logging.info(f"  Targeting season: {display_suffix} for {primary_field_name}")
        current_season_results = {'display_suffix': display_suffix}

        # Process HIGH composite
        comp_high_primary, sig_high_primary, actual_high_contrib_years_primary = None, None, []
        if event_years_high:
            slices_high_primary_local, actual_high_contrib_years_primary = select_target_data_for_compositing(
                primary_anomalies_loaded, event_years_high, target_season_code, year_offset
            )
            if slices_high_primary_local is not None and slices_high_primary_local.time.size >= max(1, n_high_input // 2):
                # Standard horizontal composite
                comp_high_primary = slices_high_primary_local.mean(dim='time')
                if primary_field_config_local.get('plot_significant_composite', True):
                    sig_high_primary = calculate_fieldwise_ttest_significance(slices_high_primary_local, popmean=0.0, alpha=alpha)
        
        current_season_results['comp_high_primary'] = comp_high_primary
        current_season_results['sig_high_primary'] = sig_high_primary
        current_season_results['actual_high_contrib_years_primary'] = actual_high_contrib_years_primary
        
        # Process LOW composite
        comp_low_primary, sig_low_primary, actual_low_contrib_years_primary = None, None, []
        if event_years_low:
            slices_low_primary_local, actual_low_contrib_years_primary = select_target_data_for_compositing(
                primary_anomalies_loaded, event_years_low, target_season_code, year_offset
            )
            if slices_low_primary_local is not None and slices_low_primary_local.time.size >= max(1, n_low_input // 2):
                # Standard horizontal composite
                comp_low_primary = slices_low_primary_local.mean(dim='time')
                if primary_field_config_local.get('plot_significant_composite', True):
                    sig_low_primary = calculate_fieldwise_ttest_significance(slices_low_primary_local, popmean=0.0, alpha=alpha)
        
        current_season_results['comp_low_primary'] = comp_low_primary
        current_season_results['sig_low_primary'] = sig_low_primary
        current_season_results['actual_low_contrib_years_primary'] = actual_low_contrib_years_primary

        # Process DIFF composite
        comp_diff_primary, sig_diff_primary = None, None
        if comp_high_primary is not None and comp_low_primary is not None and \
           slices_high_primary_local is not None and slices_low_primary_local is not None:
            comp_diff_primary = comp_high_primary - comp_low_primary
            if primary_field_config_local.get('plot_significant_composite', True):
                if primary_field_config_local.get('is_vertical_section', False):
                    # For vertical sections, use equatorially averaged data
                    lat_band = (-5, 5)  # Default equatorial band for legacy compatibility
                    slices_high_eq = extract_latitude_band(slices_high_primary_local, lat_band)
                    slices_low_eq = extract_latitude_band(slices_low_primary_local, lat_band)
                    sig_diff_primary = calculate_fieldwise_ttest_significance(slices_high_eq, slices_low_eq, alpha=alpha)
                else:
                    sig_diff_primary = calculate_fieldwise_ttest_significance(slices_high_primary_local, slices_low_primary_local, alpha=alpha)
        
        current_season_results['comp_diff_primary'] = comp_diff_primary
        current_season_results['sig_diff_primary'] = sig_diff_primary

        # Process wind overlays
        current_season_results['wind_overlays_data'] = []
        for u_wind_key in wind_overlay_u_component_keys:
            u_wind_config = target_spatial_fields_config_global.get(u_wind_key)
            if not u_wind_config or not u_wind_config.get('is_vector_u_component'):
                continue
            
            v_wind_key = u_wind_config['v_component_key']
            v_wind_config = target_spatial_fields_config_global.get(v_wind_key)
            if not v_wind_config or not v_wind_config.get('is_vector_v_component'):
                continue

            plot_type = u_wind_config['vector_plot_type']
            special_scaling = u_wind_config.get('special_scaling')
            
            u_anom_dask = get_spatial_anomalies_field(u_wind_key, u_wind_config, clim_start_str, clim_end_str, analysis_slice)
            v_anom_dask = get_spatial_anomalies_field(v_wind_key, v_wind_config, clim_start_str, clim_end_str, analysis_slice)

            if u_anom_dask is None or v_anom_dask is None:
                continue
            
            u_anom_loaded = u_anom_dask.load()
            v_anom_loaded = v_anom_dask.load()
            del u_anom_dask, v_anom_dask

            wind_overlay_results = {
                'u_component_key': u_wind_key,
                'v_component_key': v_wind_key,
                'vector_plot_type': plot_type,
                'special_scaling': special_scaling
            }
            
            # Process each case
            for case, event_yrs, n_input_yrs in [
                ('high', event_years_high, n_high_input),
                ('low', event_years_low, n_low_input)
            ]:
                comp_u, comp_v, sig_mask_wind = None, None, None
                if event_yrs:
                    slices_u_this_case, _ = select_target_data_for_compositing(u_anom_loaded, event_yrs, target_season_code, year_offset)
                    slices_v_this_case, _ = select_target_data_for_compositing(v_anom_loaded, event_yrs, target_season_code, year_offset)

                    if slices_u_this_case is not None and slices_v_this_case is not None and \
                       slices_u_this_case.time.size >= max(1, n_input_yrs // 2) and slices_v_this_case.time.size >= max(1, n_input_yrs // 2):
                        comp_u = slices_u_this_case.mean(dim='time')
                        comp_v = slices_v_this_case.mean(dim='time')
                        if primary_field_config_local.get('plot_significant_composite', True):
                            pvals_u = calculate_fieldwise_ttest_significance(slices_u_this_case, popmean=0.0, alpha=alpha, return_pvals=True)
                            pvals_v = calculate_fieldwise_ttest_significance(slices_v_this_case, popmean=0.0, alpha=alpha, return_pvals=True)
                            if pvals_u is not None and pvals_v is not None:
                                sig_mask_wind = (pvals_u < alpha) | (pvals_v < alpha)
                            elif pvals_u is not None:
                                sig_mask_wind = (pvals_u < alpha)
                            elif pvals_v is not None:
                                sig_mask_wind = (pvals_v < alpha)
                            else:
                                sig_mask_wind = xr.full_like(comp_u, False, dtype=bool) if comp_u is not None else None

                wind_overlay_results[f'comp_{case}_u'] = comp_u
                wind_overlay_results[f'comp_{case}_v'] = comp_v
                wind_overlay_results[f'sig_mask_{case}_wind'] = sig_mask_wind
            
            # Process DIFF for winds
            comp_high_u = wind_overlay_results.get('comp_high_u')
            comp_low_u = wind_overlay_results.get('comp_low_u')
            comp_high_v = wind_overlay_results.get('comp_high_v')
            comp_low_v = wind_overlay_results.get('comp_low_v')
            
            if comp_high_u is not None and comp_low_u is not None and \
               comp_high_v is not None and comp_low_v is not None:
                wind_overlay_results['comp_diff_u'] = comp_high_u - comp_low_u
                wind_overlay_results['comp_diff_v'] = comp_high_v - comp_low_v
                
                if primary_field_config_local.get('plot_significant_composite', True) and event_years_high and event_years_low:
                    slices_high_u_for_diff, _ = select_target_data_for_compositing(u_anom_loaded, event_years_high, target_season_code, year_offset)
                    slices_low_u_for_diff, _ = select_target_data_for_compositing(u_anom_loaded, event_years_low, target_season_code, year_offset)
                    slices_high_v_for_diff, _ = select_target_data_for_compositing(v_anom_loaded, event_years_high, target_season_code, year_offset)
                    slices_low_v_for_diff, _ = select_target_data_for_compositing(v_anom_loaded, event_years_low, target_season_code, year_offset)
                    
                    if slices_high_u_for_diff is not None and slices_low_u_for_diff is not None and \
                       slices_high_v_for_diff is not None and slices_low_v_for_diff is not None:
                        pvals_u_diff = calculate_fieldwise_ttest_significance(slices_high_u_for_diff, slices_low_u_for_diff, alpha=alpha, return_pvals=True)
                        pvals_v_diff = calculate_fieldwise_ttest_significance(slices_high_v_for_diff, slices_low_v_for_diff, alpha=alpha, return_pvals=True)

                        if pvals_u_diff is not None and pvals_v_diff is not None:
                            wind_overlay_results['sig_mask_diff_wind'] = (pvals_u_diff < alpha) | (pvals_v_diff < alpha)
                        elif pvals_u_diff is not None:
                            wind_overlay_results['sig_mask_diff_wind'] = (pvals_u_diff < alpha)
                        elif pvals_v_diff is not None:
                            wind_overlay_results['sig_mask_diff_wind'] = (pvals_v_diff < alpha)
                        else:
                            wind_overlay_results['sig_mask_diff_wind'] = xr.full_like(wind_overlay_results['comp_diff_u'], False, dtype=bool) if wind_overlay_results.get('comp_diff_u') is not None else None

            current_season_results['wind_overlays_data'].append(wind_overlay_results)
        
        collected_seasonal_data_for_plotting.append(current_season_results)

    del primary_anomalies_loaded

    if not collected_seasonal_data_for_plotting:
        logging.warning(f"No data collected for any season for {primary_field_name}. Skipping plots and NetCDF.")
        return None

    # Generate outputs with single panel support
    if args and getattr(args, 'single_panel', False):
        # Single panel mode: generate individual files for each season and composite type
        season_filter = getattr(args, 'seasons', None) if args else None
        if season_filter:
            # Filter collected data to requested seasons
            filtered_data = []
            for season_data in collected_seasonal_data_for_plotting:
                season_code = season_data['display_suffix'].replace('(', '_').replace(')', '').replace('-', '_m')
                if any(s in season_code for s in season_filter):
                    filtered_data.append(season_data)
            if filtered_data:
                collected_seasonal_data_for_plotting = filtered_data
        
        # Generate separate plots for each season and composite type
        for season_data in collected_seasonal_data_for_plotting:
            single_season_data = [season_data]
            
            # High composite
            plot_horizontal_composite_figure(single_season_data, 'High',
                                           primary_field_name, primary_field_config_local, target_spatial_fields_config_global,
                                           composite_type_tag, output_dir_plots, alpha, n_high_input, 
                                           show_all_winds=show_all_winds, args=args)
            
            # Low composite
            plot_horizontal_composite_figure(single_season_data, 'Low',
                                           primary_field_name, primary_field_config_local, target_spatial_fields_config_global,
                                           composite_type_tag, output_dir_plots, alpha, n_low_input, 
                                           show_all_winds=show_all_winds, args=args)
            
            # Diff composite
            plot_horizontal_composite_figure(single_season_data, 'Diff',
                                           primary_field_name, primary_field_config_local, target_spatial_fields_config_global,
                                           composite_type_tag, output_dir_plots, alpha, n_high_input, n_low_input, 
                                           show_all_winds=show_all_winds, args=args)
    else:
        # Standard multi-panel plotting
        plot_horizontal_composite_figure(collected_seasonal_data_for_plotting, 'High',
                                         primary_field_name, primary_field_config_local, target_spatial_fields_config_global,
                                         composite_type_tag, output_dir_plots, alpha, n_high_input, 
                                         show_all_winds=show_all_winds, args=args)
        
        plot_horizontal_composite_figure(collected_seasonal_data_for_plotting, 'Low',
                                         primary_field_name, primary_field_config_local, target_spatial_fields_config_global,
                                         composite_type_tag, output_dir_plots, alpha, n_low_input, 
                                         show_all_winds=show_all_winds, args=args)
        
        plot_horizontal_composite_figure(collected_seasonal_data_for_plotting, 'Diff',
                                         primary_field_name, primary_field_config_local, target_spatial_fields_config_global,
                                         composite_type_tag, output_dir_plots, alpha, n_high_input, n_low_input, 
                                         show_all_winds=show_all_winds, args=args)
    
    save_consolidated_composites_netcdf(collected_seasonal_data_for_plotting, primary_field_name,
                                        primary_field_config_local, target_spatial_fields_config_global,
                                        composite_type_tag, output_dir_nc, alpha,
                                        clim_start_str, clim_end_str, analysis_slice,
                                        show_all_winds=show_all_winds)

    return collected_seasonal_data_for_plotting

# --- MAIN SCRIPT ---

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
        if conf.get('primary_plot_field', False) and not conf.get('is_vertical_section', False)
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
                temp_collected_data = generate_seasonal_composites_and_outputs(
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
                target_spatial_fields_config_global=TARGET_SPATIAL_FIELDS_CONFIG,
                clim_start_str=CLIM_START_STR,
                clim_end_str=CLIM_END_STR,
                analysis_slice=ANALYSIS_SLICE,
                show_all_winds=args.show_all_winds,
                args=args,
                precursor_name=precursor_name,
                use_precursor_seasons=getattr(args, 'precursor_based_seasons', False),
                seasons_filter=getattr(args, 'seasons', None)
                )
                if collected_data_for_csv_part1 is None and temp_collected_data:
                    collected_data_for_csv_part1 = temp_collected_data

            if collected_data_for_csv_part1:
                composite_type_tag_part1 = f"TopBottom{N_YEARS}_{precursor_name.replace(' ', '_').replace('(', '').replace(')', '')}"
                save_contributing_years_csv(
                    collected_data_for_csv_part1,
                    composite_type_tag_part1,
                    output_plots_dir_part1
                )


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
                temp_collected_data = generate_seasonal_composites_and_outputs(
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
                    target_spatial_fields_config_global=TARGET_SPATIAL_FIELDS_CONFIG,
                    clim_start_str=CLIM_START_STR,
                    clim_end_str=CLIM_END_STR,
                    analysis_slice=ANALYSIS_SLICE,
                    show_all_winds=args.show_all_winds,
                    args=args
                )
                if collected_data_for_csv_part3 is None and temp_collected_data:
                    collected_data_for_csv_part3 = temp_collected_data
    
            if collected_data_for_csv_part3:
                composite_type_tag_part3 = f"Tercile_3of4_{set_name}"
                save_contributing_years_csv(
                    collected_data_for_csv_part3,
                    composite_type_tag_part3,
                    output_plots_dir_part3
                )

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

        collected_data = generate_seasonal_composites_and_outputs(
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
            target_spatial_fields_config_global=TARGET_SPATIAL_FIELDS_CONFIG,
            clim_start_str=CLIM_START_STR,
            clim_end_str=CLIM_END_STR,
            analysis_slice=ANALYSIS_SLICE,
            show_all_winds=args.show_all_winds,
            args=args,
            precursor_name=precursor_name,
            use_precursor_seasons=getattr(args, 'precursor_based_seasons', False),
            seasons_filter=getattr(args, 'seasons', None)
        )

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
        
        collected_data = generate_seasonal_composites_and_outputs(
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
            target_spatial_fields_config_global=TARGET_SPATIAL_FIELDS_CONFIG,
            clim_start_str=CLIM_START_STR,
            clim_end_str=CLIM_END_STR,
            analysis_slice=ANALYSIS_SLICE,
            show_all_winds=args.show_all_winds,
            args=args,
            seasons_filter=getattr(args, 'seasons', None)
        )
        
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
                if conf.get('primary_plot_field', False) and not conf.get('is_vertical_section', False)
            ]:
                jobs.append(('Individual', precursor_name, field_name))

    # Set-based jobs (3 of 4 criterion)
    if args.job_type in ['all', 'set', 'set_3of4']:
        for set_name in PRECURSOR_SETS_3OF4_DICT.keys():
            for field_name in [args.field_key] if args.field_key else [
                key for key, conf in TARGET_SPATIAL_FIELDS_CONFIG.items()
                if conf.get('primary_plot_field', False) and not conf.get('is_vertical_section', False)
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
    parser.add_argument('--show_all_winds', action='store_true', default=False,
                        help="Show all wind vectors, with non-significant winds in lighter colors (default: show only significant)")
    parser.add_argument('--job_type', type=str, default='all',
                        choices=['all', 'individual', 'set', 'set_3of4'],
                        help="Type of jobs to run: 'all' (default), 'individual' (Part 1 only), or 'set'/'set_3of4' (Part 2 only)")
    parser.add_argument('--precursor_based_seasons', action='store_true', default=False,
                        help='Start season sequence from precursor definition season for individual precursors (default: start from JJA(-1))')
    parser.add_argument('--seasons', nargs='+', 
                        choices=['JJA(-1)', 'SON(-1)', 'DJF(0)', 'MAM(0)', 'JJA(0)', 'SON(0)', 'DJF(+1)'],
                        default=None,
                        help='Generate only specific seasons (overrides precursor-based selection)')
    
    # Add standard plotting optimization arguments (skip seasons since we define our own)
    add_plotting_arguments(parser, skip_seasons=True)

    args = parser.parse_args()
    launch_driver(args)
