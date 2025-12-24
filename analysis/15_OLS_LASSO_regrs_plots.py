"""
Plot multivariate regression results from OLS/Fixed-LASSO analysis.
This script loads results for OLS (EP/CP pathways) and LASSO_fixed_alpha (COMBINED pathway)
and produces:
  1. Individual predictor‐effect maps (OLS: significance masked, LASSO: non-zero masked)
  2. R² maps
for each applicable regression set and pathway.

ENHANCED VERSION with figure size optimization:
- Supports PNG, PDF, SVG, and "both" (PNG+PDF) output formats
- Uses hybrid rendering: rasterized contours + vector overlays for optimal file sizes
- Typical file size reduction: 90% smaller (e.g., 35MB → 3MB for complex figures)

Usage examples:
  # Default: process all variables with all lag sets (OLS only, including combined)
  python script.py

  # Process only SST variable (previous hardcoded behavior)
  python script.py --variables SST

  # Process specific variables
  python script.py --variables SST low_clouds prec

  # Process specific lag configurations
  python script.py --lag_sets lag6 lag6-3

  # Use both OLS and LASSO models
  python script.py --model_types OLS LASSO_fixed_alpha

  # Process without COMBINED pathways
  python script.py --no_combined

  # PDF output with hybrid rendering (recommended for LaTeX)
  python script.py --output_format pdf

  # Save both PNG and PDF versions
  python script.py --output_format both

  # Generate only DJF(0) and MAM(0) season as PDF
  python script.py --output_format pdf --seasons DJF_0 MAM_0

  # Generate DJF(0) and MAM(0) as separate PDF files
  python script.py --output_format pdf --seasons DJF_0 MAM_0 --single_panel

  # PDF with custom DPI for rasterized elements
  python script.py --output_format pdf --raster_dpi 200
"""

import os
import sys
from pathlib import Path
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
import cartopy
import cartopy.crs as ccrs
import logging
import warnings
import gc
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import argparse

# ----
# Suppress specific Cartopy warnings that are harmless but annoying
# ----
# The rasterized parameter works for file size reduction, but Cartopy's GeoAxes
# doesn't recognize it and produces warnings. This is a known Cartopy limitation.
warnings.filterwarnings('ignore', 
                       message='.*kwargs were not used by contour.*rasterized.*',
                       category=UserWarning,
                       module='cartopy.mpl.geoaxes')

# Add project root to Python path (src directory)
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.paths import get_data_path, get_results_path
    from utils.composite_utils import apply_equatorial_mask
    from utils.TBI_functions import add_cyclic_point_xr, add_cyclic_point_to_vectors
except ImportError:
    print("Could not import from utils, trying src.utils...")
    # Fallback if run from different directory
    from utils.paths import get_data_path, get_results_path
    from utils.composite_utils import apply_equatorial_mask
    from utils.TBI_functions import add_cyclic_point_xr, add_cyclic_point_to_vectors

# ----
# Global configuration
# ----
# Point to the directory where regression analysis saved its results
DATA_DIR = get_data_path('time_series/seas_multilag_regr_coeffs/')
SAVE_DIR = get_results_path('regressions/multilag_regr_plots_OLS_LASSO/')
os.makedirs(SAVE_DIR, exist_ok=True)

DEFAULT_SIGN_LEVEL = 95    # Significance level for OLS masking
DEFAULT_WIND_LEVEL = '10m'
DEFAULT_YEARS = ('1945', '2024')

# ----
# Figure output configuration
# ----
DEFAULT_OUTPUT_FORMAT = 'png'    # Default output format
RASTER_DPI = 200                 # DPI for rasterized elements in vector formats
VECTOR_DPI = 300                 # DPI for pure raster formats
SUPPORTED_FORMATS = ['png', 'pdf', 'svg', 'both']

# Season codes from regression analysis
SEASON_CODES = ['JJA_m1', 'SON_m1', 'DJF_0', 'MAM_0', 'JJA_0', 'SON_0', 'DJF_1']
# Map to display names
SEASON_MAPPING = {
    'JJA_m1': 'JJA(-1)', 'SON_m1': 'SON(-1)', 'DJF_0': 'DJF(0)',
    'MAM_0': 'MAM(0)', 'JJA_0': 'JJA(0)', 'SON_0': 'SON(0)', 'DJF_1': 'DJF(1)'
}

# Valid target variables for processing
VALID_VARIABLES = ['SST', 'low_clouds', 'prec', 'surf_pres', 'vp_200', 'sf_200', 'RWS_200']
# Valid lag configurations
VALID_LAG_SETS = ['lag6', 'lag6-5', 'lag6-4', 'lag6-3']
# Valid model types
VALID_MODEL_TYPES = ['OLS', 'LASSO_fixed_alpha']


def build_regression_sets(lag_sets: List[str], model_types: List[str], include_combined: bool) -> List[Tuple[str, str, None]]:
    """
    Build REGRESSION_SETS dynamically from CLI arguments.

    Args:
        lag_sets: List of lag configurations (e.g., ['lag6', 'lag6-3'])
        model_types: List of model types (e.g., ['OLS', 'LASSO_fixed_alpha'])
        include_combined: Whether to include COMBINED pathway sets

    Returns:
        List of (lag_range_str, method, None) tuples
    """
    regression_sets = []

    for lag in lag_sets:
        for model in model_types:
            # Individual pathway sets
            regression_sets.append((lag, model, None))
            # Combined pathway sets
            if include_combined:
                regression_sets.append((f"{lag}_COMBINED", model, None))

    return regression_sets

def get_season_code(season_input: str) -> str:
    """Convert an input season name (either code or display) into a standard season code."""
    if season_input in SEASON_CODES:
        return season_input
    for code, disp in SEASON_MAPPING.items():
        if season_input == disp:
            return code
    logging.warning(f"Unknown season format: {season_input}. Defaulting to {SEASON_CODES[0]}")
    return SEASON_CODES[0]

# ----
# Data loading
# ----
def load_regression_results(pathway: str, target_var: str, lag_range_str: str, method: str
    ) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray], Optional[xr.DataArray], Optional[xr.DataArray]]:
    """
    Load regression results for a given pathway, target variable, lag range, and method.
    Constructs the filename exactly as saved by regression analysis.

    Args:
        pathway (str): Pathway context ('EP', 'CP', 'COMBINED').
        target_var (str): Name of the target variable.
        lag_range_str (str): Lag range identifier (e.g., 'lag6-4', 'lag6-3_COMBINED').
        method (str): Method identifier ('OLS', 'LASSO_fixed_alpha').

    Returns:
        tuple: (regr_coefs, p_values, r_squared, fixed_alpha)
               p_values is None for LASSO. fixed_alpha is None for OLS.
    """
    results_subdir = f"{pathway}_pathway"
    # Construct the filename exactly as saved by regression analysis
    filename = f"{target_var}_{pathway}_{lag_range_str}_{method}.nc"
    filepath = os.path.join(DATA_DIR, results_subdir, filename)

    if not os.path.exists(filepath):
        logging.error(f"Regression results file not found: {filepath}")
        return None, None, None, None
    try:
        with xr.open_dataset(filepath) as ds:
            regr_coefs = ds['regr_coefs'].load()
            r_squared = ds['r_squared'].load()
            p_vals = None
            fixed_alpha_val = None

            if method.upper() == 'OLS':
                if 'p_values' in ds:
                    p_vals = ds['p_values'].load()
                else:
                    logging.warning(f"OLS results file missing 'p_values': {filepath}")
            # Check specifically for the fixed alpha method
            elif method.upper() == 'LASSO_FIXED_ALPHA':
                if 'fixed_alpha' in ds:
                    fixed_alpha_val = ds['fixed_alpha'].load()
                else:
                    # Log warning if fixed_alpha is expected but missing
                    logging.warning(f"LASSO_fixed_alpha results file missing 'fixed_alpha': {filepath}")

        # Return fixed_alpha_val (which might be None for OLS) as the 4th element
        return regr_coefs, p_vals, r_squared, fixed_alpha_val
    except Exception as e:
        logging.error(f"Error loading regression results from {filepath}: {e}")
        return None, None, None, None


def load_all_wind_data(pathway: str, lag_range_str: str, method: str,
                       wind_level: str, upper_wind_level: str) -> Dict[str, xr.DataArray]:
    """Load wind regression results (coefficients and p-values/fixed_alphas)."""
    wind_fields = []
    # Determine lower-level wind fields
    if wind_level == 'WAF':
        lower_winds = ['WAFx_200', 'WAFy_200']
    elif wind_level == '10m':
        lower_winds = ['U_10m', 'V_10m']
    elif wind_level == '850':
        lower_winds = ['U_850', 'V_850']
    else:
        logging.warning(f"Unsupported wind level {wind_level}. Defaulting to '10m'.")
        lower_winds = ['U_10m', 'V_10m']
    # Upper-level wind fields
    upper_winds = [f"U_{upper_wind_level}", f"V_{upper_wind_level}"]
    wind_fields.extend(lower_winds)
    wind_fields.extend(upper_winds)

    winds_data = {}
    for field in wind_fields:
        try:
            # Load coefficients, p-values (OLS) or fixed_alpha (LASSO_fixed_alpha)
            regr, p_vals, _, fixed_alpha = load_regression_results(pathway, field, lag_range_str, method)
            if regr is not None:
                winds_data[field] = regr # Store coefficients
                if p_vals is not None:
                    winds_data[f"pvals_{field}"] = p_vals # Store OLS p-values
                if fixed_alpha is not None:
                     # Store fixed alpha (though not used for masking, might be useful info)
                     winds_data[f"fixed_alpha_{field}"] = fixed_alpha

            else:
                 logging.warning(f"Failed to load any data for wind field {field} ({method}, {lag_range_str}, {pathway}).")

        except Exception as e:
            logging.warning(f"Could not load regression data for wind field {field} ({method}, {lag_range_str}, {pathway}): {e}")

    return winds_data

# ----
# Figure saving utilities
# ----

def save_figure_optimized(fig, base_filepath: str, output_format: str, 
                          raster_dpi: int, vector_dpi: int) -> None:
    """
    Save figure in the specified format(s) with optimized settings.
    Includes safety checks for PDF compatibility.
    
    Args:
        fig: matplotlib figure object
        base_filepath: base file path without extension
        output_format: 'png', 'pdf', 'svg', or 'both'
        raster_dpi: DPI for rasterized elements in vector formats
        vector_dpi: DPI for pure raster formats
    """
    save_params_base = {
        'bbox_inches': 'tight',
        'pad_inches': 0.1
    }
    
    # For PDF/SVG formats, check for potential infinite values
    if output_format in ['pdf', 'svg', 'both']:
        try:
            # This will catch any remaining problematic values before saving
            fig.canvas.draw()
        except ValueError as e:
            if "finite" in str(e).lower():
                logging.error(f"Non-finite values detected in figure data. This may cause PDF save issues: {e}")
                raise e
    
    if output_format == 'both':
        # Save both PNG and PDF
        try:
            # PNG version with high DPI
            png_filepath = f"{base_filepath}.png"
            fig.savefig(png_filepath, dpi=vector_dpi, **save_params_base)
            logging.info(f"Saved PNG: {png_filepath}")
            
            # PDF version with hybrid rendering
            pdf_filepath = f"{base_filepath}.pdf"
            fig.savefig(pdf_filepath, dpi=raster_dpi, **save_params_base)
            logging.info(f"Saved PDF: {pdf_filepath}")
        except Exception as save_err:
            logging.error(f"Failed to save figure in both formats: {save_err}")
    
    elif output_format == 'pdf':
        try:
            pdf_filepath = f"{base_filepath}.pdf"
            fig.savefig(pdf_filepath, dpi=raster_dpi, **save_params_base)
            logging.info(f"Saved PDF: {pdf_filepath}")
        except Exception as save_err:
            logging.error(f"Failed to save PDF: {save_err}")
    
    elif output_format == 'svg':
        try:
            svg_filepath = f"{base_filepath}.svg"
            fig.savefig(svg_filepath, dpi=raster_dpi, **save_params_base)
            logging.info(f"Saved SVG: {svg_filepath}")
        except Exception as save_err:
            logging.error(f"Failed to save SVG: {save_err}")
    
    else:  # Default to PNG
        try:
            png_filepath = f"{base_filepath}.png"
            fig.savefig(png_filepath, dpi=vector_dpi, **save_params_base)
            logging.info(f"Saved PNG: {png_filepath}")
        except Exception as save_err:
            logging.error(f"Failed to save PNG: {save_err}")

# ----
# Plotting functions
# ----

def plot_predictor_effect(
    regr_coefs: xr.DataArray,
    pvals: Optional[xr.DataArray], # Only used for OLS
    var_name: str,
    predictor_name: str,
    wind_data: Dict[str, xr.DataArray],
    method: str,    # Method identifier ('OLS', 'LASSO_fixed_alpha')
    lag_range_str: str, # Full lag range string
    sign_level: float = DEFAULT_SIGN_LEVEL, # Used only for OLS
    start_season: str = SEASON_CODES[0],
    path_save: str = SAVE_DIR,
    wind_level: str = DEFAULT_WIND_LEVEL,
    upper_wind_level: str = '200',
    config_name: str = "", # Corresponds to pathway ('EP', 'CP', 'COMBINED')
    years: Tuple[str, str] = DEFAULT_YEARS,
    show_all_winds: bool = False,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    raster_dpi: int = RASTER_DPI,
    vector_dpi: int = VECTOR_DPI,
    seasons_filter: Optional[List[str]] = None,  # New: specific seasons to plot
    single_panel_mode: bool = False,             # New: save panels individually
    panel_suffix: str = ''                       # New: suffix for panel files
) -> None:
    """
    Plot regression coefficients for a single predictor across seasons.
    Applies significance masking for OLS (using pvals and sign_level).
    Applies non-zero masking for LASSO_fixed_alpha (coefficients != 0).
    """
    # --- Settings dictionary remains the same ---
    if var_name in ['prec', 'surf_pres', 'SST', 'vp_200', 'low_clouds', 'sf_200', 'RWS_200']:
        if var_name == 'prec':
            settings = {
            'cmap': clrs.LinearSegmentedColormap.from_list("", ["maroon", "brown", "white", "limegreen", "seagreen"]),
            'levels': np.arange(-4, 4.1, 0.1), 'contour_levels': np.arange(-4, 4.2, 0.2),
            'conv_factor': 1.0, 'label': 'Precipitation [mm day$^{-1}$]' }
        elif var_name == 'SST':
            settings = {
            'cmap': 'coolwarm', 'levels': np.arange(-1.2, 1.3, 0.1), 'contour_levels': np.arange(-1.4, 1.6, 0.2),
            'conv_factor': 1.0, 'label': 'SST [K]' }
        elif var_name == 'RWS_200':
            settings = {
            'cmap': clrs.LinearSegmentedColormap.from_list("", ["darkblue", "blue", "white", "red", "darkred"]),
            'levels': np.arange(-10, 10.5, 0.5), 'contour_levels': None,
            'conv_factor': 1e-11, 'label': 'Rossby Wave Source (200 hPa) [10$^{-11}$ s$^{-2}$]' }
        elif var_name == 'low_clouds':
            settings = {
            'cmap': clrs.LinearSegmentedColormap.from_list("", ["blue", "deepskyblue", "white", "darkgrey", "dimgray"]),
            'levels': np.arange(-10, 11, 1), 'contour_levels': np.arange(-10, 12, 2),
            'conv_factor': 1.0, 'label': 'Low clouds [%]' }
        elif var_name == 'surf_pres':
            settings = {
                'cmap': clrs.LinearSegmentedColormap.from_list("", ["teal", "turquoise", "white", "hotpink", "purple"]),
                'levels': np.arange(-3, 3.25, 0.25), 'contour_levels': np.arange(-3, 3.5, 0.5),
                'conv_factor': 1.0, 'label': 'Surface pressure [hPa]'
            }
        elif var_name == 'vp_200':
            settings = {
                'cmap': 'BrBG_r',
                'levels': np.arange(-30, 32, 2), 'contour_levels': np.arange(-30, 40, 10),
                'conv_factor': 1e5, 'label': 'Velocity potential (200 hPa) [10$^{5}$ m$^{2}$ s$^{-1}$]'
            }
        elif var_name == 'sf_200':
            settings = {
                'cmap': 'coolwarm',
                'levels': np.arange(-60, 65, 5), 'contour_levels': np.arange(-60, 70, 10),
                'conv_factor': 1e5, 'label': 'Streamfunction (200 hPa) [10$^{5}$ m$^{2}$ s$^{-1}$]'
            }
        else: # Default for other specified vars
            settings = {
            'cmap': 'RdBu_r', 'levels': np.linspace(-2, 2, 21), 'contour_levels': np.linspace(-2, 2, 11),
            'conv_factor': 1.0, 'label': var_name }
    else: # Fallback default for unspecified vars
        settings = {
        'cmap': 'RdBu_r', 'levels': np.linspace(-2, 2, 21), 'contour_levels': np.linspace(-2, 2, 11),
        'conv_factor': 1.0, 'label': var_name }


    start_season_code = get_season_code(start_season)
    try:
        start_idx = SEASON_CODES.index(start_season_code)
    except ValueError:
        logging.warning(f"Start season code {start_season_code} not found in {SEASON_CODES}. Defaulting to first season.")
        start_idx = 0
    
    # Apply season filter if provided
    if seasons_filter:
        # Use only the specified seasons
        seasons_to_plot = [s for s in seasons_filter if s in SEASON_CODES]
        if not seasons_to_plot:
            logging.error(f"No valid seasons in filter: {seasons_filter}")
            return
    else:
        # Default behavior: all seasons from start_idx onwards
        seasons_to_plot = SEASON_CODES[start_idx:]

    if single_panel_mode:
        # Create and save individual figures for each season
        for season_code in seasons_to_plot:
            season_disp = SEASON_MAPPING.get(season_code, season_code)
            
            # Create single-panel figure
            fig, ax = plt.subplots(1, 1, figsize=(21, 6),
                                  subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
            
            # Plot single season using the same logic as the loop below
            try:
                # Select coefficient data for this season and predictor
                regr_var = regr_coefs.sel(season=season_code, predictor=predictor_name)

                # Apply masking based on method (same logic as original)
                if method.upper() == 'OLS':
                    if pvals is not None:
                        p_value = pvals.sel(season=season_code, predictor=predictor_name)
                        regr_var = regr_var.where(p_value < (1 - sign_level / 100.0))
                        mask_info = f" (masked p>{1-sign_level/100:.2f})"
                    else:
                        logging.warning(f"OLS method selected but p-values not available for {var_name}, {predictor_name}, {season_code}. No masking applied.")
                        mask_info = " (p-vals missing)"
                elif 'LASSO' in method.upper():
                    regr_var = regr_var.where(regr_var != 0)
                    mask_info = " (masked zero coefs)"
                else:
                     mask_info = f" (unknown method: {method})"

                # Apply conversion factor AFTER masking
                regr_var_scaled = regr_var / settings['conv_factor']

                # Check if all data is masked out
                if regr_var_scaled.isnull().all():
                     logging.warning(f"All data masked out for {var_name}, {predictor_name}, {season_code}. Skipping this season.")
                     plt.close(fig)
                     continue

                # Add cyclic point and plot (same logic as original)
                regr_var_scaled_cyclic = add_cyclic_point_xr(regr_var_scaled)
                
                # Plotting with hybrid rendering
                use_rasterized = output_format in ['pdf', 'svg', 'both']
                
                contourf_kwargs = {
                    'transform': ccrs.PlateCarree(),
                    'levels': settings['levels'],
                    'cmap': settings['cmap'],
                    'extend': 'both'
                }
                if use_rasterized:
                    contourf_kwargs['rasterized'] = True
                
                cf = ax.contourf(regr_var_scaled_cyclic.longitude, regr_var_scaled_cyclic.latitude, regr_var_scaled_cyclic.values,
                                 **contourf_kwargs)

                # Plot contours if defined
                if settings.get('contour_levels') is not None:
                    try:
                        cs = ax.contour(regr_var_scaled_cyclic.longitude, regr_var_scaled_cyclic.latitude, regr_var_scaled_cyclic.values,
                                        transform=ccrs.PlateCarree(), levels=settings['contour_levels'],
                                        colors='darkgray', linewidths=0.5)
                        plt.clabel(cs, settings['contour_levels'], fmt='%1.1f', fontsize='small', colors='dimgrey',
                                   inline=True, inline_spacing=4)
                    except ValueError as e:
                        logging.warning(f"Could not draw/label contours for {var_name} in season {season_code}: {e}")

                # Add wind plotting (same logic as original, condensed)
                if show_all_winds:
                    u_field_all, v_field_all = get_wind_vectors(wind_data, season_code, predictor_name, method, sign_level, wind_level, apply_mask=False, equatorial_mask_deg=10.0 if wind_level == 'WAF' else None)
                    u_upper_all, v_upper_all, lw_all = get_upper_circulation(wind_data, season_code, predictor_name, method, sign_level, upper_wind_level, apply_mask=False)
                    
                    # Plot all wind quivers in light gray (condensed logic)
                    if u_field_all is not None and v_field_all is not None and not (np.isnan(u_field_all.values).all() or np.isnan(v_field_all.values).all()):
                        u_field_all_cyclic, v_field_all_cyclic = add_cyclic_point_to_vectors(u_field_all, v_field_all)
                        scale = 30 if wind_level == '10m' else 37
                        ax.quiver(u_field_all_cyclic.longitude, u_field_all_cyclic.latitude,
                                  u_field_all_cyclic.values, v_field_all_cyclic.values,
                                  transform=ccrs.PlateCarree(), color='gray', alpha=0.4,
                                  scale=scale, zorder=3)
                    
                    # Plot all upper circulation streamlines in light green (condensed)
                    if u_upper_all is not None and v_upper_all is not None and lw_all is not None and not (np.isnan(u_upper_all.values).all() or np.isnan(v_upper_all.values).all()):
                        try:
                            u_upper_all_cyclic, v_upper_all_cyclic = add_cyclic_point_to_vectors(u_upper_all, v_upper_all)
                            lw_all_cyclic = add_cyclic_point_xr(lw_all)
                            u_filled = np.nan_to_num(u_upper_all_cyclic.values, nan=0.0)
                            v_filled = np.nan_to_num(v_upper_all_cyclic.values, nan=0.0)
                            lw_filled = np.nan_to_num(lw_all_cyclic.values, nan=0.0)
                            ax.streamplot(u_upper_all_cyclic.longitude, u_upper_all_cyclic.latitude,
                                          u_filled, v_filled,
                                          density=1.1, color='lightgreen', maxlength=1.5,
                                          arrowsize=1.7, linewidth=lw_filled * 0.5,
                                          transform=ccrs.PlateCarree(), zorder=2)
                        except Exception as e:
                            logging.warning(f"Failed to plot all winds streamlines: {e}")

                # Plot significant winds (condensed logic)
                u_field, v_field = get_wind_vectors(wind_data, season_code, predictor_name, method, sign_level, wind_level, apply_mask=True, equatorial_mask_deg=10.0 if wind_level == 'WAF' else None)
                u_upper, v_upper, lw = get_upper_circulation(wind_data, season_code, predictor_name, method, sign_level, upper_wind_level, apply_mask=True)

                # Plot significant wind quivers in black (condensed)
                if u_field is not None and v_field is not None and not (np.isnan(u_field.values).all() or np.isnan(v_field.values).all()):
                    u_field_cyclic, v_field_cyclic = add_cyclic_point_to_vectors(u_field, v_field)
                    scale = 30 if wind_level == '10m' else 37
                    ax.quiver(u_field_cyclic.longitude, u_field_cyclic.latitude,
                              u_field_cyclic.values, v_field_cyclic.values,
                              transform=ccrs.PlateCarree(), color='black', scale=scale, zorder=5)

                # Plot significant upper circulation streamlines (condensed)
                if u_upper is not None and v_upper is not None and lw is not None and not (np.isnan(u_upper.values).all() or np.isnan(v_upper.values).all()):
                    try:
                        u_upper_cyclic, v_upper_cyclic = add_cyclic_point_to_vectors(u_upper, v_upper)
                        lw_cyclic = add_cyclic_point_xr(lw)
                        u_filled = np.nan_to_num(u_upper_cyclic.values, nan=0.0)
                        v_filled = np.nan_to_num(v_upper_cyclic.values, nan=0.0)
                        lw_filled = np.nan_to_num(lw_cyclic.values, nan=0.0)
                        ax.streamplot(u_upper_cyclic.longitude, u_upper_cyclic.latitude,
                                      u_filled, v_filled,
                                      density=1.1, color='forestgreen', maxlength=1.5,
                                      arrowsize=1.7, linewidth=lw_filled,
                                      transform=ccrs.PlateCarree(), zorder=4)
                    except Exception as e:
                        logging.warning(f"Failed to plot significant winds streamlines: {e}")

                # Axis setup
                ax.set_title(f"Season {season_disp}", fontsize=18)
                # Variable-specific extent: extend south to 70S for RWS/sf
                lat_min, lat_max = (-70, 30) if var_name in ['RWS_200', 'sf_200'] else (-50, 50)
                ax.set_extent([0, 359.99, lat_min, lat_max], crs=ccrs.PlateCarree())
                ax.add_feature(cartopy.feature.BORDERS, linewidth=0.6)
                ax.coastlines()
                gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
                gl.xlabel_style = {'size': 14}
                gl.ylabel_style = {'size': 14}
                gl.top_labels = False
                gl.right_labels = False

                # Add equatorial boundary indicators for WAF vectors
                if wind_level == 'WAF':
                    ax.axhline(y=10, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
                    ax.axhline(y=-10, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

                # Add colorbar 
                cbar = fig.colorbar(cf, ax=ax, orientation='horizontal', pad=0.10, fraction=0.05)
                cbar.set_label(settings['label'], fontsize=16)
                cbar.ax.tick_params(labelsize=14)

                # Save individual panel
                if method.upper() == 'OLS':
                    mask_desc = f"{sign_level}% conf"
                elif 'LASSO' in method.upper():
                    mask_desc = "non-zero coefs"
                else:
                    mask_desc = "unknown"

                # All information is preserved in the filename and individual panel title

                # Define output directory and filename for single panel
                dir_components = [path_save, config_name, method.lower(), lag_range_str]
                output_dir = os.path.join(*dir_components)
                os.makedirs(output_dir, exist_ok=True)

                mask_desc_clean = mask_desc.replace('% conf','pct').replace(' ','_')
                all_winds_suffix = "_all_winds" if show_all_winds else ""
                season_disp_clean = season_disp.replace('(', '').replace(')', '').replace('-', 'm')
                filename_base = f"{method.lower()}_{var_name}_partial_{predictor_name.replace('.', '_')}_{config_name}_{lag_range_str}_{season_disp_clean}_{wind_level}_UL{upper_wind_level}_mask_{mask_desc_clean}{all_winds_suffix}{panel_suffix}"
                filepath_base = os.path.join(output_dir, filename_base)

                # Save the single panel figure
                save_figure_optimized(fig, filepath_base, output_format, raster_dpi, vector_dpi)

            except Exception as plot_err:
                logging.error(f"Error plotting season {season_code} for {var_name}, predictor {predictor_name}: {plot_err}", exc_info=True)
            finally:
                plt.close(fig)
                gc.collect()
        
        return  # Exit function after handling single panel mode
    
    # Multi-panel
    n_seasons = len(seasons_to_plot)
    fig_height = 6 * n_seasons
    fig, axs = plt.subplots(n_seasons, 1, figsize=(21, fig_height),
                            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    if n_seasons == 1:
        axs = [axs] # Ensure axs is always iterable

    cf_mappable = None # To store the last contourf for the colorbar

    for i, season_code in enumerate(seasons_to_plot):
        season_disp = SEASON_MAPPING.get(season_code, season_code)
        ax = axs[i]

        try:
            # Select coefficient data for this season and predictor
            regr_var = regr_coefs.sel(season=season_code, predictor=predictor_name)

            # --- Apply Masking based on Method ---
            if method.upper() == 'OLS':
                if pvals is not None:
                    p_value = pvals.sel(season=season_code, predictor=predictor_name)
                    regr_var = regr_var.where(p_value < (1 - sign_level / 100.0))
                    mask_info = f" (masked p>{1-sign_level/100:.2f})"
                else:
                    logging.warning(f"OLS method selected but p-values not available for {var_name}, {predictor_name}, {season_code}. No masking applied.")
                    mask_info = " (p-vals missing)"
            elif 'LASSO' in method.upper(): # Catch LASSO_fixed_alpha
                regr_var = regr_var.where(regr_var != 0)
                mask_info = " (masked zero coefs)"
            else:
                 mask_info = f" (unknown method: {method})"


            # Apply conversion factor AFTER masking
            regr_var_scaled = regr_var / settings['conv_factor']

            # Check if all data is masked out
            if regr_var_scaled.isnull().all():
                 logging.warning(f"All data masked out for {var_name}, {predictor_name}, {season_code}. Skipping plot for this season.")
                 ax.text(0.5, 0.5, "All data masked", transform=ax.transAxes, ha='center', va='center', fontsize=12, color='gray')
                 ax.set_title(f"Season {season_code}{mask_info}", fontsize=16)
                 continue # Skip to next season


            # Add cyclic point to eliminate dateline gap
            regr_var_scaled_cyclic = add_cyclic_point_xr(regr_var_scaled)
            
            # Plotting the main variable (contourf) with hybrid rendering
            # Use rasterization for filled contours in vector formats to reduce file size
            use_rasterized = output_format in ['pdf', 'svg', 'both']
            
            # Only pass rasterized parameter when it's True to avoid Cartopy warnings
            contourf_kwargs = {
                'transform': ccrs.PlateCarree(),
                'levels': settings['levels'],
                'cmap': settings['cmap'],
                'extend': 'both'
            }
            if use_rasterized:
                contourf_kwargs['rasterized'] = True
            
            cf = ax.contourf(regr_var_scaled_cyclic.longitude, regr_var_scaled_cyclic.latitude, regr_var_scaled_cyclic.values,
                             **contourf_kwargs)
            cf_mappable = cf # Store for colorbar

            # Plot contours if defined
            if settings.get('contour_levels') is not None:
                try:
                    cs = ax.contour(regr_var_scaled_cyclic.longitude, regr_var_scaled_cyclic.latitude, regr_var_scaled_cyclic.values,
                                    transform=ccrs.PlateCarree(), levels=settings['contour_levels'],
                                    colors='darkgray', linewidths=0.5)
                    plt.clabel(cs, settings['contour_levels'], fmt='%1.1f', fontsize='small', colors='dimgrey',
                               inline=True, inline_spacing=4)
                except ValueError as e:
                    # This might catch cases where contouring is impossible (e.g., all NaNs)
                    logging.warning(f"Could not draw/label contours for {var_name} in season {season_code}: {e}")

            # --- Two-Pass Wind Plotting Logic ---
            if show_all_winds:
                # First pass: Plot all winds with lighter colors
                u_field_all, v_field_all = get_wind_vectors(wind_data, season_code, predictor_name, method, sign_level, wind_level, apply_mask=False, equatorial_mask_deg=10.0 if wind_level == 'WAF' else None)
                u_upper_all, v_upper_all, lw_all = get_upper_circulation(wind_data, season_code, predictor_name, method, sign_level, upper_wind_level, apply_mask=False)
                
                # Plot all wind quivers in light gray
                if u_field_all is not None and v_field_all is not None:
                    if not (np.isnan(u_field_all.values).all() or np.isnan(v_field_all.values).all()):
                        # Add cyclic points to vector components
                        u_field_all_cyclic, v_field_all_cyclic = add_cyclic_point_to_vectors(u_field_all, v_field_all)
                        
                        scale = 30 if wind_level == '10m' else 37
                        ax.quiver(u_field_all_cyclic.longitude, u_field_all_cyclic.latitude,
                                  u_field_all_cyclic.values, v_field_all_cyclic.values,
                                  transform=ccrs.PlateCarree(), color='gray', alpha=0.4,
                                  scale=scale, zorder=3)
                
                # Plot all upper circulation streamlines in light green
                if u_upper_all is not None and v_upper_all is not None and lw_all is not None:
                    if not (np.isnan(u_upper_all.values).all() or np.isnan(v_upper_all.values).all()):
                        try:
                            # Add cyclic points to vector components and linewidth
                            u_upper_all_cyclic, v_upper_all_cyclic = add_cyclic_point_to_vectors(u_upper_all, v_upper_all)
                            lw_all_cyclic = add_cyclic_point_xr(lw_all)
                            
                            # Replace NaN values with zeros for streamplot
                            u_filled = np.nan_to_num(u_upper_all_cyclic.values, nan=0.0)
                            v_filled = np.nan_to_num(v_upper_all_cyclic.values, nan=0.0)
                            lw_filled = np.nan_to_num(lw_all_cyclic.values, nan=0.0)
                            
                            ax.streamplot(u_upper_all_cyclic.longitude, u_upper_all_cyclic.latitude,
                                          u_filled, v_filled,
                                          density=1.1, color='lightgreen', maxlength=1.5,
                                          arrowsize=1.7, linewidth=lw_filled * 0.5,
                                          transform=ccrs.PlateCarree(), zorder=2)
                        except Exception as e:
                            logging.warning(f"Failed to plot all winds streamlines: {e}")

            # Second pass (or only pass if show_all_winds is False): Plot significant winds
            u_field, v_field = get_wind_vectors(wind_data, season_code, predictor_name, method, sign_level, wind_level, apply_mask=True, equatorial_mask_deg=10.0 if wind_level == 'WAF' else None)
            u_upper, v_upper, lw = get_upper_circulation(wind_data, season_code, predictor_name, method, sign_level, upper_wind_level, apply_mask=True)

            # Plot significant wind quivers in black
            if u_field is not None and v_field is not None:
                if not (np.isnan(u_field.values).all() or np.isnan(v_field.values).all()):
                    # Add cyclic points to vector components
                    u_field_cyclic, v_field_cyclic = add_cyclic_point_to_vectors(u_field, v_field)
                    
                    scale = 30 if wind_level == '10m' else 37
                    ax.quiver(u_field_cyclic.longitude, u_field_cyclic.latitude,
                              u_field_cyclic.values, v_field_cyclic.values,
                              transform=ccrs.PlateCarree(), color='black', scale=scale, zorder=5)

            # Plot significant upper circulation streamlines in forest green
            if u_upper is not None and v_upper is not None and lw is not None:
                if not (np.isnan(u_upper.values).all() or np.isnan(v_upper.values).all()):
                    try:
                        # Add cyclic points to vector components and linewidth
                        u_upper_cyclic, v_upper_cyclic = add_cyclic_point_to_vectors(u_upper, v_upper)
                        lw_cyclic = add_cyclic_point_xr(lw)
                        
                        # Replace NaN values with zeros for PDF compatibility
                        u_filled = np.nan_to_num(u_upper_cyclic.values, nan=0.0)
                        v_filled = np.nan_to_num(v_upper_cyclic.values, nan=0.0)
                        lw_filled = np.nan_to_num(lw_cyclic.values, nan=0.0)
                        
                        ax.streamplot(u_upper_cyclic.longitude, u_upper_cyclic.latitude,
                                      u_filled, v_filled,
                                      density=1.1, color='forestgreen', maxlength=1.5,
                                      arrowsize=1.7, linewidth=lw_filled,
                                      transform=ccrs.PlateCarree(), zorder=4)
                    except Exception as e:
                        logging.warning(f"Failed to plot significant winds streamlines: {e}")

            # Axis setup
            ax.set_title(f"Season {season_disp}", fontsize=18)
            # Variable-specific extent: extend south to 70S for RWS/sf
            lat_min, lat_max = (-70, 30) if var_name in ['RWS_200', 'sf_200'] else (-50, 50)
            ax.set_extent([0, 359.99, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax.add_feature(cartopy.feature.BORDERS, linewidth=0.6)
            ax.coastlines()
            gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
            gl.xlabel_style = {'size': 14}
            gl.ylabel_style = {'size': 14}
            gl.top_labels = False
            gl.right_labels = False

            # Add equatorial boundary indicators for WAF vectors
            if wind_level == 'WAF':
                # Add thin dotted lines at equatorial mask boundaries
                ax.axhline(y=10, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
                ax.axhline(y=-10, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

        except Exception as plot_err:
             # Log other potential errors during plotting for a season
             logging.error(f"Error plotting season {season_code} for {var_name}, predictor {predictor_name}: {plot_err}", exc_info=True)
             ax.text(0.5, 0.5, "Plotting Error", transform=ax.transAxes, ha='center', va='center', fontsize=12, color='red')
             ax.set_title(f"Season {season_code} - ERROR", fontsize=16)


    # Add a single colorbar at the bottom after the loop
    if cf_mappable is not None:
        # cbar = fig.colorbar(cf_mappable, ax=axs, orientation='horizontal', pad=0.10, fraction=0.015) # 5 panels
        cbar = fig.colorbar(cf_mappable, ax=axs, orientation='horizontal', pad=0.10, fraction=0.015) # 4 panels
        # cbar = fig.colorbar(cf_mappable, ax=axs, orientation='horizontal', pad=0.10, fraction=0.08) # single panel
        cbar.set_label(settings['label'], fontsize=16)
        cbar.ax.tick_params(labelsize=14)
    else:
        logging.warning(f"No valid data plotted across seasons for {var_name}, predictor {predictor_name}. Skipping colorbar.")

    # Overall figure title
    if method.upper() == 'OLS':
        mask_desc = f"{sign_level}% conf"
    elif 'LASSO' in method.upper():
        mask_desc = "non-zero coefs"
    else:
        mask_desc = "unknown"

    pathway_str = f" [{config_name}]" if config_name else ""
    # plt.suptitle(f"{predictor_name} partial effect on {var_name} ({method}, {lag_range_str}){pathway_str} with {wind_level} winds (UL: {upper_wind_level}hPa) " +
    #              f"at {mask_desc}) [{years[0]}-{years[1]}]", fontsize=18, y=0.99)
    # plt.subplots_adjust(left=0.04, bottom=0.14, right=0.96, top=0.95, hspace=0.16) # 5 panels
    plt.subplots_adjust(left=0.04, bottom=0.14, right=0.96, top=0.95, hspace=0.16) # 4 panels
    # plt.subplots_adjust(left=0.04, bottom=0.23, right=0.96, top=0.95, hspace=0.15) # 1 panel

    # Define output directory and filename
    dir_components = [path_save, config_name, method.lower(), lag_range_str]
    output_dir = os.path.join(*dir_components)
    os.makedirs(output_dir, exist_ok=True)

    mask_desc_clean = mask_desc.replace('% conf','pct').replace(' ','_')
    all_winds_suffix = "_all_winds" if show_all_winds else ""
    filename_base = f"{method.lower()}_{var_name}_partial_{predictor_name.replace('.', '_')}_{config_name}_{lag_range_str}_{wind_level}_UL{upper_wind_level}_mask_{mask_desc_clean}{all_winds_suffix}"
    filepath_base = os.path.join(output_dir, filename_base)

    # Save the figure using optimized format handling
    try:
        save_figure_optimized(fig, filepath_base, output_format, raster_dpi, vector_dpi)
    except Exception as save_err:
        logging.error(f"Failed to save plot {filepath_base}: {save_err}")
    finally:
        plt.close(fig)
        gc.collect()

def plot_r_squared(
    rsq: xr.DataArray,
    var_name: str,
    predictors: List[str], # List of predictors included in this model run
    method: str,
    lag_range_str: str, # Full lag range string
    start_season: str = SEASON_CODES[0],
    path_save: str = SAVE_DIR,
    config_name: str = "", # Corresponds to pathway ('EP', 'CP', 'COMBINED')
    years: Tuple[str, str] = DEFAULT_YEARS,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
    raster_dpi: int = RASTER_DPI,
    vector_dpi: int = VECTOR_DPI,
    seasons_filter: Optional[List[str]] = None,  # New: specific seasons to plot
    single_panel_mode: bool = False,             # New: save panels individually
    panel_suffix: str = ''                       # New: suffix for panel files
) -> None:
    """
    Plot the spatial distribution of R² across seasons.
    """
    logging.info(f"Plotting R² for {var_name} ({method}, {lag_range_str}, Pathway: {config_name})")
    # Pre-check for all NaNs or Infs
    if rsq is None or np.all(np.isnan(rsq)):
        logging.error(f"R² data is None or all NaN for {var_name} ({method}, {lag_range_str}, {config_name}). Skipping R² plot.")
        return
    rsq = rsq.where(~np.isinf(rsq), np.nan) # Replace infinities with NaN

    start_season_code = get_season_code(start_season)
    try:
        start_idx = SEASON_CODES.index(start_season_code)
    except ValueError:
        logging.warning(f"Start season code {start_season_code} not found. Defaulting to first season.")
        start_idx = 0
    
    # Apply season filter if provided (same logic as plot_predictor_effect)
    if seasons_filter:
        # Use only the specified seasons
        seasons_to_plot = [s for s in seasons_filter if s in SEASON_CODES]
        if not seasons_to_plot:
            logging.error(f"No valid seasons in filter: {seasons_filter}")
            return
    else:
        # Default behavior: all seasons from start_idx onwards
        seasons_to_plot = SEASON_CODES[start_idx:]

    n_seasons = len(seasons_to_plot)
    fig_height = 6 * n_seasons
    fig, axs = plt.subplots(n_seasons, 1, figsize=(21, fig_height),
                            subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    if n_seasons == 1:
        axs = [axs] # Ensure axs is always iterable

    valid_cf = None # To store the last pcolormesh for the colorbar

    for i, season_code in enumerate(seasons_to_plot):
        season_disp = SEASON_MAPPING.get(season_code, season_code)
        ax = axs[i]
        try:
            r_data = rsq.sel(season=season_code)
            # Check if data for this season is all NaN
            if np.all(np.isnan(r_data)):
                logging.warning(f"All R² values are NaN for {var_name}, season {season_code}")
                ax.text(0.5, 0.5, "No valid R² data", transform=ax.transAxes,
                        ha='center', va='center', fontsize=14, color='gray')
                ax.set_title(f"{season_disp}: R² for {var_name}", fontsize=16)
                continue # Skip plotting for this season

            # Add cyclic point to eliminate dateline gap
            r_data_cyclic = add_cyclic_point_xr(r_data)
            
            # Use pcolormesh for robustness with hybrid rendering
            use_rasterized = output_format in ['pdf', 'svg', 'both']
            
            # Only pass rasterized parameter when it's True to avoid Cartopy warnings
            pcolormesh_kwargs = {
                'transform': ccrs.PlateCarree(),
                'cmap': 'Blues',
                'vmin': 0,
                'vmax': 1,
                'shading': 'auto'
            }
            if use_rasterized:
                pcolormesh_kwargs['rasterized'] = True
            
            cf = ax.pcolormesh(r_data_cyclic.longitude, r_data_cyclic.latitude, r_data_cyclic.values,
                               **pcolormesh_kwargs)
            valid_cf = cf # Store the mappable object

            # Add contours
            try:
                cs = ax.contour(r_data_cyclic.longitude, r_data_cyclic.latitude, r_data_cyclic.values,
                                transform=ccrs.PlateCarree(), levels=np.arange(0.1, 1.0, 0.1),
                                colors='darkgray', linewidths=0.5)
                plt.clabel(cs, fmt='%.1f', fontsize='small', colors='dimgrey', inline=True)
            except Exception as contour_e:
                # Log if contouring or labeling fails for other reasons
                logging.warning(f"Could not draw/label R² contours for {var_name}, season {season_code}: {contour_e}")

            # Axis setup
            ax.set_title(f"{season_disp}: R² for {var_name}", fontsize=18)
            lat_min, lat_max = (-70, 30) if var_name in ['RWS_200', 'sf_200'] else (-50, 50)
            ax.set_extent([0, 359.99, lat_min, lat_max], crs=ccrs.PlateCarree())
            ax.add_feature(cartopy.feature.BORDERS, linewidth=0.6)
            ax.coastlines()
            gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
            gl.xlabel_style = {'size': 14}
            gl.ylabel_style = {'size': 14}
            gl.top_labels = False
            gl.right_labels = False

        except Exception as plot_err:
             logging.error(f"Error plotting R² for season {season_code}, {var_name}: {plot_err}", exc_info=True)
             ax.text(0.5, 0.5, "Plotting Error", transform=ax.transAxes, ha='center', va='center', fontsize=12, color='red')
             ax.set_title(f"{season_disp}: R² for {var_name} - ERROR", fontsize=16)


    # Add colorbar at the bottom
    if valid_cf is not None:
        cbar = fig.colorbar(valid_cf, ax=axs, orientation='horizontal', pad=0.20, fraction=0.08)
        cbar.set_label('R² (explained variance)', fontsize=16)
        cbar.ax.tick_params(labelsize=14)
    else:
        logging.warning(f"No valid R² plot created for {var_name}; skipping colorbar.")

    # Format predictor list for title
    if len(predictors) > 3:
         predictor_title_part = f"{len(predictors)} predictors"
    else:
         predictor_title_part = "_and_".join([p.replace('.', '_') for p in predictors])

    # Include pathway (config_name) in title
    pathway_str = f" [{config_name}]" if config_name else ""
    # plt.suptitle(f"Explained variance (R²) of {var_name} by {predictor_title_part}{pathway_str} " +
    #              f"({method}, {lag_range_str}) [{years[0]}-{years[1]}]", fontsize=18, y=0.99)
    plt.subplots_adjust(left=0.04, bottom=0.20, right=0.96, top=0.95, hspace=0.15)

    # Define output directory and filename
    # Include pathway (config_name) in directory structure
    dir_components = [path_save, config_name, method.lower(), lag_range_str, "r_squared"]
    output_dir = os.path.join(*dir_components)
    os.makedirs(output_dir, exist_ok=True)

    filename_base = f"r_squared_{method.lower()}_{lag_range_str}_{config_name}_{var_name}_by_{predictor_title_part}"
    filepath_base = os.path.join(output_dir, filename_base)

    # Save the figure using optimized format handling
    try:
        save_figure_optimized(fig, filepath_base, output_format, raster_dpi, vector_dpi)
    except Exception as e:
        logging.error(f"Failed to save R² plot {filepath_base}: {e}")
    finally:
        plt.close(fig)
        gc.collect()


# --- Wind Masking Functions ---
def get_wind_vectors(wind_data: Dict[str, xr.DataArray],
                     season: str,
                     predictor: str,
                     method: str, # Method identifier ('OLS', 'LASSO_fixed_alpha')
                     sign_level: float, # Used for OLS
                     wind_level: str,
                     apply_mask: bool = True,
                     equatorial_mask_deg: Optional[float] = None
                     ) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray]]:
    """
    Retrieve lower-level wind vector coefficients for plotting.
    Applies masking based on method: p-value for OLS, non-zero for LASSO variants.
    Applies scaling for WAF vectors using atmospheric dynamics methodology.
    """
    # Determine field keys based on wind_level
    if wind_level == 'WAF':
        u_key, v_key = 'WAFx_200', 'WAFy_200'
    elif wind_level == '10m':
        u_key, v_key = 'U_10m', 'V_10m'
    elif wind_level == '850':
        u_key, v_key = 'U_850', 'V_850'
    else:
        logging.warning(f"Unsupported wind level {wind_level} in get_wind_vectors.")
        return None, None

    # Check if coefficient data exists
    if u_key not in wind_data or v_key not in wind_data:
        logging.warning(f"Missing wind coefficient data for {u_key} or {v_key} ({method}, {season}, {predictor}).")
        return None, None

    # Select coefficients for the specific season and predictor
    u_coeffs = wind_data[u_key].sel(season=season, predictor=predictor)
    v_coeffs = wind_data[v_key].sel(season=season, predictor=predictor)

    # Apply masking based on method
    mask = None
    if apply_mask:  # Add this condition
        if method.upper() == 'OLS':
            u_pval_key, v_pval_key = f"pvals_{u_key}", f"pvals_{v_key}"
            if u_pval_key in wind_data and v_pval_key in wind_data:
                u_pval = wind_data[u_pval_key].sel(season=season, predictor=predictor)
                v_pval = wind_data[v_pval_key].sel(season=season, predictor=predictor)
                mask = (u_pval < (1 - sign_level / 100.0)) | (v_pval < (1 - sign_level / 100.0))
            else:
                logging.warning(f"Missing OLS p-values for wind masking ({u_key}, {v_key}). No mask applied.")
        elif 'LASSO' in method.upper(): # Catch LASSO_fixed_alpha
            # Mask where *either* coefficient is non-zero
            mask = (u_coeffs != 0) | (v_coeffs != 0)

    # Apply the mask if it was created
    if mask is not None:
        u_coeffs_masked = u_coeffs.where(mask)
        v_coeffs_masked = v_coeffs.where(mask)
    else: # No mask applied
        u_coeffs_masked = u_coeffs
        v_coeffs_masked = v_coeffs

    # Apply equatorial masking for WAF vectors
    if wind_level == 'WAF' and equatorial_mask_deg is not None:
        u_coeffs_masked, v_coeffs_masked = apply_equatorial_mask(
            u_coeffs_masked, v_coeffs_masked,
            equator_mask_deg=equatorial_mask_deg
        )

    # --- START: WAF Scaling Logic (atmospheric dynamics methodology) ---
    if wind_level == 'WAF':
        # Compute vector magnitude on the masked data
        magnitude = np.sqrt(u_coeffs_masked**2 + v_coeffs_masked**2)

        # Find percentiles for scaling (avoid NaN values)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore warnings from nanquantile on all-NaN slices
            # Use nanquantile to find percentiles, handling NaN values appropriately
            m05 = np.nanquantile(magnitude.values, 0.05) if not np.all(np.isnan(magnitude)) else 0
            m95 = np.nanquantile(magnitude.values, 0.95) if not np.all(np.isnan(magnitude)) else 1

        # Define scaling parameters (atmospheric dynamics standard)
        min_len = 0.6
        max_len = 1.4
        if m95 > m05:  # Avoid division by zero
            slope = (max_len - min_len) / (m95 - m05)
            intercept = max_len - slope * m95

            # Clip magnitude using xr.where for DataArray compatibility
            magnitude_clipped = xr.where(
                magnitude < m05, m05,
                xr.where(magnitude > m95, m95, magnitude)
            )

            # Calculate new magnitude
            new_magn = magnitude_clipped * slope + intercept

            # Calculate ratio for scaling, handle potential division by zero if magnitude is 0
            ratio = xr.where(magnitude != 0, new_magn / magnitude, 0)
            ratio = ratio.fillna(0.) # Fill NaNs in ratio (e.g., where magnitude was NaN) with 0

            # Apply scaling
            u_coeffs_scaled = u_coeffs_masked * ratio
            v_coeffs_scaled = v_coeffs_masked * ratio
        else: # If m95 <= m05 (e.g., all vectors are zero or NaN), don't scale
            u_coeffs_scaled = u_coeffs_masked
            v_coeffs_scaled = v_coeffs_masked

        # Use the scaled vectors for subsampling
        u_to_subsample = u_coeffs_scaled
        v_to_subsample = v_coeffs_scaled
        subsample_step = 7 # Use larger skip for WAF vectors
    else:
        # Use the masked (but not scaled) vectors for other wind levels
        u_to_subsample = u_coeffs_masked
        v_to_subsample = v_coeffs_masked
        subsample_step = 5 # Standard subsampling step
    # --- END: WAF Scaling Logic ---

    # Subsample for clearer quiver plotting
    u_wind_final = u_to_subsample[::subsample_step, ::subsample_step]
    v_wind_final = v_to_subsample[::subsample_step, ::subsample_step]

    return u_wind_final, v_wind_final

def get_upper_circulation(wind_data: Dict[str, xr.DataArray],
                          season: str,
                          predictor: str,
                          method: str, # Method identifier ('OLS', 'LASSO_fixed_alpha')
                          sign_level: float, # Used for OLS
                          upper_wind_level: str = '200',
                          apply_mask: bool = True
                          ) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray], Optional[xr.DataArray]]:
    """
    Retrieve upper-level wind coefficients for streamlines.
    Applies masking based on method: p-value for OLS, non-zero for LASSO variants.
    Calculates linewidth based on masked wind speed.
    """
    u_field_key = f"U_{upper_wind_level}"
    v_field_key = f"V_{upper_wind_level}"

    # Check if coefficient data exists
    if u_field_key not in wind_data or v_field_key not in wind_data:
        logging.warning(f"Missing upper-level wind coefficient data for {u_field_key} or {v_field_key} ({method}, {season}, {predictor}).")
        return None, None, None

    # Select coefficients
    u_coeffs = wind_data[u_field_key].sel(season=season, predictor=predictor)
    v_coeffs = wind_data[v_field_key].sel(season=season, predictor=predictor)

    # Apply masking based on method
    mask = None
    if apply_mask:  # Add this condition
        if method.upper() == 'OLS':
            u_pval_key, v_pval_key = f"pvals_{u_field_key}", f"pvals_{v_field_key}"
            if u_pval_key in wind_data and v_pval_key in wind_data:
                u_pval = wind_data[u_pval_key].sel(season=season, predictor=predictor)
                v_pval = wind_data[v_pval_key].sel(season=season, predictor=predictor)
                mask = (u_pval < (1 - sign_level / 100.0)) | (v_pval < (1 - sign_level / 100.0))
            else:
                logging.warning(f"Missing OLS p-values for upper wind masking ({u_field_key}, {v_field_key}). No mask applied.")
        elif 'LASSO' in method.upper(): # Catch LASSO_fixed_alpha
            mask = (u_coeffs != 0) | (v_coeffs != 0)

    # Apply mask
    if mask is not None:
        u_upper_masked = u_coeffs.where(mask)
        v_upper_masked = v_coeffs.where(mask)
    else:
        u_upper_masked = u_coeffs
        v_upper_masked = v_coeffs

    # Calculate line width based on the *masked* wind speed
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        regr_wind_speed = np.sqrt(u_upper_masked**2 + v_upper_masked**2)
        max_speed = np.nanmax(regr_wind_speed) if not np.all(np.isnan(regr_wind_speed)) else 0

    if max_speed and max_speed > 0:
        lw = 3 * regr_wind_speed / max_speed
        # Replace NaN with 0 instead of leaving NaN (critical for PDF compatibility)
        lw = lw.where(~np.isnan(regr_wind_speed), 0)
    else:
        # Always use 0 for missing linewidths (no NaN values)
        lw = xr.full_like(u_upper_masked, 0)

    return u_upper_masked, v_upper_masked, lw


# ----
# Main processing function (process_variable remains the same internally)
# ----
def process_variable(var_name: str,
                     pathway: str, # Pathway context ('EP', 'CP', 'COMBINED')
                     lag_range_str: str, # Full lag range string
                     method: str, # Method identifier ('OLS', 'LASSO_fixed_alpha')
                     start_year: str,
                     end_year: str,
                     sign_level: float = DEFAULT_SIGN_LEVEL,
                     wind_level: str = DEFAULT_WIND_LEVEL,
                     upper_wind_level: str = '200',
                     path_save: str = SAVE_DIR,
                     show_all_winds: bool = False,
                     output_format: str = DEFAULT_OUTPUT_FORMAT,
                     raster_dpi: int = RASTER_DPI,
                     vector_dpi: int = VECTOR_DPI,
                     seasons_filter: Optional[List[str]] = None,
                     single_panel_mode: bool = False,
                     panel_suffix: str = '') -> None:
    """
    Process a single target variable for a specific pathway, lag range, and method.
    Loads results, loads winds, plots effects, plots R².
    """
    # Use pathway as the config_name for directory structure and titles
    config_name = pathway
    logging.info(f"Processing: Var={var_name}, Pathway={pathway}, Set={lag_range_str} {method}")
    try:
        # Load regression results
        regr_coefs, pvals, rsq, _ = load_regression_results(pathway, var_name, lag_range_str, method)

        if regr_coefs is None or rsq is None:
             logging.error(f"Failed to load primary regression results for {var_name}, {pathway}, {lag_range_str}, {method}. Skipping.")
             return

    except Exception as e:
        logging.error(f"Critical error loading regression results for {var_name}: {e}")
        return

    # Load associated wind data
    wind_data = load_all_wind_data(pathway, lag_range_str, method, wind_level, upper_wind_level)

    # Determine the start season based on the minimum lag in the range
    clean_lag_range = lag_range_str.replace('_COMBINED', '')
    if '-' in clean_lag_range:
        _, min_lag_str = clean_lag_range.replace('lag','').split('-')
        min_lag = int(min_lag_str)
    else:
        min_lag = int(clean_lag_range.replace('lag',''))

    LAG_TO_SEASON = { 6: "JJA_m1", 5: "SON_m1", 4: "DJF_0", 3: "MAM_0", 2: "JJA_0", 1: "SON_0", 0: "DJF_1" }
    start_season = LAG_TO_SEASON.get(min_lag, SEASON_CODES[0])

    # --- Determine Predictors to Plot ---
    if hasattr(regr_coefs, 'predictor') and regr_coefs.predictor.size > 0:
        predictors_to_plot = regr_coefs.predictor.values.tolist()
    else:
        logging.error(f"Could not find 'predictor' coordinate in data for {var_name}. Cannot determine predictors to plot.")
        predictors_to_plot = []

    # Plot effect for each predictor found in the data
    for predictor in predictors_to_plot:
        plot_predictor_effect(
            regr_coefs=regr_coefs,
            pvals=pvals,
            var_name=var_name,
            predictor_name=predictor,
            wind_data=wind_data,
            method=method,
            lag_range_str=lag_range_str, # Pass full lag string
            sign_level=sign_level,
            start_season=start_season,
            path_save=path_save,
            wind_level=wind_level,
            upper_wind_level=upper_wind_level,
            config_name=config_name, # Use pathway as config name
            years=(start_year, end_year),
            show_all_winds=show_all_winds,
            output_format=output_format,
            raster_dpi=raster_dpi,
            vector_dpi=vector_dpi,
            seasons_filter=seasons_filter,
            single_panel_mode=single_panel_mode,
            panel_suffix=panel_suffix
        )

    # Plot R-squared
    if predictors_to_plot:
         plot_r_squared(
             rsq=rsq,
             var_name=var_name,
             predictors=predictors_to_plot,
             method=method,
             lag_range_str=lag_range_str, # Pass full lag string
             start_season=start_season,
             path_save=path_save,
             config_name=config_name, # Use pathway as config name
             years=(start_year, end_year),
             output_format=output_format,
             raster_dpi=raster_dpi,
             vector_dpi=vector_dpi,
             seasons_filter=seasons_filter,
             single_panel_mode=single_panel_mode,
             panel_suffix=panel_suffix
         )
    else:
         logging.warning(f"Skipping R² plot for {var_name} as predictor list could not be determined.")

    del regr_coefs, pvals, rsq, wind_data, predictors_to_plot
    gc.collect()


# ----
# Main Execution Logic
# ----
def main(args=None):
    """
    Main function: iterates over regression sets, determines the
    pathway context(s) for each set, and processes target variables accordingly.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Plot multivariate regression results with optional wind plotting controls."
    )
    parser.add_argument('--show_all_winds', action='store_true', default=False,
                        help="Show all wind vectors, with non-significant winds in lighter colors (default: show only significant)")
    parser.add_argument('--output_format', type=str, default=DEFAULT_OUTPUT_FORMAT,
                        choices=SUPPORTED_FORMATS,
                        help=f"Output format for figures. Options: {', '.join(SUPPORTED_FORMATS)}. 'both' saves PNG and PDF. (default: {DEFAULT_OUTPUT_FORMAT})")
    parser.add_argument('--raster_dpi', type=int, default=RASTER_DPI,
                        help=f"DPI for rasterized elements in vector formats (default: {RASTER_DPI})")
    parser.add_argument('--vector_dpi', type=int, default=VECTOR_DPI,
                        help=f"DPI for pure raster formats (default: {VECTOR_DPI})")
    parser.add_argument('--seasons', nargs='+', 
                        choices=SEASON_CODES,
                        default=None,
                        help='Generate only specific seasons (e.g., --seasons DJF_0 MAM_0). '
                             'Default: all seasons from start season onwards')
    parser.add_argument('--single_panel', action='store_true', default=False,
                        help='Save each season as a separate file instead of multi-panel figure')
    parser.add_argument('--panel_suffix', type=str, default='',
                        help='Add custom suffix to single panel filenames (e.g., "_zoom")')
    parser.add_argument('--variables', nargs='+',
                        default=VALID_VARIABLES,
                        choices=VALID_VARIABLES,
                        help=f'Target variables to process (default: all). Choices: {", ".join(VALID_VARIABLES)}')
    parser.add_argument('--lag_sets', nargs='+',
                        default=VALID_LAG_SETS,
                        choices=VALID_LAG_SETS,
                        help=f'Lag configurations to process (default: all). Choices: {", ".join(VALID_LAG_SETS)}')
    parser.add_argument('--model_types', nargs='+',
                        default=['OLS'],
                        choices=VALID_MODEL_TYPES,
                        help=f'Model types to include (default: OLS). Choices: {", ".join(VALID_MODEL_TYPES)}')
    parser.add_argument('--include_combined', action='store_true', default=True,
                        help='Include COMBINED pathway sets (default: True)')
    parser.add_argument('--no_combined', action='store_true', default=False,
                        help='Exclude COMBINED pathway sets')

    parsed_args = parser.parse_args(args)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting multi-regression plots script (OLS/Fixed-LASSO version)")
    logging.info(f"Output format: {parsed_args.output_format}, Raster DPI: {parsed_args.raster_dpi}, Vector DPI: {parsed_args.vector_dpi}")

    # List of target variables to process (from CLI)
    variables = parsed_args.variables
    logging.info(f"Variables to process: {variables}")

    # Build REGRESSION_SETS from CLI arguments
    include_combined = parsed_args.include_combined and not parsed_args.no_combined
    regression_sets = build_regression_sets(
        lag_sets=parsed_args.lag_sets,
        model_types=parsed_args.model_types,
        include_combined=include_combined
    )
    logging.info(f"Regression sets to process: {len(regression_sets)} configurations")
    logging.info(f"  Lag sets: {parsed_args.lag_sets}, Model types: {parsed_args.model_types}, Include combined: {include_combined}")

    # Define wind levels
    default_wind_levels = ['10m', '850']
    upper_wind_levels = ['200']

    # Iterate over defined regression sets first
    for lag_range_str, method, _ in regression_sets:
        logging.info(f"==== Processing Regression Set: {lag_range_str} {method} ====")

        # Determine the pathway context(s) for this regression set
        pathways_for_this_set = []
        if method.upper() == 'OLS':
            # If this is a combined‐only OLS run, use COMBINED; else EP & CP
            if '_COMBINED' in lag_range_str:
                pathways_for_this_set = ['COMBINED']
            else:
                pathways_for_this_set = ['EP', 'CP']
        elif method.upper() == 'LASSO_FIXED_ALPHA':
            # Fixed‐alpha LASSO can be done for both individual and combined pathways
            if '_COMBINED' in lag_range_str:
                pathways_for_this_set = ['COMBINED']
            else:
                pathways_for_this_set = ['EP', 'CP']
        else:
            logging.warning(f"Unknown method '{method}' in REGRESSION_SETS. Skipping set '{lag_range_str}'.")
            continue

        # Now, loop through the applicable pathway(s) for this set
        for pathway in pathways_for_this_set:
            logging.info(f"  -- Applying Pathway Context: {pathway} --")

            # Use tqdm for progress bar over variables for this set/pathway combination
            for var_name in tqdm(variables, desc=f"{pathway} {lag_range_str} {method}"):

                # Determine wind levels to plot
                if var_name in ['RWS_200', 'WAFx_200', 'WAFy_200']:
                    effective_wind_levels = ['WAF']
                else:
                    effective_wind_levels = default_wind_levels

                # Loop through wind level combinations
                for wind_level in effective_wind_levels:
                    for upper_wind_level in upper_wind_levels:
                        # Call process_variable
                        process_variable(
                            var_name=var_name,
                            pathway=pathway, # Pass the determined pathway context
                            lag_range_str=lag_range_str,
                            method=method,
                            start_year=DEFAULT_YEARS[0],
                            end_year=DEFAULT_YEARS[1],
                            sign_level=DEFAULT_SIGN_LEVEL,
                            wind_level=wind_level,
                            upper_wind_level=upper_wind_level,
                            path_save=SAVE_DIR,
                            show_all_winds=parsed_args.show_all_winds,
                            output_format=parsed_args.output_format,
                            raster_dpi=parsed_args.raster_dpi,
                            vector_dpi=parsed_args.vector_dpi,
                            seasons_filter=parsed_args.seasons,
                            single_panel_mode=parsed_args.single_panel,
                            panel_suffix=parsed_args.panel_suffix
                        )
                        gc.collect()

    logging.info("==== All processing completed. ====")

if __name__ == "__main__":
    main()