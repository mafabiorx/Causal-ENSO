"""
Specialized plotting script for multi-level RWS (Rossby Wave Source) regression results.
Creates pressure-longitude cross-sections averaged over multiple latitude bands
shifting 10 southward: (-5,5), (-15,-5), (-25,-15), (-35,-25) to examine
Rossby wave generation and propagation across different regions.

RWS spans pressure levels from 500 hPa to 200 hPa (200, 250, 300, 400, 500).
Uses the same colormap and levels as RWS_200 for consistency.

Outputs are organized in separate folders for each latitude band.
Uses transparency approach for non-significant areas with hatching.

Usage examples:
  # Default: process all lag sets with OLS, including combined
  python script.py

  # Process specific lag configurations
  python script.py --lag_sets lag6 lag6-4

  # Use both OLS and LASSO models
  python script.py --model_types OLS LASSO_fixed_alpha

  # Process without COMBINED pathways
  python script.py --no_combined

  # Combine with other options
  python script.py --lag_sets lag6-4 --output_format pdf --seasons DJF_0 MAM_0
"""

import os
import sys
from pathlib import Path
import warnings
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
import matplotlib.colors as clrs
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from matplotlib.ticker import LogFormatterMathtext
import logging
import gc
import argparse
from typing import List, Tuple, Optional
from tqdm import tqdm

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Custom formatter for pressure axis that never uses scientific notation
class PlainLogFormatter(LogFormatterMathtext):
    """Log formatter that always returns plain integers for pressure values."""
    def __call__(self, x, pos=None):
        # Always return plain integer format for our pressure range
        if 100 <= x <= 1000:
            return f'{int(x)}'
        return ''

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from utils.paths import get_data_path, get_results_path
    from utils.TBI_functions import (
        get_symmetric_levels_fixed_spacing
    )
    from utils.plotting_optimization import (
        save_figure_optimized, add_plotting_arguments, 
        clean_data_for_pdf, filter_seasons_to_plot,
        create_clean_panel_title, create_descriptive_filename
    )
except ImportError:
    print("Could not import from utils, trying utils...")
    from utils.paths import get_data_path, get_results_path
    from utils.TBI_functions import (
        get_symmetric_levels_fixed_spacing
    )
    from utils.plotting_optimization import (
        save_figure_optimized, add_plotting_arguments, 
        clean_data_for_pdf, filter_seasons_to_plot,
        create_clean_panel_title, create_descriptive_filename
    )

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = get_data_path('time_series/seas_multilag_regr_coeffs/')
SAVE_DIR = get_results_path('regressions/multilag_regr_plots_OLS_LASSO/')
os.makedirs(SAVE_DIR, exist_ok=True)

# Default parameters
DEFAULT_SIGN_LEVEL = 95    # Significance level for OLS masking
DEFAULT_YEARS = ('1945', '2024')

SEASON_CODES = ['JJA_m1', 'SON_m1', 'DJF_0', 'MAM_0', 'JJA_0', 'SON_0', 'DJF_1']

# Map to display names
SEASON_MAPPING = {
    'JJA_m1': 'JJA(-1)', 'SON_m1': 'SON(-1)', 'DJF_0': 'DJF(0)',
    'MAM_0': 'MAM(0)', 'JJA_0': 'JJA(0)', 'SON_0': 'SON(0)', 'DJF_1': 'DJF(1)'
}

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


# RWS specific settings (matching RWS_200 from 16_OLS_LASSO_regrs_plots.py)
RWS_SETTINGS = {
    'cmap': clrs.LinearSegmentedColormap.from_list("", ["darkblue", "blue", "white", "red", "darkred"]),
    'levels': np.arange(-10, 10.5, 0.5),  # From -10 to 10 with step 0.5
    'contour_levels': None,  # No contour lines for RWS
    'conv_factor': 1e-11,  # Convert to 10^-11 s^-2
    'label': 'Rossby Wave Source [10$^{-11}$ s$^{-2}$]',
    'alpha_nonsig': 0.4,  # Transparency for non-significant areas
    'alpha_sig': 1.0     # Opacity for significant areas
}

# Latitude bands for averaging (10° southward shifts)
LAT_BANDS = [
    (-5, 5),      # Equatorial band
    (-15, -5),    # Northern tropics
    (-25, -15),   # Northern subtropics  
    (-35, -25)    # Northern extratropics
]

# Define descriptive names for each band (for folder naming)
LAT_BAND_NAMES = {
    (-5, 5): "equatorial_5S-5N",
    (-15, -5): "northern_tropics_15S-5S",
    (-25, -15): "northern_subtropics_25S-15S",
    (-35, -25): "northern_extratropics_35S-25S"
}

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_regression_results(pathway: str, target_var: str, lag_range_str: str, method: str
    ) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray], Optional[xr.DataArray], Optional[xr.DataArray]]:
    """
    Load regression results for a given pathway, target variable, lag range, and method.

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
            elif method.upper() == 'LASSO_FIXED_ALPHA':
                if 'fixed_alpha' in ds:
                    fixed_alpha_val = ds['fixed_alpha'].load()
                else:
                    logging.warning(f"LASSO_fixed_alpha results file missing 'fixed_alpha': {filepath}")

        return regr_coefs, p_vals, r_squared, fixed_alpha_val
    except Exception as e:
        logging.error(f"Error loading regression results from {filepath}: {e}")
        return None, None, None, None

# =============================================================================
# LATITUDE BAND EXTRACTION
# =============================================================================

def extract_latitude_band(data_array, lat_min=-5, lat_max=5):
    """
    Extract and average data over specified latitude band with cosine weighting.
    
    Parameters:
    -----------
    data_array : xr.DataArray
        Input data with latitude dimension
    lat_min, lat_max : float
        Latitude bounds for averaging (default: 5°S to 5°N)
    
    Returns:
    --------
    xr.DataArray : Averaged over specified latitude band
    """
    # Select latitude band
    band_data = data_array.sel(latitude=slice(lat_min, lat_max))
    
    # Weight by cosine of latitude for proper spherical averaging
    weights = np.cos(np.deg2rad(band_data.latitude))
    weighted_data = band_data.weighted(weights)
    
    # Average over latitude dimension
    return weighted_data.mean(dim='latitude')

# =============================================================================
# SIGNIFICANCE HANDLING
# =============================================================================

def prepare_data_for_plotting(regr_coef, pvals, method, lat_band, p_threshold=0.05):
    """
    Prepare regression data with significance information preserved.
    
    Returns:
    - coef_eq: Equatorial average of all coefficients
    - coef_sig_eq: Equatorial average with NaN for non-significant areas
    - is_significant_eq: Boolean mask after averaging
    """
    # Determine significance based on method
    if method.upper() == 'OLS' and pvals is not None:
        is_significant = pvals < p_threshold
    else:  # LASSO: non-zero coefficients are significant
        is_significant = regr_coef != 0
    
    # Extract latitude band for both data and significance mask
    coef_eq = extract_latitude_band(regr_coef, *lat_band)
    
    # Convert boolean to float for averaging
    sig_mask_float = is_significant.astype(float)
    sig_mask_eq = extract_latitude_band(sig_mask_float, *lat_band)
    
    # Create significant-only version (NaN where not significant)
    coef_sig = regr_coef.where(is_significant)
    coef_sig_eq = extract_latitude_band(coef_sig, *lat_band)
    
    # Determine if averaged values should be considered significant
    # (e.g., if >50% of contributing points were significant)
    is_significant_eq = sig_mask_eq > 0.5
    
    return coef_eq, coef_sig_eq, is_significant_eq

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_pressure_longitude_section_rws(ax, lon, pressure, coef_all, coef_sig,
                                        settings, predictor_name, season, output_format='png'):
    """
    Create pressure-longitude plot for RWS with hatching for significant areas.
    Modified for RWS: 500-200 hPa range, no contour lines.
    """
    # Convert to display units
    coef_all_scaled = coef_all / settings['conv_factor']
    coef_sig_scaled = coef_sig / settings['conv_factor']  # keep NaNs for significance mask
    
    # Clean only the filled field for PDF compatibility; preserve NaNs in significance
    coef_all_for_plot = clean_data_for_pdf(coef_all_scaled)[0]
    
    # Use fixed levels for RWS (from settings)
    levels = settings['levels']
    
    # Check for sufficient valid data points
    if np.all(np.isnan(coef_all_scaled)):
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                ha='center', va='center', fontsize=12, color='gray')
        return None
    
    # Main plot: All coefficients with full opacity for smooth contours
    # Apply rasterization for vector formats
    contourf_kwargs = {
        'levels': levels,
        'cmap': settings['cmap'],
        'extend': 'both'
    }
    
    # Apply rasterization for PDF/SVG formats
    if output_format in ['pdf', 'svg', 'both']:
        contourf_kwargs['rasterized'] = True
    
    cf = ax.contourf(
        lon, pressure, coef_all_for_plot,
        **contourf_kwargs
    )
    
    # Apply hatching pattern to statistically significant regions
    # Create a mask where data IS significant
    sig_mask = ~np.isnan(coef_sig_scaled)
    
    if np.any(sig_mask):
        # Use contourf with a single level to create hatching regions
        # Convert boolean mask to float (1 for significant, 0 for non-significant)
        mask_data = sig_mask.astype(float)
        
        # Add hatching using contourf with hatches parameter
        ax.contourf(
            lon, pressure, mask_data,
            levels=[0.5, 1.5],  # This will capture all values of 1 (significant)
            colors='none',  # No fill color
            hatches=['///'],  # Diagonal lines, less dense than dots
        )
    
    # No contour lines for RWS (contour_levels is None)
    
    # Configure axes for RWS pressure range
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_ylim(500, 200)  # RWS pressure range
    ax.set_xlim(-180, 180)
    
    # Set explicit major ticks for RWS levels
    ax.set_yticks([500, 400, 300, 250, 200])
    
    # Set minor ticks to control all possible tick values
    ax.set_yticks([500, 450, 400, 350, 300, 250, 200], minor=True)
    
    # Apply our custom formatter to ALL ticks (major and minor)
    ax.yaxis.set_major_formatter(PlainLogFormatter())
    ax.yaxis.set_minor_formatter(PlainLogFormatter())
    
    # Control the log locator to ensure consistent tick placement
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numticks=12))
    
    # Labels
    ax.set_xlabel('Longitude', fontsize=16)
    ax.set_ylabel('Pressure (hPa)', fontsize=16)
    ax.set_title(f'Season {season}', fontsize=18, pad=10)
    ax.tick_params(labelsize=14)
    
    # Add zero line for reference
    ax.axhline(y=300, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    return cf  # Return for colorbar

def add_walker_circulation_annotations(ax):
    """Add reference annotations for Walker circulation centers."""
    # Pacific center western hemisphere
    ax.axvline(x=-160, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(-160, 350, 'Pacific', ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Pacific center eastern hemisphere
    ax.axvline(x=160, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(160, 350, 'Pacific', ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Atlantic center
    ax.axvline(x=-30, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(-30, 350, 'Atlantic', ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Indian Ocean
    ax.axvline(x=90, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(90, 350, 'Indian', ha='center', va='bottom', fontsize=16, fontweight='bold')

# =============================================================================
# MAIN PLOTTING FUNCTIONS
# =============================================================================

def plot_predictor_effect(
    regr_coefs: xr.DataArray,
    pvals: Optional[xr.DataArray],  # Only used for OLS
    predictor_name: str,
    method: str,  # Method identifier ('OLS', 'LASSO_fixed_alpha')
    lag_range_str: str,  # Full lag range string
    lat_band: Tuple[float, float] = (-5, 5),  # Latitude band for averaging
    sign_level: float = DEFAULT_SIGN_LEVEL,  # Used only for OLS
    start_season: str = SEASON_CODES[0],
    path_save: str = SAVE_DIR,
    config_name: str = "",  # Corresponds to pathway ('EP', 'CP', 'COMBINED')
    years: Tuple[str, str] = DEFAULT_YEARS,
    output_format: str = 'png',
    single_panel: bool = False,
    panel_suffix: str = '',
    seasons_filter: Optional[List[str]] = None,
    raster_dpi: int = 150,
    vector_dpi: int = 300
) -> None:
    """
    Plot regression coefficients for a single predictor across seasons.
    Applies significance masking for OLS (using pvals and sign_level).
    Applies non-zero masking for LASSO_fixed_alpha (coefficients != 0).
    Creates pressure-longitude cross-sections averaged over specified latitude band.
    """
    start_season_code = get_season_code(start_season)
    try:
        start_idx = SEASON_CODES.index(start_season_code)
    except ValueError:
        logging.warning(f"Start season code {start_season_code} not found in {SEASON_CODES}. Defaulting to first season.")
        start_idx = 0
    
    # Apply season filtering
    seasons_to_plot = filter_seasons_to_plot(seasons_filter, SEASON_CODES, start_idx)
    
    # Setup for single panel or multi-panel mode
    if single_panel:
        # Process each season separately
        for season_code in seasons_to_plot:
            season_disp = SEASON_MAPPING.get(season_code, season_code)
            
            # Create single panel figure
            fig, ax = plt.subplots(1, 1, figsize=(21, 6), subplot_kw={'aspect': 'auto'})
            
            try:
                # Select coefficient data for this season and predictor
                regr_var = regr_coefs.sel(season=season_code, predictor=predictor_name)
                
                # Apply masking based on method
                if method.upper() == 'OLS':
                    if pvals is not None:
                        p_value = pvals.sel(season=season_code, predictor=predictor_name)
                    else:
                        p_value = None
                else:
                    p_value = None
                
                # Prepare data with transparency approach
                coef_eq, coef_sig_eq, is_significant_eq = prepare_data_for_plotting(
                    regr_var, p_value, method, lat_band, p_threshold=(1 - sign_level / 100.0)
                )
                
                # Check if all data is masked out
                if coef_eq.isnull().all():
                    logging.warning(f"All data masked out for {predictor_name}, {season_code}. Skipping plot for this season.")
                    ax.text(0.5, 0.5, "All data masked", transform=ax.transAxes,
                            ha='center', va='center', fontsize=12, color='gray')
                    ax.set_title(create_clean_panel_title(season_disp), fontsize=16)
                else:
                    # Create plot with RWS-specific function
                    cf = plot_pressure_longitude_section_rws(
                        ax, coef_eq.longitude, coef_eq.pressure_level,
                        coef_eq.values, coef_sig_eq.values,
                        RWS_SETTINGS, predictor_name, season_disp, output_format
                    )
                    
                    # Simple title for single panel
                    ax.set_title(create_clean_panel_title(season_disp), fontsize=18)
                    
                    # Add Walker circulation annotations
                    add_walker_circulation_annotations(ax)
                    
                    # Add colorbar for single panel
                    if cf is not None:
                        cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.1)
                        cbar.set_label(RWS_SETTINGS['label'], fontsize=16)
                        cbar.ax.tick_params(labelsize=14)
                
            except Exception as plot_err:
                logging.error(f"Error plotting season {season_code} for {predictor_name}: {plot_err}", exc_info=True)
                ax.text(0.5, 0.5, "Plotting Error", transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, color='red')
                ax.set_title(create_clean_panel_title(season_disp), fontsize=16)
            
            # Save single panel figure
            plt.tight_layout()
            
            # Define output directory with latitude band subfolder
            dir_components = [path_save, config_name, method.lower(), lag_range_str, "rws_multi"]
            base_output_dir = os.path.join(*dir_components)
            
            # Create latitude band subdirectory inside rws_multi
            band_name = LAT_BAND_NAMES.get(lat_band, f"lat{lat_band[0]}-{lat_band[1]}")
            output_dir = os.path.join(base_output_dir, band_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Create descriptive filename for single panel
            if method.upper() == 'OLS':
                mask_desc = f"{sign_level}pct"
            elif 'LASSO' in method.upper():
                mask_desc = "non_zero"
            else:
                mask_desc = "unknown"
            
            filename = create_descriptive_filename(
                base_name='rws_multi',
                method=method.lower(),
                var_name='',
                predictor=predictor_name,
                pathway=config_name,
                lag=lag_range_str,
                season=season_code,
                mask_type=mask_desc,
                suffix=panel_suffix
            )
            
            filepath = os.path.join(output_dir, filename)
            
            # Save with optimization
            save_figure_optimized(fig, filepath, output_format, raster_dpi, vector_dpi)
            plt.close(fig)
            gc.collect()
        
        return  # Exit function after processing single panels
    
    # Original multi-panel code
    n_seasons = len(seasons_to_plot)
    fig_height = 6 * n_seasons
    fig, axs = plt.subplots(n_seasons, 1, figsize=(21, fig_height),
                            subplot_kw={'aspect': 'auto'})
    if n_seasons == 1:
        axs = [axs]  # Ensure axs is always iterable
    
    cf_mappable = None  # To store the last contourf for the colorbar
    
    for i, season_code in enumerate(seasons_to_plot):
        season_disp = SEASON_MAPPING.get(season_code, season_code)
        ax = axs[i]
        
        try:
            # Select coefficient data for this season and predictor
            regr_var = regr_coefs.sel(season=season_code, predictor=predictor_name)
            
            # Apply masking based on method
            if method.upper() == 'OLS':
                if pvals is not None:
                    p_value = pvals.sel(season=season_code, predictor=predictor_name)
                else:
                    p_value = None
            else:
                p_value = None
            
            # Prepare data with transparency approach
            coef_eq, coef_sig_eq, is_significant_eq = prepare_data_for_plotting(
                regr_var, p_value, method, lat_band, p_threshold=(1 - sign_level / 100.0)
            )
            
            # Check if all data is masked out
            if coef_eq.isnull().all():
                logging.warning(f"All data masked out for {predictor_name}, {season_code}. Skipping plot for this season.")
                ax.text(0.5, 0.5, "All data masked", transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, color='gray')
                ax.set_title(f"{season_disp}: {predictor_name} effect", fontsize=16)
                continue
            
            # Create plot with RWS-specific function
            cf = plot_pressure_longitude_section_rws(
                ax, coef_eq.longitude, coef_eq.pressure_level,
                coef_eq.values, coef_sig_eq.values,
                RWS_SETTINGS, predictor_name, season_disp, output_format
            )
            
            if cf is not None:
                cf_mappable = cf
            
            # Add Walker circulation annotations
            add_walker_circulation_annotations(ax)
            
        except Exception as plot_err:
            # Log other potential errors during plotting for a season
            logging.error(f"Error plotting season {season_code} for {predictor_name}: {plot_err}", exc_info=True)
            ax.text(0.5, 0.5, "Plotting Error", transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='red')
            ax.set_title(f"{season_disp}: {predictor_name} effect - ERROR", fontsize=16)
    
    # Add a single colorbar at the bottom after the loop
    if cf_mappable is not None:
        cax = fig.add_axes([0.3, 0.02, 0.4, 0.025])  # Position: [left, bottom, width, height]
        cbar = plt.colorbar(cf_mappable, cax=cax, orientation='horizontal')
        cbar.set_label(RWS_SETTINGS['label'], fontsize=16)
        cbar.ax.tick_params(labelsize=14)
    else:
        logging.warning(f"No valid data plotted across seasons for {predictor_name}. Skipping colorbar.")
    
    # Overall figure title
    if method.upper() == 'OLS':
        mask_desc = f"{sign_level}% conf"
    elif 'LASSO' in method.upper():
        mask_desc = "non-zero coefs"
    else:
        mask_desc = "unknown"
    
    pathway_str = f" [{config_name}]" if config_name else ""
    lat_str = f"{abs(lat_band[0])}°{'S' if lat_band[0] < 0 else 'N'}-{abs(lat_band[1])}°{'S' if lat_band[1] < 0 else 'N'}"
    # plt.suptitle(f"RWS Response to {predictor_name}{pathway_str}\n" +
    #              f"({lat_str} average, {method}, {lag_range_str}, {mask_desc}) [{years[0]}-{years[1]}]",
    #              fontsize=18, y=0.99)
    plt.subplots_adjust(left=0.04, bottom=0.11, right=0.96, top=0.95, hspace=0.25)
    
    # Define output directory with latitude band subfolder
    dir_components = [path_save, config_name, method.lower(), lag_range_str, "rws_multi"]
    base_output_dir = os.path.join(*dir_components)
    
    # Create latitude band subdirectory inside rws_multi
    band_name = LAT_BAND_NAMES.get(lat_band, f"lat{lat_band[0]}-{lat_band[1]}")
    output_dir = os.path.join(base_output_dir, band_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Filename doesn't need band name since it's in the folder structure now
    filename = f"rws_multi_{method.lower()}_{predictor_name.replace('.', '_')}_{config_name}_{lag_range_str}_{mask_desc.replace('% conf','pct').replace(' ','_')}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save the figure with optimization
    try:
        # Remove extension from filepath for save_figure_optimized
        base_filepath = os.path.splitext(filepath)[0]
        save_figure_optimized(fig, base_filepath, output_format, raster_dpi, vector_dpi)
    except Exception as save_err:
        logging.error(f"Failed to save plot {filepath}: {save_err}")
    finally:
        plt.close(fig)
        gc.collect()

def plot_r_squared_rws(
    rsq: xr.DataArray,
    predictors: List[str],  # List of predictors included in this model run
    method: str,
    lag_range_str: str,  # Full lag range string
    lat_band: Tuple[float, float] = (-5, 5),  # Latitude band for averaging
    start_season: str = SEASON_CODES[0],
    path_save: str = SAVE_DIR,
    config_name: str = "",  # Corresponds to pathway ('EP', 'CP', 'COMBINED')
    years: Tuple[str, str] = DEFAULT_YEARS,
    output_format: str = 'png',
    single_panel: bool = False,
    panel_suffix: str = '',
    seasons_filter: Optional[List[str]] = None,
    raster_dpi: int = 150,
    vector_dpi: int = 300
) -> None:
    """
    Plot the spatial distribution of R² across seasons for RWS.
    """
    logging.info(f"Plotting R² for RWS ({method}, {lag_range_str}, Pathway: {config_name})")
    
    # Pre-check for all NaNs or Infs
    if rsq is None or np.all(np.isnan(rsq)):
        logging.error(f"R² data is None or all NaN for RWS ({method}, {lag_range_str}, {config_name}). Skipping R² plot.")
        return
    rsq = rsq.where(~np.isinf(rsq), np.nan)  # Replace infinities with NaN
    
    start_season_code = get_season_code(start_season)
    try:
        start_idx = SEASON_CODES.index(start_season_code)
    except ValueError:
        logging.warning(f"Start season code {start_season_code} not found. Defaulting to first season.")
        start_idx = 0
    
    # Apply season filtering
    seasons_to_plot = filter_seasons_to_plot(seasons_filter, SEASON_CODES, start_idx)
    
    # Handle single panel mode for R-squared plots
    if single_panel:
        # Process each season separately
        for season_code in seasons_to_plot:
            season_disp = SEASON_MAPPING.get(season_code, season_code)
            
            # Create single panel figure
            fig, ax = plt.subplots(1, 1, figsize=(21, 6), subplot_kw={'aspect': 'auto'})
            
            try:
                # Extract R² for this season
                r2_season = rsq.sel(season=season_code)
                
                # Extract latitude band
                r2_eq = extract_latitude_band(r2_season, *lat_band)
                
                # Check if data for this season is all NaN
                if np.all(np.isnan(r2_eq)):
                    logging.warning(f"All R² values are NaN for RWS, season {season_code}")
                    ax.text(0.5, 0.5, "No valid R² data", transform=ax.transAxes,
                            ha='center', va='center', fontsize=14, color='gray')
                    ax.set_title(create_clean_panel_title(season_disp), fontsize=16)
                else:
                    # R² specific settings
                    r2_settings = {
                        'cmap': 'YlOrRd',  # Sequential colormap
                        'levels': np.linspace(0, 0.5, 11),  # 0 to 50% variance
                        'alpha_threshold': 0.05,  # Only show R² > 5%
                    }
                    
                    # Mask low R² values
                    r2_masked = r2_eq.where(r2_eq > r2_settings['alpha_threshold'])
                    
                    # Clean data for PDF compatibility
                    r2_values_clean = clean_data_for_pdf(r2_eq.values)[0]
                    
                    # Plot with optional rasterization
                    pcolormesh_kwargs = {
                        'cmap': r2_settings['cmap'],
                        'vmin': 0,
                        'vmax': 1,
                        'shading': 'auto'
                    }
                    
                    # Apply rasterization for PDF/SVG formats
                    if output_format in ['pdf', 'svg', 'both']:
                        pcolormesh_kwargs['rasterized'] = True
                    
                    cf = ax.pcolormesh(r2_eq.longitude, r2_eq.pressure_level, r2_values_clean,
                                      **pcolormesh_kwargs)
                    
                    # Contour lines
                    try:
                        cs = ax.contour(r2_eq.longitude, r2_eq.pressure_level, r2_eq.values,
                                        levels=[0.1, 0.2, 0.3, 0.4],
                                        colors='black',
                                        linewidths=0.8,
                                        alpha=0.7)
                        ax.clabel(cs, fmt='%.1f', fontsize='small')
                    except ValueError:
                        logging.warning(f"Could not draw/label R² contours for season {season_code}")
                    
                    # Configure axes for RWS pressure range
                    ax.set_yscale('log')
                    ax.invert_yaxis()
                    ax.set_ylim(500, 200)  # RWS pressure range
                    ax.set_xlim(-180, 180)
                    
                    # Set y-axis ticks for RWS
                    major_ticks = [500, 400, 300, 250, 200]
                    ax.set_yticks(major_ticks)
                    all_possible_ticks = [500, 450, 400, 350, 300, 250, 200]
                    ax.set_yticks(all_possible_ticks, minor=True)
                    
                    # Custom formatter
                    def format_pressure_tick(x, pos):
                        if x >= 100:
                            return f'{int(x)}'
                        return ''
                    
                    ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_pressure_tick))
                    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(format_pressure_tick))
                    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numticks=12))
                    
                    # Labels
                    ax.set_xlabel('Longitude', fontsize=16)
                    ax.set_ylabel('Pressure (hPa)', fontsize=16)

                    # Simple title for single panel
                    ax.set_title(create_clean_panel_title(season_disp), fontsize=18)
                    ax.tick_params(labelsize=14)

                    # Add Walker circulation references
                    add_walker_circulation_annotations(ax)
                    
                    # Grid
                    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
                    
                    # Add colorbar for single panel
                    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.1)
                    cbar.set_label('R² (explained variance)', fontsize=16)
                    cbar.ax.tick_params(labelsize=14)
                
            except Exception as plot_err:
                logging.error(f"Error plotting R² for season {season_code}: {plot_err}", exc_info=True)
                ax.text(0.5, 0.5, "Plotting Error", transform=ax.transAxes,
                        ha='center', va='center', fontsize=12, color='red')
                ax.set_title(create_clean_panel_title(season_disp), fontsize=16)
            
            # Save single panel figure
            plt.tight_layout()
            
            # Define output directory with latitude band subfolder
            dir_components = [path_save, config_name, method.lower(), lag_range_str, "r_squared"]
            base_output_dir = os.path.join(*dir_components)
            
            # Create latitude band subdirectory
            band_name = LAT_BAND_NAMES.get(lat_band, f"lat{lat_band[0]}-{lat_band[1]}")
            output_dir = os.path.join(base_output_dir, band_name)
            os.makedirs(output_dir, exist_ok=True)
            
            # Format predictor list for filename
            if len(predictors) > 3:
                predictor_title_part = f"{len(predictors)}_predictors"
            else:
                predictor_title_part = "_and_".join([p.replace('.', '_') for p in predictors])
            
            # Create descriptive filename for single panel
            filename = create_descriptive_filename(
                base_name='r_squared_rws_multi',
                method=method.lower(),
                var_name='',
                predictor='',
                pathway=config_name,
                lag=lag_range_str,
                season=season_code,
                suffix=f"by_{predictor_title_part}{panel_suffix}"
            )
            
            filepath = os.path.join(output_dir, filename)
            
            # Save with optimization
            save_figure_optimized(fig, filepath, output_format, raster_dpi, vector_dpi)
            plt.close(fig)
            gc.collect()
        
        return  # Exit function after processing single panels
    
    # Original multi-panel code
    n_seasons = len(seasons_to_plot)
    fig_height = 6 * n_seasons
    fig, axs = plt.subplots(n_seasons, 1, figsize=(21, fig_height),
                            subplot_kw={'aspect': 'auto'})
    if n_seasons == 1:
        axs = [axs]  # Ensure axs is always iterable
    
    # R² specific settings
    r2_settings = {
        'cmap': 'YlOrRd',  # Sequential colormap
        'levels': np.linspace(0, 0.5, 11),  # 0 to 50% variance
        'alpha_threshold': 0.05,  # Only show R² > 5%
    }
    
    valid_cf = None  # To store the last pcolormesh for the colorbar
    
    for i, season_code in enumerate(seasons_to_plot):
        season_disp = SEASON_MAPPING.get(season_code, season_code)
        ax = axs[i]
        try:
            # Extract R² for this season
            r2_season = rsq.sel(season=season_code)
            
            # Extract latitude band
            r2_eq = extract_latitude_band(r2_season, *lat_band)
            
            # Check if data for this season is all NaN
            if np.all(np.isnan(r2_eq)):
                logging.warning(f"All R² values are NaN for RWS, season {season_code}")
                ax.text(0.5, 0.5, "No valid R² data", transform=ax.transAxes,
                        ha='center', va='center', fontsize=14, color='gray')
                ax.set_title(f"{season_disp}: R²", fontsize=16)
                continue
            
            # Mask low R² values
            r2_masked = r2_eq.where(r2_eq > r2_settings['alpha_threshold'])
            
            # Clean data for PDF compatibility
            r2_values_clean = clean_data_for_pdf(r2_eq.values)[0]
            
            # Plot with optional rasterization
            pcolormesh_kwargs = {
                'cmap': r2_settings['cmap'],
                'vmin': 0,
                'vmax': 1,
                'shading': 'auto'
            }
            
            # Apply rasterization for PDF/SVG formats
            if output_format in ['pdf', 'svg', 'both']:
                pcolormesh_kwargs['rasterized'] = True
            
            cf = ax.pcolormesh(r2_eq.longitude, r2_eq.pressure_level, r2_values_clean,
                              **pcolormesh_kwargs)
            valid_cf = cf  # Store the mappable object
            
            # Contour lines
            try:
                cs = ax.contour(r2_eq.longitude, r2_eq.pressure_level, r2_eq.values,
                                levels=[0.1, 0.2, 0.3, 0.4],
                                colors='black',
                                linewidths=0.8,
                                alpha=0.7)
                ax.clabel(cs, fmt='%.1f', fontsize='small')
            except ValueError:
                # Log if contouring or labeling fails
                logging.warning(f"Could not draw/label R² contours for season {season_code}")
            
            # Configure axes for RWS pressure range
            ax.set_yscale('log')
            ax.invert_yaxis()
            ax.set_ylim(500, 200)  # RWS pressure range
            ax.set_xlim(-180, 180)
            
            # Set y-axis ticks for RWS
            # Major ticks
            major_ticks = [500, 400, 300, 250, 200]
            ax.set_yticks(major_ticks)
            
            # Minor ticks - include intermediate values that matplotlib might add
            all_possible_ticks = [500, 450, 400, 350, 300, 250, 200]
            ax.set_yticks(all_possible_ticks, minor=True)
            
            # Custom formatter that handles ANY tick value
            def format_pressure_tick(x, pos):
                """Format any pressure value as plain integer."""
                if x >= 100:  # Only format values in our pressure range
                    return f'{int(x)}'
                return ''  # Hide very small values
            
            # Apply formatter to both major and minor ticks
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(format_pressure_tick))
            ax.yaxis.set_minor_formatter(ticker.FuncFormatter(format_pressure_tick))
            
            # Control minor tick visibility
            ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), numticks=12))
            
            # Labels
            ax.set_xlabel('Longitude', fontsize=16)
            ax.set_ylabel('Pressure (hPa)', fontsize=16)
            ax.set_title(f'{season_disp}: R²', fontsize=18, pad=10)
            ax.tick_params(labelsize=14)
            
            # Add Walker circulation references
            add_walker_circulation_annotations(ax)
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            
        except Exception as plot_err:
            logging.error(f"Error plotting R² for season {season_code}: {plot_err}", exc_info=True)
            ax.text(0.5, 0.5, "Plotting Error", transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='red')
            ax.set_title(f"{season_disp}: R² - ERROR", fontsize=16)
    
    # Add colorbar at the bottom
    if valid_cf is not None:
        cax = fig.add_axes([0.3, 0.02, 0.4, 0.03])
        cbar = plt.colorbar(valid_cf, cax=cax, orientation='horizontal')
        cbar.set_label('R² (explained variance)', fontsize=16)
        cbar.ax.tick_params(labelsize=14)
    else:
        logging.warning(f"No valid R² plot created; skipping colorbar.")
    
    # Format predictor list for title
    if len(predictors) > 3:
        predictor_title_part = f"{len(predictors)} predictors"
    else:
        predictor_title_part = "_and_".join([p.replace('.', '_') for p in predictors])
    
    # Include pathway (config_name) and latitude band in title
    pathway_str = f" [{config_name}]" if config_name else ""
    lat_str = f"{abs(lat_band[0])}°{'S' if lat_band[0] < 0 else 'N'}-{abs(lat_band[1])}°{'S' if lat_band[1] < 0 else 'N'}"
    # plt.suptitle(f"Explained variance (R²) of RWS by {predictor_title_part}{pathway_str}\n" +
    #              f"({lat_str} average, {method}, {lag_range_str}) [{years[0]}-{years[1]}]",
    #              fontsize=18, y=0.99)
    plt.subplots_adjust(left=0.04, bottom=0.06, right=0.96, top=0.95, hspace=0.25)
    
    # Define output directory with latitude band subfolder
    dir_components = [path_save, config_name, method.lower(), lag_range_str, "r_squared"]
    base_output_dir = os.path.join(*dir_components)
    
    # Create latitude band subdirectory inside r_squared
    band_name = LAT_BAND_NAMES.get(lat_band, f"lat{lat_band[0]}-{lat_band[1]}")
    output_dir = os.path.join(base_output_dir, band_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Filename doesn't need band name since it's in the folder structure now
    filename = f"r_squared_rws_multi_{method.lower()}_{lag_range_str}_{config_name}_by_{predictor_title_part}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save the figure with optimization
    try:
        # Remove extension from filepath for save_figure_optimized
        base_filepath = os.path.splitext(filepath)[0]
        save_figure_optimized(fig, base_filepath, output_format, raster_dpi, vector_dpi)
    except Exception as e:
        logging.error(f"Failed to save R² plot {filepath}: {e}")
    finally:
        plt.close(fig)
        gc.collect()

# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def process_rws_multi_level(pathway: str,  # Pathway context ('EP', 'CP', 'COMBINED')
                           lag_range_str: str,  # Full lag range string
                           method: str,  # Method identifier ('OLS', 'LASSO_fixed_alpha')
                           lat_band: Tuple[float, float],  # Latitude band for averaging
                           start_year: str,
                           end_year: str,
                           sign_level: float = DEFAULT_SIGN_LEVEL,
                           path_save: str = SAVE_DIR,
                           output_format: str = 'png',
                           single_panel: bool = False,
                           panel_suffix: str = '',
                           seasons_filter: Optional[List[str]] = None,
                           raster_dpi: int = 150,
                           vector_dpi: int = 300) -> None:
    """
    Process multi-level RWS for a specific pathway, lag range, method, and latitude band.
    Loads results, plots effects, plots R².
    """
    # Target variable is RWS_multi
    target_var = 'RWS_multi'
    # Use pathway as the config_name for directory structure and titles
    config_name = pathway
    
    # The lat_band directory will be created inside the rws_multi subdirectory
    # during plotting, not at this level
    
    lat_str = f"{abs(lat_band[0])}°{'S' if lat_band[0] < 0 else 'N'}-{abs(lat_band[1])}°{'S' if lat_band[1] < 0 else 'N'}"
    logging.info(f"Processing: Var={target_var}, Pathway={pathway}, Set={lag_range_str} {method}, Lat band={lat_str}")
    
    try:
        # Load regression results
        regr_coefs, pvals, rsq, _ = load_regression_results(pathway, target_var, lag_range_str, method)
        
        if regr_coefs is None or rsq is None:
            logging.error(f"Failed to load primary regression results for {target_var}, {pathway}, {lag_range_str}, {method}. Skipping.")
            return
            
    except Exception as e:
        logging.error(f"Critical error loading regression results for {target_var}: {e}")
        return
    
    # Determine the start season based on the minimum lag in the range
    clean_lag_range = lag_range_str.replace('_COMBINED', '')
    if '-' in clean_lag_range:
        _, min_lag_str = clean_lag_range.replace('lag','').split('-')
        min_lag = int(min_lag_str)
    else:
        min_lag = int(clean_lag_range.replace('lag',''))
    
    LAG_TO_SEASON = {6: "JJA_m1", 5: "SON_m1", 4: "DJF_0", 3: "MAM_0", 2: "JJA_0", 1: "SON_0", 0: "DJF_1"}
    start_season = LAG_TO_SEASON.get(min_lag, SEASON_CODES[0])
    
    # Determine Predictors to Plot
    if hasattr(regr_coefs, 'predictor') and regr_coefs.predictor.size > 0:
        predictors_to_plot = regr_coefs.predictor.values.tolist()
    else:
        logging.error(f"Could not find 'predictor' coordinate in data for {target_var}. Cannot determine predictors to plot.")
        predictors_to_plot = []
    
    # Plot effect for each predictor found in the data
    for predictor in predictors_to_plot:
        plot_predictor_effect(
            regr_coefs=regr_coefs,
            pvals=pvals,
            predictor_name=predictor,
            method=method,
            lag_range_str=lag_range_str,
            lat_band=lat_band,
            sign_level=sign_level,
            start_season=start_season,
            path_save=path_save,
            config_name=config_name,
            years=(start_year, end_year),
            output_format=output_format,
            single_panel=single_panel,
            panel_suffix=panel_suffix,
            seasons_filter=seasons_filter,
            raster_dpi=raster_dpi,
            vector_dpi=vector_dpi
        )
    
    # Plot R-squared
    if predictors_to_plot:
        plot_r_squared_rws(
            rsq=rsq,
            predictors=predictors_to_plot,
            method=method,
            lag_range_str=lag_range_str,
            lat_band=lat_band,
            start_season=start_season,
            path_save=path_save,
            config_name=config_name,
            years=(start_year, end_year),
            output_format=output_format,
            single_panel=single_panel,
            panel_suffix=panel_suffix,
            seasons_filter=seasons_filter,
            raster_dpi=raster_dpi,
            vector_dpi=vector_dpi
        )
    else:
        logging.warning(f"Skipping R² plot for {target_var} as predictor list could not be determined.")
    
    del regr_coefs, pvals, rsq, predictors_to_plot
    gc.collect()

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """
    Main function: iterates over regression sets, determines the
    pathway context(s) for each set, and processes multi-level RWS accordingly.
    """
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Generate RWS multi-level regression plots')

    # Add standard plotting arguments
    add_plotting_arguments(parser)

    # Add regression set configuration arguments
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

    # Parse arguments
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting RWS multi-level regression plots script")
    logging.info(f"Output format: {args.output_format}")
    if args.single_panel:
        logging.info("Single panel mode enabled")
    if args.seasons:
        logging.info(f"Filtering seasons: {args.seasons}")
    logging.info(f"Raster DPI: {args.raster_dpi}, Vector DPI: {args.vector_dpi}")

    # Build REGRESSION_SETS from CLI arguments
    include_combined = args.include_combined and not args.no_combined
    regression_sets = build_regression_sets(
        lag_sets=args.lag_sets,
        model_types=args.model_types,
        include_combined=include_combined
    )
    logging.info(f"Regression sets to process: {len(regression_sets)} configurations")
    logging.info(f"  Lag sets: {args.lag_sets}, Model types: {args.model_types}, Include combined: {include_combined}")

    # Iterate over defined regression sets first
    for lag_range_str, method, _ in regression_sets:
        logging.info(f"==== Processing Regression Set: {lag_range_str} {method} ====")
        
        # Determine the pathway context(s) for this regression set
        pathways_for_this_set = []
        if method.upper() == 'OLS':
            # If this is a combined-only OLS run, use COMBINED; else EP & CP
            if '_COMBINED' in lag_range_str:
                pathways_for_this_set = ['COMBINED']
            else:
                pathways_for_this_set = ['EP', 'CP']
        elif method.upper() == 'LASSO_FIXED_ALPHA':
            # Fixed-alpha LASSO was done only for the combined pathway
            if '_COMBINED' in lag_range_str:
                pathways_for_this_set = ['COMBINED']
            else:
                logging.warning(f"Method is {method} but lag range '{lag_range_str}' is missing '_COMBINED'. Skipping this set.")
                continue
        else:
            logging.warning(f"Unknown method '{method}' in REGRESSION_SETS. Skipping set '{lag_range_str}'.")
            continue
        
        # Now, loop through the applicable pathway(s) for this set
        for pathway in pathways_for_this_set:
            logging.info(f"  -- Applying Pathway Context: {pathway} --")
            
            # Loop through latitude bands
            for lat_band in LAT_BANDS:
                lat_str = f"{abs(lat_band[0])}°{'S' if lat_band[0] < 0 else 'N'}-{abs(lat_band[1])}°{'S' if lat_band[1] < 0 else 'N'}"
                logging.info(f"    -- Processing Latitude Band: {lat_str} --")
                
                # Use tqdm for progress bar
                for _ in tqdm([1], desc=f"{pathway} {lag_range_str} {method} {lat_str}"):
                    # Call process_rws_multi_level with optimization parameters
                    process_rws_multi_level(
                        pathway=pathway,
                        lag_range_str=lag_range_str,
                        method=method,
                        lat_band=lat_band,
                        start_year=DEFAULT_YEARS[0],
                        end_year=DEFAULT_YEARS[1],
                        sign_level=DEFAULT_SIGN_LEVEL,
                        path_save=SAVE_DIR,
                        output_format=args.output_format,
                        single_panel=args.single_panel,
                        panel_suffix=args.panel_suffix,
                        seasons_filter=args.seasons,
                        raster_dpi=args.raster_dpi,
                        vector_dpi=args.vector_dpi
                    )
                    gc.collect()
    
    logging.info("==== All processing completed. ====")
    logging.info(f"Results saved to: {SAVE_DIR}")

if __name__ == "__main__":
    main()
