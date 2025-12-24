"""
Specialized plotting script for vertical streamfunction regression results.
Creates pressure-longitude cross-sections averaged over multiple latitude bands
shifting 10 southward: (-5,5), (-15,-5), (-25,-15), (-35,-25) to examine
Kelvin Wave propagation and Walker circulation modulation across different regions.

Outputs are organized in separate folders for each latitude band.
Uses transparency approach for non-significant areas.

Usage examples:
  # Default: process all lag sets with OLS, including combined
  python script.py

  # Process specific lag configurations
  python script.py --lag_sets lag6 lag6-3

  # Use both OLS and LASSO models
  python script.py --model_types OLS LASSO_fixed_alpha

  # Process without COMBINED pathways
  python script.py --no_combined

  # Combine with other options
  python script.py --lag_sets lag6-3 --output_format pdf --seasons DJF_0 MAM_0
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
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as ticker
from matplotlib.ticker import LogFormatterMathtext
import logging
import gc
from typing import List, Tuple, Optional
from tqdm import tqdm
import argparse

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
        get_symmetric_levels_fixed_spacing,
        load_era_field
    )
    from utils.plotting_optimization import (
        setup_cartopy_warnings, save_figure_optimized, 
        add_plotting_arguments, apply_rasterization_settings, 
        clean_data_for_pdf, filter_seasons_to_plot,
        setup_single_panel_figure, create_clean_panel_title,
        create_descriptive_filename, get_contourf_kwargs, get_pcolormesh_kwargs
    )
except ImportError:
    print("Could not import from utils, trying utils...")
    from utils.paths import get_data_path, get_results_path
    from utils.TBI_functions import (
        get_symmetric_levels_fixed_spacing,
        load_era_field
    )
    from utils.plotting_optimization import (
        setup_cartopy_warnings, save_figure_optimized, 
        add_plotting_arguments, apply_rasterization_settings, 
        clean_data_for_pdf, filter_seasons_to_plot,
        setup_single_panel_figure, create_clean_panel_title,
        create_descriptive_filename, get_contourf_kwargs, get_pcolormesh_kwargs
    )

# Setup Cartopy warnings suppression
setup_cartopy_warnings()

# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_DIR = get_data_path('time_series/seas_multilag_regr_coeffs/')
SAVE_DIR = get_results_path('regressions/multilag_regr_plots_OLS_LASSO/')
os.makedirs(SAVE_DIR, exist_ok=True)
SURFACE_ELEV_FILE = get_data_path("additional/Surface_elevation_as_geopot.nc", data_type="raw")

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


# Vertical streamfunction specific settings
VERTICAL_SF_SETTINGS = {
    'cmap': 'RdBu_r',  # Red for positive (enhanced eastward flow aloft)
    'level_spacing': 2.5e10,  # Spacing for contour levels
    'contour_spacing_factor': 4,  # Contour lines every 4th level
    'conv_factor': 1e11,  # Convert to 10¹¹ kg/s for display
    'label': 'Vertical Streamfunction Response [10¹¹ kg s⁻¹]',
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

# Physical constants for elevation conversion
G = 9.80665  # Standard gravity (m/s²)
P0 = 1013.25  # Sea level pressure (hPa)
L = 0.0065    # Temperature lapse rate (K/m)
T0 = 288.15   # Sea level temperature (K)
M = 0.0289644  # Molar mass of air (kg/mol)
R = 8.31447   # Universal gas constant J/(mol·K)

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
# SURFACE ELEVATION PROCESSING
# =============================================================================

def process_surface_elevation_for_plot(lat_band=(-5, 5)):
    """
    Process high-resolution surface elevation data for pressure-longitude plots.
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
        # Weight by cosine of latitude
        weights = np.cos(np.deg2rad(surface_pressure.latitude))
        surf_press_weighted = surface_pressure.weighted(weights)
        surf_press_lat_avg = surf_press_weighted.mean(dim='latitude')
        
        # Step 2: Regrid to 1° longitude resolution
        # Create target longitude grid (-180 to 180 convention)
        lon_1deg = np.arange(-180, 180, 1.0)
        
        # Interpolate to 1° grid
        surf_press_1deg = surf_press_lat_avg.interp(
            longitude=lon_1deg,
            method='linear'
        )
        
        return surf_press_1deg, surf_press_lat_avg
        
    except Exception as e:
        logging.warning(f"Could not load surface elevation data: {e}")
        return None, None

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_pressure_longitude_section(ax, lon, pressure, coef_all, coef_sig,
                                   settings, predictor_name, season, output_format='png'):
    """
    Create pressure-longitude plot with hatching for non-significant areas.
    """
    # Convert to display units
    coef_all_scaled = coef_all / settings['conv_factor']
    coef_sig_scaled = coef_sig / settings['conv_factor']
    
    # Use fixed spacing symmetric levels
    levels, contour_levels = get_symmetric_levels_fixed_spacing(
        coef_all_scaled,
        spacing=settings['level_spacing'] / settings['conv_factor']
    )
    
    # Check for sufficient valid data points
    if np.all(np.isnan(coef_all_scaled)):
        ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                ha='center', va='center', fontsize=14, color='gray')
        return None
    
    # Main plot: All coefficients with full opacity for smooth contours
    # Apply conditional rasterization for PDF output
    contourf_kwargs = {
        'levels': levels,
        'cmap': settings['cmap'],
        'extend': 'both'
    }
    apply_rasterization_settings(contourf_kwargs, output_format)
    cf = ax.contourf(lon, pressure, coef_all_scaled, **contourf_kwargs)
    
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
    
    # Contour lines on all data
    try:
        cs = ax.contour(
            lon, pressure, coef_all_scaled,
            levels=contour_levels,
            colors='black',
            linewidths=0.8,
            alpha=0.7
        )
        # Label contours
        ax.clabel(cs, contour_levels[::2], fmt='%1.1f',
                 fontsize='small', colors='black',
                 inline=True, inline_spacing=4)
    except ValueError:
        pass
    
    # Configure axes
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_ylim(1000, 200)
    ax.set_xlim(-180, 180)
    
    # Set explicit major ticks
    ax.set_yticks([1000, 850, 700, 500, 300, 200])
    
    # Set minor ticks to control all possible tick values
    ax.set_yticks([1000, 900, 850, 800, 700, 600, 500, 400, 300, 200], minor=True)
    
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
    ax.axhline(y=500, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
    
    return cf  # Return for colorbar

def add_orography_to_plot(ax, surf_press_1deg, show_fine_structure=False,
                         surf_press_highres=None):
    """
    Add surface topography representation to pressure-longitude plot.
    """
    if surf_press_1deg is None:
        return
    
    # Main topography fill (1° resolution)
    ax.fill_between(
        surf_press_1deg.longitude,
        surf_press_1deg.values,
        1050,  # Below plot bottom
        where=(surf_press_1deg.values < 1000),  # Only where surface is above 1000 hPa
        color='black',
        alpha=0.4,
        step='mid',
        label='Surface elevation'
    )
    
    # Optional: Show fine structure with thin line
    if show_fine_structure and surf_press_highres is not None:
        ax.plot(surf_press_highres.longitude,
                surf_press_highres.values,
                color='black',
                linewidth=0.5,
                alpha=0.7)
    
    # Add contour at specific elevations
    elev_contours = [850, 700]  # Pressure levels corresponding to elevations
    for p_level in elev_contours:
        if np.any(surf_press_1deg.values < p_level):
            ax.axhline(y=p_level, color='brown', linestyle=':',
                      alpha=0.3, linewidth=0.5)
    
    # Annotate major topographic features in equatorial band
    # Equatorial averaging may smooth topographic signals
    # Annotation based on known geographic locations and elevation patterns
    features = {
        'Andes': {'lon': -75, 'search_range': 10, 'text_y': 400},
        'E.Africa': {'lon': 35, 'search_range': 15, 'text_y': 500},
        'Maritime\nContinent': {'lon': 120, 'search_range': 20, 'text_y': 600},
    }
    
    for name, info in features.items():
        try:
            # Use boolean indexing
            lon_vals = surf_press_1deg.longitude.values
            lon_mask = (lon_vals >= info['lon']-info['search_range']) & (lon_vals <= info['lon']+info['search_range'])
            
            if np.any(lon_mask):
                # Extract data in the longitude window
                local_data = surf_press_1deg.values[lon_mask]
                local_lons = lon_vals[lon_mask]
                
                if len(local_data) > 0 and not np.all(np.isnan(local_data)):
                    # Find the minimum pressure (highest elevation) in the region
                    local_min = np.nanmin(local_data)
                    
                    # Always annotate if there's ANY topography signal
                    if local_min < 1000:  # Any elevation above sea level
                        # Find exact longitude of minimum
                        min_idx = np.nanargmin(local_data)
                        min_lon = local_lons[min_idx]
                        
                        # Place annotation with enhanced visibility
                        ax.annotate(name,
                                   xy=(min_lon, max(local_min, 950)),  # Don't let arrow go below 950
                                   xytext=(info['lon'], info['text_y']),  # Fixed text position
                                   fontsize=16,
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
                                   zorder=1000)  # Very high z-order
                        
                    else:
                        logging.info(f"No significant topography found for {name} in equatorial average")
        except Exception as e:
            logging.warning(f"Could not annotate {name}: {e}")

def add_walker_circulation_annotations(ax):
    """Add reference annotations for Walker circulation centers."""
    # Pacific center western hemisphere
    ax.axvline(x=-160, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(-160, 850, 'Pacific', ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Pacific center eastern hemisphere
    ax.axvline(x=160, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(160, 850, 'Pacific', ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Atlantic center
    ax.axvline(x=-30, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(-30, 850, 'Atlantic', ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Indian Ocean
    ax.axvline(x=90, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(90, 850, 'Indian', ha='center', va='bottom', fontsize=16, fontweight='bold')

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
    include_topography: bool = True,
    show_fine_topography: bool = False,
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
    
    # Handle single panel mode
    if single_panel:
        # Process each season separately
        for season_code in seasons_to_plot:
            season_disp = SEASON_MAPPING.get(season_code, season_code)
            
            # Create single panel figure
            fig, ax = setup_single_panel_figure(figsize=(21, 6))
            
            # Process surface elevation data
            if include_topography:
                surf_press_1deg, surf_press_highres = process_surface_elevation_for_plot(lat_band)
            else:
                surf_press_1deg, surf_press_highres = None, None
            
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
                    plt.close(fig)
                    continue
                
                # Create plot with transparency layers
                cf = plot_pressure_longitude_section(
                    ax, coef_eq.longitude, coef_eq.pressure_level,
                    coef_eq.values, coef_sig_eq.values,
                    VERTICAL_SF_SETTINGS, predictor_name, season_disp, output_format
                )
                
                # Add Walker circulation annotations
                add_walker_circulation_annotations(ax)
                
                # Add surface representation
                if include_topography and surf_press_1deg is not None:
                    add_orography_to_plot(ax, surf_press_1deg,
                                        show_fine_structure=show_fine_topography,
                                        surf_press_highres=surf_press_highres)
                
                # Simple title for single panel
                ax.set_title(create_clean_panel_title(season_disp), fontsize=18)

                # Add colorbar
                if cf is not None:
                    cax = fig.add_axes([0.3, 0.02, 0.4, 0.02])
                    cbar = plt.colorbar(cf, cax=cax, orientation='horizontal')
                    cbar.set_label(VERTICAL_SF_SETTINGS['label'], fontsize=16)
                    cbar.ax.tick_params(labelsize=14)
                
                # Create descriptive filename
                if method.upper() == 'OLS':
                    mask_desc = f"{sign_level}pct"
                elif 'LASSO' in method.upper():
                    mask_desc = "non_zero_coefs"
                else:
                    mask_desc = "unknown"
                
                lat_str = f"{abs(lat_band[0])}{'S' if lat_band[0] < 0 else 'N'}_{abs(lat_band[1])}{'S' if lat_band[1] < 0 else 'N'}"
                
                filename = create_descriptive_filename(
                    base_name='vertical_sf',
                    method=method,
                    var_name='streamfunction',
                    predictor=predictor_name,
                    pathway=config_name,
                    lag=lag_range_str,
                    season=season_code,
                    mask_type=mask_desc,
                    suffix=f"{lat_str}{panel_suffix}"
                )
                
                # Define output directory
                dir_components = [path_save, config_name, method.lower(), lag_range_str, "vert_zon_sf"]
                base_output_dir = os.path.join(*dir_components)
                band_name = LAT_BAND_NAMES.get(lat_band, f"lat{lat_band[0]}-{lat_band[1]}")
                output_dir = os.path.join(base_output_dir, band_name)
                os.makedirs(output_dir, exist_ok=True)
                
                # Save the figure
                filepath = os.path.join(output_dir, filename)
                save_figure_optimized(fig, filepath, output_format, raster_dpi, vector_dpi)
                
            except Exception as plot_err:
                logging.error(f"Error plotting single panel for season {season_code}: {plot_err}", exc_info=True)
            finally:
                plt.close(fig)
                gc.collect()
        
        return  # Exit after processing single panels
    
    # Multi-panel mode
    n_seasons = len(seasons_to_plot)
    fig_height = 6 * n_seasons
    fig, axs = plt.subplots(n_seasons, 1, figsize=(21, fig_height),
                            subplot_kw={'aspect': 'auto'})
    if n_seasons == 1:
        axs = [axs]  # Ensure axs is always iterable
    
    # Process surface elevation data once (before loop)
    if include_topography:
        surf_press_1deg, surf_press_highres = process_surface_elevation_for_plot(lat_band)
    else:
        surf_press_1deg, surf_press_highres = None, None
    
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
                        ha='center', va='center', fontsize=16, color='gray')
                ax.set_title(f"{season_disp}: {predictor_name} effect", fontsize=18)
                continue
            
            # Create plot with transparency layers
            cf = plot_pressure_longitude_section(
                ax, coef_eq.longitude, coef_eq.pressure_level,
                coef_eq.values, coef_sig_eq.values,
                VERTICAL_SF_SETTINGS, predictor_name, season_disp, output_format
            )
            
            if cf is not None:
                cf_mappable = cf
            
            # Add Walker circulation annotations first (behind topography)
            add_walker_circulation_annotations(ax)
            
            # Add surface representation
            if include_topography and surf_press_1deg is not None:
                add_orography_to_plot(ax, surf_press_1deg,
                                    show_fine_structure=show_fine_topography,
                                    surf_press_highres=surf_press_highres)
            
        except Exception as plot_err:
            # Log other potential errors during plotting for a season
            logging.error(f"Error plotting season {season_code} for {predictor_name}: {plot_err}", exc_info=True)
            ax.text(0.5, 0.5, "Plotting Error", transform=ax.transAxes,
                    ha='center', va='center', fontsize=12, color='red')
            ax.set_title(f"{season_disp}: {predictor_name} effect - ERROR", fontsize=16)
    
    # Add a single colorbar at the bottom after the loop
    if cf_mappable is not None:
        cax = fig.add_axes([0.25, 0.02, 0.5, 0.03])  # Position: [left, bottom, width, height]
        cbar = plt.colorbar(cf_mappable, cax=cax, orientation='horizontal')
        cbar.set_label(VERTICAL_SF_SETTINGS['label'], fontsize=16)
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
    # plt.suptitle(f"Vertical Streamfunction Response to {predictor_name}{pathway_str}\n" +
    #              f"({lat_str} average, {method}, {lag_range_str}, {mask_desc}) [{years[0]}-{years[1]}]",
    #              fontsize=18, y=0.99)
    plt.subplots_adjust(left=0.04, bottom=0.12, right=0.96, top=0.95, hspace=0.25)
    
    # Define output directory with latitude band subfolder
    dir_components = [path_save, config_name, method.lower(), lag_range_str, "vert_zon_sf"]
    base_output_dir = os.path.join(*dir_components)
    
    # Create latitude band subdirectory inside vert_zon_sf
    band_name = LAT_BAND_NAMES.get(lat_band, f"lat{lat_band[0]}-{lat_band[1]}")
    output_dir = os.path.join(base_output_dir, band_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Filename doesn't need band name since it's in the folder structure now
    filename = f"vertical_sf_{method.lower()}_{predictor_name.replace('.', '_')}_{config_name}_{lag_range_str}_{mask_desc.replace('% conf','pct').replace(' ','_')}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save the figure
    try:
        save_figure_optimized(fig, filepath.replace('.png', ''), output_format, raster_dpi, vector_dpi)
    except Exception as save_err:
        logging.error(f"Failed to save plot {filepath}: {save_err}")
    finally:
        plt.close(fig)
        gc.collect()

def plot_r_squared_vertical(
    rsq: xr.DataArray,
    predictors: List[str],  # List of predictors included in this model run
    method: str,
    lag_range_str: str,  # Full lag range string
    lat_band: Tuple[float, float] = (-5, 5),  # Latitude band for averaging
    start_season: str = SEASON_CODES[0],
    path_save: str = SAVE_DIR,
    config_name: str = "",  # Corresponds to pathway ('EP', 'CP', 'COMBINED')
    years: Tuple[str, str] = DEFAULT_YEARS,
    include_topography: bool = True,
    output_format: str = 'png',
    single_panel: bool = False,
    panel_suffix: str = '',
    seasons_filter: Optional[List[str]] = None,
    raster_dpi: int = 150,
    vector_dpi: int = 300
) -> None:
    """
    Plot the spatial distribution of R² across seasons.
    """
    logging.info(f"Plotting R² for vertical streamfunction ({method}, {lag_range_str}, Pathway: {config_name})")
    
    # Pre-check for all NaNs or Infs
    if rsq is None or np.all(np.isnan(rsq)):
        logging.error(f"R² data is None or all NaN for vertical streamfunction ({method}, {lag_range_str}, {config_name}). Skipping R² plot.")
        return
    rsq = rsq.where(~np.isinf(rsq), np.nan)  # Replace infinities with NaN
    
    start_season_code = get_season_code(start_season)
    try:
        start_idx = SEASON_CODES.index(start_season_code)
    except ValueError:
        logging.warning(f"Start season code {start_season_code} not found. Defaulting to first season.")
        start_idx = 0
    seasons_to_plot = SEASON_CODES[start_idx:]
    
    n_seasons = len(seasons_to_plot)
    fig_height = 6 * n_seasons
    fig, axs = plt.subplots(n_seasons, 1, figsize=(21, fig_height),
                            subplot_kw={'aspect': 'auto'})
    if n_seasons == 1:
        axs = [axs]  # Ensure axs is always iterable
    
    # Process surface elevation data
    if include_topography:
        surf_press_1deg, _ = process_surface_elevation_for_plot(lat_band)
    else:
        surf_press_1deg = None
    
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
                logging.warning(f"All R² values are NaN for vertical streamfunction, season {season_code}")
                ax.text(0.5, 0.5, "No valid R² data", transform=ax.transAxes,
                        ha='center', va='center', fontsize=14, color='gray')
                ax.set_title(f"{season_disp}: R²", fontsize=18)
                continue
            
            # Mask low R² values
            r2_masked = r2_eq.where(r2_eq > r2_settings['alpha_threshold'])
            
            # Plot with conditional rasterization
            pcolormesh_kwargs = get_pcolormesh_kwargs(0, 1, r2_settings['cmap'], None, output_format)
            cf = ax.pcolormesh(r2_eq.longitude, r2_eq.pressure_level, r2_eq.values, **pcolormesh_kwargs)
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
            
            # Configure axes
            ax.set_yscale('log')
            ax.invert_yaxis()
            ax.set_ylim(1000, 200)
            ax.set_xlim(-180, 180)
            
            # Set y-axis ticks - include ALL ticks that might appear
            # Major ticks
            major_ticks = [1000, 850, 700, 500, 300, 200]
            ax.set_yticks(major_ticks)
            
            # Minor ticks - include intermediate values that matplotlib might add
            all_possible_ticks = [1000, 900, 850, 800, 700, 600, 500, 400, 300, 200]
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
            
            # Add surface topography
            if surf_press_1deg is not None:
                add_orography_to_plot(ax, surf_press_1deg)
            
            # Add Walker circulation references
            add_walker_circulation_annotations(ax)
            
            # Grid
            ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
            
        except Exception as plot_err:
            logging.error(f"Error plotting R² for season {season_code}: {plot_err}", exc_info=True)
            ax.text(0.5, 0.5, "Plotting Error", transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='red')
            ax.set_title(f"{season_disp}: R² - ERROR", fontsize=16)
    
    # Add colorbar at the bottom
    if valid_cf is not None:
        cax = fig.add_axes([0.3, 0.02, 0.4, 0.02])
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
    # plt.suptitle(f"Explained variance (R²) of Vertical Streamfunction by {predictor_title_part}{pathway_str}\n" +
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
    filename = f"r_squared_vertical_sf_{method.lower()}_{lag_range_str}_{config_name}_by_{predictor_title_part}.png"
    filepath = os.path.join(output_dir, filename)
    
    # Save the figure
    try:
        save_figure_optimized(fig, filepath.replace('.png', ''), output_format, raster_dpi, vector_dpi)
    except Exception as e:
        logging.error(f"Failed to save R² plot {filepath}: {e}")
    finally:
        plt.close(fig)
        gc.collect()

# =============================================================================
# PROCESSING FUNCTIONS
# =============================================================================

def process_vertical_streamfunction(pathway: str,  # Pathway context ('EP', 'CP', 'COMBINED')
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
    Process vertical streamfunction for a specific pathway, lag range, method, and latitude band.
    Loads results, plots effects, plots R².
    """
    # Target variable is always vertical_streamfunction
    target_var = 'vertical_streamfunction'
    # Use pathway as the config_name for directory structure and titles
    config_name = pathway
    
    # The lat_band directory will be created inside the vert_zon_sf subdirectory
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
        plot_kwargs = {
            'regr_coefs': regr_coefs,
            'pvals': pvals,
            'predictor_name': predictor,
            'method': method,
            'lag_range_str': lag_range_str,
            'lat_band': lat_band,
            'sign_level': sign_level,
            'start_season': start_season,
            'path_save': path_save,
            'config_name': config_name,
            'years': (start_year, end_year),
            'include_topography': True,
            'show_fine_topography': False,
            'output_format': output_format,
            'single_panel': single_panel,
            'panel_suffix': panel_suffix,
            'seasons_filter': seasons_filter,
            'raster_dpi': raster_dpi,
            'vector_dpi': vector_dpi
        }
        
        plot_predictor_effect(**plot_kwargs)
    
    # Plot R-squared
    if predictors_to_plot:
        r2_kwargs = {
            'rsq': rsq,
            'predictors': predictors_to_plot,
            'method': method,
            'lag_range_str': lag_range_str,
            'lat_band': lat_band,
            'start_season': start_season,
            'path_save': path_save,
            'config_name': config_name,
            'years': (start_year, end_year),
            'include_topography': True,
            'output_format': output_format,
            'single_panel': single_panel,
            'panel_suffix': panel_suffix,
            'seasons_filter': seasons_filter,
            'raster_dpi': raster_dpi,
            'vector_dpi': vector_dpi
        }
        
        plot_r_squared_vertical(**r2_kwargs)
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
    pathway context(s) for each set, and processes vertical streamfunction accordingly.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Starting vertical streamfunction regression plots script")

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Generate vertical streamfunction regression plots')
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

    args = parser.parse_args()

    # Log CLI arguments received
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
                    # Call process_vertical_streamfunction
                    process_vertical_streamfunction(
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
