"""
Dynamic Predictive Modeling for ENSO Diversity Indices

This script builds and evaluates causal prediction models for ENSO diversity
indices (E, C) using predictors identified from dynamically generated or
pre-existing PCMCI+ graphs.

Features:
- Multiple model types (linear, Lasso, Ridge, Random Forest, Gradient Boosting)
- Configurable cross-validation strategies
- Dynamic causal graph generation or use of existing graphs
- Hyperparameter tuning for applicable models
- Comprehensive metadata tracking

Usage Examples:
    # Use default LassoCV model with dynamic graph generation
    python 20_predictive_model_dynamic_sets.py

    # Use Random Forest with thorough cross-validation
    python 20_predictive_model_dynamic_sets.py --model random_forest --cv-preset thorough

    # Use existing graphs with linear model
    python 20_predictive_model_dynamic_sets.py --model linear \
        --use-existing-graph --graph-suffix "PCMCI_hard"

    # Custom model parameters
    python 20_predictive_model_dynamic_sets.py --model ridge \
        --model-params '{"alpha": 10.0, "solver": "svd"}'

Dependencies:
    - utils.model_configs: Model configuration management
    - utils.model_factory: Model instantiation
    - utils.causal_discovery_utils: Causal graph generation
    - utils.paths: Path management
    - utils.predictor_configs: Predictor set definitions
"""

# --- Imports ---
import os
import sys
import logging
import argparse
import json
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
import gc
from tigramite import data_processing as pp
from tigramite.models import Models

# Set matplotlib backend before importing pyplot to avoid threading issues
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from typing import List, Tuple, Dict, Optional, Set, Any

# --- Add project root and utils to Python path ---
try:
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Also add src directory to path to find utils
    src_path = os.path.join(project_root, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from utils.paths import get_data_path, get_results_path
    from utils import causal_discovery_utils as cdu
    from utils.model_configs import (ModelConfig, get_available_models,
                                   get_available_cv_presets, print_model_info)
    from utils.model_factory import ModelFactory, adjust_param_grid_for_pipeline
    from utils.predictor_configs import (PredictorConfig, get_alpha_preset,
                                        get_available_alpha_presets)
    from utils.plotting_optimization import (
        save_figure_optimized, add_plotting_arguments,
        filter_seasons_to_plot, create_clean_panel_title,
        create_descriptive_filename
    )
except ImportError as e:
    logging.error(f"Could not import required modules. Ensure utils/ is accessible. Details: {e}")
    sys.exit(1)
except Exception as e:
    logging.error(f"An unexpected error occurred during path setup or import: {e}")
    sys.exit(1)


# --- Argument Parsing ---
def parse_arguments():
    """Parse command-line arguments for prediction model configuration."""
    parser = argparse.ArgumentParser(
        description='Dynamic Predictive Modeling with Configurable Models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available model presets:
  lassocv          : Lasso with built-in CV (default)
  linear           : Simple linear regression
  ridge            : Ridge regression with hyperparameter tuning
  random_forest    : Random Forest with hyperparameter tuning
  gradient_boosting: Gradient Boosting with hyperparameter tuning

Available CV presets:
  quick    : 3-fold outer, 2-fold inner CV
  standard : 5-fold outer, 3-fold inner CV (default)
  thorough : 10-fold outer, 5-fold inner CV

Examples:
  python 20_predictive_model_dynamic_sets.py --model lassocv
  python 20_predictive_model_dynamic_sets.py --model random_forest --cv-preset thorough
  python 20_predictive_model_dynamic_sets.py --model ridge --predictor-set known
        """
    )
    
    # Model selection arguments
    parser.add_argument(
        '--model', '-m',
        choices=get_available_models(),
        default='lassocv',
        help='Prediction model preset to use (default: lassocv)'
    )
    
    parser.add_argument(
        '--cv-preset',
        choices=get_available_cv_presets(),
        default='standard',
        help='Cross-validation preset (default: standard)'
    )
    
    # Predictor set arguments (reuse from PCMCI+)
    parser.add_argument(
        '--predictor-set', '-p',
        choices=['known', 'new', 'combined', 'custom'],
        default='combined',
        help='Predictor set to use (default: combined)'
    )
    
    parser.add_argument(
        '--predictor-config',
        type=str,
        help='Path to custom predictor configuration file'
    )
    
    # Model customization
    parser.add_argument(
        '--model-params',
        type=json.loads,
        help='JSON string of custom model parameters'
    )
    
    parser.add_argument(
        '--param-grid',
        type=json.loads,
        help='JSON string of custom parameter grid for tuning'
    )
    
    # Graph generation arguments
    parser.add_argument(
        '--use-existing-graph',
        action='store_true',
        help='Use existing causal graphs instead of regenerating'
    )
    
    parser.add_argument(
        '--graph-suffix',
        type=str,
        help='Suffix of existing graph files to use'
    )
    
    # Alpha configuration
    parser.add_argument(
        '--alpha-preset',
        choices=['mild', 'hard', 'custom'],
        default='hard',
        help='Significance level preset: mild/hard/custom (default: hard)'
    )
    
    parser.add_argument(
        '--alpha-values',
        nargs='+',
        type=float,
        help='Custom alpha values (required when --alpha-preset=custom)'
    )
    
    # Conditional independence test
    parser.add_argument(
        '--cond-ind-test',
        choices=['robust_parcorr', 'gpdc'],
        default='robust_parcorr',
        help='Conditional independence test (default: robust_parcorr)'
    )
    
    # FDR configuration
    parser.add_argument(
        '--enable-fdr',
        action='store_true',
        help='Activate post-hoc FDR-BH pruning step (default off)'
    )
    
    parser.add_argument(
        '--fdr-pc-alpha',
        type=float,
        default=0.10,
        help='Significance level for FDR-BH method (default 0.10, only used if --enable-fdr)'
    )
    
    # Output control
    parser.add_argument(
        '--output-suffix',
        type=str,
        help='Custom suffix for output files'
    )
    
    parser.add_argument(
        '--list-models',
        action='store_true',
        help='List available models and exit'
    )
    
    # Add standard plotting arguments for optimization
    add_plotting_arguments(parser)
    
    # Legend control arguments
    parser.add_argument(
        '--show-legend',
        action='store_true',
        default=True,
        help='Show legend in comparison plots (default: True)'
    )
    parser.add_argument(
        '--no-legend',
        dest='show_legend',
        action='store_false',
        help='Hide legend in comparison plots'
    )

    # Categorical metrics arguments
    parser.add_argument(
        '--event-threshold',
        type=float,
        default=0.5,
        help='Threshold (in s.d.) for defining ENSO events (default: 0.5)'
    )

    parser.add_argument(
        '--compute-categorical',
        action='store_true',
        help='Compute categorical forecast metrics (hit rate) in addition to correlation'
    )

    # Multi-panel figure generation
    parser.add_argument(
        '--generate-multipanel',
        action='store_true',
        help='Generate multi-panel comparison figure from existing LassoCV and RF results'
    )

    parser.add_argument(
        '--lasso-results-path',
        type=str,
        help='Path to LassoCV results CSV (required with --generate-multipanel)'
    )

    parser.add_argument(
        '--rf-results-path',
        type=str,
        help='Path to Random Forest results CSV (required with --generate-multipanel)'
    )

    return parser.parse_args()


# --- Default Configuration ---
# These values will be updated based on command-line arguments in main execution
DATA_FILE_PATH = get_data_path('time_series/PCMCI_data_ts_st.nc', data_type="processed")

# Analysis Parameters (defaults, will be overridden by CLI args)
TARGET_VARIABLES = ['E-ind DJF(1)', 'C-ind DJF(1)']
TAU_MIN = 1  # Min lag for causal discovery
TAU_MAX = 6  # Max lag for causal discovery AND prediction model fitting
VERBOSITY = 0  # Tigramite verbosity (for prediction Models)
CAUSAL_DISCOVERY_VERBOSITY = 0  # Verbosity for PCMCI+ runs in cdu

# Default values that will be set by command-line arguments
MODEL_NAME = 'LassoCV'  # Will be set by model configuration
N_CV_SPLITS = 5  # Will be set by CV configuration
INNER_CV_SPLITS = 3  # Will be set by CV configuration
# ALPHA_VALUES will be set based on preset or custom values
BASE_SUFFIX = 'PCMCI_hard'  # Base suffix for output files

# Output filename templates (will use dynamic values)
COMPARISON_RESULTS_FILENAME = "prediction_comparison_{model_name}_{graph_suffix}.csv"
PREDICTIONS_FILENAME_TEMPLATE = "predictions_{target}_{predictor_set}_{max_season}_{model_name}_{graph_suffix}.csv"
COMPARISON_PLOT_FILENAME = "prediction_skill_comparison_{model_name}_{graph_suffix}.png"
SCATTER_PLOT_FILENAME_TEMPLATE = "prediction_scatter_{target}_{predictor_set}_{max_season}_{model_name}_{graph_suffix}.png"

# --- Configuration Using PredictorConfig ---
# Create predictor configurations using the utility module
PREDICTOR_CONFIGS = {
    'Known': PredictorConfig('known'),
    'New': PredictorConfig('new'), 
    'Combined': PredictorConfig('combined')
}
PREDICTOR_SET_TYPES = ['Known', 'New', 'Combined']

# Define cumulative seasons for iteration
CUMULATIVE_SEASONS_ORDER = ['JJA(-1)', 'SON(-1)', 'DJF(0)', 'MAM(0)']

# Use combined config to get the complete variable-season mapping and season-months mapping
_combined_config = PREDICTOR_CONFIGS['Combined']
VARIABLE_SEASON_MAP = _combined_config.get_variables()
SEASON_MONTHS_MAP = _combined_config.get_season_months()

# Convert old season names to new ones for this script
# The script uses different season naming: JJA(-1), SON(-1), DJF(0), MAM(0), DJF_effect
SEASON_NAME_MAPPING = {
    'JJA': 'JJA(-1)',
    'SON': 'SON(-1)', 
    'DJF': 'DJF(0)',
    'MAM': 'MAM(0)',
    'DJF_effect': 'DJF_effect'
}

# Update variable season map with script-specific season names
VARIABLE_SEASON_MAP = {var: SEASON_NAME_MAPPING.get(season, season) 
                      for var, season in VARIABLE_SEASON_MAP.items()}

# Update season months map with script-specific season names
SEASON_MONTHS_MAP = {SEASON_NAME_MAPPING.get(season, season): months 
                    for season, months in SEASON_MONTHS_MAP.items()}

# Helper functions to get predictor lists from configs
def get_predictor_groups_for_config(config: PredictorConfig) -> Dict[str, List[str]]:
    """Get variable groups for a predictor configuration."""
    groups = config.get_variable_groups()
    return {
        'JJA_PREDICTORS': groups.get('JJA_PREDICTORS', []),
        'SON_PREDICTORS': groups.get('SON_PREDICTORS', []),
        'DJF_CONFOUNDERS': groups.get('DJF_CONFOUNDERS', []),
        'MAM_MEDIATORS': groups.get('MAM_MEDIATORS', []),
        'DJF_EFFECTS': groups.get('DJF_EFFECTS', [])
    }

# Get groups from combined config for global use
_combined_groups = get_predictor_groups_for_config(_combined_config)
JJA_PREDICTORS = _combined_groups['JJA_PREDICTORS']
SON_PREDICTORS = _combined_groups['SON_PREDICTORS'] 
DJF_CONFOUNDERS = _combined_groups['DJF_CONFOUNDERS']
MAM_MEDIATORS = _combined_groups['MAM_MEDIATORS']
DJF_EFFECTS = _combined_groups['DJF_EFFECTS']

# --- Basic Logging Setup ---
# Detailed logging setup is handled in main execution after argument parsing
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Global variable for var_names (from FULL dataset load) ---
# This list is crucial for mapping names in the generated graph CSV back to indices
# in the full_dataframe used for prediction.
var_names_full: List[str] = []

# --- Helper Functions ---


def build_and_save_causal_graph(
    predictors: List[str],
    targets: List[str],
    save_dir: str,
    suffix: str,
    data_file_path: str,
    variable_season_map: Dict[str, str],
    season_months_map: Dict[str, List[int]],
    tau_min: int,
    tau_max: int,
    alpha_values: List[float],
    cond_ind_test: str = 'robust_parcorr',
    enable_fdr: bool = False,
    fdr_pc_alpha: float = 0.10,
    # Optional args for cdu functions if not using defaults:
    max_combinations: Optional[int] = None,
    persistence_args: Optional[Dict] = None,
    filter_args: Optional[Dict] = None,
    verbosity: int = 0
) -> Optional[str]:
    """
    Builds a causal graph using PCMCI+ for a given set of predictors and targets,
    and saves the filtered, robust graph to a CSV file.

    Args:
        predictors (List[str]): List of predictor variable names.
        targets (List[str]): List of target variable names.
        save_dir (str): Directory to save the output graph CSV.
        suffix (str): Suffix for the output filename (e.g., 'known', 'new').
        data_file_path (str): Path to the source NetCDF data.
        variable_season_map (Dict[str, str]): Map of var names to seasons.
        season_months_map (Dict[str, List[int]]): Map of seasons to months.
        tau_min (int): Minimum lag for PCMCI+.
        tau_max (int): Maximum lag for PCMCI+.
        alpha_values (List[float]): Alpha levels for iterative PCMCI+.
        cond_ind_test (str): Conditional independence test name.
        enable_fdr (bool): Whether to run FDR post-hoc pruning.
        fdr_pc_alpha (float): Alpha level for FDR-BH method.
        max_combinations (Optional[int]): Max combinations for PCMCI+.
        persistence_args (Optional[Dict]): Args for analyze_edge_persistence.
        filter_args (Optional[Dict]): Args for filter_graph.
        verbosity (int): Verbosity level for cdu functions.

    Returns:
        Optional[str]: Path to the saved graph CSV file, or None on failure.
    """
    logging.info(f"--- Building Causal Graph for Predictor Set: {suffix} ---")
    variables_for_graph = list(set(predictors + targets)) # Unique list

    # 1. Load & subset the data for these specific variables
    # 1. Load & subset the data for these specific variables
    #    -- maintain consistent predictor and target variable ordering
    variables_for_graph = list(dict.fromkeys(predictors + targets))
    data_array, mask_array, var_names_subset = cdu.load_and_prepare_data(
        file_path=data_file_path,
        variables_to_load=variables_for_graph,
        variable_seasons_map=variable_season_map,
        season_months_map=season_months_map,
        # Pass exclusion lists for dataset consistency
        jja_predictors=JJA_PREDICTORS,
        son_predictors=SON_PREDICTORS,
        djf_confounders=DJF_CONFOUNDERS,
        djf_effects=DJF_EFFECTS
    )
    if data_array is None or mask_array is None or var_names_subset is None:
        logging.error(f"Failed to load data for graph building ({suffix}). Skipping graph generation.")
        return None
    if len(var_names_subset) == 0:
        logging.error(f"No variables loaded for graph building ({suffix}). Skipping graph generation.")
        return None

    # 2. Create Tigramite DataFrame for the subset
    dataframe_subset = pp.DataFrame(
        data_array,
        mask=mask_array,
        var_names=var_names_subset # Use the names corresponding to this subset
    )
    logging.info(f"Subset Tigramite DataFrame created for {suffix}: {dataframe_subset.N} vars, T={dataframe_subset.T}")
    # Check T value
    if not isinstance(dataframe_subset.T, dict) or 0 not in dataframe_subset.T or dataframe_subset.T[0] <= 0:
         logging.error(f"Subset data for {suffix} has zero, negative, or unexpected time steps structure ({dataframe_subset.T}). Skipping graph generation.")
         return None


    # 3. Run iterative PCMCI+ (always needed for robust graph)
    # First run iterative PCMCI+ to get robust graph
    logging.info("Running iterative PCMCI+ for robust graph generation")
    run_args = {
        'dataframe': dataframe_subset, 
        'tau_min': tau_min, 
        'tau_max': tau_max, 
        'alpha_values': alpha_values, 
        'cond_ind_test_name': cond_ind_test,
        'verbosity': verbosity
    }
    if max_combinations is not None:
        run_args['max_combinations'] = max_combinations
    results_list = cdu.run_iterative_pcmciplus(**run_args)

    # Check if all runs failed
    if all(r is None for r in results_list):
        logging.error(f"All PCMCI+ runs failed for predictor set {suffix}. Cannot build graph.")
        return None
    # Filter out None results before proceeding
    successful_results = [r for r in results_list if r is not None]
    if not successful_results:
         logging.error(f"No successful PCMCI+ runs for predictor set {suffix}. Cannot build graph.")
         return None


    # 4. Track edge significance
    # Use var_names_subset corresponding to the dataframe_subset
    edge_tracking_df = cdu.track_edge_significance(
        results_list=results_list, # Pass original list (with Nones) or filtered list? cdu handles Nones.
        var_names=var_names_subset,
        alpha_values=alpha_values,
        tau_min=tau_min,
        tau_max=tau_max
    )

    # 5. Analyze edge persistence
    persist_args_merged = {'edge_tracking_df': edge_tracking_df, 'num_total_runs': len(alpha_values)}
    if persistence_args: # Merge user-provided args, overriding defaults in cdu if needed
        persist_args_merged.update(persistence_args)
    persistence_df = cdu.analyze_edge_persistence(**persist_args_merged)

    # 6. Create robust graph
    # Use var_names_subset and its length
    robust_graph, robust_val_matrix, robust_p_matrix = cdu.create_robust_graph(
        persistence_df=persistence_df,
        num_vars=len(var_names_subset),
        var_names=var_names_subset,
        tau_max=tau_max
    )

    # 7. Filter graph (optional, but usually desired)
    # Use var_names_subset here too
    filter_args_merged = {
        'graph': robust_graph,
        'val_matrix': robust_val_matrix,
        'p_matrix': robust_p_matrix,
        'var_names': var_names_subset,
        'target_variables': targets, # Filter paths leading to these targets
        'tau_min': tau_min,
        'tau_max': tau_max,
        'max_cumulative_lag': tau_max # Default max cumulative = tau_max
        # min_cumulative_lag will use default from cdu unless overridden in filter_args
    }
    if filter_args:
        filter_args_merged.update(filter_args)
    filtered_graph, filtered_val, filtered_p = cdu.filter_graph(**filter_args_merged)

    # 7.5. Apply FDR pruning if enabled
    if enable_fdr:
        logging.info(f"Applying FDR pruning with pc_alpha = {fdr_pc_alpha}")
        try:
            final_graph, final_val, final_p, fdr_info = cdu.apply_fdr_pruning(
                dataframe=dataframe_subset,
                robust_graph=filtered_graph,
                robust_val_matrix=filtered_val,
                robust_p_matrix=filtered_p,
                var_names=var_names_subset,
                target_variables=targets,
                tau_min=tau_min,
                tau_max=tau_max,
                fdr_pc_alpha=fdr_pc_alpha,
                cond_ind_test_name=cond_ind_test,
                max_cumulative_lag=filter_args_merged.get('max_cumulative_lag', tau_max),
                min_cumulative_lag=filter_args_merged.get('min_cumulative_lag', 3),
                max_combinations=max_combinations,
                verbosity=verbosity
            )
            # Update references for final output
            filtered_graph = final_graph
            filtered_val = final_val
            filtered_p = final_p
            logging.info(f"FDR pruning applied successfully for {suffix}")
        except Exception as e:
            logging.error(f"FDR pruning failed: {e}. Using non-FDR filtered graph.")
            # Continue with non-FDR results (filtered_graph remains unchanged)

    # 8. Write filtered graph CSV
    out_csv_path = os.path.join(save_dir, f'filtered_robust_graph_{suffix}.csv')
    cdu.write_graph_with_p_values(
        graph=filtered_graph,
        val_matrix=filtered_val,
        p_matrix=filtered_p,
        var_names=var_names_subset, # Use subset names for writing
        save_name=out_csv_path,
        tau_min=tau_min,
        tau_max=tau_max
    )

    if os.path.exists(out_csv_path):
        logging.info(f"--- Successfully built and saved graph: {out_csv_path} ---")
        return out_csv_path
    else:
        logging.error(f"--- Failed to save graph CSV for suffix {suffix} ---")
        return None




# --- Other Helper Functions (calculate_metrics, plot_comparison, generate_scatter_plot) ---
# These functions provide core functionality as they don't
# need modification for the dynamic graph generation logic. Ensure they use the
# correct global/local variables if accessed (e.g., MODEL_NAME, RESULTS_SAVE_DIR, etc.)

def calculate_metrics(y_true, y_pred):
    """Calculates RMSE and Pearson Correlation, handling NaNs."""
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        # Reduced verbosity as this can happen normally in CV folds
        # logging.warning(f"Cannot calculate metrics with less than 2 finite points (found {mask.sum()}).")
        return np.nan, np.nan, np.nan
    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    if np.std(y_true_valid) < 1e-9 or np.std(y_pred_valid) < 1e-9:
        # logging.warning("Cannot calculate Pearson correlation because standard deviation of true or predicted values is close to zero.")
        corr, p_value = np.nan, np.nan
    else:
        try:
            corr, p_value = pearsonr(y_true_valid, y_pred_valid)
        except ValueError as e:
            # logging.warning(f"Could not calculate Pearson correlation: {e}")
            corr, p_value = np.nan, np.nan
    return rmse, corr, p_value


def calculate_categorical_metrics(y_true, y_pred, threshold=0.5):
    """
    Calculate categorical forecast verification metrics.

    Events are defined as |value| > threshold (standardized units).
    Both El Nino (positive) and La Nina (negative) events are captured.

    Args:
        y_true: Array of observed values
        y_pred: Array of predicted values
        threshold: Event threshold in standard deviations (default 0.5)

    Returns:
        dict with hit_rate, miss_rate, contingency table counts
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return {
            'hit_rate': np.nan,
            'miss_rate': np.nan,
            'hits': 0,
            'misses': 0,
            'false_alarms': 0,
            'correct_negatives': 0,
            'n_events_observed': 0
        }

    y_true_valid = y_true[mask]
    y_pred_valid = y_pred[mask]

    # Event = |value| > threshold (captures both El Nino and La Nina)
    observed_event = np.abs(y_true_valid) > threshold
    forecast_event = np.abs(y_pred_valid) > threshold

    # Contingency table
    hits = np.sum(observed_event & forecast_event)
    misses = np.sum(observed_event & ~forecast_event)
    false_alarms = np.sum(~observed_event & forecast_event)
    correct_negatives = np.sum(~observed_event & ~forecast_event)

    # Hit Rate = Hits / (Hits + Misses)
    n_events_observed = hits + misses
    hit_rate = hits / n_events_observed if n_events_observed > 0 else np.nan
    miss_rate = 1 - hit_rate if not np.isnan(hit_rate) else np.nan

    return {
        'hit_rate': hit_rate,
        'miss_rate': miss_rate,
        'hits': int(hits),
        'misses': int(misses),
        'false_alarms': int(false_alarms),
        'correct_negatives': int(correct_negatives),
        'n_events_observed': int(n_events_observed)
    }


def plot_comparison(results_df: pd.DataFrame, save_path: str, output_format: str = 'png',
                   raster_dpi: int = 150, vector_dpi: int = 300, show_legend: bool = True):
    """Generates and saves the comparison plot with optimization support.
    
    Args:
        results_df: DataFrame with prediction results
        save_path: Path to save the figure (without extension)
        output_format: Output format ('png', 'pdf', 'svg', 'both')
        raster_dpi: DPI for rasterized elements in vector formats
        vector_dpi: DPI for pure raster formats
    """
    logging.info("Generating prediction skill comparison plot...")
    
    # Exact dimensions in inches for guaranteed square plot area
    AXES_SIZE = 8.0
    MARGIN_LEFT = 1.0    
    MARGIN_RIGHT = 1.0
    MARGIN_TOP = 0.5
    MARGIN_BOTTOM = 0.5
    LEGEND_HEIGHT = 1.7  # Calculated for title (14pt) + 3 rows (10pt each)
    LEGEND_GAP = 0.5     # Gap between axes and legend
    
    if show_legend:
        fig_width = MARGIN_LEFT + AXES_SIZE + MARGIN_RIGHT  # 10.0 inches
        fig_height = MARGIN_TOP + AXES_SIZE + LEGEND_GAP + LEGEND_HEIGHT + MARGIN_BOTTOM  # 11.2 inches
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Set exact position to guarantee square axes
        ax.set_position([
            MARGIN_LEFT / fig_width,                                    # 0.10
            (MARGIN_BOTTOM + LEGEND_HEIGHT + LEGEND_GAP) / fig_height,  # 0.241
            AXES_SIZE / fig_width,                                      # 0.80
            AXES_SIZE / fig_height                                      # 0.714
        ])
    else:
        fig_width = MARGIN_LEFT + AXES_SIZE + MARGIN_RIGHT  # 10.0 inches
        fig_height = MARGIN_TOP + AXES_SIZE + MARGIN_BOTTOM  # 9.0 inches
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        ax.set_position([
            MARGIN_LEFT / fig_width,   # 0.10
            MARGIN_BOTTOM / fig_height, # 0.056
            AXES_SIZE / fig_width,      # 0.80
            AXES_SIZE / fig_height      # 0.889
        ])

    colors = {'E-ind DJF(1)': 'green', 'C-ind DJF(1)': 'red'}
    markers = {'Known': 'o', 'New': 's', 'Combined': '^'}
    linestyles = {'Known': ':', 'New': '--', 'Combined': '-'}

    # Ensure MaxLagSeason is treated as categorical in the correct order
    results_df['MaxLagSeason'] = pd.Categorical(results_df['MaxLagSeason'], categories=CUMULATIVE_SEASONS_ORDER, ordered=True)
    results_df = results_df.sort_values(['Target', 'PredictorSet', 'MaxLagSeason']) # Sort for consistent plotting

    # Correlation line plot
    for target in TARGET_VARIABLES:
        for pset in PREDICTOR_SET_TYPES:
            subset = results_df[(results_df['Target'] == target) & (results_df['PredictorSet'] == pset)].dropna(subset=['Correlation_Overall'])
            if not subset.empty:
                label = f"{target.split('_')[0]} ({pset})"
                ax.plot(subset['MaxLagSeason'], subset['Correlation_Overall'],
                        label=label,
                        color=colors.get(target, 'black'), # Use get for safety
                        marker=markers.get(pset, '.'),
                        linestyle=linestyles.get(pset, '-'))

    ax.set_xlabel("Predictors up to Season", fontsize=12)
    ax.set_ylabel("Prediction Skill ($r$)", fontsize=12)
    # ax.set_title(f"ENSO Prediction Skill Comparison ({MODEL_NAME}) - Correlation") 
    if show_legend:
        leg = ax.legend(title="Target (Predictor Set)", 
                       bbox_to_anchor=(0.5, -0.12),  # Adjusted for new spacing
                       loc='upper center', 
                       ncol=2,  # Horizontal layout with 2 columns
                       frameon=True,
                       fancybox=True,
                       shadow=False,
                       fontsize=10)
        leg.get_title().set_fontsize(14)  # Set legend title fontsize to 14
    # No tight_layout needed with explicit positioning
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45, ha='right')  # Adjust rotation

    try:
        # Remove extension from save_path for save_figure_optimized
        base_path = os.path.splitext(save_path)[0] if save_path.endswith(('.png', '.pdf', '.svg')) else save_path
        save_figure_optimized(fig, base_path, output_format, raster_dpi, vector_dpi)
        logging.info(f"Correlation comparison plot saved with format: {output_format}")
    except Exception as e:
        logging.error(f"Failed to save correlation comparison plot: {e}")
    plt.close(fig)
    if 'gc' in sys.modules:
        gc.collect()

    # RMSE line plot
    # Exact dimensions in inches for guaranteed square plot area (same as correlation plot)
    AXES_SIZE = 8.0
    MARGIN_LEFT = 1.0    
    MARGIN_RIGHT = 1.0
    MARGIN_TOP = 0.5
    MARGIN_BOTTOM = 0.5
    LEGEND_HEIGHT = 1.7  # Calculated for title (14pt) + 3 rows (10pt each)
    LEGEND_GAP = 0.5     # Gap between axes and legend
    
    if show_legend:
        fig_width = MARGIN_LEFT + AXES_SIZE + MARGIN_RIGHT  # 10.0 inches
        fig_height = MARGIN_TOP + AXES_SIZE + LEGEND_GAP + LEGEND_HEIGHT + MARGIN_BOTTOM  # 11.2 inches
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # Set exact position to guarantee square axes
        ax.set_position([
            MARGIN_LEFT / fig_width,                                    # 0.10
            (MARGIN_BOTTOM + LEGEND_HEIGHT + LEGEND_GAP) / fig_height,  # 0.241
            AXES_SIZE / fig_width,                                      # 0.80
            AXES_SIZE / fig_height                                      # 0.714
        ])
    else:
        fig_width = MARGIN_LEFT + AXES_SIZE + MARGIN_RIGHT  # 10.0 inches
        fig_height = MARGIN_TOP + AXES_SIZE + MARGIN_BOTTOM  # 9.0 inches
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        ax.set_position([
            MARGIN_LEFT / fig_width,   # 0.10
            MARGIN_BOTTOM / fig_height, # 0.056
            AXES_SIZE / fig_width,      # 0.80
            AXES_SIZE / fig_height      # 0.889
        ])
    for target in TARGET_VARIABLES:
        for pset in PREDICTOR_SET_TYPES:
            subset = results_df[(results_df['Target'] == target) & (results_df['PredictorSet'] == pset)].dropna(subset=['RMSE_Overall'])
            if not subset.empty:
                label = f"{target.split('_')[0]} ({pset})"
                ax.plot(subset['MaxLagSeason'], subset['RMSE_Overall'],
                        label=label,
                        color=colors.get(target, 'black'),
                        marker=markers.get(pset, '.'),
                        linestyle=linestyles.get(pset, '-'))

    ax.set_xlabel("Predictors up to Season")
    ax.set_ylabel("Prediction Skill (RMSE)")
    ax.set_title(f"ENSO Prediction Skill Comparison ({MODEL_NAME}) - RMSE")
    if show_legend:
        leg = ax.legend(title="Target (Predictor Set)", 
                       bbox_to_anchor=(0.5, -0.12),  # Adjusted for new spacing
                       loc='upper center', 
                       ncol=2,  # Horizontal layout with 2 columns
                       frameon=True,
                       fancybox=True,
                       shadow=False,
                       fontsize=10)
        leg.get_title().set_fontsize(14)  # Set legend title fontsize to 14
    # No tight_layout needed with explicit positioning
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(rotation=45, ha='right')  # Adjust rotation

    try:
        # Create RMSE-specific save path
        base_path = os.path.splitext(save_path)[0] if save_path.endswith(('.png', '.pdf', '.svg')) else save_path
        rmse_base_path = f"{base_path}_RMSE"
        save_figure_optimized(fig, rmse_base_path, output_format, raster_dpi, vector_dpi)
        logging.info(f"RMSE comparison plot saved with format: {output_format}")
    except Exception as e:
        logging.error(f"Failed to save RMSE comparison plot: {e}")
    plt.close(fig)
    if 'gc' in sys.modules:
        gc.collect()


def generate_scatter_plot(
    y_true_agg: np.ndarray,
    y_pred_agg: np.ndarray,
    target_name: str,
    predictor_set_type: str,
    max_season: str,
    model_name: str,
    graph_suffix: str, # Use the DYNAMIC_GRAPH_SUFFIX here
    overall_corr: float,
    overall_rmse: float,
    results_save_dir: str, # Pass the correct output dir
    output_format: str = 'png',
    raster_dpi: int = 150,
    vector_dpi: int = 300
):
    """Generates and saves a scatter plot of true vs. predicted values with optimization.
    
    Args:
        y_true_agg: True values array
        y_pred_agg: Predicted values array
        target_name: Name of target variable
        predictor_set_type: Type of predictor set used
        max_season: Maximum season included
        model_name: Name of the model
        graph_suffix: Suffix for the graph
        overall_corr: Overall correlation coefficient
        overall_rmse: Overall RMSE
        results_save_dir: Directory to save results
        output_format: Output format ('png', 'pdf', 'svg', 'both')
        raster_dpi: DPI for rasterized elements in vector formats
        vector_dpi: DPI for pure raster formats
    """
    # Check for sufficient valid data points
    valid_plot_mask = np.isfinite(y_true_agg) & np.isfinite(y_pred_agg)
    num_valid_points = np.sum(valid_plot_mask)

    if num_valid_points < 2:
        logging.warning(f"    Skipping scatter plot for {target_name}, Set: {predictor_set_type}, Max Season: {max_season} due to insufficient valid points ({num_valid_points}).")
        return

    logging.info(f"    Generating scatter plot for Target: {target_name}, Set: {predictor_set_type}, Max Season: {max_season}")
    try:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.gca()
        y_true_plot = y_true_agg[valid_plot_mask]
        y_pred_plot = y_pred_agg[valid_plot_mask]
        
        # Determine if rasterization is needed based on point count and format
        use_rasterized = (num_valid_points > 1000) and (output_format in ['pdf', 'svg', 'both'])
        
        # Scatter plot with optional rasterization for many points
        scatter = ax.scatter(y_true_plot, y_pred_plot, alpha=0.5, 
                           label=f'Data (n={num_valid_points})',
                           rasterized=use_rasterized if use_rasterized else None)

        min_val = min(np.min(y_true_plot), np.min(y_pred_plot))
        max_val = max(np.max(y_true_plot), np.max(y_pred_plot))
        if np.isclose(min_val, max_val): # Use isclose for float comparison
            min_val -= 0.1 * abs(min_val) if not np.isclose(min_val, 0) else 0.1
            max_val += 0.1 * abs(max_val) if not np.isclose(max_val, 0) else 0.1
        # Keep 1:1 line as vector for clarity
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 Line')

        ax.set_xlabel("True Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title(f"{target_name} Prediction ({model_name} - {predictor_set_type} - {max_season})\nCorr={overall_corr:.3f}, RMSE={overall_rmse:.3f}")
        ax.legend()
        ax.grid(True)
        ax.axis('equal') # Ensure aspect ratio is equal

        # Use the template and the DYNAMIC_GRAPH_SUFFIX
        plot_filename = SCATTER_PLOT_FILENAME_TEMPLATE.format(
            target=target_name,
            predictor_set=predictor_set_type,
            max_season=max_season,
            model_name=model_name,
            graph_suffix=graph_suffix # Pass DYNAMIC_GRAPH_SUFFIX here
        )
        # Remove extension for save_figure_optimized
        plot_filename_base = os.path.splitext(plot_filename)[0] if plot_filename.endswith('.png') else plot_filename
        plot_save_path = os.path.join(results_save_dir, plot_filename_base)
        
        save_figure_optimized(fig, plot_save_path, output_format, raster_dpi, vector_dpi)
        plt.close(fig)
        if 'gc' in sys.modules:
            gc.collect()
        logging.info(f"    Prediction scatter plot saved with format: {output_format}")
    except Exception as plot_err:
        logging.error(f"    Failed to generate or save scatter plot for {target_name}, Set: {predictor_set_type}, Max Season: {max_season}: {plot_err}", exc_info=True)
        plt.close(fig) # Ensure plot is closed on error
        if 'gc' in sys.modules:
            gc.collect()


def _plot_metric_panel(ax, df, metric_col, colors, markers, linestyles,
                       ylabel, title='', xlabel='Predictors up to Season'):
    """
    Helper function to plot a single panel in the comparison figure.

    Args:
        ax: matplotlib axes object
        df: DataFrame with results for one model
        metric_col: Column name for the metric to plot
        colors: Dict mapping target names to colors
        markers: Dict mapping predictor set types to markers
        linestyles: Dict mapping predictor set types to linestyles
        ylabel: Y-axis label
        title: Panel title (optional, default='' for no title)
        xlabel: X-axis label
    """
    df = df.copy()
    df['MaxLagSeason'] = pd.Categorical(
        df['MaxLagSeason'],
        categories=CUMULATIVE_SEASONS_ORDER,
        ordered=True
    )
    df = df.sort_values(['Target', 'PredictorSet', 'MaxLagSeason'])

    for target in TARGET_VARIABLES:
        for pset in PREDICTOR_SET_TYPES:
            subset = df[(df['Target'] == target) &
                       (df['PredictorSet'] == pset)].dropna(subset=[metric_col])
            if not subset.empty:
                # Shorter label: "E (Known)" instead of full name
                target_short = 'E' if 'E-ind' in target else 'C'
                label = f"{target_short} ({pset})"
                ax.plot(subset['MaxLagSeason'], subset[metric_col],
                        label=label,
                        color=colors.get(target, 'black'),
                        marker=markers.get(pset, '.'),
                        linestyle=linestyles.get(pset, '-'),
                        markersize=6,
                        linewidth=1.5)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold', loc='center')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.tick_params(axis='both', labelsize=12)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Set y-axis limits for consistency
    if 'Correlation' in metric_col:
        ax.set_ylim([0, 1])
    elif 'HitRate' in metric_col:
        ax.set_ylim([0, 1])


def plot_comparison_row(
    results_df_lasso: pd.DataFrame,
    results_df_rf: pd.DataFrame,
    metric_col: str,
    save_path: str,
    ylabel: str,
    left_title: str = 'LassoCV',
    right_title: str = 'Random Forest',
    output_format: str = 'pdf',
    raster_dpi: int = 150,
    vector_dpi: int = 300
):
    """
    Generate a single row (1x2) comparison plot with shared legend below.

    NO panel labels - LaTeX will handle (a), (b) via subfigure package.

    Args:
        results_df_lasso: DataFrame with LassoCV model results
        results_df_rf: DataFrame with Random Forest model results
        metric_col: Column name for metric to plot ('Correlation_Overall' or 'HitRate_Overall')
        save_path: Path to save figure (without extension)
        ylabel: Y-axis label for the metric
        left_title: Title for left panel (default: 'LassoCV')
        right_title: Title for right panel (default: 'Random Forest')
        output_format: Output format ('png', 'pdf', 'svg', 'both')
        raster_dpi: DPI for rasterized elements
        vector_dpi: DPI for pure raster formats
    """
    logging.info(f"Generating row figure for {metric_col}...")

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))

    # Styling
    colors = {'E-ind DJF(1)': '#2E8B57', 'C-ind DJF(1)': '#DC143C'}  # forest green, crimson
    markers = {'Known': 'o', 'New': 's', 'Combined': '^'}
    linestyles = {'Known': ':', 'New': '--', 'Combined': '-'}

    # Left panel: LassoCV
    _plot_metric_panel(axes[0], results_df_lasso, metric_col,
                       colors, markers, linestyles,
                       ylabel=ylabel, title=left_title)

    # Right panel: Random Forest
    _plot_metric_panel(axes[1], results_df_rf, metric_col,
                       colors, markers, linestyles,
                       ylabel='', title=right_title)

    # Shared legend below both panels (6 columns for horizontal layout)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='lower center',
               bbox_to_anchor=(0.5, 0.01),
               ncol=6,
               frameon=True,
               fontsize=12)

    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Leave space for legend at bottom

    # Save figure
    try:
        save_figure_optimized(fig, save_path, output_format, raster_dpi, vector_dpi)
        logging.info(f"Row figure saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save row figure: {e}")

    plt.close(fig)
    gc.collect()


def save_analysis_metadata(output_dir: str, args: argparse.Namespace,
                          model_config: ModelConfig, results_summary: List[Dict],
                          predictor_sets_used: List[str], graph_suffix: str):
    """Save complete analysis metadata."""
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'command_line_args': vars(args),
        'model_config': model_config.get_metadata(),
        'predictor_sets_used': predictor_sets_used,
        'target_variables': TARGET_VARIABLES,
        'results_summary': results_summary,
        'graph_generation': {
            'generated_new': not args.use_existing_graph,
            'alpha_values': args.alpha_values if not args.use_existing_graph else None,
            'graph_suffix': graph_suffix
        }
    }
    
    metadata_path = os.path.join(output_dir, 'analysis_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4, default=str)  # default=str handles numpy types
    
    logging.info(f"Saved analysis metadata to {metadata_path}")


# --- Main Execution ---
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    
    # Global variable for suffix tag
    global suffix_tag
    
    # Handle list models option
    if args.list_models:
        print_model_info()
        sys.exit(0)

    # Handle multi-panel figure generation from existing results
    if args.generate_multipanel:
        logging.info("--- Generating Multi-Panel Figure from Existing Results ---")

        # Validate required paths
        if not args.lasso_results_path or not args.rf_results_path:
            logging.error("Both --lasso-results-path and --rf-results-path are required with --generate-multipanel")
            sys.exit(1)

        if not os.path.exists(args.lasso_results_path):
            logging.error(f"LassoCV results file not found: {args.lasso_results_path}")
            sys.exit(1)

        if not os.path.exists(args.rf_results_path):
            logging.error(f"Random Forest results file not found: {args.rf_results_path}")
            sys.exit(1)

        # Load results
        try:
            results_df_lasso = pd.read_csv(args.lasso_results_path)
            results_df_rf = pd.read_csv(args.rf_results_path)
            logging.info(f"Loaded LassoCV results: {len(results_df_lasso)} rows")
            logging.info(f"Loaded Random Forest results: {len(results_df_rf)} rows")
        except Exception as e:
            logging.error(f"Error loading results: {e}")
            sys.exit(1)

        # Check if hit rate data is available
        include_hit_rate = 'HitRate_Overall' in results_df_lasso.columns and 'HitRate_Overall' in results_df_rf.columns
        if args.compute_categorical and not include_hit_rate:
            logging.warning("Hit rate columns not found in results. Generating correlation-only figure.")
            include_hit_rate = False

        # Generate output paths
        output_dir = os.path.dirname(args.lasso_results_path)

        # Generate correlation row figure
        corr_save_path = os.path.join(output_dir, 'prediction_skill_correlation')
        plot_comparison_row(
            results_df_lasso=results_df_lasso,
            results_df_rf=results_df_rf,
            metric_col='Correlation_Overall',
            save_path=corr_save_path,
            ylabel='Correlation ($r$)',
            left_title='',
            right_title='',
            output_format=args.output_format,
            raster_dpi=args.raster_dpi,
            vector_dpi=args.vector_dpi
        )
        logging.info("Saved correlation row figure")

        # Generate hit rate row figure (if categorical metrics available)
        if include_hit_rate:
            hr_save_path = os.path.join(output_dir, 'prediction_skill_hitrate')
            plot_comparison_row(
                results_df_lasso=results_df_lasso,
                results_df_rf=results_df_rf,
                metric_col='HitRate_Overall',
                save_path=hr_save_path,
                ylabel=f'Hit Rate (threshold={args.event_threshold} s.d.)',
                left_title='',
                right_title='',
                output_format=args.output_format,
                raster_dpi=args.raster_dpi,
                vector_dpi=args.vector_dpi
            )
            logging.info("Saved hit rate row figure")

        logging.info("--- Row Figure Generation Complete ---")
        sys.exit(0)

    # Create model configuration
    try:
        model_config = ModelConfig(
            model_preset=args.model,
            cv_preset=args.cv_preset,
            custom_params=args.model_params or {},
            custom_param_grid=args.param_grid or {}
        )
        logging.info(f"Created model configuration: {model_config.model_preset} with CV preset: {model_config.cv_preset}")
    except Exception as e:
        logging.error(f"Error creating model configuration: {e}")
        sys.exit(1)
    
    # Get model and CV configurations
    model_info = model_config.get_model_info()
    cv_config = model_config.get_cv_config()
    
    # Update global configuration variables from model config
    MODEL_NAME = model_info['name']
    N_CV_SPLITS = cv_config['n_splits']
    INNER_CV_SPLITS = cv_config['inner_cv_splits']
    
    # Validate alpha configuration
    if args.alpha_preset == 'custom' and not args.alpha_values:
        logging.error("Alpha values required when --alpha-preset=custom")
        sys.exit(1)
    
    # Handle alpha values based on preset
    if args.alpha_preset == 'custom':
        ALPHA_VALUES = args.alpha_values
    else:
        ALPHA_VALUES = get_alpha_preset(args.alpha_preset)
    
    logging.info(f"Alpha preset: {args.alpha_preset}, values: {ALPHA_VALUES}")
    logging.info(f"Conditional independence test: {args.cond_ind_test}")
    if args.enable_fdr:
        logging.info(f"FDR enabled with pc_alpha: {args.fdr_pc_alpha}")
    
    # Set up graph generation or use existing graphs
    if args.use_existing_graph:
        if not args.graph_suffix:
            logging.error("--graph-suffix required when --use-existing-graph is set")
            sys.exit(1)
        
        # Set paths for existing graphs
        GRAPH_SUFFIX = args.graph_suffix
        CAUSAL_GRAPH_SAVE_DIR = get_results_path(
            f'PCMCIplus/causal_prediction_comparison/{MODEL_NAME}/',
            result_type="figures"
        )
        
        # Verify graphs exist
        for set_type in PREDICTOR_SET_TYPES:
            graph_path = os.path.join(CAUSAL_GRAPH_SAVE_DIR, 
                                     f'filtered_robust_graph_{set_type.lower()}.csv')
            if not os.path.exists(graph_path):
                logging.warning(f"Expected graph not found: {graph_path}")
        
        logging.info(f"Using existing causal graphs with suffix: {GRAPH_SUFFIX}")
        skip_graph_generation = True
    else:
        # Generate output paths using utility function
        predictor_config = PredictorConfig(args.predictor_set)
        paths = cdu.generate_output_paths(BASE_SUFFIX, args.alpha_preset, args.cond_ind_test, predictor_config)
        GRAPH_SUFFIX = paths['suffix']
        skip_graph_generation = False
        logging.info("Will generate new causal graphs")
    
    # Set up output directories with proper structure
    if args.enable_fdr:
        suffix_tag = f"{GRAPH_SUFFIX}_fdr_pc_alpha{str(args.fdr_pc_alpha).replace('.','p')}"
    else:
        suffix_tag = GRAPH_SUFFIX
    
    # Create directory structure: figures/PCMCIplus/causal_prediction_comparison/{model_name}/{suffix}/
    RESULTS_SAVE_DIR = get_results_path(
        f'PCMCIplus/causal_prediction_comparison/{MODEL_NAME}/{suffix_tag}/',
        result_type="figures"
    )
    os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
    
    # Graph save directory is the same as results directory
    CAUSAL_GRAPH_SAVE_DIR = RESULTS_SAVE_DIR
    
    # Update log file with model-specific name
    log_file = os.path.join(RESULTS_SAVE_DIR, f'log_{MODEL_NAME}_{suffix_tag}.txt')
    
    # Update logging to include file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    logging.getLogger().addHandler(file_handler)
    
    logging.info("--- Starting Dynamic Causal Prediction Analysis ---")
    logging.info(f"Model: {model_config.model_preset} ({MODEL_NAME})")
    logging.info(f"CV Strategy: {model_config.cv_preset} ({N_CV_SPLITS} outer, {INNER_CV_SPLITS} inner splits)")
    logging.info(f"Predictor Set: {args.predictor_set}")
    logging.info(f"Output Directory: {RESULTS_SAVE_DIR}")

    # 1. Load Full Data for Prediction Tasks
    # Use the utility function with all required variables
    all_required_vars = set(_combined_config.get_variable_list()) | set(TARGET_VARIABLES)
    full_dataframe, var_names_full = cdu.load_full_dataset_for_prediction(
        file_path=DATA_FILE_PATH,
        variable_season_map=VARIABLE_SEASON_MAP,
        season_months_map=SEASON_MONTHS_MAP,
        required_variables=list(all_required_vars)
    )
    if full_dataframe is None or var_names_full is None:
        logging.error("Failed to load full data for prediction. Exiting.")
        sys.exit(1)

    # Extract total time steps (T) from the full dataframe
    T_dict = full_dataframe.T
    if isinstance(T_dict, dict) and 0 in T_dict:
        T = T_dict[0]
        logging.info(f"Total time steps in full dataset (T): {T}")
    else:
        logging.error(f"Unexpected format for time steps in full dataframe: {T_dict}. Expected a dict with key 0.")
        sys.exit(1)

    # Check for sufficient samples for TimeSeriesSplit
    if T <= N_CV_SPLITS:
        logging.error(f"Insufficient total samples ({T}) for {N_CV_SPLITS}-fold TimeSeriesSplit. Need T > N_CV_SPLITS.")
        sys.exit(1)

    # 2. Build Causal Graphs for each Predictor Set (or use existing)
    graph_csv_paths = {}
    
    if skip_graph_generation:
        # Use existing graphs
        logging.info("Loading paths to existing causal graphs...")
        for set_type in PREDICTOR_SET_TYPES:
            suffix = set_type.lower()
            graph_csv_path = os.path.join(CAUSAL_GRAPH_SAVE_DIR, 
                                         f'filtered_robust_graph_{suffix}.csv')
            if os.path.exists(graph_csv_path):
                graph_csv_paths[set_type] = graph_csv_path
                logging.info(f"Found existing graph for {set_type}: {graph_csv_path}")
            else:
                logging.warning(f"Graph not found for {set_type}: {graph_csv_path}")
                graph_csv_paths[set_type] = None
    else:
        # Generate new graphs
        logging.info("Generating new causal graphs...")
        
    for set_type in PREDICTOR_SET_TYPES: # ['Known', 'New', 'Combined']
        if skip_graph_generation:
            continue  # Skip generation if using existing graphs
        predictors = PREDICTOR_CONFIGS[set_type].get_variable_list()
        suffix = set_type.lower() # 'known', 'new', 'combined'

        # Define args for persistence and filtering if defaults are not desired
        # persistence_args = {'min_persistence': 0.8, ...} # Example
        # filter_args = {'min_cumulative_lag': 2, ...} # Example

        graph_csv = build_and_save_causal_graph(
            predictors=predictors,
            targets=TARGET_VARIABLES,
            save_dir=CAUSAL_GRAPH_SAVE_DIR,
            suffix=suffix,
            data_file_path=DATA_FILE_PATH,
            variable_season_map=VARIABLE_SEASON_MAP,
            season_months_map=SEASON_MONTHS_MAP,
            tau_min=TAU_MIN,
            tau_max=TAU_MAX,
            alpha_values=ALPHA_VALUES,
            cond_ind_test=args.cond_ind_test,
            enable_fdr=args.enable_fdr,
            fdr_pc_alpha=args.fdr_pc_alpha,
            max_combinations=10,  # Standard default value
            verbosity=CAUSAL_DISCOVERY_VERBOSITY
        )
        if graph_csv:
            graph_csv_paths[set_type] = graph_csv
        else:
            logging.error(f"Failed to generate causal graph for predictor set '{set_type}'. This set will be skipped in prediction.")
            # Store None to indicate failure
            graph_csv_paths[set_type] = None

    # 3. Initialize Results Storage for all prediction runs
    all_run_results = []

    # 4. Loop through Target Variables
    for target_name in TARGET_VARIABLES:
        logging.info(f"\n==== Processing Target: {target_name} ====")
        if target_name not in var_names_full:
            logging.error(f"Target variable '{target_name}' not found in full DataFrame columns: {var_names_full}. Skipping.")
            continue
        # Get target index from the GLOBAL list
        target_idx = var_names_full.index(target_name)

        # 5. Outer Loops: Iterate through Predictor Sets and Cumulative Seasons
        for predictor_set_type in PREDICTOR_SET_TYPES:
            logging.info(f"\n--- Predictor Set Type: {predictor_set_type} ---")

            # Check if graph generation failed for this set
            current_graph_csv = graph_csv_paths.get(predictor_set_type)
            if current_graph_csv is None:
                logging.warning(f"Skipping predictor set '{predictor_set_type}' because its causal graph failed to generate.")
                # Add NaN results for all seasons for this skipped set/target combo
                for max_season in CUMULATIVE_SEASONS_ORDER:
                     all_run_results.append({
                        'Target': target_name, 'PredictorSet': predictor_set_type, 'MaxLagSeason': max_season,
                        'RMSE_Overall': np.nan, 'Correlation_Overall': np.nan, 'P_value_Overall': np.nan,
                        'HitRate_Overall': np.nan, 'MissRate_Overall': np.nan, 'N_Events_Observed': 0,
                        'Event_Threshold': args.event_threshold if args.compute_categorical else np.nan,
                        'N_CV_Splits': N_CV_SPLITS, 'Num_Parents_Used': 0, 'Parents_Used': 'Graph Generation Failed'
                    })
                continue # Move to the next predictor set

            # 5a. Identify ALL Causal Parents from the *dynamically generated* graph for this target & set
            # Use the utility function with explicit var_names_full parameter
            all_target_parents_dict = cdu.parse_causal_parents_from_csv(current_graph_csv, target_name, var_names_full)

            if not all_target_parents_dict or target_idx not in all_target_parents_dict:
                logging.warning(f"Could not retrieve any causal parents dict entry for target '{target_name}' (Index: {target_idx}) from graph {current_graph_csv}. Skipping prediction runs for this set.")
                # Add NaN results for all seasons for this target/set combo
                for max_season in CUMULATIVE_SEASONS_ORDER:
                     all_run_results.append({
                        'Target': target_name, 'PredictorSet': predictor_set_type, 'MaxLagSeason': max_season,
                        'RMSE_Overall': np.nan, 'Correlation_Overall': np.nan, 'P_value_Overall': np.nan,
                        'HitRate_Overall': np.nan, 'MissRate_Overall': np.nan, 'N_Events_Observed': 0,
                        'Event_Threshold': args.event_threshold if args.compute_categorical else np.nan,
                        'N_CV_Splits': N_CV_SPLITS, 'Num_Parents_Used': 0, 'Parents_Used': 'No Parents Found in Graph'
                    })
                continue # Move to the next predictor set

            all_parents_list = all_target_parents_dict.get(target_idx, [])
            if not all_parents_list:
                logging.warning(f"No causal parents listed for target '{target_name}' in the graph {current_graph_csv}. Skipping prediction runs for this set.")
                # Add NaN results for all seasons
                for max_season in CUMULATIVE_SEASONS_ORDER:
                     all_run_results.append({
                        'Target': target_name, 'PredictorSet': predictor_set_type, 'MaxLagSeason': max_season,
                        'RMSE_Overall': np.nan, 'Correlation_Overall': np.nan, 'P_value_Overall': np.nan,
                        'HitRate_Overall': np.nan, 'MissRate_Overall': np.nan, 'N_Events_Observed': 0,
                        'Event_Threshold': args.event_threshold if args.compute_categorical else np.nan,
                        'N_CV_Splits': N_CV_SPLITS, 'Num_Parents_Used': 0, 'Parents_Used': '[]'
                    })
                continue # Move to the next predictor set

            # Get the list of signal names for the current set type from config
            # This is used ONLY to filter the parents based on season below.
            # The actual parents came from the graph built ONLY with this set's signals.
            current_signal_list_names = PREDICTOR_CONFIGS[predictor_set_type].get_variable_list()

            allowed_seasons: Set[str] = set() # Start with empty set for cumulative seasons

            for max_season in CUMULATIVE_SEASONS_ORDER:
                # Add the current season to the set of allowed seasons
                allowed_seasons.add(max_season)
                logging.info(f"  -- Max Season Included: {max_season} (Allowed: {sorted(list(allowed_seasons))}) --")

                # 6. Filter Parents based on Allowed Seasons
                # We already know the parents came from the correct graph (built on the set).
                # Now, just filter these parents based on the cumulative season rule.
                predictors_for_this_run_list = []
                
                # Define lag to season mapping for proper filtering
                # For target in DJF(1):
                # lag -6 → JJA(0) which is JJA(-1) relative to DJF(1)
                # lag -5 → SON(0) which is SON(-1) relative to DJF(1)  
                # lag -4 → DJF(0) which is DJF(0) relative to DJF(1)
                # lag -3 → MAM(1) which is MAM(0) relative to DJF(1)
                # lag -2 → JJA(1) - would be future, shouldn't occur
                # lag -1 → SON(1) - would be future, shouldn't occur
                lag_to_season_map = {
                    -6: 'JJA(-1)',
                    -5: 'SON(-1)', 
                    -4: 'DJF(0)',
                    -3: 'MAM(0)',
                    -2: 'JJA(0)',  # Shouldn't occur for DJF(1) target
                    -1: 'SON(0)'   # Shouldn't occur for DJF(1) target
                }
                
                for parent_tuple in all_parents_list:
                    parent_idx, parent_lag = parent_tuple
                    # Get parent name for logging
                    parent_name = var_names_full[parent_idx]
                    
                    # Determine the actual season based on lag, not variable name
                    actual_parent_season = lag_to_season_map.get(parent_lag)
                    
                    if actual_parent_season is None:
                        logging.warning(f"    Unexpected lag {parent_lag} for parent '{parent_name}'. Skipping.")
                        continue

                    # Check: Is the parent's actual season in the allowed cumulative set?
                    if actual_parent_season in allowed_seasons:
                        predictors_for_this_run_list.append(parent_tuple)
                        logging.debug(f"    Including '{parent_name}' at lag {parent_lag} (season {actual_parent_season})")
                    else:
                        logging.debug(f"    Excluding '{parent_name}' at lag {parent_lag} (season {actual_parent_season}) - not in allowed seasons {sorted(list(allowed_seasons))}")

                predictors_for_this_run = {target_idx: predictors_for_this_run_list}
                num_current_predictors = len(predictors_for_this_run_list)

                # Format parent list string using global names for logging/results
                parents_used_str = str([(var_names_full[p[0]], p[1]) for p in predictors_for_this_run_list])

                if num_current_predictors == 0:
                    logging.warning(f"    No predictors remaining for Target: {target_name}, Set: {predictor_set_type}, Max Season: {max_season} after seasonal filtering. Skipping this run.")
                    all_run_results.append({
                        'Target': target_name, 'PredictorSet': predictor_set_type, 'MaxLagSeason': max_season,
                        'RMSE_Overall': np.nan, 'Correlation_Overall': np.nan, 'P_value_Overall': np.nan,
                        'HitRate_Overall': np.nan, 'MissRate_Overall': np.nan, 'N_Events_Observed': 0,
                        'Event_Threshold': args.event_threshold if args.compute_categorical else np.nan,
                        'N_CV_Splits': N_CV_SPLITS, 'Num_Parents_Used': 0, 'Parents_Used': '[]'
                    })
                    continue # Skip to the next season

                logging.info(f"    Using {num_current_predictors} predictors for this run: {parents_used_str}")
                
                # Check for duplicate predictors at the season level
                season_unique_predictors = {}
                for pred_idx, lag in predictors_for_this_run_list:
                    if pred_idx not in season_unique_predictors:
                        season_unique_predictors[pred_idx] = []
                    season_unique_predictors[pred_idx].append(lag)
                
                season_duplicate_found = False
                for pred_idx, lags in season_unique_predictors.items():
                    if len(lags) > 1:
                        pred_name = var_names_full[pred_idx]
                        logging.warning(f"    Predictor '{pred_name}' appears at multiple lags {lags} for Target: {target_name}, Set: {predictor_set_type}, Season: {max_season}")
                        season_duplicate_found = True

                # 7. Time Series Cross-Validation Setup
                tscv = TimeSeriesSplit(n_splits=N_CV_SPLITS)
                all_y_true_cv = []
                all_y_pred_cv = []
                fold_metrics_cv = [] # Optional: store fold metrics if needed

                logging.info(f"    Starting {N_CV_SPLITS}-fold Time Series Cross-Validation...")

                # Use the data array and mask from the FULL dataframe load
                raw_values = full_dataframe.values
                data_to_split = raw_values[0] if isinstance(raw_values, dict) and 0 in raw_values else raw_values
                raw_mask = full_dataframe.mask
                base_mask_array = raw_mask[0] if isinstance(raw_mask, dict) and 0 in raw_mask else raw_mask

                if not isinstance(data_to_split, np.ndarray) or not isinstance(base_mask_array, np.ndarray):
                    logging.error(f"    Data or mask for splitting is not a numpy array (type: {type(data_to_split)}, {type(base_mask_array)}). Skipping run.")
                    all_run_results.append({
                        'Target': target_name, 'PredictorSet': predictor_set_type, 'MaxLagSeason': max_season,
                        'RMSE_Overall': np.nan, 'Correlation_Overall': np.nan, 'P_value_Overall': np.nan,
                        'HitRate_Overall': np.nan, 'MissRate_Overall': np.nan, 'N_Events_Observed': 0,
                        'Event_Threshold': args.event_threshold if args.compute_categorical else np.nan,
                        'N_CV_Splits': N_CV_SPLITS, 'Num_Parents_Used': num_current_predictors, 'Parents_Used': parents_used_str
                    })
                    continue  # Skip to the next season

                try:
                    for fold, (train_indices, test_indices) in enumerate(tscv.split(data_to_split)):
                        # logging.debug(f"    --- Fold {fold+1}/{N_CV_SPLITS} ---") # Debug level
                        if len(train_indices) == 0 or len(test_indices) == 0:
                            logging.warning(f"    Fold {fold+1} resulted in empty train/test indices. Skipping fold.")
                            continue
                        # logging.debug(f"    Train indices: {len(train_indices)}, Test indices: {len(test_indices)}")

                        # Prepare Train/Test DataFrames for this fold using masks from full_dataframe
                        combined_train_mask = base_mask_array.copy()
                        combined_test_mask = base_mask_array.copy()
                        combined_train_mask[test_indices, :] = True # Mask test indices in train set
                        combined_test_mask[train_indices, :] = True # Mask train indices in test set

                        # Use GLOBAL var_names_full here
                        df_train = pp.DataFrame(data_to_split, mask=combined_train_mask, var_names=var_names_full)
                        df_test = pp.DataFrame(data_to_split, mask=combined_test_mask, var_names=var_names_full)

                        # Check for sufficient unmasked data in train/test for the target
                        train_mask_fold = df_train.mask[0] if isinstance(df_train.mask, dict) else df_train.mask
                        test_mask_fold = df_test.mask[0] if isinstance(df_test.mask, dict) else df_test.mask
                        unmasked_train_samples = np.sum(~train_mask_fold[:, target_idx])
                        unmasked_test_samples = np.sum(~test_mask_fold[:, target_idx])

                        # Check against number of predictors being used in *this specific run*
                        if unmasked_train_samples <= num_current_predictors:
                            logging.warning(f"    Fold {fold+1}: Insufficient unmasked train samples ({unmasked_train_samples}) for {num_current_predictors} predictors. Skipping fold.")
                            continue
                        if unmasked_test_samples == 0:
                            logging.warning(f"    Fold {fold+1}: No unmasked test samples for target. Skipping fold.")
                            continue

                        # 8. Initialize and Fit Model on Training Data
                        try:
                            # Use either sklearn or Tigramite models based on configuration
                            if model_info['type'] == 'sklearn':
                                # Extract training and test data using sklearn approach
                                predictors_list = predictors_for_this_run.get(target_idx, [])
                                
                                # Check for duplicate predictors and warn
                                unique_predictors = {}
                                for pred_idx, lag in predictors_list:
                                    if pred_idx not in unique_predictors:
                                        unique_predictors[pred_idx] = []
                                    unique_predictors[pred_idx].append(lag)
                                
                                duplicate_found = False
                                for pred_idx, lags in unique_predictors.items():
                                    if len(lags) > 1:
                                        pred_name = var_names_full[pred_idx]
                                        logging.warning(f"    Fold {fold+1}: Sklearn model - Predictor '{pred_name}' appears at multiple lags: {lags}. Creating separate features.")
                                        duplicate_found = True
                                
                                # Extract X_train and y_train using Tigramite data
                                X_train_fold = []
                                y_train_fold = []
                                X_test_fold = []
                                y_test_fold = []
                                
                                data_values = df_train.values[0] if isinstance(df_train.values, dict) else df_train.values
                                train_mask = df_train.mask[0] if isinstance(df_train.mask, dict) else df_train.mask
                                test_data_values = df_test.values[0] if isinstance(df_test.values, dict) else df_test.values
                                test_mask = df_test.mask[0] if isinstance(df_test.mask, dict) else df_test.mask
                                
                                T_total = data_values.shape[0]
                                
                                # Build training data
                                for t in range(T_total):
                                    if train_mask[t, target_idx]:  # Skip if target is masked
                                        continue
                                    
                                    row_train = []
                                    valid_sample = True
                                    
                                    for pred_idx, lag in predictors_list:
                                        t_lag = t + lag  # lag is negative
                                        if 0 <= t_lag < T_total:
                                            row_train.append(data_values[t_lag, pred_idx])
                                        else:
                                            valid_sample = False
                                            break
                                    
                                    if valid_sample and len(row_train) == len(predictors_list):
                                        X_train_fold.append(row_train)
                                        y_train_fold.append(data_values[t, target_idx])
                                
                                # Build test data
                                for t in range(T_total):
                                    if test_mask[t, target_idx]:  # Skip if target is masked
                                        continue
                                    
                                    row_test = []
                                    valid_sample = True
                                    
                                    for pred_idx, lag in predictors_list:
                                        t_lag = t + lag  # lag is negative
                                        if 0 <= t_lag < T_total:
                                            row_test.append(test_data_values[t_lag, pred_idx])
                                        else:
                                            valid_sample = False
                                            break
                                    
                                    if valid_sample and len(row_test) == len(predictors_list):
                                        X_test_fold.append(row_test)
                                        y_test_fold.append(test_data_values[t, target_idx])
                                
                                # Convert to numpy arrays
                                if not X_train_fold or not y_train_fold:
                                    logging.warning(f"    Fold {fold+1}: No valid training samples after extraction. Skipping fold.")
                                    continue
                                if not X_test_fold or not y_test_fold:
                                    logging.warning(f"    Fold {fold+1}: No valid test samples after extraction. Skipping fold.")
                                    continue
                                
                                X_train_array = np.array(X_train_fold)
                                y_train_array = np.array(y_train_fold)
                                X_test_array = np.array(X_test_fold)
                                y_test_array = np.array(y_test_fold)
                                
                                # Log data shapes for debugging
                                logging.debug(f"    Fold {fold+1}: X_train shape: {X_train_array.shape}, y_train shape: {y_train_array.shape}")
                                logging.debug(f"    Fold {fold+1}: X_test shape: {X_test_array.shape}, y_test shape: {y_test_array.shape}")
                                
                                # Log success with multi-lag predictors
                                if duplicate_found and X_train_array.shape[0] > 0:
                                    logging.info(f"    Fold {fold+1}: Successfully extracted {X_train_array.shape[0]} training samples with multi-lag predictors")
                                
                                # Check for potential issues
                                if X_train_array.shape[1] != len(predictors_list):
                                    logging.error(f"    Fold {fold+1}: Feature count mismatch. Expected {len(predictors_list)}, got {X_train_array.shape[1]}")
                                    continue
                                
                                # Create and fit model
                                prediction_model = ModelFactory.create_model_with_tuning(
                                    model_info, 
                                    inner_cv_splits=INNER_CV_SPLITS,
                                    scoring='neg_root_mean_squared_error'
                                )
                                
                                # Fit the model with error handling for multicollinearity
                                try:
                                    prediction_model.fit(X_train_array, y_train_array)
                                except Exception as sklearn_fit_err:
                                    logging.error(f"    Sklearn model fit failed for fold {fold+1}: {sklearn_fit_err}")
                                    if duplicate_found:
                                        logging.error(f"    This may be due to duplicate predictors at different lags causing multicollinearity.")
                                    continue
                                
                                # Make predictions
                                try:
                                    y_pred_fold = prediction_model.predict(X_test_array)
                                    y_true_fold = y_test_array
                                except Exception as sklearn_pred_err:
                                    logging.error(f"    Sklearn model predict failed for fold {fold+1}: {sklearn_pred_err}")
                                    continue
                                
                            else:
                                # Use Tigramite Models for backward compatibility
                                from sklearn.linear_model import LassoCV
                                tigramite_model = LassoCV(cv=5, random_state=42)  # Default for Tigramite compatibility
                                model_fitter = Models(dataframe=df_train, model=tigramite_model, mask_type='y', verbosity=VERBOSITY)
                                
                                # Check for duplicate predictors before fitting
                                predictors_list = predictors_for_this_run.get(target_idx, [])
                                unique_predictors = {}
                                for pred_idx, lag in predictors_list:
                                    if pred_idx not in unique_predictors:
                                        unique_predictors[pred_idx] = []
                                    unique_predictors[pred_idx].append(lag)
                                
                                duplicate_found = False
                                for pred_idx, lags in unique_predictors.items():
                                    if len(lags) > 1:
                                        pred_name = var_names_full[pred_idx]
                                        logging.warning(f"    Fold {fold+1}: Predictor '{pred_name}' appears at multiple lags: {lags}. This may cause fitting issues.")
                                        duplicate_found = True
                                
                                try:
                                    fitted_model_info = model_fitter.fit_full_model(
                                        all_parents=predictors_for_this_run,
                                        selected_variables=[target_idx],
                                        tau_max=TAU_MAX,
                                        cut_off='max_lag_or_tau_max'
                                    )
                                except Exception as tigramite_fit_err:
                                    logging.error(f"    Tigramite fit_full_model failed for fold {fold+1}: {tigramite_fit_err}")
                                    if duplicate_found:
                                        logging.error(f"    This may be due to duplicate predictors at different lags.")
                                    continue
                                
                                # Predict using Tigramite
                                try:
                                    y_pred_fold_list = model_fitter.predict_full_model(
                                        new_data=df_test,
                                        cut_off='max_lag_or_tau_max'
                                    )
                                except Exception as tigramite_pred_err:
                                    logging.error(f"    Tigramite predict_full_model failed for fold {fold+1}: {tigramite_pred_err}")
                                    continue
                                    
                                if not y_pred_fold_list:
                                    logging.error(f"    Prediction failed for fold {fold+1}, returned empty list.")
                                    continue
                                y_pred_fold = y_pred_fold_list[0]
                                
                                # Retrieve true values from the fitter's stored test array
                                if target_idx not in model_fitter.stored_test_array:
                                    logging.error(f"    Could not find stored test array for target index {target_idx} in fold {fold+1}.")
                                    continue
                                test_array_constructed = model_fitter.stored_test_array[target_idx]
                                if len(test_array_constructed) < 2:
                                    logging.error(f"    Stored test array for target index {target_idx} in fold {fold+1} has unexpected structure.")
                                    continue
                                y_true_fold = test_array_constructed[1]
                                
                        except Exception as fit_err:
                            logging.error(f"    Error during model fitting/prediction for fold {fold+1}: {fit_err}", exc_info=False)
                            continue # Skip to next fold

                        # 9. Validate prediction results
                        try:
                            if len(y_true_fold) != len(y_pred_fold):
                                logging.warning(f"    Fold {fold+1}: Length mismatch between true ({len(y_true_fold)}) and predicted ({len(y_pred_fold)}). Skipping fold.")
                                continue
                            if len(y_true_fold) == 0:
                                logging.warning(f"    Fold {fold+1}: Both true and predicted arrays are empty after masking/prediction. Skipping fold.")
                                continue

                            # Append valid fold results
                            all_y_true_cv.append(y_true_fold)
                            all_y_pred_cv.append(y_pred_fold)

                        except Exception as val_err:
                            logging.error(f"    Error during validation for fold {fold+1}: {val_err}", exc_info=False)
                            continue # Skip to next fold
                    # --- End of CV fold loop ---

                except ValueError as ve: # Catch specific errors like from tscv.split
                    logging.error(f"    ValueError during TimeSeriesSplit setup or iteration: {ve}. Skipping run for this season.")
                    all_run_results.append({
                        'Target': target_name, 'PredictorSet': predictor_set_type, 'MaxLagSeason': max_season,
                        'RMSE_Overall': np.nan, 'Correlation_Overall': np.nan, 'P_value_Overall': np.nan,
                        'HitRate_Overall': np.nan, 'MissRate_Overall': np.nan, 'N_Events_Observed': 0,
                        'Event_Threshold': args.event_threshold if args.compute_categorical else np.nan,
                        'N_CV_Splits': N_CV_SPLITS, 'Num_Parents_Used': num_current_predictors, 'Parents_Used': parents_used_str
                    })
                    continue # Skip to the next season
                except Exception as cv_err: # Catch any other unexpected errors during the loop
                    logging.error(f"    Unexpected error during cross-validation loop: {cv_err}. Skipping run for this season.", exc_info=True)
                    all_run_results.append({
                        'Target': target_name, 'PredictorSet': predictor_set_type, 'MaxLagSeason': max_season,
                        'RMSE_Overall': np.nan, 'Correlation_Overall': np.nan, 'P_value_Overall': np.nan,
                        'HitRate_Overall': np.nan, 'MissRate_Overall': np.nan, 'N_Events_Observed': 0,
                        'Event_Threshold': args.event_threshold if args.compute_categorical else np.nan,
                        'N_CV_Splits': N_CV_SPLITS, 'Num_Parents_Used': num_current_predictors, 'Parents_Used': parents_used_str
                    })
                    continue # Skip to the next season


                # 10. Aggregate and Evaluate Overall Performance for this Run (Target, Set, Season)
                logging.debug(f"    CV fold results: {len(all_y_true_cv)} successful folds out of {N_CV_SPLITS} total")

                # Initialize categorical metrics to NaN (will be computed if args.compute_categorical)
                overall_hit_rate = np.nan
                overall_miss_rate = np.nan
                n_events = 0

                if not all_y_true_cv or not all_y_pred_cv:
                    logging.warning(f"    No successful CV predictions collected for Target: {target_name}, Set: {predictor_set_type}, Max Season: {max_season}. Storing NaN.")
                    if season_duplicate_found:
                        logging.warning(f"    This may be due to duplicate predictors at different lags causing all folds to fail.")
                    overall_rmse, overall_corr, overall_pval = np.nan, np.nan, np.nan
                    y_true_agg, y_pred_agg = None, None # Ensure these are None if aggregation fails
                else:
                    try:
                        # Filter out potential empty arrays from skipped folds before concatenating
                        valid_y_true = [arr for arr in all_y_true_cv if isinstance(arr, np.ndarray) and arr.size > 0]
                        valid_y_pred = [arr for arr in all_y_pred_cv if isinstance(arr, np.ndarray) and arr.size > 0]

                        if not valid_y_true or not valid_y_pred:
                             logging.warning(f"    No non-empty fold results to aggregate for {target_name}, Set: {predictor_set_type}, Max Season: {max_season}. Storing NaN.")
                             overall_rmse, overall_corr, overall_pval = np.nan, np.nan, np.nan
                             y_true_agg, y_pred_agg = None, None
                        else:
                            y_true_agg = np.concatenate(valid_y_true)
                            y_pred_agg = np.concatenate(valid_y_pred)
                            overall_rmse, overall_corr, overall_pval = calculate_metrics(y_true_agg, y_pred_agg)
                            logging.info(f"    Overall Performance (Set: {predictor_set_type}, Max Season: {max_season}): RMSE={overall_rmse:.4f}, Corr={overall_corr:.4f} (p={overall_pval:.3g})")

                            # Calculate categorical metrics if enabled
                            if args.compute_categorical:
                                cat_metrics = calculate_categorical_metrics(
                                    y_true_agg, y_pred_agg,
                                    threshold=args.event_threshold
                                )
                                overall_hit_rate = cat_metrics['hit_rate']
                                overall_miss_rate = cat_metrics['miss_rate']
                                n_events = cat_metrics['n_events_observed']
                                logging.info(f"    Categorical (threshold={args.event_threshold}): "
                                             f"HitRate={overall_hit_rate:.3f}, Events={n_events}")

                            # Save aggregated predictions (only if aggregation succeeded)
                            preds_df = pd.DataFrame({'y_true': y_true_agg, 'y_pred': y_pred_agg})
                            preds_save_path = os.path.join(RESULTS_SAVE_DIR, PREDICTIONS_FILENAME_TEMPLATE.format(
                                target=target_name, predictor_set=predictor_set_type, max_season=max_season,
                                model_name=MODEL_NAME, graph_suffix=suffix_tag))
                            try:
                                preds_df.to_csv(preds_save_path, index=False, float_format='%.4f')
                                logging.info(f"    Saved aggregated predictions to {preds_save_path}")
                            except Exception as save_err:
                                logging.error(f"    Failed to save aggregated predictions: {save_err}")


                            # Generate Scatter Plot (if metrics are valid)
                            if not np.isnan(overall_corr) and not np.isnan(overall_rmse):
                                generate_scatter_plot(
                                    y_true_agg=y_true_agg,
                                    y_pred_agg=y_pred_agg,
                                    target_name=target_name,
                                    predictor_set_type=predictor_set_type,
                                    max_season=max_season,
                                    model_name=MODEL_NAME,
                                    graph_suffix=suffix_tag, # Use the configured suffix
                                    overall_corr=overall_corr,
                                    overall_rmse=overall_rmse,
                                    results_save_dir=RESULTS_SAVE_DIR, # Pass correct dir
                                    output_format=args.output_format,
                                    raster_dpi=args.raster_dpi,
                                    vector_dpi=args.vector_dpi
                                )

                    except ValueError as concat_err:
                        logging.error(f"    Error concatenating fold results: {concat_err}. Storing NaN.")
                        overall_rmse, overall_corr, overall_pval = np.nan, np.nan, np.nan
                        y_true_agg, y_pred_agg = None, None
                    except Exception as agg_err:
                        logging.error(f"    Error during aggregation/metric calculation: {agg_err}. Storing NaN.")
                        overall_rmse, overall_corr, overall_pval = np.nan, np.nan, np.nan
                        y_true_agg, y_pred_agg = None, None

                # Store results for this specific run configuration (even if metrics are NaN)
                all_run_results.append({
                    'Target': target_name,
                    'PredictorSet': predictor_set_type,
                    'MaxLagSeason': max_season, # Store the season name
                    'RMSE_Overall': overall_rmse,
                    'Correlation_Overall': overall_corr,
                    'P_value_Overall': overall_pval,
                    'HitRate_Overall': overall_hit_rate,
                    'MissRate_Overall': overall_miss_rate,
                    'N_Events_Observed': n_events,
                    'Event_Threshold': args.event_threshold if args.compute_categorical else np.nan,
                    'N_CV_Splits': N_CV_SPLITS,
                    'Num_Parents_Used': num_current_predictors,
                    'Parents_Used': parents_used_str # Store the string representation
                })
            # --- End of cumulative season loop ---
        # --- End of predictor set loop ---
    # --- End of target variable loop ---

    # 11. Save Overall Comparison Results
    if all_run_results:
        summary_df = pd.DataFrame(all_run_results)
        # 1) Convert to ordered categorical
        summary_df['MaxLagSeason'] = pd.Categorical(
            summary_df['MaxLagSeason'],
            categories=CUMULATIVE_SEASONS_ORDER,
            ordered=True
        )
        # 2) Sort by the three column names
        summary_df = summary_df.sort_values(
            by=['Target','PredictorSet','MaxLagSeason']
        )
        comparison_filename = COMPARISON_RESULTS_FILENAME.format(
            model_name=MODEL_NAME, 
            graph_suffix=suffix_tag
        )
        summary_save_path = os.path.join(RESULTS_SAVE_DIR, comparison_filename)

        try:
            # save CSV
            summary_df.to_csv(summary_save_path, index=False, float_format='%.4f')
            logging.info(f"\nOverall prediction comparison results saved to: {summary_save_path}")

            # and then plot
            plot_filename = COMPARISON_PLOT_FILENAME.format(
                model_name=MODEL_NAME,
                graph_suffix=suffix_tag
            )
            # Remove .png extension for save_figure_optimized
            plot_filename_base = plot_filename.replace('.png', '')
            plot_save_path = os.path.join(RESULTS_SAVE_DIR, plot_filename_base)
            plot_comparison(summary_df, plot_save_path, 
                          output_format=args.output_format,
                          raster_dpi=args.raster_dpi,
                          vector_dpi=args.vector_dpi,
                          show_legend=args.show_legend)

        except Exception as e:
            logging.error(f"Failed to save summary comparison results or plot: {e}")

    else:
        logging.warning("No results generated across all targets, sets, and seasons to save.")
        all_run_results = []  # Set to empty for metadata saving
    
    # 12. Save comprehensive metadata
    try:
        save_analysis_metadata(
            output_dir=RESULTS_SAVE_DIR,
            args=args,
            model_config=model_config,
            results_summary=all_run_results,
            predictor_sets_used=PREDICTOR_SET_TYPES,
            graph_suffix=suffix_tag
        )
    except Exception as e:
        logging.error(f"Failed to save analysis metadata: {e}")
    
    logging.info("--- Dynamic Causal Prediction Analysis Complete ---")