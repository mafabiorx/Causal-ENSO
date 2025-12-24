"""
Sequential Multivariate Regression Analysis: Climate Field Response to ENSO Precursors

Core Functionality:
- Apply pathway-specific regression to quantify spatial climate field responses to ENSO precursors
- Use OLS regression for individual EP and CP pathway analysis
- Apply LASSO regression for combined pathway analysis with variable selection
- Remove E/C index confounders to isolate pure precursor effects

Key Features:
- Sequential temporal analysis from JJA(-1) through DJF(+1) lead times
- Spatially-derived LASSO alpha parameter optimization for regularization
- Comprehensive significance testing with FDR correction for spatial fields
- Generate regression coefficient maps for physical interpretation

Output: Spatial regression coefficient fields and significance maps for pathway interpretation

Usage examples:
  # Default: process U_10m and V_10m target variables
  python script.py

  # Process all available target variables
  python script.py --target-variables SST prec surf_pres vp_200 low_clouds U_200 V_200 \\
                                      U_10m V_10m sf_200 U_850 V_850 RWS_200 WAFx_200 \\
                                      WAFy_200 vertical_streamfunction RWS_multi

  # Process specific target variables
  python script.py --target-variables SST prec low_clouds

  # Dry run to see what jobs would be submitted
  python script.py --target-variables SST prec --dry-run

  # Run with specific worker count
  python script.py --target-variables SST --max-workers 4
"""

import numpy as np
import xarray as xr
import pandas as pd
import warnings
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LassoCV
import os
import sys
from pathlib import Path
from dask.diagnostics import ProgressBar
from tqdm import tqdm
import logging
import functools # For potentially using partial in apply_ufunc if needed
import traceback

# --- Imports for parallel execution ---
import concurrent.futures
import argparse
import psutil
import multiprocessing # For get_context if needed

# --- Add project root to Python path ---
sys.path.append(str(Path(__file__).parent.parent))

# --- Import utility functions ---
try:
    from utils.TBI_functions import (
    remove_signal,
    load_era_field,
    standardize_dim
    )
    from utils.paths import get_data_path, get_results_path
except ImportError:
    print("Could not import from utils, trying utils...")
    try:
        from utils.TBI_functions import (
        remove_signal,
        load_era_field,
        standardize_dim
        )
        from utils.paths import get_data_path, get_results_path
    except ImportError as e_fallback:
        print(f"Fallback import failed: {e_fallback}")
        print("Please ensure TBI_functions.py and paths.py are in the Python path.")
        sys.exit(1)


# --- Processing Configuration Dataclass ---
# Encapsulates all worker configuration to reduce global state complexity
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class ProcessingConfig:
    """
    Configuration container for multiprocessing workers.

    This dataclass encapsulates all the configuration parameters that were
    previously scattered as global variables. It provides:
    - Type hints for better IDE support and documentation
    - Immutability for thread safety (when frozen=True is added)
    - Clear documentation of what configuration each worker needs

    The global variables below are maintained for backward compatibility
    but new code should use the config object directly.
    """
    # Data paths
    data_dir_gridded: str = ""
    data_dir_ts: str = ""

    # Time period configuration
    time_period_start: Optional[Any] = None  # pd.Timestamp
    time_period_end: Optional[Any] = None    # pd.Timestamp

    # LASSO configuration
    max_iter_lasso: int = 10000
    cv_folds_lasso: int = 5

    # Season/Lag mapping configuration
    seasons: List[str] = field(default_factory=list)
    regression_sets: List[Any] = field(default_factory=list)
    pacific_bbox_alpha: Dict[str, Any] = field(default_factory=dict)
    lag_to_season: Dict[int, str] = field(default_factory=dict)
    season_to_month: Dict[str, int] = field(default_factory=dict)
    target_season_suffix_to_lag: Dict[str, int] = field(default_factory=dict)
    predictor_season_lag_map: Dict[str, int] = field(default_factory=dict)

    # Predictor/confounder paths (for worker reconstruction)
    predictor_paths_structure: Dict[str, Any] = field(default_factory=dict)
    target_variable_defs: Dict[str, Any] = field(default_factory=dict)
    confounder_paths: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for worker initialization."""
        return {
            'DATA_DIR_GRIDDED': self.data_dir_gridded,
            'DATA_DIR_TS': self.data_dir_ts,
            'TIME_PERIOD_START': self.time_period_start,
            'TIME_PERIOD_END': self.time_period_end,
            'MAX_ITER_LASSO': self.max_iter_lasso,
            'CV_FOLDS_LASSO': self.cv_folds_lasso,
            'TARGET_SEASONS_TO_MAP': self.seasons,
            'REGRESSION_SETS': self.regression_sets,
            'PACIFIC_BBOX_ALPHA': self.pacific_bbox_alpha,
            'LAG_TO_SEASON': self.lag_to_season,
            'season_to_month': self.season_to_month,
            'target_season_suffix_to_lag': self.target_season_suffix_to_lag,
            'predictor_season_lag_map': self.predictor_season_lag_map,
            'predictor_paths_structure': self.predictor_paths_structure,
            'target_variable_defs': self.target_variable_defs,
            'confounder_paths': self.confounder_paths,
        }


# --- Global variables for worker processes ---
# These are maintained for backward compatibility with existing helper functions.
# New code should prefer passing config explicitly where possible.
# These will be set by load_data_in_worker for each worker process.
CONFOUNDERS = {}
PREDICTORS = {}
PREDICTOR_LAG_MAP = {}

# Configuration globals that will be set in worker from config_dict
SEASONS = []
REGRESSION_SETS = []
DATA_DIR_GRIDDED = ""
DATA_DIR_TS = ""

# Valid target variables for CLI selection
VALID_TARGET_VARIABLES = [
    'SST', 'prec', 'surf_pres', 'vp_200', 'low_clouds',
    'U_200', 'V_200', 'U_10m', 'V_10m', 'sf_200',
    'U_850', 'V_850', 'RWS_200', 'WAFx_200', 'WAFy_200',
    'vertical_streamfunction', 'RWS_multi'
]
TIME_PERIOD_START = None
TIME_PERIOD_END = None
MAX_ITER_LASSO = 10000
CV_FOLDS_LASSO = 5
PACIFIC_BBOX_ALPHA = {}
LAG_TO_SEASON = {}
season_to_month = {}
target_season_suffix_to_lag = {}
predictor_season_lag_map = {}
# predictor_paths and target_variable_defs are reconstructed/used locally in load_data_in_worker

# Worker-local config reference (set during initialization)
_worker_config: Optional[ProcessingConfig] = None

# --- Environment Setter for Worker Processes ---
def _set_single_thread_env():
    """Sets environment variables to limit threads used by numerical libraries."""
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['NUMEXPR_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

# --- Worker-specific Logging Setup ---
def setup_worker_logging(worker_pid, debug_mode, output_dir_for_logs):
    """Configures logging for a single worker process."""
    # Remove existing handlers from root logger to avoid duplicate messages or conflicts
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        handler.close()

    log_dir = os.path.join(output_dir_for_logs, "worker_logs")
    os.makedirs(log_dir, exist_ok=True)
    worker_log_file = os.path.join(log_dir, f'worker_{worker_pid}.log')
    
    logging.basicConfig(
        level=logging.DEBUG if debug_mode else logging.INFO,
        format=f'%(asctime)s [Worker-{worker_pid}] [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(worker_log_file, mode='w'),
            # logging.StreamHandler() # Uncomment for debugging to see worker logs in console
        ],
        force=True
    )
    # Suppress specific warnings again for the worker
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning, module='dask')
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom <= 0 for slice')
    logging.info(f"Worker {worker_pid} logging initialized.")


# --- Helper Functions ---

def load_predictors(predictor_paths_dict, base_time_coord):
    """Loads predictor time series from paths, aligns to common time coordinate."""
    predictors_da = {}
    for pred_name, file_path in predictor_paths_dict.items():
        if not os.path.exists(file_path):
            logging.error(f"Predictor file does not exist: {file_path} for predictor {pred_name}")
            raise FileNotFoundError(f"Missing predictor file: {file_path}")
        try:
            with xr.open_dataset(file_path) as ds:
                data_vars = [v for v in ds.data_vars if v not in ds.coords]
                if not data_vars:
                    raise ValueError(f"No suitable data variable found in {file_path}")
                var_name = data_vars[0]
                predictor_da = ds[var_name].load()

                if 'time' not in predictor_da.coords:
                    raise ValueError(f"Predictor {pred_name} missing 'time' coordinate.")
                if not pd.api.types.is_datetime64_any_dtype(predictor_da['time']):
                    try:
                        predictor_da['time'] = pd.to_datetime(predictor_da['time'].values)
                    except Exception as time_e:
                        raise ValueError(f"Could not convert time for {pred_name}: {time_e}")
                
                predictors_da[pred_name] = predictor_da.reindex(time=base_time_coord, method=None)
        except Exception as e:
            logging.error(f"Error loading predictor {pred_name} from {file_path}: {e}")
            raise
    return predictors_da

def align_data_for_regression(target_y_spatial, target_season_str,
    predictor_dict_set, predictor_lag_map_local, # Use local copy of PREDICTOR_LAG_MAP
    confounder_indices_dict, confounder_lag):
    """
    Aligns target, predictors, and MULTIPLE confounders based on lags.
    Uses inner join logic to find common time steps across ALL inputs.
    Relies on global: target_season_suffix_to_lag, season_to_month
    """
    try:
        target_lag = target_season_suffix_to_lag[target_season_str]
        base_target_season = target_season_str.split('_')[0]
        target_month = season_to_month[base_target_season]

        y_target_season = target_y_spatial.where(target_y_spatial.time.dt.month == target_month, drop=True)
        if y_target_season.time.size == 0:
            logging.warning(f"No target data for {target_season_str} (Month: {target_month})")
            return None, None, None, None
        base_target_time = y_target_season.time

        data_arrays_to_align = [y_target_season]
        aligned_predictors_dict = {}
        aligned_confounders_dict = {}

        for pred_name, predictor_da in predictor_dict_set.items():
            original_lag = predictor_lag_map_local.get(pred_name) # Use passed local map
            if original_lag is None:
                logging.warning(f"Predictor {pred_name} missing from lag map, skipping.")
                continue
            shift_seasons = original_lag - target_lag
            shifted_predictor = predictor_da.shift(time=shift_seasons) if shift_seasons != 0 else predictor_da
            aligned_pred = shifted_predictor.reindex(time=base_target_time, method=None)
            if aligned_pred.notnull().any():
                aligned_predictors_dict[pred_name] = aligned_pred
                data_arrays_to_align.append(aligned_pred)
            else:
                logging.warning(f"Predictor {pred_name} skipped for {target_season_str}: no overlap after shift {shift_seasons}.")

        if not aligned_predictors_dict:
            logging.warning(f"No predictors successfully aligned for target {target_season_str}.")
            return None, None, None, None

        confounder_shift = confounder_lag - target_lag
        for conf_name, confounder_index in confounder_indices_dict.items():
            shifted_confounder = confounder_index.shift(time=confounder_shift) if confounder_shift != 0 else confounder_index
            temp_aligned_confounder = shifted_confounder.reindex(time=base_target_time, method=None)
            if temp_aligned_confounder.isnull().all():
                logging.error(f"Confounder '{conf_name}' alignment resulted in all NaNs for target {target_season_str} using lag {confounder_lag}. Cannot proceed.")
                return None, None, None, None
            aligned_confounders_dict[conf_name] = temp_aligned_confounder
            data_arrays_to_align.append(temp_aligned_confounder)

        aligned_data = xr.align(*data_arrays_to_align, join='inner')
        common_time = aligned_data[0].time

        if common_time.size == 0:
            logging.error(f"No common time steps found after inner join alignment for {target_season_str}.")
            return None, None, None, None
        logging.info(f"Aligned data for {target_season_str} has {common_time.size} time steps.")

        y_aligned = aligned_data[0]
        num_predictors = len(aligned_predictors_dict)
        X_aligned_dict = {name: aligned_data[i+1] for i, name in enumerate(aligned_predictors_dict.keys())}
        confounders_aligned_dict = {name: aligned_data[i + 1 + num_predictors] for i, name in enumerate(aligned_confounders_dict.keys())}

        return y_aligned, X_aligned_dict, confounders_aligned_dict, common_time

    except Exception as e:
        logging.error(f"Error during data alignment for target {target_season_str}: {e}", exc_info=True)
        return None, None, None, None

def get_predictors_for_set(lag_range_str, pathway_name):
    """
    Selects the appropriate predictor DataArrays based on the lag range and pathway context.
    Relies on global: PREDICTORS, PREDICTOR_LAG_MAP
    """
    clean_lag_range = lag_range_str.replace('_COMBINED', '')
    if '-' in clean_lag_range:
        max_lag, min_lag = map(int, clean_lag_range.replace('lag', '').split('-'))
        included_lags = list(range(max_lag, min_lag - 1, -1))
    else:
        included_lags = [int(clean_lag_range.replace('lag', ''))]

    predictors_set_dict = {}
    if pathway_name == 'COMBINED':
        for pathway in ['EP', 'CP']:
            if pathway in PREDICTORS:
                for name, predictor_da in PREDICTORS[pathway].items():
                    original_lag = PREDICTOR_LAG_MAP.get(name)
                    if original_lag in included_lags:
                        if name not in predictors_set_dict:
                            predictors_set_dict[name] = predictor_da
    elif pathway_name in PREDICTORS:
        for name, predictor_da in PREDICTORS[pathway_name].items():
            original_lag = PREDICTOR_LAG_MAP.get(name)
            if original_lag in included_lags:
                predictors_set_dict[name] = predictor_da
    else:
        logging.error(f"Invalid pathway name '{pathway_name}' provided to get_predictors_for_set.")
        return {}

    if not predictors_set_dict:
        logging.warning(f"No predictors found for pathway '{pathway_name}' matching lags {included_lags} (from {lag_range_str}).")
    logging.info(f"Selected {len(predictors_set_dict)} predictors for pathway '{pathway_name}': {sorted(list(predictors_set_dict.keys()))}")
    return predictors_set_dict

def align_data_for_season(target_var, season, lag_range_str, pathway_name):
    """
    Prepares aligned data for a specific target season, lag range, and pathway context.
    Relies on global: PREDICTOR_LAG_MAP, CONFOUNDERS
    """
    predictors_set_dict = get_predictors_for_set(lag_range_str, pathway_name)
    if not predictors_set_dict:
        logging.error(f"No predictors selected for {pathway_name}, {lag_range_str}. Cannot proceed for {season}.")
        return None, None, None

    clean_lag_range = lag_range_str.replace('_COMBINED', '')
    if '-' in clean_lag_range:
        _, min_lag = map(int, clean_lag_range.replace('lag', '').split('-'))
    else:
        min_lag = int(clean_lag_range.replace('lag', ''))
    confounder_lag = min_lag

    y_aligned, X_aligned_dict, confounders_aligned_dict, _ = align_data_for_regression(
        target_var, season, predictors_set_dict, PREDICTOR_LAG_MAP, # Pass global PREDICTOR_LAG_MAP
        CONFOUNDERS, confounder_lag
    )
    return y_aligned, X_aligned_dict, confounders_aligned_dict

def create_predictor_array(X_aligned_dict):
    """
    Convert dictionary of aligned predictor DataArrays into a single DataArray.
    """
    if not X_aligned_dict:
        logging.warning("create_predictor_array received an empty dictionary.")
        return None
    predictor_list = []
    predictor_names = sorted(list(X_aligned_dict.keys()))
    for pred_name in predictor_names:
        predictor_list.append(X_aligned_dict[pred_name])
    try:
        combined_predictors = xr.concat(predictor_list, dim='predictor')
        combined_predictors = combined_predictors.assign_coords(predictor=predictor_names)
        return combined_predictors
    except Exception as e:
        logging.error(f"Error concatenating predictors: {e}")
        return None

def clean_aligned_data(y_data, confounder_data_dict, other_datadict=None):
    """
    Removes time steps where ANY of the provided confounders have NaN values.
    """
    if not confounder_data_dict:
        logging.warning("No confounder data provided to clean_aligned_data. Returning original data.")
        return y_data, confounder_data_dict, other_datadict

    confounder_list = list(confounder_data_dict.values())
    try:
        stacked_confounders = xr.concat(confounder_list, dim='_temp_conf_check')
        valid_mask = stacked_confounders.notnull().all(dim='_temp_conf_check')
    except Exception as e:
        logging.error(f"Error creating valid mask from confounders: {e}")
        return None, None, None

    if valid_mask.all():
        return y_data, confounder_data_dict, other_datadict

    invalid_count = (~valid_mask).sum().item()
    valid_count = valid_mask.sum().item()
    total_points = y_data.time.size
    logging.info(f"Cleaning data: Found {invalid_count} time points with NaNs in at least one confounder (out of {total_points}). Keeping {valid_count} valid points.")

    cleaned_y = y_data.where(valid_mask, drop=True)
    cleaned_confounders_dict = {name: da.where(valid_mask, drop=True) for name, da in confounder_data_dict.items()}
    
    cleaned_other = None
    if other_datadict:
        cleaned_other = {}
        for k, v in other_datadict.items():
            if 'time' in v.dims:
                cleaned_other[k] = v.where(valid_mask, drop=True)
            else:
                cleaned_other[k] = v
    
    if cleaned_y.time.size == 0:
        logging.warning("Cleaning process removed all time points. Returning None.")
        return None, None, None
    return cleaned_y, cleaned_confounders_dict, cleaned_other

def ols_regress_1D(y, X):
    """
    Performs Ordinary Least Squares (OLS) regression for a single grid point's time series.
    """
    n_predictors = X.shape[1]
    mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
    n_samples = np.sum(mask)

    if n_samples < n_predictors + 2:
        return (np.full(n_predictors, np.nan, dtype=np.float32),
                np.full(n_predictors, np.nan, dtype=np.float32),
                np.float32(np.nan))
    y_clean = y[mask]
    X_clean = X[mask, :]
    X_const = sm.add_constant(X_clean, prepend=True, has_constant='skip')

    try:
        model = sm.OLS(y_clean, X_const).fit()
        coef_start_index = 1 if X_const.shape[1] == X_clean.shape[1] + 1 else 0
        coefs = model.params[coef_start_index:]
        pvals = model.pvalues[coef_start_index:]
        r2    = model.rsquared

        final_coefs = np.full(n_predictors, np.nan, dtype=np.float32)
        final_pvals = np.full(n_predictors, np.nan, dtype=np.float32)
        num_fitted_coefs = len(coefs)

        if num_fitted_coefs == n_predictors:
            final_coefs = coefs
            final_pvals = pvals
        elif num_fitted_coefs < n_predictors:
            logging.debug(f"OLS fit returned fewer coefficients ({num_fitted_coefs}) than predictors ({n_predictors}). Padding with NaN.")
            final_coefs[:num_fitted_coefs] = coefs
            final_pvals[:num_fitted_coefs] = pvals
        else: # num_fitted_coefs > n_predictors
            logging.warning(f"OLS returned more coefficients ({num_fitted_coefs}) than predictors ({n_predictors}). Truncating.")
            final_coefs = coefs[:n_predictors]
            final_pvals = pvals[:n_predictors]
        
        coefs = final_coefs
        pvals = final_pvals

    except Exception as e:
        logging.debug(f"OLS fit failed: {e}")
        coefs = np.full(n_predictors, np.nan, dtype=np.float32)
        pvals = np.full(n_predictors, np.nan, dtype=np.float32)
        r2    = np.float32(np.nan)
    return (coefs.astype(np.float32), pvals.astype(np.float32), np.float32(r2))

def lasso_cv_regress_1D(y, X):
    """
    Performs LASSO regression using LassoCV. Relies on global CV_FOLDS_LASSO, MAX_ITER_LASSO.
    """
    n_predictors = X.shape[1]
    mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
    n_samples = np.sum(mask)
    min_samples_needed = max(CV_FOLDS_LASSO + 2, n_predictors + 1, 10)

    if n_samples < min_samples_needed:
        logging.debug(f"Skipping LassoCV: Insufficient samples ({n_samples} < {min_samples_needed})")
        return (np.full(n_predictors, np.nan, dtype=np.float32),
                np.float32(np.nan), np.float32(np.nan))
    y_clean = y[mask]
    X_clean = X[mask, :]

    if np.std(y_clean) < 1e-10:
        logging.debug("Skipping LassoCV: Target variable has near-zero variance.")
        return (np.zeros(n_predictors, dtype=np.float32),
                np.float32(np.nan), np.float32(0.0))
    
    model = LassoCV(cv=CV_FOLDS_LASSO, max_iter=MAX_ITER_LASSO, random_state=42,
                    tol=1e-3, selection='random', fit_intercept=True, n_jobs=1)
    try:
        model.fit(X_clean, y_clean)
        coefs = model.coef_
        alpha_opt = model.alpha_
        r2 = model.score(X_clean, y_clean)
    except Exception as e:
        logging.debug(f"LassoCV fit failed: {e}")
        coefs = np.full(n_predictors, np.nan, dtype=np.float32)
        alpha_opt = np.float32(np.nan)
        r2 = np.float32(np.nan)
    return (coefs.astype(np.float32), np.float32(alpha_opt), np.float32(r2))

def lasso_fixed_alpha_regress_1D(y, X, fixed_alpha):
    """
    Performs LASSO regression using a pre-determined fixed alpha. Relies on global MAX_ITER_LASSO.
    """
    n_predictors = X.shape[1]
    mask = ~np.any(np.isnan(X), axis=1) & ~np.isnan(y)
    n_samples = np.sum(mask)
    min_samples_needed = n_predictors + 1

    if n_samples < min_samples_needed:
        logging.debug(f"Skipping Lasso (fixed alpha): Insufficient samples ({n_samples} < {min_samples_needed})")
        return (np.full(n_predictors, np.nan, dtype=np.float32),
                np.float32(fixed_alpha), np.float32(np.nan))
    y_clean = y[mask]
    X_clean = X[mask, :]

    if np.std(y_clean) < 1e-10:
        logging.debug("Skipping Lasso (fixed alpha): Target variable has near-zero variance.")
        return (np.zeros(n_predictors, dtype=np.float32),
                np.float32(fixed_alpha), np.float32(0.0))

    model = Lasso(alpha=fixed_alpha, max_iter=MAX_ITER_LASSO, random_state=42,
                  tol=1e-3, selection='random', fit_intercept=True)
    try:
        model.fit(X_clean, y_clean)
        coefs = model.coef_
        r2 = model.score(X_clean, y_clean)
    except Exception as e:
        logging.debug(f"Lasso (fixed alpha) fit failed: {e}")
        coefs = np.full(n_predictors, np.nan, dtype=np.float32)
        r2 = np.float32(np.nan)
    return (coefs.astype(np.float32), np.float32(fixed_alpha), np.float32(r2))

def calculate_spatial_alpha(y_target_spatial, lag_range_str, target_season_str):
    """
    Calculates a single representative LASSO alpha.
    Relies on global: PACIFIC_BBOX_ALPHA, CV_FOLDS_LASSO, MAX_ITER_LASSO
    """
    logging.info(f"Calculating spatial alpha using LassoCV on Pacific subset for {lag_range_str}, season {target_season_str}...")
    y_aligned, X_aligned_dict, confounders_aligned_dict = align_data_for_season(
        y_target_spatial, target_season_str, lag_range_str, 'COMBINED'
    )
    if y_aligned is None or not X_aligned_dict or not confounders_aligned_dict:
        logging.error("Alignment failed during spatial alpha calculation. Cannot proceed.")
        return np.nan

    try:
        lat_slice = slice(PACIFIC_BBOX_ALPHA['lat_min'], PACIFIC_BBOX_ALPHA['lat_max'])
        lon_slice = slice(PACIFIC_BBOX_ALPHA['lon_min'], PACIFIC_BBOX_ALPHA['lon_max'])
        
        # Handle pressure levels for LASSO alpha calculation
        if 'pressure_level' in y_aligned.dims:
            # For RWS and other multi-level variables, average over all available levels
            # This provides a stable estimate representing the full vertical structure
            y_subset = y_aligned.sel(
                latitude=lat_slice, 
                longitude=lon_slice
            ).mean(dim='pressure_level')  # Average over all levels for stable alpha
            available_levels = y_aligned.pressure_level.values
            logging.info(f"Using averaged pressure levels {available_levels} for spatial alpha calculation")
        else:
            y_subset = y_aligned.sel(latitude=lat_slice, longitude=lon_slice)
            
        X_subset_dict = X_aligned_dict
        confounders_subset_dict = confounders_aligned_dict
        if y_subset.latitude.size == 0 or y_subset.longitude.size == 0:
            logging.error("Spatial subsetting for alpha calculation resulted in zero grid points.")
            return np.nan
    except Exception as e:
        logging.error(f"Error during spatial subsetting for alpha calculation: {e}", exc_info=True)
        return np.nan

    confounders_std_dict = {}
    valid_confounders = True
    for conf_name, conf_da in confounders_subset_dict.items():
        try:
            conf_std = standardize_dim(conf_da, 'time')
            if conf_std.isnull().all():
                valid_confounders = False; break
            confounders_std_dict[conf_name] = conf_std
        except: valid_confounders = False; break
    if not valid_confounders: return np.nan

    y_cleaned_subset, confounders_std_cleaned_dict, X_cleaned_subset_dict = clean_aligned_data(
        y_subset, confounders_std_dict, X_subset_dict
    )
    if y_cleaned_subset is None or not confounders_std_cleaned_dict or not X_cleaned_subset_dict or y_cleaned_subset.time.size == 0:
        logging.error("Cleaning subset data failed or resulted in no time steps.")
        return np.nan

    try:
        confounder_names = sorted(list(confounders_std_cleaned_dict.keys()))
        confounder_list_for_concat = [confounders_std_cleaned_dict[name] for name in confounder_names]
        combined_confounders_da = xr.concat(confounder_list_for_concat, dim='confounder_id').assign_coords(confounder_id=confounder_names).transpose('time', 'confounder_id')
        
        y_cleaned_subset_chunked = y_cleaned_subset.chunk({'time': -1, 'latitude':'auto', 'longitude':'auto'})
        combined_confounders_da_chunked = combined_confounders_da.chunk({'time': -1, 'confounder_id': -1})
        y_final_subset = remove_signal(y_cleaned_subset_chunked, combined_confounders_da_chunked, 'time', False)
        if y_final_subset is None or y_final_subset.isnull().all(): return np.nan
    except Exception as e:
        logging.error(f"Error removing confounder signal for subset: {e}. Skipping alpha calc.")
        return np.nan

    X_processed_subset_da = create_predictor_array(X_cleaned_subset_dict)
    if X_processed_subset_da is None or X_processed_subset_da.size == 0: return np.nan

    n_predictors = X_processed_subset_da.sizes['predictor']
    common_apply_kwargs = {
        "input_core_dims": [['time'], ['time', 'predictor']], "vectorize": True, "dask": 'parallelized',
        "dask_gufunc_kwargs": {'output_sizes': {'predictor': n_predictors}, 'allow_rechunk': True},
        "output_core_dims": [['predictor'], [], []], "output_dtypes": [np.float32, np.float32, np.float32]
    }
    try:
        y_final_subset_chunked = y_final_subset.chunk({'latitude': 'auto', 'longitude': 'auto', 'time': -1})
        X_processed_subset_da_chunked = X_processed_subset_da.chunk({'time': -1, 'predictor': -1})
        
        regr_result_subset = xr.apply_ufunc(
            lasso_cv_regress_1D, y_final_subset_chunked, X_processed_subset_da_chunked,
            **common_apply_kwargs
        )
        with ProgressBar(): alpha_optimal_da = regr_result_subset[1].compute()
    except Exception as e:
        logging.error(f"Error during LassoCV regression on subset: {e}", exc_info=True)
        return np.nan

    try:
        median_alpha = np.nanmedian(alpha_optimal_da.values)
        if np.isnan(median_alpha): return np.nan
        logging.info(f"Calculated Median Optimal Alpha from Pacific subset: {median_alpha:.5f}")
        return float(median_alpha)
    except Exception as e:
        logging.error(f"Error calculating median alpha: {e}")
        return np.nan

def run_regression_set(y_final, X_processed_da, predictor_names, method, season, fixed_alpha=None):
    """
    Applies the specified regression method (OLS or LASSO with fixed alpha).
    """
    if X_processed_da is None or X_processed_da.size == 0 or 'predictor' not in X_processed_da.dims or X_processed_da.sizes['predictor'] == 0:
        logging.warning(f"Skipping regression for season {season}: Invalid or empty predictor array.")
        return None
    if y_final is None or y_final.size == 0:
        logging.warning(f"Skipping regression for season {season}: Invalid or empty target array.")
        return None

    n_predictors = X_processed_da.sizes['predictor']
    logging.info(f"Running {method} regression with {n_predictors} predictors for {season}.")

    apply_kwargs = {
        "input_core_dims": [['time'], ['time', 'predictor']], "vectorize": True, "dask": 'parallelized',
        "dask_gufunc_kwargs": {'output_sizes': {'predictor': n_predictors}, 'allow_rechunk': True},
        "kwargs": {}
    }
    try:
        # Adaptive chunking for pressure levels if present
        chunk_dict = {'time': -1}  # Always load full time dimension
        if 'pressure_level' in y_final.dims:
            chunk_dict['pressure_level'] = min(5, y_final.sizes['pressure_level'])  # Process 5 levels at a time
            chunk_dict['latitude'] = 'auto'
            chunk_dict['longitude'] = 'auto'
        else:
            chunk_dict['latitude'] = 'auto'
            chunk_dict['longitude'] = 'auto'
        
        y_final_chunked = y_final.chunk(chunk_dict)
        X_processed_da_chunked = X_processed_da.chunk({'time': -1, 'predictor': -1})

        if method == 'OLS':
            regression_func = ols_regress_1D
            output_core_dims = [['predictor'], ['predictor'], []]; output_dtypes = [np.float32, np.float32, np.float32]
            result_vars = ['regr_coefs', 'p_values', 'r_squared']
        elif method == 'LASSO_fixed_alpha':
            if fixed_alpha is None or np.isnan(fixed_alpha):
                logging.error(f"Fixed alpha required but not provided/NaN for LASSO_fixed_alpha in {season}.")
                return None
            regression_func = lasso_fixed_alpha_regress_1D
            output_core_dims = [['predictor'], [], []]; output_dtypes = [np.float32, np.float32, np.float32]
            result_vars = ['regr_coefs', 'fixed_alpha', 'r_squared']
            apply_kwargs["kwargs"] = {'fixed_alpha': fixed_alpha}
        else: raise ValueError(f"Unsupported regression method: {method}")

        regr_result = xr.apply_ufunc(
            regression_func, y_final_chunked, X_processed_da_chunked,
            output_core_dims=output_core_dims, output_dtypes=output_dtypes, **apply_kwargs
        )
        logging.info(f"Computation starting for {method} on {season}...")
        with ProgressBar():
            computed_data = [res.compute() for res in (regr_result if isinstance(regr_result, tuple) else (regr_result,))]
        logging.info(f"Computation finished for {season}.")

        results = []
        for i, var_name in enumerate(result_vars):
            computed_da = computed_data[i]
            if 'predictor' in computed_da.dims:
                computed_da = computed_da.assign_coords(predictor=predictor_names)
            results.append(computed_da.expand_dims(season=[season]))
        return results
    except Exception as e:
        logging.error(f"Error during {method} regression for {season}: {e}", exc_info=True)
        return None

def process_target_variable(target_var, pathway_name, target_name, output_base_path):
    """
    Processes a single target variable through all regression sets and seasons.
    Relies on global: REGRESSION_SETS, LAG_TO_SEASON, SEASONS, etc.
    """
    logging.info(f"\n--- Processing Target Variable: {target_name} (Pathway Context: {pathway_name}) ---")
    
    # Check if target variable has pressure levels
    has_pressure_levels = 'pressure_level' in target_var.dims
    if has_pressure_levels:
        logging.info(f"Target variable {target_name} has {target_var.sizes['pressure_level']} pressure levels")

    for set_idx, regr_set_tuple in enumerate(REGRESSION_SETS): # REGRESSION_SETS is global
        lag_range_str, method, _ = regr_set_tuple

        is_combined_set = '_COMBINED' in lag_range_str
        if (pathway_name == 'COMBINED' and not is_combined_set) or \
           (pathway_name != 'COMBINED' and is_combined_set):
            logging.debug(f"Skipping set {lag_range_str} ({method}) for {pathway_name} pathway context.")
            continue
        
        logging.info(f"--> Running Regression Set {set_idx+1}: Lags={lag_range_str}, Method={method} for Pathway={pathway_name}")
        clean_lag_range = lag_range_str.replace('_COMBINED', '')
        min_lag = int(clean_lag_range.replace('lag', '').split('-')[-1]) # Gets min_lag

        store_results = {'regr_coefs': [], 'r_squared': []}
        if method == 'OLS': store_results['p_values'] = []
        if method == 'LASSO_fixed_alpha': store_results['fixed_alpha'] = []
        
        filename_base = f"{target_name}_{pathway_name}_{lag_range_str}_{method}"
        start_season = LAG_TO_SEASON.get(min_lag) # LAG_TO_SEASON is global
        if start_season is None: continue
        try:
            start_idx = SEASONS.index(start_season) # SEASONS is global
            applicable_seasons = SEASONS[start_idx:]
        except ValueError: continue

        fixed_lasso_alpha = np.nan
        if method == 'LASSO_fixed_alpha':
            # calculate_spatial_alpha uses global PACIFIC_BBOX_ALPHA, CV_FOLDS_LASSO, MAX_ITER_LASSO
            fixed_lasso_alpha = calculate_spatial_alpha(
                target_var, lag_range_str, applicable_seasons[0]
            )
            if np.isnan(fixed_lasso_alpha):
                logging.error(f"Failed to calculate spatial alpha for {lag_range_str}. Skipping set.")
                continue
        
        for season in applicable_seasons:
            logging.info(f"  -- Processing Target Season: {season} --")
            y_aligned, X_aligned_dict, confounders_aligned_dict = align_data_for_season(
                target_var, season, lag_range_str, pathway_name
            )
            if y_aligned is None or X_aligned_dict is None or confounders_aligned_dict is None:
                logging.warning(f"    Alignment failed for {season}, skipping.")
                continue

            confounders_std_dict = {}
            valid_confounders = True
            for conf_name, conf_da in confounders_aligned_dict.items():
                try:
                    conf_std = standardize_dim(conf_da, 'time')
                    if conf_std.isnull().all(): valid_confounders = False; break
                    confounders_std_dict[conf_name] = conf_std
                except: valid_confounders = False; break
            if not valid_confounders: continue

            y_cleaned, confounders_std_cleaned_dict, X_aligned_cleaned_dict = clean_aligned_data(
                y_aligned, confounders_std_dict, X_aligned_dict
            )
            if y_cleaned is None or not confounders_std_cleaned_dict or not X_aligned_cleaned_dict or y_cleaned.time.size == 0:
                logging.warning(f"    Cleaning failed or removed all data for {season}. Skipping.")
                continue

            try:
                confounder_names = sorted(list(confounders_std_cleaned_dict.keys()))
                conf_list = [confounders_std_cleaned_dict[name] for name in confounder_names]
                combined_confounders_da = xr.concat(conf_list, dim='confounder_id').assign_coords(confounder_id=confounder_names).transpose('time', 'confounder_id')
                
                # Adaptive chunking for pressure levels if present
                chunk_dict_clean = {'time': -1}
                if 'pressure_level' in y_cleaned.dims:
                    chunk_dict_clean['pressure_level'] = min(5, y_cleaned.sizes['pressure_level'])
                    chunk_dict_clean['latitude'] = 'auto'
                    chunk_dict_clean['longitude'] = 'auto'
                else:
                    chunk_dict_clean['latitude'] = 'auto'
                    chunk_dict_clean['longitude'] = 'auto'
                y_cleaned_chunked = y_cleaned.chunk(chunk_dict_clean)
                combined_confounders_da_chunked = combined_confounders_da.chunk({'time': -1, 'confounder_id': -1})
                y_final = remove_signal(y_cleaned_chunked, combined_confounders_da_chunked, 'time', False)
                if y_final is None or y_final.isnull().all(): logging.warning(f"    Removing confounder signal all NaNs for {season}. Skipping."); continue
            except Exception as e: logging.error(f"    Error processing confounders for {season}: {e}. Skipping."); continue

            predictor_names = sorted(list(X_aligned_cleaned_dict.keys()))
            X_processed_da = create_predictor_array(X_aligned_cleaned_dict)
            if X_processed_da is None or X_processed_da.size == 0: logging.warning(f"    Predictor array empty for {season}. Skipping."); continue

            results = run_regression_set(
                y_final, X_processed_da, predictor_names, method, season,
                fixed_alpha=fixed_lasso_alpha if method == 'LASSO_fixed_alpha' else None
            )
            if results:
                logging.info(f"    Regression successful for {season}.")
                store_results['regr_coefs'].append(results[0])
                store_results['r_squared'].append(results[-1]) # R-squared is always last
                if method == 'OLS': store_results['p_values'].append(results[1])
                elif method == 'LASSO_fixed_alpha': store_results['fixed_alpha'].append(results[1]) # Fixed alpha value
            else:
                logging.warning(f"    Regression failed for {season}.")

        if store_results['regr_coefs']:
            ds_results = xr.Dataset()
            try:
                ds_results['regr_coefs'] = xr.concat(store_results['regr_coefs'], dim='season', join='exact')
                ds_results['r_squared'] = xr.concat(store_results['r_squared'], dim='season', join='exact')
                if method == 'OLS':
                    ds_results['p_values'] = xr.concat(store_results['p_values'], dim='season', join='exact')
                elif method == 'LASSO_fixed_alpha':
                    ds_results['fixed_alpha'] = xr.concat(store_results['fixed_alpha'], dim='season', join='exact')
                
                output_subdir = f"{pathway_name}_pathway"
                output_path = os.path.join(output_base_path, output_subdir)
                os.makedirs(output_path, exist_ok=True)
                out_file = os.path.join(output_path, f"{filename_base}.nc")
                ds_results.to_netcdf(out_file)
                logging.info(f"--> Saved Set {set_idx+1} results to: {out_file}")
            except Exception as e:
                logging.error(f"    Error concatenating or saving results for {filename_base}: {e}", exc_info=True)
        else:
            logging.warning(f"--> No valid results for regression set {set_idx+1} ({filename_base}). No file saved.")


# --- Data Loading Function for Worker ---
def load_data_in_worker(target_name_to_load, worker_config):
    """
    Loads data necessary for a single worker job.
    Populates global variables: CONFOUNDERS, PREDICTORS, PREDICTOR_LAG_MAP,
    SEASONS, REGRESSION_SETS, and other path/config globals.
    Returns the specific target_var DataArray.
    """
    # Declare which globals this function will modify for the worker's process
    global CONFOUNDERS, PREDICTORS, PREDICTOR_LAG_MAP, SEASONS, REGRESSION_SETS
    global DATA_DIR_GRIDDED, DATA_DIR_TS, TIME_PERIOD_START, TIME_PERIOD_END
    global MAX_ITER_LASSO, CV_FOLDS_LASSO, PACIFIC_BBOX_ALPHA, LAG_TO_SEASON
    global season_to_month, target_season_suffix_to_lag, predictor_season_lag_map
    
    pid = os.getpid()
    logging.info(f"Worker {pid}: Initializing data load for target '{target_name_to_load}'.")

    # Populate global config variables from worker_config
    DATA_DIR_GRIDDED = worker_config['DATA_DIR_GRIDDED']
    DATA_DIR_TS = worker_config['DATA_DIR_TS']
    # OUTPUT_DIR is handled by process_target_variable's argument
    TIME_PERIOD_START = pd.Timestamp(worker_config['TIME_PERIOD_START'])
    TIME_PERIOD_END = pd.Timestamp(worker_config['TIME_PERIOD_END'])
    
    MAX_ITER_LASSO = worker_config['MAX_ITER_LASSO']
    CV_FOLDS_LASSO = worker_config['CV_FOLDS_LASSO']
    PACIFIC_BBOX_ALPHA = worker_config['PACIFIC_BBOX_ALPHA']
    LAG_TO_SEASON = worker_config['LAG_TO_SEASON']
    
    season_to_month = worker_config['season_to_month']
    target_season_suffix_to_lag = worker_config['target_season_suffix_to_lag']
    predictor_season_lag_map = worker_config['predictor_season_lag_map']
    
    SEASONS = worker_config['TARGET_SEASONS_TO_MAP']
    REGRESSION_SETS = worker_config['REGRESSION_SETS']

    # Reconstruct predictor_paths
    _predictor_paths_structure = worker_config['predictor_paths_structure']
    # This local `predictor_paths` is used to feed `load_predictors`
    local_predictor_paths_fully_resolved = {} 
    for p_pathway, p_seasons in _predictor_paths_structure.items():
        local_predictor_paths_fully_resolved[p_pathway] = {}
        for p_season, p_files in p_seasons.items():
            local_predictor_paths_fully_resolved[p_pathway][p_season] = {}
            for p_name, p_file_suffix in p_files.items():
                local_predictor_paths_fully_resolved[p_pathway][p_season][p_name] = os.path.join(DATA_DIR_TS, p_file_suffix)

    _target_variable_defs = worker_config['target_variable_defs']

    # --- Load Confounders (E/C indices) ---
    logging.info(f"Worker {pid}: Loading confounder indices...")
    try:
        E_index_st_path = os.path.join(DATA_DIR_TS, worker_config['confounder_paths']['E_index_st'])
        C_index_st_path = os.path.join(DATA_DIR_TS, worker_config['confounder_paths']['C_index_st'])

        E_index_st = xr.open_dataset(E_index_st_path)['E_index']
        C_index_st = xr.open_dataset(C_index_st_path)['C_index']

        # Define base_time_coord using one index and specified period
        # Ensure base_time_coord is consistent for all data alignment
        time_slice = slice(str(TIME_PERIOD_START.year), str(TIME_PERIOD_END.year))
        # Attempt to create a full time range for robust reindexing
        try:
            # Use a known frequency if possible, e.g., 'MS' for month start
            # For seasonal data, this might be more complex. Assuming time coord is sufficient.
            ref_time_for_base = E_index_st.time.sel(time=time_slice)
            if ref_time_for_base.size == 0 and C_index_st.time.sel(time=time_slice).size > 0:
                 ref_time_for_base = C_index_st.time.sel(time=time_slice)
            elif ref_time_for_base.size == 0: # Both empty for slice
                 raise ValueError("Both E and C indices are empty for the selected time period before creating base_time_coord.")
            base_time_coord = ref_time_for_base
            logging.info(f"Worker {pid}: Base time coordinate from {TIME_PERIOD_START} to {TIME_PERIOD_END} has {base_time_coord.size} steps.")

        except Exception as e_time:
            logging.error(f"Worker {pid}: Error creating base_time_coord: {e_time}", exc_info=True)
            raise

        E_index_st = E_index_st.reindex(time=base_time_coord, method=None).rename('E_index')
        C_index_st = C_index_st.reindex(time=base_time_coord, method=None).rename('C_index')

        if E_index_st.time.size == 0 or C_index_st.time.size == 0:
            raise ValueError("Confounder indices are empty after time selection/reindexing.")
        CONFOUNDERS = {'E_index': E_index_st, 'C_index': C_index_st}
        logging.info(f"Worker {pid}: Loaded E_index ({E_index_st.time.size} steps), C_index ({C_index_st.time.size} steps).")
    except Exception as e:
        logging.error(f"Worker {pid}: FATAL - Failed to load E/C confounder indices: {e}", exc_info=True)
        raise 

    # --- Load Predictors ---
    logging.info(f"Worker {pid}: Loading predictor time series...")
    # These will populate the global PREDICTORS and PREDICTOR_LAG_MAP
    _all_predictors_ep = {}
    _all_predictors_cp = {}
    _all_predictor_lag_map = {}
    try:
        # Iterate through the fully resolved paths structure
        for pathway, pred_seasons_dict in local_predictor_paths_fully_resolved.items():
            current_pathway_predictors = {} # For this pathway EP or CP
            for season_lag_name, preds_in_season_files in pred_seasons_dict.items():
                if not preds_in_season_files: continue
                # load_predictors expects dict of {name: filepath}
                loaded_preds = load_predictors(preds_in_season_files, base_time_coord)
                current_pathway_predictors.update(loaded_preds)

                lag_val = predictor_season_lag_map.get(season_lag_name) # Uses global predictor_season_lag_map
                if lag_val is not None:
                    for pred_name in loaded_preds:
                        _all_predictor_lag_map[pred_name] = lag_val
                else:
                    logging.warning(f"Worker {pid}: Predictor season '{season_lag_name}' not in predictor_season_lag_map.")
            
            if pathway == 'EP':
                _all_predictors_ep.update(current_pathway_predictors)
            elif pathway == 'CP':
                _all_predictors_cp.update(current_pathway_predictors)
        
        PREDICTORS = {'EP': _all_predictors_ep, 'CP': _all_predictors_cp}
        PREDICTOR_LAG_MAP = _all_predictor_lag_map
        logging.info(f"Worker {pid}: Loaded {len(_all_predictors_ep)} EP, {len(_all_predictors_cp)} CP predictors. Total {len(_all_predictor_lag_map)} in lag map.")

    except Exception as e:
        logging.error(f"Worker {pid}: FATAL - Failed during predictor loading: {e}", exc_info=True)
        raise

    # --- Load specific target variable ---
    logging.info(f"Worker {pid}: Loading target spatial variable: {target_name_to_load}")
    target_var_to_return = None
    try:
        definition = _target_variable_defs[target_name_to_load]
        var_path = os.path.join(DATA_DIR_GRIDDED, definition['filename']) # Uses global DATA_DIR_GRIDDED
        if not os.path.exists(var_path):
            raise FileNotFoundError(f"Target file not found: {var_path}")

        # load_era_field is an existing utility function
        loaded_var = load_era_field(filepath=var_path, var_name=definition['var_name'], **definition['kwargs'])
        if loaded_var.name == '__xarray_dataarray_variable__': # Default name if not set
            loaded_var = loaded_var.rename(target_name_to_load)
        
        loaded_var = loaded_var.sel(time=slice(str(TIME_PERIOD_START.year), str(TIME_PERIOD_END.year)))
        reindexed_var = loaded_var.reindex(time=base_time_coord, method=None)

        if reindexed_var.time.size == 0:
            raise ValueError(f"Target variable '{target_name_to_load}' is empty after time selection/reindexing.")
        target_var_to_return = reindexed_var
        logging.info(f"Worker {pid}: Loaded target {target_name_to_load} with shape {target_var_to_return.shape}")
    except Exception as e:
        logging.error(f"Worker {pid}: FATAL - Failed to load target variable '{target_name_to_load}': {e}", exc_info=True)
        raise
        
    # Apply Dask config for ASUS Zenbook if specified
    if worker_config.get('apply_dask_config_asus', False):
        import dask
        dask.config.set({"array.slicing.split_large_chunks": False})
        logging.info(f"Worker {pid}: Applied dask.config array.slicing.split_large_chunks=False")

    logging.info(f"Worker {pid}: Data loading complete for target '{target_name_to_load}'.")
    return target_var_to_return

# --- Core Worker Function ---
def run_outer_job(pathway_name, target_name, config_dict, debug_mode):
    """The function executed by each worker process."""
    worker_pid = os.getpid()
    # Setup logging for this specific worker process
    # This must be one of the first things called in the new process.
    setup_worker_logging(worker_pid, debug_mode, config_dict['OUTPUT_DIR'])
    
    _set_single_thread_env()

    logging.info(f"Worker {worker_pid}: Starting job for Pathway: {pathway_name}, Target: {target_name}")
    
    try:
        # Load data within the worker. This sets the necessary global variables
        # (CONFOUNDERS, PREDICTORS, PREDICTOR_LAG_MAP, SEASONS, REGRESSION_SETS, etc.)
        # for process_target_variable to use.
        target_var_da = load_data_in_worker(target_name, config_dict)
        
        # The output_base_path for process_target_variable is the main script's output dir
        # process_target_variable will create subdirectories like EP_pathway within it.
        process_target_variable(target_var_da, pathway_name, target_name, config_dict['OUTPUT_DIR'])
        
        logging.info(f"Worker {worker_pid}: Successfully completed job for Pathway: {pathway_name}, Target: {target_name}")
        return (pathway_name, target_name, "SUCCESS", None)
    except Exception as e:
        # Log the full traceback within the worker's log file
        tb_str = traceback.format_exc()
        logging.error(f"Worker {worker_pid}: FAILED job for Pathway: {pathway_name}, Target: {target_name}. Error: {e}\nTraceback:\n{tb_str}")
        # Return error information for the main process to summarize
        return (pathway_name, target_name, "FAILED", f"Error: {e}\n{tb_str}")

# --- Static Input Aggregation ---
def gather_static_inputs(args):
    """
    Gathers all configurations, paths, and definitions that are static across jobs.
    This dictionary will be passed (pickled) to each worker.
    """
    config = {}
    
    # Resolve paths using get_data_path (ensure it's available)
    try:
        config['DATA_DIR_GRIDDED'] = get_data_path('1_deg_seasonal/', data_type="interim")
        config['DATA_DIR_TS'] = get_data_path('time_series', data_type="processed")
    except NameError: # If get_data_path is not defined (e.g. utils not imported)
        logging.error("get_data_path is not defined. Ensure utils.paths is correctly imported.")
        raise
    
    # Main output directory for the parallel script run
    base_output_dir = os.path.join(config['DATA_DIR_TS'], 'seas_multilag_regr_coeffs')
    os.makedirs(base_output_dir, exist_ok=True)
    config['OUTPUT_DIR'] = base_output_dir

    # Analysis Parameters
    config['TARGET_SEASONS_TO_MAP'] = ['JJA_m1', 'SON_m1', 'DJF_0', 'MAM_0', 'JJA_0', 'SON_0', 'DJF_1']
    config['TIME_PERIOD_START'] = '1945-06-01'
    config['TIME_PERIOD_END'] = '2024-02-01'
    config['MAX_ITER_LASSO'] = 10000
    config['CV_FOLDS_LASSO'] = 5
    config['PACIFIC_BBOX_ALPHA'] = {'lat_min': -25, 'lat_max': 10, 'lon_min': -120, 'lon_max': -75}
    config['LAG_TO_SEASON'] = {
        6: "JJA_m1", 5: "SON_m1", 4: "DJF_0", 3: "MAM_0",
        2: "JJA_0", 1: "SON_0", 0: "DJF_1"
    }
    
    config['season_to_month'] = {'JJA': 7, 'SON': 10, 'DJF': 1, 'MAM': 4}
    config['target_season_suffix_to_lag'] = {
        'JJA_m1': 6, 'SON_m1': 5, 'DJF_0': 4, 'MAM_0': 3,
        'JJA_0': 2, 'SON_0': 1, 'DJF_1': 0
    }
    config['predictor_season_lag_map'] = {'JJA_m1': 6, 'SON_m1': 5, 'DJF_0': 4, 'MAM_0': 3}

    # Predictor paths structure (relative to DATA_DIR_TS)
    config['predictor_paths_structure'] = {
        'EP': {
            'JJA_m1': {'REOF SST JJA': 'E-ind/REOF_SST_ts_mode4_JJA.nc'},
            'SON_m1': {'MCA WAF-RWS SON': 'E-ind/MCA_RWS_WAF_ts_WAF_mode1_SON.nc'},
            'DJF_0': {'MCA RWS-WAF DJF': 'E-ind/MCA_RWS_200_WAF_ts_RWS_200_mode2_DJF.nc'},
            'MAM_0': {'MCA RWS-prec MAM(E)': 'E-ind/MCA_prec_RWS_200_ts_RWS_200_mode2_MAM.nc'}
        },
        'CP': {
            'JJA_m1': {'REOF SST JJA': 'E-ind/REOF_SST_ts_mode4_JJA.nc'},
            'SON_m1': {'MCA prec-RWS SON': 'C-ind/MCA_prec_RWS_200_ts_prec_mode2_SON.nc'},
            'DJF_0': {'MCA RWS-prec DJF': 'C-ind/MCA_prec_RWS_200_ts_RWS_200_mode2_DJF.nc'},
            'MAM_0': {'MCA RWS-prec MAM(C)': 'C-ind/MCA_prec_RWS_200_ts_RWS_200_mode2_MAM.nc'}
        }
    }
    
    # Full target variable definitions - filter based on args.target_variables CLI argument
    all_target_variable_defs = {
        'SST': {'filename': 'SST_seas_1deg.nc', 'var_name': 'sst', 'kwargs': {}},
        'prec': {'filename': 'prec_seas_1deg.nc', 'var_name': 'tp', 'kwargs': {}},
        'surf_pres': {'filename': 'surf_pres_seas_1deg.nc', 'var_name': 'sp', 'kwargs': {}},
        'vp_200': {'filename': 'vp_200_seas_1deg.nc', 'var_name': 'velocity_potential', 'kwargs': {'lat_slice':(-50, 50)}},
        'low_clouds': {'filename': 'low_cloud_cover_seas_1deg.nc', 'var_name': 'lcc', 'kwargs': {}},
        'U_200': {'filename': 'U_200_seas_1deg.nc', 'var_name': 'u', 'kwargs': {'lat_slice':(-80, 50)}},
        'V_200': {'filename': 'V_200_seas_1deg.nc', 'var_name': 'v', 'kwargs': {'lat_slice':(-80, 50)}},
        'U_10m': {'filename': 'U_10m_seas_1deg.nc', 'var_name': 'u10', 'kwargs': {}},
        'V_10m': {'filename': 'V_10m_seas_1deg.nc', 'var_name': 'v10', 'kwargs': {}},
        'sf_200': {'filename': 'sf_200_seas_1deg.nc', 'var_name': 'streamfunction', 'kwargs': {'lat_slice':(-80, 50)}},
        'U_850': {'filename': 'U_850_seas_1deg.nc', 'var_name': 'u', 'kwargs': {}},
        'V_850': {'filename': 'V_850_seas_1deg.nc', 'var_name': 'v', 'kwargs': {}},
        'RWS_200': {'filename': 'RWS_200_seas_1deg.nc', 'var_name': 'RWS', 'kwargs': {'lat_slice':(-80, 50)}},
        'WAFx_200': {'filename': 'WAF_200_components_1deg.nc', 'var_name': 'WAFx', 'kwargs': {'lat_slice':(-80, 50)}},
        'WAFy_200': {'filename': 'WAF_200_components_1deg.nc', 'var_name': 'WAFy', 'kwargs': {'lat_slice':(-80, 50)}},
        'vertical_streamfunction': {'filename': 'vertical_streamfunction_seas_1deg.nc', 'var_name': 'psi', 'kwargs': {}},
        'RWS_multi': {'filename': 'RWS_seas_1deg.nc', 'var_name': 'RWS', 'kwargs': {}}
    }

    # Filter to selected target variables from CLI (args.target_variables)
    selected_vars = getattr(args, 'target_variables', ['U_10m', 'V_10m'])
    config['target_variable_defs'] = {
        var: all_target_variable_defs[var]
        for var in selected_vars
        if var in all_target_variable_defs
    }
    logging.info(f"Target variables to process: {list(config['target_variable_defs'].keys())}")

    config['confounder_paths'] = {
        'E_index_st': 'E_index_st_ts.nc',
        'C_index_st': 'C_index_st_ts.nc'
    }

    config['REGRESSION_SETS'] = [
        ('lag6', 'OLS', None), ('lag6-5', 'OLS', None), ('lag6-4', 'OLS', None), ('lag6-3', 'OLS', None),
        ('lag6-4', 'LASSO_fixed_alpha', None), ('lag6-3', 'LASSO_fixed_alpha', None),
        ('lag6_COMBINED', 'OLS', None), ('lag6-5_COMBINED', 'OLS', None),
        ('lag6-4_COMBINED', 'LASSO_fixed_alpha', None), ('lag6-3_COMBINED', 'LASSO_fixed_alpha', None),
    ]
    
    config['apply_dask_config_asus'] = args.asus_memory_config
    return config

# --- Job Launcher ---
def launch_driver(args, job_specs, config_dict):
    """Orchestrates the parallel execution of jobs."""
    num_jobs = len(job_specs)
    if args.dry_run:
        logging.info("--- DRY RUN MODE ---")
        logging.info(f"Would attempt to run {num_jobs} jobs with max_workers = {args.max_workers or 'auto'}:")
        for spec in job_specs:
            logging.info(f"  - Pathway: {spec[0]}, Target: {spec[1]}")
        logging.info("--- END DRY RUN ---")
        return [], []

    if args.max_workers == 1 and args.scheduler == 'processpool': # Fallback to serial execution
        logging.info("Running in serial mode (max_workers=1). Worker logs will be in logs/worker_<pid>.log.")
        success_jobs = []
        failed_jobs = []
        _set_single_thread_env() # Set for main process too in serial mode
        for pathway_name, target_name in tqdm(job_specs, desc="Serial Jobs Progress"):
            # Call run_outer_job directly. It handles its own logging.
            result = run_outer_job(pathway_name, target_name, config_dict, args.debug)
            if result[2] == "SUCCESS": # result is (pathway, target, status, message)
                success_jobs.append(result[:2]) # Store (pathway, target)
            else:
                failed_jobs.append(result) # Store (pathway, target, status, message)
        return success_jobs, failed_jobs

    # Determine number of workers for parallel execution
    try:
        physical_cores = psutil.cpu_count(logical=False)
        if physical_cores is None:
            physical_cores = psutil.cpu_count(logical=True)
            logging.info(f"Could not detect physical cores, using logical cores: {physical_cores}")
        else:
            logging.info(f"Detected {physical_cores} physical cores.")
    except Exception as e:
        logging.warning(f"Could not detect CPU cores using psutil: {e}. Defaulting to 2 cores.")
        physical_cores = 2
        
    max_workers_to_use = args.max_workers if args.max_workers is not None else physical_cores
    # Limit to physical CPU cores for optimal computational performance
    # For CPU-bound, final_workers = min(max_workers_to_use, physical_cores) is safer.
    # Given the spec, allow user to override:
    final_workers = max_workers_to_use
    logging.info(f"Using {final_workers} worker processes.")

    success_jobs = []
    failed_jobs = []

    if args.scheduler == 'processpool':
        with concurrent.futures.ProcessPoolExecutor(max_workers=final_workers, initializer=_set_single_thread_env) as executor:
            futures = {
                executor.submit(run_outer_job, pathway_name, target_name, config_dict, args.debug): (pathway_name, target_name)
                for pathway_name, target_name in job_specs
            }
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=num_jobs, desc="Parallel Jobs Progress"):
                pathway_name, target_name = futures[future] # Get original job spec for this future
                try:
                    # result is (pathway_name, target_name, status, message_or_none)
                    job_pathway, job_target, status, message = future.result()
                    if status == "SUCCESS":
                        success_jobs.append((job_pathway, job_target))
                        # Main process log
                        logging.info(f"Job COMPLETED: Pathway={job_pathway}, Target={job_target}")
                    else:
                        failed_jobs.append((job_pathway, job_target, message)) # message includes traceback
                        logging.error(f"Job FAILED: Pathway={job_pathway}, Target={job_target}. See worker log and summary. Error snippet: {str(message).splitlines()[0] if message else 'N/A'}")
                except Exception as e:
                    # This catches errors in the future itself (e.g., pickling error, process crashed hard)
                    tb_str = traceback.format_exc()
                    failed_jobs.append((pathway_name, target_name, f"Critical Executor Error: {e}\n{tb_str}"))
                    logging.error(f"Job SUBMISSION/CRITICAL FAILURE for Pathway={pathway_name}, Target={target_name}: {e}", exc_info=True)
    else:
        logging.error(f"Unsupported scheduler: {args.scheduler}. Only 'processpool' is currently implemented.")

    return success_jobs, failed_jobs

# --- Main Execution Block (New Parallel Version) ---
def main():
    parser = argparse.ArgumentParser(description="Run multi-lag regression with parallel processing.")

    # --- Use the same logic as mca_optimization_workflow_RWS_WAF.py for worker count ---
    def available_workers():
        import psutil
        # Memory-based: leave 4 GB for OS, 1 GB per worker
        max_by_mem = (psutil.virtual_memory().total // 2**30 - 4) // 1
        # CPU-based: use half the logical cores
        max_by_cpu = max(1, os.cpu_count() // 2)
        # Clamp to a reasonable upper bound (8)
        return max(1, min(8, max_by_mem, max_by_cpu))

    default_workers = available_workers()

    parser.add_argument(
        "--max-workers",
        type=int,
        default=default_workers,
        help=(
            f"Max number of worker processes. "
            f"Defaults to min(8, available memory in GB minus 4, half the logical CPU cores) "
            f"(default: {default_workers}). "
            "Can be set lower for memory-constrained systems like ASUS Zenbook (e.g., 3 or 4)."
        )
    )
    parser.add_argument("--scheduler", choices=['processpool'], default='processpool',
                        help="Scheduler to use for parallelism (only 'processpool' currently).")
    parser.add_argument("--debug", action='store_true', help="Enable debug logging for main process and workers.")
    parser.add_argument("--dry-run", action='store_true', help="Print jobs to be run without executing them.")
    parser.add_argument("--asus-memory-config", action='store_true',
                        help="Apply Dask config 'array.slicing.split_large_chunks=False' for ASUS Zenbook (low memory).")
    parser.add_argument("--target-variables", nargs='+',
                        default=['U_10m', 'V_10m'],
                        choices=VALID_TARGET_VARIABLES,
                        help=f"Target variables to process (default: U_10m, V_10m). "
                             f"Choices: {', '.join(VALID_TARGET_VARIABLES)}")
    args = parser.parse_args()

    # Main process logging setup
    # Determine output directory for main log based on where get_data_path points
    try:
        _temp_data_dir_ts_for_log = get_data_path('time_series', data_type="processed")
        main_log_base_dir = os.path.join(_temp_data_dir_ts_for_log, 'seas_multilag_regr_coeffs')
    except Exception as e:
        logging.warning(f"Could not determine log path using get_data_path: {e}. Defaulting log path to './logs_main'.")
        main_log_base_dir = 'logs_main'
        
    os.makedirs(main_log_base_dir, exist_ok=True)
    log_file = os.path.join(main_log_base_dir, 'main_regression_log.txt')

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s [MainProcess-%(process)d] [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    logging.info(f"--- Starting Parallel Multi-Lag Regression Script with args: {args} ---")
    
    # Suppress specific warnings for the main process
    warnings.filterwarnings('ignore', category=FutureWarning)
    warnings.filterwarnings('ignore', category=UserWarning, module='dask') # Dask warnings
    warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom <= 0 for slice') # Statsmodels

    # 1. Gather static inputs and configurations
    logging.info("Gathering static inputs and configurations...")
    try:
        config_dict = gather_static_inputs(args)
        # The main output directory for results is now config_dict['OUTPUT_DIR']
        logging.info(f"Results will be saved under: {config_dict['OUTPUT_DIR']}")
    except Exception as e:
        logging.error(f"Failed to gather static inputs: {e}", exc_info=True)
        return

    # 2. Define job specifications (pathway_name, target_name)
    job_specs = []
    pathway_contexts = ['EP', 'CP', 'COMBINED']
    target_var_names = list(config_dict['target_variable_defs'].keys())

    for pathway_name in pathway_contexts:
        # Basic check for predictor definitions (actual loading/failure is in worker)
        if pathway_name == 'EP' and not config_dict['predictor_paths_structure'].get('EP'):
            logging.warning(f"No EP predictors defined in config. EP pathway jobs might be skipped by worker or fail.")
        if pathway_name == 'CP' and not config_dict['predictor_paths_structure'].get('CP'):
            logging.warning(f"No CP predictors defined in config. CP pathway jobs might be skipped by worker or fail.")
        if pathway_name == 'COMBINED' and \
           not (config_dict['predictor_paths_structure'].get('EP') or config_dict['predictor_paths_structure'].get('CP')):
            logging.warning(f"No EP or CP predictors defined. COMBINED pathway jobs might be skipped by worker or fail.")

        for target_name in target_var_names:
            job_specs.append((pathway_name, target_name))
    
    logging.info(f"Generated {len(job_specs)} job specifications.")
    if not job_specs:
        logging.warning("No job specifications generated. Exiting.")
        return

    # 3. Launch driver to execute jobs
    success_jobs, failed_jobs = launch_driver(args, job_specs, config_dict)

    # 4. Report summary
    logging.info("--- Run Summary ---")
    logging.info(f"Total jobs submitted: {len(job_specs)}")
    logging.info(f"Successful jobs: {len(success_jobs)}")
    logging.info(f"Failed jobs: {len(failed_jobs)}")
    if failed_jobs:
        logging.error("--- Details of Failed Jobs ---")
        for job_info in failed_jobs:
            # job_info is (pathway_name, target_name, error_message_with_traceback)
            p_name, t_name, err_msg = job_info
            logging.error(f"  Job: Pathway='{p_name}', Target='{t_name}'")
            logging.error(f"  Error: {err_msg.splitlines()[0] if err_msg else 'Unknown error'}") # First line of error
            # Full traceback is in worker log and also in err_msg
            # For main log, just a snippet is fine. For detailed debugging, check worker log.
            logging.error(f"  (Full traceback for this failure is in the corresponding worker log: logs/worker_<pid>.log and potentially above if critical)")
            logging.error("-" * 30)

    logging.info("--- Parallel Multi-Lag Regression Script Completed ---")
    logging.info(f"Main log file: {log_file}")
    logging.info(f"Worker logs are in: {os.path.abspath('logs')}")
    logging.info(f"Results are in: {os.path.abspath(config_dict['OUTPUT_DIR'])}")


if __name__ == "__main__":
    main()