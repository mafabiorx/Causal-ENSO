# Library imports
import os
import sys
import json
import logging
import re
from typing import List, Tuple, Dict, Sequence # Added Sequence
from contextlib import redirect_stdout, redirect_stderr
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
from tigramite import data_processing as pp
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.pcmci import PCMCI
from tigramite import plotting as tp

# Add the project root directory to the Python path
# This assumes thresholds.py is in the project root or a location accessible via PYTHONPATH
# Adjust if your project structure is different.
try:
    # This will work if the script is run from its directory or project root
    # and thresholds.py are in the same directory as this script or higher up
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_candidates = [
        os.path.abspath(os.path.join(current_script_dir, '../../../../')), # Original path
        os.path.abspath(os.path.join(current_script_dir, '../../..')),
        os.path.abspath(os.path.join(current_script_dir, '../..')),
        os.path.abspath(os.path.join(current_script_dir, '..')),
        os.path.abspath(os.path.join(current_script_dir))
    ]
    thresholds_found = False
    for pr_path in project_root_candidates:
        if pr_path not in sys.path:
            sys.path.insert(0, pr_path)
        try:
            from thresholds import RobustThresholds # Try importing
            from utils.paths import get_results_path # Try importing another key component
            project_root = pr_path # Found a working project root
            thresholds_found = True
            # Using logging.info for setup messages that are not warnings
            logging.info(f"Using project root: {project_root} for script 16")
            break
        except ImportError:
            if pr_path in sys.path: # Clean up if import failed
                sys.path.remove(pr_path)
            continue
    if not thresholds_found:
        # Using logging.error for critical setup failures
        logging.error("Could not find thresholds.py or src.utils.paths. Please check your PYTHONPATH and project structure.")
        logging.error(f"sys.path (script 16): {sys.path}")
        logging.error(f"Attempted project_root_candidates (script 16): {project_root_candidates}")
        # Fallback to try importing directly, assuming PYTHONPATH is set
        try:
            from thresholds import RobustThresholds
            from utils.paths import get_results_path
            logging.info("Imported thresholds and src.utils directly (assuming in PYTHONPATH).")
        except ImportError as e_direct:
            logging.error(f"Direct import also failed: {e_direct}")
            sys.exit(1)


except NameError: # __file__ is not defined
    logging.warning("__file__ not defined in script 16. Assuming thresholds and src.utils are in PYTHONPATH.")
    from thresholds import RobustThresholds
    from utils.paths import get_results_path


THR = RobustThresholds()

# Get environment variables for file paths
target_descriptor = os.environ.get("TARGET_DESCRIPTOR")
generation = os.environ.get("GA_GENERATION")
individual = os.environ.get("GA_INDIVIDUAL")
trial = os.environ.get("TRIAL")

# Ensure environment variables are set
if not all([target_descriptor, generation, individual, trial]):
    logging.error("Error: Required environment variables not set in script 16. "
                  f"TARGET_DESCRIPTOR: {target_descriptor}, GA_GENERATION: {generation}, "
                  f"GA_INDIVIDUAL: {individual}, TRIAL: {trial}")
    sys.exit(1)

# Extract lag value from target descriptor (used for MCA variable season assignment)
target_lag_env = os.environ.get("TARGET_LAG") # Get TARGET_LAG directly
target_lag = None
if target_lag_env:
    try:
        target_lag = int(target_lag_env)
    except ValueError:
        logging.warning(f"TARGET_LAG environment variable '{target_lag_env}' is not a valid integer. Using default lag logic.")
# Fallback to regex if TARGET_LAG is not set (less robust)
if target_lag is None and target_descriptor:
    lag_match = re.search(r'lag_(\d+)', target_descriptor)
    if lag_match:
        target_lag = int(lag_match.group(1))
        logging.info(f"Extracted lag {target_lag} from target_descriptor using regex.")
    else:
        logging.warning("Could not extract lag from target descriptor via regex. MCA season might be default.")


# Analysis parameters
MAX_COMBINATIONS = 10
TAU_MIN = 1
TAU_MAX = 6

# Define paths using target descriptor
VERSION = os.environ.get("VERSION", 'v0')
SUFFIX = 'all_lags_iterative_step1_noFDR_MCA_RWS_WAF'

target_dir_results_base = get_results_path(f'MCA_RWS_WAF/{trial}/{target_descriptor}/{VERSION}')

PATH_SAVE = os.path.join(target_dir_results_base, 'PCMCIplus_results')
os.makedirs(PATH_SAVE, exist_ok=True)

# --- Custom Independence Tests ---
class CustomRobustParCorr(RobustParCorr):
    """
    Custom implementation of RobustParCorr that enforces forward-in-time conditioning sets
    and handles the "degrees of freedom <= 0" warning.
    """
    def run_test(self, X, Y, Z, tau_max=None, **kwargs):
        if len(X) != 1 or len(Y) != 1:
            pass 

        tau_x = X[0][1] 
        Z_filtered = [z_item for z_item in Z if z_item[1] <= tau_x]

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Degrees of freedom <= 0 for slice", 
                category=RuntimeWarning
            )
            warnings.filterwarnings(
                "ignore",
                message="invalid value encountered in scalar divide",
                category=RuntimeWarning
            )
            return super().run_test(X, Y, Z_filtered, tau_max=tau_max, **kwargs)

def get_season_from_lag(lag_val_func: int) -> Tuple[str, str]:
    """Determine season based on lag value for REOF/MCA variable assignment."""
    if lag_val_func == 6: return "JJA", "JJA_PREDICTORS"
    elif lag_val_func == 5: return "SON", "SON_PREDICTORS"
    elif lag_val_func == 4: return "DJF", "DJF_CONFOUNDERS" 
    elif lag_val_func == 3: return "MAM", "MAM_MEDIATORS"   
    else:
        logging.warning(f"Unsupported lag value for REOF/MCA season assignment: {lag_val_func}. Using SON as default.")
        return "SON", "SON_PREDICTORS"

mca_season, mca_group_name = "SON", "SON_PREDICTORS" 
if target_lag is not None:
    mca_season, mca_group_name = get_season_from_lag(target_lag)
    logging.info(f"Using {mca_season} season for REOF variables (assigned to group {mca_group_name}) based on target_lag: {target_lag}")
else:
    logging.warning("No target_lag found or parsed. Using SON as default season for REOF variables.")

VARIABLE_SEASONS = {
    'REOF SST JJA': 'JJA', 'DMI JJA': 'JJA', 
    'MCA WAF-RWS SON': 'SON', 'MCA prec-RWS SON': 'SON', 'SIOD MAM': 'MAM',
    'SASDI_DJF_0': 'DJF', 'Atl3_DJF_0': 'DJF',
    'NPMM_SST_DJF_0': 'DJF', 'WNP_DJF_0': 'DJF',
    'NPMM_wind_MAM': 'MAM', 'NTA_MAM': 'MAM', 'SPO_MAM': 'MAM', 
    'SPMM_SST_MAM': 'MAM', 'SPMM_wind_MAM': 'MAM',
    'E-ind DJF(1)': 'DJF_effect', 'C-ind DJF(1)': 'DJF_effect',
} # , 'MCA2 prec-RWS SON': 'SON',

# Dynamically add MCA variables with the correct season
VARIABLE_SEASONS[f'MCA1_RWS'] = mca_season
VARIABLE_SEASONS[f'MCA2_RWS'] = mca_season
VARIABLE_SEASONS[f'MCA1_WAF'] = mca_season
VARIABLE_SEASONS[f'MCA2_WAF'] = mca_season

ALL_VARIABLES = list(VARIABLE_SEASONS.keys())

# Variable groups with dynamic MCA assignment
JJA_PREDICTORS = ['REOF SST JJA', 'DMI JJA']
SON_PREDICTORS = ['MCA WAF-RWS SON', 'MCA prec-RWS SON', 'SIOD MAM']
DJF_CONFOUNDERS = ['Atl3_DJF_0', 'SASDI_DJF_0', 'NPMM_SST_DJF_0', 'WNP_DJF_0']
MAM_MEDIATORS = ['NPMM_wind_MAM', 'NTA_MAM', 'SPO_MAM', 'SPMM-SST_MAM', 'SPMM-wind_MAM']
DJF_EFFECTS = ['E-ind DJF(1)', 'C-ind DJF(1)']

# Add MCA variables to the appropriate group based on lag
mca_variables = ['MCA1_RWS', 'MCA2_RWS', 'MCA1_WAF', 'MCA2_WAF']
if mca_group_name == "JJA_PREDICTORS":
    JJA_PREDICTORS.extend(mca_variables)
elif mca_group_name == "SON_PREDICTORS":
    SON_PREDICTORS.extend(mca_variables)
elif mca_group_name == "DJF_CONFOUNDERS":
    DJF_CONFOUNDERS.extend(mca_variables)
elif mca_group_name == "MAM_MEDIATORS":
    MAM_MEDIATORS.extend(mca_variables)

# Season to month mapping
SEASON_MONTHS = {
    'JJA': [7],
    'SON': [10],
    'DJF': [1],
    'MAM': [4],
    'DJF_effect': [1],
}

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_prepare_data(file_path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load data from netCDF, apply seasonal masks, and convert to numpy."""
    try:
        data_xr = xr.open_dataset(file_path)
    except FileNotFoundError:
        logging.error(f"Data file not found: {file_path}")
        raise 
    
    missing_vars = [var for var in ALL_VARIABLES if var not in data_xr.data_vars]
    if missing_vars:
        logging.error(f"Missing variables in dataset {file_path}: {missing_vars}")
        raise ValueError(f"Dataset {file_path} is missing required variables: {missing_vars}")

    data_xr = data_xr.fillna(0) 
    time_length = len(data_xr.time)
    num_variables = len(ALL_VARIABLES)

    data_array = np.zeros((time_length, num_variables))
    mask_array = np.zeros((time_length, num_variables), dtype=bool) 

    for idx, var_name in enumerate(ALL_VARIABLES):
        season = VARIABLE_SEASONS[var_name]
        months_for_season = SEASON_MONTHS[season]
        
        is_in_season_months = data_xr.time.dt.month.isin(months_for_season)
        
        time_indices = pd.RangeIndex(start=0, stop=time_length)
        exclusion_mask = pd.Series(True, index=time_indices) 

        if var_name in JJA_PREDICTORS and time_length >= 3: exclusion_mask.iloc[-3] = False 
        if var_name in SON_PREDICTORS and time_length >= 2: exclusion_mask.iloc[-2] = False 
        if var_name in DJF_CONFOUNDERS and time_length >= 1: exclusion_mask.iloc[-1] = False 
        if var_name in DJF_EFFECTS and time_length >=2: 
             exclusion_mask.iloc[1] = False 

        final_keep_mask = is_in_season_months.values & exclusion_mask.values
        
        data_array[:, idx] = data_xr[var_name].data
        mask_array[:, idx] = ~final_keep_mask 

        effective_samples = np.sum(final_keep_mask)
        if effective_samples <= MAX_COMBINATIONS + 2 : 
            logging.warning(f"Variable {var_name} has only {effective_samples} effective samples. "
                           f"Max_combinations={MAX_COMBINATIONS}. May cause DoF issues.")
    return data_array, mask_array, ALL_VARIABLES


def run_pcmciplus_iterative(dataframe: pp.DataFrame,
                            var_names: List[str],
                            ind_test_cls, 
                            alpha_seq: List[float]) -> List[Dict]:
    """
    Run PCMCI+ multiple times with different alpha values using a specific independence test.
    """
    results_list = []
    for alpha in alpha_seq:
        logging.info(f"Running PCMCI+ with {ind_test_cls.__name__}, alpha = {alpha}")
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=ind_test_cls(significance='analytic', mask_type='y'), 
            verbosity=0 
        )
        
        current_results = pcmci.run_pcmciplus(
            tau_min=TAU_MIN,
            tau_max=TAU_MAX,
            pc_alpha=alpha,
            max_combinations=MAX_COMBINATIONS
        )
        results_list.append(current_results)
        logging.info(f"Completed PCMCI+ run with {ind_test_cls.__name__}, alpha = {alpha}")
    return results_list

def track_edge_significance(results_list: List[Dict], var_names: List[str], 
                          alpha_values: List[float]) -> pd.DataFrame:
    """
    Track edge significance across multiple PCMCI+ runs.

    Args:
        results_list: List of PCMCI+ results for different alpha values
        var_names: List of variable names
        alpha_values: List of alpha values used in each run

    Returns:
        DataFrame containing edge persistence information
    """
    edge_records = []
    
    for run_idx, (results, alpha) in enumerate(zip(results_list, alpha_values)):
        val_matrix = results['val_matrix']
        p_matrix = results['p_matrix']
        graph = results['graph']
        
        for i in range(len(var_names)):
            for j in range(len(var_names)):
                for tau in range(TAU_MIN, TAU_MAX + 1):
                    if graph[i, j, tau] != '':
                        edge_records.append({
                            'cause': var_names[i],
                            'effect': var_names[j],
                            'lag': tau,
                            'alpha': alpha,
                            'causal_effect': val_matrix[i, j, tau],
                            'p_value': p_matrix[i, j, tau]
                        })
    
    return pd.DataFrame(edge_records)

def analyze_persistence_single(par_df: pd.DataFrame, n_alphas: int) -> pd.DataFrame:
    """
    Analyze persistence for a single test (ParCorr only).
    Input:
        par_df: DataFrame with columns ['cause','effect','lag','alpha','causal_effect','p_value']
        n_alphas: number of alpha levels (len(alpha_seq))
    Output columns (one row per unique (cause,effect,lag)):
        - persistence_ratio = count(rows)/n_alphas
        - avg_effect, abs_effect = mean(causal_effect), abs(mean)
        - avg_p_value = mean(p_value)
        - effect_consistency = fraction of runs whose sign(causal_effect) == sign(avg_effect) (tie-break zeros)
        - is_persistent = persistence_ratio ≥ THR.min_persistence
        - has_low_pvalue = avg_p_value ≤ THR.max_avg_p
        - has_strong_effect = abs_effect ≥ THR.min_effect
        - is_consistent = effect_consistency ≥ THR.min_direction
        - is_robust = all four flags above True
    """
    if par_df.empty:
        logging.warning("Edge DataFrame for persistence analysis is empty.")
        return pd.DataFrame(columns=[
            'cause', 'effect', 'lag', 'persistence_ratio', 'avg_effect',
            'abs_effect', 'avg_p_value', 'effect_consistency',
            'is_persistent', 'has_low_pvalue', 'is_consistent',
            'has_strong_effect', 'is_robust'
        ])

    edge_groups = par_df.groupby(['cause', 'effect', 'lag'])
    persistence_records = []

    for (cause, effect, lag), group in edge_groups:
        num_significant_at_alpha = len(group)
        persistence_ratio = num_significant_at_alpha / n_alphas
        avg_effect = group['causal_effect'].mean() if num_significant_at_alpha > 0 else 0.0
        avg_p_value = group['p_value'].mean() if num_significant_at_alpha > 0 else 1.0
        abs_effect = np.abs(avg_effect)

        effect_consistency = 0.0
        if num_significant_at_alpha > 0:
            effect_directions = np.sign(group['causal_effect'])
            dominant_direction = np.sign(avg_effect)
            if dominant_direction == 0 and len(group['causal_effect']) > 0:
                if not np.all(group['causal_effect'] == 0):
                    effect_consistency = 0.0
                else:
                    effect_consistency = 1.0
            elif dominant_direction != 0:
                effect_consistency = np.mean(effect_directions == dominant_direction)

        is_persistent = persistence_ratio >= THR.min_persistence
        has_low_pvalue = avg_p_value <= THR.max_avg_p
        has_strong_effect = abs_effect >= THR.min_effect
        is_consistent = effect_consistency >= THR.min_direction
        is_robust = is_persistent and has_low_pvalue and has_strong_effect and is_consistent

        persistence_records.append({
            'cause': cause, 'effect': effect, 'lag': lag,
            'persistence_ratio': persistence_ratio,
            'avg_effect': avg_effect,
            'abs_effect': abs_effect,
            'avg_p_value': avg_p_value,
            'effect_consistency': effect_consistency,
            'is_persistent': is_persistent,
            'has_low_pvalue': has_low_pvalue,
            'has_strong_effect': has_strong_effect,
            'is_consistent': is_consistent,
            'is_robust': is_robust
        })
    return pd.DataFrame(persistence_records)

def create_robust_graph(persistence_df: pd.DataFrame,
                        original_shape: tuple, 
                        var_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a new graph containing only robust edges. Values from persistence_df (ParCorr based)."""
    robust_graph = np.full(original_shape, '', dtype=object)
    robust_val_matrix = np.zeros(original_shape)
    robust_p_matrix = np.ones(original_shape) 

    robust_edges = persistence_df[persistence_df['is_robust']]
    for _, edge in robust_edges.iterrows():
        try:
            i = var_names.index(edge['cause'])
            j = var_names.index(edge['effect'])
            tau_idx = int(edge['lag']) 

            if 0 <= i < original_shape[0] and \
               0 <= j < original_shape[1] and \
               0 <= tau_idx < original_shape[2]: 
                robust_graph[i, j, tau_idx] = '-->' 
                robust_val_matrix[i, j, tau_idx] = edge['avg_effect']
                robust_p_matrix[i, j, tau_idx] = edge['avg_p_value']
            else:
                logging.warning(f"Index out of bounds when creating robust graph for edge: {edge}. Skipping.")
        except ValueError: 
            logging.warning(f"Variable name not found in var_names list for edge: {edge}. Skipping.")
        except IndexError:
            logging.warning(f"IndexError while populating robust graph for edge: {edge}. Skipping.")
    return robust_graph, robust_val_matrix, robust_p_matrix

def filter_graph(graph: np.ndarray, val_matrix: np.ndarray, p_matrix: np.ndarray,
                var_names: List[str], tau_min_filt: int, tau_max_filt: int,
                max_cumulative_lag: int = TAU_MAX) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Filters graph: only pathways to E_ind_DJF_1 or C_ind_DJF_1 within max_cumulative_lag."""
    targets = ['E-ind DJF(1)', 'C-ind DJF(1)']
    try:
        target_indices = [var_names.index(t) for t in targets if t in var_names]
    except ValueError as e:
        logging.error(f"Target variable not found in var_names: {e}")
        return (np.full_like(graph, '', dtype=object),
                np.zeros_like(val_matrix),
                np.ones_like(p_matrix))

    if not target_indices:
        logging.warning("No target variables found for filtering. Returning original graph.")
        return graph, val_matrix, p_matrix

    links_to_keep = set() 
    for target_idx in target_indices:
        queue: List[Tuple[int, int]] = [(target_idx, 0)]
        visited_for_path = set() 

        while queue:
            current_var_idx, cum_lag = queue.pop(0)
            if (current_var_idx, cum_lag) in visited_for_path and cum_lag > 0 : 
                continue
            visited_for_path.add((current_var_idx, cum_lag))

            for source_var_idx in range(len(var_names)):
                for lag_val in range(tau_min_filt, tau_max_filt + 1):
                    if graph[source_var_idx, current_var_idx, lag_val] != '':
                        new_cum_lag = cum_lag + lag_val
                        if new_cum_lag <= max_cumulative_lag:
                            links_to_keep.add((source_var_idx, current_var_idx, lag_val))
                            if source_var_idx not in target_indices: 
                                queue.append((source_var_idx, new_cum_lag))
    
    new_graph = np.full_like(graph, '', dtype=object)
    new_val_matrix = np.zeros_like(val_matrix)
    new_p_matrix = np.ones_like(p_matrix)

    for i, j, tau_link in links_to_keep:
        new_graph[i, j, tau_link] = graph[i, j, tau_link]
        new_val_matrix[i, j, tau_link] = val_matrix[i, j, tau_link]
        new_p_matrix[i, j, tau_link] = p_matrix[i, j, tau_link]
        
    return new_graph, new_val_matrix, new_p_matrix

def create_causal_graph_visualizations(graph_matrix: np.ndarray, 
                                       value_matrix: np.ndarray, 
                                       variable_names_list: List[str],
                                       save_path_dir: str, 
                                       file_suffix: str): 
    """Creates and saves causal graph and time series graph visualizations."""
    node_positions = get_node_positions(variable_names_list) 

    plt.figure(figsize=(30, 25)) 
    tp.plot_graph(
        val_matrix=value_matrix,
        graph=graph_matrix,
        var_names=[f"{var}" for var in variable_names_list], 
        link_colorbar_label='Robust Causal Effect (ParCorr based)',
        figsize=(25, 20), 
        node_colorbar_label='Autocorrelation (not shown if all zero)',
        node_pos=node_positions,
        node_size=0.15, 
        alpha=0.7,      
        node_label_size=18,
        arrow_linewidth=20, 
        label_fontsize=18,
        link_label_fontsize=18,
        curved_radius=0.2 
    )
    plt.tight_layout()
    plot_filename_network = os.path.join(save_path_dir, f'causal_graph_{file_suffix}_gen{generation}_ind{individual}.png')
    plt.savefig(plot_filename_network, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved causal network graph to {plot_filename_network}")

    plt.figure(figsize=(20, 20))
    tp.plot_time_series_graph(
        graph=graph_matrix,
        val_matrix=value_matrix,
        var_names=variable_names_list,
        link_colorbar_label='Robust Causal Effect (ParCorr based)',
        node_size=0.05, 
        arrow_linewidth=10,
        label_fontsize=22,
        figsize=(20,20)
    )
    plt.tight_layout()
    plot_filename_ts = os.path.join(save_path_dir, f'ts_graph_{file_suffix}_gen{generation}_ind{individual}.png')
    plt.savefig(plot_filename_ts, dpi=300, bbox_inches='tight')
    plt.close()
    logging.info(f"Saved time series graph to {plot_filename_ts}")


def get_node_positions(var_names_node: List[str]) -> Dict[str, List[float]]:
    """Calculate node positions for graph visualization based on season."""
    season_y_positions = {'JJA': 0.9, 'SON': 0.7, 'DJF': 0.5, 'MAM': 0.3, 'DJF_effect': 0.1}
    x_coords, y_coords = [], []
    
    season_counts = {season: 0 for season in season_y_positions}
    season_totals = {season: sum(1 for v_name in var_names_node if VARIABLE_SEASONS.get(v_name) == season)
                     for season in season_y_positions}

    for var in var_names_node:
        season = VARIABLE_SEASONS.get(var, 'DJF_effect') 
        y = season_y_positions.get(season, 0.0) 

        current_count_in_season = season_counts[season]
        total_in_season = season_totals.get(season, 1)
        
        if total_in_season <= 1:
            x = 0.5
        else:
            x = current_count_in_season / (total_in_season - 1) if total_in_season > 1 else 0.5
        
        x_coords.append(x)
        y_coords.append(y)
        season_counts[season] += 1
        
    return {'x': x_coords, 'y': y_coords}

def save_persistence_results_to_csv(persistence_summary_df: pd.DataFrame,
                                    path_to_save: str, 
                                    filename_suffix: str):
    """Save edge persistence analysis results and a summary."""
    if persistence_summary_df.empty:
        logging.warning("Persistence DataFrame is empty. Skipping CSV save.")
        return

    persistence_summary_df.to_csv(
        os.path.join(path_to_save, f'edge_persistence_summary_{filename_suffix}.csv'),
        index=False
    )
    
    robust_edges_df = persistence_summary_df[persistence_summary_df['is_robust']].sort_values(
        by=['abs_effect', 'persistence_ratio'], ascending=[False, False] 
    )
    robust_edges_df.to_csv(
        os.path.join(path_to_save, f'robust_edges_details_{filename_suffix}.csv'),
        index=False
    )
    
    summary_stats = {
        'total_unique_edges_considered': len(persistence_summary_df),
        'robust_edges_found': len(robust_edges_df),
        'avg_persistence_ratio_all': persistence_summary_df['persistence_ratio'].mean() if not persistence_summary_df.empty else 0,
        'median_persistence_ratio_all': persistence_summary_df['persistence_ratio'].median() if not persistence_summary_df.empty else 0,
        'avg_persistence_ratio_robust': robust_edges_df['persistence_ratio'].mean() if not robust_edges_df.empty else 0,
    }
    with open(os.path.join(path_to_save, f'persistence_stats_{filename_suffix}.json'), 'w') as f:
        json.dump(summary_stats, f, indent=4)
    logging.info(f"Saved persistence results and stats with suffix '{filename_suffix}'.")


def write_final_graph_to_csv(graph_to_write: np.ndarray, 
                             val_matrix_to_write: np.ndarray,
                             p_matrix_to_write: np.ndarray,
                             var_names_list: List[str],
                             full_save_path: str, 
                             digits: int = 4):
    """Write the details of the (filtered, robust) causal graph to a CSV file."""
    rows = []
    for i in range(len(var_names_list)): 
        for j in range(len(var_names_list)): 
            for lag_val_csv in range(TAU_MIN, TAU_MAX + 1): 
                if graph_to_write[i, j, lag_val_csv] != '':
                    rows.append({
                        'Cause': var_names_list[i],
                        'Effect': var_names_list[j],
                        'Lag': lag_val_csv,
                        'Causal_Effect': round(val_matrix_to_write[i, j, lag_val_csv], digits),
                        'p_value': round(p_matrix_to_write[i, j, lag_val_csv], digits)
                    })
    df_to_save = pd.DataFrame(rows)
    df_to_save.to_csv(full_save_path, index=False)
    logging.info(f"Saved final graph details to {full_save_path}")


def main() -> None:
    """
    Main function that orchestrates the entire analysis pipeline.
    """
    logging.info("Running single-test pipeline: RobustParCorr only.")
    
    data_file_path = os.path.join(target_dir_results_base, f'ds_caus_MCA_40_24_gen{generation}_ind{individual}.nc')

    logging.info(f"Loading data from {data_file_path}")
    try:
        data_array, mask_array, var_names = load_and_prepare_data(data_file_path)
    except (FileNotFoundError, ValueError) as e: 
        logging.error(f"Failed to load or prepare data: {e}. Exiting.")
        empty_df = pd.DataFrame(columns=['Cause', 'Effect', 'Lag', 'Causal_Effect', 'p_value'])
        error_csv_path = os.path.join(PATH_SAVE, f'filtered_robust_graph_{SUFFIX}.csv')
        empty_df.to_csv(error_csv_path, index=False)
        logging.info(f"Created empty results CSV at {error_csv_path} due to data loading error.")
        sys.exit(1) 
    
    # Create Tigramite dataframe
    dataframe = pp.DataFrame(
        data_array,
        datatime=np.arange(len(data_array)),
        var_names=var_names,
        mask=mask_array
    )
    logging.info("Created Tigramite dataframe")
    
    alpha_seq = [0.1, 0.05, 0.025, 0.01] 
    logging.info(f"Using alpha sequence for PCMCI+: {alpha_seq}")

    results_parcorr = run_pcmciplus_iterative(dataframe, var_names, CustomRobustParCorr, alpha_seq)
    edges_parcorr_df = track_edge_significance(results_parcorr, var_names, alpha_seq)
    logging.info(f"Tracked {len(edges_parcorr_df)} edges from RobustParCorr runs.")

    # Sanity check: if no edges found, exit with empty CSV
    if edges_parcorr_df.empty:
        logging.warning("No edges found from RobustParCorr runs; exiting with empty output.")
        empty_df = pd.DataFrame(columns=['Cause', 'Effect', 'Lag', 'Causal_Effect', 'p_value'])
        empty_csv_path = os.path.join(PATH_SAVE, f'filtered_robust_graph_{SUFFIX}.csv')
        empty_df.to_csv(empty_csv_path, index=False)
        logging.info(f"Created empty results CSV at {empty_csv_path} due to no edges.")
        sys.exit(0)
    
    # Pass the actual number of alpha levels to analyze_edge_persistence
    persistence_df = analyze_persistence_single(edges_parcorr_df, len(alpha_seq))
    logging.info(f"Persistence analysis complete. Found {len(persistence_df[persistence_df['is_robust']])} robust edges.")

    save_persistence_results_to_csv(persistence_df, PATH_SAVE, SUFFIX)

    original_graph_shape = results_parcorr[0]['graph'].shape if results_parcorr else (len(var_names), len(var_names), TAU_MAX + 1)

    robust_graph, robust_val_matrix, robust_p_matrix = create_robust_graph(
        persistence_df, original_graph_shape, var_names
    )
    logging.info("Created robust graph matrices (ParCorr based values).")

    filtered_robust_graph, filtered_val_matrix, filtered_p_matrix = filter_graph(
        robust_graph, robust_val_matrix, robust_p_matrix,
        var_names, TAU_MIN, TAU_MAX, 
        max_cumulative_lag=TAU_MAX
    )
    logging.info("Applied pathway filtering to the robust graph.")

    create_causal_graph_visualizations(
        filtered_robust_graph, filtered_val_matrix, var_names,
        PATH_SAVE, f"{SUFFIX}_filtered_robust" 
    )

    final_csv_path = os.path.join(PATH_SAVE, f'filtered_robust_graph_{SUFFIX}.csv')
    write_final_graph_to_csv(
        filtered_robust_graph, filtered_val_matrix, filtered_p_matrix,
        var_names, final_csv_path
    )
    
    logging.info("PCMCI+ analysis pipeline completed successfully.")

if __name__ == "__main__":
    try:
        main()
        sys.exit(0) 
    except Exception as e:
        logging.error(f"Error in main execution of script 16: {e}", exc_info=True)
        try:
            if 'PATH_SAVE' in globals() and 'SUFFIX' in globals() and PATH_SAVE and SUFFIX : # Check if these are defined
                 empty_df = pd.DataFrame(columns=['Cause', 'Effect', 'Lag', 'Causal_Effect', 'p_value'])
                 error_csv_path = os.path.join(PATH_SAVE, f'filtered_robust_graph_{SUFFIX}.csv')
                 if not os.path.exists(error_csv_path): 
                     empty_df.to_csv(error_csv_path, index=False)
                     logging.info(f"Created empty results CSV at {error_csv_path} due to main execution error.")
        except Exception as e_csv:
            logging.error(f"Failed to write empty CSV on error: {e_csv}")
        sys.exit(1)