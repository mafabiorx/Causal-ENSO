"""
PCMCI+ Sliding Window Analysis for Climate Time Series

This script performs sliding window causal discovery analysis on climate time series
data using the PCMCI+ algorithm. It analyzes temporal evolution of causal relationships
and creates consensus graphs based on link frequency across windows.

Key features:
  - Sliding window analysis with configurable window lengths and steps
  - Consensus graph creation based on link frequency
  - Temporal evolution visualization of causal relationships
  - Sensitivity analysis across multiple window configurations
  - Integration with pruned variable sets from robust graph analysis

Scientific context:
This analysis investigates temporal dynamics of causal pathways linking the South
American Monsoon System (SAMS) to ENSO development, providing insights into how
causal relationships evolve over different time periods.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.max_open_warning'] = 0
import gc
import math

from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite import plotting as tp
from tigramite.independence_tests.robust_parcorr import RobustParCorr

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PATH SETUP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sys.path.append(str(Path(__file__).parent.parent))

# Import utilities
try:
    from utils.paths import get_data_path, get_results_path
    from utils.predictor_configs import (
        PredictorConfig, 
        get_available_predictor_sets,
        load_custom_config_from_file
    )
    from utils.causal_discovery_utils import (
        load_and_prepare_data,
        load_pruned_variables,
        reduce_variable_lists,
        calculate_effective_samples,
        get_frequency_threshold,
        filter_graph,
        create_custom_test_class,
        create_causal_graph_w_map
    )
    from utils.plotting_optimization import (
        save_figure_optimized, add_plotting_arguments,
        filter_seasons_to_plot, create_clean_panel_title,
        create_descriptive_filename
    )
except ImportError as err:
    logging.error(f"Failed to import required modules: {err}")
    raise RuntimeError("Could not import utils. Ensure PYTHONPATH includes project root.") from err

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ GLOBAL PARAMETERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analysis parameters
TAU_MIN, TAU_MAX = 1, 6
MAX_COMBINATIONS = 10
PC_ALPHA = 0.05
ALPHA_LEVEL = 0.05

# Window parameters (can be overridden by CLI)
WIN_LEN = 35  # Smooth sliding window length (years)
WIN_STEP = 5  # Smooth sliding window step (years)
LEN_STEP = 38  # Period analysis window (years)

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ COMMAND LINE INTERFACE ━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for sliding window analysis configuration.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        prog='13_PCMCIplus_sliding_window.py',
        description='PCMCI+ Sliding Window Analysis with Configurable Predictor Sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available predictor sets:
  known     : Well-established ENSO precursors from literature
  new       : Novel predictive modes from this research
  combined  : All available predictors (default)
  custom    : User-defined configuration from JSON file

FDR loading options:
  --load-fdr             Load FDR-pruned results from causal discovery analysis
  --fdr-pc-alpha 0.05    Specify the FDR threshold used (default: 0.10)

Examples:
  python 13_PCMCIplus_sliding_window.py --predictor-set known
  python 13_PCMCIplus_sliding_window.py --window-configs 20,10 30,15 40,20
  python 13_PCMCIplus_sliding_window.py --predictor-set custom --config my_predictors.json
  python 13_PCMCIplus_sliding_window.py --suffix PCMCI_hard_combined --load-fdr --fdr-pc-alpha 0.01
  python 13_PCMCIplus_sliding_window.py --suffix PCMCI_mild_known --load-fdr --fdr-pc-alpha 0.05
        """
    )
    
    # Predictor configuration
    parser.add_argument(
        '-p', '--predictor-set',
        choices=get_available_predictor_sets(),
        default='combined',
        help='Predictor set to use for analysis (default: combined)'
    )
    parser.add_argument(
        '-c', '--config',
        type=str,
        help='Path to custom predictor configuration JSON (required with --predictor-set=custom)'
    )
    
    # Window configuration
    parser.add_argument(
        '--window-configs',
        nargs='+',
        default=['20,10', '30,15', '40,20'],
        help='Window configurations as "length,step" in years (default: 20,10 30,15 40,20)'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--pc-alpha',
        type=float,
        default=PC_ALPHA,
        help=f'PC alpha value for conditional independence testing (default: {PC_ALPHA})'
    )
    parser.add_argument(
        '--alpha-level',
        type=float,
        default=ALPHA_LEVEL,
        help=f'General significance level (default: {ALPHA_LEVEL})'
    )
    parser.add_argument(
        '--max-combinations',
        type=int,
        default=MAX_COMBINATIONS,
        help=f'Maximum conditioning set combinations (default: {MAX_COMBINATIONS})'
    )
    
    # Data configuration
    parser.add_argument(
        '--data-file',
        default='PCMCI_data_ts_st.nc',
        help='Input data file name (default: PCMCI_data_ts_st.nc)'
    )
    parser.add_argument(
        '--suffix',
        default='PCMCI_hard_combined',
        help='Base suffix for input/output files'
    )
    
    # FDR loading configuration
    parser.add_argument(
        '--load-fdr',
        action='store_true',
        help='Load FDR-pruned results from causal discovery analysis (default: load standard robust filtered results)'
    )
    parser.add_argument(
        '--fdr-pc-alpha',
        type=float,
        default=0.10,
        help='FDR pc_alpha value used in causal discovery analysis (default: 0.10, only used with --load-fdr)'
    )
    
    # Temporal evolution configuration
    parser.add_argument(
        '--temporal-configs',
        nargs='+',
        default=[f'{WIN_LEN},{WIN_STEP}', f'{LEN_STEP},{LEN_STEP}'],
        help=f'Temporal evolution window configs as "length,step" (default: {WIN_LEN},{WIN_STEP} {LEN_STEP},{LEN_STEP})'
    )
    
    # Add standard plotting optimization arguments
    add_plotting_arguments(parser)
    
    return parser.parse_args()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ SLIDING WINDOW FUNCTIONS ━━━━━━━━━━━━━━━━━━━━━━━━━━



def plot_sliding_window_results(
    results: Dict,
    var_names: List[str],
    var_seasons_mapping: Dict[str, str],
    path_save: str,
    suffix: str = 'primary',
    predictor_set: str = 'combined',
    cond_ind_test: str = 'robust_parcorr',
    output_format: str = 'png'
) -> None:
    """
    Plot causal graphs for each window in the sliding window results.

    Args:
        results: Dictionary from pcmci.run_sliding_window_of
        var_names: List of variable names used in the analysis
        var_seasons_mapping: Dictionary mapping variable names to seasons
        path_save: Directory to save results
        suffix: Base suffix for the output files
        predictor_set: Predictor set used for analysis (for node positioning)
        cond_ind_test: Conditional independence test used
    """
    if 'window_results' not in results or not results['window_results']:
        logging.warning(f"No 'window_results' found in results for suffix {suffix}. Skipping plotting.")
        return

    graphs = results['window_results'].get('graph')
    val_matrices = results['window_results'].get('val_matrix')
    p_matrices = results['window_results'].get('p_matrix')

    if graphs is None or val_matrices is None or p_matrices is None:
        logging.warning(f"Missing matrices in window_results for {suffix}. Skipping plotting.")
        return

    n_windows = min(len(graphs), len(val_matrices), len(p_matrices))
    logging.info(f"Plotting results for {n_windows} windows for configuration {suffix}...")

    for w in range(n_windows):
        graph = graphs[w]
        val_matrix = val_matrices[w]
        p_matrix = p_matrices[w]

        window_suffix = f"{suffix}_window_{w}"

        try:
            # Filter the graph for this window
            filtered_graph, filtered_val_matrix, _ = filter_graph(
                graph,
                val_matrix,
                p_matrix,
                var_names,
                target_variables=['E-ind DJF(1)', 'C-ind DJF(1)'],
                tau_min=TAU_MIN,
                tau_max=TAU_MAX,
                max_cumulative_lag=TAU_MAX
            )
            logging.debug(f"Applied filtering to graph for window {w} of {suffix}")

            # Plot the filtered graph for this window using map-based visualization
            create_causal_graph_w_map(
                graph=filtered_graph,
                val_matrix=filtered_val_matrix,
                var_names=var_names,
                save_dir=path_save,
                suffix=window_suffix,
                predictor_set=predictor_set,
                cond_ind_test=cond_ind_test
            )
            
            # Also create time series graph for comparison
            plt.figure(figsize=(25, 20))
            try:
                tp.plot_time_series_graph(
                    val_matrix=filtered_val_matrix,
                    graph=filtered_graph,
                    var_names=[f"{var}" for var in var_names],
                    link_colorbar_label='Robust partial correlation',
                    figsize=(25, 20),
                )
                plt.tight_layout()
                ts_save_path = os.path.join(path_save, f'ts_graph_{window_suffix}')
                save_figure_optimized(plt.gcf(), ts_save_path, output_format)
                logging.info(f"Saved time series graph to {ts_save_path}")
            except Exception as e:
                logging.error(f"Error plotting time series graph for {window_suffix}: {e}", exc_info=True)
            finally:
                plt.close()
                # Memory management for multiple windows
                if 'gc' in sys.modules:
                    gc.collect()
        except Exception as e:
            logging.error(f"Error processing or plotting window {w} for {suffix}: {e}", exc_info=True)

    logging.info(f"Finished plotting individual window results for {suffix}.")


def create_consensus_graph(
    results: Dict,
    var_names: List[str],
    window_length_years: int,
    path_save: str
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Create a consensus graph based on frequency of links across sliding windows.

    Args:
        results: Results dictionary from PCMCI sliding window analysis
        var_names: List of variable names used in the analysis
        window_length_years: Length of the window in years (used for threshold)
        path_save: Directory to save frequency results

    Returns:
        Tuple: (consensus_graph, consensus_val_matrix, link_width_matrix)
               Returns (None, None, None) if results are invalid
    """
    logging.info(f"Creating consensus graph using frequency method for window length {window_length_years}y...")
    
    if 'window_results' not in results or not results['window_results']:
        logging.warning("No 'window_results' found in results. Cannot create consensus graph.")
        return None, None, None

    graphs = results['window_results'].get('graph')
    val_matrices = results['window_results'].get('val_matrix')

    # Validate input structure
    if graphs is None or val_matrices is None:
        logging.warning("Missing 'graph' or 'val_matrix' in window_results. Cannot create consensus graph.")
        return None, None, None

    if not isinstance(graphs, np.ndarray) or not isinstance(val_matrices, np.ndarray):
        logging.warning("'graph' or 'val_matrix' not in numpy array format. Cannot create consensus graph.")
        return None, None, None

    if graphs.size == 0 or val_matrices.size == 0:
        logging.warning("Empty graph or val_matrix arrays. Cannot create consensus graph.")
        return None, None, None

    # Get number of windows from first dimension
    if graphs.shape[0] != val_matrices.shape[0]:
        logging.warning("Inconsistent number of windows between graph and val_matrix. Cannot create consensus graph.")
        return None, None, None
    n_windows = graphs.shape[0]
    if n_windows == 0:
        logging.warning("Zero valid windows found. Cannot create consensus graph.")
        return None, None, None

    # Get dimensions from the 4D arrays
    try:
        # Extract dimensions from the 4D array
        if graphs.ndim != 4 or val_matrices.ndim != 4:
            raise ValueError(f"Expected 4D arrays, got shapes: graphs={graphs.shape}, val_matrices={val_matrices.shape}")
        
        _, n_vars, n_vars2, n_lags = graphs.shape
        if n_vars != n_vars2:
            raise ValueError(f"Non-square matrix dimension: {n_vars} != {n_vars2}")
        
        # Check against expected dimensions
        expected_per_window_shape = (len(var_names), len(var_names), TAU_MAX + 1)
        actual_per_window_shape = (n_vars, n_vars, n_lags)
        if actual_per_window_shape != expected_per_window_shape:
            raise ValueError(f"Unexpected per-window matrix shape. Expected {expected_per_window_shape}, got {actual_per_window_shape}")
    except (IndexError, AttributeError, ValueError, TypeError) as e:
        logging.error(f"Could not determine valid dimensions from results: {e}")
        return None, None, None

    # Initialize counters and value accumulators
    link_counts = np.zeros((n_vars, n_vars, n_lags), dtype=int)
    value_sums = np.zeros((n_vars, n_vars, n_lags), dtype=float)

    # Count occurrences and sum values across windows
    valid_windows_counted = 0
    for w in range(n_windows):
        try:
            # Extract window w from the 4D arrays
            graph_w = graphs[w]
            val_matrix_w = val_matrices[w]
            
            # Count where links exist (non-empty strings in graph)
            link_present = (graph_w != '')
            link_counts += link_present.astype(int)
            # Sum values where links exist, handle potential NaNs
            value_sums += np.where(link_present, np.nan_to_num(val_matrix_w), 0)
            valid_windows_counted += 1
        except Exception as e:
            logging.warning(f"Error processing window {w}: {e}")

    if valid_windows_counted == 0:
        logging.warning("No windows with consistent shapes/types found. Cannot create consensus graph.")
        return None, None, None

    logging.info(f"Aggregated results from {valid_windows_counted} valid windows.")

    # Calculate frequencies
    link_frequencies = link_counts / valid_windows_counted

    # Determine frequency threshold based on window length
    frequency_threshold = get_frequency_threshold(window_length_years)

    # Create consensus graph based on frequency threshold
    consensus_shape = (n_vars, n_vars, n_lags)
    consensus_graph = np.full(consensus_shape, '', dtype=object)
    frequent_links = (link_frequencies >= frequency_threshold)
    consensus_graph[frequent_links] = '-->'    

    # Calculate average values ONLY for links that appeared at least once
    consensus_val_matrix = np.zeros(consensus_shape, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_values = np.where(link_counts > 0, value_sums / link_counts, 0)
    consensus_val_matrix[frequent_links] = avg_values[frequent_links]

    # Calculate link power as product of frequency and absolute effect
    link_power = np.zeros_like(consensus_val_matrix)
    link_power[frequent_links] = link_frequencies[frequent_links] * np.abs(consensus_val_matrix[frequent_links])

    # Normalize link_power for link width (0 to 1)
    max_power = np.nanmax(link_power)
    if max_power > 0:
        link_width_matrix = np.nan_to_num(link_power / max_power)
    else:
        link_width_matrix = np.zeros_like(link_power)

    logging.info(f"Consensus graph created with {np.sum(frequent_links)} links meeting threshold {frequency_threshold}.")

    # Save link frequencies and properties
    try:
        freq_df_data = []
        for i in range(n_vars):
            for j in range(n_vars):
                for k in range(n_lags):
                    if frequent_links[i, j, k]:
                        freq_df_data.append({
                            'from': var_names[i],
                            'to': var_names[j],
                            'lag': k,
                            'frequency': link_frequencies[i, j, k],
                            'average_effect': consensus_val_matrix[i, j, k],
                            'link_power': link_power[i, j, k]
                        })
        
        if freq_df_data:
            freq_df = pd.DataFrame(freq_df_data)
            freq_filename = os.path.join(path_save, 
                f'consensus_link_freq_w{window_length_years}y_thresh{int(frequency_threshold*100)}.csv')
            freq_df.to_csv(freq_filename, index=False)
            logging.info(f"Saved consensus link frequencies to {freq_filename}")
        else:
            logging.info("No frequent links found to save in frequency CSV.")
    except Exception as e:
        logging.error(f"Failed to save link frequency data: {e}", exc_info=True)

    return consensus_graph, consensus_val_matrix, link_width_matrix


def extract_target_causal_pairs(
    csv_path: str,
    target_variables: List[str] = None
) -> List[Tuple[str, str, int]]:
    """
    Extract all causal pairs that terminate at target variables from robust graph CSV.
    
    Args:
        csv_path: Path to the filtered robust graph CSV file
        target_variables: List of target variable names to find incoming links for
                         Defaults to ['E-ind DJF(1)', 'C-ind DJF(1)']
        
    Returns:
        List of tuples (cause, effect, lag) for all links to target variables
    """
    if target_variables is None:
        target_variables = ['E-ind DJF(1)', 'C-ind DJF(1)']
        
    try:
        df = pd.read_csv(csv_path)
        
        # Filter for rows where Effect is one of our target variables
        target_links = df[df['Effect'].isin(target_variables)]
        
        # Extract unique (Cause, Effect, Lag) tuples
        var_pairs = []
        for _, row in target_links.iterrows():
            var_pairs.append((row['Cause'], row['Effect'], int(row['Lag'])))
        
        # Sort by effect variable, then by lag (descending), then by cause
        var_pairs.sort(key=lambda x: (x[1], -x[2], x[0]))
        
        logging.info(f"Extracted {len(var_pairs)} causal pairs targeting {target_variables}")
        return var_pairs
        
    except FileNotFoundError:
        logging.error(f"Error: Robust graph CSV not found at: {csv_path}")
        return []
    except Exception as e:
        logging.error(f"Error extracting causal pairs from {csv_path}: {e}")
        return []


def _plot_single_temporal_pair(
    results: Dict,
    var1: str,
    var2: str,
    lag_specified: int,
    time_axis: np.ndarray,
    window_step: int,
    window_length: int,
    var_names: List[str],
    path_save: str,
    suffix: str,
    pair_idx: int,
    output_format: str,
    panel_suffix: str
) -> None:
    """
    Helper function to plot a single temporal evolution pair.
    """
    if 'window_results' not in results or not results['window_results']:
        return
    
    val_matrices = results['window_results'].get('val_matrix')
    graphs = results['window_results'].get('graph')
    
    if not isinstance(val_matrices, np.ndarray) or val_matrices.ndim != 4:
        return
    if not isinstance(graphs, np.ndarray) or graphs.ndim != 4:
        return
    
    try:
        var1_idx = var_names.index(var1)
        var2_idx = var_names.index(var2)
    except ValueError:
        logging.warning(f"Variable pair ({var1}, {var2}) not found in var_names list")
        return
    
    # Extract dimensions
    n_windows = val_matrices.shape[0]
    ref_shape_3d = val_matrices.shape[1:]
    
    # Create figure for this pair
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Get colormap
    try:
        cmap = matplotlib.colormaps.get_cmap('tab20')
    except:
        cmap = None
    
    # Marker mapping for lag encoding (τ=3..6 → o, s, ^, D)
    def marker_for_lag(lag: int) -> str:
        if lag == 3:
            return 'o'
        if lag == 4:
            return 's'
        if lag == 5:
            return '^'
        if lag == 6:
            return 'D'
        return 'o'
    
    def is_key_mode_pair(cause: str, effect: str) -> bool:
        # Emphasize only EP pathways for the key modes
        if cause == 'REOF SST JJA' and effect == 'E-ind DJF(1)':
            return True
        if cause in {'MCA RWS-prec MAM(E)', 'MCA prec-RWS MAM(E)'} and effect == 'E-ind DJF(1)':
            return True
        return False
    
    values = []
    time_points_window = []
    window_indices = []
    
    lag_index = lag_specified
    if not (TAU_MIN <= lag_index <= TAU_MAX):
        plt.close(fig)
        return
    
    if lag_index >= ref_shape_3d[2]:
        plt.close(fig)
        return
    
    # Process each window
    for w in range(n_windows):
        try:
            val_matrix = val_matrices[w, :, :, :]
            graph = graphs[w, :, :, :]
            
            if graph.shape != ref_shape_3d or val_matrix.shape != ref_shape_3d:
                continue
                
        except (IndexError, ValueError):
            continue
        
        # Check if link exists and get value
        if (var1_idx < graph.shape[0] and var2_idx < graph.shape[1] and lag_index < graph.shape[2]):
            link_element = graph[var1_idx, var2_idx, lag_index]
            
            link_exists = False
            if isinstance(link_element, str):
                link_exists = (link_element != '')
            
            if link_exists:
                value = val_matrix[var1_idx, var2_idx, lag_index]
                values.append(value)
                time_point = time_axis[w * window_step + window_length // 2] if w * window_step + window_length // 2 < len(time_axis) else time_axis[-1]
                time_points_window.append(time_point)
                window_indices.append(w)
    
    # Generate plot visualization when data points exist
    if values and time_points_window:
        color = cmap(pair_idx % cmap.N) if cmap else None
        
        # Helper function for consecutive sequences
        def split_consecutive(vals, times, indices):
            if not vals:
                return []
            sequences = []
            curr_vals = [vals[0]]
            curr_times = [times[0]]
            for i in range(1, len(vals)):
                if indices[i] == indices[i-1] + 1:
                    curr_vals.append(vals[i])
                    curr_times.append(times[i])
                else:
                    sequences.append((curr_vals, curr_times))
                    curr_vals = [vals[i]]
                    curr_times = [times[i]]
            sequences.append((curr_vals, curr_times))
            return sequences
        
        sequences = split_consecutive(values, time_points_window, window_indices)
        
        for seq_idx, (seq_values, seq_times) in enumerate(sequences):
            marker = marker_for_lag(lag_specified)
            is_key_mode = is_key_mode_pair(var1, var2)
            line_width = 3 if is_key_mode else 2
            marker_size = 6 if is_key_mode else 4
            scatter_size = 65 if is_key_mode else 50
            if len(seq_values) == 1:
                ax.scatter(seq_times, seq_values, marker=marker, s=scatter_size, color=color)
            else:
                ax.plot(seq_times, seq_values, marker=marker, linewidth=line_width, markersize=marker_size, color=color)
        
        # No title for publication-style figure
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Causal Effect Strength', fontsize=12)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Create descriptive filename
        filename = create_descriptive_filename(
            base_name='temporal_evolution',
            method='pcmci',
            var_name=var2.replace(' ', '_'),
            predictor=var1.replace(' ', '_'),
            pathway=suffix,
            lag=f"lag{lag_specified}",
            season='',
            suffix=panel_suffix
        )
        
        output_path = os.path.join(path_save, filename)
        save_figure_optimized(fig, output_path, output_format)
        logging.info(f"Saved individual temporal evolution plot to {output_path}")
    
    plt.close(fig)
    
    # Memory management
    if 'gc' in sys.modules:
        gc.collect()


def plot_temporal_evolution(
    results: Dict,
    var_pairs: List[Tuple[str, str, int]],
    time_axis: np.ndarray,
    window_step: int,
    window_length: int,
    var_names: List[str],
    path_save: str,
    suffix: str = 'primary',
    output_format: str = 'png',
    single_panel: bool = False,
    panel_suffix: str = ''
) -> None:
    """
    Plot temporal evolution of specific causal relationships at specified lags.

    Args:
        results: Dictionary from pcmci.run_sliding_window_of
        var_pairs: List of tuples (var1, var2, lag_specified)
        time_axis: Numpy array representing the time points for the x-axis
        window_step: Step size of the sliding window (in data points)
        window_length: Length of the sliding window (in data points)
        var_names: List of variable names used in the analysis
        path_save: Directory to save plots
        suffix: Suffix for the output filename
    """
    logging.info(f"Plotting temporal evolution for {len(var_pairs)} pairs for suffix {suffix}...")
    
    if 'window_results' not in results or not results['window_results']:
        logging.warning(f"No 'window_results' found in results for suffix {suffix}. Skipping temporal evolution plot.")
        return

    val_matrices = results['window_results'].get('val_matrix')
    graphs = results['window_results'].get('graph')

    # Validate input structure (expect 4D NumPy arrays)
    if not isinstance(val_matrices, np.ndarray) or val_matrices.ndim != 4:
        logging.error(f"({suffix}) Expected 'val_matrix' to be a 4D NumPy array, got type {type(val_matrices)} with ndim={getattr(val_matrices, 'ndim', 'N/A')}.")
        return
    
    if not isinstance(graphs, np.ndarray) or graphs.ndim != 4:
        logging.error(f"({suffix}) Expected 'graph' to be a 4D NumPy array, got type {type(graphs)} with ndim={getattr(graphs, 'ndim', 'N/A')}.")
        return

    # Extract dimensions from the 4D array
    try:
        n_windows = val_matrices.shape[0]
        n_vars = val_matrices.shape[1]
        n_lags = val_matrices.shape[3]
        ref_shape_3d = val_matrices.shape[1:]

        # Validate dimensions
        expected_shape = (n_windows, len(var_names), len(var_names), TAU_MAX + 1)
        if val_matrices.shape != expected_shape:
            raise ValueError(f"Unexpected 4D matrix dimensions. Got {val_matrices.shape}, expected {expected_shape}")
        
        if graphs.shape != val_matrices.shape:
            raise ValueError(f"Shape mismatch between graphs {graphs.shape} and val_matrices {val_matrices.shape}")

    except (IndexError, ValueError) as e:
        logging.error(f"({suffix}) Error validating dimensions of input arrays: {e}")
        return

    if n_windows == 0:
        logging.warning(f"Zero windows found in results array for {suffix}. Skipping temporal plot.")
        return

    n_pairs = len(var_pairs)
    if n_pairs == 0:
        logging.warning(f"No variable pairs provided for temporal evolution plot ({suffix}). Skipping.")
        return

    # Support for single panel mode (each var pair in separate plot)
    if single_panel:
        # Process each variable pair separately
        for i, (var1, var2, lag_specified) in enumerate(var_pairs):
            _plot_single_temporal_pair(
                results, var1, var2, lag_specified,
                time_axis, window_step, window_length, var_names,
                path_save, suffix, i, output_format, panel_suffix
            )
        return
    
    # Original multi-pair plot (taller for better separation)
    fig, ax = plt.subplots(figsize=(12, 10))
    # Legend layout knobs (tune these to adjust spacing)
    ENC_LEGEND_Y = 0.26  # y for encoding legend (figure coords)
    MAIN_LEGEND_Y = 0.14  # y for main legend (figure coords)
    BOTTOM_BASE = 0.1  # base bottom space for legends
    BOTTOM_PER_ROW = 0.05  # extra bottom per main-legend row

    # Use tab20 colormap for 20 distinct colors
    try:
        cmap = matplotlib.colormaps.get_cmap('tab20')
        if cmap is None:
            raise ValueError("Colormap 'tab20' not found")
    except Exception as e:
        logging.error(f"Failed to get colormap 'tab20': {e}. Falling back to tab10.")
        try:
            cmap = matplotlib.colormaps.get_cmap('tab10')
        except Exception as e2:
            logging.error(f"Failed to get fallback colormap 'tab10': {e2}. Using default colors.")
            cmap = None

    # Marker mapping for lag encoding (τ=3..6 → o, s, ^, D)
    def marker_for_lag(lag: int) -> str:
        if lag == 3:
            return 'o'
        if lag == 4:
            return 's'
        if lag == 5:
            return '^'
        if lag == 6:
            return 'D'
        # Default marker if outside specified mapping
        return 'o'

    # Key modes emphasis: only EP pathways should be bold/emphasized
    def is_key_mode_pair(cause: str, effect: str) -> bool:
        if cause == 'REOF SST JJA' and effect == 'E-ind DJF(1)':
            return True
        if cause in {'MCA RWS-prec MAM(E)', 'MCA prec-RWS MAM(E)'} and effect == 'E-ind DJF(1)':
            return True
        return False

    # Helper function to split data into consecutive sequences
    def split_into_consecutive_sequences(values, time_points, window_indices):
        """
        Split data into consecutive sequences based on window indices.
        
        Returns list of (values_seq, time_points_seq) tuples for each consecutive sequence.
        """
        if not values:
            return []
        
        sequences = []
        current_values = [values[0]]
        current_times = [time_points[0]]
        
        for i in range(1, len(values)):
            # Check if current window is consecutive to previous
            if window_indices[i] == window_indices[i-1] + 1:
                # Consecutive - add to current sequence
                current_values.append(values[i])
                current_times.append(time_points[i])
            else:
                # Gap detected - save current sequence and start new one
                sequences.append((current_values, current_times))
                current_values = [values[i]]
                current_times = [time_points[i]]
        
        # Don't forget the last sequence
        sequences.append((current_values, current_times))
        
        return sequences
    
    all_time_points_plot = []
    legend_entries = []  # For building ordered main legend (by stability)

    for i, (var1, var2, lag_specified) in enumerate(var_pairs):
        try:
            var1_idx = var_names.index(var1)
            var2_idx = var_names.index(var2)
        except ValueError:
            logging.warning(f"({suffix}) Variable pair ({var1}, {var2}) skipped: not found in var_names list")
            continue

        values = []
        time_points_window = []
        window_indices = []  # Track which window each value comes from

        lag_index = lag_specified
        if not (TAU_MIN <= lag_index <= TAU_MAX):
            logging.warning(f"({suffix}) Specified lag {lag_specified} for pair ({var1}, {var2}) is outside range [{TAU_MIN}, {TAU_MAX}]. Skipping.")
            continue

        if lag_index >= ref_shape_3d[2]:
            logging.warning(f"({suffix}) Specified lag {lag_specified} for pair ({var1}, {var2}) is out of bounds for matrix dimension {ref_shape_3d[2]}. Skipping.")
            continue

        # Process each window
        for w in range(n_windows):
            try:
                val_matrix = val_matrices[w, :, :, :]
                graph = graphs[w, :, :, :]

                if graph.shape != ref_shape_3d or val_matrix.shape != ref_shape_3d:
                    raise ValueError(f"Window {w}: Unexpected slice shape")

            except (IndexError, ValueError) as e:
                logging.warning(f"({suffix}) Skipping window {w} for pair ({var1}, {var2}) due to: {e}")
                continue

            # Check if link exists and get value
            if (var1_idx < graph.shape[0] and var2_idx < graph.shape[1] and lag_index < graph.shape[2]):
                link_element = graph[var1_idx, var2_idx, lag_index]
                
                link_exists = False
                if isinstance(link_element, np.ndarray):
                    logging.warning(f"({suffix}) Window {w}, Pair ({var1}, {var2}), Lag {lag_index}: Graph element is unexpected array. Treating as NO link.")
                elif isinstance(link_element, str):
                    link_exists = (link_element != '')
                else:
                    logging.warning(f"({suffix}) Window {w}, Pair ({var1}, {var2}), Lag {lag_index}: Graph element has unexpected type {type(link_element)}. Treating as NO link.")

                if link_exists:
                    value = val_matrix[var1_idx, var2_idx, lag_index]
                    values.append(value)
                    # Calculate time point for this window
                    time_point = time_axis[w * window_step + window_length // 2] if w * window_step + window_length // 2 < len(time_axis) else time_axis[-1]
                    time_points_window.append(time_point)
                    window_indices.append(w)  # Track window index
            else:
                logging.warning(f"({suffix}) Window {w}, Pair ({var1}, {var2}), Lag {lag_index}: Indices out of bounds")

        # Plot this variable pair
        if values and time_points_window:
            color = cmap(i % cmap.N) if cmap else None
            marker = marker_for_lag(lag_specified)
            is_key_mode = is_key_mode_pair(var1, var2)
            line_width = 3 if is_key_mode else 2
            marker_size = 6 if is_key_mode else 4
            scatter_size = 65 if is_key_mode else 50
            
            # Split into consecutive sequences
            sequences = split_into_consecutive_sequences(values, time_points_window, window_indices)
            
            # Plot each sequence
            for seq_idx, (seq_values, seq_times) in enumerate(sequences):
                # Only add label to first sequence to avoid duplicate legend entries
                label = f'{var1} → {var2} (lag {lag_specified})' if seq_idx == 0 else None
                
                if len(seq_values) == 1:
                    # Single point - plot as scatter to ensure it's visible
                    ax.scatter(seq_times, seq_values,
                               marker=marker, s=scatter_size,
                               label=label,
                               color=color)
                else:
                    # Multiple consecutive points - plot with line
                    ax.plot(seq_times, seq_values,
                            marker=marker, linewidth=line_width, markersize=marker_size,
                            label=label,
                            color=color)
            
            all_time_points_plot.extend(time_points_window)
            # Collect legend entry with stability count
            stability_count = len(values)
            logging.debug(f"({suffix}) Legend candidate: {var1} -> {var2} (lag {lag_specified}), stability_count={stability_count}")
            legend_entries.append({
                'count': stability_count,
                'label': f'{var1} → {var2} (lag {lag_specified})',
                'marker': marker,
                'color': color,
                'line_width': line_width,
                'marker_size': marker_size,
                'is_key': is_key_mode
            })
            # Specific debug for NPMM-wind MAM visibility
            if var1 == 'NPMM-wind MAM':
                logging.info(f"({suffix}) NPMM-wind MAM → {var2} (lag {lag_specified}) has {stability_count} visible window points")
        else:
            logging.info(f"({suffix}) No valid data points for pair ({var1}, {var2}) at lag {lag_specified}")

    # Finalize plot
    if all_time_points_plot:
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('Causal Effect Strength', fontsize=12)
        # Build legends below the plot using figure-level legends to avoid replacement conflicts
        from matplotlib.lines import Line2D
        # Compact encoding legend (row of semantics) – figure-level
        enc_handles = [
            Line2D([0], [0], linestyle='None', marker='o', color='k', label='Single window'),
            Line2D([0], [0], linestyle='-', marker='o', color='k', label='Consecutive windows'),
            Line2D([0], [0], linestyle='None', marker=marker_for_lag(3), color='k', label='τ=3'),
            Line2D([0], [0], linestyle='None', marker=marker_for_lag(4), color='k', label='τ=4'),
            Line2D([0], [0], linestyle='None', marker=marker_for_lag(5), color='k', label='τ=5'),
            Line2D([0], [0], linestyle='None', marker=marker_for_lag(6), color='k', label='τ=6'),
        ]
        enc_cols = len(enc_handles)
        # Main legend entries ordered by stability (descending)
        main_handles = []
        main_labels = []
        main_flags = []
        if legend_entries:
            legend_entries.sort(key=lambda d: (-d['count'], d['label']))
            for d in legend_entries:
                linestyle = '-' if d['count'] > 1 else 'None'
                h = Line2D([0], [0], linestyle=linestyle, marker=d['marker'],
                           color=d['color'], linewidth=d['line_width'], markersize=d['marker_size'])
                main_handles.append(h)
                main_labels.append(d['label'])
                main_flags.append(d['is_key'])
        main_cols = min(3, len(main_handles)) if main_handles else 1
        # Estimate rows to reserve space
        main_rows = math.ceil(len(main_handles) / max(1, main_cols))
        # Place legends: encoding slightly above main
        fig_enc_leg = fig.legend(handles=enc_handles, labels=[h.get_label() for h in enc_handles],
                                 loc='lower center', bbox_to_anchor=(0.5, ENC_LEGEND_Y), ncol=enc_cols,
                                 frameon=True)
        # Preserve encoding legend when adding another figure legend
        try:
            fig.add_artist(fig_enc_leg)
        except Exception as e:
            logging.warning(f"({suffix}) Could not add encoding legend as artist: {e}")
        # Reorder to make visual layout row-wise across columns
        if main_handles and main_cols > 1 and main_rows > 1:
            items = list(zip(main_handles, main_labels, main_flags))
            items_row_major = items[:]  # already sorted by stability desc
            items_col_major = []
            for c in range(main_cols):
                for r in range(main_rows):
                    idx = r * main_cols + c
                    if idx < len(items_row_major):
                        items_col_major.append(items_row_major[idx])
            main_handles, main_labels, main_flags = zip(*items_col_major)
            main_handles, main_labels, main_flags = list(main_handles), list(main_labels), list(main_flags)
        fig_main_leg = None
        if main_handles:
            fig_main_leg = fig.legend(handles=main_handles, labels=main_labels,
                                      loc='lower center', bbox_to_anchor=(0.5, MAIN_LEGEND_Y), ncol=main_cols,
                                      frameon=True)
            # Emphasize key modes in legend text based on pair-specific flags
            if fig_main_leg is not None:
                texts = fig_main_leg.get_texts()
                for idx, text in enumerate(texts):
                    if idx < len(main_flags) and main_flags[idx]:
                        text.set_fontweight('bold')
        # Log top-3 by stability for traceability and first-row labels
        if legend_entries:
            top_sorted = sorted(legend_entries, key=lambda d: (-d['count'], d['label']))
            top3 = top_sorted[:3]
            first_row_labels = [lab for lab in (main_labels[:main_cols] if main_handles else [])]
            logging.info(f"({suffix}) Top-3 by stability: {[t['label'] for t in top3]}")
            logging.info(f"({suffix}) First-row legend labels (row-wise intent): {first_row_labels}")
        logging.info(f"({suffix}) Legend debug: enc_cols={enc_cols}, main_cols={main_cols}, main_rows={main_rows}, main_entries={len(main_handles)}")
        ax.grid(True, alpha=0.3)
        # Ensure legends don't get cut off – reserve dynamic bottom space
        plt.tight_layout()
        bottom_needed = BOTTOM_BASE + BOTTOM_PER_ROW * max(1, main_rows)  # heuristic spacing (extra for encoding legend above)
        bottom_needed = min(max(bottom_needed, 0.25), 0.5)
        logging.info(f"({suffix}) Applying subplots_adjust bottom={bottom_needed:.2f}")
        plt.subplots_adjust(bottom=bottom_needed)
        
        output_path = os.path.join(path_save, f'temporal_evolution_{suffix}')
        save_figure_optimized(fig, output_path, output_format)
        logging.info(f"Saved temporal evolution plot to {output_path}")
    else:
        logging.warning(f"({suffix}) No valid data to plot for temporal evolution")

    plt.close(fig)
    
    # Memory management
    if 'gc' in sys.modules:
        gc.collect()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ MAIN PIPELINE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """
    Main analysis pipeline for sliding window PCMCI+ analysis.
    
    Orchestrates:
    1. Configuration loading and validation
    2. Data preparation with pruned variables
    3. Sliding window sensitivity analysis
    4. Consensus graph creation and visualization
    5. Temporal evolution analysis
    """
    # Parse arguments
    global args  # Make args globally accessible for nested functions
    args = parse_arguments()
    
    # Validate configuration
    if args.predictor_set == 'custom' and not args.config:
        logging.error("Custom configuration file required when --predictor-set=custom")
        sys.exit(1)
    
    # Setup logging header
    logging.info("=" * 70)
    logging.info("PCMCI+ SLIDING WINDOW ANALYSIS")
    logging.info("=" * 70)
    
    # Load predictor configuration
    try:
        custom_config = None
        if args.predictor_set == 'custom':
            custom_config = load_custom_config_from_file(args.config)
            logging.info(f"Loaded custom configuration from {args.config}")
        
        config = PredictorConfig(args.predictor_set, custom_config)
        logging.info(f"Predictor configuration: {config}")
        
    except Exception as e:
        logging.error(f"Failed to load predictor configuration: {e}")
        sys.exit(1)
    
    # Get configuration components
    variable_seasons = config.get_variables()
    all_variables = config.get_variable_list()
    variable_groups = config.get_variable_groups()
    season_months = config.get_season_months()
    
    logging.info(f"Analyzing {len(all_variables)} variables from {args.predictor_set} set")
    
    # Setup paths
    data_dir = get_data_path('time_series/', data_type="processed")
    load_path = get_results_path(os.path.join('PCMCIplus', args.suffix), result_type="figures")
    
    # Add FDR indicator to save path if loading FDR results
    save_subdir = 'sliding_window'
    if args.load_fdr:
        save_subdir = f'sliding_window_fdr_pc_alpha{str(args.fdr_pc_alpha).replace(".", "p")}'
    
    save_path = get_results_path(
        os.path.join('PCMCIplus', args.suffix, save_subdir, f'pc_alpha_{args.pc_alpha}'),
        result_type="figures"
    )
    os.makedirs(save_path, exist_ok=True)
    
    # Load and apply pruned variables
    # Construct filename based on FDR settings
    base_filename = f'filtered_robust_graph_{args.suffix}_robust_filtered'
    if args.load_fdr:
        # Match causal discovery analysis naming convention: decimal point replaced with 'p'
        fdr_suffix = f"_fdr_pc_alpha{str(args.fdr_pc_alpha).replace('.','p')}"
        base_filename += fdr_suffix
        logging.info(f"Loading FDR-pruned results with pc_alpha={args.fdr_pc_alpha}")
    
    robust_csv_path = os.path.join(load_path, f'{base_filename}.csv')
    logging.info(f"Loading pruned variables list from: {robust_csv_path}")
    pruned_variables = load_pruned_variables(robust_csv_path)
    
    # Filter variables and groups based on pruned list
    filtered_variable_seasons, filtered_jja, filtered_son, filtered_djf, _, filtered_djf_eff = reduce_variable_lists(
        pruned_variables,
        variable_seasons,
        variable_groups.get('JJA_PREDICTORS', []),
        variable_groups.get('SON_PREDICTORS', []),
        variable_groups.get('DJF_CONFOUNDERS', []),
        variable_groups.get('MAM_MEDIATORS', []),
        variable_groups.get('DJF_EFFECTS', [])
    )
    
    final_variable_list = sorted(list(filtered_variable_seasons.keys()))
    if not final_variable_list:
        logging.error("No variables remaining after pruning. Exiting.")
        sys.exit(1)
    
    # Load and prepare data
    data_file_path = os.path.join(data_dir, args.data_file)
    logging.info(f"Loading and preparing data for {len(final_variable_list)} variables from: {data_file_path}")
    
    data_array, mask_array, var_names = load_and_prepare_data(
        file_path=data_file_path,
        variables_to_load=final_variable_list,
        variable_seasons_map=filtered_variable_seasons,
        season_months_map=season_months,
        jja_predictors=filtered_jja,
        son_predictors=filtered_son,
        djf_confounders=filtered_djf,
        djf_effects=filtered_djf_eff,
        jja1_mediators=[]  # No JJA(1) mediators in sliding window analysis
    )
    
    if data_array is None:
        logging.error("Failed to load data")
        sys.exit(1)
    
    # Setup Tigramite
    dataframe = pp.DataFrame(
        data_array,
        datatime=np.arange(len(data_array)),
        var_names=var_names,
        mask=mask_array
    )
    logging.info(f"Created Tigramite dataframe with shape {data_array.shape} and mask shape {mask_array.shape}")
    
    # Create custom test class using the utility function
    custom_robust_parcorr = create_custom_test_class(RobustParCorr)
    
    cond_ind_test = custom_robust_parcorr(significance='analytic', mask_type='y')
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test, verbosity=0)
    logging.info("Initialized PCMCI+ with custom RobustParCorr.")
    
    # Calculate effective samples and time axis
    effective_samples_per_var = calculate_effective_samples(mask_array)
    total_years = len(data_array) / 4.0
    samples_per_year_per_var = effective_samples_per_var / total_years if total_years > 0 else np.ones_like(effective_samples_per_var)
    valid_samples_per_year = samples_per_year_per_var[samples_per_year_per_var > 0]
    min_samples_per_year = np.min(valid_samples_per_year) if valid_samples_per_year.size > 0 else 1.0
    
    logging.info(f"Estimated minimum samples per year across variables: {min_samples_per_year:.2f}")
    if min_samples_per_year <= 0:
        logging.warning("Minimum samples per year is zero or negative. Using 1.0 as fallback.")
        min_samples_per_year = 1.0
    
    # Create time axis (assuming quarterly data starting 1945)
    start_year = 1945
    time_axis = np.arange(start_year, start_year + len(data_array) / 4.0, 0.25)
    logging.info(f"Created time axis from {time_axis.min()} to {time_axis.max()}")
    
    # Parse window configurations
    try:
        window_configs = []
        for config_str in args.window_configs:
            w_len, w_step = map(int, config_str.split(','))
            window_configs.append((w_len, w_step))
        logging.info(f"Window configurations: {window_configs}")
    except ValueError:
        logging.error("Invalid window configuration format. Use 'length,step' format.")
        sys.exit(1)
    
    # Sensitivity Analysis Loop
    results_sensitivity = {}
    logging.info("\n--- Starting Sensitivity Analysis ---")
    
    for w_length_yr, w_step_yr in window_configs:
        config_suffix = f'w{w_length_yr}y_s{w_step_yr}y'
        logging.info(f"\nProcessing config: {config_suffix}")
        
        # Adjust window length and step from years to data points
        adj_length = int(np.ceil(w_length_yr * 4 / min_samples_per_year))
        adj_step = int(np.ceil(w_step_yr * 4 / min_samples_per_year))
        adj_step = max(1, adj_step)
        
        logging.info(f"Target window: {w_length_yr}y / {w_step_yr}y.")
        logging.info(f"Adjusted window (data points): length={adj_length}, step={adj_step}")
        logging.info(f"Adjusted window approx duration: {adj_length / 4.0:.1f} years")
        
        # Check feasibility
        if adj_length > len(data_array):
            logging.warning(f"Adjusted window length {adj_length} exceeds data length {len(data_array)}. Skipping config {config_suffix}.")
            continue
        
        if adj_length <= max(TAU_MAX, 2):
            logging.warning(f"Adjusted window length {adj_length} is too small. Skipping config {config_suffix}.")
            continue
        
        # Run sliding window analysis
        logging.info(f"Running pcmci.run_sliding_window_of for {config_suffix}...")
        try:
            results = pcmci.run_sliding_window_of(
                method='run_pcmciplus',
                method_args={
                    'tau_min': TAU_MIN,
                    'tau_max': TAU_MAX,
                    'pc_alpha': args.pc_alpha,
                    'max_combinations': args.max_combinations,
                    'reset_lagged_links': True
                },
                window_length=adj_length,
                window_step=adj_step,
            )
            logging.info(f"Finished run_sliding_window_of for {config_suffix}.")
            results_sensitivity[(w_length_yr, w_step_yr)] = results
        except Exception as e:
            logging.error(f"Error during run_sliding_window_of for {config_suffix}: {e}", exc_info=True)
            continue
        
        # Plot results for individual windows
        logging.info(f"Plotting individual window results for {config_suffix}...")
        plot_sliding_window_results(
            results, 
            var_names, 
            filtered_variable_seasons, 
            save_path, 
            suffix=config_suffix,
            predictor_set=args.predictor_set,
            cond_ind_test='robust_parcorr',
            output_format=args.output_format
        )
        
        # Create and plot consensus graph
        logging.info(f"Creating consensus graph for {config_suffix}...")
        consensus_graph, consensus_val_matrix, link_width_matrix = create_consensus_graph(
            results, var_names, window_length_years=w_length_yr, path_save=save_path
        )
        
        if consensus_graph is not None and consensus_val_matrix is not None:
            logging.info(f"Filtering consensus graph for {config_suffix}...")
            dummy_p_matrix = np.ones_like(consensus_val_matrix)
            
            filtered_consensus_graph, filtered_consensus_val_matrix, _ = filter_graph(
                consensus_graph,
                consensus_val_matrix,
                dummy_p_matrix,
                var_names,
                target_variables=['E-ind DJF(1)', 'C-ind DJF(1)'],
                tau_min=TAU_MIN,
                tau_max=TAU_MAX,
                max_cumulative_lag=TAU_MAX
            )
            
            logging.info(f"Plotting filtered consensus graph for {config_suffix}...")
            consensus_plot_suffix = f'{config_suffix}_thresh{int(get_frequency_threshold(w_length_yr)*100)}_consensus'
            create_causal_graph_w_map(
                graph=filtered_consensus_graph,
                val_matrix=filtered_consensus_val_matrix,
                var_names=var_names,
                save_dir=save_path,
                suffix=consensus_plot_suffix,
                predictor_set=args.predictor_set,
                cond_ind_test='robust_parcorr'
            )
        else:
            logging.warning(f"Consensus graph creation failed for {config_suffix}. Skipping filtering and plotting.")
        
        logging.info(f"Completed processing for config {config_suffix}.")
    
    logging.info("\n--- Finished Sensitivity Analysis ---")
    
    # Temporal Evolution Analysis
    logging.info("\n--- Starting Temporal Evolution Analysis ---")
    
    # Dynamically extract variable pairs from the robust graph
    logging.info("Extracting causal pairs from robust graph for temporal evolution analysis...")
    var_pairs_initial = extract_target_causal_pairs(robust_csv_path)
    
    # Debug presence checks for specific modes
    target_check = 'NPMM-wind MAM'
    logging.info(f"Presence check: '{target_check}' in var_names? {'YES' if target_check in var_names else 'NO'}")
    initial_hits = sum(1 for v1, v2, l in var_pairs_initial if v1 == target_check)
    logging.info(f"Initial pairs containing '{target_check}' as cause: {initial_hits}")

    if not var_pairs_initial:
        logging.warning("No causal pairs found targeting E-ind DJF(1) or C-ind DJF(1). ")
        # No fallback pairs - temporal evolution will be skipped
    else:
        logging.info(f"Found {len(var_pairs_initial)} causal relationships to target variables:")
        for cause, effect, lag in var_pairs_initial:
            logging.info(f"  {cause} → {effect} (lag {lag})")
    
    # Filter variable pairs to only include pairs where both variables are in the final list
    var_pairs_filtered = [
        (v1, v2, lag) for v1, v2, lag in var_pairs_initial
        if v1 in var_names and v2 in var_names
    ]
    filtered_hits = sum(1 for v1, v2, l in var_pairs_filtered if v1 == target_check)
    logging.info(f"Filtered to {len(var_pairs_filtered)} variable pairs where both variables exist in the analysis (retained '{target_check}': {filtered_hits})")
    
    if not var_pairs_filtered:
        logging.warning("No valid variable pairs remain after filtering. Skipping temporal evolution analysis.")
    else:
        # Parse temporal evolution configurations
        try:
            temporal_configs = []
            for config_str in args.temporal_configs:
                t_len, t_step = map(int, config_str.split(','))
                temporal_configs.append((t_len, t_step, f'{"smooth" if t_len != t_step else "periods"}_w{t_len}y_s{t_step}y'))
        except ValueError:
            logging.error("Invalid temporal configuration format. Use 'length,step' format.")
            sys.exit(1)
        
        for t_len_yr, t_step_yr, t_suffix in temporal_configs:
            logging.info(f"\nRunning analysis for temporal evolution plot: {t_suffix}")
            
            t_adj_length = int(np.ceil(t_len_yr * 4 / min_samples_per_year))
            t_adj_step = int(np.ceil(t_step_yr * 4 / min_samples_per_year))
            t_adj_step = max(1, t_adj_step)
            
            logging.info(f"Target window: {t_len_yr}y / {t_step_yr}y.")
            logging.info(f"Adjusted window (data points): length={t_adj_length}, step={t_adj_step}")
            logging.info(f"Adjusted window approx duration: {t_adj_length / 4.0:.1f} years")
            
            # Check feasibility
            if t_adj_length > len(data_array) or t_adj_length <= max(TAU_MAX, 2):
                logging.warning(f"Adjusted window length {t_adj_length} is invalid. Skipping temporal config {t_suffix}.")
                continue
            
            # Run sliding window analysis for this configuration
            try:
                results_temporal = pcmci.run_sliding_window_of(
                    method='run_pcmciplus',
                    method_args={
                        'tau_min': TAU_MIN,
                        'tau_max': TAU_MAX,
                        'pc_alpha': args.pc_alpha,
                        'max_combinations': args.max_combinations,
                        'reset_lagged_links': True
                    },
                    window_length=t_adj_length,
                    window_step=t_adj_step,
                )
                logging.info(f"Finished run_sliding_window_of for {t_suffix}.")
                
                # Plot temporal evolution using the filtered pairs
                plot_temporal_evolution(
                    results_temporal,
                    var_pairs_filtered,
                    time_axis,
                    t_adj_step,
                    t_adj_length,
                    var_names,
                    save_path,
                    suffix=t_suffix,
                    output_format=args.output_format,
                    single_panel=args.single_panel,
                    panel_suffix=args.panel_suffix
                )
                
            except Exception as e:
                logging.error(f"Error during temporal analysis run for {t_suffix}: {e}", exc_info=True)
                continue
    
    logging.info("\n--- Finished Temporal Evolution Analysis ---")
    logging.info(f"\nResults saved to: {save_path}")
    logging.info("Pipeline finished successfully.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ENTRY POINT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Unhandled error in main execution: {e}", exc_info=True)
        raise
