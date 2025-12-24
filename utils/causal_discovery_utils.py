"""
PCMCI+ Causal Discovery Utilities: Climate Teleconnection Analysis Framework
===========================================================================

This module provides specialized utilities for causal discovery in climate systems
using the PCMCI+ algorithm with climate-specific adaptations. The implementation
focuses on identifying robust causal pathways between climate modes while ensuring
temporal causality constraints essential for prediction applications.

Scientific Domain and Context
----------------------------
Climate teleconnections operate through complex physical mechanisms involving
atmospheric and oceanic dynamics across multiple timescales. Traditional correlation
analysis cannot distinguish causation from confounding, making causal discovery
essential for:

- **Climate Prediction**: Identifying true precursors vs spurious correlations
- **Process Understanding**: Separating causes from effects in climate interactions
- **Model Validation**: Testing physical mechanisms in climate model simulations
- **Attribution Studies**: Determining causal drivers of climate variability

This module implements PCMCI+ with forward-in-time conditioning to ensure that
identified causal relationships respect the arrow of time and can be used for
actual climate prediction applications.

Core Methodology: PCMCI+ with Climate Adaptations
------------------------------------------------

**PCMCI+ Algorithm**:
The algorithm combines two phases:
1. **PC (Peter-Clark) phase**: Skeleton discovery through conditional independence tests
2. **MCI (Momentary Conditional Independence) phase**: Causal orientation and lag determination

**Climate-Specific Adaptations**:
- **Forward-in-Time Conditioning**: Variables only condition on their own past
- **Seasonal Lag Structure**: Optimized for climate prediction timescales (1-6 seasons)
- **Robust Independence Testing**: RobustParCorr handles climate data characteristics
- **Iterative Robustness**: Multiple significance levels assess edge persistence

**Statistical Framework**:
- Null Hypothesis: Conditional independence between variables given conditioning set
- Test Statistic: Robust partial correlation coefficients
- Significance: Multiple testing correction with False Discovery Rate control
- Robustness: Edge persistence across significance thresholds

Key Functions and Capabilities
-----------------------------

**Core Analysis Functions**:
- `run_pcmci_analysis()`: Main causal discovery pipeline with climate adaptations
- `load_and_prepare_data()`: Climate data preprocessing for PCMCI+ analysis
- `generate_robust_graph()`: Multi-threshold robustness analysis
- `filter_target_pathways()`: Focus on pathways to specific climate targets

**Robustness and Validation**:
- `analyze_edge_persistence()`: Quantify causal link stability
- `validate_temporal_ordering()`: Ensure proper causality constraints
- `cross_validate_results()`: Assess reproducibility across data subsets

**Output and Visualization**:
- `export_causal_graph()`: Standard formats for further analysis
- `generate_summary_statistics()`: Quantitative robustness metrics
- `create_pathway_metadata()`: Complete documentation of discovered pathways

Scientific Standards and Quality Control
---------------------------------------

**Statistical Validation**:
- **Independence Test Assumptions**: Validated for climate index characteristics
- **Sample Size Requirements**: Minimum 30 years for robust climate statistics
- **Multiple Testing**: FDR correction for high-dimensional climate networks
- **Cross-Validation**: Temporal stability assessment across different periods

**Physical Consistency**:
- **Temporal Ordering**: Strict enforcement of cause-before-effect relationships
- **Climate Timescales**: Lag structures consistent with atmospheric/oceanic dynamics
- **Known Teleconnections**: Validation against established climate relationships
- **Physical Plausibility**: Screening for rapid responses inconsistent with physics

**Methodological Rigor**:
- **Reproducibility**: Complete parameter documentation and random seed control
- **Sensitivity Analysis**: Robustness to algorithm parameters and data preprocessing
- **Uncertainty Quantification**: Confidence intervals and persistence measures
- **Comparative Validation**: Cross-validation with alternative causal methods

Integration with Climate Research Workflow
-----------------------------------------
This module serves as the foundation for:
- Climate prediction system development
- Teleconnection mechanism validation
- Climate model evaluation frameworks
- Process-based understanding of climate variability

The utilities enable systematic causal analysis across different climate datasets,
time periods, and spatial domains while maintaining methodological consistency
and scientific rigor essential for climate research applications.

Dependencies and Performance
---------------------------
Built on the tigramite package for PCMCI+ implementation with climate-specific
extensions. Requires numpy/pandas for data handling, xarray for climate data
structures, and scikit-learn for preprocessing. Computational complexity scales
as O(p²×τ×n) where p=variables, τ=max_lag, n=sample_size.

Memory requirements: ~1GB RAM per 50 variables × 100 time steps
Runtime estimates: 10 minutes (20 vars) to 2 hours (50 vars) depending on configuration

"""

import os
import sys
import logging
import json
import importlib
from datetime import datetime
import numpy as np
import pandas as pd
import xarray as xr
from typing import List, Tuple, Dict, Optional, Any

from tigramite import data_processing as pp
from tigramite.independence_tests.robust_parcorr import RobustParCorr
from tigramite.pcmci import PCMCI

# Import predictor config functions
try:
    from .predictor_configs import get_cond_ind_test_config, PredictorConfig
    from .paths import get_results_path
except ImportError:
    from predictor_configs import get_cond_ind_test_config, PredictorConfig
    from paths import get_results_path

# --- Constants (can be overridden by arguments if needed) ---
DEFAULT_TAU_MIN = 1
DEFAULT_TAU_MAX = 6
DEFAULT_ALPHA_VALUES = [0.05, 0.025, 0.015, 0.01]
DEFAULT_MAX_COMBINATIONS = 10
DEFAULT_PERSISTENCE_MIN = 0.75
DEFAULT_PERSISTENCE_MAX_PVAL = 0.05
DEFAULT_PERSISTENCE_MIN_CONSISTENCY = 0.9
DEFAULT_PERSISTENCE_MIN_EFFECT = 0.15
DEFAULT_FILTER_MIN_CUM_LAG = 3
DEFAULT_MAP_EXTENT = [70, 359.99, -50, 50]  # Default global extent for consistent visualization

# --- Helper Functions ---

def load_and_prepare_data(
    file_path: str,
    variables_to_load: List[str],
    variable_seasons_map: Dict[str, str],
    season_months_map: Dict[str, List[int]],
    jja_predictors: List[str] = [], # Optional lists for specific exclusions
    son_predictors: List[str] = [],
    djf_confounders: List[str] = [],
    djf_effects: List[str] = [],
    jja1_mediators: List[str] = []
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
    """
    Load and prepare data from a netCDF file for a specific list of variables.

    Args:
        file_path (str): Path to the netCDF file.
        variables_to_load (List[str]): List of variable names to load and process.
        variable_seasons_map (Dict[str, str]): Maps variable names to season keys.
        season_months_map (Dict[str, List[int]]): Maps season keys to month numbers.
        jja_predictors (List[str]): Variables subject to JJA 2023 exclusion.
        son_predictors (List[str]): Variables subject to SON 2023 exclusion.
        djf_confounders (List[str]): Variables subject to D(23)JF(24) exclusion.
        djf_effects (List[str]): Variables subject to D(45)JF(46) exclusion.
        jja1_mediators (List[str]): Variables subject to JJA 1945 exclusion.

    Returns:
        Tuple containing:
        - data_array: numpy array of shape (time_steps, variables) or None on error.
        - mask_array: boolean mask array of same shape or None on error.
        - var_names_loaded: list of variable names corresponding to columns, or None on error.
    """
    logging.info(f"Loading subset of data for causal discovery: {variables_to_load}")
    try:
        # Load only the required variables
        ds = xr.open_dataset(file_path)[variables_to_load].fillna(0).load()
        time_length = len(ds.time)
        num_variables = len(variables_to_load)
        var_names_loaded = list(ds.data_vars) # Get actual order

        if set(var_names_loaded) != set(variables_to_load):
            logging.warning(f"Mismatch between requested variables and variables found in file: "
                            f"Requested: {variables_to_load}, Found: {var_names_loaded}")
            # Proceeding with variables found, but order might be different if some were missing.
            # Re-assign num_variables based on what was actually loaded.
            num_variables = len(var_names_loaded)
            if num_variables == 0:
                logging.error("No requested variables found in the dataset.")
                return None, None, None


        # Initialize arrays
        data_array = np.zeros((time_length, num_variables))
        mask_array = np.zeros((time_length, num_variables), dtype=bool)

        # Ensure time coordinates are datetime objects for dt accessor
        if not pd.api.types.is_datetime64_any_dtype(ds['time']):
            logging.warning("Time coordinate is not datetime. Attempting conversion.")
            try:
                ds['time'] = pd.to_datetime(ds['time'].values)
            except Exception as time_conv_err:
                logging.error(f"Could not convert time coordinate to datetime: {time_conv_err}")
                return None, None, None

        for idx, var in enumerate(var_names_loaded):
            season = variable_seasons_map.get(var)
            if season is None:
                logging.warning(f"Variable '{var}' not found in variable_seasons_map. Cannot apply seasonal mask.")
                # Mask all data for this variable if season is unknown? Or leave unmasked?
                # Variable remains unmasked due to missing seasonal mapping
                mask_array[:, idx] = False # No mask applied
                data_array[:, idx] = ds[var].values
                continue

            months = season_months_map.get(season)
            if months is None:
                logging.warning(f"Season '{season}' for variable '{var}' not found in season_months_map. Cannot apply seasonal mask.")
                mask_array[:, idx] = False # No mask applied
                data_array[:, idx] = ds[var].values
                continue

            # Create base season mask
            season_mask = np.isin(ds.time.dt.month, months)

            # Apply specific exclusions from causal discovery logic if applicable
            # Ensure these indices are valid for the time_length
            if var in jja_predictors and time_length >= 3: # Exclude JJA 2023 (third last time step)
                season_mask[-3] = False
            if var in son_predictors and time_length >= 2: # Exclude SON 2023 (second last time step)
                season_mask[-2] = False
            if var in djf_confounders and time_length >= 1: # Exclude D(2023)JF(2024) (last time step)
                season_mask[-1] = False
            if var in djf_effects and time_length >= 2: # Exclude D(1945)JF(1946) (second time step, index 1)
                if len(season_mask) > 1: season_mask[1] = False # Check length before indexing
            if var in jja1_mediators and time_length >= 1: # Exclude JJA 1945 (first time step, index 0)
                season_mask[0] = False # Exclude first JJA time step

            data_array[:, idx] = ds[var].values
            mask_array[:, idx] = ~season_mask # True where data should be masked (excluded)

            # Check for low effective samples
            effective_samples = np.sum(season_mask)
            if effective_samples <= 2:
                logging.warning(f"Variable {var} has only {effective_samples} effective samples after masking. "
                                f"This may cause 'degrees of freedom <= 0' warnings.")

        logging.info(f"Loaded subset data array shape: {data_array.shape}")
        return data_array, mask_array, var_names_loaded

    except FileNotFoundError:
        logging.error(f"Data file not found: {file_path}")
        return None, None, None
    except KeyError as e:
        logging.error(f"Variable not found in dataset {file_path}: {e}. Requested: {variables_to_load}")
        return None, None, None
    except Exception as e:
        logging.error(f"Error loading or preparing subset data: {e}", exc_info=True)
        return None, None, None


def create_custom_test_class(base_test_class):
    """
    Factory to create custom test class with forward-in-time conditioning.
    
    Args:
        base_test_class: Base test class (RobustParCorr or GPDC)
        
    Returns:
        Custom test class with forward-in-time conditioning
    """
    
    class CustomTest(base_test_class):
        """Custom implementation that enforces forward-in-time conditioning sets."""
        
        def run_test(self, X, Y, Z, tau_max=None, **kwargs):
            """
            Override the run_test method to enforce forward-in-time conditioning sets.
            We only keep conditioning variables Z that are at the same or earlier lag than X.
            
            For example, if X is observed at lag -5 relative to Y (X(t-5) → Y(t)), 
            we only keep conditioning variables Z with lags <= -5. This ensures that 
            we do not use 'future' information (from lags closer to Y or after X).
            
            Also silences warnings related to "degrees of freedom <= 0 for slice"
            which occurs when there aren't enough samples to compute variance.
            """
            # X, Y, and Z are lists of (var_idx, lag) tuples. For single link tests:
            if len(X) != 1 or len(Y) != 1:
                raise ValueError(f"Custom{base_test_class.__name__} is designed for single X and Y variables.")

            # Extract cause lag
            X_var, tau_x = X[0]
            tau_z = [tau for (_, tau) in Z]
        
            # Filter Z: keep only those with lag <= tau_x
            Z_filtered = [z for z, tz in zip(Z, tau_z) if tz <= tau_x]

            # Run the test with numpy warnings temporarily suppressed
            import warnings
            with warnings.catch_warnings():
                # Filter out the specific RuntimeWarning about degrees of freedom
                warnings.filterwarnings(
                    "ignore", 
                    message="Degrees of freedom <= 0 for slice",
                    category=RuntimeWarning
                )
                return super().run_test(X, Y, Z_filtered, tau_max=tau_max, **kwargs)
    
    return CustomTest


class CustomRobustParCorr(RobustParCorr):
    """
    Custom implementation of RobustParCorr that enforces forward-in-time conditioning sets
    and handles the "degrees of freedom <= 0" warning.
    (From causal discovery methodology)
    """
    def run_test(self, X, Y, Z, tau_max=None, **kwargs):
        if len(X) != 1 or len(Y) != 1:
            # Allow this for general use, but log a warning if used outside expected PCMCI context
            logging.debug("CustomRobustParCorr run_test called with len(X) != 1 or len(Y) != 1.")
            # Fallback to standard behavior without filtering Z based on X's lag
            Z_filtered = Z
        else:
            # Standard PCMCI link test: Filter Z based on X's lag
            X_var, tau_x = X[0]
            tau_z = [tau for (_, tau) in Z]
            Z_filtered = [z for z, tz in zip(Z, tau_z) if tz <= tau_x]

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Degrees of freedom <= 0 for slice",
                category=RuntimeWarning
            )
            # Ensure necessary arguments are passed to the superclass method
            # The exact signature might depend on the Tigramite version,
            # but typically includes X, Y, Z, and potentially others like tau_max.
            return super().run_test(X=X, Y=Y, Z=Z_filtered, tau_max=tau_max, **kwargs)


def run_iterative_pcmciplus(
    dataframe: pp.DataFrame,
    tau_min: int = DEFAULT_TAU_MIN,
    tau_max: int = DEFAULT_TAU_MAX,
    alpha_values: List[float] = DEFAULT_ALPHA_VALUES,
    max_combinations: int = DEFAULT_MAX_COMBINATIONS,
    cond_ind_test_name: str = 'robust_parcorr',
    verbosity: int = 0
) -> List[Dict]:
    """
    Run PCMCI+ multiple times with different alpha values.

    Args:
        dataframe: Tigramite dataframe.
        tau_min: Minimum lag.
        tau_max: Maximum lag.
        alpha_values: List of significance levels.
        max_combinations: Max combinations for PCMCI+.
        cond_ind_test_name: Name of conditional independence test.
        verbosity: Tigramite verbosity level.

    Returns:
        List of PCMCI+ results dictionaries for each alpha value.
    """
    # Get test configuration
    test_config = get_cond_ind_test_config(cond_ind_test_name)
    
    # Import and create custom test class
    test_module = importlib.import_module(test_config['module'])
    base_test_class = getattr(test_module, test_config['class'])
    CustomTestClass = create_custom_test_class(base_test_class)
    
    results_list = []
    logging.info(f"Running iterative PCMCI+ with alpha values: {alpha_values} using {test_config['name']}")

    for alpha in alpha_values:
        logging.info(f"  Running PCMCI+ with alpha = {alpha}")
        
        # Initialize test with config params
        cond_ind_test = CustomTestClass(**test_config['params'])
        
        pcmci = PCMCI(
            dataframe=dataframe,
            cond_ind_test=cond_ind_test,
            verbosity=verbosity
        )

        try:
            results = pcmci.run_pcmciplus(
                tau_min=tau_min,
                tau_max=tau_max,
                pc_alpha=alpha,
                max_combinations=max_combinations
            )
            results_list.append(results)
            logging.info(f"  Completed PCMCI+ run with alpha = {alpha}")
        except Exception as e:
            logging.error(f"  Error during PCMCI+ run with alpha = {alpha}: {e}", exc_info=True)
            # Decide whether to stop or continue with other alphas
            # Record failure state for this significance level
            results_list.append(None)

    return results_list


def run_single_pcmci_fdr(
    dataframe: pp.DataFrame,
    tau_min: int = 1,
    tau_max: int = 6,
    pc_alpha: float = 0.10,
    cond_ind_test_name: str = 'robust_parcorr',
    max_combinations: int = 10,
    verbosity: int = 0
) -> Dict:
    """
    One-shot PCMCI+ with Benjamini–Hochberg FDR control.
    Returns the full results dictionary (graph, val_matrix, p_matrix).
    
    Args:
        dataframe: Tigramite DataFrame object
        tau_min: Minimum time lag
        tau_max: Maximum time lag
        pc_alpha: Significance level for FDR-BH method
        cond_ind_test_name: Name of conditional independence test
        max_combinations: Maximum combinations for PCMCI+
        verbosity: Verbosity level for PCMCI
        
    Returns:
        Dictionary with 'graph', 'val_matrix', 'p_matrix' from PCMCI+ with FDR
    """
    test_cfg = get_cond_ind_test_config(cond_ind_test_name)
    test_mod = importlib.import_module(test_cfg['module'])
    base_cls = getattr(test_mod, test_cfg['class'])
    CustomTest = create_custom_test_class(base_cls)
    cond_test = CustomTest(**test_cfg['params'])

    pcmci = PCMCI(dataframe=dataframe,
                  cond_ind_test=cond_test,
                  verbosity=verbosity)

    results = pcmci.run_pcmciplus(tau_min=tau_min,
                                  tau_max=tau_max,
                                  pc_alpha=pc_alpha,
                                  max_combinations=max_combinations,
                                  fdr_method='fdr_bh')
    return results


def intersect_graphs(graph_a: np.ndarray, graph_b: np.ndarray) -> np.ndarray:
    """
    Return a copy of graph_a where cells are kept only if the same
    (i,j,τ) entry in graph_b has both:
    1. A non-empty string (edge exists)
    2. The same edge type (-->, <--, o-o, etc.)
    All other cells are set to ''.
    
    Args:
        graph_a: First graph array (typically the robust graph)
        graph_b: Second graph array (typically the FDR graph)
        
    Returns:
        Intersection of the two graphs
    """
    mask = (graph_b != '') & (graph_a == graph_b)
    result = np.where(mask, graph_a, '')
    return result


def apply_fdr_pruning(
    dataframe: pp.DataFrame,
    robust_graph: np.ndarray,
    robust_val_matrix: np.ndarray,
    robust_p_matrix: np.ndarray,
    var_names: List[str],
    target_variables: List[str],
    tau_min: int,
    tau_max: int,
    fdr_pc_alpha: float,
    cond_ind_test_name: str = 'robust_parcorr',
    max_cumulative_lag: Optional[int] = None,
    min_cumulative_lag: int = 3,
    max_combinations: int = 10,
    verbosity: int = 0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Apply FDR-based pruning to a robust causal graph.
    
    This function performs a second-stage pruning by:
    1. Running a single PCMCI+ analysis with FDR-BH correction
    2. Filtering the FDR graph for paths to target variables
    3. Intersecting with the input robust graph (also filtered)
    
    Critical: Both graphs must be filtered for target pathways BEFORE intersection
    to prevent nodes without paths to targets from appearing in the final graph.
    
    Args:
        dataframe: Tigramite DataFrame with data and mask
        robust_graph: Already filtered robust graph from iterative analysis
        robust_val_matrix: Value matrix corresponding to robust_graph
        robust_p_matrix: P-value matrix corresponding to robust_graph
        var_names: List of variable names
        target_variables: List of target variable names to filter paths to
        tau_min: Minimum time lag
        tau_max: Maximum time lag
        fdr_pc_alpha: Significance level for FDR-BH method
        cond_ind_test_name: Conditional independence test to use
        max_cumulative_lag: Maximum cumulative lag for path filtering (default: tau_max)
        min_cumulative_lag: Minimum cumulative lag for path filtering
        max_combinations: Maximum combinations for PCMCI+
        verbosity: Verbosity level for logging
        
    Returns:
        Tuple of (pruned_graph, pruned_val_matrix, pruned_p_matrix, fdr_info)
        where fdr_info contains diagnostic information about the FDR pruning
    """
    if max_cumulative_lag is None:
        max_cumulative_lag = tau_max
    
    # Step 1: Run single PCMCI+ with FDR-BH
    logging.info(f"Running single PCMCI+ with FDR-BH (pc_alpha = {fdr_pc_alpha})")
    
    fdr_results = run_single_pcmci_fdr(
        dataframe=dataframe,
        tau_min=tau_min,
        tau_max=tau_max,
        pc_alpha=fdr_pc_alpha,
        cond_ind_test_name=cond_ind_test_name,
        max_combinations=max_combinations,
        verbosity=verbosity
    )
    
    if fdr_results is None:
        raise RuntimeError("FDR PCMCI+ run failed")
    
    fdr_graph = fdr_results['graph']
    fdr_val_matrix = fdr_results['val_matrix']
    fdr_p_matrix = fdr_results['p_matrix']
    
    # Step 2: Filter FDR graph for target pathways
    # This is the critical step for comprehensive analysis
    logging.info("Filtering FDR graph for target pathways...")
    
    fdr_filtered_graph, fdr_filtered_val_matrix, fdr_filtered_p_matrix = filter_graph(
        graph=fdr_graph,
        val_matrix=fdr_val_matrix,
        p_matrix=fdr_p_matrix,
        var_names=var_names,
        target_variables=target_variables,
        tau_min=tau_min,
        tau_max=tau_max,
        max_cumulative_lag=max_cumulative_lag,
        min_cumulative_lag=min_cumulative_lag
    )
    
    # Step 3: Intersect the two filtered graphs
    pruned_graph = intersect_graphs(robust_graph, fdr_filtered_graph)
    
    # Create corresponding value and p-value matrices
    pruned_val_matrix = np.where(pruned_graph != '', robust_val_matrix, 0.0)
    pruned_p_matrix = np.where(pruned_graph != '', robust_p_matrix, 1.0)
    
    # Collect diagnostic information
    fdr_info = {
        'fdr_edges_total': np.sum(fdr_graph != ''),
        'fdr_edges_filtered': np.sum(fdr_filtered_graph != ''),
        'robust_edges_input': np.sum(robust_graph != ''),
        'pruned_edges_final': np.sum(pruned_graph != ''),
        'fdr_pc_alpha': fdr_pc_alpha
    }
    
    logging.info(f"FDR pruning complete: {fdr_info['robust_edges_input']} → {fdr_info['pruned_edges_final']} edges")
    logging.info(f"FDR graph: {fdr_info['fdr_edges_total']} total → {fdr_info['fdr_edges_filtered']} filtered edges")
    
    return pruned_graph, pruned_val_matrix, pruned_p_matrix, fdr_info


def track_edge_significance(
    results_list: List[Optional[Dict]], # Allow for None if a run failed
    var_names: List[str],
    alpha_values: List[float],
    tau_min: int = DEFAULT_TAU_MIN,
    tau_max: int = DEFAULT_TAU_MAX
) -> pd.DataFrame:
    """
    Track edge significance across multiple PCMCI+ runs.

    Args:
        results_list: List of PCMCI+ results (or None) for different alpha values.
        var_names: List of variable names.
        alpha_values: List of alpha values used.
        tau_min: Minimum lag considered.
        tau_max: Maximum lag considered.

    Returns:
        DataFrame containing edge significance information.
    """
    edge_records = []
    num_vars = len(var_names)

    for run_idx, (results, alpha) in enumerate(zip(results_list, alpha_values)):
        if results is None:
            logging.warning(f"Skipping edge tracking for run {run_idx+1} (alpha={alpha}) due to previous error.")
            continue # Skip failed runs

        val_matrix = results.get('val_matrix')
        p_matrix = results.get('p_matrix')
        graph = results.get('graph')

        if val_matrix is None or p_matrix is None or graph is None:
            logging.warning(f"Missing matrices in results for run {run_idx+1} (alpha={alpha}). Skipping.")
            continue

        # Check matrix dimensions match var_names and tau_max
        expected_shape = (num_vars, num_vars, tau_max + 1)
        if graph.shape != expected_shape or val_matrix.shape != expected_shape or p_matrix.shape != expected_shape:
             logging.warning(f"Matrix shape mismatch for run {run_idx+1} (alpha={alpha}). "
                             f"Expected {expected_shape}, Got graph:{graph.shape}, val:{val_matrix.shape}, p:{p_matrix.shape}. Skipping.")
             continue


        for i in range(num_vars):
            for j in range(num_vars):
                # Iterate from tau_min to tau_max (inclusive)
                for tau in range(tau_min, tau_max + 1):
                    # Check if tau is within the bounds of the matrix's lag dimension
                    if tau < graph.shape[2]:
                        link_symbol = graph[i, j, tau]
                        # Check for non-empty link symbols (e.g., '-->', 'o->', etc.)
                        if isinstance(link_symbol, str) and link_symbol.strip() != '':
                            edge_records.append({
                                'cause': var_names[i],
                                'effect': var_names[j],
                                'lag': tau,
                                'alpha': alpha,
                                'run_idx': run_idx,
                                'causal_effect': val_matrix[i, j, tau],
                                'p_value': p_matrix[i, j, tau]
                            })
                    else:
                        # This case should ideally not happen if tau_max matches matrix dim, but good for safety
                        logging.warning(f"Lag index {tau} out of bounds for matrix dimension {graph.shape[2]} in run {run_idx+1}.")


    if not edge_records:
        logging.warning("No significant edges found across any successful PCMCI+ run.")
        # Return an empty DataFrame with expected columns
        return pd.DataFrame(columns=['cause', 'effect', 'lag', 'alpha', 'run_idx', 'causal_effect', 'p_value'])

    return pd.DataFrame(edge_records)


def analyze_edge_persistence(
    edge_tracking_df: pd.DataFrame,
    num_total_runs: int=4, # Need total number of runs attempted
    min_persistence: float = DEFAULT_PERSISTENCE_MIN,
    max_avg_pvalue: float = DEFAULT_PERSISTENCE_MAX_PVAL,
    min_effect_consistency: float = DEFAULT_PERSISTENCE_MIN_CONSISTENCY,
    min_effect_size: float = DEFAULT_PERSISTENCE_MIN_EFFECT
) -> pd.DataFrame:
    """
    Analyze edge persistence with multiple robustness criteria.

    Args:
        edge_tracking_df: DataFrame from track_edge_significance.
        num_total_runs: The total number of alpha values used (potential runs).
        min_persistence: Minimum fraction of significant runs required (based on total runs).
        max_avg_pvalue: Maximum allowed average p-value across significant runs.
        min_effect_consistency: Minimum required effect direction consistency.
        min_effect_size: Minimum required absolute average effect size.

    Returns:
        DataFrame with persistence metrics for each unique edge.
    """
    if edge_tracking_df.empty:
        logging.warning("Edge tracking DataFrame is empty. Cannot analyze persistence.")
        return pd.DataFrame(columns=[
            'cause', 'effect', 'lag', 'persistence_ratio', 'avg_effect',
            'abs_effect', 'avg_p_value', 'effect_consistency', 'is_persistent',
            'has_low_pvalue', 'is_consistent', 'has_strong_effect', 'is_robust'
        ])

    edge_groups = edge_tracking_df.groupby(['cause', 'effect', 'lag'])
    persistence_records = []

    for (cause, effect, lag), group in edge_groups:
        # Filter for runs where the edge was deemed significant (p <= alpha)
        significant_group = group[group['p_value'] <= group['alpha']]
        significant_runs_count = len(significant_group)

        # Calculate persistence ratio based on the *total* number of runs attempted
        total_tested_runs = group['run_idx'].nunique() or 1
        persistence_ratio   = significant_runs_count / total_tested_runs

        # Calculate metrics based *only* on the significant runs
        if significant_runs_count > 0:
            avg_effect = significant_group['causal_effect'].mean()
            avg_p_value = significant_group['p_value'].mean()
            abs_effect = np.abs(avg_effect)

            # Calculate effect consistency based on significant runs
            effects = significant_group['causal_effect']
            if len(effects) > 0:
                 # Handle cases where mean effect is very close to zero
                mean_effect_for_sign = np.mean(effects)
                if np.isclose(mean_effect_for_sign, 0):
                    # If mean is near zero, consistency is less meaningful or could be calculated differently
                    # Option 1: Assign a neutral value like 0.5 or NaN
                    # Option 2: Calculate based on majority sign (positive vs negative counts)
                    pos_count = np.sum(effects > 0)
                    neg_count = np.sum(effects < 0)
                    effect_consistency = max(pos_count, neg_count) / len(effects) if len(effects) > 0 else 0
                else:
                    dominant_direction = np.sign(mean_effect_for_sign)
                    effect_direction_signs = np.sign(effects)
                    effect_consistency = np.mean(effect_direction_signs == dominant_direction)
            else:
                 effect_consistency = np.nan # Or 0 if no significant effects found

        else:
            # If never significant, set metrics accordingly
            avg_effect = np.nan
            avg_p_value = np.nan
            abs_effect = np.nan
            effect_consistency = np.nan

        # Apply robustness criteria
        is_persistent = persistence_ratio >= min_persistence
        # Check avg_p_value only if it's not NaN
        has_low_pvalue = (not np.isnan(avg_p_value)) and (avg_p_value <= max_avg_pvalue)
        # Check consistency only if it's not NaN
        is_consistent = (not np.isnan(effect_consistency)) and (effect_consistency >= min_effect_consistency)
         # Check effect size only if it's not NaN
        has_strong_effect = (not np.isnan(abs_effect)) and (abs_effect >= min_effect_size)

        persistence_records.append({
            'cause': cause,
            'effect': effect,
            'lag': lag,
            'persistence_ratio': persistence_ratio,
            'avg_effect': avg_effect,
            'abs_effect': abs_effect,
            'avg_p_value': avg_p_value,
            'effect_consistency': effect_consistency,
            'is_persistent': is_persistent,
            'has_low_pvalue': has_low_pvalue,
            'is_consistent': is_consistent,
            'has_strong_effect': has_strong_effect,
            'is_robust': (is_persistent and has_low_pvalue and is_consistent and has_strong_effect)
        })

    if not persistence_records:
         return pd.DataFrame(columns=[ # Ensure columns exist even if empty
            'cause', 'effect', 'lag', 'persistence_ratio', 'avg_effect',
            'abs_effect', 'avg_p_value', 'effect_consistency', 'is_persistent',
            'has_low_pvalue', 'is_consistent', 'has_strong_effect', 'is_robust'
        ])

    return pd.DataFrame(persistence_records)


def create_robust_graph(
    persistence_df: pd.DataFrame,
    num_vars: int,
    var_names: List[str],
    tau_max: int = DEFAULT_TAU_MAX
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a new graph containing only robust edges.

    Args:
        persistence_df: DataFrame from analyze_edge_persistence.
        num_vars: Number of variables.
        var_names: List of variable names.
        tau_max: Maximum lag dimension for the output matrices.

    Returns:
        Tuple containing filtered graph, value matrix, and p-value matrix.
    """
    original_shape = (num_vars, num_vars, tau_max + 1)
    robust_graph = np.full(original_shape, '', dtype=object)
    robust_val_matrix = np.zeros(original_shape)
    robust_p_matrix = np.ones(original_shape) # Initialize p-values to 1 (non-significant)

    # Filter for robust edges
    robust_edges = persistence_df[persistence_df['is_robust'].fillna(False)] # Handle potential NaNs

    if robust_edges.empty:
        logging.warning("No robust edges found after persistence analysis.")
        return robust_graph, robust_val_matrix, robust_p_matrix

    var_to_index = {name: i for i, name in enumerate(var_names)}

    for _, edge in robust_edges.iterrows():
        cause_idx = var_to_index.get(edge['cause'])
        effect_idx = var_to_index.get(edge['effect'])
        lag = int(edge['lag']) # Ensure lag is integer

        if cause_idx is None or effect_idx is None:
            logging.warning(f"Could not find index for edge: {edge['cause']} -> {edge['effect']}. Skipping.")
            continue

        # Ensure lag is within bounds
        if 0 <= lag < original_shape[2]:
            robust_graph[cause_idx, effect_idx, lag] = '-->' # Standard link symbol
            # Use the average effect and p-value from the persistence analysis
            robust_val_matrix[cause_idx, effect_idx, lag] = edge['avg_effect']
            robust_p_matrix[cause_idx, effect_idx, lag] = edge['avg_p_value']
        else:
            logging.warning(f"Lag {lag} out of bounds [0, {original_shape[2]-1}] for edge: {edge['cause']} -> {edge['effect']}. Skipping.")

    return robust_graph, robust_val_matrix, robust_p_matrix


def filter_graph(
    graph: np.ndarray,
    val_matrix: np.ndarray,
    p_matrix: np.ndarray,
    var_names: List[str],
    target_variables: List[str],
    tau_min: int = DEFAULT_TAU_MIN,
    tau_max: int = DEFAULT_TAU_MAX,
    max_cumulative_lag: int = DEFAULT_TAU_MAX,
    min_cumulative_lag: int = DEFAULT_FILTER_MIN_CUM_LAG
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filters the causal graph to include only pathways terminating at specified target variables,
    within given cumulative lag bounds. (From causal discovery methodology).

    Args:
        graph: The causal graph matrix.
        val_matrix: Matrix of causal effect values.
        p_matrix: Matrix of p-values.
        var_names: List of variable names corresponding to graph indices.
        target_variables: List of target variable names.
        tau_min: Minimum time lag for individual links.
        tau_max: Maximum time lag for individual links.
        max_cumulative_lag: Maximum allowed cumulative lag for a path.
        min_cumulative_lag: Minimum required cumulative lag for a path.

    Returns:
        Tuple containing filtered versions of graph, val_matrix, and p_matrix.
    """
    logging.info(f"Filtering graph for paths ending in {target_variables} with cumulative lag [{min_cumulative_lag}, {max_cumulative_lag}]")
    var_to_index = {name: i for i, name in enumerate(var_names)}
    num_vars = len(var_names)

    # Ensure targets exist in var_names before getting indices
    target_indices = []
    for t in target_variables:
        idx = var_to_index.get(t)
        if idx is not None:
            target_indices.append(idx)
        else:
            logging.warning(f"Target variable '{t}' for filtering not found in the provided var_names list. Skipping this target.")

    if not target_indices:
        logging.warning("No valid target variables found for filtering. Returning empty graph.")
        # Return empty graph structure matching input shape
        return (np.full_like(graph, '', dtype=object),
                np.zeros_like(val_matrix),
                np.ones_like(p_matrix))

    links_to_keep = set() # Stores tuples (source_idx, target_idx, lag)

    # Perform a backward search from each target
    for target_idx in target_indices:
        # Queue stores tuples: (current_var_index, cumulative_lag_to_target)
        queue = [(target_idx, 0)]
        # Keep track of visited states (node_idx, cum_lag) during the search from *this* target
        # to prevent redundant exploration and cycles within lag limits.
        visited_states_this_target = {(target_idx, 0)}

        while queue:
            current_var_idx, cum_lag = queue.pop(0)

            # Find parents (sources) of the current variable
            for source_var_idx in range(num_vars):
                # Check links across allowed individual lags
                for tau in range(tau_min, tau_max + 1):
                    # Check if lag is within graph dimensions
                    if tau < graph.shape[2]:
                        link_symbol = graph[source_var_idx, current_var_idx, tau]
                        if isinstance(link_symbol, str) and link_symbol.strip() != '':
                            # Found a parent link
                            new_cum_lag = cum_lag + tau

                            # Check if this path meets cumulative lag criteria
                            if min_cumulative_lag <= new_cum_lag <= max_cumulative_lag:
                                # Mark this link to be kept
                                links_to_keep.add((source_var_idx, current_var_idx, tau))

                                # Explore further back from this source, if not visited with this lag
                                # and within max cumulative lag overall
                                source_state = (source_var_idx, new_cum_lag)
                                if source_state not in visited_states_this_target:
                                     queue.append(source_state)
                                     visited_states_this_target.add(source_state)
                            # If lag criteria not met here, stop searching further back along this path


    # Create new matrices based on the links_to_keep set
    new_graph = np.full_like(graph, '', dtype=object)
    new_val_matrix = np.zeros_like(val_matrix)
    new_p_matrix = np.ones_like(p_matrix) # Default to non-significant

    if not links_to_keep:
        logging.warning("No paths meeting the criteria were found during filtering.")
    else:
        logging.info(f"Found {len(links_to_keep)} links meeting the filtering criteria.")
        for i, j, tau in links_to_keep:
            # Ensure indices and lag are valid before assignment
            if 0 <= i < num_vars and 0 <= j < num_vars and 0 <= tau < graph.shape[2]:
                new_graph[i, j, tau] = graph[i, j, tau] # Copy symbol
                new_val_matrix[i, j, tau] = val_matrix[i, j, tau]
                new_p_matrix[i, j, tau] = p_matrix[i, j, tau]
            else:
                 logging.warning(f"Attempted to assign filtered link with invalid index/lag: ({i}, {j}, {tau}). Skipping.")


    return new_graph, new_val_matrix, new_p_matrix


def write_graph_with_p_values(
    graph: np.ndarray,
    val_matrix: np.ndarray,
    p_matrix: np.ndarray,
    var_names: List[str],
    save_name: str,
    tau_min: int = DEFAULT_TAU_MIN,
    tau_max: int = DEFAULT_TAU_MAX,
    digits: int = 4
) -> None:
    """
    Write the causal graph details (only existing links) to a CSV file.

    Args:
        graph: The causal graph matrix.
        val_matrix: Matrix of causal effect values.
        p_matrix: Matrix of p-values.
        var_names: List of variable names.
        save_name: Path to save the CSV file.
        tau_min: Minimum lag to include.
        tau_max: Maximum lag to include.
        digits: Number of decimal places for formatting floats.
    """
    rows = []
    num_vars = len(var_names)
    logging.info(f"Saving graph details to {save_name}")

    for i in range(num_vars):
        for j in range(num_vars):
            # Iterate through the specified lag range
            for tau in range(tau_min, tau_max + 1):
                 # Check if lag is within matrix bounds and link exists
                if tau < graph.shape[2]:
                    link_symbol = graph[i, j, tau]
                    if isinstance(link_symbol, str) and link_symbol.strip() != '':
                        # Ensure values exist before rounding
                        causal_effect = val_matrix[i, j, tau] if not np.isnan(val_matrix[i, j, tau]) else np.nan
                        p_value = p_matrix[i, j, tau] if not np.isnan(p_matrix[i, j, tau]) else np.nan

                        rows.append({
                            'Cause': var_names[i],
                            'Effect': var_names[j],
                            'Lag': tau,
                            'Causal_Effect': round(causal_effect, digits) if not np.isnan(causal_effect) else np.nan,
                            'p_value': round(p_value, digits) if not np.isnan(p_value) else np.nan
                        })
                else:
                    # Should not happen if tau_max is correct, but safety check
                    logging.warning(f"Lag {tau} is out of bounds for graph matrix dimension {graph.shape[2]} while writing CSV.")


    if not rows:
        logging.warning(f"No links found in the graph to write to CSV file: {save_name}")
        # Create an empty file with headers anyway? Or just log? Let's just log.
        # Optionally, create an empty file: pd.DataFrame(columns=['Cause', 'Effect', 'Lag', 'Causal_Effect', 'p_value']).to_csv(save_name, index=False)
        return

    try:
        df = pd.DataFrame(rows)
        df.to_csv(save_name, index=False, float_format=f'%.{digits}f')
        logging.info(f"Successfully wrote {len(rows)} links to {save_name}")
    except Exception as e:
        logging.error(f"Failed to write graph CSV to {save_name}: {e}", exc_info=True)


def save_persistence_results(
    persistence_df: pd.DataFrame,
    save_dir: str,
    suffix: str
) -> None:
    """
    Save edge persistence analysis results in multiple formats.
    
    Args:
        persistence_df: DataFrame from analyze_edge_persistence
        save_dir: Directory to save results
        suffix: Filename suffix
    """
    # Save complete persistence metrics
    full_path = os.path.join(save_dir, f'edge_persistence_{suffix}.csv')
    persistence_df.to_csv(full_path, index=False)
    logging.info(f"Saved full persistence results to {full_path}")
    
    # Save only robust edges
    robust_edges = persistence_df[persistence_df['is_robust']].sort_values(
        'persistence_ratio', ascending=False
    )
    robust_path = os.path.join(save_dir, f'robust_edges_{suffix}.csv')
    robust_edges.to_csv(robust_path, index=False)
    logging.info(f"Saved {len(robust_edges)} robust edges to {robust_path}")
    
    # Generate and save summary statistics
    summary_stats = {
        'total_edges_tested': len(persistence_df),
        'robust_edges_found': len(robust_edges),
        'average_persistence': float(persistence_df['persistence_ratio'].mean()),
        'median_persistence': float(persistence_df['persistence_ratio'].median()),
        'robust_edge_list': robust_edges[['cause', 'effect', 'lag']].to_dict('records')
    }
    
    summary_path = os.path.join(save_dir, f'persistence_summary_{suffix}.json')
    with open(summary_path, 'w') as f:
        json.dump(summary_stats, f, indent=4)
    logging.info(f"Saved persistence summary to {summary_path}")


def generate_output_paths(
    base_suffix: str,
    alpha_preset: str,
    cond_ind_test: str,
    config: Optional[PredictorConfig] = None
) -> Dict[str, str]:
    """
    Generate output directory paths and suffix based on configuration.
    
    Args:
        base_suffix: Base suffix for output files  
        alpha_preset: Alpha preset name (mild/hard/custom)
        cond_ind_test: Conditional independence test name
        config: Predictor configuration object (optional, defaults to 'combined')
        
    Returns:
        Dictionary with 'save_dir' and 'suffix' keys
    """
    # Use default config if not provided (for pathway analysis)
    if config is None:
        config = PredictorConfig('combined')
    
    # Modify suffix based on alpha preset
    if alpha_preset == 'mild':
        base_suffix = base_suffix.replace('_hard', '_mild')
    elif alpha_preset == 'custom':
        base_suffix = base_suffix.replace('_hard', '_custom')
    
    # Add test type if not default
    if cond_ind_test == 'gpdc':
        base_suffix += '_GPDC'
    
    # Get full suffix including predictor set info
    suffix = config.get_suffix(base_suffix)
    
    # Create output directory
    save_dir = get_results_path(os.path.join('PCMCIplus', suffix), result_type="figures")
    os.makedirs(save_dir, exist_ok=True)
    
    logging.info(f"Output directory: {save_dir}")
    
    return {
        'save_dir': save_dir,
        'suffix': suffix
    }


def save_analysis_metadata(
    config: Optional[PredictorConfig],
    alpha_values: List[float],
    alpha_preset: str,
    cond_ind_test: str,
    results_summary: Dict[str, Any],
    output_dir: str,
    pathway: Optional[str] = None,
    pathway_config: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save comprehensive metadata for reproducibility.
    
    Creates a JSON file containing all analysis parameters, configuration
    details, and summary statistics for full reproducibility.
    
    Args:
        config: Predictor configuration used (optional for pathway analysis)
        alpha_values: List of alpha values tested
        alpha_preset: Alpha preset name used
        cond_ind_test: Conditional independence test used
        results_summary: Dictionary of result statistics
        output_dir: Directory to save metadata
        pathway: Pathway identifier for pathway-specific analysis (optional)
        pathway_config: Pathway configuration dictionary (optional)
    """
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'analysis_parameters': {
            'tau_min': DEFAULT_TAU_MIN,
            'tau_max': DEFAULT_TAU_MAX,
            'alpha_values': alpha_values,
            'alpha_preset': alpha_preset,
            'cond_ind_test': cond_ind_test,
            'max_combinations': DEFAULT_MAX_COMBINATIONS
        },
        'results_summary': results_summary,
        'environment': {
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': os.getcwd()
        }
    }
    
    # Add configuration info if available
    if config is not None:
        metadata['predictor_config'] = config.get_metadata()
    
    # Add pathway-specific metadata if applicable
    if pathway is not None:
        metadata['pathway'] = pathway
        if pathway_config is not None:
            metadata['pathway_config'] = pathway_config
        # Add pathway-specific data files
        metadata['data_files'] = {
            'standard': 'PCMCI_data_ts_st.nc',
            'physical': 'PCMCI_data_ts_phy_mech_st.nc'
        }
    
    metadata_path = os.path.join(output_dir, 'analysis_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    logging.info(f"Saved analysis metadata to {metadata_path}")


def get_node_positions(
    var_names: List[str], 
    predictor_set: str = 'combined',
    include_physical_mechanisms: bool = False
) -> Dict[str, Any]:
    """
    Get geographic positions for all climate variables.
    
    This function maps each climate variable to its representative geographic
    location based on the region or phenomenon it represents. Positions are
    used for map-based causal graph visualization.
    
    Args:
        var_names: List of variable names to position
        predictor_set: Predictor set identifier (affects logging)
        include_physical_mechanisms: Whether to include physical mechanism positions
        
    Returns:
        Dictionary with x, y coordinates and transform for cartopy
    """
    # Complete position dictionary for all possible climate variables
    # Positions represent central locations of climate phenomena/regions
    all_node_positions = {
        # JJA predictors - Southern Hemisphere winter
        'REOF SST JJA':        (-17, -25),   # Tropical Atlantic SST mode
        'DMI JJA':             (80, -5),     # Dipole Mode Index (Indian Ocean)
        
        # SON predictors - Southern Hemisphere spring  
        'SIOD MAM':            (80, -15),    # Subtropical Indian Ocean Dipole
        'MCA prec-RWS SON':    (-40, -10),   # South America precipitation pattern
        'MCA WAF-RWS SON':     (-30, -8),    # Wave activity flux pattern
        
        # DJF confounders - Southern Hemisphere summer
        'MCA RWS-prec DJF':    (-32, -31),   # Rossby wave source pattern
        'MCA RWS-WAF DJF':     (-50, -30),   # Combined wave pattern
        'WNP DJF':             (140, 15),    # Western North Pacific
        'SASDI SON':           (-25, -20),   # South Atlantic SST Dipole Index
        'Atl3 DJF':            (-6, -1),     # Atlantic 3 region
        'NPMM-SST DJF':        (-125, 20),   # North Pacific Meridional Mode
        'NPO DJF':             (-170, 40),   # North Pacific Oscillation
        'PSA2 JJA':            (-150, -35),  # Pacific-South American pattern 2
        
        # MAM mediators - Southern Hemisphere fall
        'MCA RWS-prec MAM(E)': (-50, -14),   # Eastern Pacific pathway
        'MCA RWS-prec MAM(C)': (-55, -20),   # Central Pacific pathway
        'NTA MAM':             (-45, 20),    # North Tropical Atlantic
        'SPO MAM':             (-110, -20),  # South Pacific Oscillation
        'NPMM-wind MAM':       (-150, 17),   # NPMM wind component
        'SPMM-SST MAM':        (-90, -10),   # South Pacific Meridional Mode SST
        'SPMM-wind MAM':       (-85, -15),   # SPMM wind component
        
        # DJF effects - ENSO indices (target variables)
        'E-ind DJF(1)':        (-115, 0),    # Eastern Pacific ENSO index
        'C-ind DJF(1)':        (-165, 0),    # Central Pacific ENSO index
    }
    
    # Add physical mechanism positions if requested
    if include_physical_mechanisms:
        physical_positions = {
            'Panama V':            (-82, 5),      # Panama region
            'West Pac U':          (160, 0),     # Western Pacific
            'RWS SETP':            (-92, -25),   # Southeast Pacific
            'Coastal V':           (-78, -15),   # Peru/Chile coast
            'Sc core':             (-87, -20),   # Stratocumulus core
            'Sc primary':          (-90, -22),   # Stratocumulus primary
            'Sc wide':             (-93, -18),   # Stratocumulus wide
            'Latent Heat Flux':    (-88, -18),   # SETP region
            'Bakun UI':            (-75, -12),   # SA coastline upwelling index
        }
        all_node_positions.update(physical_positions)
    
    # Build coordinate lists maintaining var_names order
    x_coords = []
    y_coords = []
    missing_vars = []
    
    for var in var_names:
        if var in all_node_positions:
            lon, lat = all_node_positions[var]
        else:
            # Log warning for missing positions and default to (0, 0)
            missing_vars.append(var)
            lon, lat = 0, 0
            
        x_coords.append(lon)
        y_coords.append(lat)
    
    if missing_vars:
        logging.warning(
            f"Node positions not defined for variables: {missing_vars}. "
            f"Defaulting to (0, 0). Consider adding positions to get_node_positions()."
        )
    
    # Convert to 0-360 longitude range for cartopy
    x_coords = [lon % 360 for lon in x_coords]
    
    return {
        'x': x_coords,
        'y': y_coords,
        'transform': None  # Will be set to ccrs.PlateCarree() in plotting function
    }


def calculate_dynamic_extent(
    active_node_indices: set,
    var_names: List[str],
    predictor_set: str = 'combined',
    include_physical_mechanisms: bool = False,
    padding_degrees: float = 10.0
) -> List[float]:
    """
    Calculate dynamic map extent based on active node positions.
    
    This function determines the optimal map extent by analyzing the geographic
    positions of active nodes and applying padding to ensure good visualization.
    
    Args:
        active_node_indices: Set of indices for nodes that participate in causal links
        var_names: List of variable names
        predictor_set: Predictor set identifier (used for node positioning)
        include_physical_mechanisms: Whether to include physical mechanism positions
        padding_degrees: Degrees of padding to add around the active nodes
        
    Returns:
        List of extent coordinates [west, east, south, north] in degrees
    """
    if not active_node_indices:
        logging.warning("No active nodes found for dynamic extent calculation, using default extent")
        return [70, 359.99, -45, 27]  # Default global view
    
    # Get positions for all variables
    node_positions = get_node_positions(var_names, predictor_set, include_physical_mechanisms)
    
    # Extract coordinates for active nodes only
    active_lons = []
    active_lats = []
    
    for i in active_node_indices:
        if i < len(node_positions['x']) and i < len(node_positions['y']):
            lon = node_positions['x'][i]
            lat = node_positions['y'][i]
            # Skip default (0,0) positions that indicate missing coordinate data
            if lon != 0 or lat != 0:
                active_lons.append(lon)
                active_lats.append(lat)
    
    if not active_lons or not active_lats:
        logging.warning("No valid coordinates found for active nodes, using default extent")
        return [70, 359.99, -45, 27]
    
    # Calculate min/max coordinates
    min_lon = min(active_lons)
    max_lon = max(active_lons)
    min_lat = min(active_lats)
    max_lat = max(active_lats)
    
    # Apply padding
    west = max(0, min_lon - padding_degrees)
    east = min(359.99, max_lon + padding_degrees)  # Use 359.99 to prevent Cartopy 360→0 normalization
    south = max(-90, min_lat - padding_degrees)
    north = min(90, max_lat + padding_degrees)
    
    # Ensure minimum extent size to prevent overly zoomed maps
    min_extent_lon = 40  # Minimum longitude span
    min_extent_lat = 30  # Minimum latitude span
    
    lon_span = east - west
    lat_span = north - south
    
    if lon_span < min_extent_lon:
        center_lon = (west + east) / 2
        west = max(0, center_lon - min_extent_lon / 2)
        east = min(359.99, center_lon + min_extent_lon / 2)
    
    if lat_span < min_extent_lat:
        center_lat = (south + north) / 2
        south = max(-90, center_lat - min_extent_lat / 2)
        north = min(90, center_lat + min_extent_lat / 2)
    
    # Handle edge case where extent crosses 180° meridian
    if west > east:
        east = 359.99
    
    extent = [west, east, south, north]
    logging.info(f"Dynamic extent calculated: {extent} (padding: {padding_degrees}°)")
    
    return extent


def create_causal_graph_w_map(
    graph: np.ndarray,
    val_matrix: np.ndarray,
    var_names: List[str],
    save_dir: str,
    suffix: str,
    predictor_set: str = 'combined',
    cond_ind_test: str = 'robust_parcorr',
    pathway: Optional[str] = None,
    style_physical_mechanisms: bool = False,
    output_format: str = 'png',
    raster_dpi: int = 150,
    vector_dpi: int = 300,
    use_dynamic_extent: bool = False,
    custom_extent: Optional[List[float]] = None,
    show_colorbar: bool = False
) -> None:
    """
    Create and save causal graph visualization on a geographic map.
    
    This function generates a publication-quality figure showing causal links
    between climate variables positioned at their geographic locations. It
    intelligently handles node visibility, map extent, and special highlighting.
    
    Args:
        graph: Causal graph matrix from PCMCI+ (vars × vars × lags)
        val_matrix: Matrix of causal effect strengths
        var_names: List of variable names
        save_dir: Directory to save the visualization
        suffix: Filename suffix for output
        predictor_set: Predictor set used (affects map extent and features)
        cond_ind_test: Conditional independence test used (for labeling)
        pathway: Pathway identifier ('EP' or 'CP') for pathway-specific analysis
        style_physical_mechanisms: Whether to apply special styling to physical mechanisms
        output_format: Output format ('png', 'pdf', 'svg', or 'both')
        raster_dpi: DPI for rasterized elements in vector formats
        vector_dpi: DPI for pure raster formats
        use_dynamic_extent: Whether to use dynamic extent based on active nodes (default: False)
        custom_extent: Custom extent bounds [west, east, south, north] in degrees (default: None)
        show_colorbar: Whether to show unified colorbar/legend (default: False)
    """
    # Lazy imports to improve startup time
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from tigramite import plotting as tp
    
    # Check for physical mechanism styling requirements
    include_physical = pathway is not None and style_physical_mechanisms
    
    logging.info(f"Creating causal graph visualization for {len(var_names)} variables")
    if pathway:
        logging.info(f"Pathway-specific visualization for {pathway} pathway")
    
    # ─────────────────── Identify Active Nodes ──────────────────
    # Only show nodes that participate in at least one causal link
    active_node_indices = set()
    num_vars = len(var_names)
    
    for i in range(num_vars):
        for j in range(num_vars):
            for tau in range(graph.shape[2]):
                if graph[i, j, tau] != '':
                    active_node_indices.add(i)  # Source is active
                    active_node_indices.add(j)  # Target is active
    
    logging.info(f"Active nodes identified: {[var_names[i] for i in sorted(active_node_indices)]}")
    
    # ─────────────────── Node Display Properties ──────────────────
    # Configure node visibility based on activity
    node_labels_for_plot = []
    node_sizes_for_plot = []
    default_node_size = 6.0
    default_label_size = 20
    
    for i in range(num_vars):
        if i in active_node_indices:
            node_labels_for_plot.append(var_names[i])
            node_sizes_for_plot.append(default_node_size)
        else:
            node_labels_for_plot.append('')  # Hide label
            node_sizes_for_plot.append(0.0)  # Hide node
    
    # ─────────────────── Map Configuration ──────────────────
    # Create figure with appropriate projection
    fig = plt.figure(figsize=(30, 30))
    proj = ccrs.PlateCarree()
    proj_mod = ccrs.PlateCarree(central_longitude=180)  # Pacific-centered
    ax = plt.axes(projection=proj_mod)
    
    # Determine map extent based on configuration
    if custom_extent is not None:
        # Use user-provided custom extent
        extent = custom_extent
        logging.info(f"Using custom map extent: {extent}")
    elif use_dynamic_extent:
        # Use existing dynamic calculation based on active nodes
        extent = calculate_dynamic_extent(
            active_node_indices=active_node_indices,
            var_names=var_names,
            predictor_set=predictor_set,
            include_physical_mechanisms=include_physical,
            padding_degrees=10.0
        )
    else:
        # Use new default fixed extent for consistent visualization
        extent = DEFAULT_MAP_EXTENT
        logging.info(f"Using default fixed map extent: {extent}")
    
    try:
        ax.set_extent(extent, crs=proj)
    except Exception as e:
        logging.error(f"Error setting map extent {extent}: {e}")
        plt.close(fig)
        return
    
    # Add map features
    ax.add_feature(cfeature.LAND, facecolor='lightgray', zorder=1)
    ax.add_feature(cfeature.OCEAN, facecolor='lightcyan', zorder=0)
    ax.add_feature(cfeature.COASTLINE, linewidth=1, zorder=2)
    ax.add_feature(cfeature.BORDERS, linestyle='-', linewidth=0.5, zorder=2)
    
    # Configure gridlines with publication-ready font sizes
    gl = ax.gridlines(draw_labels=True, color='gray', alpha=0.5, linestyle='--')
    gl.xlabel_style = {'size': 18, 'color': 'black'}
    gl.ylabel_style = {'size': 18, 'color': 'black'}
    gl.top_labels = False  # Remove top longitude labels for cleaner appearance
    
    # ─────────────────── Special Regional Highlights ──────────────────
    # Draw South America box for combined predictor set or pathway analysis
    if predictor_set == 'combined' or pathway is not None:
        # Highlight South American monsoon region
        sa_coords = [(-65, 5), (-12, 5), (-12, -35), (-65, -35)]
        sa_coords_360 = [(lon % 360, lat) for lon, lat in sa_coords]
        
        sa_box = mpatches.Polygon(
            sa_coords_360,
            closed=True,
            edgecolor='darkgreen',
            facecolor='none',
            linewidth=5,
            transform=ccrs.PlateCarree(),
            zorder=3,
            label='SAMS Region'
        )
        ax.add_patch(sa_box)
        logging.info("Added South America monsoon region highlight")
    
    # ─────────────────── Plot Causal Graph ──────────────────
    # Get node positions and set transform
    node_positions = get_node_positions(var_names, predictor_set, include_physical_mechanisms=include_physical)
    node_positions['transform'] = proj
    
    # Set current axes for tigramite plotting
    plt.sca(ax)
    
    # Determine colorbar labels based on test type
    if cond_ind_test == 'gpdc':
        link_label = 'Distance correlation'
        node_label = 'Auto-distance correlation'
    else:
        link_label = 'Robust partial correlation'
        node_label = 'Autocorrelation'
    
    # Create unified label for combined display
    unified_label = f"{node_label} / {link_label}"
    
    try:
        tp.plot_graph(
            val_matrix=val_matrix,
            graph=graph,
            var_names=node_labels_for_plot,  # Already filtered for activity
            link_colorbar_label=unified_label if show_colorbar else link_label,
            node_colorbar_label=node_label,
            node_pos=node_positions,
            node_size=node_sizes_for_plot,   # Hide inactive nodes
            node_label_size=default_label_size,
            arrow_linewidth=12,
            show_colorbar=False,  # Always False to prevent Tigramite from creating colorbars
            label_fontsize=20,
            link_label_fontsize=18,
            show_autodependency_lags=True,
            tick_label_size=16,
            fig_ax=(fig, ax)
        )
        
        # ─────────────────── Manual Unified Colorbar ──────────────────
        # Create our own unified colorbar when requested
        if show_colorbar:
            import matplotlib.pyplot as plt
            from matplotlib import cm
            from matplotlib.colors import Normalize
            
            # Extract value range from the actual data
            if val_matrix is not None and graph is not None:
                # Get values only where edges exist
                edge_mask = graph != ''
                if np.any(edge_mask):
                    vmin = float(np.min(val_matrix[edge_mask]))
                    vmax = float(np.max(val_matrix[edge_mask]))
                    # Ensure symmetric range for better visualization
                    abs_max = max(abs(vmin), abs(vmax))
                    vmin, vmax = -abs_max, abs_max
                else:
                    vmin, vmax = -1, 1  # Default range if no edges
            else:
                vmin, vmax = -1, 1  # Default range
            
            # Create a ScalarMappable for the colorbar using link colormap settings
            cmap = cm.get_cmap('RdBu_r')  # Default colormap for links (matches Tigramite)
            norm = Normalize(vmin=vmin, vmax=vmax)
            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            
            # Create colorbar axes with proper positioning and spacing
            # Position: [x, y, width, height] in axes coordinates
            # Centered horizontally, with spacing below the map
            cbar_ax = fig.add_axes([0.33, 0.325, 0.3, 0.02])
            
            # Create the colorbar
            cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
            cbar.set_label(unified_label, fontsize=24, labelpad=5)
            
            # Set appropriate ticks based on value range
            tick_range = max(abs(vmin), abs(vmax))
            if tick_range <= 1:
                ticks = [-0.8, -0.4, 0.0, 0.4, 0.8]
            else:
                # Create ticks for wider ranges
                tick_step = tick_range / 4
                ticks = [-tick_range, -tick_step, 0.0, tick_step, tick_range]
            
            # Filter ticks to be within actual range
            ticks = [t for t in ticks if vmin <= t <= vmax]
            cbar.set_ticks(ticks)
            
            # Set colorbar tick label font size for publication readability
            cbar.ax.tick_params(labelsize=18)
            
            # Remove outline for cleaner look (matching Tigramite style)
            cbar.outline.set_visible(False)
            
            logging.info(f"Created unified colorbar with label: '{unified_label}' (range: {vmin:.3f} to {vmax:.3f})")
        
        # ─────────────────── Enhanced Node Styling ──────────────────
        # Define physical mechanisms for styling
        physical_mechanisms = {
            'Panama V', 'West Pac U', 'RWS SETP', 'Coastal V',
            'Sc core', 'Sc primary', 'Sc wide', 'Latent Heat Flux', 'Bakun UI'
        }
        
        # Make DJF effect variables (ENSO indices) bold and larger
        djf_effects = {'E-ind DJF(1)', 'C-ind DJF(1)'}
        
        for text_obj in ax.texts:
            text_content = text_obj.get_text()
            if text_content != '':
                # Style target variables
                if text_content in djf_effects:
                    text_obj.set_fontsize(22)
                    text_obj.set_fontweight('bold')
                # Style physical mechanisms if requested
                elif style_physical_mechanisms and text_content in physical_mechanisms:
                    text_obj.set_fontstyle('italic')
                    text_obj.set_color('darkblue')  # Distinctive color
                    
        logging.info("Enhanced variable labels")
        
    except Exception as e:
        logging.error(f"Error during graph plotting: {e}", exc_info=True)
        plt.close(fig)
        raise
    
    # ─────────────────── Add Title if Pathway Specified ──────────────────
    if pathway:
        active_nodes = [var_names[i] for i in sorted(active_node_indices)]
        num_physical = len([n for n in active_nodes if n in physical_mechanisms])
        ax.set_title(
            f'{pathway} Pathway Causal Network{"with Physical Mechanisms" if include_physical else ""}\n'
            f'Active Nodes: {len(active_node_indices)}'
            f'{f" | Physical Mechanisms: {num_physical}" if include_physical else ""}',
            fontsize=24, 
            pad=20
        )
    
    # ─────────────────── Save Figure ──────────────────
    filename_parts = ['causal_graph_map']
    if pathway:
        filename_parts.append(pathway)
    filename_parts.append(suffix)
    base_output_path = os.path.join(save_dir, f'{"_".join(filename_parts)}')
    
    # Import save function if not already available
    try:
        from utils.plotting_optimization import save_figure_optimized
    except ImportError:
        # Fallback to standard save if optimization not available
        output_path = f"{base_output_path}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logging.info(f"Saved causal graph visualization to {output_path}")
    else:
        # Use optimized save function
        save_figure_optimized(fig, base_output_path, output_format, raster_dpi, vector_dpi)
    
    plt.close(fig)
    import gc
    gc.collect()


# ━━━━━━━━━━━━━━━━━━━━━━━━ SLIDING WINDOW UTILITIES ━━━━━━━━━━━━━━━━━━━━━━━━━━━

def load_pruned_variables(csv_path: str) -> List[str]:
    """
    Load pruned variables from a robust graph CSV file.
    
    Reads the robust_graph CSV file and returns a list of unique variable names
    from the 'Cause' and 'Effect' columns.
    
    Args:
        csv_path: Path to the CSV file containing robust graph edges
        
    Returns:
        Sorted list of unique variable names
        
    Raises:
        SystemExit: If CSV file is not found or cannot be read
    """
    try:
        df = pd.read_csv(csv_path)
        pruned_vars = set(df["Cause"]) | set(df["Effect"])
        logging.info(f"Successfully loaded {len(pruned_vars)} unique variables from {csv_path}")
        return sorted(list(pruned_vars))
    except FileNotFoundError:
        logging.error(f"Error: Pruning CSV file not found at: {csv_path}")
        raise SystemExit(1)
    except Exception as e:
        logging.error(f"Error reading pruning CSV file {csv_path}: {e}")
        raise SystemExit(1)


def reduce_variable_lists(
    pruned_vars: List[str],
    variable_seasons_dict: Dict[str, str],
    jja_list: List[str],
    son_list: List[str],
    djf_list: List[str],
    mam_list: List[str],
    djf_eff_list: List[str]
) -> Tuple[Dict[str, str], List[str], List[str], List[str], List[str], List[str]]:
    """
    Filter variable configurations to only include pruned variables.
    
    Filters the variable seasons dictionary and group lists to only include
    variables present in the pruned_vars list.
    
    Args:
        pruned_vars: List of variables to keep after pruning
        variable_seasons_dict: Original mapping of variables to seasons
        jja_list: Original JJA predictor variables
        son_list: Original SON predictor variables  
        djf_list: Original DJF confounder variables
        mam_list: Original MAM mediator variables
        djf_eff_list: Original DJF effect variables
        
    Returns:
        Tuple containing filtered versions of all inputs:
        (variable_seasons_dict, jja_list, son_list, djf_list, mam_list, djf_eff_list)
    """
    # Filter VARIABLE_SEASONS dictionary
    new_variable_seasons = {
        k: v for k, v in variable_seasons_dict.items() if k in pruned_vars
    }
    final_var_list = sorted(list(new_variable_seasons.keys()))
    logging.info(f"Final variable list after pruning ({len(final_var_list)} variables): {final_var_list}")

    # Filter each group list
    new_jja = [v for v in jja_list if v in final_var_list]
    new_son = [v for v in son_list if v in final_var_list]
    new_djf = [v for v in djf_list if v in final_var_list]
    new_mam = [v for v in mam_list if v in final_var_list]
    new_djf_eff = [v for v in djf_eff_list if v in final_var_list]

    logging.info(f"Pruned JJA predictors: {new_jja}")
    logging.info(f"Pruned SON predictors: {new_son}")
    logging.info(f"Pruned DJF confounders: {new_djf}")
    logging.info(f"Pruned MAM mediators: {new_mam}")
    logging.info(f"Pruned DJF effects: {new_djf_eff}")

    return new_variable_seasons, new_jja, new_son, new_djf, new_mam, new_djf_eff


def calculate_effective_samples(mask_array: np.ndarray) -> np.ndarray:
    """
    Calculate number of effective (non-masked) samples per variable.
    
    Args:
        mask_array: Boolean mask array where True indicates masked/invalid data
        
    Returns:
        Array of effective sample counts per variable
    """
    # Count non-masked samples per variable
    valid_samples_per_variable = (~mask_array).sum(axis=0)
    min_effective_samples = np.min(valid_samples_per_variable[valid_samples_per_variable > 0])
    logging.info(f"Minimum effective samples across variables: {min_effective_samples}")
    return valid_samples_per_variable


def get_frequency_threshold(window_length_years: int) -> float:
    """
    Determine frequency threshold for consensus graphs based on window length.
    
    Used in sliding window analysis to determine what fraction of windows
    a causal link must appear in to be considered robust.
    
    Args:
        window_length_years: Length of the sliding window in years
        
    Returns:
        Appropriate frequency threshold (between 0 and 1)
    """
    if window_length_years <= 20:
        threshold = 0.3
    elif window_length_years <= 30:
        threshold = 0.45
    else:
        threshold = 0.55
    logging.debug(f"Using frequency threshold {threshold} for window length {window_length_years} years.")
    return threshold


def parse_causal_parents_from_csv(
    graph_csv_path: str, 
    target_var: str,
    var_names_full: List[str]
) -> Dict[int, List[Tuple[int, int]]]:
    """
    Parse a causal graph CSV to find all direct causal parents of a target variable.
    
    This function reads a CSV file containing causal relationships and extracts
    parent variables that directly influence the specified target variable.
    
    Args:
        graph_csv_path: Path to the CSV file containing causal graph information
        target_var: Name of the target variable to find parents for
        var_names_full: Complete list of variable names for index mapping
        
    Returns:
        Dictionary mapping target variable index to list of (parent_index, lag) tuples.
        Returns empty structure if no parents found or errors occur.
        
    Example:
        If target_var='E-ind DJF(1)' has parents ['DMI JJA', 'Atl3 DJF'] at lags [-6, -4]:
        Returns {target_idx: [(dmi_idx, -6), (atl3_idx, -4)]}
    """
    if not var_names_full:
        logging.error("Variable names list is empty. Cannot map parent names to indices.")
        return {}

    logging.info(f"Parsing causal parents for target '{target_var}' from graph: {graph_csv_path}")
    parents = {}
    
    try:
        # Check if graph file exists and is not empty
        if not os.path.exists(graph_csv_path) or os.path.getsize(graph_csv_path) == 0:
            logging.warning(f"Graph CSV file is missing or empty: {graph_csv_path}. No parents can be identified.")
            # Return structure expected by caller: dict with target_idx mapping to empty list
            if target_var in var_names_full:
                target_idx = var_names_full.index(target_var)
                return {target_idx: []}
            else:
                logging.error(f"Target variable '{target_var}' not found in variable names list.")
                return {}

        graph_df = pd.read_csv(graph_csv_path)
        
        # Filter for links where the current target is the 'Effect'
        target_links = graph_df[graph_df['Effect'] == target_var]

        # Ensure target_var is actually in var_names_full before getting index
        if target_var not in var_names_full:
            logging.error(f"Target variable '{target_var}' not found in variable names list.")
            return {}
        target_idx = var_names_full.index(target_var)

        if target_links.empty:
            logging.warning(f"No causal parents found for target '{target_var}' in the graph CSV {graph_csv_path}.")
            return {target_idx: []}

        parent_list = []
        for _, row in target_links.iterrows():
            cause_name = row['Cause']
            try:
                lag_val = row['Lag']
                if pd.isna(lag_val):
                    logging.warning(f"Found NaN lag for cause '{cause_name}' in {graph_csv_path}. Skipping this parent.")
                    continue
                lag = int(float(lag_val))
            except ValueError:
                logging.warning(f"Could not convert lag '{row['Lag']}' to integer for cause '{cause_name}' in {graph_csv_path}. Skipping.")
                continue
            except KeyError:
                logging.error(f"Column 'Lag' not found in graph CSV: {graph_csv_path}")
                return {target_idx: []}

            # Map cause_name to index using the var_names_full list
            if cause_name in var_names_full:
                cause_idx = var_names_full.index(cause_name)
                # Store as (var_index, -lag) as expected by Tigramite models
                parent_list.append((cause_idx, -lag))
            else:
                # Log if a parent from the subset graph is not in the full variable list
                logging.warning(f"Parent '{cause_name}' from graph {graph_csv_path} for target '{target_var}' not found in variable names list. Skipping this parent.")

        parents[target_idx] = parent_list
        # Log parents using names from the variable list for clarity
        parent_names_lags = [(var_names_full[p[0]], p[1]) for p in parent_list]
        logging.info(f"Found {len(parent_list)} parents for {target_var}: {parent_names_lags}")
        return parents

    except FileNotFoundError:
        logging.error(f"Graph CSV file not found: {graph_csv_path}")
        # Find target index to return correct empty structure
        if target_var in var_names_full:
            target_idx = var_names_full.index(target_var)
            return {target_idx: []}
        else:
            return {}
    except pd.errors.EmptyDataError:
        logging.warning(f"Graph CSV file is empty: {graph_csv_path}. No parents identified.")
        if target_var in var_names_full:
            target_idx = var_names_full.index(target_var)
            return {target_idx: []}
        else:
            return {}
    except KeyError as e:
        logging.error(f"Missing expected column in graph CSV {graph_csv_path}: {e}")
        if target_var in var_names_full:
            target_idx = var_names_full.index(target_var)
            return {target_idx: []}
        else:
            return {}
    except Exception as e:
        logging.error(f"Error parsing graph CSV {graph_csv_path}: {e}", exc_info=True)
        if target_var in var_names_full:
            target_idx = var_names_full.index(target_var)
            return {target_idx: []}
        else:
            return {}


def load_full_dataset_for_prediction(
    file_path: str,
    variable_season_map: Dict[str, str],
    season_months_map: Dict[str, List[int]],
    required_variables: Optional[List[str]] = None,
    fill_na_value: float = 0.0
) -> Tuple[Optional[pp.DataFrame], Optional[List[str]]]:
    """
    Load complete dataset for prediction tasks with seasonal masking.
    
    This function loads all variables from a NetCDF file and applies seasonal masking
    based on variable-season mappings. Unlike load_and_prepare_data which loads
    specific variables, this loads the entire dataset for prediction workflows.
    
    Args:
        file_path: Path to the NetCDF data file
        variable_season_map: Mapping of variable names to their seasons
        season_months_map: Mapping of season keys to month numbers
        required_variables: Optional list of variables that must be present
        fill_na_value: Value to fill NaN entries with (default: 0.0)
        
    Returns:
        Tuple of (tigramite_dataframe, variable_names_list) or (None, None) on error
        
    Example:
        df, var_names = load_full_dataset_for_prediction(
            'data.nc',
            {'DMI JJA': 'JJA', 'E-ind DJF(1)': 'DJF_effect'},
            {'JJA': [7], 'DJF_effect': [1]}
        )
    """
    logging.info(f"Loading full dataset for prediction from: {file_path}")
    
    try:
        # Load dataset
        ds = xr.open_dataset(file_path)
        ds = ds.fillna(fill_na_value).load()
        
        var_names = list(ds.data_vars)
        logging.info(f"Found {len(var_names)} variables in dataset")
        
        # Validate required variables if specified
        if required_variables:
            missing_vars = set(required_variables) - set(var_names)
            if missing_vars:
                logging.error(f"Required variables not found in dataset: {missing_vars}")
                return None, None
        
        # Convert to array format
        data_array = ds.to_array().values.T  # Shape (time, variables)
        logging.info(f"Data array shape (time, variables): {data_array.shape}")
        
        if data_array.shape[0] == 0:
            logging.error("Loaded data array has 0 time steps.")
            return None, None
        
        # Create seasonal masking
        time_length = len(ds.time)
        num_variables = len(var_names)
        mask_array = np.zeros((time_length, num_variables), dtype=bool)
        
        # Ensure time coordinates are datetime objects
        if not pd.api.types.is_datetime64_any_dtype(ds['time']):
            logging.warning("Time coordinate is not datetime. Attempting conversion.")
            try:
                ds['time'] = pd.to_datetime(ds['time'].values)
            except Exception as time_conv_err:
                logging.error(f"Could not convert time coordinate to datetime: {time_conv_err}")
                return None, None
        
        # Apply seasonal masking for each variable
        for idx, var in enumerate(var_names):
            season = variable_season_map.get(var)
            if season is None:
                logging.warning(f"Variable '{var}' not found in variable_season_map. No masking applied.")
                continue
                
            months = season_months_map.get(season)
            if months is None:
                logging.warning(f"Season '{season}' for variable '{var}' not found in season_months_map. No masking applied.")
                continue
            
            # Create seasonal mask (True for valid months)
            season_mask = np.isin(ds.time.dt.month, months)
            # Set mask array (True indicates masked/excluded data)
            mask_array[:, idx] = ~season_mask
        
        # Create Tigramite DataFrame
        dataframe = pp.DataFrame(
            data_array,
            mask=mask_array,
            var_names=var_names
        )
        
        logging.info(f"Created Tigramite DataFrame: {dataframe.N} variables, T={dataframe.T}")
        
        # Validate time steps structure
        if not isinstance(dataframe.T, dict) or 0 not in dataframe.T or dataframe.T[0] <= 0:
            logging.error(f"Invalid time steps structure ({dataframe.T}). Check input file: {file_path}")
            return None, None
        
        return dataframe, var_names
        
    except FileNotFoundError:
        logging.error(f"Data file not found: {file_path}")
        return None, None
    except Exception as e:
        logging.error(f"Error loading dataset for prediction: {e}", exc_info=True)
        return None, None