"""
PCMCI+ Causal Discovery for Climate Teleconnection Analysis

Core Functionality:
- Implement PCMCI+ algorithm with forward-in-time conditioning for climate data
- Identify robust causal pathways between climate predictors and ENSO indices
- Apply iterative significance testing with multiple alpha levels for robustness
- Filter networks to focus on pathways terminating at target variables

Key Features:
- Custom forward-in-time conditioning ensuring temporal causality for predictions
- Multi-level significance testing to assess edge persistence and reduce false positives
- Climate-adapted lag structure optimized for seasonal-to-interannual timescales
- Modular configuration system supporting different predictor sets and target variables

Output: Filtered causal graphs and statistical metadata for robust climate teleconnections
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
import gc

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ PATH SETUP ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sys.path.append(str(Path(__file__).parent.parent))

# Import configuration utilities
try:
    from utils.paths import get_data_path, get_results_path
    from utils.plotting_optimization import (
        save_figure_optimized, add_plotting_arguments,
        setup_cartopy_warnings, apply_rasterization_settings
    )
    from utils.predictor_configs import (
        PredictorConfig,
        load_custom_config_from_file,
        get_alpha_preset,
        get_available_alpha_presets,
        get_available_predictor_sets,
        print_predictor_set_info
    )
    from utils.causal_discovery_utils import (
        load_and_prepare_data,
        run_iterative_pcmciplus,
        run_single_pcmci_fdr,
        intersect_graphs,
        apply_fdr_pruning,
        track_edge_significance,
        analyze_edge_persistence,
        create_robust_graph,
        filter_graph,
        write_graph_with_p_values,
        save_persistence_results,
        generate_output_paths,
        save_analysis_metadata,
        get_node_positions,
        create_causal_graph_w_map
    )
except ImportError as err:
    logging.error(f"Failed to import required modules: {err}")
    raise RuntimeError("Could not import utils. Ensure PYTHONPATH includes project root.") from err

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ GLOBAL PARAMETERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Analysis parameters - these define the temporal scope of causal discovery
TAU_MIN, TAU_MAX = 1, 6  # Lag range in seasons (3-18 months)
MAX_COMBINATIONS = 10    # Max conditioning set size for computational feasibility
BASE_SUFFIX = 'PCMCI_hard'
DATA_DIR = get_data_path('time_series/', data_type='processed')

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━ COMMAND LINE INTERFACE ━━━━━━━━━━━━━━━━━━━━━━━━━
def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments for PCMCI+ analysis configuration.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        prog='12_PCMCIplus.py',
        description='PCMCI+ Causal Discovery with Configurable Predictor Sets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available predictor sets:
  known     : Well-established ENSO precursors from literature
  new       : Novel predictive modes from this research  
  combined  : All available predictors (default)
  custom    : User-defined configuration from JSON file

FDR options:
  --enable-fdr           Activate post-hoc FDR pruning (default off)
  --fdr-pc-alpha 0.10    Significance level for FDR-BH method (only used if --enable-fdr)

Examples:
  python 12_PCMCIplus.py --predictor-set known
  python 12_PCMCIplus.py --predictor-set new --alpha-preset mild
  python 12_PCMCIplus.py --predictor-set custom --config my_predictors.json
  python 12_PCMCIplus.py --predictor-set known --alpha-preset hard --enable-fdr --fdr-pc-alpha 0.10
  python 12_PCMCIplus.py --list-sets  # Show available sets
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
    
    # Alpha value configuration
    parser.add_argument(
        '--alpha-preset',
        choices=get_available_alpha_presets() + ['custom'],
        default='hard',
        help='Significance level preset: mild/hard/custom (default: hard)'
    )
    parser.add_argument(
        '--alpha-values',
        nargs='+',
        type=float,
        help='Custom alpha values (required when --alpha-preset=custom)'
    )
    
    # Test configuration
    parser.add_argument(
        '--cond-ind-test',
        choices=['robust_parcorr', 'gpdc'],
        default='robust_parcorr',
        help='Conditional independence test (default: robust_parcorr)'
    )
    
    # Output configuration
    parser.add_argument(
        '--base-suffix',
        default=BASE_SUFFIX,
        help=f'Base suffix for output files (default: {BASE_SUFFIX})'
    )
    parser.add_argument(
        '--data-file',
        default='PCMCI_data_ts_st.nc',
        help='Input data file name in DATA_DIR (default: PCMCI_data_ts_st.nc)'
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
    
    # Utility options
    parser.add_argument(
        '--list-sets',
        action='store_true',
        help='List available predictor sets with descriptions and exit'
    )
    
    # Map visualization options
    parser.add_argument(
        '--map-extent',
        choices=['fixed', 'dynamic', 'custom'],
        default='fixed',
        help='Map extent mode: fixed (default, 70E-360E, 50S-50N), dynamic (adapt to nodes), or custom'
    )
    parser.add_argument(
        '--extent-bounds',
        nargs=4,
        type=float,
        metavar=('WEST', 'EAST', 'SOUTH', 'NORTH'),
        help='Custom extent bounds [west, east, south, north] in degrees (requires --map-extent=custom)'
    )
    parser.add_argument(
        '--show-legend',
        action='store_true',
        help='Show colorbar/legend with unified label in causal graph visualization (default: hidden)'
    )
    
    # Add standard plotting arguments for optimization
    add_plotting_arguments(parser)
    
    return parser.parse_args()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━ VISUALIZATION HELPERS ━━━━━━━━━━━━━━━━━━━━━━━━━




# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ OUTPUT HELPERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━






# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ MAIN PIPELINE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def main() -> None:
    """
    Main analysis pipeline orchestrating the complete PCMCI+ workflow.
    
    This function coordinates:
    1. Configuration loading and validation
    2. Data preparation with seasonal masking
    3. Iterative PCMCI+ analysis across alpha values
    4. Edge persistence and robustness analysis
    5. Graph filtering for target pathways
    6. Visualization and output generation
    """
    # ─────────────────── Parse Arguments ──────────────────
    args = parse_arguments()
    
    # Setup warning suppression for clean output
    setup_cartopy_warnings()
    
    # Handle utility options
    if args.list_sets:
        print_predictor_set_info()
        return
    
    # ─────────────────── Validate Configuration ──────────────────
    # Check custom configuration requirements
    if args.predictor_set == 'custom' and not args.config:
        logging.error("Custom configuration file required when --predictor-set=custom")
        sys.exit(1)
    
    # Check alpha value requirements
    if args.alpha_preset == 'custom' and not args.alpha_values:
        logging.error("Alpha values required when --alpha-preset=custom")
        sys.exit(1)
    
    # Check map extent requirements
    if args.map_extent == 'custom' and not args.extent_bounds:
        logging.error("Custom extent requires --extent-bounds")
        sys.exit(1)
    
    # ─────────────────── Load Configuration ──────────────────
    logging.info("="*70)
    logging.info("PCMCI+ CAUSAL DISCOVERY ANALYSIS")
    logging.info("="*70)
    
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
    
    # ─────────────────── Setup Alpha Values ──────────────────
    if args.alpha_preset == 'custom':
        alpha_values = args.alpha_values
    else:
        alpha_values = get_alpha_preset(args.alpha_preset)
    
    logging.info(f"Alpha values for iterative analysis: {alpha_values}")
    logging.info(f"Conditional independence test: {args.cond_ind_test}")
    
    # ─────────────────── Generate Output Paths ──────────────────
    paths = generate_output_paths(args.base_suffix, args.alpha_preset, args.cond_ind_test, config)
    save_dir = paths['save_dir']
    suffix = paths['suffix']
    
    # ─────────────────── Load and Prepare Data ──────────────────
    data_file_path = os.path.join(DATA_DIR, args.data_file)
    logging.info(f"Loading data from {data_file_path}")
    
    try:
        # Use utility function to load data with proper seasonal masking
        data_array, mask_array, var_names = load_and_prepare_data(
            file_path=data_file_path,
            variables_to_load=all_variables,
            variable_seasons_map=variable_seasons,
            season_months_map=season_months,
            jja_predictors=variable_groups.get('JJA_PREDICTORS', []),
            son_predictors=variable_groups.get('SON_PREDICTORS', []),
            djf_confounders=variable_groups.get('DJF_CONFOUNDERS', []),
            djf_effects=variable_groups.get('DJF_EFFECTS', []),
            jja1_mediators=variable_groups.get('JJA(1)_MEDIATORS', [])
        )
        
        if data_array is None:
            logging.error("Failed to load data")
            sys.exit(1)
            
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Create Tigramite dataframe
    from tigramite import data_processing as pp
    dataframe = pp.DataFrame(
        data=data_array,
        datatime=np.arange(len(data_array)),
        var_names=var_names,
        mask=mask_array
    )
    logging.info(f"Created Tigramite dataframe: {data_array.shape[0]} timesteps × {data_array.shape[1]} variables")
    
    # ─────────────────── Run Iterative PCMCI+ ──────────────────
    logging.info("\n" + "-"*50)
    logging.info("RUNNING ITERATIVE PCMCI+ ANALYSIS")
    logging.info("-"*50)
    
    results_list = run_iterative_pcmciplus(
        dataframe=dataframe,
        tau_min=TAU_MIN,
        tau_max=TAU_MAX,
        alpha_values=alpha_values,
        max_combinations=MAX_COMBINATIONS,
        cond_ind_test_name=args.cond_ind_test,
        verbosity=1  # Enable tigramite output
    )
    
    successful_runs = sum(1 for r in results_list if r is not None)
    logging.info(f"Completed {successful_runs}/{len(alpha_values)} PCMCI+ runs successfully")
    
    # ─────────────────── Edge Persistence Analysis ──────────────────
    logging.info("\n" + "-"*50)
    logging.info("ANALYZING EDGE PERSISTENCE")
    logging.info("-"*50)
    
    # Track edges across all runs
    edge_tracking_df = track_edge_significance(
        results_list=results_list,
        var_names=var_names,
        alpha_values=alpha_values,
        tau_min=TAU_MIN,
        tau_max=TAU_MAX
    )
    
    if edge_tracking_df.empty:
        logging.warning("No significant edges found in any run")
        # Still save empty results for completeness
        save_persistence_results(pd.DataFrame(), save_dir, suffix)
        return
    
    logging.info(f"Tracked {len(edge_tracking_df)} edge occurrences across runs")
    
    # Analyze persistence with robustness criteria
    persistence_df = analyze_edge_persistence(
        edge_tracking_df=edge_tracking_df,
        num_total_runs=len(alpha_values),
        min_persistence=0.75,      # Present in 75% of runs
        max_avg_pvalue=0.05,       # Average p-value < 0.05
        min_effect_consistency=0.9, # Same sign in 90% of runs
        min_effect_size=0.15       # |effect| > 0.15
    )
    
    num_robust = len(persistence_df[persistence_df['is_robust']])
    logging.info(f"Found {num_robust} robust edges out of {len(persistence_df)} unique edges")
    
    # ─────────────────── Create Robust Graph ──────────────────
    logging.info("\n" + "-"*50)
    logging.info("CREATING ROBUST CAUSAL GRAPH")
    logging.info("-"*50)
    
    # Use first successful result for matrix dimensions
    first_result = next((r for r in results_list if r is not None), None)
    if first_result is None:
        logging.error("No successful PCMCI+ results to create graph")
        return
    
    robust_graph, robust_val_matrix, robust_p_matrix = create_robust_graph(
        persistence_df=persistence_df,
        num_vars=len(var_names),
        var_names=var_names,
        tau_max=TAU_MAX
    )
    
    # ─────────────────── Filter Graph for Target Pathways ──────────────────
    logging.info("\n" + "-"*50)
    logging.info("FILTERING GRAPH FOR TARGET PATHWAYS")
    logging.info("-"*50)
    
    # Define target variables (ENSO indices)
    target_variables = ['E-ind DJF(1)', 'C-ind DJF(1)']
    
    # Apply cumulative lag filtering
    filtered_graph, filtered_val_matrix, filtered_p_matrix = filter_graph(
        graph=robust_graph,
        val_matrix=robust_val_matrix,
        p_matrix=robust_p_matrix,
        var_names=var_names,
        target_variables=target_variables,
        tau_min=TAU_MIN,
        tau_max=TAU_MAX,
        max_cumulative_lag=TAU_MAX,
        min_cumulative_lag=3  # At least 3 seasons lead time
    )
    
    # Count filtered edges
    filtered_edges = np.sum(filtered_graph != '')
    logging.info(f"Filtered graph contains {filtered_edges} edges leading to target variables")
    
    # ─────────────────── Post-hoc FDR pruning ──────────────────
    if args.enable_fdr:
        logging.info("\n" + "-"*50)
        logging.info("APPLYING FDR PRUNING")
        logging.info("-"*50)
        
        try:
            final_graph, final_val_matrix, final_p_matrix, fdr_info = apply_fdr_pruning(
                dataframe=dataframe,
                robust_graph=filtered_graph,
                robust_val_matrix=filtered_val_matrix,
                robust_p_matrix=filtered_p_matrix,
                var_names=var_names,
                target_variables=target_variables,
                tau_min=TAU_MIN,
                tau_max=TAU_MAX,
                fdr_pc_alpha=args.fdr_pc_alpha,
                cond_ind_test_name=args.cond_ind_test,
                max_cumulative_lag=TAU_MAX,
                min_cumulative_lag=3,
                max_combinations=MAX_COMBINATIONS,
                verbosity=1
            )
            final_edges = fdr_info['pruned_edges_final']
        except Exception as e:
            logging.error(f"FDR pruning failed: {e}")
            # Fall back to non-FDR results
            final_graph = filtered_graph
            final_val_matrix = filtered_val_matrix
            final_p_matrix = filtered_p_matrix
            final_edges = filtered_edges
    else:
        final_graph = filtered_graph
        final_val_matrix = filtered_val_matrix
        final_p_matrix = filtered_p_matrix
        final_edges = filtered_edges
    
    # ─────────────────── Generate Visualizations ──────────────────
    logging.info("\n" + "-"*50)
    logging.info("GENERATING VISUALIZATIONS")
    logging.info("-"*50)
    
    # Process map extent arguments
    use_dynamic = args.map_extent == 'dynamic'
    custom_extent = None
    if args.map_extent == 'custom':
        custom_extent = args.extent_bounds
    
    # Create suffix tag based on FDR status
    suffix_tag = f"{suffix}_robust_filtered"
    if args.enable_fdr:
        suffix_tag += f"_fdr_pc_alpha{str(args.fdr_pc_alpha).replace('.','p')}"
    
    # Create map-based visualization of final graph (with or without FDR)
    create_causal_graph_w_map(
        graph=final_graph,
        val_matrix=final_val_matrix,
        var_names=var_names,
        save_dir=save_dir,
        suffix=suffix_tag,
        predictor_set=args.predictor_set,
        cond_ind_test=args.cond_ind_test,
        output_format=args.output_format,
        raster_dpi=args.raster_dpi,
        vector_dpi=args.vector_dpi,
        use_dynamic_extent=use_dynamic,
        custom_extent=custom_extent,
        show_colorbar=args.show_legend
    )
    
    # Also create unfiltered robust graph for comparison
    create_causal_graph_w_map(
        graph=robust_graph,
        val_matrix=robust_val_matrix,
        var_names=var_names,
        save_dir=save_dir,
        suffix=f"{suffix}_robust_full",
        predictor_set=args.predictor_set,
        cond_ind_test=args.cond_ind_test,
        output_format=args.output_format,
        raster_dpi=args.raster_dpi,
        vector_dpi=args.vector_dpi,
        use_dynamic_extent=use_dynamic,
        custom_extent=custom_extent,
        show_colorbar=args.show_legend
    )
    
    # Clean up memory after visualizations
    gc.collect()
    
    # ─────────────────── Save Results ──────────────────
    logging.info("\n" + "-"*50)
    logging.info("SAVING RESULTS")
    logging.info("-"*50)
    
    # Save persistence analysis results
    save_persistence_results(persistence_df, save_dir, suffix)
    
    # Save final graph details (with FDR suffix if applicable)
    csv_path = os.path.join(save_dir, f'filtered_robust_graph_{suffix_tag}.csv')
    write_graph_with_p_values(
        graph=final_graph,
        val_matrix=final_val_matrix,
        p_matrix=final_p_matrix,
        var_names=var_names,
        save_name=csv_path,
        tau_min=TAU_MIN,
        tau_max=TAU_MAX,
        digits=4
    )
    
    # Save full robust graph details
    full_csv_path = os.path.join(save_dir, f'full_robust_graph_{suffix}.csv')
    write_graph_with_p_values(
        graph=robust_graph,
        val_matrix=robust_val_matrix,
        p_matrix=robust_p_matrix,
        var_names=var_names,
        save_name=full_csv_path,
        tau_min=TAU_MIN,
        tau_max=TAU_MAX,
        digits=4
    )
    
    # Save comprehensive metadata
    results_summary = {
        'variables_analyzed': len(all_variables),
        'successful_pcmci_runs': successful_runs,
        'total_edges_found': len(edge_tracking_df),
        'unique_edges_tested': len(persistence_df),
        'robust_edges': num_robust,
        'filtered_edges': int(filtered_edges),
        'fdr_pc_alpha': args.fdr_pc_alpha if args.enable_fdr else None,
        'filtered_edges_after_fdr': int(final_edges) if args.enable_fdr else None,
        'alpha_values_used': alpha_values
    }
    
    save_analysis_metadata(
        config=config,
        alpha_values=alpha_values,
        alpha_preset=args.alpha_preset,
        cond_ind_test=args.cond_ind_test,
        results_summary=results_summary,
        output_dir=save_dir,
        pathway=None,
        pathway_config=None
    )
    
    # ─────────────────── Final Summary ──────────────────
    logging.info("\n" + "="*70)
    logging.info("ANALYSIS COMPLETE")
    logging.info("="*70)
    logging.info(f"Results saved to: {save_dir}")
    logging.info(f"Robust edges found: {num_robust}")
    logging.info(f"Filtered edges to targets: {filtered_edges}")
    if args.enable_fdr:
        logging.info(f"Edges after FDR pruning: {final_edges}")
    logging.info("="*70)


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