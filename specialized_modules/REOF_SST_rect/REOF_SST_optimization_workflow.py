"""
Optimization Workflow for PCMCIplus Analysis using REOF

This script implements an optimization workflow to find the optimal REOF domains
for SST that maximize the causal effect of specific REOF modes
(PC1_SST to PC4_SST) on specific target ENSO diversity variables at specific lags.

The workflow iteratively:
1. Modifies the vertices in script 2_REOF_SST.py (using rectangular domains)
2. Runs the REOF analysis via script 2
3. Preprocesses the data via script 14_..._REOF_SST.py
4. Runs the PCMCIplus analysis via script 16_..._REOF_SST.py (with dual CI tests)
5. Extracts the causal effect values and updates the Rect domains for the next iteration using a genetic algorithm.
"""

import os
import sys
from pathlib import Path
import re
import numpy as np
import pandas as pd
import subprocess
import random
import shutil
import logging
import concurrent.futures
import itertools
from typing import List, Tuple, Dict, Any, Sequence

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import and set seeds for reproducibility BEFORE any random operations
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.seed_config import set_all_seeds, GLOBAL_SEED
set_all_seeds(GLOBAL_SEED)

# Add the project root directory to the Python path
try:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from rect_utils import Rect
    from utils.paths import get_results_path
    project_root = str(Path(__file__).parent.parent.parent)
except NameError:
    logging.warning("__file__ not defined. Assuming rect_utils and src.utils are in PYTHONPATH.")
    pass

from rect_utils import (Rect, rect_to_vertices, vertices_to_rect,
                        random_rect, mutate_rect, crossover_rect,
                        is_valid_rect,
                        LAT_RANGE as RECT_UTILS_LAT_RANGE, 
                        LON_RANGE as RECT_UTILS_LON_RANGE, 
                        MIN_DEG_LAT as RECT_UTILS_MIN_DEG_LAT,
                        MIN_DEG_LON as RECT_UTILS_MIN_DEG_LON)
from utils.paths import get_results_path

# Define paths and constants
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SUFFIX = 'all_lags_iterative_step1_noFDR_REOF_SST'
VERSION_BASE = 'v'
START_VERSION_NUM = 0
TRIAL = 602
RESULTS_DIR = get_results_path(f'REOF_SST/trial_{TRIAL}/')
os.makedirs(RESULTS_DIR, exist_ok=True)

INITIAL_RECT = Rect(lat_min=-35, lat_max=-20, lon_min=-50, lon_max=5)
if not is_valid_rect(INITIAL_RECT):
    logging.error(f"FATAL: INITIAL_RECT {INITIAL_RECT} is not valid according to rect_utils.is_valid_rect.")
    logging.error(f"Rect Utils LAT_RANGE: {RECT_UTILS_LAT_RANGE}, LON_RANGE: {RECT_UTILS_LON_RANGE}, MIN_DEG_LAT: {RECT_UTILS_MIN_DEG_LAT}, MIN_DEG_LON: {RECT_UTILS_MIN_DEG_LON}")
    sys.exit(1)

TARGETS_1 = {
    'PC1_SST': { 'var': 'PC1_SST', 'effect': 'E-ind DJF(1)', 'lag': 6 },
    'PC2_SST': { 'var': 'PC2_SST', 'effect': 'E-ind DJF(1)', 'lag': 6 },
    'PC3_SST': { 'var': 'PC3_SST', 'effect': 'E-ind DJF(1)', 'lag': 6 },
    'PC4_SST': { 'var': 'PC4_SST', 'effect': 'E-ind DJF(1)', 'lag': 6 },
}
TARGETS_2 = {
    'PC1_SST': { 'var': 'PC1_SST', 'effect': 'C-ind DJF(1)', 'lag': 6 },
    'PC2_SST': { 'var': 'PC2_SST', 'effect': 'C-ind DJF(1)', 'lag': 6 },
    'PC3_SST': { 'var': 'PC3_SST', 'effect': 'C-ind DJF(1)', 'lag': 6 },
    'PC4_SST': { 'var': 'PC4_SST', 'effect': 'C-ind DJF(1)', 'lag': 6 },
}
OPTIMIZED_REOF_VARS = ['PC1_SST', 'PC2_SST', 'PC3_SST', 'PC4_SST']
ALL_TARGETS = {
    "TARGETS_1": TARGETS_1,
    "TARGETS_2": TARGETS_2,
}

POPULATION_SIZE = 36
GENERATIONS = 6    
MUTATION_RATE = 0.35 
CROSSOVER_RATE = 0.75 

def roulette_wheel_selection(population: List[Sequence[Tuple[int, int]]], fitness: List[float]) -> int:
    total_fitness = sum(fitness)
    if total_fitness == 0:
        return random.randint(0, len(population) - 1)
    pick = random.uniform(0, total_fitness)
    current = 0
    for idx, fit in enumerate(fitness):
        current += fit
        if current > pick:
            return idx
    return len(population) - 1

def mutate_vertices_wrapper(vertices: Sequence[Tuple[int, int]],
                            mutation_strength: float = 0.2) -> Sequence[Tuple[int, int]]:
    try:
        rect = vertices_to_rect(vertices)
    except ValueError as e:
        logging.warning(f"Could not convert vertices to Rect in mutate_vertices_wrapper: {e}. Returning original vertices.")
        return vertices
    rect_mutated = mutate_rect(rect, mutation_strength)
    return rect_to_vertices(rect_mutated)

def crossover_vertices_wrapper(parent1_vertices: Sequence[Tuple[int, int]],
                               parent2_vertices: Sequence[Tuple[int, int]]) -> Sequence[Tuple[int, int]]:
    try:
        rect1 = vertices_to_rect(parent1_vertices)
        rect2 = vertices_to_rect(parent2_vertices)
    except ValueError as e:
        logging.warning(f"Could not convert parent vertices to Rect in crossover_vertices_wrapper: {e}. Returning parent1 vertices.")
        return parent1_vertices 
    child_rect = crossover_rect(rect1, rect2)
    return rect_to_vertices(child_rect)

def update_script_version(script_path: str, new_version: str) -> str:
    with open(script_path, 'r') as f:
        content = f.read()

    version_pattern = r"VERSION\s*=\s*['\"]v\d+['\"]"
    if not re.search(version_pattern, content):
        logging.warning(f"VERSION pattern not found in {script_path}. Cannot update version.")
        new_content = content
    else:
        new_content = re.sub(version_pattern, f"VERSION = '{new_version}'", content)

    base_name, ext = os.path.splitext(os.path.basename(script_path))
    save_dir = os.path.join(CURRENT_DIR, f"trial_{TRIAL}_script_versions")
    os.makedirs(save_dir, exist_ok=True)
    new_script_path = os.path.join(save_dir, f"{base_name}_{new_version}{ext}")

    with open(new_script_path, 'w') as f:
        f.write(new_content)
    return new_script_path

def run_script(script_path: str) -> int:
    if not os.path.exists(script_path):
        logging.error(f"Script not found at {script_path}")
        return -1

    print(f"Running {os.path.basename(script_path)}...")
    try:
        script_dir = os.path.dirname(script_path)
        env = os.environ.copy()
        current_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = f"{script_dir}{os.pathsep}{project_root}{os.pathsep}{current_pythonpath}"

        result = subprocess.run(
            [sys.executable, script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False, 
            env=env 
        )

        if result.returncode != 0:
            print(f"--- Output from {os.path.basename(script_path)} ---")
            print(result.stdout)
            print(f"--- Errors from {os.path.basename(script_path)} ---")
            print(result.stderr)
            print(f"--- End Output ---")
            logging.error(f"Script {os.path.basename(script_path)} failed with exit code {result.returncode}")
        return result.returncode
    except Exception as e:
        logging.error(f"Error running script {script_path}: {e}")
        return -1

def get_target_descriptor(targets: Dict[str, Dict[str, Any]], target_key: str) -> str:
    first_var_key = next(iter(targets.keys()))
    effect_name = targets[first_var_key]["effect"].replace(" ", "_").replace("(", "").replace(")", "") 
    lag = targets[first_var_key]["lag"]
    return f"{effect_name}_lag_{lag}"

def prepare_target_directories(base_results_dir: str, target_descriptor: str, version: str) -> str:
    target_version_dir = os.path.join(base_results_dir, target_descriptor, version)
    os.makedirs(target_version_dir, exist_ok=True)
    os.makedirs(os.path.join(target_version_dir, 'NetCDFs_REOF_modes'), exist_ok=True)
    os.makedirs(os.path.join(target_version_dir, 'Plots'), exist_ok=True)
    os.makedirs(os.path.join(target_version_dir, 'PCMCIplus_results'), exist_ok=True)
    return target_version_dir

def extract_causal_effects(csv_path: str, target_config: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    results = {var_name: 0.0 for var_name in target_config.keys()}
    if not os.path.exists(csv_path):
        logging.warning(f"Results file {csv_path} does not exist. Returning zero effects.")
        return results
    try:
        df = pd.read_csv(csv_path)
        required_cols = ['Cause', 'Effect', 'Lag', 'Causal_Effect']
        if not all(col in df.columns for col in required_cols):
             logging.error(f"Missing required columns in {csv_path}. Expected: {required_cols}")
             return results
        for var_name, config in target_config.items():
            cause_var, effect_var, lag_val = config['var'], config['effect'], config['lag']
            matching_rows = df[(df['Cause'] == cause_var) & (df['Effect'] == effect_var) & (df['Lag'] == lag_val)]
            if not matching_rows.empty:
                effect_value = abs(matching_rows.iloc[0]['Causal_Effect'])
                if pd.isna(effect_value):
                    logging.warning(f"Found NaN effect for {cause_var} -> {effect_var} (lag {lag_val}). Treating as 0.0.")
                    results[var_name] = 0.0
                else:
                    results[var_name] = float(effect_value)
    except pd.errors.EmptyDataError:
        logging.warning(f"Results file {csv_path} is empty. Returning zero effects.")
    except Exception as e:
        logging.error(f"Error processing results file {csv_path}: {e}")
        results = {var_name: 0.0 for var_name in target_config.keys()}
    return results

def run_optimization_step(individual_vertices: Sequence[Tuple[int, int]], 
                          version: str,
                          targets: Dict[str, Dict[str, Any]],
                          target_descriptor: str,
                          generation: int,
                          individual_idx: int) -> Tuple[Dict[str, float], bool]:
    first_target_key = next(iter(targets.keys()))
    target_lag = targets[first_target_key]["lag"]

    os.environ["TARGET_DESCRIPTOR"] = target_descriptor
    os.environ["GA_GENERATION"] = str(generation)
    os.environ["GA_INDIVIDUAL"] = str(individual_idx) 
    os.environ["TARGET_LAG"] = str(target_lag)
    os.environ["TRIAL"] = f'trial_{TRIAL}'
    os.environ["VERSION"] = version

    try:
        current_rect = vertices_to_rect(individual_vertices)
        os.environ["RECT_LAT_MIN"] = str(current_rect.lat_min)
        os.environ["RECT_LAT_MAX"] = str(current_rect.lat_max)
        os.environ["RECT_LON_MIN"] = str(current_rect.lon_min)
        os.environ["RECT_LON_MAX"] = str(current_rect.lon_max)
        logging.info(f"Set RECT env vars: lat_min={current_rect.lat_min}, lat_max={current_rect.lat_max}, lon_min={current_rect.lon_min}, lon_max={current_rect.lon_max}")
    except ValueError as e:
        logging.error(f"Failed to convert vertices to Rect for env var setting: {individual_vertices}. Error: {e}")
        return ({var: 0.0 for var in targets.keys()}, False)

    target_version_dir = prepare_target_directories(RESULTS_DIR, target_descriptor, version)

    script2_orig_path = os.path.join(CURRENT_DIR, '2_REOF_SST.py')
    script14_orig_path = os.path.join(CURRENT_DIR, '14_save_ds_caus_E_v_C_40_24_REOF_SST.py')
    script16_orig_path = os.path.join(CURRENT_DIR, '16_PCMCIplus_all_lags_40_24_iterative_step1_noFDR_REOF_SST.py')

    scripts_exist = True
    for p_script in [script2_orig_path, script14_orig_path, script16_orig_path]:
        if not os.path.exists(p_script):
            logging.error(f"FATAL Error: Original script not found: {p_script}")
            scripts_exist = False
    if not scripts_exist:
        return ({var: 0.0 for var in targets.keys()}, False)

    new_script2_path = update_script_version(script2_orig_path, version)
    new_script14_path = update_script_version(script14_orig_path, version)
    new_script16_path = update_script_version(script16_orig_path, version)

    current_run_success = True
    print(f"\n--- Running analysis chain for Gen {generation}, Ind {individual_idx}, Version {version} ---")

    if run_script(new_script2_path) != 0: current_run_success = False
    if current_run_success and run_script(new_script14_path) != 0: current_run_success = False
    if current_run_success and run_script(new_script16_path) != 0:
        logging.warning(f"Script 16 ({os.path.basename(new_script16_path)}) failed or had issues.")

    csv_path = os.path.join(target_version_dir, 'PCMCIplus_results', f'filtered_robust_graph_{SUFFIX}.csv')
    print(f"Attempting to extract effects from: {csv_path}")
    effects = extract_causal_effects(csv_path, targets)

    print(f"--- Finished analysis chain for Version {version}. Scripts Success: {current_run_success} ---")
    print(f"Extracted Effects: {effects}")

    return effects, current_run_success

def _set_thread_env():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

def available_workers():
    max_by_mem = (24 - 4) // 1
    max_by_cpu = os.cpu_count() // 2
    return max(1, min(8, max_by_mem, max_by_cpu))

def genetic_algorithm_optimization():
    print("Starting REOF genetic algorithm optimization...")
    version_num = START_VERSION_NUM
    all_targets_tracking = {}

    for target_key, targets_config in ALL_TARGETS.items():
        target_descriptor = get_target_descriptor(targets_config, target_key)
        first_var_key = next(iter(targets_config.keys()))
        effect_name = targets_config[first_var_key]['effect']
        lag_value = targets_config[first_var_key]['lag']

        print(f"\n{'='*10} Starting optimization for target set: {target_key} ({target_descriptor}) {'='*10}")
        print(f"Target Effect: {effect_name}, Target Lag: {lag_value}")
        print(f"Optimizing for variables: {OPTIMIZED_REOF_VARS}")

        target_tracking = {
            'target_key': target_key, 'target_descriptor': target_descriptor,
            'effect': effect_name, 'lag': lag_value,
            'best_overall_effect': 0.0, 'best_overall_var_name': None,
            'best_vertices': None, 'best_generation': -1,
            'best_individual': -1, 'best_version': None,
            'variable_best_effects': {var_name: 0.0 for var_name in OPTIMIZED_REOF_VARS}
        }

        population_vertices: List[Sequence[Tuple[int, int]]] = []
        if is_valid_rect(INITIAL_RECT):
            population_vertices.append(rect_to_vertices(INITIAL_RECT))
        else:
            logging.warning(f"INITIAL_RECT {INITIAL_RECT} is invalid. Starting with a random valid rect.")
            population_vertices.append(rect_to_vertices(random_rect()))

        while len(population_vertices) < POPULATION_SIZE:
            population_vertices.append(rect_to_vertices(random_rect()))

        all_run_results_for_target = []

        for generation in range(GENERATIONS):
            print(f"\n--- Generation {generation+1}/{GENERATIONS} for {target_descriptor} ---")
            fitness_values = []
            generation_results_log = [] 

            with concurrent.futures.ProcessPoolExecutor(
                max_workers=available_workers(),
                initializer=_set_thread_env,
            ) as pool:
                futs = {
                    pool.submit(
                        run_optimization_step,
                        vtx,
                        f"{VERSION_BASE}{vnum+i}",
                        targets_config,
                        target_descriptor,
                        generation,
                        i
                    ): i
                    for i, (vtx, vnum) in enumerate(zip(population_vertices, itertools.count(version_num)))
                }
                version_num += len(futs)
                for fut in concurrent.futures.as_completed(futs):
                    i = futs[fut]
                    effects, success_run = fut.result()
                    current_ind_vertices = population_vertices[i]
                    current_max_effect = 0.0
                    current_best_var_name = None
                    if effects and any(v > 0 for v in effects.values()):
                        current_max_effect = max(effects.values())
                        current_best_var_name = max(effects, key=effects.get)
                    fitness_values.append(current_max_effect)

                    individual_log = {
                        'version': f"{VERSION_BASE}{version_num - len(futs) + i}", 'generation': generation, 'individual': i,
                        'vertices': current_ind_vertices, 'success_run': success_run,
                        'target_descriptor': target_descriptor,
                        'max_effect_in_individual': current_max_effect,
                        'best_var_in_individual': current_best_var_name
                    }
                    for var_name_log_entry in OPTIMIZED_REOF_VARS:
                        individual_log[f'effect_{var_name_log_entry}'] = effects.get(var_name_log_entry, 0.0)
                    generation_results_log.append(individual_log)
                    all_run_results_for_target.append(individual_log)

                    if current_max_effect > target_tracking['best_overall_effect']:
                        print(f"*** New Best Overall Effect for {target_descriptor}: {current_max_effect:.4f} "
                              f"(Variable: {current_best_var_name}, Version: {f'{VERSION_BASE}{version_num - len(futs) + i}'}) ***")
                        target_tracking.update({
                            'best_overall_effect': current_max_effect,
                            'best_overall_var_name': current_best_var_name,
                            'best_vertices': list(current_ind_vertices), 
                            'best_generation': generation, 'best_individual': i,
                            'best_version': f"{VERSION_BASE}{version_num - len(futs) + i}"
                        })

                    for var_name_track in OPTIMIZED_REOF_VARS:
                        var_effect = effects.get(var_name_track, 0.0)
                        if var_effect > target_tracking['variable_best_effects'][var_name_track]:
                            target_tracking['variable_best_effects'][var_name_track] = var_effect

            best_fitness_in_gen = max(fitness_values) if fitness_values else 0.0
            avg_fitness_in_gen = np.mean(fitness_values) if fitness_values else 0.0
            print(f"\nGeneration {generation+1} Summary ({target_descriptor}):")
            print(f"  Best Fitness (Max Effect) in Generation: {best_fitness_in_gen:.4f}")
            print(f"  Average Fitness in Generation: {avg_fitness_in_gen:.4f}")
            print(f"  Best Overall Effect Found So Far: {target_tracking['best_overall_effect']:.4f} "
                  f"(Variable: {target_tracking['best_overall_var_name'] or 'None'})")
            print("  Best Effects Per Variable So Far:")
            for var_name_sum_gen, best_eff_gen in target_tracking['variable_best_effects'].items():
                print(f"    {var_name_sum_gen}: {best_eff_gen:.4f}")

            print("\nPerforming Selection, Crossover, Mutation...")
            diversity = np.std(fitness_values) if fitness_values else 0.0
            adaptive_mutation_rate = MUTATION_RATE * 2.0 if diversity < 0.01 and generation > 0 else MUTATION_RATE
            if adaptive_mutation_rate != MUTATION_RATE:
                logging.info(f"Low diversity detected (std = {diversity:.4f}). Increasing mutation rate to {adaptive_mutation_rate:.2f}.")

            new_population_vertices: List[Sequence[Tuple[int, int]]] = []
            if fitness_values: 
                best_idx_current_gen = np.argmax(fitness_values)
                new_population_vertices.append(population_vertices[best_idx_current_gen])
                logging.info(f"Elitism: Keeping individual {best_idx_current_gen} (fitness {fitness_values[best_idx_current_gen]:.4f})")

            while len(new_population_vertices) < POPULATION_SIZE:
                parent1_idx = roulette_wheel_selection(population_vertices, fitness_values)
                parent2_idx = roulette_wheel_selection(population_vertices, fitness_values)
                parent1_v = population_vertices[parent1_idx]
                parent2_v = population_vertices[parent2_idx]

                child_v: Sequence[Tuple[int, int]]
                if random.random() < CROSSOVER_RATE:
                    child_v = crossover_vertices_wrapper(parent1_v, parent2_v)
                else:
                    child_v = list(parent1_v) if random.random() < 0.5 else list(parent2_v) 

                if random.random() < adaptive_mutation_rate:
                    child_v = mutate_vertices_wrapper(child_v, mutation_strength=adaptive_mutation_rate)
                
                new_population_vertices.append(child_v)
            
            while len(new_population_vertices) < POPULATION_SIZE:
                 logging.warning(f"Population size ({len(new_population_vertices)}) incorrect after breeding. Adding random valid individuals.")
                 new_population_vertices.append(rect_to_vertices(random_rect()))

            num_immigrants = max(1, int(0.1 * POPULATION_SIZE))
            immigrant_indices = random.sample(range(POPULATION_SIZE), num_immigrants)
            logging.info(f"Injecting {num_immigrants} new random immigrants.")
            for idx in immigrant_indices:
                new_population_vertices[idx] = rect_to_vertices(random_rect())
            population_vertices = new_population_vertices

            if generation == GENERATIONS - 1:
                print("\nLast generation reached. Starting neighbourhood sweep around best solution...")

                best_rect = vertices_to_rect(target_tracking['best_vertices'])
                deltas = [-3, -2, -1, 1, 2, 3]
                neighbour_rects = set()
                for dlat in deltas:
                    for dlon in deltas:
                        for edge in ["lat_min", "lat_max", "lon_min", "lon_max"]:
                            new_vals = dict(best_rect._asdict())
                            new_vals[edge] += (dlat if 'lat' in edge else dlon)
                            candidate = Rect(**new_vals)
                            if is_valid_rect(candidate):
                                neighbour_rects.add(tuple(rect_to_vertices(candidate)))

                population_vertices = [list(v) for v in neighbour_rects]
                if len(population_vertices) > 100:
                    population_vertices = random.sample(population_vertices, 100)
                print(f"Neighbourhood sweep: evaluating {len(population_vertices)} neighbours.")

                incumbent_effect = target_tracking['best_overall_effect']
                for i, verts in enumerate(population_vertices):
                    version = f"{VERSION_BASE}{version_num}"
                    print(f"\nEvaluating neighbour {i} (Version {version}). Vertices: {verts}")
                    effects, success_run = run_optimization_step(
                        verts, version, targets_config,
                        target_descriptor, generation, i
                    )
                    current_max = max(effects.values()) if effects else 0.0
                    best_var = max(effects, key=effects.get) if current_max > 0 else None
                    entry = {
                        'version': version, 'generation': generation, 'individual': i,
                        'vertices': verts, 'success_run': success_run,
                        'target_descriptor': target_descriptor,
                        'max_effect_in_individual': current_max,
                        'best_var_in_individual': best_var
                    }
                    for vn in OPTIMIZED_REOF_VARS:
                        entry[f'effect_{vn}'] = effects.get(vn, 0.0)
                    all_run_results_for_target.append(entry)

                    if current_max > incumbent_effect:
                        print(f"*** New best neighbour effect: {current_max:.4f} "
                              f"(Variable: {best_var}, Version: {version}) ***")
                        target_tracking.update({
                            'best_overall_effect': current_max,
                            'best_overall_var_name': best_var,
                            'best_vertices': verts,
                            'best_generation': generation,
                            'best_individual': i,
                            'best_version': version
                        })
                        incumbent_effect = current_max
                    version_num += 1

                print("Neighbourhood sweep complete.")
                break

        if target_tracking['best_vertices']:
            final_version = f"{VERSION_BASE}{version_num}"
            print(f"\n{'='*10} Final Evaluation for {target_descriptor} using best vertices (Version {final_version}) {'='*10}")
            final_effects, final_success = run_optimization_step(
                target_tracking['best_vertices'], final_version, targets_config,
                target_descriptor, GENERATIONS, -1 
            )
            version_num += 1
            final_max_effect = 0.0
            final_best_var_name = None
            if final_effects and any(v > 0 for v in final_effects.values()):
                final_max_effect = max(final_effects.values())
                final_best_var_name = max(final_effects, key=final_effects.get)

            final_result_log = {
                'version': final_version, 'generation': GENERATIONS, 'individual': -1,
                'vertices': target_tracking['best_vertices'], 'success_run': final_success,
                'target_descriptor': target_descriptor,
                'max_effect_in_individual': final_max_effect,
                'best_var_in_individual': final_best_var_name
            }
            for var_name_l in OPTIMIZED_REOF_VARS:
                final_result_log[f'effect_{var_name_l}'] = final_effects.get(var_name_l, 0.0)
            all_run_results_for_target.append(final_result_log)

        target_results_savedir = os.path.join(RESULTS_DIR, target_descriptor, "optimization_summary")
        os.makedirs(target_results_savedir, exist_ok=True)
        results_df = pd.DataFrame(all_run_results_for_target)
        csv_save_path = os.path.join(target_results_savedir, f'optimization_full_log_{target_descriptor}.csv')
        results_df.to_csv(csv_save_path, index=False)
        print(f"\nFull optimization log for {target_descriptor} saved to: {csv_save_path}")
        all_targets_tracking[target_key] = target_tracking
    
    print(f"\n{'='*20} Overall Optimization Summary {'='*20}")
    global_best_tracking = {
        'target_key': None, 'var_name': None, 'effect': 0.0,
        'effect_name': None, 'lag': None, 'version': None, 'vertices': None
    }
    if not all_targets_tracking:
        print("\nNo target sets were processed.")
    else:
        for target_key_s, target_data_s in all_targets_tracking.items():
            print(f"\n--- Summary for Target Set: {target_key_s} ({target_data_s['effect']}, lag {target_data_s['lag']}) ---")
            print(f"  Best Overall Result:")
            print(f"    Variable: {target_data_s['best_overall_var_name'] or 'None'}")
            print(f"    Effect: {target_data_s['best_overall_effect']:.4f}")
            print(f"    Achieved in Version: {target_data_s['best_version'] or 'N/A'}")
            print(f"    Vertices: {target_data_s['best_vertices']}")
            print("  Best effects achieved per specific variable:")
            for var_name_s_detail in OPTIMIZED_REOF_VARS:
                var_best_s = target_data_s['variable_best_effects'][var_name_s_detail]
                print(f"    {var_name_s_detail}: {var_best_s:.4f}")

            if target_data_s['best_overall_effect'] > global_best_tracking['effect']:
                global_best_tracking.update({
                    'target_key': target_key_s,
                    'var_name': target_data_s['best_overall_var_name'],
                    'effect': target_data_s['best_overall_effect'],
                    'effect_name': target_data_s['effect'],
                    'lag': target_data_s['lag'],
                    'version': target_data_s['best_version'],
                    'vertices': target_data_s['best_vertices']
                })
        print(f"\n--- Global Best Result Across All Targets ---")
        if global_best_tracking['target_key'] is not None:
            print(f"  Target Set: {global_best_tracking['target_key']}")
            print(f"  Target Config: {global_best_tracking['effect_name']}, lag {global_best_tracking['lag']}")
            print(f"  Best Variable: {global_best_tracking['var_name']}")
            print(f"  Max Effect: {global_best_tracking['effect']:.4f}")
            print(f"  Achieved in Version: {global_best_tracking['version']}")
            print(f"  Optimal Vertices: {global_best_tracking['vertices']}")
        else:
            print("  No global best result found.")

    print(f"\n{'='*20} All Optimizations Complete {'='*20}")
    return all_targets_tracking

if __name__ == "__main__":
    print(f"Starting REOF optimization workflow for Trial {TRIAL}...")
    print(f"Results will be saved in: {RESULTS_DIR}")
    print(f"Using RECT_UTILS_LAT_RANGE: {RECT_UTILS_LAT_RANGE}, RECT_UTILS_LON_RANGE: {RECT_UTILS_LON_RANGE}, MIN_DEG_LAT: {RECT_UTILS_MIN_DEG_LAT}, MIN_DEG_LON: {RECT_UTILS_MIN_DEG_LON}")
    print(f"Initial Rect for GA: {INITIAL_RECT}")
    print(f"Optimizing variables: {OPTIMIZED_REOF_VARS}")
    print(f"Target sets to process: {list(ALL_TARGETS.keys())}")

    optimization_summary = genetic_algorithm_optimization()

    print("\nOptimization process finished.")

    script_copy_dir = os.path.join(CURRENT_DIR, f"trial_{TRIAL}_script_versions")
    if os.path.exists(script_copy_dir):
        try:
            print(f"\nCleaning up generated script copies from: {script_copy_dir}...")
            shutil.rmtree(script_copy_dir)
            print(f"Successfully removed temporary script directory: {script_copy_dir}")
        except Exception as e:
            logging.error(f"Error removing directory {script_copy_dir}: {e}")
    else:
        logging.info(f"Temporary script directory not found, skipping cleanup: {script_copy_dir}")

    print("\nWorkflow finished.")
    sys.exit(0)
