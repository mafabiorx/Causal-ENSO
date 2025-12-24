"""
Compute Multi-level Rossby Wave Source (RWS) for pressure levels [500, 400, 300, 250, 200] hPa
after removing the zonal (longitude) mean from the fields.

This script processes large atmospheric datasets efficiently using chunking and lazy loading.

Input files:
    - U_mid_up_glob_seas.nc: U wind component at 5 pressure levels
    - V_mid_up_glob_seas.nc: V wind component at 5 pressure levels

Output:
    - RWS_multi_level_seas_zonal_removed.nc: Multi-level RWS field

Formula: RWS = -η * D - (u_chi * ∂η/∂x + v_chi * ∂η/∂y)
    where:
    η = absolute vorticity
    D = horizontal divergence
    u_chi, v_chi = divergent wind components
    ∂η/∂x, ∂η/∂y = gradients of absolute vorticity

"""

import os
import sys
from pathlib import Path
import time
import glob
import gc
import psutil
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import xarray as xr
from tqdm import tqdm
from windspharm.xarray import VectorWind
import dask

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))
from utils.TBI_functions import long_0_360_to_180_180
from utils.paths import get_data_path

# -----------------------------------------------------------------------------
# 1. Configuration and Setup
# -----------------------------------------------------------------------------

# Directory configuration
INPUT_DIR = get_data_path('seasonal/', data_type="raw")
TEMP_DIR = os.path.join(INPUT_DIR, 'temp_RWS_multilevel/')
os.makedirs(TEMP_DIR, exist_ok=True)

# Processing configuration
TARGET_LEVELS = [500, 400, 300, 250, 200]  # Pressure levels in hPa
N_CHUNKS = 20  # Number of temporal chunks
MAX_MEMORY_GB = 8  # Maximum memory usage in GB
MIN_FILE_SIZE_BYTES = 10000  # Minimum valid file size

# File paths
U_FILE = os.path.join(INPUT_DIR, 'U_mid_up_glob_seas.nc')
V_FILE = os.path.join(INPUT_DIR, 'V_mid_up_glob_seas.nc')
OUTPUT_FILE = os.path.join(INPUT_DIR, 'RWS_multi_level_seas_zonal_removed.nc')

# Setup logging
log_file = os.path.join(os.path.dirname(__file__), 'RWS_multi_level_processing.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

# -----------------------------------------------------------------------------
# 2. Utility Functions
# -----------------------------------------------------------------------------

def get_memory_usage() -> float:
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / 1e9

def log_memory(message: str):
    """Log memory usage with custom message"""
    memory_gb = get_memory_usage()
    logging.info(f"{message} - Memory usage: {memory_gb:.2f} GB")

def cleanup_memory():
    """Force garbage collection and clear caches"""
    gc.collect()
    # Clear any xarray caches
    if hasattr(xr.backends.file_manager, 'FILE_CACHE'):
        xr.backends.file_manager.FILE_CACHE.clear()
    time.sleep(1)  # Give system time to release memory

def check_prerequisite_files() -> bool:
    """Check if required files exist and provide instructions if missing"""
    
    # Check if U_mid_up file exists
    if not os.path.exists(U_FILE):
        error_message = f"""
{'!'*80}
ERROR: U_mid_up_glob_seas.nc not found!
{'!'*80}
Please create this file first!
"""
        print(error_message)
        return False
    
    return True

def validate_input_files() -> bool:
    """Validate that input files exist and have correct structure"""
    
    logging.info("Validating input files...")
    
    # Check file existence
    if not os.path.exists(U_FILE):
        logging.error(f"U wind file not found: {U_FILE}")
        return False
    
    if not os.path.exists(V_FILE):
        logging.error(f"V wind file not found: {V_FILE}")
        return False
    
    # Check file structure
    try:
        with xr.open_dataset(U_FILE) as ds_u:
            u_levels = ds_u.pressure_level.values.tolist()
            if not all(level in u_levels for level in TARGET_LEVELS):
                logging.error(f"U file missing required levels. Has: {u_levels}, Need: {TARGET_LEVELS}")
                return False
            logging.info(f"U file validated: {ds_u.dims}")
        
        with xr.open_dataset(V_FILE) as ds_v:
            v_levels = ds_v.pressure_level.values.tolist()
            if not all(level in v_levels for level in TARGET_LEVELS):
                logging.error(f"V file missing required levels. Has: {v_levels}, Need: {TARGET_LEVELS}")
                return False
            logging.info(f"V file validated: {ds_v.dims}")
    
    except Exception as e:
        logging.error(f"Error validating files: {e}")
        return False
    
    logging.info("Input files validated successfully")
    return True

def get_completed_levels() -> List[int]:
    """Check which pressure levels have been completed"""
    completed = []
    
    for level in TARGET_LEVELS:
        final_file = os.path.join(TEMP_DIR, f'RWS_{level}hPa_complete.nc')
        if os.path.exists(final_file) and os.path.getsize(final_file) > MIN_FILE_SIZE_BYTES:
            try:
                with xr.open_dataset(final_file) as ds:
                    if 'RWS' in ds.data_vars:
                        completed.append(level)
            except (OSError, ValueError) as e:
                logging.debug(f"Could not validate {final_file}: {e}")
            except (KeyboardInterrupt, SystemExit):
                raise
    
    return completed

def cleanup_temp_files(level: int, chunk_files: List[str]):
    """Clean up temporary chunk files for a level"""
    
    # Remove chunk files
    for f in chunk_files:
        try:
            if os.path.exists(f):
                os.remove(f)
        except OSError as e:
            logging.warning(f"Could not remove {f}: {e}")
    
    # Remove temporary U/V files for this level
    patterns = [
        f'U_{level}hPa_chunk_*.nc',
        f'V_{level}hPa_chunk_*.nc',
        f'RWS_{level}hPa_chunk_*.nc'
    ]
    
    for pattern in patterns:
        for f in glob.glob(os.path.join(TEMP_DIR, pattern)):
            try:
                os.remove(f)
            except OSError as e:
                logging.warning(f"Could not remove {f}: {e}")

# -----------------------------------------------------------------------------
# 3. RWS Computation Functions
# -----------------------------------------------------------------------------

def compute_rws_chunk(u_chunk: xr.DataArray, v_chunk: xr.DataArray) -> xr.DataArray:
    """
    Compute RWS using windspharm with the formula:
    S = -η * D - (u_chi * ∂η/∂x + v_chi * ∂η/∂y)
    
    Following Sardeshmukh and Hoskins (1988)
    """
    
    # Create VectorWind instance for spherical harmonic calculations
    w = VectorWind(u_chunk, v_chunk)
    
    # Compute diagnostic fields
    eta = w.absolutevorticity()           # absolute vorticity (η)
    div = w.divergence()                  # horizontal divergence (D)
    uchi, vchi = w.irrotationalcomponent()  # divergent wind components (u_chi, v_chi)
    etax, etay = w.gradient(eta)          # gradients of absolute vorticity (∂η/∂x, ∂η/∂y)
    
    # Compute Rossby Wave Source: S = -η * D - (u_chi * ∂η/∂x + v_chi * ∂η/∂y)
    RWS = -eta * div - (uchi * etax + vchi * etay)
    
    # Add metadata
    RWS.name = "RWS"
    RWS.attrs["long_name"] = "Rossby Wave Source (computed on fields with zonal mean removed)"
    RWS.attrs["units"] = "s^-2"
    RWS.attrs["formula"] = "S = -η * D - (u_chi * ∂η/∂x + v_chi * ∂η/∂y)"
    
    return RWS

def process_pressure_level(level: int, retry_count: int = 0) -> Optional[str]:
    """
    Process a single pressure level with temporal chunking
    
    Returns path to completed level file or None if failed
    """
    
    level_start_time = time.time()
    logging.info(f"\n{'='*60}")
    logging.info(f"Processing pressure level: {level} hPa (attempt {retry_count + 1})")
    logging.info(f"{'='*60}")
    
    final_output = os.path.join(TEMP_DIR, f'RWS_{level}hPa_complete.nc')
    
    try:
        # 1. Load data for specific level with lazy loading
        log_memory(f"Loading data for {level} hPa")
        
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):
            # Open datasets with chunking
            ds_u = xr.open_dataset(U_FILE, chunks={'time': 12})
            ds_v = xr.open_dataset(V_FILE, chunks={'time': 12})
            
            # Select pressure level
            u_level = ds_u['u'].sel(pressure_level=level)
            v_level = ds_v['v'].sel(pressure_level=level)
            
            # Convert longitude coordinates
            u_level = long_0_360_to_180_180(u_level)
            v_level = long_0_360_to_180_180(v_level)
            
            # Remove zonal mean
            logging.info(f"Removing zonal mean for {level} hPa")
            u_eddy = u_level - u_level.mean(dim='longitude')
            v_eddy = v_level - v_level.mean(dim='longitude')
            
            # Get time dimension info
            n_times = len(u_eddy.time)
            time_chunks = np.array_split(np.arange(n_times), N_CHUNKS)
            
            logging.info(f"Processing {n_times} time steps in {len(time_chunks)} chunks")
        
        # 2. Process temporal chunks
        chunk_files = []
        
        with tqdm(total=len(time_chunks), desc=f"Level {level} hPa chunks") as pbar:
            for i, chunk_idx in enumerate(time_chunks):
                
                # Check memory before processing chunk
                if get_memory_usage() > MAX_MEMORY_GB * 0.8:
                    logging.warning(f"High memory usage detected. Cleaning up...")
                    cleanup_memory()
                    time.sleep(2)
                
                try:
                    # Extract and compute chunk
                    log_memory(f"Processing chunk {i+1}/{len(time_chunks)}")
                    
                    u_chunk = u_eddy.isel(time=chunk_idx).compute()
                    v_chunk = v_eddy.isel(time=chunk_idx).compute()
                    
                    # Compute RWS for chunk
                    rws_chunk = compute_rws_chunk(u_chunk, v_chunk)
                    
                    # Save chunk
                    chunk_file = os.path.join(TEMP_DIR, f'RWS_{level}hPa_chunk_{i:03d}.nc')
                    rws_chunk.to_netcdf(chunk_file)
                    chunk_files.append(chunk_file)
                    
                    # Clean up chunk memory
                    del u_chunk, v_chunk, rws_chunk
                    
                except Exception as e:
                    logging.error(f"Error processing chunk {i} for level {level}: {e}")
                    raise
                
                pbar.update(1)
        
        # Close datasets
        ds_u.close()
        ds_v.close()
        del u_eddy, v_eddy
        cleanup_memory()
        
        # 3. Concatenate temporal chunks
        logging.info(f"Concatenating {len(chunk_files)} chunks for {level} hPa")
        
        if not chunk_files:
            raise RuntimeError(f"No chunk files produced for level {level}")
        
        # Open and concatenate chunks
        with xr.open_mfdataset(chunk_files, combine='by_coords') as combined:
            # Sort by time
            combined = combined.sortby('time')
            
            # Add pressure level coordinate
            combined = combined.expand_dims({'pressure_level': [level]})
            
            # Save final level file
            encoding = {
                'RWS': {
                    'zlib': True,
                    'complevel': 4,
                    'dtype': 'float32'
                }
            }
            
            combined.to_netcdf(final_output, encoding=encoding)
        
        # Verify output
        output_size = os.path.getsize(final_output) / 1e6  # MB
        logging.info(f"Level {level} hPa completed: {output_size:.1f} MB")
        
        # 4. Cleanup
        cleanup_temp_files(level, chunk_files)
        
        elapsed = time.time() - level_start_time
        logging.info(f"Level {level} hPa processing time: {elapsed/60:.1f} minutes")
        
        return final_output
        
    except Exception as e:
        logging.error(f"Failed to process level {level}: {e}")
        
        # Cleanup on failure
        if 'chunk_files' in locals():
            cleanup_temp_files(level, chunk_files)
        
        # Retry logic
        if retry_count < 2:
            logging.info(f"Retrying level {level} after cleanup...")
            cleanup_memory()
            time.sleep(30)
            return process_pressure_level(level, retry_count + 1)
        
        return None

def combine_levels(level_files: Dict[int, str]) -> bool:
    """Combine all pressure levels into final output file"""
    
    logging.info("\nCombining all pressure levels into final file...")
    
    try:
        # Open all level files
        datasets = []
        for level in sorted(level_files.keys()):
            ds = xr.open_dataset(level_files[level])
            datasets.append(ds)
        
        # Concatenate along pressure dimension
        combined = xr.concat(datasets, dim='pressure_level')
        combined = combined.sortby('pressure_level', ascending=False)
        
        # Add global attributes
        combined.attrs = {
            'title': 'Multi-level Rossby Wave Source with zonal mean removed',
            'pressure_levels': ', '.join([f"{l} hPa" for l in sorted(level_files.keys())]),
            'method': 'windspharm package, following Sardeshmukh and Hoskins (1988)',
            'formula': 'RWS = -η * D - (u_chi * ∂η/∂x + v_chi * ∂η/∂y)',
            'processing_date': str(datetime.now()),
            'input_files': 'U_mid_up_glob_seas.nc, V_mid_up_glob_seas.nc',
            'temporal_chunks': str(N_CHUNKS),
            'created_by': 'Multi-level RWS processing script'
        }
        
        encoding = {
            'RWS': {
                'zlib': True,
                'complevel': 4,
                'chunksizes': (1, 12, 180, 360),  # (pressure, time, lat, lon)
                'dtype': 'float32'
            }
        }
        
        # Save final file
        logging.info(f"Saving final output to: {OUTPUT_FILE}")
        combined.to_netcdf(OUTPUT_FILE, encoding=encoding)
        
        # Close datasets
        for ds in datasets:
            ds.close()
        
        # Verify final output
        output_size = os.path.getsize(OUTPUT_FILE) / 1e9  # GB
        logging.info(f"Final output saved: {output_size:.2f} GB")
        
        return True
        
    except Exception as e:
        logging.error(f"Failed to combine levels: {e}")
        return False

# -----------------------------------------------------------------------------
# 4. Main Processing Pipeline
# -----------------------------------------------------------------------------

def main():
    """Main processing pipeline for multi-level RWS computation"""
    
    start_time = time.time()
    
    # 0. Check prerequisites first
    if not check_prerequisite_files():
        return 1
    
    logging.info("="*80)
    logging.info("Multi-level RWS Processing Started")
    logging.info(f"Target levels: {TARGET_LEVELS} hPa")
    logging.info(f"Temporal chunks: {N_CHUNKS}")
    logging.info(f"Max memory: {MAX_MEMORY_GB} GB")
    logging.info("="*80)
    
    # 1. Validate inputs
    if not validate_input_files():
        logging.error("Input validation failed. Exiting.")
        return 1
    
    # 2. Check for existing progress
    completed_levels = get_completed_levels()
    if completed_levels:
        logging.info(f"Found completed levels: {completed_levels}")
        remaining_levels = [l for l in TARGET_LEVELS if l not in completed_levels]
    else:
        remaining_levels = TARGET_LEVELS.copy()
    
    if not remaining_levels:
        logging.info("All levels already processed. Proceeding to combination step.")
    
    # 3. Process each pressure level
    successful_levels = {}
    
    # Include already completed levels
    for level in completed_levels:
        level_file = os.path.join(TEMP_DIR, f'RWS_{level}hPa_complete.nc')
        successful_levels[level] = level_file
    
    # Process remaining levels
    for level in remaining_levels:
        log_memory(f"Starting level {level}")
        
        level_file = process_pressure_level(level)
        
        if level_file and os.path.exists(level_file):
            successful_levels[level] = level_file
            logging.info(f"✓ Level {level} hPa completed successfully")
        else:
            logging.error(f"✗ Level {level} hPa failed")
        
        # Memory cleanup between levels
        cleanup_memory()
        time.sleep(5)
    
    # 4. Check if all levels completed
    if len(successful_levels) != len(TARGET_LEVELS):
        missing = set(TARGET_LEVELS) - set(successful_levels.keys())
        logging.error(f"Missing levels: {missing}. Cannot create final output.")
        return 1
    
    # 5. Combine all levels
    if combine_levels(successful_levels):
        logging.info("✓ Successfully created multi-level RWS file")
        
        # Cleanup level files after successful combination
        logging.info("Cleaning up temporary level files...")
        for level_file in successful_levels.values():
            try:
                os.remove(level_file)
            except OSError as e:
                logging.warning(f"Could not remove temp file {level_file}: {e}")
            except (KeyboardInterrupt, SystemExit):
                raise
        
        # Remove temp directory if empty
        try:
            os.rmdir(TEMP_DIR)
            logging.debug(f"Removed empty temp directory: {TEMP_DIR}")
        except OSError as e:
            logging.debug(f"Could not remove temp directory {TEMP_DIR}: {e}")
        except (KeyboardInterrupt, SystemExit):
            raise
    else:
        logging.error("✗ Failed to create final output file")
        return 1
    
    # 6. Final summary
    elapsed_time = time.time() - start_time
    logging.info("\n" + "="*80)
    logging.info("Processing Complete!")
    logging.info(f"Total time: {elapsed_time/60:.1f} minutes")
    logging.info(f"Output file: {OUTPUT_FILE}")
    logging.info(f"Processed levels: {sorted(successful_levels.keys())}")
    log_memory("Final memory usage")
    logging.info("="*80)
    
    return 0

# -----------------------------------------------------------------------------
# 5. Script Execution
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Main script execution with error handling
    """
    
    try:
        exit_code = main()
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logging.info("\nProcessing interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)