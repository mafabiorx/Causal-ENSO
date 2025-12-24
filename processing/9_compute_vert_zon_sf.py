"""
Compute vertical streamfunction from global zonal wind data.

This script processes a large (15GB) NetCDF file containing global zonal winds
at multiple pressure levels and computes the vertical streamfunction using
vertical integration: dψ/dp = -u

The implementation uses memory-efficient chunking strategies to handle the
large dataset on limited hardware.
"""

import os
import sys
import time
import gc
import glob
import logging
from pathlib import Path
from datetime import datetime
import json

import numpy as np
import xarray as xr
import dask
import dask.array as da
from scipy.integrate import cumulative_trapezoid
import psutil
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.paths import get_data_path
from utils.TBI_functions import long_0_360_to_180_180

# === CONFIGURATION ===

# Physical constants
R = 6.371e6  # Earth's radius in meters
g = 9.81     # Gravitational acceleration in m/s^2

# Paths
INPUT_DIR = get_data_path('seasonal/', data_type="raw")
INPUT_FILE = os.path.join(INPUT_DIR, 'U_all_glob_seas.nc')
OUTPUT_DIR = INPUT_DIR  # Save in same directory
TEMP_DIR = os.path.join(INPUT_DIR, 'temp_vertical_sf/')
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'vertical_streamfunction_seas.nc')
CHECKPOINT_FILE = os.path.join(TEMP_DIR, 'processing_checkpoint.json')

# Pressure levels (hPa)
PRESSURE_LEVELS_HPA = [100, 150, 200, 250, 400, 500, 600, 700, 800, 850, 925, 1000]

# Chunking parameters
N_LON_CHUNKS = 12  # 360°/30° = 12 chunks
N_TIME_CHUNKS = 20
LON_CHUNK_SIZE = 30  # degrees

# Memory management
MEMORY_LIMIT_FRACTION = 0.6  # Use max 60% of available memory
TIME_DELAY_SECONDS = 30  # Delay between chunks


# === UTILITY FUNCTIONS ===

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1e9


def get_available_memory():
    """Get available system memory in GB."""
    return psutil.virtual_memory().available / 1e9


def load_checkpoint():
    """Load processing checkpoint if it exists."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'completed_chunks': []}


def save_checkpoint(checkpoint):
    """Save processing checkpoint."""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f)


def vertical_integration(u_data, pressure_pa, latitude):
    """
    Perform vertical integration to compute mass streamfunction.

    Parameters:
    -----------
    u_data : numpy.ndarray
        Zonal wind data with shape (..., n_pressure)
    pressure_pa : numpy.ndarray
        Pressure levels in Pascals
    latitude : numpy.ndarray
        Latitude values in degrees

    Returns:
    --------
    psi : numpy.ndarray
        Vertical mass streamfunction with same shape as u_data
    """
    # Ensure pressure is increasing (top to bottom)
    if pressure_pa[0] > pressure_pa[-1]:
        pressure_pa = pressure_pa[::-1]
        u_data = u_data[..., ::-1]

    # Initialize output array
    psi = np.zeros_like(u_data)

    # Get shape for reshaping
    original_shape = u_data.shape
    n_levels = original_shape[-1]

    # Calculate latitude-dependent factor (2πR cos(lat) / g)
    lat_rad = np.deg2rad(latitude)
    cos_lat = np.cos(lat_rad)

    if len(original_shape) == 4:  # (time, lat, lon, pressure)
        cos_lat_broadcast = cos_lat[np.newaxis, :, np.newaxis, np.newaxis]
    elif len(original_shape) == 3:  # (lat, lon, pressure)
        cos_lat_broadcast = cos_lat[:, np.newaxis, np.newaxis]
    else:
        raise ValueError(f"Unexpected shape: {original_shape}")

    # Apply the factor: u * 2πR cos(lat) / g
    factor = u_data * 2 * np.pi * R * cos_lat_broadcast / g

    # Reshape to 2D for easier processing
    factor_flat = factor.reshape(-1, n_levels)
    psi_flat = psi.reshape(-1, n_levels)

    # Perform integration for each profile
    # ψ(p) = ∫[p_top to p] (u * 2πR cos(lat) / g) dp'
    for i in range(factor_flat.shape[0]):
        if not np.all(np.isnan(factor_flat[i, :])):
            integral = cumulative_trapezoid(factor_flat[i, :], pressure_pa, initial=0)
            psi_flat[i, :] = integral

    # Reshape back to original shape
    return psi_flat.reshape(original_shape)


def process_chunk(ds_chunk, chunk_id):
    """
    Process a single chunk of data.

    Parameters:
    -----------
    ds_chunk : xarray.Dataset
        Chunk of U data
    chunk_id : tuple
        (lon_idx, time_idx) identifying the chunk

    Returns:
    --------
    str : Path to saved temporary file
    """
    logging.info(f"Processing chunk {chunk_id}")
    start_time = time.time()

    # Extract zonal wind data
    u_data = ds_chunk['u'].values  # shape: (time, pressure, lat, lon)

    # Extract latitude values
    latitude = ds_chunk['latitude'].values

    # Convert pressure levels to Pa
    pressure_pa = np.array(PRESSURE_LEVELS_HPA) * 100.0

    # Rearrange dimensions for integration (put pressure last)
    u_data = np.transpose(u_data, (0, 2, 3, 1))  # (time, lat, lon, pressure)

    # Perform vertical integration
    psi_data = vertical_integration(u_data, pressure_pa, latitude)

    # Rearrange back to original dimension order
    psi_data = np.transpose(psi_data, (0, 3, 1, 2))  # (time, pressure, lat, lon)

    # Create output dataset
    psi_da = xr.DataArray(
        psi_data,
        dims=['time', 'pressure_level', 'latitude', 'longitude'],
        coords={
            'time': ds_chunk['time'],
            'pressure_level': ds_chunk['pressure_level'],
            'latitude': ds_chunk['latitude'],
            'longitude': ds_chunk['longitude']
        }
    )

    psi_da.attrs = {
        'long_name': 'Zonal Mass Streamfunction',
        'units': 'kg s-1',
        'description': 'Zonal mass streamfunction computed from zonal wind',
        'method': 'Cumulative trapezoid integration with latitude-dependent factor',
        'boundary_condition': 'psi=0 at top of atmosphere',
        'formula': 'psi(p) = integral from p_top to p of (u * 2*pi*R*cos(lat) / g) dp'
    }

    # Save to temporary file
    temp_filename = f'psi_chunk_lon{chunk_id[0]:02d}_time{chunk_id[1]:02d}.nc'
    temp_filepath = os.path.join(TEMP_DIR, temp_filename)

    psi_ds = xr.Dataset({'psi': psi_da})
    psi_ds.to_netcdf(temp_filepath, engine='netcdf4', format='NETCDF4')

    elapsed = time.time() - start_time
    logging.info(f"Chunk {chunk_id} completed in {elapsed:.1f} seconds")

    # Clean up
    del u_data, psi_data, psi_da, psi_ds
    gc.collect()

    return temp_filepath


def setup_processing_environment():
    """
    Set up the processing environment including directories and logging.

    Returns:
    --------
    logging.Logger : Configured logger instance
    """
    # Create directories
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(TEMP_DIR, 'processing.log')),
            logging.StreamHandler()
        ]
    )

    logging.info("=" * 60)
    logging.info("Starting vertical streamfunction computation")
    logging.info(f"Input file: {INPUT_FILE}")
    logging.info(f"Output file: {OUTPUT_FILE}")
    logging.info("=" * 60)

    # Check input file exists
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")


def process_all_chunks(ds, completed_chunks):
    """
    Process all data chunks with checkpointing and memory management.

    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset opened with dask
    completed_chunks : list
        List of already completed chunk IDs

    Returns:
    --------
    list : List of temporary file paths
    """
    n_time = len(ds.time)
    n_lon = len(ds.longitude)

    # Calculate chunk boundaries
    lon_chunks = np.array_split(np.arange(n_lon), N_LON_CHUNKS)
    time_chunks = np.array_split(np.arange(n_time), N_TIME_CHUNKS)

    # Process chunks
    temp_files = []
    total_chunks = N_LON_CHUNKS * N_TIME_CHUNKS

    with tqdm(total=total_chunks, desc="Processing chunks") as pbar:
        for lon_idx, lon_chunk in enumerate(lon_chunks):
            for time_idx, time_chunk in enumerate(time_chunks):
                chunk_id = (lon_idx, time_idx)

                # Skip if already completed
                if str(chunk_id) in completed_chunks:
                    logging.info(f"Skipping completed chunk {chunk_id}")
                    temp_filename = f'psi_chunk_lon{chunk_id[0]:02d}_time{chunk_id[1]:02d}.nc'
                    temp_files.append(os.path.join(TEMP_DIR, temp_filename))
                    pbar.update(1)
                    continue

                # Check memory
                available_mem = get_available_memory()
                current_mem = get_memory_usage()
                logging.info(f"Memory: using {current_mem:.1f}GB, available {available_mem:.1f}GB")

                if available_mem < 2.0:  # Less than 2GB available
                    logging.warning("Low memory, triggering garbage collection")
                    gc.collect()
                    time.sleep(10)

                # Extract chunk
                ds_chunk = ds.isel(longitude=lon_chunk, time=time_chunk)

                # Process chunk
                try:
                    temp_file = process_chunk(ds_chunk, chunk_id)
                    temp_files.append(temp_file)

                    # Update checkpoint
                    completed_chunks.append(str(chunk_id))
                    save_checkpoint({'completed_chunks': completed_chunks})

                except Exception as e:
                    logging.error(f"Error processing chunk {chunk_id}: {e}")
                    raise

                # Clean up
                ds_chunk.close()
                del ds_chunk
                gc.collect()

                # Delay between chunks
                time.sleep(TIME_DELAY_SECONDS)
                pbar.update(1)

    return temp_files


def concatenate_temp_files():
    """
    Concatenate all temporary chunk files into the final output.

    Returns:
    --------
    bool : True if concatenation successful
    """
    logging.info("=" * 60)
    logging.info("Concatenating temporary files...")

    # Get all temporary files with pattern
    pattern = os.path.join(TEMP_DIR, 'psi_chunk_*.nc')
    temp_files = sorted(glob.glob(pattern))

    if not temp_files:
        logging.error(f"No temporary files found matching pattern: {pattern}")
        raise ValueError("No temporary files to concatenate")

    logging.info(f"Found {len(temp_files)} temporary files to concatenate")

    # Organize files into nested structure for 2D concatenation
    # Files are named as psi_chunk_lon{LON_IDX}_time{TIME_IDX}.nc
    logging.info("Organizing files for concatenation...")

    # Create nested list structure: outer list is time, inner list is longitude
    file_grid = []
    for time_idx in range(N_TIME_CHUNKS):
        time_row = []
        for lon_idx in range(N_LON_CHUNKS):
            filename = f'psi_chunk_lon{lon_idx:02d}_time{time_idx:02d}.nc'
            filepath = os.path.join(TEMP_DIR, filename)
            if os.path.exists(filepath):
                time_row.append(filepath)
            else:
                logging.error(f"Missing expected file: {filepath}")
                raise FileNotFoundError(f"Missing chunk file: {filename}")
        file_grid.append(time_row)

    # Open files lazily with dask using nested structure
    logging.info("Opening files with xr.open_mfdataset...")
    psi_final = xr.open_mfdataset(
        file_grid,
        combine='nested',
        concat_dim=['time', 'longitude'],
        parallel=True,
        chunks={'time': 24, 'longitude': 60}  # Reasonable chunk sizes
    )

    # Rechunk for efficient writing
    logging.info("Rechunking for output...")
    psi_final = psi_final.chunk({
        'time': 12,
        'pressure_level': -1,  # Keep pressure levels together
        'latitude': -1,        # Keep latitude together
        'longitude': 60        # Chunk longitude
    })

    # Configure dask to handle large chunks
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        # Save final output
        logging.info(f"Saving final output to {OUTPUT_FILE}")
        psi_final.to_netcdf(
            OUTPUT_FILE,
            engine='netcdf4',
            format='NETCDF4',
            unlimited_dims=['time'],
            compute=True  # Force computation
        )

    # Close the dataset
    psi_final.close()
    logging.info("Output file saved successfully")

    return True


def cleanup_temp_files():
    """
    Clean up temporary files and directories after successful processing.
    """
    # Get list of temp files
    pattern = os.path.join(TEMP_DIR, 'psi_chunk_*.nc')
    temp_files = glob.glob(pattern)

    logging.info("Cleaning up temporary files...")
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            logging.debug(f"Removed: {temp_file}")

    # Remove checkpoint file
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)
        logging.info("Removed checkpoint file")

    # Remove temp directory if empty
    try:
        os.rmdir(TEMP_DIR)
        logging.info("Removed empty temp directory")
    except OSError:
        pass  # Directory not empty or other error


def verify_output():
    """
    Verify the output file was created successfully.

    Returns:
    --------
    bool : True if output file exists and is valid
    """
    if os.path.exists(OUTPUT_FILE):
        output_size = os.path.getsize(OUTPUT_FILE) / 1e9  # Size in GB
        logging.info(f"Output file created successfully, size: {output_size:.2f} GB")
        return True
    else:
        logging.error(f"Output file not created: {OUTPUT_FILE}")
        logging.error("Temporary files NOT removed - please check the error and retry")
        return False


# === MAIN PROCESSING ===

def main():
    """Main processing function."""
    # Setup environment
    setup_processing_environment()

    # Load checkpoint
    checkpoint = load_checkpoint()
    completed_chunks = checkpoint['completed_chunks']

    # Open dataset with dask
    logging.info("Opening dataset...")
    ds = xr.open_dataset(INPUT_FILE, chunks={'time': 24})

    # Log dimensions
    n_time = len(ds.time)
    n_lon = len(ds.longitude)
    n_lat = len(ds.latitude)
    n_pressure = len(ds.pressure_level)
    logging.info(f"Dataset dimensions: time={n_time}, lon={n_lon}, lat={n_lat}, pressure={n_pressure}")

    # Convert longitude if needed (0-360 to -180-180)
    if ds.longitude.min() >= 0 and ds.longitude.max() > 180:
        logging.info("Converting longitude from 0-360 to -180-180")
        ds = long_0_360_to_180_180(ds)

    # Process all chunks
    process_all_chunks(ds, completed_chunks)

    # Close input dataset
    ds.close()

    # Concatenate temporary files
    try:
        concatenate_temp_files()
    except Exception as e:
        logging.error(f"Error during concatenation: {e}")
        raise

    # Verify output and cleanup
    if verify_output():
        cleanup_temp_files()
    else:
        raise RuntimeError("Failed to create output file")

    # Final logging
    logging.info("=" * 60)
    logging.info("Processing complete!")
    logging.info(f"Output saved to: {OUTPUT_FILE}")

    # Final memory report
    final_mem = get_memory_usage()
    logging.info(f"Final memory usage: {final_mem:.1f}GB")


if __name__ == "__main__":
    main()
