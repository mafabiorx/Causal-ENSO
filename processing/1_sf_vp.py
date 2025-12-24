"""
Atmospheric Dynamics: Streamfunction and Velocity Potential Computation

Core Functionality:
- Compute streamfunction from U/V wind components at multiple pressure levels (200, 250, 850 hPa)
- Calculate velocity potential from wind field decomposition
- Process data in chunks to handle memory constraints for large datasets
- Apply coordinate transformations and interpolation for data quality

Output: Seasonal streamfunction and velocity potential NetCDF files for atmospheric dynamics analysis
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import xarray as xr
from tqdm import tqdm
from windspharm.xarray import VectorWind

# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))
from utils.TBI_functions import long_0_360_to_180_180
from utils.paths import get_data_path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory configuration
INPUT_DIR = get_data_path('seasonal/', data_type="raw")
OUTPUT_DIR = os.path.join(INPUT_DIR, 'temp_files/')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Processing configuration
N_CHUNKS = 20  # Number of temporal chunks for memory efficiency

# Level-specific configurations
PROCESSING_LEVELS = [
    {
        'level': 250,
        'variables': ['streamfunction'],
        'interpolate': True,
        'description': 'Upper tropospheric streamfunction at 250 hPa'
    },
    {
        'level': 200,
        'variables': ['velocity_potential', 'streamfunction'],
        'interpolate': False,
        'description': 'Upper tropospheric dynamics at 200 hPa'
    },
    {
        'level': 850,
        'variables': ['both'],  # Special case: compute both sf and vp together
        'interpolate': False,
        'use_global': True,
        'description': 'Lower tropospheric dynamics at 850 hPa'
    }
]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# =============================================================================
# CORE FUNCTIONS
# =============================================================================

def load_wind_data(level: int, use_global: bool = False) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Load U and V wind components for a specific pressure level.

    Parameters
    ----------
    level : int
        Pressure level in hPa
    use_global : bool, optional
        If True, load global data (with '_glob' suffix), by default False

    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
        U and V wind components
    """
    suffix = '_glob' if use_global else ''

    u_path = os.path.join(INPUT_DIR, f'U_{level}_seas{suffix}.nc')
    v_path = os.path.join(INPUT_DIR, f'V_{level}_seas{suffix}.nc')

    logger.info(f"Loading wind data for {level} hPa (global={use_global})")

    try:
        U = xr.open_dataset(u_path).u.sortby('latitude', ascending=True)
        V = xr.open_dataset(v_path).v.sortby('latitude', ascending=True)

        logger.info(f"  U shape: {U.shape}, V shape: {V.shape}")
        return U, V

    except FileNotFoundError as e:
        logger.error(f"Wind data files not found for level {level}")
        raise FileNotFoundError(f"Missing files: {u_path} or {v_path}") from e
    except Exception as e:
        logger.error(f"Error loading wind data: {e}")
        raise


def apply_coordinate_transforms(U: xr.DataArray, V: xr.DataArray,
                                convert_longitude: bool = True,
                                interpolate_na: bool = False) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Apply coordinate transformations to wind data.

    Parameters
    ----------
    U, V : xr.DataArray
        Wind components
    convert_longitude : bool, optional
        Convert longitude from 0-360 to -180-180, by default True
    interpolate_na : bool, optional
        Interpolate missing values, by default False

    Returns
    -------
    Tuple[xr.DataArray, xr.DataArray]
        Transformed wind components
    """
    logger.info("Applying coordinate transformations")

    if convert_longitude:
        U = long_0_360_to_180_180(U)
        V = long_0_360_to_180_180(V)

    if interpolate_na:
        logger.info("  Interpolating missing values")
        U = U.interpolate_na(dim='latitude', method='cubic').interpolate_na(dim='longitude', method='cubic')
        V = V.interpolate_na(dim='latitude', method='cubic').interpolate_na(dim='longitude', method='cubic')

    return U, V


def get_field_prefix(field_type: str) -> str:
    """
    Get the correct file prefix for a field type.

    Parameters
    ----------
    field_type : str
        Type of field: 'streamfunction', 'velocity_potential', or 'both'

    Returns
    -------
    str
        File prefix: 'sf' for streamfunction, 'vp' for velocity_potential
    """
    prefix_mapping = {
        'streamfunction': 'sf',
        'velocity_potential': 'vp'
    }

    if field_type not in prefix_mapping:
        raise ValueError(f"Unknown field_type: {field_type}. Expected 'streamfunction' or 'velocity_potential'")

    return prefix_mapping[field_type]


def compute_field_from_winds(U_chunk: xr.DataArray, V_chunk: xr.DataArray,
                             field_type: str) -> xr.DataArray:
    """
    Compute diagnostic field from wind components using windspharm.

    Parameters
    ----------
    U_chunk, V_chunk : xr.DataArray
        Wind component chunks
    field_type : str
        Type of field to compute: 'streamfunction', 'velocity_potential', or 'both'

    Returns
    -------
    xr.DataArray or Tuple[xr.DataArray, xr.DataArray]
        Computed field(s)
    """
    # Create VectorWind instance
    w = VectorWind(U_chunk, V_chunk)

    if field_type == 'streamfunction':
        return w.streamfunction()
    elif field_type == 'velocity_potential':
        return w.velocitypotential()
    elif field_type == 'both':
        return w.sfvp()  # Returns (streamfunction, velocity_potential)
    else:
        raise ValueError(f"Unknown field_type: {field_type}")


def process_level_in_chunks(U: xr.DataArray, V: xr.DataArray,
                            level: int, field_type: str,
                            interpolate: bool = False,
                            n_chunks: int = N_CHUNKS) -> List[str]:
    """
    Process a pressure level in temporal chunks.

    Parameters
    ----------
    U, V : xr.DataArray
        Full wind components
    level : int
        Pressure level in hPa
    field_type : str
        Type of field to compute
    interpolate : bool, optional
        Apply interpolation to fill missing values, by default False
    n_chunks : int, optional
        Number of temporal chunks, by default N_CHUNKS

    Returns
    -------
    List[str]
        Paths to chunk files
    """
    logger.info(f"Processing {level} hPa in {n_chunks} chunks for {field_type}")

    # Split time dimension into chunks
    time_chunks = np.array_split(np.arange(len(U.time)), n_chunks)
    chunk_files = []

    for i, chunk_idx in tqdm(enumerate(time_chunks), total=n_chunks,
                             desc=f"{level} hPa {field_type}"):
        try:
            # Extract chunk
            U_chunk = U.isel(time=chunk_idx)
            V_chunk = V.isel(time=chunk_idx)

            # Apply interpolation if requested
            if interpolate:
                U_chunk = U_chunk.interpolate_na(dim='latitude', method='cubic').interpolate_na(dim='longitude', method='cubic')
                V_chunk = V_chunk.interpolate_na(dim='latitude', method='cubic').interpolate_na(dim='longitude', method='cubic')

            # Compute field(s)
            result = compute_field_from_winds(U_chunk, V_chunk, field_type)

            # Save chunk(s)
            if field_type == 'both':
                sf, vp = result
                sf_file = os.path.join(OUTPUT_DIR, f'sf_{level}_seas_chunk_{i}.nc')
                vp_file = os.path.join(OUTPUT_DIR, f'vp_{level}_seas_chunk_{i}.nc')

                sf.to_netcdf(sf_file)
                vp.to_netcdf(vp_file)

                chunk_files.append(('sf', sf_file))
                chunk_files.append(('vp', vp_file))

                # Cleanup
                sf.close()
                vp.close()
                del sf, vp
            else:
                prefix = get_field_prefix(field_type)
                chunk_file = os.path.join(OUTPUT_DIR,
                                         f'{prefix}_{level}_seas_chunk_{i}.nc')
                result.to_netcdf(chunk_file)
                chunk_files.append((prefix, chunk_file))

                # Cleanup
                result.close()
                del result

            # Cleanup chunk data
            U_chunk.close()
            V_chunk.close()
            del U_chunk, V_chunk

        except Exception as e:
            logger.error(f"Error processing chunk {i} for level {level}: {e}")
            raise

    return chunk_files


def concatenate_chunks(chunk_files: List[Tuple[str, str]], level: int) -> None:
    """
    Concatenate temporal chunks and save final output.

    Parameters
    ----------
    chunk_files : List[Tuple[str, str]]
        List of (field_type_prefix, file_path) tuples
    level : int
        Pressure level in hPa
    """
    logger.info(f"Concatenating chunks for {level} hPa")

    # Group files by field type
    files_by_type = {}
    for field_prefix, filepath in chunk_files:
        if field_prefix not in files_by_type:
            files_by_type[field_prefix] = []
        files_by_type[field_prefix].append(filepath)

    # Concatenate each field type
    for field_prefix, files in files_by_type.items():
        logger.info(f"  Concatenating {len(files)} chunks for {field_prefix}")

        # Load and concatenate
        datasets = [xr.open_dataset(f) for f in files]
        combined = xr.concat(datasets, dim='time')

        # Save final output
        output_file = os.path.join(INPUT_DIR, f'{field_prefix}_{level}_seas.nc')
        combined.to_netcdf(output_file)

        logger.info(f"  Saved: {output_file}")

        # Cleanup
        combined.close()
        for ds in datasets:
            ds.close()


def cleanup_temp_files(output_dir: str = OUTPUT_DIR) -> None:
    """
    Delete all temporary chunk files.

    Parameters
    ----------
    output_dir : str, optional
        Directory containing temporary files, by default OUTPUT_DIR
    """
    logger.info("Cleaning up temporary files")

    deleted_count = 0
    for filename in os.listdir(output_dir):
        file_path = os.path.join(output_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
                deleted_count += 1
        except Exception as e:
            logger.warning(f'Failed to delete {file_path}: {e}')

    logger.info(f"  Deleted {deleted_count} temporary files")


def process_pressure_level(level_config: dict) -> None:
    """
    Process a single pressure level according to its configuration.

    Parameters
    ----------
    level_config : dict
        Configuration dictionary containing level, variables, and options
    """
    level = level_config['level']
    logger.info("=" * 70)
    logger.info(f"Processing {level} hPa: {level_config['description']}")
    logger.info("=" * 70)

    start_time = time.time()

    try:
        # Load wind data
        use_global = level_config.get('use_global', False)
        U, V = load_wind_data(level, use_global=use_global)

        # Apply transformations
        convert_lon = level_config.get('convert_longitude', True)
        U, V = apply_coordinate_transforms(U, V, convert_longitude=convert_lon)

        # Process based on variable type
        variables = level_config['variables']
        interpolate = level_config.get('interpolate', False)

        if 'both' in variables:
            # Special case: compute sf and vp together
            chunk_files = process_level_in_chunks(U, V, level, 'both', interpolate=interpolate)
        else:
            # Process each variable separately
            chunk_files = []
            for var in variables:
                var_chunks = process_level_in_chunks(U, V, level, var, interpolate=interpolate)
                chunk_files.extend(var_chunks)

        # Cleanup source data
        U.close()
        V.close()
        del U, V

        # Concatenate results
        concatenate_chunks(chunk_files, level)

        # Cleanup temporary files
        cleanup_temp_files()

        elapsed = time.time() - start_time
        logger.info(f"Completed {level} hPa in {elapsed/60:.1f} minutes")

    except Exception as e:
        logger.error(f"Failed to process {level} hPa: {e}")
        raise


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main processing pipeline."""
    logger.info("=" * 70)
    logger.info("STREAMFUNCTION AND VELOCITY POTENTIAL COMPUTATION")
    logger.info("=" * 70)
    logger.info(f"Input directory: {INPUT_DIR}")
    logger.info(f"Temporary directory: {OUTPUT_DIR}")
    logger.info(f"Number of temporal chunks: {N_CHUNKS}")
    logger.info(f"Processing {len(PROCESSING_LEVELS)} pressure levels")
    logger.info("=" * 70)

    overall_start = time.time()

    # Process each pressure level
    for level_config in PROCESSING_LEVELS:
        try:
            process_pressure_level(level_config)
        except Exception as e:
            logger.error(f"Stopping due to error in {level_config['level']} hPa: {e}")
            raise

    # Final summary
    overall_elapsed = time.time() - overall_start
    logger.info("=" * 70)
    logger.info("ALL PROCESSING COMPLETE")
    logger.info(f"Total time: {overall_elapsed/60:.1f} minutes")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
