"""
download_utils.py
=================

Shared utilities for robust ERA5 data downloading.

This module provides modern, type-safe infrastructure for downloading ERA5
reanalysis data via the CDS API.

Design Goals:
- Type-safe task management with dataclasses
- Robust file validation with magic byte checking
- Transparent archive handling (ZIP/TAR formats)
- Multi-engine xarray support with graceful fallbacks
- Comprehensive error classification and retry logic
- CF-compliant preprocessing for ERA5 rectilinear grids
- Integration with existing utils.paths infrastructure

Scientific Integrity:
- Maintains all validation and verification steps from original scripts
- Preserves data integrity through comprehensive file checking
- Ensures CF-compliance for interoperability with climate tools
- Follows established climate data processing best practices
"""

from __future__ import annotations

import os
import sys
import time
import glob
import shutil
import logging
import tarfile
import zipfile
import tempfile
import importlib.util
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Union, Sequence
from collections import defaultdict

import cdsapi
import xarray as xr
import dask
import pandas as pd

# Import project path utilities
from .paths import RAW_DATA_DIR, create_directories

# Ensure directories exist
create_directories()

# ================================= Constants =================================

# NetCDF file signatures for magic byte validation
NETCDF_SIGNATURES = {b'CDF\x01', b'CDF\x02', b'\x89HDF'}

# Error classification taxonomy with retry recommendations
ERROR_TAXONOMY = {
    'ssl_error': (True, 60),              # (should_retry, base_delay_seconds)
    'timeout': (True, 30),
    'connection_refused': (True, 60),
    'broken_pipe': (True, 60),
    'temp_artifact_missing': (True, 30),
    'quota_exceeded': (False, 0),         # Don't retry quota errors
    'auth_error': (False, 0),             # Don't retry auth errors
    'not_found': (False, 0),              # Don't retry 404 errors
    'unknown': (True, 60),                # Conservative retry for unknown errors
}


# ================================ Date Helpers ==============================


def parse_year_month(value: str) -> Tuple[int, int]:
    """Return (year, month) pair from a YYYY-MM string."""
    try:
        year_str, month_str = value.split('-', 1)
        year = int(year_str)
        month = int(month_str)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise ValueError(f"Invalid year-month string: {value!r}") from exc
    if not (1 <= month <= 12):
        raise ValueError(f"Month out of range in {value!r}")
    return year, month


def iter_year_months(
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> List[Tuple[int, int]]:
    """Inclusive list of (year, month) pairs covering the provided window."""
    pairs: List[Tuple[int, int]] = []
    year, month = start_year, start_month
    while True:
        pairs.append((year, month))
        if (year, month) == (end_year, end_month):
            break
        month += 1
        if month == 13:
            month = 1
            year += 1
    return pairs


def months_for_year(
    year: int,
    start_year: int,
    start_month: int,
    end_year: int,
    end_month: int,
) -> List[str]:
    """Return zero-padded months for *year* falling within the inclusive span."""
    if year < start_year or year > end_year:
        return []
    months = [m for (y, m) in iter_year_months(start_year, start_month, end_year, end_month) if y == year]
    return [f"{m:02d}" for m in months]


def months_for_year_str(year: int, start: str, end: str) -> List[str]:
    """Wrapper around months_for_year accepting YYYY-MM strings."""
    s_year, s_month = parse_year_month(start)
    e_year, e_month = parse_year_month(end)
    return months_for_year(year, s_year, s_month, e_year, e_month)


def year_range(start_year: int, end_year: int) -> List[int]:
    """Inclusive list of years between the provided bounds."""
    if start_year > end_year:
        return []
    return list(range(start_year, end_year + 1))

# Default robustness configuration
DEFAULT_CONFIG = {
    'verify_before_concat': True,
    'allow_partial_concat': False,
    'max_download_retries': 3,
    'max_global_retries': 3,
    'retry_delay_seconds': 60,
    'min_file_size_bytes': 10000,
    'exponential_backoff': True,
    'max_workers': 8,
    'retry_workers': 5,
}

# ================================= Dataclasses ===============================

@dataclass
class DownloadConfig:
    """Configuration for ERA5 download operations."""
    start_year: int
    end_year: int
    variables: List[str]
    start_month: int = 1
    end_month: int = 12
    variable_mappings: Dict[str, str] = field(default_factory=dict)
    grid: Optional[List[float]] = None
    area: Optional[List[float]] = None
    pressure_levels: Optional[List[str]] = None  # For multi-level data

    # Robustness settings
    verify_before_concat: bool = True
    allow_partial_concat: bool = False
    max_download_retries: int = 3
    max_global_retries: int = 3
    retry_delay_seconds: int = 60
    min_file_size_bytes: int = 10000
    exponential_backoff: bool = True
    max_workers: int = 8
    retry_workers: int = 5

    # Directory settings
    base_dir: Optional[str] = None
    logs_dir: Optional[str] = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

        # Set default directories if not provided
        if self.base_dir is None:
            self.base_dir = RAW_DATA_DIR
        if self.logs_dir is None:
            self.logs_dir = os.path.join(self.base_dir, 'logs')

        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

    def validate(self):
        """Validate configuration parameters."""
        if self.start_year > self.end_year:
            raise ValueError(f"start_year ({self.start_year}) must be <= end_year ({self.end_year})")

        if not (1 <= self.start_month <= 12):
            raise ValueError(f"start_month ({self.start_month}) must be within 1-12")

        if not (1 <= self.end_month <= 12):
            raise ValueError(f"end_month ({self.end_month}) must be within 1-12")

        if self.start_year == self.end_year and self.start_month > self.end_month:
            raise ValueError(
                f"start_month ({self.start_month}) must be <= end_month ({self.end_month}) when years are equal"
            )

        if not self.variables:
            raise ValueError("variables list cannot be empty")

        if self.grid is not None:
            validate_grid_parameter(self.grid)

        if self.area is not None:
            validate_area_parameter(self.area)

@dataclass
class DownloadTask:
    """Type-safe container for individual download tasks."""
    year: int
    variable: str
    months: List[str]
    dataset: str
    output_path: str
    grid: Optional[List[float]] = None
    area: Optional[List[float]] = None
    pressure_levels: Optional[List[str]] = None
    retry_count: int = 0

    def validate(self) -> None:
        """Validate task parameters before execution."""
        if not (1900 <= self.year <= 2100):
            raise ValueError(f"Invalid year: {self.year}")

        if not self.months:
            raise ValueError("months list cannot be empty")

        for month in self.months:
            if not (month.isdigit() and 1 <= int(month) <= 12):
                raise ValueError(f"Invalid month: {month}")

        if not self.variable:
            raise ValueError("variable cannot be empty")

        if self.grid is not None:
            validate_grid_parameter(self.grid)

        if self.area is not None:
            validate_area_parameter(self.area)

    def build_request(self) -> Dict:
        """Build CDS API request dictionary."""
        request = {
            'product_type': ['monthly_averaged_reanalysis'],
            'variable': [self.variable],
            'year': [str(self.year)],
            'month': self.months,
            'time': '00:00',
            'data_format': 'netcdf',
        }

        if self.pressure_levels is not None:
            request['pressure_level'] = self.pressure_levels

        if self.grid is not None:
            request['grid'] = self.grid

        if self.area is not None:
            request['area'] = self.area

        return request

# ================================= Validation Functions ======================

def validate_grid_parameter(grid: Union[List[float], Tuple[float, float]]) -> None:
    """
    Validate CDS grid parameter format.

    Parameters:
        grid: Grid resolution as [lat_res, lon_res] in degrees

    Raises:
        ValueError: If format is invalid
    """
    if not isinstance(grid, (list, tuple)):
        raise ValueError(f"Grid must be a list or tuple. Got: {type(grid).__name__}")

    if len(grid) != 2:
        raise ValueError(f"Grid must contain exactly two values [lat_res, lon_res]. Got {len(grid)} values: {grid}")

    try:
        lat_res, lon_res = [float(x) for x in grid]
    except (ValueError, TypeError):
        raise ValueError(f"Grid values must be numeric. Got: {grid}")

    if lat_res <= 0 or lon_res <= 0:
        raise ValueError(f"Grid resolution values must be positive. Got: lat_res={lat_res}, lon_res={lon_res}")

def validate_area_parameter(area: Union[List[float], Tuple[float, float, float, float]]) -> None:
    """
    Validate CDS area parameter format.

    Parameters:
        area: Spatial bounds as [N, W, S, E] in degrees

    Raises:
        ValueError: If format is invalid
    """
    if not isinstance(area, (list, tuple)):
        raise ValueError(f"Area must be a list or tuple. Got: {type(area).__name__}")

    if len(area) != 4:
        raise ValueError(f"Area must contain exactly four values [N, W, S, E]. Got {len(area)} values: {area}")

    try:
        north, west, south, east = [float(x) for x in area]
    except (ValueError, TypeError):
        raise ValueError(f"Area values must be numeric. Got: {area}")

    # Validate latitude bounds
    if not (-90 <= north <= 90) or not (-90 <= south <= 90):
        raise ValueError(f"Latitude values must be between -90 and 90 degrees. Got: North={north}, South={south}")

    # Validate longitude bounds
    if not (-180 <= west <= 180) or not (-180 <= east <= 180):
        raise ValueError(f"Longitude values must be between -180 and 180 degrees. Got: West={west}, East={east}")

    # Validate logical bounds
    if north <= south:
        raise ValueError(f"North latitude must be greater than South latitude. Got: North={north}, South={south}")

# ================================= Utility Classes ===========================

class FileValidator:
    """Advanced file validation with magic byte checking and multi-engine support."""

    def __init__(self, min_size_bytes: int = 10000):
        self.min_size_bytes = min_size_bytes
        self.preferred_engines = self._get_available_engines()

    def _get_available_engines(self) -> List[str]:
        """Determine available xarray engines in order of preference."""
        engines = []
        engine_modules = {
            'netcdf4': 'netCDF4',
            'h5netcdf': 'h5netcdf',
        }

        for engine, module in engine_modules.items():
            if importlib.util.find_spec(module) is not None:
                engines.append(engine)

        return engines if engines else [None]  # Fallback to default

    def read_file_magic(self, path: Path, nbytes: int = 4) -> bytes:
        """Read the first nbytes of a file for magic byte checking."""
        try:
            with path.open('rb') as fh:
                return fh.read(nbytes)
        except OSError:
            return b""

    def is_probably_netcdf(self, path: Path) -> bool:
        """Check if file has NetCDF/HDF5 magic signature."""
        magic = self.read_file_magic(path, 4)
        return magic in NETCDF_SIGNATURES

    def validate_size(self, path: Path) -> bool:
        """Check if file meets minimum size requirement."""
        if not path.exists():
            return False
        return path.stat().st_size >= self.min_size_bytes

    def validate_content(self, path: Path, expected_var: Optional[str] = None) -> bool:
        """
        Validate file content with multi-engine fallback.

        Parameters:
            path: Path to NetCDF file
            expected_var: Variable name to check for (optional)

        Returns:
            True if file is valid and contains expected variable
        """
        if not self.is_probably_netcdf(path):
            return False

        for engine in self.preferred_engines:
            try:
                open_kwargs = {}
                if engine is not None:
                    open_kwargs['engine'] = engine

                with xr.open_dataset(str(path), **open_kwargs) as ds:
                    data_vars = list(ds.data_vars)

                    if not data_vars:
                        return False

                    if expected_var is not None:
                        return expected_var in data_vars

                    return True

            except (OSError, ValueError):
                # Expected errors for incompatible file formats
                continue
            except Exception as e:
                # Unexpected errors - log for investigation
                logging.debug(f"Unexpected error validating {path} with engine {engine}: {e}")
                continue

        return False

    def is_valid_download(self, path: Path, expected_var: Optional[str] = None) -> bool:
        """
        Comprehensive validation combining size, magic bytes, and content.

        Parameters:
            path: Path to downloaded file
            expected_var: Expected variable name

        Returns:
            True if file passes all validation checks
        """
        return (self.validate_size(path) and
                self.validate_content(path, expected_var))

class ArchiveHandler:
    """Handle transparent extraction of ZIP and TAR archives from CDS."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def process_download(self, downloaded_path: Path, final_path: Path) -> Path:
        """
        Process downloaded file, extracting archives if needed.

        Parameters:
            downloaded_path: Path to downloaded file (possibly archive)
            final_path: Desired final path for NetCDF file

        Returns:
            Path to final NetCDF file
        """
        if not downloaded_path.exists():
            raise FileNotFoundError(f"Downloaded file not found: {downloaded_path}")

        # Ensure parent directory exists
        final_path.parent.mkdir(parents=True, exist_ok=True)

        # Check if it's a ZIP archive
        if zipfile.is_zipfile(downloaded_path):
            return self._extract_zip(downloaded_path, final_path)

        # Check if it's a TAR archive
        if tarfile.is_tarfile(downloaded_path):
            return self._extract_tar(downloaded_path, final_path)

        # Not an archive - treat as NetCDF and apply preprocessing
        return self._process_netcdf(downloaded_path, final_path)

    def _extract_zip(self, zip_path: Path, final_path: Path) -> Path:
        """Extract NetCDF files from ZIP archive."""
        self.logger.debug(f"[ARCHIVE] Extracting ZIP: {zip_path.name}")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                extracted_files = []

                with zipfile.ZipFile(zip_path) as zf:
                    nc_members = [m for m in zf.namelist() if m.lower().endswith('.nc')]

                    if not nc_members:
                        raise RuntimeError(f"ZIP archive {zip_path} contains no .nc files")

                    for member in nc_members:
                        zf.extract(member, tmpdir)
                        extracted_files.append(tmpdir_path / Path(member).name)

                self._combine_netcdf_files(extracted_files, final_path)

        finally:
            zip_path.unlink(missing_ok=True)

        return final_path

    def _extract_tar(self, tar_path: Path, final_path: Path) -> Path:
        """Extract NetCDF files from TAR archive."""
        self.logger.debug(f"[ARCHIVE] Extracting TAR: {tar_path.name}")

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                extracted_files = []

                with tarfile.open(tar_path) as tf:
                    nc_members = [m for m in tf.getmembers() if m.name.lower().endswith('.nc')]

                    if not nc_members:
                        raise RuntimeError(f"TAR archive {tar_path} contains no .nc files")

                    for member in nc_members:
                        tf.extract(member, path=tmpdir)
                        extracted_files.append(tmpdir_path / Path(member.name).name)

                self._combine_netcdf_files(extracted_files, final_path)

        finally:
            tar_path.unlink(missing_ok=True)

        return final_path

    def _process_netcdf(self, source_path: Path, final_path: Path) -> Path:
        """Process single NetCDF file with preprocessing."""
        self._combine_netcdf_files([source_path], final_path)
        source_path.unlink(missing_ok=True)
        return final_path

    def _combine_netcdf_files(self, file_paths: List[Path], output_path: Path) -> None:
        """Combine multiple NetCDF files with preprocessing."""
        file_paths = sorted(file_paths)

        if output_path.exists():
            output_path.unlink()

        # Try different engines for robust opening
        engines = ['netcdf4', 'h5netcdf'] if self._engine_available('netcdf4') else ['h5netcdf']
        if not self._engine_available('h5netcdf'):
            engines = [None]  # Fallback to default

        ds = None
        for engine in engines:
            try:
                open_kwargs = {
                    'combine': 'by_coords',
                    'preprocess': preprocess_era5_dataset,
                    'parallel': False,
                }
                if engine is not None:
                    open_kwargs['engine'] = engine

                ds = xr.open_mfdataset([str(p) for p in file_paths], **open_kwargs)
                break

            except Exception as e:
                self.logger.warning(f"Engine {engine} failed to open files: {e}")
                continue

        if ds is None:
            raise RuntimeError("All engines failed to open files")

        try:
            # Sort by time if present
            if 'time' in ds.coords:
                ds = ds.sortby('time')

            # Clean encoding to prevent write conflicts
            for var in ds.data_vars:
                ds[var].encoding.pop('coordinates', None)

            # Write with preferred engine
            write_kwargs = {'unlimited_dims': ('time',), 'compute': True}
            if self._engine_available('netcdf4'):
                ds.to_netcdf(str(output_path), engine='netcdf4', format='NETCDF4', **write_kwargs)
            else:
                ds.to_netcdf(str(output_path), **write_kwargs)

        except Exception:
            if output_path.exists():
                output_path.unlink()
            raise
        finally:
            ds.close()

    def _engine_available(self, engine: str) -> bool:
        """Check if xarray engine is available."""
        module_map = {'netcdf4': 'netCDF4', 'h5netcdf': 'h5netcdf'}
        module_name = module_map.get(engine)
        return module_name and importlib.util.find_spec(module_name) is not None

# ================================= Error Handling ============================

def classify_download_error(exception: Exception) -> Tuple[str, bool]:
    """
    Classify download errors for intelligent retry decisions.

    Parameters:
        exception: Exception that occurred during download

    Returns:
        Tuple of (error_type, should_retry)
    """
    error_str = str(exception).lower()

    if "ssl" in error_str or "certificate" in error_str:
        return "ssl_error", True
    elif "timeout" in error_str or "timed out" in error_str:
        return "timeout", True
    elif "connection" in error_str and "refused" in error_str:
        return "connection_refused", True
    elif "broken pipe" in error_str:
        return "broken_pipe", True
    elif "download artefact missing" in error_str or "temporary download artefact" in error_str:
        return "temp_artifact_missing", True
    elif "quota" in error_str or "limit" in error_str:
        return "quota_exceeded", False
    elif "authentication" in error_str or "unauthorized" in error_str or "forbidden" in error_str:
        return "auth_error", False
    elif "404" in error_str or "not found" in error_str:
        return "not_found", False
    else:
        return "unknown", True

# ================================= Preprocessing ==============================

def preprocess_era5_dataset(ds: xr.Dataset) -> xr.Dataset:
    """
    Comprehensive preprocessing for ERA5 datasets.

    Applies coordinate harmonization, CF-compliance, and cleanup operations
    optimized for ERA5 rectilinear grids.

    Parameters:
        ds: Raw ERA5 dataset

    Returns:
        Preprocessed dataset with harmonized coordinates and attributes
    """
    # 1. Remove extraneous variables that CDS sometimes includes
    drop_vars = ['number', 'expver']
    ds = ds.drop_vars(drop_vars, errors='ignore')

    # 2. Remove bounds variables (often cause issues with CDO/NCO)
    bounds_patterns = ['_bnds', '_bounds', 'bounds_']
    bounds_vars = [
        var for var in ds.variables
        if any(pattern in var.lower() for pattern in bounds_patterns)
    ]
    if bounds_vars:
        ds = ds.drop_vars(bounds_vars, errors='ignore')

    # 3. Standardize time coordinate naming
    time_renames = {}
    if 'valid_time' in ds.coords:
        time_renames['valid_time'] = 'time'
    if time_renames:
        ds = ds.rename(time_renames)

    # Ensure time is a coordinate
    if 'time' in ds.variables and 'time' not in ds.coords:
        ds = ds.set_coords('time')

    # 4. Ensure latitude is ascending (ERA5 sometimes delivers descending)
    if 'latitude' in ds.coords:
        lat_vals = ds.latitude.values
        if lat_vals.ndim == 1 and len(lat_vals) > 1 and lat_vals[0] > lat_vals[-1]:
            ds = ds.reindex(latitude=list(reversed(ds.latitude)))

    # 5. Add CF-compliant attributes for better interoperability
    coord_attrs = {
        'latitude': {
            'standard_name': 'latitude',
            'long_name': 'Latitude',
            'units': 'degrees_north',
            'axis': 'Y'
        },
        'longitude': {
            'standard_name': 'longitude',
            'long_name': 'Longitude',
            'units': 'degrees_east',
            'axis': 'X'
        },
        'time': {
            'standard_name': 'time',
            'long_name': 'Time',
            'axis': 'T'
        }
    }

    for coord_name, attrs in coord_attrs.items():
        if coord_name in ds.coords:
            for key, value in attrs.items():
                ds[coord_name].attrs.setdefault(key, value)

    # 6. Set preferred dimension order for optimal performance
    if 'pressure_level' in ds.dims:
        # Multi-level data: time, pressure_level, latitude, longitude
        preferred_order = ['time', 'pressure_level', 'latitude', 'longitude']
    else:
        # Single-level data: time, latitude, longitude
        preferred_order = ['time', 'latitude', 'longitude']

    actual_dims = [dim for dim in preferred_order if dim in ds.dims]
    remaining_dims = [dim for dim in ds.dims if dim not in preferred_order]
    if actual_dims:
        ds = ds.transpose(*actual_dims, *remaining_dims)

    # 7. Clean encoding to prevent write conflicts
    for var in list(ds.data_vars) + list(ds.coords):
        if var in ds.variables:
            # Remove problematic encoding entries
            for key in ['coordinates', 'contiguous', 'chunksizes']:
                ds[var].encoding.pop(key, None)

    return ds

def longitude_0_360_to_180_180(ds: xr.Dataset) -> xr.Dataset:
    """
    Convert longitude coordinates from [0,360) to [-180,180) and sort.

    Parameters:
        ds: Dataset with longitude coordinates

    Returns:
        Dataset with wrapped and sorted longitude coordinates
    """
    if 'longitude' not in ds.coords:
        return ds

    # Convert longitude coordinates
    ds = ds.assign_coords(longitude=(ds.longitude + 180) % 360 - 180)

    # Sort by longitude (ERA5 has 1D rectilinear coordinates)
    return ds.sortby('longitude')

# ================================= Logging Setup ==============================

def setup_dual_logger(name: str, log_file: Path, console_level: int = logging.INFO) -> logging.Logger:
    """
    Setup logger with both file (DEBUG) and console (INFO) handlers.

    Parameters:
        name: Logger name
        log_file: Path to log file
        console_level: Console logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler - detailed logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler - user-friendly output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger

# ================================= Base Downloader Class ======================

class ERA5Downloader:
    """
    Base class for robust ERA5 data downloading with modern architecture.

    This class provides shared functionality for both single-level and multi-level
    ERA5 downloads.

    Features:
    - Type-safe task management with dataclasses
    - Robust file validation and archive handling
    - Multi-engine xarray support with graceful fallbacks
    - Comprehensive error classification and retry logic
    - CF-compliant preprocessing for ERA5 data
    - Global retry orchestration for maximum robustness
    """

    def __init__(self, config: DownloadConfig):
        """
        Initialize downloader with configuration.

        Parameters:
            config: Download configuration settings
        """
        self.config = config
        self.file_validator = FileValidator(config.min_file_size_bytes)

        # Setup logging with descriptive filename
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        num_vars = len(config.variables)
        date_range = f"{config.start_year}{config.start_month:02d}-{config.end_year}{config.end_month:02d}"

        # Create descriptive log filename
        if num_vars == 1:
            # Single variable: use variable name for clarity
            log_file = Path(config.logs_dir) / f'download_{config.variables[0]}_{date_range}_{timestamp}.log'
        else:
            # Multiple variables: use generic name with count
            log_file = Path(config.logs_dir) / f'download_{num_vars}vars_{date_range}_{timestamp}.log'

        self.logger = setup_dual_logger(f"era5_downloader_{id(self)}", log_file)

        # Setup archive handler
        self.archive_handler = ArchiveHandler(self.logger)

        # Initialize CDS client
        self.client = cdsapi.Client()

        # Track download statistics
        self.stats = {
            'total_tasks': 0,
            'successful_downloads': 0,
            'failed_downloads': 0,
            'retries_by_error': defaultdict(int),
            'download_times': [],
        }

    def log_configuration(self):
        """Log comprehensive configuration summary."""
        self.logger.info('[MAIN] ===== ERA5 DOWNLOAD CONFIGURATION =====')
        self.logger.info(f'[MAIN] Date range: {self.config.start_year}-{self.config.end_year} '
                        f'({self.config.end_year - self.config.start_year + 1} years)')
        self.logger.info(f'[MAIN] Variables: {self.config.variables}')
        self.logger.info(f'[MAIN] Verification enabled: {self.config.verify_before_concat}')
        self.logger.info(f'[MAIN] Allow partial data: {self.config.allow_partial_concat}')
        self.logger.info(f'[MAIN] Max retries per year: {self.config.max_download_retries}')
        self.logger.info(f'[MAIN] Max global retries: {self.config.max_global_retries}')

        if self.config.grid:
            self.logger.info(f'[MAIN] Grid resolution: {self.config.grid[0]}° × {self.config.grid[1]}°')
        if self.config.area:
            self.logger.info(f'[MAIN] Spatial bounds: N={self.config.area[0]}, W={self.config.area[1]}, '
                           f'S={self.config.area[2]}, E={self.config.area[3]}')

        self.logger.info('[MAIN] ==========================================')

    def check_existing_files(self, variable: str) -> Tuple[List[int], List[int]]:
        """
        Check which years have been successfully downloaded for a variable.

        Parameters:
            variable: Variable name to check

        Returns:
            Tuple of (existing_years, missing_years)
        """
        existing_years = []
        missing_years = []

        input_dir = self._get_input_dir(variable)

        self.logger.info(f'[VERIFY] Checking for files in: {input_dir}')
        self.logger.info(f'[VERIFY] Looking for pattern: download_*_{variable}.nc')

        for year in range(self.config.start_year, self.config.end_year + 1):
            filepath = Path(input_dir) / f'download_{year}_{variable}.nc'

            if filepath.exists():
                file_size = filepath.stat().st_size
                self.logger.debug(f'[VERIFY] Found file: {filepath} ({file_size} bytes)')

                if file_size >= self.config.min_file_size_bytes:
                    # Get expected variable name from mapping
                    expected_var = self.config.variable_mappings.get(variable, variable)

                    if self.file_validator.validate_content(filepath, expected_var):
                        existing_years.append(year)
                        self.logger.info(f'[VERIFY] Valid file found: {year} ({file_size} bytes)')
                    else:
                        missing_years.append(year)
                        self.logger.warning(f'[VERIFY] Invalid dataset structure: {filepath}')
                else:
                    missing_years.append(year)
                    self.logger.warning(f'[VERIFY] File too small: {filepath} '
                                      f'({file_size} bytes, min required: {self.config.min_file_size_bytes})')
            else:
                missing_years.append(year)
                self.logger.debug(f'[VERIFY] Missing file: {filepath}')

        self.logger.info(f'[VERIFY] Summary: Found {len(existing_years)} existing, {len(missing_years)} missing years')
        if existing_years:
            sample_years = sorted(existing_years)[:10]
            self.logger.info(f'[VERIFY] Existing years: {sample_years}{"..." if len(existing_years) > 10 else ""}')
        if missing_years:
            self.logger.info(f'[VERIFY] Missing years: {sorted(missing_years)}')

        return existing_years, missing_years

    def build_tasks(self, variable: str, missing_years: List[int]) -> List[DownloadTask]:
        """
        Build download tasks for missing years.

        Parameters:
            variable: Variable to download
            missing_years: Years that need to be downloaded

        Returns:
            List of DownloadTask objects
        """
        tasks = []
        input_dir = self._get_input_dir(variable)

        for year in missing_years:
            months = months_for_year(
                year,
                self.config.start_year,
                self.config.start_month,
                self.config.end_year,
                self.config.end_month
            )

            if not months:
                continue

            output_path = os.path.join(input_dir, f'download_{year}_{variable}.nc')

            task = DownloadTask(
                year=year,
                variable=variable,
                months=months,
                dataset=self.get_dataset_name(),
                output_path=output_path,
                grid=self.config.grid,
                area=self.config.area,
                pressure_levels=self.config.pressure_levels,
            )

            task.validate()
            tasks.append(task)

        return tasks

    def download_single_task(self, task: DownloadTask) -> bool:
        """
        Download a single task with error handling and validation.

        Parameters:
            task: DownloadTask to execute

        Returns:
            True if successful, False if failed
        """
        start_time = time.time()

        try:
            request = task.build_request()
            tmp_path = Path(task.output_path + '.part')
            final_path = Path(task.output_path)

            # Clean up any existing partial downloads
            if tmp_path.exists():
                tmp_path.unlink()

            retry_msg = f" (retry {task.retry_count})" if task.retry_count > 0 else ""
            self.logger.info(f'[DOWNLOAD] Starting: {task.year} {task.variable}{retry_msg}')

            # Perform the download
            self.client.retrieve(task.dataset, request, str(tmp_path))

            # Process the downloaded file (handle archives, apply preprocessing)
            processed_path = self.archive_handler.process_download(tmp_path, final_path)

            # Validate the final result
            expected_var = self.config.variable_mappings.get(task.variable, task.variable)
            if self.file_validator.is_valid_download(processed_path, expected_var):
                download_time = time.time() - start_time
                self.stats['download_times'].append(download_time)
                self.stats['successful_downloads'] += 1
                self.logger.info(f'[DOWNLOAD] Complete: {processed_path} ({download_time:.1f}s)')
                return True
            else:
                self.stats['failed_downloads'] += 1
                self.logger.error(f'[DOWNLOAD] Failed validation: {processed_path}')
                return False

        except Exception as e:
            error_type, should_retry = classify_download_error(e)
            self.stats['retries_by_error'][error_type] += 1
            self.stats['failed_downloads'] += 1

            self.logger.error(f'[DOWNLOAD] Failed {task.year}-{task.variable}{retry_msg}: {error_type} - {e}')
            return False

    def download_tasks_parallel(self, tasks: List[DownloadTask], max_workers: int) -> List[DownloadTask]:
        """
        Download multiple tasks in parallel with retry logic.

        Parameters:
            tasks: List of DownloadTask objects
            max_workers: Maximum number of concurrent downloads

        Returns:
            List of failed tasks
        """
        if not tasks:
            return []

        self.stats['total_tasks'] += len(tasks)
        failures = []

        for retry_attempt in range(self.config.max_download_retries):
            if not tasks:
                break

            self.logger.info(f'[RETRY] Attempt {retry_attempt + 1}/{self.config.max_download_retries} '
                           f'for {len(tasks)} tasks')

            successful_tasks = []

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Update retry count for each task
                for task in tasks:
                    task.retry_count = retry_attempt

                future_to_task = {
                    executor.submit(self.download_single_task, task): task
                    for task in tasks
                }

                completed = 0
                for future in as_completed(future_to_task):
                    task = future_to_task[future]
                    try:
                        success = future.result()
                        if success:
                            successful_tasks.append(task)
                            self.logger.info(f'[RETRY] Success: {task.year}')
                    except Exception as e:
                        self.logger.error(f'[RETRY] Exception for {task.year}: {e}')

                    completed += 1
                    if completed % 5 == 0 or completed == len(tasks):
                        self.logger.info(f'[RETRY] Progress: {completed}/{len(tasks)} completed')

            # Update tasks list to only include failures
            tasks = [task for task in tasks if task not in successful_tasks]

            if tasks and retry_attempt < self.config.max_download_retries - 1:
                # Apply exponential backoff
                if self.config.exponential_backoff:
                    delay = self.config.retry_delay_seconds * (2 ** retry_attempt)
                else:
                    delay = self.config.retry_delay_seconds

                self.logger.info(f'[RETRY] Waiting {delay}s before next attempt. Still missing: '
                               f'{[t.year for t in tasks]}')
                time.sleep(delay)

        failures = tasks
        if failures:
            failed_years = [t.year for t in failures]
            self.logger.error(f'[RETRY] Failed to download after all attempts: {failed_years}')

        return failures

    def process_variable_with_global_retry(self, variable: str) -> bool:
        """
        Process a single variable with global retry orchestration.

        Parameters:
            variable: Variable name to process

        Returns:
            True if all downloads succeeded, False otherwise
        """
        self.logger.info(f'[MAIN] === Processing {variable} ===')

        # Phase 1: Initial download attempt (only missing years)
        if self.config.verify_before_concat:
            existing_years, missing_years = self.check_existing_files(variable)

            if missing_years:
                self.logger.info(f'[MAIN] Initial verification found {len(existing_years)} existing '
                               f'and {len(missing_years)} missing years')
                self.logger.info(f'[MAIN] Missing years: {missing_years}')

                initial_tasks = self.build_tasks(variable, missing_years)
                self.download_tasks_parallel(initial_tasks, self.config.max_workers)
            else:
                self.logger.info(f'[MAIN] All {len(existing_years)} years already present, '
                               'skipping initial download')
        else:
            # Legacy mode: download everything (not recommended)
            self.logger.warning('[MAIN] VERIFY_BEFORE_CONCAT is disabled - downloading all years')
            all_years = list(range(self.config.start_year, self.config.end_year + 1))
            initial_tasks = self.build_tasks(variable, all_years)
            self.download_tasks_parallel(initial_tasks, self.config.max_workers)

        # Phase 2: Global retry loop for maximum robustness
        if self.config.verify_before_concat:
            global_retry = 0
            while global_retry < self.config.max_global_retries:
                existing_years, missing_years = self.check_existing_files(variable)

                if not missing_years:
                    self.logger.info(f'[MAIN] All years present for {variable}')
                    break

                self.logger.warning(f'[MAIN] Global retry {global_retry + 1}/{self.config.max_global_retries}. '
                                  f'Missing years: {missing_years}')

                retry_tasks = self.build_tasks(variable, missing_years)
                still_missing_tasks = self.download_tasks_parallel(retry_tasks, self.config.retry_workers)

                if not still_missing_tasks:
                    break

                global_retry += 1
                if global_retry < self.config.max_global_retries:
                    delay = self.config.retry_delay_seconds * global_retry
                    self.logger.info(f'[MAIN] Waiting {delay}s before global retry')
                    time.sleep(delay)

        # Phase 3: Final verification
        if self.config.verify_before_concat:
            existing_years, missing_years = self.check_existing_files(variable)

            if missing_years:
                error_msg = f'Cannot proceed with concatenation. Missing years: {missing_years}'
                if self.config.allow_partial_concat:
                    self.logger.warning(f'[MAIN] {error_msg}. Proceeding with partial data.')
                    self._create_missing_years_report(variable, missing_years)
                    return False  # Partial success
                else:
                    self.logger.error(f'[MAIN] {error_msg}. Aborting.')
                    return False
            else:
                self.logger.info(f'[MAIN] All {len(existing_years)} years verified for {variable}')
                return True

        return True

    def concatenate_variable(self, variable: str) -> bool:
        """
        Concatenate yearly files for a variable into a single dataset.

        Parameters:
            variable: Variable name to concatenate

        Returns:
            True if concatenation succeeded, False otherwise
        """
        input_dir = self._get_input_dir(variable)
        pattern = os.path.join(input_dir, f'download_*_{variable}.nc')
        files = sorted(glob.glob(pattern))

        if not files:
            self.logger.error(f'[CONCAT] No files found for {variable} in {input_dir}')
            return False

        self.logger.info(f'[CONCAT] Opening {len(files)} files for {variable}')

        try:
            # Open with preprocessing
            ds = xr.open_mfdataset(
                files,
                combine='nested',
                concat_dim='time',
                parallel=True,
                preprocess=preprocess_era5_dataset,
                chunks={'time': 24}
            )

            self.logger.info(f'[CONCAT] Initial open_mfdataset complete, chunks={ds.chunks}')

            # Sort on time and rechunk to smaller blocks
            if 'time' in ds.coords:
                ds = ds.sortby('time')

            # Determine optimal chunking
            chunk_dict = {'time': 12, 'latitude': 'auto', 'longitude': 'auto'}
            if 'pressure_level' in ds.dims:
                chunk_dict['pressure_level'] = 'auto'

            ds = ds.chunk(chunk_dict)
            self.logger.debug(f'[CONCAT] After rechunk, chunks={ds.chunks}')

            # Apply longitude wrapping and spatial subsetting
            with dask.config.set(**{'array.slicing.split_large_chunks': True}):
                ds = longitude_0_360_to_180_180(ds)

            if 'latitude' in ds.coords:
                ds = ds.sortby('latitude', ascending=True)

            # Apply spatial subsetting if configured
            if hasattr(self, '_apply_spatial_subsetting'):
                ds = self._apply_spatial_subsetting(ds)

            # Write output
            output_path = self._get_output_path(variable)
            self.logger.info(f'[CONCAT] Writing output to {output_path}')

            ds.to_netcdf(
                output_path,
                engine='netcdf4',
                format='NETCDF4',
                unlimited_dims=('time',),
                compute=True
            )

            ds.close()

            # Verify concatenation succeeded
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
                self.logger.info(f'[CONCAT] Concatenation successful: {output_path} ({file_size:.1f} MB)')
                return True
            else:
                self.logger.error('[CONCAT] Concatenation failed: output file not created')
                return False

        except Exception as e:
            self.logger.error(f'[CONCAT] Concatenation failed for {variable}: {e}', exc_info=True)
            return False

    def cleanup_yearly_files(self, variable: str):
        """
        Remove per-year NetCDF files after successful concatenation.

        Parameters:
            variable: Variable name to clean up
        """
        input_dir = self._get_input_dir(variable)
        self.logger.info(f'[DELETE] Deleting input files for {variable}')

        deleted = 0
        for year in range(self.config.start_year, self.config.end_year + 1):
            filepath = os.path.join(input_dir, f'download_{year}_{variable}.nc')
            if os.path.exists(filepath):
                os.remove(filepath)
                deleted += 1
                self.logger.debug(f'[DELETE] Deleted {filepath}')
            else:
                self.logger.warning(f'[DELETE] Not found for deletion: {filepath}')

        self.logger.info(f'[DELETE] Deleted {deleted} files for {variable}')

    def process_all_variables(self) -> Dict[str, bool]:
        """
        Process all configured variables with download and concatenation.

        Returns:
            Dictionary mapping variable names to success status
        """
        start_time = time.time()
        self.log_configuration()

        results = {}

        for variable in self.config.variables:
            # Download phase
            download_success = self.process_variable_with_global_retry(variable)

            # Concatenation phase
            if download_success or self.config.allow_partial_concat:
                try:
                    self.logger.info(f'[MAIN] Starting concatenation for {variable}')
                    concat_success = self.concatenate_variable(variable)

                    if concat_success:
                        # Only delete input files if concatenation succeeded
                        self.cleanup_yearly_files(variable)
                        results[variable] = True
                    else:
                        results[variable] = False

                except Exception as e:
                    self.logger.error(f'[MAIN] Concatenation failed for {variable}: {e}', exc_info=True)
                    results[variable] = False
            else:
                results[variable] = False

            self.logger.info(f'[MAIN] === Finished {variable} ===')

        # Final summary
        elapsed_time = time.time() - start_time
        self.logger.info(f'[MAIN] All processing complete. Total time: {elapsed_time:.1f}s')

        # Log statistics
        self._log_final_statistics()

        return results

    def _get_input_dir(self, variable: str) -> str:
        """Get input directory for a variable (to be overridden by subclasses)."""
        return os.path.join(self.config.base_dir, variable.replace('_', '-'))

    def _get_output_path(self, variable: str) -> str:
        """Get output file path for a variable."""
        return os.path.join(self.config.base_dir, f'concatenated_{variable}.nc')

    def _create_missing_years_report(self, variable: str, missing_years: List[int]):
        """Create a report of missing years for partial concatenation."""
        report_path = os.path.join(self.config.logs_dir, f'missing_years_report_{variable}.txt')
        with open(report_path, 'w') as f:
            f.write(f'Missing years for {variable}:\n')
            f.write('\n'.join(map(str, missing_years)))
        self.logger.info(f'[MAIN] Missing years report saved to: {report_path}')

    def _log_final_statistics(self):
        """Log final download statistics."""
        self.logger.info('[STATS] ===== DOWNLOAD STATISTICS =====')
        self.logger.info(f'[STATS] Total tasks: {self.stats["total_tasks"]}')
        self.logger.info(f'[STATS] Successful downloads: {self.stats["successful_downloads"]}')
        self.logger.info(f'[STATS] Failed downloads: {self.stats["failed_downloads"]}')

        if self.stats['download_times']:
            avg_time = sum(self.stats['download_times']) / len(self.stats['download_times'])
            self.logger.info(f'[STATS] Average download time: {avg_time:.1f}s')

        if self.stats['retries_by_error']:
            self.logger.info('[STATS] Retries by error type:')
            for error_type, count in self.stats['retries_by_error'].items():
                self.logger.info(f'[STATS]   {error_type}: {count}')

        self.logger.info('[STATS] ================================')

    def get_dataset_name(self) -> str:
        """Get CDS dataset name (to be overridden by subclasses)."""
        raise NotImplementedError("Subclasses must implement get_dataset_name()")

# ================================= Specialized Subclasses ====================

class ERA5SingleLevelDownloader(ERA5Downloader):
    """Downloader for ERA5 single-level variables."""

    def get_dataset_name(self) -> str:
        return "reanalysis-era5-single-levels-monthly-means"

class ERA5MultiLevelDownloader(ERA5Downloader):
    """Downloader for ERA5 pressure-level variables."""

    def get_dataset_name(self) -> str:
        return "reanalysis-era5-pressure-levels-monthly-means"
