"""
Shared utilities for composite analysis scripts.
Contains functions common to both horizontal and vertical composite analysis.
"""

import sys
import os
import logging
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any
from scipy.stats import ttest_ind, ttest_1samp
import psutil
import csv
import warnings

try:
    # Try standard relative import first (when running from src or its subdirectories)
    from .TBI_functions import (
        load_era_field,
        compute_seasonal_anomalies
    )
except ImportError:
    try:
        # Try without relative import (when utils is in sys.path)
        from utils.TBI_functions import (
            load_era_field,
            compute_seasonal_anomalies
        )
    except ImportError:
        try:
            # Fallback for when script is run from parent directory
            from utils.TBI_functions import (
                load_era_field,
                compute_seasonal_anomalies
            )
        except ImportError as e:
            print(f"Warning: Could not import TBI_functions, some functionality may be limited: {e}")
            # Create placeholder functions to prevent script failure
            def load_era_field(*args, **kwargs):
                raise RuntimeError("TBI_functions not available - missing dependencies")
            def compute_seasonal_anomalies(*args, **kwargs):
                raise RuntimeError("TBI_functions not available - missing dependencies")

# Shared constants (identical in both scripts)
SEASON_TO_MIDDLE_MONTH: Dict[str, int] = {'DJF': 1, 'MAM': 4, 'JJA': 7, 'SON': 10}

PRECURSOR_TO_DEFINITION_SEASON_KEY_MAP: Dict[str, str] = {
    'REOF SST JJA': 'JJA_-1', 'DMI JJA': 'JJA_-1', 'PSA2 JJA': 'JJA_-1',
    'MCA WAF-RWS SON': 'SON_-1', 'MCA prec-RWS SON': 'SON_-1', 'SIOD MAM': 'SON_-1',
    'MCA RWS-WAF DJF': 'DJF_0', 'MCA RWS-prec DJF': 'DJF_0',
    'WNP DJF': 'DJF_0', 'SASDI SON': 'DJF_0', 'Atl3 DJF': 'DJF_0', 'NPMM-SST DJF': 'DJF_0',
    'MCA RWS-prec MAM(E)': 'MAM_0', 'MCA RWS-prec MAM(C)': 'MAM_0', 'NTA MAM': 'MAM_0',
    'SPO MAM': 'MAM_0', 'NPMM-wind MAM': 'MAM_0', 'SPMM-SST MAM': 'MAM_0', 'SPMM-wind MAM': 'MAM_0',
}

PRECURSOR_ENSO_SIGN_MAP: Dict[str, int] = {
    'REOF SST JJA': 1,
    'MCA WAF-RWS SON': 1,
    'MCA RWS-WAF DJF': -1,
    'MCA RWS-prec MAM(E)': 1,
    'MCA prec-RWS SON': 1,
    'MCA RWS-prec DJF': -1,
    'MCA RWS-prec MAM(C)': 1,
}

STANDARD_ENSO_CYCLE_PLOT_SEQUENCE: List[Tuple[str, int, str]] = [
    ('JJA', -1, 'JJA(-1)'),
    ('SON', -1, 'SON(-1)'),
    ('DJF', -1, 'DJF(0)'),
    ('MAM',  0, 'MAM(0)'),
    ('JJA',  0, 'JJA(0)'),
    ('SON',  0, 'SON(0)'),
    ('DJF',  0, 'DJF(+1)')
]

# Mapping from precursor definition season keys to plot sequence start indices
PRECURSOR_DEFINITION_TO_PLOT_START_INDEX: Dict[str, int] = {
    'JJA_-1': 0,  # Start from JJA(-1) - index 0
    'SON_-1': 1,  # Start from SON(-1) - index 1  
    'DJF_0': 2,   # Start from DJF(0) - index 2
    'MAM_0': 3,   # Start from MAM(0) - index 3
    'JJA_0': 4,   # Start from JJA(0) - index 4
    'SON_0': 5,   # Start from SON(0) - index 5
    'DJF_1': 6    # Start from DJF(+1) - index 6
}

# --- IDENTICAL FUNCTIONS (10) ---

def setup_logging(save_dir: Path) -> None:
    """Sets up logging to file and console."""
    save_dir.mkdir(parents=True, exist_ok=True)
    log_file = save_dir / 'composite_analysis.log'
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Logging setup complete for composite_analysis.py.")

def load_all_precursor_timeseries(filepath: Path) -> Optional[xr.Dataset]:
    """Loads and returns the full precursor dataset."""
    logging.info(f"Loading precursor time series from: {filepath}")
    if not filepath.exists():
        logging.error(f"FATAL: Precursor time series file not found: {filepath}")
        return None
    try:
        ds = xr.open_dataset(filepath)
        logging.info(f"Successfully loaded precursors. Variables: {list(ds.data_vars)}")
        return ds
    except Exception as e:
        logging.error(f"FATAL: Could not load precursor file {filepath}: {e}", exc_info=True)
        return None

def get_active_season_precursor_series(precursors_ds: xr.Dataset, precursor_name: str) -> xr.DataArray:
    """Extracts a single precursor series, retaining only its active seasonal values."""
    if precursor_name not in precursors_ds:
        logging.warning(f"Precursor '{precursor_name}' not found in dataset. Returning empty DataArray.")
        return xr.DataArray(np.array([]), dims=['time'], name=precursor_name, coords={'time': []})
    ts = precursors_ds[precursor_name].dropna(dim='time')
    return ts

def select_extreme_years_for_precursor(
    active_ts: xr.DataArray,
    precursor_name: str,
    n_years: int,
    precursor_season_map: Dict[str, str],
    season_month_map: Dict[str, int]
) -> Tuple[List[int], List[int]]:
    """
    Select top/bottom N years based on precursor's defining season only.
    
    Args:
        active_ts: Time series data for the precursor
        precursor_name: Name of the precursor
        n_years: Number of extreme years to select
        precursor_season_map: Maps precursor names to their season keys
        season_month_map: Maps season codes to their middle month
    
    Returns:
        Tuple of (top_years, bottom_years) as lists of years
    """
    # Get the defining season for this precursor
    season_key = precursor_season_map.get(precursor_name)
    if not season_key:
        logging.error(f"No season key found for precursor {precursor_name}")
        return [], []
    
    # Extract season code (e.g., 'SON' from 'SON_-1')
    season_code = season_key.split('_')[0]
    target_month = season_month_map.get(season_code)
    
    if target_month is None:
        logging.error(f"No month mapping found for season {season_code}")
        return [], []
    
    # Filter time series to only include the target month
    filtered_ts = active_ts.sel(time=active_ts.time.dt.month == target_month)
    
    if filtered_ts.time.size < n_years:
        logging.warning(f"Not enough data points ({filtered_ts.time.size}) for {n_years} years in {precursor_name}")
        n_available = filtered_ts.time.size
        n_to_select = min(n_years, n_available)
    else:
        n_to_select = n_years
    
    # Sort the filtered time series
    sorted_ts = filtered_ts.sortby(filtered_ts)
    
    # Extract years for top and bottom values
    top_years = sorted_ts.isel(time=slice(-n_to_select, None)).time.dt.year.values.tolist()
    bottom_years = sorted_ts.isel(time=slice(0, n_to_select)).time.dt.year.values.tolist()
    
    # Ensure uniqueness (should already be unique due to monthly filtering)
    top_years = sorted(list(set(top_years)))
    bottom_years = sorted(list(set(bottom_years)))
    
    logging.info(f"  Selected {len(top_years)} top years and {len(bottom_years)} bottom years for {precursor_name} based on {season_code} values")
    
    return top_years, bottom_years

def select_target_data_for_compositing(
    spatial_anomaly_da: xr.DataArray,
    precursor_event_years: List[int],
    target_season_code: str,
    year_offset_from_precursor_event_year: int,
    season_month_map: Optional[Dict[str, int]] = None
) -> Tuple[Optional[xr.DataArray], List[int]]:
    """
    Selects data from a spatial anomaly DataArray for specific years and a target season.
    """
    if spatial_anomaly_da is None or not precursor_event_years:
        return None, []

    # Use provided season_month_map or fall back to global constant
    if season_month_map is None:
        season_month_map = SEASON_TO_MIDDLE_MONTH
    
    target_month = season_month_map.get(target_season_code)
    if target_month is None:
        logging.error(f"Invalid target_season_code: {target_season_code}")
        return None, []

    target_data_calendar_years = []
    for p_year in precursor_event_years:
        effective_season_year = p_year + year_offset_from_precursor_event_year
        data_stamp_year = effective_season_year + 1 if target_season_code == 'DJF' else effective_season_year
        target_data_calendar_years.append(data_stamp_year)
    
    target_data_calendar_years = sorted(list(set(target_data_calendar_years)))

    selected_slices = spatial_anomaly_da.sel(
        time=(spatial_anomaly_da.time.dt.month == target_month) & 
             (spatial_anomaly_da.time.dt.year.isin(target_data_calendar_years))
    )

    if selected_slices.time.size == 0:
        return None, []

    contributing_precursor_years = []
    data_years_in_composite = selected_slices.time.dt.year.values
    for p_year in precursor_event_years:
        effective_season_year_for_p_year = p_year + year_offset_from_precursor_event_year
        data_stamp_year_for_p_year = effective_season_year_for_p_year + 1 if target_season_code == 'DJF' else effective_season_year_for_p_year
        if data_stamp_year_for_p_year in data_years_in_composite:
            contributing_precursor_years.append(p_year)
    
    return selected_slices, sorted(list(set(contributing_precursor_years)))

def get_precursor_ref_calendar_year_and_season(y_event_ref: int, precursor_def_key: str) -> Tuple[int, str]:
    """
    Determines the calendar year and season code for a precursor's data
    based on the ENSO event reference year and the precursor's definition key.
    """
    season_code, year_tag = precursor_def_key.split('_')
    if year_tag == '-1':
        calendar_year = y_event_ref - 1
    elif year_tag == '0':
        calendar_year = y_event_ref
    else:
        logging.error(f"Unknown year_tag in precursor_def_key: {precursor_def_key}")
        raise ValueError(f"Unknown year_tag in precursor_def_key: {precursor_def_key}")
    return calendar_year, season_code

def calculate_fieldwise_ttest_significance(
    data_slices_group1: xr.DataArray,
    data_slices_group2: Optional[xr.DataArray] = None,
    popmean: float = 0.0,
    alpha: float = 0.05,
    return_pvals: bool = False
) -> Optional[xr.DataArray]:
    """Performs field-wise t-test. Returns boolean mask of significance or p-values."""
    if data_slices_group1 is None or data_slices_group1.time.size < 2:
        return None

    try:
        if data_slices_group2 is not None:
            if data_slices_group2.time.size < 2:
                return None
            data1_np = data_slices_group1.transpose('time', ...).data
            data2_np = data_slices_group2.transpose('time', ...).data
            t_stat, p_value = ttest_ind(data1_np, data2_np, axis=0, nan_policy='omit', equal_var=False)
        else:
            data1_np = data_slices_group1.transpose('time', ...).data
            t_stat, p_value = ttest_1samp(data1_np, popmean=popmean, axis=0, nan_policy='omit')
        
        if return_pvals:
            result_data = p_value
        else:
            if np.all(np.isnan(p_value)):
                result_data = np.full(p_value.shape, False, dtype=bool)
            else:
                result_data = p_value < alpha
        
        # Dynamically determine coordinates and dimensions based on input data
        # (excluding 'time' dimension which was reduced by the t-test)
        coords = {}
        dims = []
        
        for dim in data_slices_group1.dims:
            if dim != 'time':
                coords[dim] = data_slices_group1[dim]
                dims.append(dim)
        
        result_da = xr.DataArray(
            result_data,
            coords=coords,
            dims=dims
        )
        return result_da
    except Exception as e:
        logging.error(f"Error during t-test: {e}", exc_info=True)
        return None

def get_processed_composite_winds(
    u_composite: xr.DataArray,
    v_composite: xr.DataArray,
    significance_mask: xr.DataArray,
    plot_type: str,
    special_scaling: Optional[str] = None,
    apply_significance_mask: bool = True,
    equatorial_mask_deg: Optional[float] = None
) -> Tuple[Optional[xr.DataArray], Optional[xr.DataArray], Optional[xr.DataArray]]:
    """Process wind composites for plotting."""
    if u_composite is None or v_composite is None or significance_mask is None:
        return None, None, None

    if apply_significance_mask:
        u_masked = u_composite.where(significance_mask)
        v_masked = v_composite.where(significance_mask)
    else:
        u_masked = u_composite
        v_masked = v_composite

    # Apply equatorial masking for WAF vectors
    if equatorial_mask_deg is not None and special_scaling == 'waf':
        u_masked, v_masked = apply_equatorial_mask(
            u_masked, v_masked, 
            equator_mask_deg=equatorial_mask_deg
        )

    u_to_subsample = u_masked
    v_to_subsample = v_masked
    subsample_step = 5

    if special_scaling == 'waf':
        subsample_step = 7
        magnitude = np.sqrt(u_masked**2 + v_masked**2)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            m05 = np.nanquantile(magnitude.data, 0.05) if not np.all(np.isnan(magnitude.data)) else 0
            m95 = np.nanquantile(magnitude.data, 0.95) if not np.all(np.isnan(magnitude.data)) else 1
        
        min_len, max_len = 0.6, 1.4
        if m95 > m05:
            slope = (max_len - min_len) / (m95 - m05)
            intercept = max_len - slope * m95
            magnitude_clipped = xr.where(magnitude < m05, m05, xr.where(magnitude > m95, m95, magnitude))
            new_magn = magnitude_clipped * slope + intercept
            ratio = xr.where(magnitude != 0, new_magn / magnitude, 0).fillna(0.)
            u_to_subsample = u_masked * ratio
            v_to_subsample = v_masked * ratio

    u_final = u_to_subsample[::subsample_step, ::subsample_step]
    v_final = v_to_subsample[::subsample_step, ::subsample_step]
    
    linewidth_da = None
    if plot_type == 'streamline':
        stream_speed = np.sqrt(u_masked**2 + v_masked**2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            max_s = np.nanmax(stream_speed.data) if not np.all(np.isnan(stream_speed.data)) else 0
        
        if max_s and max_s > 0:
            linewidth_da = (3 * stream_speed / max_s).fillna(0.)
        else:
            linewidth_da = xr.full_like(stream_speed, 0.0 if not np.all(np.isnan(stream_speed.data)) else np.nan)
        linewidth_da = linewidth_da[::subsample_step, ::subsample_step]

    return u_final, v_final, linewidth_da

def save_contributing_years_csv(
    collected_seasonal_data: Optional[List[Dict]],
    composite_type_tag: str,
    output_dir: Path
) -> None:
    """Saves a CSV file listing the contributing years for high and low composites."""
    if not collected_seasonal_data:
        return

    csv_rows = [('season_lag', 'high_years', 'low_years')]
    for sdata in collected_seasonal_data:
        high_list = sdata.get('actual_high_contrib_years_primary', [])
        low_list = sdata.get('actual_low_contrib_years_primary', [])
        csv_rows.append((
            sdata['display_suffix'],
            ';'.join(map(str, sorted(list(set(high_list))))),
            ';'.join(map(str, sorted(list(set(low_list)))))
        ))

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_fname = f"contributing_years_{composite_type_tag}.csv"
    csv_path = output_dir / csv_fname

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    logging.info(f"  Saved CSV with contributing years: {csv_path}")

def available_workers():
    """Determine a safe default number of worker processes."""
    max_by_mem = (psutil.virtual_memory().total // (2**30) - 4) // 1
    max_by_cpu = max(1, os.cpu_count() // 2)
    return max(1, min(8, max_by_mem, max_by_cpu))

def apply_equatorial_mask(u_data: xr.DataArray, v_data: xr.DataArray, 
                         equator_mask_deg: float = 10.0) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Apply hard mask to WAF vectors within specified degrees of equator.
    
    Parameters:
    - u_data: Zonal component of vectors
    - v_data: Meridional component of vectors
    - equator_mask_deg: Mask vectors within ±equator_mask_deg of equator (default 10°)
    
    Returns:
    - Tuple of masked (u, v) components with NaN within equatorial band
    """
    # Create latitude mask: True where |latitude| > equator_mask_deg
    lat_mask = np.abs(u_data.latitude) > equator_mask_deg
    
    # Apply mask to both components
    u_masked = u_data.where(lat_mask)
    v_masked = v_data.where(lat_mask)
    
    return u_masked, v_masked

def get_target_season_specs_for_precursor(
    precursor_name: str, 
    full_sequence: List[Tuple[str, int, str]] = STANDARD_ENSO_CYCLE_PLOT_SEQUENCE,
    seasons_filter: Optional[List[str]] = None
) -> List[Tuple[str, int, str]]:
    """
    Get target season specifications based on precursor timing or explicit filter.
    
    This function harmonizes season selection between composite and regression scripts
    by allowing dynamic adjustment of the starting season based on precursor definition.
    
    Args:
        precursor_name: Name of the precursor (empty string disables precursor-based selection)
        full_sequence: Full season sequence to filter from (default: STANDARD_ENSO_CYCLE_PLOT_SEQUENCE)
        seasons_filter: Explicit season display names to include (overrides precursor-based selection)
                       Example: ['DJF(0)', 'MAM(0)', 'JJA(0)']
    
    Returns:
        Filtered season specification list as [(season_code, year_offset, display_suffix), ...]
    
    Examples:
        # Precursor-based selection
        >>> get_target_season_specs_for_precursor('MCA RWS-WAF DJF')
        [('DJF', -1, 'DJF(0)'), ('MAM', 0, 'MAM(0)'), ...]  # Starting from DJF(0)
        
        # Explicit season filtering
        >>> get_target_season_specs_for_precursor('', seasons_filter=['DJF(0)', 'MAM(0)'])
        [('DJF', -1, 'DJF(0)'), ('MAM', 0, 'MAM(0)')]
        
        # Default behavior (no filtering)
        >>> get_target_season_specs_for_precursor('')
        [('JJA', -1, 'JJA(-1)'), ('SON', -1, 'SON(-1)'), ...]  # Full sequence
    """
    # Explicit season filtering takes precedence
    if seasons_filter:
        logging.info(f"Using explicit season filter: {seasons_filter}")
        filtered_specs = []
        for season_code, year_offset, display_suffix in full_sequence:
            if display_suffix in seasons_filter:
                filtered_specs.append((season_code, year_offset, display_suffix))
        return filtered_specs
    
    # Precursor-based season selection
    if precursor_name:
        precursor_def_key = PRECURSOR_TO_DEFINITION_SEASON_KEY_MAP.get(precursor_name)
        if precursor_def_key:
            start_index = PRECURSOR_DEFINITION_TO_PLOT_START_INDEX.get(precursor_def_key)
            if start_index is not None:
                logging.info(f"Starting season sequence from index {start_index} for precursor '{precursor_name}' (def: {precursor_def_key})")
                return full_sequence[start_index:]
            else:
                logging.warning(f"Unknown precursor definition season key: {precursor_def_key}. Using full sequence.")
        else:
            logging.warning(f"Precursor '{precursor_name}' not found in PRECURSOR_TO_DEFINITION_SEASON_KEY_MAP. Using full sequence.")
    
    # Default: return full sequence
    logging.info("Using full season sequence (no precursor-based filtering)")
    return full_sequence

# --- UNIFIED FUNCTIONS (2) ---

def get_spatial_anomalies_field(
    field_short_name: str,
    config: Dict[str, Any],
    clim_start_date_str: str,
    clim_end_date_str: str,
    analysis_time_slice: slice
) -> Optional[xr.DataArray]:
    """
    Loads raw spatial field, computes seasonal anomalies, and returns it as a Dask array.
    Handles both horizontal (2D/3D) and vertical section (4D) data.
    """
    logging.info(f"  Preparing spatial field (anomalies): {field_short_name}")
    raw_data_path = config['raw_data_path']
    if not raw_data_path.exists():
        logging.error(f"Raw data file not found for {field_short_name}: {raw_data_path}")
        return None
    
    try:
        # Check if this is a vertical section field
        is_vertical = config.get('is_vertical_section', False)
        
        raw_field = load_era_field(
            filepath=str(raw_data_path),
            var_name=config['var_name_in_file'],
            scale=config.get('scale', None), 
            lat_slice=config.get('lat_slice', None),
        )
        if raw_field is None:
            logging.error(f"Failed to load raw data for {field_short_name}")
            return None
        
        # For vertical sections, ensure pressure dimension is preserved
        if is_vertical:
            # Check for pressure dimension
            if 'pressure_level' not in raw_field.dims:
                logging.error(f"Vertical section field {field_short_name} missing pressure_level dimension")
                return None
            logging.info(f"  Loaded 4D vertical section data with shape: {raw_field.shape}")
            logging.info(f"  Pressure levels: {raw_field.pressure_level.values}")

        anomalies = compute_seasonal_anomalies(raw_field, clim_start_date_str, clim_end_date_str)
        del raw_field
        
        if anomalies is None:
            logging.error(f"Failed to compute anomalies for {field_short_name}")
            return None

        anomalies = anomalies.sel(time=analysis_time_slice)
        if anomalies.time.size == 0:
            logging.warning(f"No data for {field_short_name} within analysis slice")
            return None
        
        anomalies = anomalies.rename(field_short_name + '_anom')
        anomalies.attrs['units'] = config.get('units', 'Unknown')
        anomalies.attrs['is_vertical_section'] = is_vertical
        return anomalies
    except Exception as e:
        logging.error(f"Error processing spatial field {field_short_name}: {e}", exc_info=True)
        return None

def save_consolidated_composites_netcdf(
    collected_seasonal_data: List[Dict],
    primary_field_name: str,
    primary_field_config: Dict[str, Any],
    target_spatial_fields_config_global: Dict[str, Dict[str, Any]],
    composite_type_tag: str,
    output_dir_nc: Path,
    alpha: float,
    clim_start_str: str,
    clim_end_str: str,
    analysis_slice: slice,
    lat_band: Optional[Tuple[float, float]] = None,
    show_all_winds: bool = False
) -> None:
    """
    Saves all composites to a single NetCDF file.
    Handles both horizontal and vertical section metadata.
    """
    if not collected_seasonal_data:
        logging.warning(f"No data to save to NetCDF for {primary_field_name}, {composite_type_tag}")
        return

    data_vars_for_ds = {}
    
    # Find reference composite for coordinates
    ref_comp = None
    for s_data in collected_seasonal_data:
        for case in ['high', 'low', 'diff']:
            ref_comp = s_data.get(f'comp_{case}_primary')
            if ref_comp is not None:
                break
        if ref_comp is not None:
            break
    
    if ref_comp is None:
        logging.warning(f"No valid composites found for {primary_field_name}")
        return

    season_lag_coord = pd.Index([s['display_suffix'] for s in collected_seasonal_data], name='season_lag')

    # Save primary field composites
    for case in ['high', 'low', 'diff']:
        comp_list = [s.get(f'comp_{case}_primary') for s in collected_seasonal_data]
        sig_list = [s.get(f'sig_{case}_primary') for s in collected_seasonal_data]

        filled_comp_list = [c if c is not None else xr.full_like(ref_comp, np.nan) for c in comp_list]
        filled_sig_list = [s if s is not None else xr.full_like(ref_comp, False, dtype=bool) for s in sig_list]

        if filled_comp_list:
            data_vars_for_ds[f'composite_{case}_{primary_field_name}'] = xr.concat(filled_comp_list, dim=season_lag_coord).assign_attrs(
                units=primary_field_config.get('units', ''),
                description=f"{case.capitalize()} composite for {primary_field_name}"
            )
        
        if primary_field_config.get('plot_significant_composite', True) and filled_sig_list:
            data_vars_for_ds[f'significance_{case}_{primary_field_name}'] = xr.concat(filled_sig_list, dim=season_lag_coord).assign_attrs(
                description=f"Significance (p<{alpha}) for {case} {primary_field_name} composite"
            )

    # Save wind overlay composites
    if collected_seasonal_data and collected_seasonal_data[0].get('wind_overlays_data'):
        for wind_idx, wind_overlay_template in enumerate(collected_seasonal_data[0]['wind_overlays_data']):
            u_key = wind_overlay_template['u_component_key']
            v_key = wind_overlay_template['v_component_key']
            u_config = target_spatial_fields_config_global[u_key]
            v_config = target_spatial_fields_config_global[v_key]

            for case in ['high', 'low', 'diff']:
                u_comp_list = []
                v_comp_list = []
                sig_mask_list = []
                
                for s in collected_seasonal_data:
                    if wind_idx < len(s.get('wind_overlays_data', [])):
                        u_comp_list.append(s['wind_overlays_data'][wind_idx].get(f'comp_{case}_u'))
                        v_comp_list.append(s['wind_overlays_data'][wind_idx].get(f'comp_{case}_v'))
                        sig_mask_list.append(s['wind_overlays_data'][wind_idx].get(f'sig_mask_{case}_wind'))
                    else:
                        u_comp_list.append(None)
                        v_comp_list.append(None)
                        sig_mask_list.append(None)
                
                ref_comp_wind_u = None
                for c_u in u_comp_list:
                    if c_u is not None:
                        ref_comp_wind_u = c_u
                        break
                if ref_comp_wind_u is None:
                    ref_comp_wind_u = xr.full_like(ref_comp, np.nan)

                ref_comp_wind_v = None
                for c_v in v_comp_list:
                    if c_v is not None:
                        ref_comp_wind_v = c_v
                        break
                if ref_comp_wind_v is None:
                    ref_comp_wind_v = xr.full_like(ref_comp, np.nan)

                filled_u_comps = [c if c is not None else xr.full_like(ref_comp_wind_u, np.nan) for c in u_comp_list]
                filled_v_comps = [c if c is not None else xr.full_like(ref_comp_wind_v, np.nan) for c in v_comp_list]
                filled_sig_masks = [sm if sm is not None else xr.full_like(ref_comp_wind_u, False, dtype=bool) for sm in sig_mask_list]

                if filled_u_comps:
                    data_vars_for_ds[f'composite_{case}_{u_key}'] = xr.concat(filled_u_comps, dim=season_lag_coord).assign_attrs(
                        units=u_config.get('units', ''),
                        description=f"{case.capitalize()} composite for {u_key}"
                    )
                if filled_v_comps:
                    data_vars_for_ds[f'composite_{case}_{v_key}'] = xr.concat(filled_v_comps, dim=season_lag_coord).assign_attrs(
                        units=v_config.get('units', ''),
                        description=f"{case.capitalize()} composite for {v_key}"
                    )
                
                if primary_field_config.get('plot_significant_composite', True) and filled_sig_masks:
                    data_vars_for_ds[f'significance_mask_{case}_wind_{u_key}_{v_key}'] = xr.concat(filled_sig_masks, dim=season_lag_coord).assign_attrs(
                        description=f"Significance mask (p_u<{alpha} OR p_v<{alpha}) for {case} {u_key}/{v_key} wind composite"
                    )
    
    if not data_vars_for_ds:
        logging.warning(f"No data variables to save in NetCDF for {primary_field_name}, {composite_type_tag}")
        return

    ds_to_save = xr.Dataset(data_vars_for_ds)
    ds_to_save.attrs['composite_type_tag'] = composite_type_tag
    ds_to_save.attrs['primary_target_field'] = primary_field_name
    ds_to_save.attrs['significance_alpha'] = alpha
    ds_to_save.attrs['show_all_winds'] = str(show_all_winds)
    
    # Add metadata for vertical sections
    is_vertical = primary_field_config.get('is_vertical_section', False)
    ds_to_save.attrs['is_vertical_section'] = str(is_vertical)
    if is_vertical:
        # Add vertical-specific metadata
        if lat_band:
            ds_to_save.attrs['equatorial_band'] = str(lat_band)
        else:
            ds_to_save.attrs['equatorial_band'] = str((-5, 5))  # Default for legacy compatibility

    output_dir_nc.mkdir(parents=True, exist_ok=True)
    nc_fname = f"composites_multiseason_{composite_type_tag}_{primary_field_name}{'_all_winds' if show_all_winds else ''}.nc"
    save_filepath = output_dir_nc / nc_fname
    ds_to_save.to_netcdf(save_filepath)
    logging.info(f"  Saved consolidated NetCDF: {save_filepath}")