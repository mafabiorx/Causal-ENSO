"""
Climate Data Processing Core: Shared Utility Functions for ENSO-ERA5 Analysis
=============================================================================

This module centralizes frequently used climate analysis functions across the
ENSO-ERA5 research workflow, providing standardized methods for data processing,
spatial operations, and statistical transformations essential for climate science
applications including ENSO diversity analysis and causal discovery.

Scientific Domain and Context
----------------------------
Climate data analysis requires specialized handling of geophysical coordinates,
seasonal cycles, and Earth geometry. This module implements climate science
best practices for:

- **Temporal Operations**: Seasonal standardization and time coordinate conversion
- **Spatial Operations**: Area-weighted averaging respecting Earth's spherical geometry
- **Data Quality**: Proper handling of missing values and coordinate systems
- **Standardization**: Statistical transformations preserving physical meaning

The functions support the research workflow from raw ERA5 data processing through
final causal discovery analysis, ensuring methodological consistency throughout.

Core Functionality
-----------------
**Seasonal Analysis Functions**: Standardized seasonal cycle removal and
standardization following climate science conventions

**Spatial Processing Functions**: Area-weighted averaging and coordinate
transformations accounting for Earth's spherical geometry

**Data Transformation**: Time coordinate conversion and anomaly computation
maintaining CF-compliant metadata and physical units

Key Scientific Standards
-----------------------
- **Coordinate Systems**: CF-compliant handling of latitude/longitude coordinates
- **Physical Units**: Unit preservation through all transformations with proper metadata
- **Statistical Methods**: Climate-appropriate statistical procedures with proper assumptions
- **Quality Assurance**: Robust handling of missing data and boundary conditions

Integration with Research Workflow
---------------------------------
These utilities support the complete ENSO-ERA5 analysis pipeline:
- Processing scripts (1a-11): Data preparation and index calculation
- Analysis scripts (12-23): Causal discovery and predictive modeling
- Consistent methodology across all climate data transformations

Dependencies and Performance
---------------------------
Built on xarray for N-dimensional labeled arrays, numpy for numerical operations,
and pandas for time series handling. Optimized for large climate datasets
(~GB scale) with memory-efficient operations and chunked processing support.
Requires scipy for advanced statistical operations and shapely for geometric
operations.

Examples and Usage Patterns
---------------------------
Standard workflow for climate data processing:
1. Load raw climate data with proper coordinate handling
2. Apply area weighting for spatial averages respecting Earth geometry
3. Compute seasonal anomalies removing climatological cycles
4. Standardize by season for statistical analysis compatibility

"""

import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
import xesmf as xe
import statsmodels.api as sm
from shapely.geometry import Point, Polygon
from typing import Tuple, Union, List, Optional

def convert_TS_from_Gregorian(ts_data: xr.Dataset, yyyy_mm_dd: str) -> xr.Dataset:
    """
    Converts the time coordinate of an xarray Dataset from a float format into a pandas DatetimeIndex.
    
    Parameters:
    ts_data (xarray.Dataset): The original dataset with a time coordinate in float format.
    yyyy_mm_dd (str): The start date in the format 'yyyy-mm-dd'. Note that the day must be the first day of the month (i.e., '01').
    
    Returns:
    ts_data_new (xarray.Dataset): The dataset with the converted time coordinate.
    """
    
    # Convert the start date string into a pandas Timestamp object
    start_date = pd.Timestamp(yyyy_mm_dd)
    
    # Get the number of time points in the original dataset
    n_dates = len(ts_data['time'])
    
    # Generate a new time coordinate as a sequence of dates with a monthly frequency, starting from the start date
    new_time = pd.date_range(start=start_date, periods=n_dates, freq='MS')
    
    # Replace the 'time' coordinate in the dataset with the new time coordinate
    ts_data_new = ts_data.assign_coords(time=new_time)
    
    return ts_data_new

def total_average(dataarray: xr.DataArray, lat_name: str) -> float:
    """
    Compute area-weighted global average respecting Earth's spherical geometry.

    This function performs proper spatial averaging of climate data by applying cosine
    of latitude weighting to account for the varying area of grid cells on Earth's
    surface. This is essential for computing physically meaningful global and regional
    averages from gridded climate datasets.

    Physical Background
    ------------------
    Grid cells on Earth's surface have different areas depending on latitude due to
    the spherical geometry. Near the equator, grid cells are larger than near the poles.
    Without proper weighting, simple arithmetic averaging would over-represent polar
    regions and under-represent equatorial regions, leading to biased climate statistics.

    The area weighting factor cos(latitude) correctly accounts for the meridional
    variation in grid cell area, ensuring that each location contributes to the
    average proportional to the actual area it represents on Earth's surface.

    Mathematical Formulation
    -----------------------
    Area-weighted average: ⟨f⟩ = ∫∫ f(lat,lon) × cos(lat) dA / ∫∫ cos(lat) dA

    where:
    - f(lat,lon): gridded climate field
    - cos(lat): latitude weighting factor (Earth geometry)
    - dA: differential area element

    This formulation ensures conservation of global integrals and proper representation
    of physical quantities like global mean temperature, precipitation, etc.

    Parameters
    ----------
    dataarray : xarray.DataArray
        Gridded climate data to be spatially averaged.
        Coordinate requirements: Must contain latitude and longitude dimensions
        Physical units: Any climate variable (temperature, precipitation, etc.)
        Missing values: Handled automatically with skipna=True

    lat_name : str
        Name of the latitude coordinate in the input DataArray.
        Accepted values: 'latitude' (CF-standard) or 'lat' (abbreviated)
        Quality control: Function validates coordinate existence

    Returns
    -------
    float
        Area-weighted global/regional average value.
        Physical units: Same as input DataArray
        Interpretation: Representative average accounting for Earth's geometry
        Quality: Properly weighted for climate applications

    Notes
    -----
    Scientific Applications:
    - Global mean surface temperature calculation
    - Regional precipitation averaging
    - Climate model validation and inter-comparison
    - Trend analysis requiring proper spatial weighting

    Coordinate System Requirements:
    - Latitude coordinates in degrees (-90 to +90)
    - Longitude coordinates in degrees (0-360 or -180 to +180)
    - Regular or irregular grids supported through xarray weighting

    Limitations:
    - Assumes spherical Earth (appropriate for climate scales)
    - Does not account for topographic effects on effective area
    - Grid cell boundaries assumed to be at coordinate centers

    Examples
    --------
    Compute global mean surface temperature:
    >>> temp_global = xr.open_dataset('surface_temperature.nc')['temp']
    >>> global_mean_temp = total_average(temp_global, 'latitude')
    >>> print(f"Global mean temperature: {global_mean_temp:.2f} °C")

    Regional average with custom coordinates:
    >>> precip_region = precip_data.sel(lat=slice(30, 60), lon=slice(-120, -80))
    >>> regional_precip = total_average(precip_region, 'lat')


    See Also
    --------
    lat_lon_weighting : Apply area weighting without averaging for further operations
    standardize_seasonal : Seasonal standardization after spatial averaging
    """
    if lat_name == 'lat':
        averaged_data = dataarray.weighted(np.cos(np.deg2rad(dataarray.lat))).mean(dim=['lon', 'lat'], skipna=True)
    else:
        averaged_data = dataarray.weighted(np.cos(np.deg2rad(dataarray.latitude))).mean(dim=['longitude', 'latitude'], skipna=True)
    
    return averaged_data

def long_0_360_to_180_180(da: xr.DataArray) -> xr.DataArray:
    """
    Converts longitude coordinates from the [0, 360) range to [-180, 180).
    After adjustment, re-sorts the data accordingly.

    Parameters
    ----------
    da : xarray.DataArray or xarray.Dataset
        Input data with a 'longitude' dimension in [0, 360) range.

    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Output data with 'longitude' in [-180, 180) range.
    """
    da['longitude'] = (da['longitude'] + 180) % 360 - 180
    da = da.sortby('longitude')
    return da

def add_cyclic_point_xr(data: xr.DataArray, coord: str = 'longitude') -> xr.DataArray:
    """
    Add a cyclic point at 180° longitude to eliminate gaps when plotting 
    with central_longitude=180 in Cartopy.
    
    This function adds a longitude point at 180° by copying data from -180°,
    ensuring seamless plotting across the dateline for Pacific-centered maps.
    
    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        Input data with longitude dimension in [-180, 180) range.
    coord : str, default 'longitude'
        Name of the longitude coordinate dimension.
    
    Returns
    -------
    xarray.DataArray or xarray.Dataset
        Output data with cyclic longitude point added at 180°.
        If cyclic point already exists or is not needed, returns original data.
        
    Examples
    --------
    >>> sst_cyclic = add_cyclic_point_xr(sst_data)
    >>> # Now sst_cyclic has longitude from -180° to 180° (inclusive)
    """
    import numpy as np
    import xarray as xr
    
    # Check if longitude coordinate exists
    if coord not in data.dims:
        return data
    
    # Get longitude values
    lons = data[coord].values
    
    # Determine cyclic boundary requirement for global longitude coverage
    # Only add if longitude spans from ~-180 to <180 and doesn't already have 180°
    if lons.min() <= -179 and lons.max() < 180 and not np.any(np.isclose(lons, 180.0)):
        # Create new longitude array with 180° added
        lon_cyclic = np.append(lons, 180.0)
        
        # Get data at -180° longitude (leftmost point)
        leftmost_data = data.isel({coord: 0})
        
        # Concatenate the leftmost data as rightmost (at 180°)
        data_cyclic = xr.concat([data, leftmost_data.expand_dims(coord)], dim=coord)
        data_cyclic[coord] = lon_cyclic
        
        return data_cyclic
    
    return data

def add_cyclic_point_to_vectors(u_data: xr.DataArray, v_data: xr.DataArray, coord: str = 'longitude') -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Add cyclic points to U/V vector components simultaneously to ensure
    proper vector field plotting across the dateline.
    
    Parameters
    ----------
    u_data : xarray.DataArray
        U (eastward) component of vector field.
    v_data : xarray.DataArray  
        V (northward) component of vector field.
    coord : str, default 'longitude'
        Name of the longitude coordinate dimension.
    
    Returns
    -------
    tuple of (xarray.DataArray, xarray.DataArray)
        Tuple of (u_cyclic, v_cyclic) with cyclic points added at 180°.
        If cyclic points not needed, returns original arrays.
        
    Examples
    --------
    >>> u_wind_cyclic, v_wind_cyclic = add_cyclic_point_to_vectors(u_wind, v_wind)
    """
    # Apply cyclic point to both components
    u_cyclic = add_cyclic_point_xr(u_data, coord=coord)
    v_cyclic = add_cyclic_point_xr(v_data, coord=coord)
    
    return u_cyclic, v_cyclic

def load_era_field(
    filepath: str,
    var_name: str,
    rename_coords: Optional[dict] = None,
    lat_slice: Optional[tuple] = None,
    transform_func: Optional[callable] = None,
    scale: Optional[float] = None,
    drop_vars_list: Optional[list] = None,
    select_level: Optional[int] = None,
    drop_level: bool = False,
    isel_dict: Optional[dict] = None,
    mean_dim: Optional[tuple] = None,
    sort_lat: Optional[bool] = None,
    squeeze: Optional[bool] = None,
) -> xr.DataArray:
    """
    Load a single ERA5 field from a NetCDF file with a context manager, applying
    intelligent automatic transformations:
    
      1. Auto-detect and rename coordinates for consistency (lat->latitude, lon->longitude).
         Override with rename_coords if you need custom renaming.
      2. Auto-sort latitudes in ascending order (override with sort_lat=False if needed).
      3. Auto-detect longitude range and transform from [0,360] to [-180,180] if needed.
      4. Auto-squeeze singleton dimensions (override with squeeze=False if needed).
      5. Slice latitude (lat_slice = (low, high)) if provided.
      6. Apply optional transform_func for custom transformations.
      7. Multiply by 'scale' if provided (useful for converting units).
      8. Drop certain variables (drop_vars_list) if needed.
      9. Select a single vertical level (select_level), dropping 'level' dimension if drop_level=True.
      10. isel_dict: dictionary to pass to .isel(...) (e.g., {'pressure_level': 0}).
      11. mean_dim: if you want vertical averaging, pass mean_dim=('level', (300,500))
         to get .sel(level=slice(300,500)).mean(dim='level')
         
    Returns a deep copy, so the underlying file is closed and freed from memory.
    """

    with xr.open_dataset(filepath) as ds:
        # 1) Auto-detect and rename coordinates for consistency
        coord_names = list(ds.coords)
        
        # Default coordinate mapping if not explicitly provided
        if rename_coords is None:
            rename_coords = {}
            # Check for lat/lon naming variants and standardize to latitude/longitude
            if 'lat' in coord_names and 'latitude' not in coord_names:
                rename_coords['lat'] = 'latitude'
            if 'lon' in coord_names and 'longitude' not in coord_names:
                rename_coords['lon'] = 'longitude'
        
        # Apply renaming if needed
        if rename_coords:
            ds = ds.rename(rename_coords)

        # 2) Extract the variable
        da = ds[var_name]

        # 3) Drop certain vars if needed
        if drop_vars_list:
            da = da.drop_vars(drop_vars_list, errors='ignore')

        # 4) isel if provided
        if isel_dict:
            da = da.isel(**isel_dict)

        # 5) Extract specific pressure level with optional dimension removal
        if select_level is not None:
            da = da.sel(level=select_level)
            if drop_level:
                da = da.drop_vars('level', errors='ignore')

        # 6) Perform layer-mean vertical integration over specified level range
        if mean_dim is not None:
            dim_name, level_slice = mean_dim
            da = da.sel({dim_name: slice(*level_slice)}).mean(dim=dim_name)

        # 7) Auto-detect and sort latitudes if needed
        if 'latitude' in da.coords:
            # Check if latitudes are in descending order
            lat_vals = da.latitude.values
            lat_descending = lat_vals[0] > lat_vals[-1]
            
            # Auto-sort latitude unless explicitly disabled
            if sort_lat is None or sort_lat:
                if lat_descending:
                    da = da.sortby('latitude', ascending=True)
            
        # 8) Slice latitude if requested
        if lat_slice:
            da = da.sel(latitude=slice(lat_slice[0], lat_slice[1]))

        # 9) Auto-detect longitude range and transform if needed
        if 'longitude' in da.coords:
            lon_min = float(da.longitude.min())
            lon_max = float(da.longitude.max())
            
            # Auto-detect if longitude transformation is needed
            needs_transform = lon_min >= 0 and lon_max > 180
            
            # Apply automatic longitude transformation unless explicit func provided
            if transform_func is None and needs_transform:
                da = long_0_360_to_180_180(da)
            elif transform_func is not None:
                da = transform_func(da)

        # 10) Apply scaling if provided
        if scale is not None:
            da = da * scale

        # 11) Auto-squeeze singleton dimensions by default unless explicitly disabled
        if squeeze is None or squeeze:
            da = da.squeeze(drop=True)

        # Return a deep copy so the dataset is closed immediately
        return da.copy(deep=True)
    
def compute_seasonal_anomalies(dataset: xr.DataArray, start_date: str, end_date: str) -> xr.DataArray:
    """
    Compute seasonal anomalies for climate variability analysis.

    This function computes seasonal anomalies by removing the seasonal climatological
    mean from each season, following standard climate analysis procedures. The method
    preserves temporal coordinates at season centers and maintains proper statistical
    properties essential for climate index construction and teleconnection analysis.

    Physical Background
    ------------------
    Climate variables exhibit strong seasonal cycles driven by the annual solar radiation
    cycle. To study interannual and longer-term climate variability (such as ENSO, IOD,
    or decadal oscillations), the predictable seasonal cycle must be removed to reveal
    anomalous patterns that drive climate impacts and predictability.

    The seasonal anomaly represents departures from the expected seasonal conditions:
    - Positive anomalies: above-normal conditions for that season
    - Negative anomalies: below-normal conditions for that season
    - Zero anomalies: near-normal seasonal conditions

    This procedure is fundamental for climate index construction, enabling comparison
    of different years and identification of climate mode signatures.

    Statistical Methodology
    ----------------------
    Algorithm: Seasonal cycle removal via climatological subtraction
    1. Group data by season (DJF, MAM, JJA, SON)
    2. Compute climatological mean for each season over the specified period
    3. Subtract seasonal climatology from original data
    4. Preserve temporal metadata and coordinate information

    Mathematical formulation:
    Anomaly(season, year) = Value(season, year) - Climatology(season)

    where Climatology(season) = mean(Value(season, all_years))

    Quality Control and Validation:
    - Resulting anomalies have zero seasonal mean by construction
    - Preserves interannual and longer-term variability patterns
    - Maintains physical units and coordinate system metadata

    Parameters
    ----------
    dataset : xarray.DataArray
        Climate time series with temporal dimension 'time'.
        Coordinate requirements: Must have proper datetime index
        Physical units: Any climate variable (temperature, precipitation, winds, etc.)
        Temporal resolution: Monthly data recommended for seasonal analysis
        Quality control: Missing values handled through xarray groupby operations

    start_date : str
        Beginning of climatological period in 'YYYY-MM' format.
        Standard practice: Use 30-year periods (e.g., '1991-01' to '2020-12')
        Climate baseline: Should represent current climate normal period
        Statistical significance: Minimum 20 years recommended

    end_date : str
        End of climatological period in 'YYYY-MM' format.
        Period selection: Must align with start_date for proper climatology
        Data requirements: Should cover complete seasonal cycles
        Temporal coverage: Balanced representation of all seasons essential

    Returns
    -------
    xarray.DataArray
        Seasonal anomaly time series with climatology removed.
        Physical units: Same as input (°C, mm/month, m/s, etc.)
        Temporal resolution: Same as input (typically monthly)
        Statistical properties: Zero seasonal mean, preserved variability
        Coordinate system: Preserves all original coordinates and metadata

    Notes
    -----
    Climate Applications:
    - ENSO index construction (Niño 3.4, SOI, etc.)
    - Climate model evaluation and bias assessment
    - Trend analysis and change detection
    - Teleconnection pattern identification

    Temporal Considerations:
    - Time coordinates maintained at season centers for proper phasing
    - Climatological period should be recent and representative
    - Account for calendar effects (leap years) in long time series
    - Consider data quality changes over time (observational networks)

    Physical Interpretation:
    - Large positive anomalies indicate unusual warmth/wetness for the season
    - Large negative anomalies indicate unusual coldness/dryness
    - Anomaly magnitude indicates strength of climate signal
    - Temporal persistence indicates climate mode activity

    Integration with Analysis Workflow:
    - Essential preprocessing for climate index construction
    - Required input for causal discovery analysis (seasonal stationarity)
    - Foundation for composite analysis and field significance testing
    - Standard procedure before correlation and regression analysis

    Examples
    --------
    Compute SST anomalies for ENSO analysis:
    >>> sst_data = xr.open_dataset('monthly_sst.nc')['sst']
    >>> sst_anoms = compute_seasonal_anomalies(sst_data, '1991-01', '2020-12')
    >>> # Use for Niño index construction
    >>> nino34_anoms = sst_anoms.sel(lat=slice(-5, 5), lon=slice(190, 240)).mean(['lat', 'lon'])

    Process precipitation data for monsoon analysis:
    >>> precip_data = xr.open_dataset('monthly_precip.nc')['precip']
    >>> precip_anoms = compute_seasonal_anomalies(precip_data, '1981-01', '2010-12')
    >>> # Examine JJA monsoon variability
    >>> jja_monsoon = precip_anoms.where(precip_anoms.time.dt.season == 'JJA')


    See Also
    --------
    standardize_seasonal : Further standardization for statistical analysis
    total_average : Spatial averaging for index construction
    lat_lon_weighting : Area weighting for regional averages
    """
    ds_sliced = dataset.sel(time=slice(start_date, end_date))
    # Resample into 3-month means using quarter start anchored on December
    ds_seasonal = ds_sliced.resample(time='QS-DEC').mean()
    # Convert time coordinate to pandas DatetimeIndex, shift by one month, and reassign
    new_times = pd.to_datetime(ds_seasonal.time.values) + pd.DateOffset(months=1)
    ds_seasonal = ds_seasonal.assign_coords(time=new_times)
    seasonal_mean = ds_seasonal.groupby("time.season").mean("time")
    ds_anomalies = ds_seasonal.groupby("time.season") - seasonal_mean
    return ds_anomalies

def select_irregular_region(data: xr.DataArray, vertices: List[Tuple[float, float]]) -> xr.DataArray:
    """
    Selects or masks out an irregular polygonal region defined by vertices.

    Parameters
    ----------
    data : xarray.DataArray
        The data array to select from, with 'latitude' and 'longitude' dims.
    vertices : list of tuple(float, float)
        List of (lon, lat) pairs defining the polygon in order.

    Returns
    -------
    xarray.DataArray
        Data inside the polygon region; outside points are set to NaN.

    Raises
    ------
    ValueError
        If no points fall within the specified polygon.
    """
    poly = Polygon(vertices)
    mask = xr.apply_ufunc(
        lambda lon, lat: Point(lon, lat).within(poly),
        data['longitude'], data['latitude'],
        vectorize=True
    )
    if mask.sum() == 0:
        raise ValueError("No data points fall within the specified polygon.")
    return data.where(mask, drop=True)


def lat_lon_weighting(dataarray: xr.DataArray, lat_name: str = 'latitude') -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Applies sqrt(cos(lat)) weighting to handle meridional convergence of grid cells.

    Parameters
    ----------
    dataarray : xarray.DataArray
        Input data with dimension: [time, latitude, longitude] or similar.
    lat_name : str, optional
        Name of the latitude coordinate ('latitude' or 'lat').

    Returns
    -------
    weighted_data : xarray.DataArray
        The data array after multiplying by sqrt(cos(lat)).
    weights_expanded : xarray.DataArray
        The 2D weighting factor field for reference or unweighting.
    """
    lats = dataarray[lat_name]
    lons = dataarray['longitude']
    weights = np.sqrt(np.cos(np.deg2rad(lats)))
    weights_expanded = weights.expand_dims({'longitude': lons}).transpose('latitude', 'longitude')
    
    weighted_data = dataarray * weights_expanded
    return weighted_data, weights_expanded


def standardize_seasonal(da: xr.DataArray) -> xr.DataArray:
    """
    Perform seasonal standardization for climate time series analysis.

    This function implements season-specific standardization (Z-score transformation)
    following climate science best practices for removing seasonal cycles while
    preserving interannual and longer-term variability. The method ensures that
    each season (DJF, MAM, JJA, SON) has zero mean and unit variance, which is
    essential for climate index construction and causal discovery analysis.

    Physical Background
    ------------------
    Climate variables exhibit strong seasonal cycles that can mask interannual
    variability patterns crucial for understanding climate teleconnections.
    Seasonal standardization:
    - Removes the mean seasonal cycle while preserving anomaly patterns
    - Equalizes variance across seasons, preventing seasonal bias in analysis
    - Maintains physical relationships between climate modes at interannual timescales
    - Enables proper statistical comparison between different seasonal conditions

    This approach is fundamental for ENSO analysis where DJF conditions (peak ENSO)
    must be comparable to other seasons for causal pathway identification.

    Parameters
    ----------
    da : xarray.DataArray
        Seasonal climate time series with temporal dimension 'time'.
        Physical units: Any climate variable (preserved in output)
        Coordinate system: Time coordinate with seasonal resolution
        Data requirements: Minimum 30 years recommended for robust statistics
        Quality control: Function handles missing values through xarray operations

    Returns
    -------
    xarray.DataArray
        Seasonally standardized climate index time series.
        Physical units: Dimensionless standard deviations
        Temporal resolution: Same as input (seasonal)
        Interpretation: Values represent departures from seasonal climatological mean
        Statistical properties: Each season has mean=0, std=1 by construction

    Notes
    -----
    Physical Interpretation:
    - Positive values (+1σ to +3σ): Above-normal conditions for that season
    - Negative values (-1σ to -3σ): Below-normal conditions for that season
    - Extreme events (|value| > 2σ): Unusual climate conditions (~2.5% probability)

    Statistical Methodology:
    - Algorithm: Season-specific Z-score transformation: (x - μ_season) / σ_season
    - Assumptions: Seasonal normality (robust to mild deviations)
    - Limitations: Requires sufficient data per season (>20 years minimum)
    - Validation: Check that output seasons have zero mean and unit variance

    Climate Science Context:
    - Standard procedure for climate index construction
    - Essential for removing seasonal bias in teleconnection analysis
    - Required preprocessing for PCMCI+ causal discovery with seasonal data
    - Enables comparison of climate anomalies across different seasons

    Examples
    --------
    Standardize ENSO index for causal analysis:
    >>> nino34_seasonal = load_seasonal_sst_index('nino34')
    >>> nino34_std = standardize_seasonal(nino34_seasonal)
    >>> print(f"DJF variance: {nino34_std.sel(time=nino34_std.time.dt.month==12).var():.3f}")
    1.000

    Multi-year seasonal standardization:
    >>> precip_data = xr.open_dataset('seasonal_precipitation.nc')['precip']
    >>> precip_std = standardize_seasonal(precip_data)
    >>> # Verify standardization worked correctly
    >>> seasonal_stats = precip_std.groupby('time.season').agg(['mean', 'std'])


    See Also
    --------
    compute_seasonal_anomalies : Remove seasonal cycle from climate data
    lat_lon_weighting : Apply area weighting for spatial averaging
    """
    def _std_dim(data, dim='time'):
        return (data - data.mean(dim=dim)) / data.std(dim=dim)

    season_labels = da.time.dt.month
    return da.groupby(season_labels).map(_std_dim)

def get_symmetric_levels(spatial_pattern: xr.DataArray, spacing: float = 0.1, factor: float = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate symmetric levels for colorbars and contour lines based on the range of a spatial pattern.
    
    Args:
        spatial_pattern (xarray.DataArray): The spatial pattern to generate levels for.
        spacing (float): The desired spacing between levels (default: 0.1).
        factor (float): A factor to divide the maximum absolute value of the pattern (default: 1).   
    Returns:
        clevs (numpy.ndarray): Levels for the colorbar.
        levs_cont (numpy.ndarray): Levels for the contour lines.
    """
    max_abs_val = max(abs(spatial_pattern.min()), abs(spatial_pattern.max()))/factor
    max_abs_val = np.ceil(max_abs_val / spacing) * spacing
    
    clevs = np.arange(-max_abs_val, max_abs_val + spacing, spacing)
    levs_cont = np.arange(-max_abs_val, max_abs_val + spacing * 2, spacing * 2)
    
    return clevs, levs_cont

def fill_one_season_to_all_seasons(data: xr.DataArray, season_to_fill: str, var_name: str) -> xr.DataArray:
    """
    Distributes a single-season time series into a full 3-monthly series,
    leaving NaN in other seasons.

    Parameters
    ----------
    data : xarray.DataArray
        The data for the chosen season, dimension: [time].
    season_to_fill : str
        Which season to fill (e.g. 'SON', 'MAM'); must align with your time index.
    var_name : str
        Name of the output variable.

    Returns
    -------
    xarray.DataArray
        A 3-monthly DataArray from 1945-06-01 to 2024-03-01 with data only
        in the chosen season slots.
    """
    timestamps = pd.date_range(start='1945-06-01', end='2024-03-01', freq='3ME')
    seasonal_data = np.full(len(timestamps), np.nan)
    
    if season_to_fill == 'SON':
        seasonal_data[::4] = data.values
    elif season_to_fill == 'MAM':
        seasonal_data[2::4] = data.values
    else:
        raise ValueError("Unsupported season. Use 'SON' or 'MAM', etc.")
    
    da_seasonal = xr.DataArray(
        data=seasonal_data,
        coords={'time': timestamps},
        dims=['time'],
        name=var_name
    )
    return da_seasonal


def standardize_dim(x: xr.DataArray, dim: str = 'time') -> xr.DataArray:
    """
    Standardizes x along a named dimension by subtracting mean and dividing by std.

    Parameters
    ----------
    x : xarray.DataArray
        Data to be standardized.
    dim : str
        Dimension name along which to standardize (default 'time').

    Returns
    -------
    xarray.DataArray
        Standardized data = (x - mean) / std
    """
    return (x - x.mean(dim=dim)) / x.std(dim=dim)


def remove_signal(da: xr.DataArray, signal: xr.DataArray, dim: str, standardize_dim: bool = True) -> xr.DataArray:
    """
    Removes the influence of a time series from da via linear regression.
    var_corrected = var_to_modify - (model.predict signal)

    Parameters
    ----------
    da : xarray.DataArray
        The variable to be 'cleaned' of the signal, dimension includes 'time'.
    signal : xarray.DataArray
        1D time series to remove from var_to_modify.
    dim : str
        The dimension along which to do the regression (usually 'time').
    standardize_dim : bool, optional
        Whether to z-score the result after regression. Default True.

    Returns
    -------
    xarray.DataArray
        The corrected variable, optionally standardized.
    """
    
    # Input validation
    if dim not in da.dims:
        raise ValueError(f"Dimension {dim} not found in data array")
    
    # Get non-regression dimensions
    other_dims = [d for d in da.dims if d != dim]
    
    # Handle case when there are no other dimensions
    if not other_dims:
        # Prepare data for regression
        X = sm.add_constant(signal.values)
        y = da.values
        
        # Perform regression
        model = sm.OLS(y, X).fit()
        residuals = y - model.predict(X)
        
        # Create output DataArray
        result = xr.DataArray(
            residuals,
            dims=da.dims,
            coords=da.coords
        )
    else:
        # Stack multiple dimensions case
        stacked = da.stack(space=other_dims)
        X = sm.add_constant(signal.values)
        
        # Perform regression
        residuals = np.zeros(stacked.shape)
        for i in range(stacked.shape[1]):
            y = stacked[:, i].values
            model = sm.OLS(y, X).fit()
            residuals[:, i] = y - model.predict(X)
            
        # Unstack result
        result = xr.DataArray(
            residuals,
            dims=stacked.dims,
            coords=stacked.coords
        ).unstack('space')
    
    if standardize_dim:
        result = (result - result.mean(dim=dim)) / result.std(dim=dim)
    
    return result

def regress_1D(y, x):
    """
    Regress y ~ x for a single dimension and return [regr_coef, p_value, r_squared].

    If there are insufficient valid points or all NaNs, returns [nan, nan, nan].

    Parameters
    ----------
    y : np.ndarray
        Dependent variable, 1D array.
    x : np.ndarray
        Independent variable, 1D array of same size as y.

    Returns
    -------
    np.ndarray of shape (3,)
        [slope_coefficient, p_value, r_squared]
    """
    if np.all(np.isnan(y)) or np.all(np.isnan(x)):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    mask = ~np.isnan(y) & ~np.isnan(x)
    if mask.sum() < 3:
        return np.array([np.nan, np.nan, np.nan], dtype=np.float32)

    y_clean = y[mask]
    x_clean = x[mask]

    # Add constant
    x_clean = sm.add_constant(x_clean)

    model = sm.OLS(y_clean, x_clean).fit()
    regr_coef = model.params[1]
    p_val = model.f_pvalue
    r_sq = model.rsquared
    
    return np.array([regr_coef, p_val, r_sq], dtype=np.float32)

def average_over_irregular_region_pval(data, vertices, p_value=None, p_value_threshold=0.01):
    """
    Averages over an irregular region of an xarray DataArray.
    
    If p_value is provided, only grid points with p_value below the specified threshold
    are included in the average. Otherwise, the average is computed over all points
    within the polygon defined by the vertices.
    
    Parameters:
        data (xarray.DataArray): The data to average.
        vertices (list): List of (longitude, latitude) tuples defining the polygon.
        p_value (xarray.DataArray, optional): p-value data for the grid points. Defaults to None.
        p_value_threshold (float, optional): p-value threshold for including grid points. Defaults to 0.01.
    
    Returns:
        tuple: A tuple containing:
            - data_in_region (xarray.DataArray): Data values within the defined region (and passing p-value filtering if provided).
            - averaged_data (xarray.DataArray): The average over the 'latitude' (and longitude, if applicable) dimension.
    """
    # Create a polygon using the provided vertices
    poly = Polygon(vertices)

    # Create a mask that is True for points within the polygon
    polygon_mask = xr.apply_ufunc(
        lambda lon, lat: Point(lon, lat).within(poly),
        data['longitude'], data['latitude'],
        vectorize=True
    )

    # If p_value is provided, combine the polygon mask with the p_value threshold mask
    if p_value is not None:
        mask = polygon_mask & (p_value < p_value_threshold)
    else:
        mask = polygon_mask

    # Check if the mask yields any selected points
    if mask.sum() == 0:
        threshold_info = " and p_value threshold" if p_value is not None else ""
        raise ValueError(f"No data points fall within the specified polygon{threshold_info}.")

    # Select data using the mask and drop points not meeting criteria
    data_in_region = data.where(mask, drop=True)

    if data_in_region.count() == 0:
        threshold_info = " and p_value threshold" if p_value is not None else ""
        raise ValueError(f"No valid data points left after applying the mask{threshold_info}.")

    # Average over the latitude dimension (assuming total_average works appropriately here)
    averaged_data = total_average(data_in_region, 'latitude')

    return data_in_region, averaged_data

def apply_spatial_weighting(dataarray: xr.DataArray, 
                            lat_name: str = 'latitude') -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Apply spatial weighting to account for decreasing grid cell area with latitude.
    """
    lats = dataarray[lat_name]
    lons = dataarray['longitude']
    
    weights = np.sqrt(np.cos(np.deg2rad(lats)))
    weights_expanded = weights.expand_dims({'longitude': lons}).transpose('latitude', 'longitude')
    
    weighted_data = dataarray * weights_expanded
    
    return weighted_data, weights_expanded

def regrid_to_target(da, target_da, lat_name='latitude', lon_name='longitude'):
    """
    Regrid an xarray DataArray or Dataset (da) to the same grid as target_da 
    using xESMF bilinear interpolation.
    """
    # 1) Rename input dataset coords to xESMF's expected 'lat' and 'lon'
    ds_in = da.rename({lat_name: 'lat', lon_name: 'lon'})
    
    # 2) Rename target dataset coords for xESMF
    ds_out_grid = target_da.rename({lat_name: 'lat', lon_name: 'lon'})
    
    # 3) Create regridder with the target grid
    regridder = xe.Regridder(ds_in, ds_out_grid, 'bilinear', reuse_weights=False)
    
    # 4) Regrid
    ds_regridded = regridder(ds_in)
    
    # 5) Rename coords back
    ds_regridded = ds_regridded.rename({'lat': lat_name, 'lon': lon_name})
    
    return ds_regridded

def mask_data_over_land(sst_data, data_to_mask):
    """
    Masks the data over land based on the NaN values in the SST data.

    Args:
        sst_data (xr.DataArray): The SST data used to create the land mask.
        data_to_mask (xr.DataArray): The data to be masked over land.

    Returns:
        xr.DataArray: The masked data, where values over the ocean are set to NaN.
    """
    # Create a mask for land areas
    land_mask = np.isnan(sst_data.values)

    # Apply the land mask to the data
    masked_data = data_to_mask.where(land_mask)

    return masked_data

def get_symmetric_levels_fixed_spacing(spatial_pattern, spacing=0.05, factor=1):
    """
    From REOF_SST.py: symmetric contour & colorbar levels
    """
    import logging
    logger = logging.getLogger(__name__)

    data = spatial_pattern.values if hasattr(spatial_pattern, 'values') else spatial_pattern

    # Add diagnostic logging
    max_abs = np.nanmax(np.abs(data)) / factor
    logger.debug(f"Level calc: data shape={data.shape}, max_abs={max_abs:.6f}, spacing={spacing}")
    logger.debug(f"Data range: [{np.nanmin(data):.6f}, {np.nanmax(data):.6f}]")
    logger.debug(f"NaN ratio in data: {np.isnan(data).sum() / data.size:.3f}")
    logger.debug(f"All zeros in data: {np.all(data == 0)}")
    logger.debug(f"Factor: {factor}, original max_abs: {np.nanmax(np.abs(data)):.6f}")

    bound = np.round(max_abs / spacing) * spacing
    logger.debug(f"Computed bound: {bound:.6f}")

    clevs = np.arange(-bound, bound + spacing, spacing)
    levs_cont = np.arange(-bound, bound + 2*spacing, 2*spacing)

    logger.debug(f"Generated {len(clevs)} levels: {clevs[:3] if len(clevs) > 3 else clevs} ...")
    logger.debug(f"Generated {len(levs_cont)} contour levels: {levs_cont[:3] if len(levs_cont) > 3 else levs_cont} ...")

    if len(clevs) < 2:
        logger.warning(f"CRITICAL: Only {len(clevs)} level(s) generated! This will cause matplotlib contourf to fail")
        logger.warning(f"  Data max_abs: {max_abs:.6f}, bound: {bound:.6f}, spacing: {spacing}")

    return clevs, levs_cont
