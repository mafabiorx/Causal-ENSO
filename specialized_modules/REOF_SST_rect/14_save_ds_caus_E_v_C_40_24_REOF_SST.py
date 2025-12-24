import xarray as xr
import os
import sys
from pathlib import Path
# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parent.parent.parent))
from utils.paths import get_data_path, get_results_path

## Loading and preprocessing data ## 

# Get environment variables for file paths
target_descriptor = os.environ.get("TARGET_DESCRIPTOR")
generation = os.environ.get("GA_GENERATION")
individual = os.environ.get("GA_INDIVIDUAL")
trial = os.environ.get("TRIAL")

# Ensure environment variables are set
if not all([target_descriptor, generation, individual]):
    print("Error: Required environment variables not set")
    print(f"TARGET_DESCRIPTOR: {target_descriptor}")
    print(f"GA_GENERATION: {generation}")
    print(f"GA_INDIVIDUAL: {individual}")
    sys.exit(1)

print(f"Processing all REOF modes")

VERSION = os.environ.get("VERSION", 'v0')

# Define paths
DATA_DIR_TS = get_data_path('time_series', data_type="processed")
RESULTS_DIR = get_results_path('REOF_SST/')
target_dir = os.path.join(RESULTS_DIR, trial, target_descriptor, VERSION)
SAVE_DIR = os.path.join(target_dir, 'NetCDFs_REOF_modes')

# Load the signals
DMI_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'DMI_st_ts.nc'))['DMI']
SASDI_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'SASDI_st_ts.nc'))['SASDI']
Atl3_DJF_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'Atl3_st_ts.nc'))['ATL3']
NTA_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'NTA_st_ts.nc'))['NTA']
SIOD_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'SIOD_st_ts.nc'))['SIOD']
WNP_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'WNP_st_ts.nc'))['WNP']

SPO_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'SPO_st_ts.nc')).rename({'__xarray_dataarray_variable__':'SPO'}).SPO
NPMM_SST_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'NPMM_SST_st_ts.nc')).rename({'__xarray_dataarray_variable__':'NPMM_SST'}).NPMM_SST
NPMM_wind_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'NPMM_wind_st_ts.nc')).rename({'__xarray_dataarray_variable__':'NPMM_wind'}).NPMM_wind
SPMM_SST_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'SPMM_SST_st_ts.nc')).rename({'__xarray_dataarray_variable__':'SPMM_SST'}).SPMM_SST
SPMM_wind_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'SPMM_wind_st_ts.nc')).rename({'__xarray_dataarray_variable__':'SPMM_wind'}).SPMM_wind

E_ind_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'E_index_st_ts.nc')).rename({'E_index':'E_ind'}).E_ind
C_ind_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'C_index_st_ts.nc')).rename({'C_index':'C_ind'}).C_ind

# Load first 4 REOF modes
PC1_SST_st = xr.open_dataset(os.path.join(SAVE_DIR, f'REOF_SST_timeseries_mode1_gen{generation}_ind{individual}.nc'))['PC1_SST']
PC2_SST_st = xr.open_dataset(os.path.join(SAVE_DIR, f'REOF_SST_timeseries_mode2_gen{generation}_ind{individual}.nc'))['PC2_SST']
PC3_SST_st = xr.open_dataset(os.path.join(SAVE_DIR, f'REOF_SST_timeseries_mode3_gen{generation}_ind{individual}.nc'))['PC3_SST']
PC4_SST_st = xr.open_dataset(os.path.join(SAVE_DIR, f'REOF_SST_timeseries_mode4_gen{generation}_ind{individual}.nc'))['PC4_SST']

# Create an xarray Dataset of all signals (predictor and responses)
data_st = xr.Dataset({
    # JJA predictors
    'PC1_SST': PC1_SST_st,
    'PC2_SST': PC2_SST_st,
    'PC3_SST': PC3_SST_st,
    'PC4_SST': PC4_SST_st,
    'DMI_JJA': DMI_st,
    # SON predictors
    'MCA2 prec-RWS SON': MCA2_prec_RWS_SON_C_st,
    'SIOD_SON': SIOD_st,
    # DJF confounders
    'SASDI_DJF_0': SASDI_st,
    'Atl3_DJF_0': Atl3_DJF_st,
    'NPMM_SST_DJF_0': NPMM_SST_st,
    'WNP_DJF_0': WNP_st,
    # MAM mediators
    'NPMM_wind_MAM': NPMM_wind_st,
    'NTA_MAM': NTA_st,
    'SPO_MAM': SPO_st,
    'SPMM_SST_MAM': SPMM_SST_st,
    'SPMM_wind_MAM': SPMM_wind_st,  
    # DJF effects
    'E-ind DJF(1)': E_ind_st,
    'C-ind DJF(1)': C_ind_st
})

# Export the xarray Dataset to a NetCDF file in the target directory
output_file = os.path.join(target_dir, f'ds_caus_REOF_40_24_gen{generation}_ind{individual}.nc')
data_st.to_netcdf(output_file)
print(f'Data exported to {output_file}')