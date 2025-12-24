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

print(f"Processing all MCA modes")

VERSION = os.environ.get("VERSION", 'v0')

# Define paths
DATA_DIR_TS = get_data_path('time_series', data_type="processed")
RESULTS_DIR = get_results_path('MCA_prec_RWS_200/')
target_dir = os.path.join(RESULTS_DIR, trial, target_descriptor, VERSION)
SAVE_DIR = os.path.join(target_dir, 'NetCDFs_MCA_modes')

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

# Load the JJA predictors
REOF_SST_ts_mode4_JJA = xr.open_dataset(os.path.join(DATA_DIR_TS, 'E-ind',
                                'REOF_SST_ts_mode4_JJA.nc'))['PC4_SST']

# Load the SON predictor
MCA1_WAF_RWS_SON_E_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'E-ind',
                                 'MCA_RWS_WAF_ts_WAF_mode1_SON.nc'))['pc_WAF']

MCA2_prec_RWS_SON_C_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'C-ind',
                                 'MCA_prec_RWS_200_ts_prec_mode2_SON.nc'))['pc_prec']

# Load the DJF predictors
MCA2_RWS_WAF_E_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'E-ind',
                                 'MCA_RWS_200_WAF_ts_RWS_200_mode2_DJF.nc'))['pc_RWS']

MCA2_prec_RWS_DJF_C_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'C-ind',
                                 'MCA_prec_RWS_200_ts_RWS_200_mode2_DJF.nc'))['pc_RWS_200']

# Load first 2 MCA modes of prec and first 2 MCA modes of RWS_200
# prec modes
MCA1_prec = xr.open_dataset(os.path.join(SAVE_DIR, 
                                f'MCA_prec_RWS_200_timeseries_prec_mode1_gen{generation}_ind{individual}.nc'))['pc_prec']
MCA2_prec = xr.open_dataset(os.path.join(SAVE_DIR, 
                                f'MCA_prec_RWS_200_timeseries_prec_mode2_gen{generation}_ind{individual}.nc'))['pc_prec']

# RWS_200 modes                              
MCA1_RWS_200 = xr.open_dataset(os.path.join(SAVE_DIR, 
                                f'MCA_prec_RWS_200_timeseries_RWS_200_mode1_gen{generation}_ind{individual}.nc'))['pc_RWS_200']
MCA2_RWS_200 = xr.open_dataset(os.path.join(SAVE_DIR, 
                                f'MCA_prec_RWS_200_timeseries_RWS_200_mode2_gen{generation}_ind{individual}.nc'))['pc_RWS_200']

# Create an xarray Dataset of all signals (predictor and responses)
data_st = xr.Dataset({
    # JJA predictors
    'REOF SST JJA': REOF_SST_ts_mode4_JJA,
    'DMI_JJA': DMI_st,
    # SON predictors
    'MCA WAF-RWS SON': MCA1_WAF_RWS_SON_E_st,
    'MCA prec-RWS SON': MCA2_prec_RWS_SON_C_st,
    'SIOD_SON': SIOD_st,
    # DJF confounders
    'MCA1_prec': MCA1_prec,
    'MCA2_prec': MCA2_prec,
    'MCA1_RWS_200': MCA1_RWS_200,
    'MCA2_RWS_200': MCA2_RWS_200,
    'MCA2 RWS-WAF DJF': MCA2_RWS_WAF_E_st,
    # 'MCA2 RWS-prec DJF': MCA2_prec_RWS_DJF_C_st,
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
output_file = os.path.join(target_dir, f'ds_caus_MCA_40_24_gen{generation}_ind{individual}.nc')
data_st.to_netcdf(output_file)
print(f'Data exported to {output_file}')