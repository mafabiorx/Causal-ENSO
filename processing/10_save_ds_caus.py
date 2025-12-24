import xarray as xr
import os
import sys
from pathlib import Path
# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.paths import get_data_path

# Define paths
DATA_DIR_TS = get_data_path('time_series/', data_type="processed")

# Load the signals
DMI_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'DMI_st_ts.nc'))['DMI']
SASDI_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'SASDI_st_ts.nc'))['SASDI']
Atl3_DJF_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'Atl3_st_ts.nc'))['ATL3']
NTA_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'NTA_st_ts.nc'))['NTA']
SIOD_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'SIOD_st_ts.nc'))['SIOD']
WNP_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'WNP_st_ts.nc'))['WNP']

SPO_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'SPO_st_ts.nc')).rename({'__xarray_dataarray_variable__':'SPO'}).SPO
NPO_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'NPO_st_ts.nc')).rename({'__xarray_dataarray_variable__':'NPO'}).NPO
NPMM_SST_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'NPMM_SST_st_ts.nc')).rename({'__xarray_dataarray_variable__':'NPMM_SST'}).NPMM_SST
NPMM_wind_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'NPMM_wind_st_ts.nc')).rename({'__xarray_dataarray_variable__':'NPMM_wind'}).NPMM_wind
SPMM_SST_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'SPMM_SST_st_ts.nc')).rename({'__xarray_dataarray_variable__':'SPMM_SST'}).SPMM_SST
SPMM_wind_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'SPMM_wind_st_ts.nc')).rename({'__xarray_dataarray_variable__':'SPMM_wind'}).SPMM_wind

E_ind_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'E_index_st_ts.nc')).rename({'E_index':'E_ind'}).E_ind
C_ind_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'C_index_st_ts.nc')).rename({'C_index':'C_ind'}).C_ind

PSA2_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'PSA2_st_ts.nc')).rename({'__xarray_dataarray_variable__':'PSA2'}).PSA2

# Load the JJA predictors
REOF_SST_ts_mode4_JJA = xr.open_dataset(os.path.join(DATA_DIR_TS, 'E-ind',
                                'REOF_SST_ts_mode4_JJA.nc'))['PC4_SST']

# Load the SON predictor
MCA1_WAF_RWS_SON_E_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'E-ind',
                                 'MCA_RWS_WAF_ts_WAF_mode1_SON.nc'))['pc_WAF']

MCA2_prec_RWS_SON_C_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'C-ind',
                                 'MCA_prec_RWS_200_ts_prec_mode2_SON.nc'))['pc_prec']

# Load the DJF predictors
MCA2_RWS_WAF_DJF_E_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'E-ind',
                                 'MCA_RWS_200_WAF_ts_RWS_200_mode2_DJF.nc'))['pc_RWS']

MCA2_RWS_prec_DJF_C_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'C-ind',
                                 'MCA_prec_RWS_200_ts_RWS_200_mode2_DJF.nc'))['pc_RWS_200']

# Load the MAM predictors
MCA2_RWS_prec_MAM_E_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'E-ind',
                                 'MCA_prec_RWS_200_ts_RWS_200_mode2_MAM.nc'))['pc_RWS_200']
MCA2_RWS_prec_MAM_C_st = xr.open_dataset(os.path.join(DATA_DIR_TS, 'C-ind',
                                 'MCA_prec_RWS_200_ts_RWS_200_mode2_MAM.nc'))['pc_RWS_200']

# Create an xarray Dataset of all signals (predictor and responses)
data_st = xr.Dataset({
    # JJA predictors
    'REOF SST JJA': REOF_SST_ts_mode4_JJA,
    'DMI JJA': DMI_st,
    'PSA2 JJA': PSA2_st,
    # SON predictors
    'MCA WAF-RWS SON': MCA1_WAF_RWS_SON_E_st,
    'MCA prec-RWS SON': MCA2_prec_RWS_SON_C_st,
    'SIOD MAM': SIOD_st,
    # DJF predictors
    'MCA RWS-WAF DJF': MCA2_RWS_WAF_DJF_E_st,
    'MCA RWS-prec DJF': MCA2_RWS_prec_DJF_C_st,    
    'SASDI SON': SASDI_st,
    'Atl3 DJF': Atl3_DJF_st,
    'NPMM-SST DJF': NPMM_SST_st,
    'WNP DJF': WNP_st,
    'NPO DJF': NPO_st,
    # MAM predictors
    'NPMM-wind MAM': NPMM_wind_st,
    'MCA RWS-prec MAM(E)': MCA2_RWS_prec_MAM_E_st,
    'MCA RWS-prec MAM(C)': MCA2_RWS_prec_MAM_C_st,
    'NTA MAM': NTA_st,
    'SPO MAM': SPO_st,
    'SPMM-SST MAM': SPMM_SST_st,
    'SPMM-wind MAM': SPMM_wind_st,
    'E-ind DJF(1)': E_ind_st,
    'C-ind DJF(1)': C_ind_st
})

# Export the xarray Dataset to a NetCDF file
data_st.to_netcdf(os.path.join(DATA_DIR_TS, 'PCMCI_data_ts_st.nc'))

## Create Physical Mechanisms Dataset ##

# Load physical mechanism time series
panama_v = xr.open_dataset(os.path.join(DATA_DIR_TS, 'panama_jet_panama_jet_V_st.nc')).rename({'v10':'panama_v'}).panama_v
west_pac_u = xr.open_dataset(os.path.join(DATA_DIR_TS, 'westerlies_west_Pac_U_st.nc')).rename({'u10':'west_pac_u'}).west_pac_u
rws_setp = xr.open_dataset(os.path.join(DATA_DIR_TS, 'RWS_200_SETP_st.nc')).rename({'RWS':'rws_setp'}).rws_setp
coastal_v = xr.open_dataset(os.path.join(DATA_DIR_TS, 'coastal_winds_V_st.nc')).rename({'v10':'coastal_v'}).coastal_v
sc_core = xr.open_dataset(os.path.join(DATA_DIR_TS, 'sc_core_st.nc')).rename({'lcc':'sc_core'}).sc_core
sc_primary = xr.open_dataset(os.path.join(DATA_DIR_TS, 'sc_primary_st.nc')).rename({'lcc':'sc_primary'}).sc_primary
sc_wide = xr.open_dataset(os.path.join(DATA_DIR_TS, 'sc_wide_st.nc')).rename({'lcc':'sc_wide'}).sc_wide

# Load Surface latent heat flux
latent_heat_flux = xr.open_dataset(os.path.join(DATA_DIR_TS, 'latent_heat_flux_SETP_st.nc')).rename({'avg_slhtf':'latent_heat_flux'}).latent_heat_flux

# Load Bakun Upwelling Index  
bakun_ui = xr.open_dataset(os.path.join(DATA_DIR_TS, 'bakun_ui_SA_coastline_st.nc'))['bakun_ui']

# Create physical mechanisms dataset
phy_mech_st = xr.Dataset({
    # Physical mechanisms
    'Panama V': panama_v,
    'West Pac U': west_pac_u,
    'RWS SETP': rws_setp,
    'Coastal V': coastal_v,
    'Sc core': sc_core,
    'Sc primary': sc_primary,
    'Sc wide': sc_wide,
    'Latent Heat Flux': latent_heat_flux,
    'Bakun UI': bakun_ui,
    # Target variables
    'E-ind DJF(1)': E_ind_st,
    'C-ind DJF(1)': C_ind_st
})

# Export the physical mechanisms dataset to a NetCDF file
phy_mech_st.to_netcdf(os.path.join(DATA_DIR_TS, 'PCMCI_data_ts_phy_mech_st.nc'))
