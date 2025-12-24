"""
Multi-Basin Climate Index Computation: SST-Based Modes

Core Functionality:
- Compute Dipole Mode Index (DMI) for Indian Ocean variability
- Calculate South Atlantic Subtropical Dipole Index (SASDI)
- Extract Atlantic tropical indices (ATL3, NTA)
- Process Southern Indian Ocean Dipole (SIOD) and Western North Pacific (WNP) modes

Output: Standardized climate mode time series for multi-basin teleconnection analysis
"""

# Import the tools
import numpy as np
import xarray as xr
import pandas as pd
import statsmodels.api as sm
import warnings
import os
import sys
import logging
from pathlib import Path
warnings.filterwarnings("ignore", category=FutureWarning, module="xarray")
# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))
from utils.TBI_functions import total_average, compute_seasonal_anomalies, standardize_seasonal, convert_TS_from_Gregorian
from utils.paths import get_data_path, get_results_path
from utils.processing_results import ProcessingResult, ProcessingSummary, ProcessingStatus

# Setup logging
logger = logging.getLogger('4_SST-based_ts')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(handler)

# Set up paths (module level for potential imports)
DATA_DIR_TS = get_data_path('', data_type="external")
DATA_DIR = get_data_path('seasonal/', data_type="raw")
SAVE_DIR = get_data_path('time_series/')


def main():
    """Main function for SST-based climate index computation."""
    # Initialize processing summary
    summary = ProcessingSummary(script_name='4_SST-based_ts.py')

    # --- DMI (Dipole Mode Index) ---
    logger.info("Processing DMI...")
    dmi_filepath = os.path.join(DATA_DIR_TS, 'DMI_hadsst.nc')
    if os.path.exists(dmi_filepath):
        try:
            DMI = xr.open_dataset(dmi_filepath, decode_times=False).rename({'diff':'DMI'})['DMI']
            DMI = convert_TS_from_Gregorian(DMI, '1870-01-01').sel(time=slice('1945-06', '2024-02'))
            DMI_seas = compute_seasonal_anomalies(DMI, '1945-06-01', '2024-02-29').drop_vars('season', errors='ignore')
            DMI_seas_st = standardize_seasonal(DMI_seas)
            save_path = os.path.join(SAVE_DIR, 'DMI_st_ts.nc')
            DMI_seas_st.to_netcdf(save_path)
            logger.info("DMI processing complete.")
            summary.add_result(ProcessingResult('DMI', ProcessingStatus.SUCCESS, output_path=save_path))
        except Exception as e:
            logger.error(f"Error processing DMI: {e}", exc_info=True)
            summary.add_result(ProcessingResult('DMI', ProcessingStatus.FAILED, error_message=str(e)))
    else:
        logger.error(f"DMI file not found at {dmi_filepath}")
        summary.add_result(ProcessingResult('DMI', ProcessingStatus.FAILED, error_message=f"File not found: {dmi_filepath}"))

    # --- Load Base SST Data ---
    logger.info("Loading base SST data...")
    sst_filepath = os.path.join(DATA_DIR, 'SST_seas.nc') # Assuming raw seasonal SST data exists here
    if os.path.exists(sst_filepath):
        try:
            # Load base SST data
            with xr.open_dataset(sst_filepath) as ds:
                # Select variable, ensure latitude is ascending, load into memory if needed
                SST = ds.sst.sortby('latitude', ascending=True).load()
                logger.info(f"Base SST data loaded successfully from {sst_filepath}")

            # Compute seasonal anomalies ONCE for efficiency
            # Date range must cover complete seasonal cycles for accurate anomaly computation
            logger.info("Computing seasonal SST anomalies...")
            SST_seas = compute_seasonal_anomalies(SST, '1945-06-01', '2024-02-29').drop_vars('season', errors='ignore')
            logger.info("Seasonal SST anomalies computed.")

            # --- SASDI Ham 2021 (South Atlantic Subtropical Dipole Index) ---
            logger.info("Processing SASDI (Ham 2021)...")
            try:
                # Define poles based on Ham et al., 2021
                SW_pole_SASD_Ham = total_average(SST_seas.sel(latitude=slice(-45, -35), longitude=slice(-60, 0)), 'latitude')
                NE_pole_SASD_Ham = total_average(SST_seas.sel(latitude=slice(-30, -20), longitude=slice(-40, 20)), 'latitude')

                # Calculate index (SW - NE)
                SASDI_Ham_seas = SW_pole_SASD_Ham - NE_pole_SASD_Ham
                SASDI_Ham_seas.name = 'SASDI' # Assign a name

                # Standardize seasonally
                SASDI_Ham_seas_st = standardize_seasonal(SASDI_Ham_seas)

                # Save the standardized index
                save_path = os.path.join(SAVE_DIR, 'SASDI_st_ts.nc')
                SASDI_Ham_seas_st.to_netcdf(save_path)
                logger.info(f"SASDI processing complete. Saved to {save_path}")
                summary.add_result(ProcessingResult('SASDI', ProcessingStatus.SUCCESS, output_path=save_path))
            except Exception as e:
                logger.error(f"Error processing SASDI: {e}", exc_info=True)
                summary.add_result(ProcessingResult('SASDI', ProcessingStatus.FAILED, error_message=str(e)))

            # --- Atlantic Nino 3 (ATL3) Index ---
            logger.info("Processing ATL3...")
            try:
                # Define ATL3 region
                lat_bounds_atl3 = [-3, 3]
                lon_bounds_atl3 = [-20, 0]

                # Select data and compute area average
                atl3_sst = SST_seas.sel(latitude=slice(*lat_bounds_atl3), longitude=slice(*lon_bounds_atl3))
                atl3_index = total_average(atl3_sst, 'latitude')
                atl3_index.name = 'ATL3' # Assign a name

                # Standardize seasonally
                atl3_seas_st = standardize_seasonal(atl3_index)

                # Save the standardized index
                save_path = os.path.join(SAVE_DIR, 'Atl3_st_ts.nc')
                atl3_seas_st.to_netcdf(save_path)
                logger.info(f"ATL3 processing complete. Saved to {save_path}")
                summary.add_result(ProcessingResult('ATL3', ProcessingStatus.SUCCESS, output_path=save_path))
            except Exception as e:
                logger.error(f"Error processing ATL3: {e}", exc_info=True)
                summary.add_result(ProcessingResult('ATL3', ProcessingStatus.FAILED, error_message=str(e)))

            # --- Northern Tropical Atlantic (NTA) Index ---
            logger.info("Processing NTA...")
            try:
                # Define NTA region
                lat_bounds_nta = [6, 18]
                lon_bounds_nta = [-60, -20]

                # Select data and compute area average
                sst_nta = SST_seas.sel(latitude=slice(*lat_bounds_nta), longitude=slice(*lon_bounds_nta))
                nta_index = total_average(sst_nta, 'latitude')
                nta_index.name = 'NTA' # Assign a name

                # Standardize seasonally
                NTA_seas_st = standardize_seasonal(nta_index)

                # Save the standardized index
                save_path = os.path.join(SAVE_DIR, 'NTA_st_ts.nc')
                NTA_seas_st.to_netcdf(save_path)
                logger.info(f"NTA processing complete. Saved to {save_path}")
                summary.add_result(ProcessingResult('NTA', ProcessingStatus.SUCCESS, output_path=save_path))
            except Exception as e:
                logger.error(f"Error processing NTA: {e}", exc_info=True)
                summary.add_result(ProcessingResult('NTA', ProcessingStatus.FAILED, error_message=str(e)))

            # --- Southern Indian Ocean Dipole (SIOD) Index ---
            logger.info("Processing SIOD...")
            try:
                # Define the geographical bounds based on the paper (e.g., Jo et al., 2022)
                # Southcentral IO pole: 65°-85° E, 25°-10° S
                lat_bounds_siod_sc = [-25, -10]
                lon_bounds_siod_sc = [65, 85]
                # Southeastern IO pole: 90°-120° E, 30°-5° S
                lat_bounds_siod_se = [-30, -5]
                lon_bounds_siod_se = [90, 120]

                # Select data for each pole
                sst_siod_sc_pole = SST_seas.sel(latitude=slice(*lat_bounds_siod_sc), longitude=slice(*lon_bounds_siod_sc))
                sst_siod_se_pole = SST_seas.sel(latitude=slice(*lat_bounds_siod_se), longitude=slice(*lon_bounds_siod_se))

                # Compute area-averaged SST anomalies for each pole
                siod_sc_avg = total_average(sst_siod_sc_pole, 'latitude')
                siod_se_avg = total_average(sst_siod_se_pole, 'latitude')

                # Compute the SIOD index (Southcentral minus Southeastern)
                SIOD_seas = siod_sc_avg - siod_se_avg
                SIOD_seas.name = 'SIOD' # Assign a name

                # Standardize the seasonal index
                SIOD_seas_st = standardize_seasonal(SIOD_seas)

                # Save the standardized index
                save_path = os.path.join(SAVE_DIR, 'SIOD_st_ts.nc')
                SIOD_seas_st.to_netcdf(save_path)
                logger.info(f"SIOD processing complete. Saved to {save_path}")
                summary.add_result(ProcessingResult('SIOD', ProcessingStatus.SUCCESS, output_path=save_path))
            except Exception as e:
                logger.error(f"Error processing SIOD: {e}", exc_info=True)
                summary.add_result(ProcessingResult('SIOD', ProcessingStatus.FAILED, error_message=str(e)))

            # --- Western North Pacific (WNP) Index ---
            logger.info("Processing WNP...")
            try:
                # Define WNP region based on Wang et al. (2012), GRL
                lat_bounds_wnp = [18, 28]  # 18°N to 28°N
                lon_bounds_wnp = [122, 132] # 122°E to 132°E

                # Select data for the WNP region
                sst_wnp_region = SST_seas.sel(latitude=slice(*lat_bounds_wnp), longitude=slice(*lon_bounds_wnp))

                # Compute the area-averaged SST anomalies for the region
                wnp_index_seas = total_average(sst_wnp_region, 'latitude')
                wnp_index_seas.name = 'WNP' # Assign a name

                # Standardize the seasonal index
                WNP_seas_st = standardize_seasonal(wnp_index_seas)

                # Save the standardized index
                save_path = os.path.join(SAVE_DIR, 'WNP_st_ts.nc')
                WNP_seas_st.to_netcdf(save_path)
                logger.info(f"WNP processing complete. Saved to {save_path}")
                summary.add_result(ProcessingResult('WNP', ProcessingStatus.SUCCESS, output_path=save_path))
            except Exception as e:
                logger.error(f"Error processing WNP: {e}", exc_info=True)
                summary.add_result(ProcessingResult('WNP', ProcessingStatus.FAILED, error_message=str(e)))

        except FileNotFoundError:
            logger.error(f"Base SST file not found at {sst_filepath}. Cannot process derived indices.")
            for idx in ['SASDI', 'ATL3', 'NTA', 'SIOD', 'WNP']:
                summary.add_result(ProcessingResult(idx, ProcessingStatus.FAILED, error_message="Base SST file not found"))
        except KeyError as e:
            logger.error(f"Variable 'sst' not found in {sst_filepath}. Check the NetCDF file structure. {e}")
            for idx in ['SASDI', 'ATL3', 'NTA', 'SIOD', 'WNP']:
                summary.add_result(ProcessingResult(idx, ProcessingStatus.FAILED, error_message=f"Variable not found: {e}"))
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading or processing SST data: {e}", exc_info=True)
            for idx in ['SASDI', 'ATL3', 'NTA', 'SIOD', 'WNP']:
                summary.add_result(ProcessingResult(idx, ProcessingStatus.FAILED, error_message=str(e)))

    else:
        logger.error(f"Base SST file not found at {sst_filepath}. Cannot process derived indices (SASDI, ATL3, NTA, SIOD, WNP).")
        for idx in ['SASDI', 'ATL3', 'NTA', 'SIOD', 'WNP']:
            summary.add_result(ProcessingResult(idx, ProcessingStatus.FAILED, error_message=f"File not found: {sst_filepath}"))

    # Log summary and exit with appropriate code
    summary.log_summary(logger)
    sys.exit(summary.get_exit_code())


if __name__ == "__main__":
    main()
