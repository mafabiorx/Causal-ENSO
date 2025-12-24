"""
Atmospheric Dynamics: Rossby Wave Source and Wave Activity Flux Computation
===========================================================================

This module computes Rossby Wave Source (RWS) and Wave Activity Flux (WAF) at 200 hPa
following established atmospheric dynamics theory to quantify wave generation and
propagation mechanisms essential for understanding climate teleconnections. The analysis
focuses on tropical-extratropical interactions and their role in ENSO teleconnection
pathways and global climate predictability.

Scientific Domain and Context
----------------------------
Atmospheric Rossby waves are fundamental to global climate teleconnections, carrying
energy and momentum from tropical heat sources to extratropical regions. Understanding
wave generation (RWS) and propagation (WAF) is crucial for:

- **ENSO Teleconnections**: How tropical Pacific heating generates global responses
- **Monsoon Dynamics**: Wave propagation affecting regional circulation patterns
- **Climate Predictability**: Long-range forecast skill through wave-mediated connections
- **Extreme Events**: Wave train patterns influencing regional weather extremes

The 200 hPa level represents the upper tropospheric waveguide where Rossby waves
propagate most efficiently, making it optimal for teleconnection analysis.

Core Functionality
-----------------
**Rossby Wave Source (RWS) Computation**: Quantifies vorticity generation through
diabatic heating and vorticity advection following Sardeshmukh & Hoskins (1988)

**Wave Activity Flux (WAF) Analysis**: Computes energy propagation vectors using
Takaya & Nakamura (2001) formulation for stationary wave activity flux

**Zonal Anomaly Analysis**: Removes zonal mean to focus on wave-like perturbations
and teleconnection patterns rather than zonal mean circulation changes

Physical Background
------------------

**Rossby Wave Source (RWS) Theory**:
Mathematical formulation: S = -η*D - ∇·(u_χ*∇η)
where:
- η: absolute vorticity (planetary + relative vorticity)
- D: divergence field (linked to diabatic heating through mass continuity)
- u_χ: irrotational (divergent) wind component
- ∇: horizontal gradient operator

Physical interpretation: RWS quantifies how diabatic heating (through divergence)
and vorticity advection by divergent flow generate Rossby wave sources. Positive
RWS indicates wave generation, with tropical convection being a primary source.

**Wave Activity Flux (WAF) Theory**:
Based on Takaya & Nakamura (2001) formulation for stationary wave activity flux:
WAF = (ρ₀/2|U|) * {U(ψ_x² - ψψ_xx) + V(ψ_xψ_y - ψψ_xy);
                    U(ψ_xψ_y - ψψ_xy) + V(ψ_y² - ψψ_yy)}

where:
- ρ₀: reference density
- U, V: basic state zonal and meridional winds
- ψ: streamfunction anomaly
- Subscripts denote partial derivatives

Physical interpretation: WAF vectors point in the direction of Rossby wave energy
propagation, enabling tracking of teleconnection pathways from source to response
regions. Great circle propagation characterizes planetary-scale wave trains.

Teleconnection Applications
--------------------------
**ENSO-Global Connections**: El Niño/La Niña heating anomalies generate RWS over
the tropical Pacific, launching wave trains that propagate to extratropical regions
affecting North American, European, and Asian climate patterns.

**Seasonal Prediction**: Understanding RWS generation and WAF propagation pathways
provides physical basis for seasonal climate prediction, particularly for
temperature and precipitation patterns in downstream regions.

**Climate Change Context**: Changes in tropical heating patterns and basic state
winds affect both wave generation and propagation characteristics, influencing
future teleconnection strength and patterns.

Integration with Research Workflow
---------------------------------
This atmospheric dynamics analysis provides crucial process understanding for:
- Validation of causal pathways identified through PCMCI+ analysis
- Physical interpretation of statistical teleconnections
- Mechanistic support for ENSO diversity prediction improvements

The RWS and WAF fields serve as process diagnostics linking tropical forcing
to extratropical responses in the South American monsoon - ENSO pathway analysis.

Dependencies and Performance
---------------------------
Requires metpy for atmospheric calculations, cartopy for map projections, and
xarray for efficient handling of 3D atmospheric fields. Memory requirements
scale with horizontal resolution (~500MB for 1°x1° global fields). Optimized
for seasonal-mean diagnostics with proper handling of spherical geometry.
"""

import os, glob, sys, time
from pathlib import Path
import numpy as np
import xarray as xr
from tqdm import tqdm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature  # for coastlines and borders
from windspharm.xarray import VectorWind
# Add the project root directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))
from utils.TBI_functions import long_0_360_to_180_180, get_symmetric_levels
from utils.paths import get_data_path, get_results_path


# -----------------------------------------------------------------------------
# 1. Setup Directories and Global Parameters
# -----------------------------------------------------------------------------
# Input directory for seasonal, detrended fields
INPUT_DIR = get_data_path('seasonal/', data_type="raw")
# Temporary folder for chunk files (reuse the same one as before)
OUTPUT_DIR = os.path.join(INPUT_DIR, 'temp_files/')
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Directory for plots
PLOT_DIR = get_results_path('main_climate_modes/')

N_CHUNKS = 20
START_CHUNK = 0  # Useful if the script execution is restarted

# -----------------------------------------------------------------------------
# 2. RWS COMPUTATION AT 200 hPa (with zonal mean removed)
# -----------------------------------------------------------------------------
print("[INFO] Starting RWS computation at 200 hPa (zonal mean removed)...")

# Load the U and V wind components from the 200 hPa files
U_200 = xr.open_dataset(os.path.join(INPUT_DIR, 'U_200_seas.nc'))['u']
U_200 = long_0_360_to_180_180(U_200)
# Remove the zonal (longitude) mean to isolate eddy components
U_200 = U_200 - U_200.mean(dim='longitude')

V_200 = xr.open_dataset(os.path.join(INPUT_DIR, 'V_200_seas.nc'))['v']
V_200 = long_0_360_to_180_180(V_200)
V_200 = V_200 - V_200.mean(dim='longitude')

# Split the time axis into N_CHUNKS chunks for memory efficiency
chunks = np.array_split(np.arange(len(U_200.time)), N_CHUNKS)

# Write out each chunk for U and V
for i, chunk in tqdm(enumerate(chunks[START_CHUNK:], start=START_CHUNK), total=N_CHUNKS-START_CHUNK):
    U_chunk = U_200.isel(time=chunk)
    V_chunk = V_200.isel(time=chunk)
    U_chunk.to_netcdf(os.path.join(OUTPUT_DIR, f'U_chunk_{i}.nc'))
    V_chunk.to_netcdf(os.path.join(OUTPUT_DIR, f'V_chunk_{i}.nc'))
    U_chunk.close()
    V_chunk.close()
    del U_chunk, V_chunk
    time.sleep(3)

U_200.close()
V_200.close()
del U_200, V_200

# Gather input chunk files for U and V
u_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "U_chunk_*.nc")))
v_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "V_chunk_*.nc")))
if len(u_files) == 0 or len(v_files) == 0:
    raise FileNotFoundError("No U or V wind chunk files found. Check file naming or path.")
if len(u_files) != len(v_files):
    raise ValueError("Mismatch between number of U and V files. Ensure proper pairing based on time period.")

# Process each chunk to compute the Rossby Wave Source (RWS)
temp_files = []  # store paths of temporary RWS chunk outputs
for u_file, v_file in zip(u_files, v_files):
    ds_u = xr.open_dataset(u_file)
    ds_v = xr.open_dataset(v_file)
    # Extract the single variable from each dataset
    u_var = list(ds_u.data_vars)[0]
    v_var = list(ds_v.data_vars)[0]
    u_da = ds_u[u_var]
    v_da = ds_v[v_var]
    # Create VectorWind instance for differentiation
    w = VectorWind(u_da, v_da)
    # Compute diagnostic fields
    eta   = w.absolutevorticity()    # absolute vorticity (η)
    div   = w.divergence()           # horizontal divergence (D)
    uchi, vchi = w.irrotationalcomponent()   # divergent wind components
    etax, etay = w.gradient(eta)     # gradients of absolute vorticity

    # Compute Rossby Wave Source: S = -η * D - (u_chi * ∂η/∂x + v_chi * ∂η/∂y)
    RWS = -eta * div - (uchi * etax + vchi * etay)
    RWS.name = "RWS"
    RWS.attrs["long_name"] = "Rossby Wave Source (computed on fields with zonal mean removed)"
    RWS.attrs["units"] = "s^-2"
    # Save this chunk’s RWS to a temporary NetCDF file
    chunk_id = os.path.basename(u_file)
    out_name = chunk_id.replace("U", "RWS").replace(".nc", "_RWS.nc")
    out_path = os.path.join(OUTPUT_DIR, out_name)
    RWS.to_netcdf(out_path)
    temp_files.append(out_path)
    ds_u.close()
    ds_v.close()

# Concatenate all temporary RWS files along the time dimension and save final RWS file
if not temp_files:
    raise RuntimeError("No temporary RWS files were produced. Check earlier steps.")

combined = xr.open_mfdataset(temp_files, combine='by_coords')
final_rws_path = os.path.join(INPUT_DIR, "RWS_200_seas_zonal_removed.nc")
combined.to_netcdf(final_rws_path)
combined.close()
print(f"[INFO] RWS field saved: {final_rws_path}")

# (Clean up temporary RWS chunk files)
for fpath in temp_files:
    try:
        os.remove(fpath)
    except OSError as e:
        print(f"Error removing file {fpath}: {e}")


# Cleanup: Remove temporary U/V chunk files
uv_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "U_chunk_*.nc"))) + \
           sorted(glob.glob(os.path.join(OUTPUT_DIR, "V_chunk_*.nc")))
for f in uv_files:
    try:
        os.remove(f)
    except OSError as e:
        print(f"Error removing temporary file {f}: {e}")

# Cleanup: Remove temporary RWS chunk files
for f in temp_files:
    try:
        os.remove(f)
    except OSError as e:
        print(f"Error removing temporary file {f}: {e}")


# --- RWS Climatology Visualization ---
print("[INFO] Generating RWS seasonal climatology plot ...")
final_rws_path = os.path.join(INPUT_DIR, "RWS_200_seas_zonal_removed.nc")
RWS = xr.open_dataset(final_rws_path)['RWS']
RWS_seas_clim = RWS.groupby('time.season').mean('time')
seasons = RWS_seas_clim.season.values

fig, axs = plt.subplots(4, 1, figsize=(20, len(seasons)*5), subplot_kw={'projection': ccrs.PlateCarree()})
for i, season in enumerate(seasons):
    to_plot = RWS_seas_clim.sel(season=season)
    ax = axs[i]
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.coastlines(linewidth=0.8)
    gl = ax.gridlines(draw_labels=True)
    gl.top_labels = False
    gl.right_labels = False
    clevs, lev_cont = get_symmetric_levels(to_plot, spacing=5e-10, factor=10)
    cf = to_plot.plot.contourf(ax=ax, transform=ccrs.PlateCarree(), levels=clevs,
                               cmap='RdBu_r', extend='both', add_colorbar=False)
    #cs = to_plot.plot.contour(ax=ax, levels=lev_cont, colors='k')
    ax.set_title(season, fontsize=16)
plt.subplots_adjust(left=0.04, bottom=0.04, right=0.96, top=0.95, hspace=0.12)
cax = fig.add_axes([0.3, 0.02, 0.4, 0.01])
plt.colorbar(cf, cax=cax, orientation='horizontal', shrink=0.1, label='RWS [s$^{-2}$]')
plt.suptitle('RWS Climatology at 200 hPa (zonal mean removed)', fontsize=20)
rws_plot_path = os.path.join(PLOT_DIR, 'RWS_climatology_zonal_removed.png')
plt.savefig(rws_plot_path, bbox_inches='tight', pad_inches=0.1, dpi=300)
plt.close(fig)
print(f"[INFO] RWS climatology plot saved: {rws_plot_path}")

# -----------------------------------------------------------------------------
# 3. WAF COMPUTATION AT 200 hPa (with time-mean anomaly subtraction)
# -----------------------------------------------------------------------------

print("[INFO] Starting WAF computation at 200 hPa (with time-mean anomalies)...")

# File paths for U, V, and Z at 200 hPa
u_file = os.path.join(INPUT_DIR, "U_200_seas.nc")
v_file = os.path.join(INPUT_DIR, "V_200_seas.nc")
z_file = os.path.join(INPUT_DIR, "Z_200_seas.nc")

print("[INFO] Loading U, V, Z data from NetCDF with chunking...")

# Open datasets
ds_u = xr.open_dataset(u_file)
ds_v = xr.open_dataset(v_file)
ds_z = xr.open_dataset(z_file)

# Convert longitudes for U and V; for Z, drop any pressure_level variable and squeeze
ds_u = long_0_360_to_180_180(ds_u)
ds_v = long_0_360_to_180_180(ds_v)
ds_z = ds_z.drop_vars('pressure_level', errors='ignore').squeeze()

# Merge datasets; assume variable names: 'u', 'v', 'z'
ds = xr.merge([ds_u, ds_v, ds_z])

# Remove the zonal mean from each field (background removal to get the eddy component)
ds['u'] = ds['u'] - ds['u'].mean(dim='longitude')
ds['v'] = ds['v'] - ds['v'].mean(dim='longitude')
ds['z'] = ds['z'] - ds['z'].mean(dim='longitude')

# Apply chunking to latitude and longitude for heavy differentiation steps
ds = ds.chunk({'latitude': 32, 'longitude': 32})
print(ds)
print("[INFO] Data loaded and chunked successfully.\n")

# --- Compute Climatology and Anomalies ---
print("[INFO] Computing time-mean fields and anomalies for U, V, Z...")
U_clim = ds['u'].mean(dim='time')
V_clim = ds['v'].mean(dim='time')
Z_clim = ds['z'].mean(dim='time')

U_prime = ds['u'] - U_clim
V_prime = ds['v'] - V_clim
Z_prime = ds['z'] - Z_clim

# --- Compute Streamfunction Anomaly (psi') from geopotential anomaly ---
print("[INFO] Computing streamfunction anomaly from geopotential anomaly...")
Omega = 7.292e-5  # Earth's angular velocity (s^-1)
lat_rad = np.deg2rad(ds['latitude'])
f = 2.0 * Omega * np.sin(lat_rad)  # Coriolis parameter
# Log statistics for Coriolis parameter f
f_abs = np.abs(f.values)
EPS = 1e-6
f_b = f.broadcast_like(Z_prime)
psi = (Z_prime / f_b).where(np.abs(f_b) > EPS)

print("[INFO] Differentiating streamfunction anomaly with respect to longitude and latitude...")

deg2rad = np.pi / 180.0
dpsi_dlon = psi.differentiate('longitude') * deg2rad
dpsi_dlat = psi.differentiate('latitude') * deg2rad
d2psi_dlon2    = dpsi_dlon.differentiate('longitude') * deg2rad
d2psi_dlat2    = dpsi_dlat.differentiate('latitude') * deg2rad
d2psi_dlondlat = dpsi_dlat.differentiate('longitude') * deg2rad

# Compute intermediate terms for the flux formulation
termxu = (dpsi_dlon**2) - (psi * d2psi_dlon2)
termxv = (dpsi_dlon * dpsi_dlat) - (psi * d2psi_dlondlat)
termyv = (dpsi_dlat**2) - (psi * d2psi_dlat2)

# --- Compute the Horizontal Wave Activity Flux (WAF) ---
print("[INFO] Computing WAF components using Takaya & Nakamura (2001) formulation...")

# Basic state wind magnitude from the time-mean winds
wind_mag = np.sqrt(U_clim**2 + V_clim**2)
# Log statistics for wind magnitude
small_wind_count = int((wind_mag < 1e-4).sum().values)
# Create a safe wind magnitude to avoid division by zero:
wind_thresh = 1e-4
wind_mag_safe = xr.where(wind_mag < wind_thresh, wind_thresh, wind_mag)
coslat = np.cos(lat_rad)

a = 6.371e6    # Earth radius in meters
p = 200.0 / 1000.0   # 200 hPa level as fraction of 1000 hPa

coeff = (p * coslat) / (2.0 * wind_mag_safe)

F_x = (coeff / (a * a * coslat)) * ((U_clim/coslat) * termxu + V_clim * termxv)
F_y = (coeff / (a * a)) * ((U_clim/coslat) * termxv + V_clim * termyv)

F_x.name = 'WAFx'
F_y.name = 'WAFy'
F_x.attrs['long_name'] = "Wave Activity Flux (zonal component) on fields with time anomalies"
F_y.attrs['long_name'] = "Wave Activity Flux (meridional component) on fields with time anomalies"
F_x.attrs['units'] = "m^2 s^-2"
F_y.attrs['units'] = "m^2 s^-2"

ds_out = xr.Dataset({'WAFx': F_x, 'WAFy': F_y})
ds_out = ds_out.compute()
print("[INFO] WAF computation completed.\n")

# Save the WAF components to NetCDF with a descriptive filename.
waf_nc_path = os.path.join(INPUT_DIR, "WAF_200_components_zonal_removed.nc")
print(f"[INFO] Saving WAF fields to {waf_nc_path}")
ds_out.to_netcdf(waf_nc_path)
print("[INFO] WAF NetCDF saved.\n")

# -----------------------------------------------------------------------------
# 4. Visualization: Plot WAF Magnitude and Streamlines (with original specs)
# -----------------------------------------------------------------------------

waf_nc_path = os.path.join(INPUT_DIR, "WAF_200_components_zonal_removed.nc")
WAF = xr.open_dataset(waf_nc_path)

def create_equatorial_taper_mask(latitude, equator_width=5, transition_width=5):
    """
    Create a weight mask that gradually tapers values near the equator.
    """
    import numpy as np
    import xarray as xr
    
    # Create a template array with same shape/coords as input
    result = xr.ones_like(latitude)
    
    # Get absolute latitude values (distance from equator)
    abs_lat = np.abs(latitude.values)
    
    # Create mask array
    weights = np.ones_like(abs_lat)
    
    # Set weights to 0 within equator_width
    weights[abs_lat < equator_width] = 0
    
    # Calculate transition zone weights
    transition_mask = (abs_lat >= equator_width) & (abs_lat < (equator_width + transition_width))
    if np.any(transition_mask):
        # Linear interpolation in transition zone
        weights[transition_mask] = (abs_lat[transition_mask] - equator_width) / transition_width
    
    # Assign values to the result array
    result.values = weights
    
    return result

def add_equatorial_band(ax, lat_band=5, color='lightgray', alpha=0.2):
    """Add a shaded band indicating the equatorial region."""
    import matplotlib.patches as mpatches
    import cartopy.crs as ccrs
    
    # Create rectangle spanning all longitudes in the equatorial band
    rect = mpatches.Rectangle(
        xy=(-180, -lat_band), width=360, height=2*lat_band,
        transform=ccrs.PlateCarree(), facecolor=color, alpha=alpha,
        edgecolor='none', zorder=0
    )
    ax.add_patch(rect)
    
    # Add subtle dotted lines at the edges of the band
    ax.axhline(y=lat_band, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)
    ax.axhline(y=-lat_band, color='gray', linestyle=':', linewidth=0.8, alpha=0.5)

def waf_visualization(ds_plot, plot_path, equator_width=5, transition_width=5):
    """
    Create an improved WAF visualization with better equatorial handling.
    
    Parameters:
    -----------
    ds_plot : xarray.Dataset
        Dataset containing WAFx and WAFy components
    plot_path : str
        Path to save the plot
    equator_width : float
        Width of equatorial region to shade (degrees)
    transition_width : float
        Width of transition zone for tapering (degrees)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    
    # Apply equatorial tapering to WAF components
    taper_mask = create_equatorial_taper_mask(
        ds_plot.latitude, 
        equator_width=equator_width, 
        transition_width=transition_width
    )
    
    # Create tapered versions for visualization
    WAFx_tapered = ds_plot['WAFx'] * taper_mask
    WAFy_tapered = ds_plot['WAFy'] * taper_mask
    
    # Compute WAF magnitude from tapered components
    WAF_mag = np.sqrt(WAFx_tapered**2 + WAFy_tapered**2)
    
    # Adaptive scaling for line widths
    max_WAF = WAF_mag.max()
    if np.isnan(max_WAF) or max_WAF < 1e-10:
        max_WAF = 1.0  # Fallback if max is too small or NaN
    
    lw = 3 * WAF_mag / max_WAF
    
    # Create the plot
    fig = plt.figure(figsize=(12, 8))
    proj = ccrs.PlateCarree()
    ax = plt.axes(projection=proj)
    
    # Set map extent and add features
    ax.set_extent([-180, 180, -60, 60], crs=proj)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    
    # Add equatorial band shading
    add_equatorial_band(ax, lat_band=equator_width, color='lightgray', alpha=0.2)
    
    # Plot WAF magnitude as filled contours
    cf = ax.contourf(
        ds_plot['longitude'], 
        ds_plot['latitude'], 
        WAF_mag,
        levels=20, 
        cmap='viridis', 
        extend='max', 
        transform=proj
    )
    
    # Add streamlines with varying linewidth
    ax.streamplot(
        ds_plot['longitude'], 
        ds_plot['latitude'],
        WAFx_tapered.values, 
        WAFy_tapered.values,
        density=1.2, 
        color='white',  # White streamlines are more visible on colormap
        linewidth=lw.values, 
        arrowsize=1.5,
        transform=proj
    )
    
    # Add a second streamplot for just the near-equatorial region with fixed width
    # This provides context for wave propagation near equator where values are tapered
    near_equator = (np.abs(ds_plot.latitude) <= (equator_width + transition_width))
    if np.any(near_equator):
        try:
            ax.streamplot(
                ds_plot['longitude'], 
                ds_plot['latitude'].where(near_equator, drop=True),
                ds_plot['WAFx'].where(near_equator, drop=True).values, 
                ds_plot['WAFy'].where(near_equator, drop=True).values,
                density=0.8, 
                color='gold',  # Distinct color for equatorial region
                linewidth=0.5,  # Thin lines indicate lower confidence
                arrowsize=1.0,
                transform=proj,
                zorder=5
            )
        except (ValueError, TypeError) as e:
            print(f"Note: Could not plot equatorial streamlines: {e}")
    
    # Add colorbar and annotations
    cbar = plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.07, shrink=0.8)
    cbar.set_label('WAF Magnitude (m² s⁻²)')
    
    # Add annotations explaining equatorial handling
    ax.text(
        0.5, 0.02, 
        f"Note: WAF values tapered within {equator_width}° of equator with {transition_width}° transition",
        transform=ax.transAxes, 
        ha='center', 
        fontsize=9,
        bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5')
    )
    
    # Add title
    time_val = str(ds_plot['time'].values) if 'time' in ds_plot.coords else "instantaneous"
    title_txt = "Wave Activity Flux (Takaya & Nakamura, 2001) @ 200 hPa\n"
    title_txt += "(zonal mean removed with equatorial tapering)"
    if time_val != "instantaneous":
        title_txt += f"\n{time_val[:10]}"
    ax.set_title(title_txt, fontsize=14)
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linestyle=':', color='gray', alpha=0.6)
    gl.top_labels = False
    gl.right_labels = False
    
    # Save figure
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"[INFO] Enhanced WAF plot saved: {plot_path}")
    
    # Return data for potential further analysis
    return {
        'WAFx_tapered': WAFx_tapered, 
        'WAFy_tapered': WAFy_tapered, 
        'WAF_mag': WAF_mag
    }

print("[INFO] Creating WAF visualization...")
ds_plot = WAF.isel(time=-1).sel(latitude=slice(-50, 50))
plot_path = os.path.join(PLOT_DIR, "WAF_streamlines_climatology.png")
waf_visualization(ds_plot, plot_path, equator_width=5, transition_width=5)