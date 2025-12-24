# Data Acquisition Guide

This guide explains how to obtain and prepare ERA5 reanalysis data for reproducing the ENSO diversity prediction analysis.

**Quick Facts:**
- **Source**: ERA5 monthly means (Copernicus Climate Data Store)
- **Period**: June 1945 – February 2024 (seasonal aggregation)
- **Resolution**: 1°×1° (regridded from native 0.25°×0.25°)
- **Storage**: ~15-20 GB raw data, ~25-30 GB total with outputs
- **Time**: ~8-12 hours download (depending on CDS queue and network)

**Choose your path:**
- 🚀 [Quick Start (Automated)](#quick-start-automated) - Recommended
- 📋 [Manual Download](#manual-download-via-cds-web-interface) - Alternative
- 🔧 [Troubleshooting](#troubleshooting) - Common issues

---

## Prerequisites

### 1. CDS Account Setup (one-time, ~5 min)
1. Register: https://cds.climate.copernicus.eu/user/register
2. Accept Terms & Conditions: https://cds.climate.copernicus.eu/api-how-to
3. Get API credentials: https://cds.climate.copernicus.eu/user

### 2. Configure API Access
Create `~/.cdsapirc` with your credentials:
```bash
cat > ~/.cdsapirc << 'EOF'
url: https://cds.climate.copernicus.eu/api/v2
key: YOUR_UID:YOUR_API_KEY
EOF
chmod 600 ~/.cdsapirc
```

Replace `YOUR_UID:YOUR_API_KEY` with values from Step 3.

### 3. Verify Installation
```bash
python -c "import cdsapi; print('✓ CDS API ready')"
```

---

## Required Variables

The analysis requires **7 ERA5 variables** across 2 datasets:

### Single-Level Dataset (`reanalysis-era5-single-levels-monthly-means`)
| Variable | CDS Name | Used For |
|----------|----------|----------|
| Sea Surface Temperature | `sea_surface_temperature` | ENSO indices, REOF modes |
| Total Precipitation | `total_precipitation` | SACZ/Amazon convection |
| 10m U-wind | `10m_u_component_of_wind` | Surface coupling |
| 10m V-wind | `10m_v_component_of_wind` | Surface coupling |

### Pressure-Level Dataset (`reanalysis-era5-pressure-levels-monthly-means`)
| Variable | CDS Name | Levels | Used For |
|----------|----------|--------|----------|
| U-wind | `u_component_of_wind` | 850, 200 hPa | Atmospheric circulation |
| V-wind | `v_component_of_wind` | 850, 200 hPa | Atmospheric circulation |
| Geopotential | `geopotential` | 200 hPa | Wave patterns |

### Derived Variables (computed by pipeline)
These are **not** downloaded, but calculated in `processing/`:
- **Streamfunction** (from U/V winds via `windspharm`) - Script 1
- **Rossby Wave Source** (from vorticity) - Script 7
- **Wave Activity Flux** (from streamfunction) - Script 7

---

## Quick Start (Automated)

The repository includes robust download utilities with validation and retry logic.

### Step 1: Prepare Environment
```bash
cd /path/to/TBI_repo
source TBI_env/bin/activate  # Created via `uv venv TBI_env` (or `conda activate TBI_env`)
```

### Step 2: Download Single-Level Variables
Create `download_single_level.py`:
```python
from utils.download_utils import ERA5SingleLevelDownloader, DownloadConfig

config = DownloadConfig(
    start_year=1945,
    end_year=2024,
    start_month=6,              # June 1945
    end_month=2,                # February 2024
    variables=[
        'sea_surface_temperature',
        'total_precipitation',
        '10m_u_component_of_wind',
        '10m_v_component_of_wind'
    ],
    grid=[1.0, 1.0],           # 1°×1° resolution
    verify_before_concat=True,  # Validate each file
    max_global_retries=3       # Retry failed downloads
)

downloader = ERA5SingleLevelDownloader(config)
results = downloader.process_all_variables()
print(f"Success: {sum(results.values())}/{len(results)} variables")
```

Run:
```bash
python download_single_level.py
```

**Expected output:**
- Progress logs: `data/raw/logs/download_*.log`
- Downloaded files: `data/raw/[variable]/download_[year]_[variable].nc`
- Concatenated output: `data/raw/concatenated_[variable].nc`

### Step 3: Download Pressure-Level Variables
Create `download_pressure_level.py`:
```python
from utils.download_utils import ERA5MultiLevelDownloader, DownloadConfig

config = DownloadConfig(
    start_year=1945,
    end_year=2024,
    start_month=6,
    end_month=2,
    variables=[
        'u_component_of_wind',
        'v_component_of_wind',
        'geopotential'
    ],
    pressure_levels=['850', '200'],  # hPa
    grid=[1.0, 1.0],
    verify_before_concat=True,
    max_global_retries=3
)

downloader = ERA5MultiLevelDownloader(config)
results = downloader.process_all_variables()
print(f"Success: {sum(results.values())}/{len(results)} variables")
```

**Time estimate:** 8-12 hours total (CDS queuing + download + validation)

**Pro tip:** Run overnight. Downloads resume if interrupted.

---

## Manual Download via CDS Web Interface

If automated download fails or you prefer manual control:

### Single-Level Variables
1. Navigate to: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels-monthly-means
2. Configure request:
   - **Product type**: Monthly averaged reanalysis
   - **Variable**: Select one at a time (SST, precipitation, winds)
   - **Year**: 1945-2024 (select all)
   - **Month**: 01-12 (select all)
   - **Time**: 00:00
   - **Sub-region extraction**:
     - North: 90, West: -180, South: -90, East: 180
     - Grid: 1.0/1.0 (degrees)
   - **Format**: NetCDF
3. Submit → Download when ready
4. Repeat for each variable

### Pressure-Level Variables
1. Navigate to: https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels-monthly-means
2. Configure request:
   - **Pressure level**: 850, 200 hPa
   - (Other settings same as single-level)

### After Manual Download
Place files in expected locations:
```bash
mkdir -p data/raw/seasonal
mv downloaded_sst.nc data/raw/seasonal/SST_seas.nc
mv downloaded_precip.nc data/raw/seasonal/precip_seas.nc
# ... etc for all variables
```

---

## Directory Structure

Paths are managed by `utils/paths.py`. Default structure (auto-created):

```
TBI_repo/
├── data/                          # ⚠️ In .gitignore (not tracked)
│   ├── raw/
│   │   ├── seasonal/              # Final monthly aggregates
│   │   │   ├── SST_seas.nc
│   │   │   ├── precip_seas.nc
│   │   │   └── ...
│   │   ├── logs/                  # Download logs
│   │   └── [variable]/            # Per-variable yearly files (temp)
│   ├── interim/                   # Processing intermediates
│   └── processed/
│       └── causal_dataset.nc      # Final compiled dataset
└── results/                       # ⚠️ In .gitignore
    ├── figures/
    ├── causal_graphs/
    └── predictions/
```

**Key points:**
- `data/` and `results/` are excluded from git (large files)
- Yearly files in `data/raw/[variable]/` are deleted after successful concatenation
- Scripts will create directories automatically

---

## Processing Pipeline

Once raw data is acquired, execute scripts sequentially:

### Phase 1: Streamfunction + velocity potential (Script 1)
```bash
# Compute streamfunction and velocity potential products
python processing/1_sf_vp.py
```
**Output:** Derived wind diagnostics used by later steps

### Phase 2: Climate Indices (Scripts 2-9)
```bash
# ENSO diversity indices (E/C decomposition)
python processing/2_E_C_ts.py

# Extratropical modes (SPO, NPO)
python processing/3_PCA_SPO_NPO_ts.py

# SST-based indices (DMI, SIOD, ATL3, NTA, etc.)
python processing/4_SST-based_ts.py

# Meridional modes (NPMM, SPMM via MCA)
python processing/5_NPMM_n_SPMM_ts.py

# PSA indices
python processing/6_PSA.py

# Rossby Wave Source + Wave Activity Flux
python processing/7_RWS_WAF.py

# Multi-level RWS calculations
python processing/8_RWS_multi_level.py

# Vertical/zonally averaged streamfunction
python processing/9_compute_vert_zon_sf.py
```
**Output:** Standardized seasonal time series for each index

### Phase 3: Dataset Compilation (Script 10)
```bash
python processing/10_save_ds_caus.py
```
**Output:** `data/processed/causal_dataset.nc` (~500 MB)
- All indices aligned to seasonal time steps
- Ready for PCMCI+ causal discovery (Analysis scripts 12-20)

### Verification
```bash
# Check causal dataset structure
ncdump -h data/processed/causal_dataset.nc | grep "UNLIMITED"
# Should show time dimension with ~316 seasons (1945-2024)

# Verify key variables present
ncdump -h data/processed/causal_dataset.nc | grep "float"
# Should list: E_index, C_index, REOF_SST_JJA, MCA_RWS_prec_MAM, etc.
```

**Runtime:** 2-4 hours depending on machine (parallelized where possible)

---

## Troubleshooting

### CDS API Issues

**Error: "Invalid token" or "Unauthorized"**
```bash
# Check API key format
cat ~/.cdsapirc
# Should be: key: 12345:abcd-efgh-1234-5678
# NOT:      key: 12345 abcd-efgh-1234-5678 (no colon)
```

**Error: "Request too large"**
- CDS limits requests to ~120,000 gridpoints × timesteps
- Solution: Download year-by-year (automated downloader does this)

**Error: "Quota exceeded"**
- CDS limits: ~100 GB/day or ~1000 requests/day
- Solution: Wait 24 hours, then retry (state is preserved)

**Slow downloads / stuck "queued"**
- CDS can have 2-6 hour queues during peak times (9am-5pm CET)
- Solution: Run overnight, or use `verify_before_concat=True` to resume

### Processing Issues

**Error: "windspharm not found"**
```bash
# Requires gfortran compiler
sudo apt-get install gfortran build-essential

# Reinstall with legacy flags (Python 3.10-3.11 only)
FFLAGS="-fallow-argument-mismatch -std=legacy" \
FCFLAGS="-fallow-argument-mismatch -std=legacy" \
pip install --no-build-isolation windspharm
```
If `windspharm` build issues appear, verify `gfortran` is installed and set `FFLAGS/FCFLAGS` for legacy Fortran compatibility.

**Error: "NetCDF file is invalid"**
```bash
# Validate file integrity
ncdump -h data/raw/seasonal/SST_seas.nc
# If error: re-download that variable

# Check file size (should be ~300-800 MB per variable)
du -h data/raw/seasonal/*.nc
```

**Memory errors during concatenation**
```bash
# Reduce dask chunk size in processing scripts
# Edit script, find: .chunk({'time': 24})
# Change to:        .chunk({'time': 12})
```

### Getting Help

For download utilities issues:
- Check logs in `data/raw/logs/`
- Review docstrings: `python -c "from utils.download_utils import ERA5Downloader; help(ERA5Downloader)"`
- Open issue with log excerpt

For data format issues:
- Post example with `ncdump -h` output
- Include script that failed and error message

---

## Storage Requirements

- **Raw ERA5 data**: ~15-20 GB (1945-2024, 1°×1° resolution, all variables)
- **Processed indices**: ~500 MB (seasonal time series)
- **Analysis outputs**: ~2-5 GB (spatial regression maps, composites)
- **Total recommended**: ~25-30 GB free space

---

## References

**ERA5 Documentation:**
- Hersbach, H., et al. (2020). The ERA5 global reanalysis. *Quarterly Journal of the Royal Meteorological Society*, 146(730), 1999-2049.
- CDS Knowledge Base: https://confluence.ecmwf.int/display/CKB/ERA5

**Download Utilities:**
- Implemented in `utils/download_utils.py`
- Based on CDS API client: https://github.com/ecmwf/cdsapi
