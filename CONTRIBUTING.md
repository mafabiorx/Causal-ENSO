# Contributing to ENSO-ERA5 Climate Analysis

Thank you for your interest in contributing to this research codebase! This project investigates novel South Atlantic-SAMS pathways that precondition ENSO diversity at 3-6 season leads.

**Before you begin:**
- ⚠️ This is scientific research code. Contributions must maintain **methodological rigor** and **reproducibility**.
- 📊 Changes to analysis algorithms require validation against published benchmarks.
- 🧪 All modifications should be testable and well-documented.

## Quick Navigation
- [Scientific Standards](#scientific-standards) - **Read first**
- [Development Setup](#development-setup)
- [Contribution Workflow](#contribution-workflow)
- [Code Conventions](#code-conventions)
- [Validation Benchmarks](#validation-benchmarks)

---

## Scientific Standards

### Core Principles

**All contributions must be:**

✅ **Scientifically sound**
- Never bypass proper problem understanding with quick fixes
- Identify root causes through systematic analysis
- Maintain statistical validity of algorithms

✅ **Properly validated**
- Use logging and diagnostics to verify assumptions
- Test against known results (see [Validation Benchmarks](#validation-benchmarks))
- Document methodology with literature references

✅ **Reproducible**
- Preserve backwards compatibility where possible
- Document any changes to processing pipeline
- Include instructions for reproducing new results

### Unacceptable Practices

❌ **Never do these:**
- Remove autocorrelations or statistical connections without scientific justification
- Implement patches that hide problems instead of solving them
- Make assumptions about data behavior without verification
- Modify algorithms without understanding their scientific purpose
- Add comments referencing when/why you made changes (e.g., "added this on Oct 1" or "based on script X")

### Required Approach for Bug Fixes

When fixing bugs in analysis code, follow this process:

**1. Understand the algorithm**
- Read docstrings and referenced papers
- Trace data flow through the processing pipeline
- Identify the scientific purpose of the computation

**2. Diagnose the root cause**
```python
# Add diagnostic logging
import logging
logging.info(f"Input shape: {data.shape}, dtype: {data.dtype}")
logging.info(f"Valid range: [{data.min():.3f}, {data.max():.3f}]")

# Visualize intermediate results
import matplotlib.pyplot as plt
plt.plot(data.time, data.values)
plt.savefig('debug_timeseries.png')
```

**3. Propose solution maintaining validity**
- Explain why the fix preserves scientific correctness
- Provide references if methodology changes
- Show before/after comparison

**4. Test thoroughly**
- Verify outputs match expected physical patterns
- Check skill metrics remain stable (±0.05 tolerance)
- Run on subset of years for faster iteration

### Example - Good vs. Bad Fix

❌ **Bad:**
```python
# Remove autocorrelation to fix p-value issue
data = data.diff('time').dropna('time')
```
This destroys temporal information needed for causal inference!

✅ **Good:**
```python
# Account for autocorrelation in significance testing
# Following Bretherton et al. (1999) effective DOF method
from utils.stats_helpers import effective_dof
n_eff = effective_dof(data, method='bretherton')
p_value_corrected = adjust_pvalue(p_value, n=n_eff)
```
This preserves data integrity while addressing statistical issue.

---

## Development Setup

### Prerequisites
- **Python**: 3.10 or 3.11 (3.12+ not supported - see [Why](#why-python-310-311))
- **Fortran compiler**: `gfortran` (required for `windspharm` dependency)
- **Storage**: 30+ GB free for data and outputs
- **RAM**: 16+ GB recommended (8 GB minimum)

### Environment Installation

**Option 1: Using UV (recommended)**
```bash
# Clone repository
git clone https://github.com/mafabiorx/Causal-ENSO.git
cd Causal-ENSO

# Create environment
uv venv TBI_env
source TBI_env/bin/activate
uv pip install -r requirements.txt

# Verify installation
python -c "import xarray, tigramite, windspharm; print('✓ All dependencies ready')"
```

**Option 2: Using pip + virtualenv**
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Option 3: Using Conda**
```bash
conda env create -f environment.yml
conda activate TBI_env
```

If `windspharm` fails to build, ensure `gfortran` is installed and set `FFLAGS/FCFLAGS` for legacy Fortran compatibility.

#### Why Python 3.10-3.11?

The `windspharm` package depends on `pyspharm`, which uses `numpy.distutils` (removed in NumPy 2.0 / Python 3.12+). The package wraps Fortran 77 code from NCAR's SPHEREPACK library for spherical harmonics. Until `pyspharm` migrates to `meson` or `scikit-build-core`, we're constrained to Python 3.10-3.11.

### Data Acquisition

Follow `DATA.md` to download ERA5 reanalysis data. For development:

```bash
# Start the processing pipeline with streamfunction/VP derivation
python processing/1_sf_vp.py
# Full pipeline: run scripts 1-11 sequentially
```

### Verify Setup
```bash
# Verify core dependencies import cleanly
python -c "import xarray, tigramite, windspharm; print('✓ Core dependencies ready')"
```

---

## Contribution Workflow

### Before Starting

**For bug fixes:**
- Check existing issues to avoid duplication
- Reproduce the bug with minimal example
- Identify affected scripts/functions

**For new features:**
- Open an issue to discuss approach
- Explain scientific motivation
- Get maintainer approval before implementing

### Making Changes

**1. Fork and branch**
```bash
# Fork repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/Causal-ENSO.git
cd Causal-ENSO
git remote add upstream https://github.com/mafabiorx/Causal-ENSO.git
git checkout -b fix/descriptive-name  # or feat/descriptive-name
```

**2. Develop with validation**
```bash
# Activate environment
source TBI_env/bin/activate

# Make changes
# ... edit files ...

# Run affected scripts
python processing/2_E_C_ts.py       # if you modified index calculation
python analysis/12_PCMCIplus.py     # if you modified causal discovery

# Check outputs
ls -lh data/processed/causal_dataset.nc  # Should exist and be ~500 MB
```

**3. Validate against benchmarks**
Manually re-run the affected analysis scripts and record the benchmark metrics
listed below (skills, causal effects, and spatial pattern checks). Include the
values in your PR so reviewers can confirm the expected ranges.

### Commit Guidelines

**Format:**
```
<type>(<scope>): <short summary>

<detailed description>

<validation results>
```

**Types:**
- `fix`: Bug fixes
- `feat`: New features
- `refactor`: Code restructuring (no behavior change)
- `docs`: Documentation only
- `perf`: Performance improvements

**Example - Good commit:**
```
fix(preprocessing): Handle descending latitude coordinates

ERA5 datasets sometimes arrive with descending latitude order,
causing spatial alignment issues in downstream analysis. Added
ascending reindex to preprocess_era5_dataset() with explicit
sorting to ensure consistency.

Tested on:
- 1945-2024 SST data (ascending → no change)
- 1979-2023 precip data (descending → corrected)
- Verified E-index correlation: r=0.81 (expected: 0.80±0.05)

Fixes #42
```

**Example - Bad commit:**
```
fixed bug
```

### Pull Request Process

**1. Prepare PR**
```bash
# Sync with upstream
git fetch upstream
git rebase upstream/main

# Push to your fork
git push origin fix/descriptive-name
```

**2. Open PR on GitHub**
Include:
- **Summary**: What changed and why
- **Motivation**: Scientific rationale or bug description
- **Validation**: Benchmark results (recorded from manual runs)
- **Related Issues**: `Fixes #42` or `Closes #17`

**Template:**
```markdown
## Summary
Brief description of changes.

## Motivation
Why this change is needed (scientific or technical).

## Changes Made
- Modified `processing/2_E_C_ts.py` to handle edge case
- Updated `utils/TBI_functions.py` with new helper function
- Added documentation to README

## Validation Results
```
✓ E-index correlation: r=0.804 (expected: 0.80±0.05)
✓ C-index correlation: r=0.751 (expected: 0.75±0.05)  
✓ REOF SST lag-6 effect: ρ=0.497 (expected: 0.50±0.08)
✓ Skill improvement: Δr=0.148 (expected: 0.15±0.03)
```

## Related Issues
Fixes #42
```

**3. Address Review Feedback**
- Respond to comments within 48 hours
- Push additional commits to same branch
- Request re-review when ready

**4. Merge**
- Maintainers will merge after approval
- Delete your branch after merge

---

## Code Conventions

### Style Guide
- **PEP 8** compliance (100-character line limit for readability)
- **Type hints** for function signatures
- **Docstrings** for all public functions (NumPy style)

### Documentation Format

**Function docstrings:**
```python
def compute_seasonal_anomalies(
    data: xr.Dataset,
    clim_period: tuple[int, int] = (1981, 2010)
) -> xr.Dataset:
    """
    Compute seasonal anomalies relative to climatological period.
    
    Parameters
    ----------
    data : xr.Dataset
        Input dataset with 'time' coordinate
    clim_period : tuple of int, default (1981, 2010)
        Start and end years for climatology (inclusive)
    
    Returns
    -------
    xr.Dataset
        Dataset with anomalies computed for each season
    
    Notes
    -----
    Follows methodology in Takahashi et al. (2011) for ENSO diversity
    indices. Detrended linearly before computing climatology to remove
    long-term trends.
    
    References
    ----------
    Takahashi, K., et al. (2011). ENSO regimes: Reinterpreting the
    canonical and Modoki El Niño. Geophys. Res. Lett., 38, L10704.
    """
```

### Code Organization

**Imports:**
```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party (alphabetical)
import numpy as np
import pandas as pd
import xarray as xr

# Local (absolute imports)
from utils.paths import RAW_DATA_DIR, PROCESSED_DATA_DIR
from utils.TBI_functions import detrend_data, standardize_index
```

**Comments:**
```python
# Good: Explain WHY, reference methodology
# Use Bretherton et al. (1999) effective DOF to account for autocorrelation
n_eff = effective_dof(residuals)

# Bad: Explain WHAT (code should be self-explanatory)
# Loop over years
for year in years:
    ...
```

### Error Handling
```python
# Validate inputs
if not isinstance(data, xr.Dataset):
    raise TypeError(f"Expected xr.Dataset, got {type(data).__name__}")

# Handle missing files gracefully
try:
    ds = xr.open_dataset(filepath)
except FileNotFoundError:
    logger.error(f"Data file not found: {filepath}")
    logger.info("Run `python processing/11_save_ds_caus.py` to generate")
    sys.exit(1)
```

### Performance
- Use `dask` for datasets >1 GB
- Chunk spatially (`{'latitude': 'auto', 'longitude': 'auto'}`)
- Profile before optimizing (`python -m cProfile script.py`)

---

## Validation Benchmarks

All changes to analysis code must reproduce these key results within tolerance:

### Skill Metrics (from paper Table 2)
```python
# Combined predictor set (Known + New)
E_index_skill = 0.80 ± 0.05  # Cross-validated correlation
C_index_skill = 0.75 ± 0.05

# Skill improvement over Known-only
Delta_E = 0.15 ± 0.03
Delta_C = 0.15 ± 0.03
```

### Causal Effects (from paper Figures 2c, 3c)
```python
# Key pathways (RPC, FDR-corrected)
REOF_SST_JJA_to_E_lag6 = 0.50 ± 0.08      # Strongest long-lead
MCA_RWS_prec_MAM_to_E_lag3 = 0.55 ± 0.08  # Strongest short-lead
NPMM_wind_MAM_to_C_lag3 = 0.40 ± 0.08     # Known predictor baseline
```

### Spatial Patterns (qualitative)
- **REOF SST JJA mode 4**: Subtropical dipole at 35°S-22°S
- **MCA precipitation MAM**: SACZ enhancement, SESA drying
- **Composite El Niño SST**: Peak warming 5°N-5°S, 170°W-90°W

### Running Validation

**Manual check:**
```bash
# Re-run causal discovery
python analysis/12_PCMCIplus.py

# Compare output to paper Figure 2c
# Check: results/causal_graphs/pcmci_rpc_fdr.pdf
```

**Tolerance rationale:**
- ±0.05 for skill: Accounts for cross-validation fold variance
- ±0.08 for causal effects: Accounts for bootstrap uncertainty
- ±0.10 spatial correlation: Accounts for regridding differences

---

## Questions or Issues?

- **Data acquisition**: See `DATA.md`
- **Environment setup**: See `README.md`
- **Bug reports**: Open an issue with reproducible example
- **Feature requests**: Open an issue describing scientific motivation
- **General questions**: Check existing issues or open a discussion

Thank you for contributing to advancing ENSO prediction science!
