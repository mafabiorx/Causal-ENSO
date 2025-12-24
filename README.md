# ENSO-ERA5 Climate Analysis Codebase

## Research Snapshot
This repository contains the workflow behind *South Tropical Atlantic and South Atlantic Convergence Zone: Opportunities for Long-Lead ENSO Diversity Prediction* (Bellacanzone, Bordoni & Shepherd, 2025). The pipeline couples ERA5 reanalysis (1945–2024) with PCMCI+ causal discovery and targeted MCA/REOF searches to isolate South Atlantic–SAMS pathways that precondition ENSO diversity. Reproducing the analysis should recover:
- A JJA(-1) subtropical South Atlantic SST REOF mode that leads Eastern-Pacific ENSO events by six seasons.
- SON–MAM South Atlantic Convergence Zone and Amazon convection modes (via MCA of precipitation and 200 hPa RWS/WAF) supplying 3–5 season precursors.
- ≈0.15 correlation-skill gains when restricting statistical forecasts to PCMCI+-identified parents relative to canonical precursors.

## Repository Layout
```
processing/                 # Seasonal preprocessing pipeline (scripts 1–10)
analysis/                   # Causal discovery, diagnostics, prediction (scripts 12–20)
analysis/precursors_modes/  # Genetic search + MCA/REOF domain optimizers
specialized_modules/        # Experimental toolkits supporting optimization
utils/                      # Shared helpers (paths, predictors, plotting, downloads)
CONTRIBUTING.md             # Scientific standards and contribution guidelines
DATA.md                     # ERA5 data acquisition guide
```

## Environment Setup
1. Ensure `git`, `gcc`, and `proj`/`geos` dependencies needed by `cartopy` are available.
2. Create the UV environment (recommended):
   ```bash
   uv venv TBI_env
   source TBI_env/bin/activate
   uv pip install -r requirements.txt
   ```
   Alternatively, you can use Conda with `environment.yml` if you prefer.
3. For Fortran compiler configuration needed by `windspharm`, ensure `gfortran` is available and set `FFLAGS/FCFLAGS` if your compiler requires legacy flags.

## Data Requirements
- **Source**: ERA5 monthly means (Copernicus Climate Data Store), 1945–2024.
- **Variables**: SST, total precipitation, u/v wind (850 & 200 hPa), geopotential height, streamfunction, Rossby Wave Source, Wave Activity Flux.
- **Processing**: Bilinear regridding to 1°×1°, seasonal averaging, linear detrending, standardisation.
- **Storage**: Place raw inputs under `data/raw/` (outside git). `utils/paths.py` will initialise `data/interim/`, `data/processed/`, and `results/` on import. CDS credentials (`~/.cdsapirc`) are required for scripted downloads via `utils/download_utils.py`.

## Reproducing the Workflow

### Script Execution Order
1. **Seasonal Indices** – Execute the numbered scripts in `processing/` sequentially (e.g., `python processing/1_sf_vp.py` … `python processing/10_save_ds_caus.py`). This assembles the causal-ready dataset and canonical precursor suite.
2. **Causal Discovery** – Run `python analysis/12_PCMCIplus.py` to regenerate the PCMCI+ graphs with forward-in-time conditioning, followed by supporting diagnostics:
   - `12_PCMCIplus.py` – Main PCMCI+ causal discovery
   - `13_PCMCIplus_sliding_window.py` – Temporal stability analysis
3. **Regression & Composites** – Generate spatial patterns and diagnostics:
   - `14_OLS_LASSO_regr_coef_E-C.py` – Regression coefficients
   - `15_OLS_LASSO_regrs_plots.py` – Regression visualizations
   - `16_OLS_LASSO_regrs_plots.py` – Additional regression plots
   - `17_RWS_multi_level_regrs_plots.py` – Multi-level RWS patterns
   - `18_vertical_streamfunction_regrs_plots.py` – Vertical structure
   - `19_Horizontal_Composites.py`, `19_Vertical_Composites.py` – Composite analysis
4. **Prediction & Skill** – Use `python analysis/20_predictive_model_dynamic_sets.py` to refresh forecast experiments comparing canonical vs. PCMCI+ predictor sets.

### Script Numbering Notes
- **Parallel scripts**: Scripts 16 and 19 have paired versions representing complementary analysis approaches that can be run independently.

## Development & Contributions
- Activate the environment (`source TBI_env/bin/activate`) before running scripts and launch utilities with `python -m ...` from the repository root so relative paths resolve.
- Follow the scientific standards and contribution workflow in `CONTRIBUTING.md` (includes validation benchmarks, code conventions, and PR requirements).
- New features should include appropriate testing and documentation. See `CONTRIBUTING.md` for validation requirements.

## License & Citation
The project is released under the MIT License (`LICENSE`). Cite the work as *Bellacanzone, F., Bordoni, S., & Shepherd, T. (2025). ENSO Diversity Prediction via South Atlantic–SAMS Pathways (in review)* until the formal reference and DOI become available.

## Contact
Questions, collaboration, or data access: [Open an issue](https://github.com/mafabiorx/Causal-ENSO/issues)
