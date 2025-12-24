# Precursor Modes Analysis

This folder contains advanced statistical analysis scripts that identify climate precursor patterns and modes of variability within the ENSO/precipitation system using multivariate techniques.

## Analysis Methods

### 1. Maximum Covariance Analysis (MCA/SVD)

**Scripts:**
- `MCA_RWS_200_WAF.py` - Scalar-vector MCA between Rossby Wave Source (RWS) and Wave Activity Flux (WAF) at 200 hPa
- `MCA_prec_RWS_200.py` - Scalar-scalar MCA between precipitation and RWS at 200 hPa

**Purpose:** MCA identifies coupled patterns of variability between two fields by finding spatial patterns that maximize covariance. This reveals how atmospheric dynamics (RWS/WAF) relate to precipitation patterns.

**Key Features:**
- Regional MCA analysis over irregular domains using polygon vertices
- Seasonal standardization and latitude weighting
- Selection of specific modes (configurable MODE parameter)
- Regression mapping of significant relationships
- Both scalar-scalar and scalar-vector implementations

### 2. Rotated Empirical Orthogonal Functions (REOF)

**Script:**
- `REOF_SST.py` - REOF analysis of Sea Surface Temperature (SST) anomalies

**Purpose:** REOF improves upon standard EOF by rotating the principal components to achieve simpler, more physically interpretable spatial patterns using Varimax rotation.

**Key Features:**
- Regional REOF over irregular domains
- Varimax rotation for enhanced interpretability
- Regression of global SST onto selected rotated PC
- Significance testing at 95% confidence level

## Common Workflow Pattern

1. **Data Loading**: Load seasonal climate data (NetCDF format)
2. **Anomaly Computation**: Calculate seasonal anomalies from climatology
3. **Regional Selection**: Focus analysis on specific climate regions using irregular polygons
4. **Statistical Analysis**: Apply MCA or REOF with latitude weighting
5. **Mode Selection**: Extract specific modes based on explained variance
6. **Regression Analysis**: Map relationships between modes and global fields
7. **Significance Testing**: Apply statistical significance (95% confidence)
8. **Output Generation**: Save NetCDF results and publication-quality plots

## Key Parameters

- **MODE/PC_MODE**: Selects which statistical mode to analyze
- **PLOT_EXTENT**: Geographic bounds for visualization `[lon_min, lon_max, lat_min, lat_max]`
- **vertices**: Irregular polygon defining regional analysis domain
- **SIGNIFICANCE_LEVEL**: Statistical threshold (typically 0.05)

## Output Products

- **NetCDF Files**: Regression maps and statistical metadata
- **PNG Plots**: Cartopy-based geospatial visualizations with:
  - Filled contours for primary variable
  - Vector fields (WAF) or contour lines (RWS) for secondary variables
  - Significance masking
  - Professional cartographic styling

## Climate Science Context

These analyses identify **precursor patterns** - statistical relationships between different climate variables that can indicate developing climate conditions. The focus on atmospheric dynamics (RWS/WAF) and ocean-atmosphere coupling (SST) helps understand the physical mechanisms driving precipitation variability in the study region.