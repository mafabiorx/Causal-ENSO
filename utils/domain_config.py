"""
Centralized Geographic Domain Configuration for Climate Analysis

This module provides standardized geographic domain definitions and time period
configurations used across the analysis pipeline. Centralizing these parameters
ensures consistency and makes it easier to modify analysis domains.

Usage:
    from utils.domain_config import DOMAINS, get_domain, TIME_PERIOD_START, TIME_PERIOD_END

    # Get a specific domain
    domain = get_domain('tropical_pacific_eq')
    sst_region = sst.sel(latitude=slice(domain.lat_min, domain.lat_max),
                         longitude=slice(domain.lon_min, domain.lon_max))

    # Get time period for analysis
    start, end = get_time_period()
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Optional

# === TIME PERIOD CONFIGURATION ===
# Standard analysis period used across all processing scripts
TIME_PERIOD_START = '1945-06-01'
TIME_PERIOD_END = '2024-02-29'


@dataclass(frozen=True)
class GeographicDomain:
    """
    Immutable definition of a geographic analysis domain.

    Attributes:
        name: Short identifier for the domain
        lat_min: Southern boundary (degrees, -90 to 90)
        lat_max: Northern boundary (degrees, -90 to 90)
        lon_min: Western boundary (degrees, varies by convention)
        lon_max: Eastern boundary (degrees, varies by convention)
        description: Human-readable description of the domain
        lon_convention: Longitude convention ('0_360' or '-180_180')
    """
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    description: str
    lon_convention: str = '0_360'  # Default to 0-360 convention

    def to_slice(self) -> Tuple[slice, slice]:
        """Return latitude and longitude slices for xarray selection."""
        return (slice(self.lat_min, self.lat_max),
                slice(self.lon_min, self.lon_max))


# === DOMAIN DEFINITIONS ===
# All geographic domains used in the analysis pipeline

DOMAINS: Dict[str, GeographicDomain] = {
    # === ENSO-related domains ===
    'tropical_pacific_eq': GeographicDomain(
        name='tropical_pacific_eq',
        lat_min=-5, lat_max=5,
        lon_min=170, lon_max=280,
        description='Equatorial Pacific for E/C indices',
        lon_convention='0_360'
    ),
    'cold_tongue': GeographicDomain(
        name='cold_tongue',
        lat_min=-6, lat_max=6,
        lon_min=180, lon_max=270,
        description='Cold Tongue Index region (CTI)',
        lon_convention='0_360'
    ),

    # === Pacific Oscillation domains ===
    'south_pacific_spo': GeographicDomain(
        name='south_pacific_spo',
        lat_min=-45, lat_max=-10,
        lon_min=-160, lon_max=-70,
        description='South Pacific Oscillation domain',
        lon_convention='-180_180'
    ),
    'north_pacific_npo': GeographicDomain(
        name='north_pacific_npo',
        lat_min=20, lat_max=85,
        lon_min=120, lon_max=240,
        description='North Pacific Oscillation domain',
        lon_convention='0_360'
    ),

    # === Meridional Mode domains ===
    'npmm': GeographicDomain(
        name='npmm',
        lat_min=-21, lat_max=32,
        lon_min=175, lon_max=265,
        description='North Pacific Meridional Mode',
        lon_convention='0_360'
    ),
    'spmm': GeographicDomain(
        name='spmm',
        lat_min=-35, lat_max=-10,
        lon_min=180, lon_max=290,
        description='South Pacific Meridional Mode',
        lon_convention='0_360'
    ),

    # === Indian Ocean domains ===
    'dmi_west': GeographicDomain(
        name='dmi_west',
        lat_min=-10, lat_max=10,
        lon_min=50, lon_max=70,
        description='Western Indian Ocean pole for DMI',
        lon_convention='0_360'
    ),
    'dmi_east': GeographicDomain(
        name='dmi_east',
        lat_min=-10, lat_max=0,
        lon_min=90, lon_max=110,
        description='Eastern Indian Ocean pole for DMI',
        lon_convention='0_360'
    ),
    'siod_southcentral': GeographicDomain(
        name='siod_southcentral',
        lat_min=-25, lat_max=-10,
        lon_min=65, lon_max=85,
        description='Southcentral Indian Ocean pole for SIOD',
        lon_convention='0_360'
    ),
    'siod_southeast': GeographicDomain(
        name='siod_southeast',
        lat_min=-30, lat_max=-5,
        lon_min=90, lon_max=120,
        description='Southeastern Indian Ocean pole for SIOD',
        lon_convention='0_360'
    ),

    # === Atlantic domains ===
    'atl3': GeographicDomain(
        name='atl3',
        lat_min=-3, lat_max=3,
        lon_min=-20, lon_max=0,
        description='Atlantic Nino 3 region',
        lon_convention='-180_180'
    ),
    'nta': GeographicDomain(
        name='nta',
        lat_min=6, lat_max=18,
        lon_min=-60, lon_max=-20,
        description='Northern Tropical Atlantic',
        lon_convention='-180_180'
    ),
    'sasdi_sw': GeographicDomain(
        name='sasdi_sw',
        lat_min=-45, lat_max=-35,
        lon_min=-60, lon_max=0,
        description='Southwest pole for SASDI (Ham 2021)',
        lon_convention='-180_180'
    ),
    'sasdi_ne': GeographicDomain(
        name='sasdi_ne',
        lat_min=-30, lat_max=-20,
        lon_min=-40, lon_max=20,
        description='Northeast pole for SASDI (Ham 2021)',
        lon_convention='-180_180'
    ),

    # === Western North Pacific ===
    'wnp': GeographicDomain(
        name='wnp',
        lat_min=18, lat_max=28,
        lon_min=122, lon_max=132,
        description='Western North Pacific (Wang et al., 2012)',
        lon_convention='0_360'
    ),

    # === Pacific-South American domains ===
    'psa_southern_hemisphere': GeographicDomain(
        name='psa_southern_hemisphere',
        lat_min=-70, lat_max=-10,
        lon_min=-180, lon_max=180,
        description='Southern Hemisphere for PSA modes',
        lon_convention='-180_180'
    ),
}


def get_domain(name: str) -> GeographicDomain:
    """
    Retrieve a geographic domain by name.

    Args:
        name: Domain identifier (must be a key in DOMAINS)

    Returns:
        GeographicDomain object with the domain configuration

    Raises:
        KeyError: If the domain name is not found
    """
    if name not in DOMAINS:
        available = ', '.join(sorted(DOMAINS.keys()))
        raise KeyError(f"Domain '{name}' not found. Available domains: {available}")
    return DOMAINS[name]


def get_time_period() -> Tuple[str, str]:
    """
    Get the standard analysis time period.

    Returns:
        Tuple of (start_date, end_date) as strings
    """
    return TIME_PERIOD_START, TIME_PERIOD_END


def list_domains() -> Dict[str, str]:
    """
    List all available domains with their descriptions.

    Returns:
        Dictionary mapping domain names to descriptions
    """
    return {name: domain.description for name, domain in DOMAINS.items()}
