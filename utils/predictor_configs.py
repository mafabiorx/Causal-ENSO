"""
Predictor configuration module for PCMCI+ analysis.

This module defines different predictor sets for causal discovery analysis:
- Known predictors: Well-established ENSO precursors from literature
- New predictors: Novel predictive modes identified through research
- Combined set: All available predictors
- Custom sets: User-defined predictor combinations
"""

import hashlib
import json
from typing import Dict, List, Optional, Union
import logging

# Complete predictor set definitions
PREDICTOR_SETS = {
    'known': {
        'name': 'Known Predictors',
        'description': 'Well-established ENSO precursors from climate science literature',
        'variables': {
            # JJA predictors
            'DMI JJA': 'JJA',
            'PSA2 JJA': 'JJA',
            # SON predictors  
            'SIOD MAM': 'MAM',
            # DJF predictors
            'WNP DJF': 'DJF',
            'SASDI SON': 'SON',
            'Atl3 DJF': 'DJF',
            'NPMM-SST DJF': 'DJF',
            'NPO DJF': 'DJF',
            # MAM mediators
            'NTA MAM': 'MAM',
            'SPO MAM': 'MAM',
            'NPMM-wind MAM': 'MAM',
            'SPMM-SST MAM': 'MAM',
            'SPMM-wind MAM': 'MAM',
            # DJF effects
            'E-ind DJF(1)': 'DJF_effect',
            'C-ind DJF(1)': 'DJF_effect',
        }
    },
    'new': {
        'name': 'Novel Predictors',
        'description': 'New predictive modes identified through research',
        'variables': {
            # JJA predictors
            'REOF SST JJA': 'JJA',
            # SON predictors
            'MCA WAF-RWS SON': 'SON',
            'MCA prec-RWS SON': 'SON',
            # DJF predictors
            'MCA RWS-WAF DJF': 'DJF',
            'MCA RWS-prec DJF': 'DJF',
            # MAM mediators
            'MCA RWS-prec MAM(E)': 'MAM',
            'MCA RWS-prec MAM(C)': 'MAM',
            # DJF effects
            'E-ind DJF(1)': 'DJF_effect',
            'C-ind DJF(1)': 'DJF_effect',
        }
    },
    'atlantic': {
        'name': 'Atlantic Predictors',
        'description': 'ENSO precursors from the Atlantic ocean basin',
        'variables': {
            # JJA predictors
            'REOF SST JJA': 'JJA',
            # SON predictors
            'MCA WAF-RWS SON': 'SON',
            'MCA prec-RWS SON': 'SON',
            # DJF predictors
            'SASDI SON': 'SON',
            'Atl3 DJF': 'DJF',
            'MCA RWS-WAF DJF': 'DJF',
            'MCA RWS-prec DJF': 'DJF',
            # MAM mediators
            'NTA MAM': 'MAM',
            # DJF effects
            'E-ind DJF(1)': 'DJF_effect',
            'C-ind DJF(1)': 'DJF_effect',
        }
    },
    'south_america': {
        'name': 'South american Predictors',
        'description': 'ENSO precursors from the South American (Amazon) region',
        'variables': {
            # MAM mediators
            'MCA RWS-prec MAM(E)': 'MAM',
            'MCA RWS-prec MAM(C)': 'MAM',
            # DJF effects
            'E-ind DJF(1)': 'DJF_effect',
            'C-ind DJF(1)': 'DJF_effect',
        }
    }
}

# Generate combined set programmatically
def _generate_combined_set():
    """Generate combined predictor set from known and new sets."""
    combined_vars = {}
    for set_name in ['known', 'new']:
        combined_vars.update(PREDICTOR_SETS[set_name]['variables'])
    
    return {
        'name': 'All Predictors',
        'description': 'Complete set of all available predictors (known + new)',
        'variables': combined_vars
    }

PREDICTOR_SETS['combined'] = _generate_combined_set()

# Alpha value presets
ALPHA_PRESETS = {
    'mild': {
        'name': 'Mild Significance',
        'description': 'Less stringent alpha values for exploratory analysis',
        'values': [0.1, 0.05, 0.025, 0.01]
    },
    'hard': {
        'name': 'Hard Significance', 
        'description': 'More stringent alpha values for conservative analysis',
        'values': [0.05, 0.025, 0.015, 0.01]
    }
}

# Conditional independence test configurations
COND_IND_TESTS = {
    'robust_parcorr': {
        'name': 'Robust Partial Correlation',
        'class': 'RobustParCorr',
        'module': 'tigramite.independence_tests.robust_parcorr',
        'params': {'significance': 'analytic', 'mask_type': 'y'}
    },
    'gpdc': {
        'name': 'Gaussian Process Distance Correlation',
        'class': 'GPDC',
        'module': 'tigramite.independence_tests.gpdc',
        'params': {'significance': 'analytic', 'mask_type': 'y'}
    }
}

# Season to month mapping
SEASON_MONTHS = {
    'JJA': [7],
    'SON': [10],
    'DJF': [1],
    'MAM': [4],
    'DJF_effect': [1],
}

class PredictorConfig:
    """
    Configuration class for managing predictor sets in PCMCI+ analysis.
    
    Handles predictor set selection, validation, and provides structured
    access to variable configurations.
    """
    
    def __init__(self, predictor_set: str = 'combined', custom_config: Optional[Dict] = None):
        """
        Initialize predictor configuration.
        
        Args:
            predictor_set: One of 'known', 'new', 'combined', or 'custom'
            custom_config: Optional dictionary for custom predictor configuration
        """
        self.predictor_set = predictor_set
        self.custom_config = custom_config
        self.config = self._load_config()
        self._validate_config()
        
    def _load_config(self) -> Dict:
        """Load configuration based on predictor set."""
        if self.predictor_set == 'custom':
            if self.custom_config is None:
                raise ValueError("Custom configuration required when predictor_set='custom'")
            return self.custom_config
        elif self.predictor_set in PREDICTOR_SETS:
            return PREDICTOR_SETS[self.predictor_set]
        else:
            available_sets = list(PREDICTOR_SETS.keys()) + ['custom']
            raise ValueError(f"Unknown predictor set '{self.predictor_set}'. "
                           f"Available sets: {available_sets}")
    
    def _validate_config(self) -> None:
        """Validate the loaded configuration."""
        required_fields = ['name', 'description', 'variables']
        for field in required_fields:
            if field not in self.config:
                raise ValueError(f"Configuration missing required field: {field}")
        
        if not isinstance(self.config['variables'], dict):
            raise ValueError("Configuration 'variables' must be a dictionary")
        
        # Validate seasons
        valid_seasons = set(SEASON_MONTHS.keys())
        for var, season in self.config['variables'].items():
            if season not in valid_seasons:
                raise ValueError(f"Invalid season '{season}' for variable '{var}'. "
                               f"Valid seasons: {valid_seasons}")
    
    def get_variables(self) -> Dict[str, str]:
        """Get variable-season mapping for this configuration."""
        return self.config['variables']
    
    def get_variable_list(self) -> List[str]:
        """Get list of variable names."""
        return list(self.config['variables'].keys())
    
    def get_variable_groups(self) -> Dict[str, List[str]]:
        """
        Generate variable groups based on seasons.
        
        Returns:
            Dictionary with seasonal groups (JJA_PREDICTORS, SON_PREDICTORS, etc.)
        """
        groups = {
            'JJA_PREDICTORS': [],
            'SON_PREDICTORS': [],
            'DJF_CONFOUNDERS': [],
            'MAM_MEDIATORS': [],
            'DJF_EFFECTS': []
        }
        
        for var, season in self.config['variables'].items():
            if season == 'JJA':
                groups['JJA_PREDICTORS'].append(var)
            elif season == 'SON':
                groups['SON_PREDICTORS'].append(var)
            elif season == 'DJF':
                groups['DJF_CONFOUNDERS'].append(var)
            elif season == 'MAM':
                groups['MAM_MEDIATORS'].append(var)
            elif season == 'DJF_effect':
                groups['DJF_EFFECTS'].append(var)
        
        return groups
    
    def get_season_months(self) -> Dict[str, List[int]]:
        """Get season to month mapping."""
        return SEASON_MONTHS
    
    def get_config_hash(self) -> str:
        """Generate hash for custom configurations."""
        if self.predictor_set != 'custom':
            return self.predictor_set
        
        # Create deterministic hash of custom configuration
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    def get_suffix(self, base_suffix: str) -> str:
        """Generate suffix for output files."""
        if self.predictor_set == 'custom':
            return f"{base_suffix}_custom_{self.get_config_hash()}"
        else:
            return f"{base_suffix}_{self.predictor_set}"
    
    def get_metadata(self) -> Dict:
        """Get configuration metadata for output tracking."""
        return {
            'predictor_set': self.predictor_set,
            'name': self.config['name'],
            'description': self.config['description'],
            'variable_count': len(self.config['variables']),
            'variables': self.config['variables'],
            'config_hash': self.get_config_hash()
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return (f"PredictorConfig(set='{self.predictor_set}', "
                f"name='{self.config['name']}', "
                f"variables={len(self.config['variables'])})")
    
    def __repr__(self) -> str:
        return self.__str__()

def load_custom_config_from_file(file_path: str) -> Dict:
    """
    Load custom predictor configuration from JSON file.
    
    Args:
        file_path: Path to JSON configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(file_path, 'r') as f:
        config = json.load(f)
    
    # Validate required structure
    required_fields = ['name', 'description', 'variables']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Configuration file missing required field: {field}")
    
    return config

def get_alpha_preset(preset_name: str) -> List[float]:
    """
    Get alpha values for a given preset.
    
    Args:
        preset_name: Name of the alpha preset ('mild' or 'hard')
        
    Returns:
        List of alpha values
        
    Raises:
        ValueError: If preset_name is unknown
    """
    if preset_name not in ALPHA_PRESETS:
        available_presets = list(ALPHA_PRESETS.keys())
        raise ValueError(f"Unknown alpha preset: {preset_name}. "
                        f"Available presets: {available_presets}")
    return ALPHA_PRESETS[preset_name]['values']

def get_cond_ind_test_config(test_name: str) -> Dict:
    """
    Get configuration for a conditional independence test.
    
    Args:
        test_name: Name of the test ('robust_parcorr' or 'gpdc')
        
    Returns:
        Configuration dictionary for the test
        
    Raises:
        ValueError: If test_name is unknown
    """
    if test_name not in COND_IND_TESTS:
        available_tests = list(COND_IND_TESTS.keys())
        raise ValueError(f"Unknown test: {test_name}. "
                        f"Available tests: {available_tests}")
    return COND_IND_TESTS[test_name]

def get_available_predictor_sets() -> List[str]:
    """Get list of available predictor sets."""
    return list(PREDICTOR_SETS.keys()) + ['custom']

def get_available_alpha_presets() -> List[str]:
    """Get list of available alpha presets."""
    return list(ALPHA_PRESETS.keys())

def get_available_cond_ind_tests() -> List[str]:
    """Get list of available conditional independence tests."""
    return list(COND_IND_TESTS.keys())

def print_predictor_set_info(predictor_set: str = None) -> None:
    """
    Print information about predictor sets.
    
    Args:
        predictor_set: Specific set to show, or None for all sets
    """
    if predictor_set:
        if predictor_set not in PREDICTOR_SETS:
            logging.error(f"Unknown predictor set: {predictor_set}")
            return
        sets_to_show = [predictor_set]
    else:
        sets_to_show = PREDICTOR_SETS.keys()
    
    for set_name in sets_to_show:
        config = PREDICTOR_SETS[set_name]
        print(f"\n{set_name.upper()} SET:")
        print(f"  Name: {config['name']}")
        print(f"  Description: {config['description']}")
        print(f"  Variables ({len(config['variables'])}):")
        
        # Group by season for better display
        by_season = {}
        for var, season in config['variables'].items():
            if season not in by_season:
                by_season[season] = []
            by_season[season].append(var)
        
        for season in ['JJA', 'SON', 'DJF', 'MAM', 'DJF_effect']:
            if season in by_season:
                print(f"    {season}: {by_season[season]}")

if __name__ == "__main__":
    # Example usage and testing
    print("=== PCMCI+ Predictor Configuration ===")
    print_predictor_set_info()
    
    # Test configurations
    print("\n=== Testing Configurations ===")
    for set_name in ['known', 'new', 'combined']:
        config = PredictorConfig(set_name)
        print(f"{set_name}: {config}")
        print(f"  Variables: {len(config.get_variable_list())}")
        print(f"  Groups: {list(config.get_variable_groups().keys())}")