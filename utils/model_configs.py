"""
Model configuration management for predictive modeling.
Provides presets for various model types and hyperparameter configurations.
"""

from typing import Dict, List, Any, Optional, Union
import json
import logging
from dataclasses import dataclass, field, asdict

# Model type definitions
MODEL_PRESETS = {
    'lassocv': {
        'name': 'LassoCV',
        'type': 'sklearn',
        'module': 'sklearn.linear_model',
        'class': 'LassoCV',
        'params': {
            'cv': 5,
            'random_state': 42,
            'max_iter': 2000
        },
        'requires_scaling': True,
        'grid_search': False
    },
    'linear': {
        'name': 'LinearRegression',
        'type': 'sklearn',
        'module': 'sklearn.linear_model',
        'class': 'LinearRegression',
        'params': {},
        'requires_scaling': False,
        'grid_search': False
    },
    'ridge': {
        'name': 'Ridge',
        'type': 'sklearn',
        'module': 'sklearn.linear_model',
        'class': 'Ridge',
        'params': {
            'random_state': 42
        },
        'requires_scaling': True,
        'grid_search': True,
        'param_grid': {
            'alpha': [0.1, 1.0, 10.0, 100.0]
        }
    },
    'random_forest': {
        'name': 'RandomForestRegressor',
        'type': 'sklearn',
        'module': 'sklearn.ensemble',
        'class': 'RandomForestRegressor',
        'params': {
            'random_state': 42,
            'n_jobs': -1
        },
        'requires_scaling': False,
        'grid_search': True,
        'param_grid': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_leaf': [1, 3, 5],
            'max_features': ['sqrt', 0.5]
        }
    },
    'gradient_boosting': {
        'name': 'GradientBoostingRegressor',
        'type': 'sklearn',
        'module': 'sklearn.ensemble',
        'class': 'GradientBoostingRegressor',
        'params': {
            'random_state': 42,
            'n_estimators': 100
        },
        'requires_scaling': False,
        'grid_search': True,
        'param_grid': {
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 1.0]
        }
    },
    'tigramite_linear': {
        'name': 'TigramiteLinear',
        'type': 'tigramite',
        'class': 'LinearRegression',
        'params': {},
        'requires_scaling': False,
        'grid_search': False
    }
}

# Cross-validation configurations
CV_PRESETS = {
    'quick': {
        'n_splits': 3,
        'inner_cv_splits': 2
    },
    'standard': {
        'n_splits': 5,
        'inner_cv_splits': 3
    },
    'thorough': {
        'n_splits': 10,
        'inner_cv_splits': 5
    }
}

@dataclass
class ModelConfig:
    """Configuration for a prediction model."""
    model_preset: str
    cv_preset: str = 'standard'
    custom_params: Dict[str, Any] = field(default_factory=dict)
    custom_param_grid: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.model_preset not in MODEL_PRESETS:
            raise ValueError(f"Unknown model preset: {self.model_preset}")
        if self.cv_preset not in CV_PRESETS:
            raise ValueError(f"Unknown CV preset: {self.cv_preset}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get complete model information including custom overrides."""
        info = MODEL_PRESETS[self.model_preset].copy()
        
        # Deep copy params and param_grid to avoid modifying presets
        if 'params' in info:
            info['params'] = info['params'].copy()
        if 'param_grid' in info:
            info['param_grid'] = info['param_grid'].copy()
        
        # Apply custom parameters
        if self.custom_params:
            info['params'].update(self.custom_params)
        
        # Apply custom parameter grid
        if self.custom_param_grid and 'param_grid' in info:
            info['param_grid'].update(self.custom_param_grid)
        
        return info
    
    def get_cv_config(self) -> Dict[str, int]:
        """Get cross-validation configuration."""
        return CV_PRESETS[self.cv_preset].copy()
    
    def get_suffix(self, base_suffix: str = '') -> str:
        """Generate a suffix based on model configuration."""
        suffix_parts = [base_suffix] if base_suffix else []
        suffix_parts.append(self.model_preset)
        suffix_parts.append(f"cv{self.cv_preset}")
        return '_'.join(filter(None, suffix_parts))
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get configuration metadata for saving."""
        return {
            'model_preset': self.model_preset,
            'cv_preset': self.cv_preset,
            'model_info': self.get_model_info(),
            'cv_config': self.get_cv_config(),
            'custom_params': self.custom_params,
            'custom_param_grid': self.custom_param_grid
        }

# Helper functions
def get_available_models() -> List[str]:
    """Get list of available model presets."""
    return list(MODEL_PRESETS.keys())

def get_available_cv_presets() -> List[str]:
    """Get list of available CV presets."""
    return list(CV_PRESETS.keys())

def load_custom_model_config(filepath: str) -> Dict[str, Any]:
    """Load custom model configuration from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def print_model_info():
    """Print information about available models."""
    print("\nAvailable Model Presets:")
    print("-" * 50)
    
    for name, config in MODEL_PRESETS.items():
        print(f"\n{name}:")
        print(f"  Full Name: {config['name']}")
        print(f"  Type: {config['type']}")
        print(f"  Requires Scaling: {config['requires_scaling']}")
        print(f"  Grid Search: {config['grid_search']}")
        if config['grid_search'] and 'param_grid' in config:
            print(f"  Hyperparameters: {list(config['param_grid'].keys())}")
    
    print("\n\nAvailable CV Presets:")
    print("-" * 50)
    for name, config in CV_PRESETS.items():
        print(f"\n{name}:")
        print(f"  Outer CV Splits: {config['n_splits']}")
        print(f"  Inner CV Splits: {config['inner_cv_splits']}")