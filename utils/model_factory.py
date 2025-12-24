"""
Factory module for creating prediction models with unified interface.
"""

import importlib
import logging
from typing import Any, Dict, Optional, Union, Tuple
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import numpy as np


class TigramiteFitError(Exception):
    """Raised when Tigramite model fitting fails."""
    pass


class TigramitePredictError(Exception):
    """Raised when Tigramite model prediction fails."""
    pass

class ModelFactory:
    """Factory for creating prediction models."""
    
    @staticmethod
    def create_model(model_config: Dict[str, Any], 
                    use_pipeline: bool = True) -> Union[Any, Pipeline]:
        """
        Create a model instance from configuration.
        
        Args:
            model_config: Model configuration dictionary
            use_pipeline: Whether to wrap in sklearn Pipeline
            
        Returns:
            Model instance or Pipeline
        """
        model_type = model_config['type']
        
        if model_type == 'sklearn':
            return ModelFactory._create_sklearn_model(model_config, use_pipeline)
        elif model_type == 'tigramite':
            return ModelFactory._create_tigramite_model(model_config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def _create_sklearn_model(config: Dict[str, Any], 
                             use_pipeline: bool) -> Union[Any, Pipeline]:
        """Create sklearn model with optional pipeline."""
        # Import model class
        module = importlib.import_module(config['module'])
        model_class = getattr(module, config['class'])
        
        # Create model instance
        model = model_class(**config['params'])
        
        # Wrap in pipeline if requested and scaling needed
        if use_pipeline and config.get('requires_scaling', False):
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            return pipeline
        elif use_pipeline:
            # Still wrap in pipeline for consistency
            pipeline = Pipeline([
                ('model', model)
            ])
            return pipeline
        
        return model
    
    @staticmethod
    def _create_tigramite_model(config: Dict[str, Any]) -> Any:
        """Create Tigramite model wrapper."""
        # Create Tigramite model wrapper with sklearn-compatible interface
        # that provides sklearn-like interface
        return TigramiteModelWrapper(config)

    @staticmethod
    def create_model_with_tuning(model_config: Dict[str, Any],
                                inner_cv_splits: int = 3,
                                scoring: str = 'neg_root_mean_squared_error',
                                n_jobs: int = -1) -> Union[GridSearchCV, Pipeline, Any]:
        """
        Create a model with optional hyperparameter tuning.
        
        Args:
            model_config: Model configuration dictionary
            inner_cv_splits: Number of splits for inner CV
            scoring: Scoring metric for GridSearchCV
            n_jobs: Number of parallel jobs
            
        Returns:
            GridSearchCV object if tuning needed, otherwise model/pipeline
        """
        base_model = ModelFactory.create_model(model_config, use_pipeline=True)
        
        # Check if grid search is needed
        if model_config.get('grid_search', False) and 'param_grid' in model_config:
            param_grid = model_config['param_grid'].copy()
            
            # Adjust parameter names for pipeline if needed
            if isinstance(base_model, Pipeline):
                param_grid = adjust_param_grid_for_pipeline(param_grid)
            
            # Create inner CV strategy
            inner_cv = TimeSeriesSplit(n_splits=inner_cv_splits)
            
            # Create GridSearchCV
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring=scoring,
                cv=inner_cv,
                n_jobs=n_jobs,
                refit=True,
                verbose=0
            )
            
            return grid_search
        
        return base_model

class TigramiteModelWrapper:
    """Wrapper to provide sklearn-like interface for Tigramite models."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.dataframe = None
        self.parents = None
        self.target_idx = None
        self.tau_max = None
        self.fitted = False
        
    def fit(self, X, y, **kwargs):
        """Fit method compatible with sklearn interface."""
        # Store training data for Tigramite integration
        self.X_train = X
        self.y_train = y
        self.fitted = True
        
        logging.info("TigramiteModelWrapper: Training data stored. Ready for Tigramite Models integration.")
        return self
    
    def predict(self, X):
        """Predict method compatible with sklearn interface."""
        if not self.fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logging.info("TigramiteModelWrapper: Using simple linear regression fallback for prediction.")
        
        # Fallback to simple linear regression if Tigramite context not available
        from sklearn.linear_model import LinearRegression
        
        if hasattr(self, 'X_train') and hasattr(self, 'y_train'):
            model = LinearRegression()
            model.fit(self.X_train, self.y_train)
            return model.predict(X)
        else:
            logging.warning("TigramiteModelWrapper: No training data available. Returning zeros.")
            return np.zeros(X.shape[0])
    
    def fit_with_tigramite(self, dataframe, target_predictors, tau_max):
        """
        Fit using Tigramite Models class with proper dataframe context.
        
        Args:
            dataframe: Tigramite DataFrame
            target_predictors: Dictionary mapping target indices to predictor lists
            tau_max: Maximum lag for model fitting
        """
        from tigramite.models import Models
        
        try:
            # Create Tigramite Models instance
            models = Models(
                dataframe=dataframe,
                model=self.config.get('class', 'LinearRegression'),
                data_transform=None,
                verbosity=0
            )
            
            # Fit the model
            self.models_instance = models
            self.fit_results = models.fit(
                target_predictors=target_predictors,
                tau_max=tau_max
            )
            self.fitted = True
            self.tigramite_fitted = True
            
            logging.info("TigramiteModelWrapper: Successfully fitted with Tigramite Models class.")
            return self
            
        except Exception as e:
            logging.error(f"TigramiteModelWrapper: Error fitting with Tigramite: {e}")
            raise TigramiteFitError(f"Tigramite fitting failed: {e}") from e
    
    def predict_with_tigramite(self, dataframe_new, target_predictors):
        """
        Predict using Tigramite Models class with proper dataframe context.
        
        Args:
            dataframe_new: New Tigramite DataFrame for prediction
            target_predictors: Dictionary mapping target indices to predictor lists
            
        Returns:
            Dictionary with predictions for each target
        """
        if not hasattr(self, 'tigramite_fitted') or not self.tigramite_fitted:
            raise ValueError("Model must be fitted with Tigramite before prediction")
            
        try:
            predictions = self.models_instance.predict(
                dataframe_new=dataframe_new,
                target_predictors=target_predictors
            )
            return predictions
            
        except Exception as e:
            logging.error(f"TigramiteModelWrapper: Error predicting with Tigramite: {e}")
            raise TigramitePredictError(f"Tigramite prediction failed: {e}") from e
    
    def set_tigramite_params(self, dataframe, parents, target_idx, tau_max):
        """Set Tigramite-specific parameters."""
        self.dataframe = dataframe
        self.parents = parents
        self.target_idx = target_idx
        self.tau_max = tau_max

# Helper function for parameter grid adjustment
def adjust_param_grid_for_pipeline(param_grid: Dict[str, Any], 
                                  pipeline_step_name: str = 'model') -> Dict[str, Any]:
    """
    Adjust parameter grid keys for pipeline compatibility.
    
    Args:
        param_grid: Original parameter grid
        pipeline_step_name: Name of the model step in pipeline
        
    Returns:
        Adjusted parameter grid
    """
    adjusted_grid = {}
    for key, value in param_grid.items():
        # Check if key already has pipeline prefix
        if '__' not in key:
            adjusted_key = f"{pipeline_step_name}__{key}"
        else:
            adjusted_key = key
        adjusted_grid[adjusted_key] = value
    return adjusted_grid