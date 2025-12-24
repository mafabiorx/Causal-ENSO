"""
Centralized seed configuration for reproducibility.

This module provides a single source of truth for random seed values
used across the codebase. All random operations (Python's random module,
NumPy, and scikit-learn) should use these seeds to ensure reproducible results.

Scientific Note:
    Reproducibility is critical for scientific research. This module ensures
    that genetic algorithm optimizations, cross-validation splits, and any
    other stochastic operations produce identical results across runs.
"""

import random
import numpy as np
from typing import Optional
import logging

# Global seed constant - used across all random operations
GLOBAL_SEED = 42

logger = logging.getLogger(__name__)


def set_all_seeds(seed: Optional[int] = None) -> int:
    """
    Set seeds for all random number generators.

    This function sets seeds for both Python's built-in random module
    and NumPy's random generator to ensure reproducibility.

    Parameters
    ----------
    seed : int, optional
        The seed value to use. If None, uses GLOBAL_SEED (42).

    Returns
    -------
    int
        The seed value that was set.

    Example
    -------
    >>> from utils.seed_config import set_all_seeds
    >>> set_all_seeds()  # Uses default GLOBAL_SEED=42
    42
    >>> set_all_seeds(123)  # Uses custom seed
    123
    """
    if seed is None:
        seed = GLOBAL_SEED

    random.seed(seed)
    np.random.seed(seed)

    logger.debug(f"Random seeds set to {seed}")
    return seed


def get_seed() -> int:
    """
    Get the global seed value.

    Returns
    -------
    int
        The global seed constant (42).

    Example
    -------
    >>> from utils.seed_config import get_seed
    >>> seed = get_seed()
    >>> print(seed)
    42
    """
    return GLOBAL_SEED


def get_random_state() -> int:
    """
    Get the random_state parameter for scikit-learn estimators.

    This is an alias for get_seed() but named to match scikit-learn's
    convention for the random_state parameter.

    Returns
    -------
    int
        The random state value (42).

    Example
    -------
    >>> from utils.seed_config import get_random_state
    >>> from sklearn.linear_model import LassoCV
    >>> model = LassoCV(random_state=get_random_state())
    """
    return GLOBAL_SEED
