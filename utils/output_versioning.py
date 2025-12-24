"""
Output versioning utilities for reproducible scientific outputs.

This module provides infrastructure for versioned output filenames and
accompanying metadata files. Versioning is opt-in via the --versioned flag
to maintain backward compatibility with existing workflows.

Scientific Note:
    Proper output versioning is critical for scientific reproducibility.
    Each output file can be traced back to the exact parameters and
    timestamp that generated it, enabling audit trails and preventing
    accidental data loss through overwriting.
"""

import os
import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


def get_git_hash() -> Optional[str]:
    """
    Get the current git commit hash.

    Returns
    -------
    str or None
        The short git hash if in a git repository, None otherwise.
    """
    try:
        result = subprocess.run(
            ['git', 'rev-parse', '--short', 'HEAD'],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).parent.parent
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@dataclass
class OutputMetadata:
    """
    Metadata for a versioned output file.

    Attributes
    ----------
    script_name : str
        Name of the script that generated the output.
    output_path : str
        Path to the output file.
    timestamp : datetime
        When the output was generated.
    parameters : dict
        Parameters used to generate the output.
    git_hash : str or None
        Git commit hash at time of generation.
    data_shape : dict or None
        Shape of the data (if applicable).
    """
    script_name: str
    output_path: str
    timestamp: datetime = field(default_factory=datetime.now)
    parameters: Dict[str, Any] = field(default_factory=dict)
    git_hash: Optional[str] = field(default_factory=get_git_hash)
    data_shape: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary for JSON serialization."""
        return {
            'script_name': self.script_name,
            'output_path': self.output_path,
            'timestamp': self.timestamp.isoformat(),
            'parameters': self.parameters,
            'git_hash': self.git_hash,
            'data_shape': self.data_shape
        }

    def save(self, metadata_path: Optional[str] = None) -> str:
        """
        Save metadata to a JSON file.

        Parameters
        ----------
        metadata_path : str, optional
            Path for the metadata file. If not provided, derives from output_path.

        Returns
        -------
        str
            Path to the saved metadata file.
        """
        if metadata_path is None:
            # Replace extension with _metadata.json
            base = os.path.splitext(self.output_path)[0]
            metadata_path = f"{base}_metadata.json"

        with open(metadata_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

        logger.debug(f"Metadata saved to {metadata_path}")
        return metadata_path


class OutputRegistry:
    """
    Registry for tracking all outputs from processing sessions.

    This provides a centralized manifest of all generated outputs,
    enabling audit trails and reproducibility tracking.

    Attributes
    ----------
    registry_path : str
        Path to the registry JSON file.
    entries : list
        List of output metadata entries.
    """

    def __init__(self, registry_path: str):
        """
        Initialize the output registry.

        Parameters
        ----------
        registry_path : str
            Path to the registry JSON file.
        """
        self.registry_path = registry_path
        self.entries: List[Dict[str, Any]] = []
        self._load_existing()

    def _load_existing(self) -> None:
        """Load existing registry entries if file exists."""
        if os.path.exists(self.registry_path):
            try:
                with open(self.registry_path, 'r') as f:
                    self.entries = json.load(f)
                logger.debug(f"Loaded {len(self.entries)} existing entries from registry")
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Could not load existing registry: {e}")
                self.entries = []

    def add_entry(self, metadata: OutputMetadata) -> None:
        """
        Add a new entry to the registry.

        Parameters
        ----------
        metadata : OutputMetadata
            Metadata for the output to register.
        """
        self.entries.append(metadata.to_dict())
        self._save()

    def _save(self) -> None:
        """Save the registry to disk."""
        os.makedirs(os.path.dirname(self.registry_path), exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.entries, f, indent=2, default=str)

    def get_entries_for_script(self, script_name: str) -> List[Dict[str, Any]]:
        """Get all entries generated by a specific script."""
        return [e for e in self.entries if e.get('script_name') == script_name]

    def get_recent_entries(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the n most recent entries."""
        sorted_entries = sorted(
            self.entries,
            key=lambda x: x.get('timestamp', ''),
            reverse=True
        )
        return sorted_entries[:n]


def generate_versioned_path(
    base_dir: str,
    base_name: str,
    extension: str = '.nc',
    use_timestamp: bool = False,
    parameters: Optional[Dict[str, Any]] = None
) -> str:
    """
    Generate a path for an output file, optionally with versioning.

    Parameters
    ----------
    base_dir : str
        Directory for the output file.
    base_name : str
        Base name of the output file (without extension).
    extension : str, optional
        File extension including the dot. Default is '.nc'.
    use_timestamp : bool, optional
        If True, include timestamp in filename. Default is False (opt-in).
    parameters : dict, optional
        Parameters to hash for filename (if use_timestamp is True).

    Returns
    -------
    str
        Full path to the output file.

    Example
    -------
    >>> # Without versioning (default)
    >>> generate_versioned_path('/data', 'output')
    '/data/output.nc'
    >>> # With versioning
    >>> generate_versioned_path('/data', 'output', use_timestamp=True)
    '/data/output_20251214_120000.nc'
    """
    if use_timestamp:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{base_name}_{timestamp}{extension}"
    else:
        filename = f"{base_name}{extension}"

    return os.path.join(base_dir, filename)


def save_with_metadata(
    data,  # xarray.DataArray or Dataset
    output_path: str,
    script_name: str,
    parameters: Dict[str, Any],
    registry: Optional[OutputRegistry] = None,
    versioned: bool = False,
    save_metadata: bool = True
) -> OutputMetadata:
    """
    Save data with accompanying metadata file.

    Parameters
    ----------
    data : xarray.DataArray or xarray.Dataset
        The data to save.
    output_path : str
        Path for the output file.
    script_name : str
        Name of the script generating the output.
    parameters : dict
        Parameters used to generate the output.
    registry : OutputRegistry, optional
        Registry to record the output in.
    versioned : bool, optional
        If True, modify output_path to include timestamp. Default is False.
    save_metadata : bool, optional
        If True, save accompanying metadata JSON file. Default is True.

    Returns
    -------
    OutputMetadata
        Metadata object for the saved output.

    Example
    -------
    >>> import xarray as xr
    >>> data = xr.DataArray([1, 2, 3])
    >>> metadata = save_with_metadata(
    ...     data,
    ...     '/data/output.nc',
    ...     script_name='1_process.py',
    ...     parameters={'region': 'pacific'},
    ...     versioned=True
    ... )
    """
    # Modify path if versioned
    if versioned:
        base, ext = os.path.splitext(output_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"{base}_{timestamp}{ext}"

    # Get data shape if available
    data_shape = None
    if hasattr(data, 'dims') and hasattr(data, 'sizes'):
        data_shape = dict(data.sizes)

    # Save the data
    data.to_netcdf(output_path)
    logger.info(f"Data saved to {output_path}")

    # Create metadata
    metadata = OutputMetadata(
        script_name=script_name,
        output_path=output_path,
        parameters=parameters,
        data_shape=data_shape
    )

    # Save metadata file
    if save_metadata:
        metadata.save()

    # Register if registry provided
    if registry:
        registry.add_entry(metadata)

    return metadata


def add_versioning_args(parser) -> None:
    """
    Add standard versioning arguments to an argparse parser.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The argument parser to add arguments to.

    Example
    -------
    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> add_versioning_args(parser)
    >>> args = parser.parse_args(['--versioned'])
    >>> print(args.versioned)
    True
    """
    parser.add_argument(
        '--versioned',
        action='store_true',
        help='Enable versioned output filenames with timestamps and metadata'
    )
    parser.add_argument(
        '--no-metadata',
        action='store_true',
        help='Disable saving metadata JSON files alongside outputs'
    )


def hash_parameters(parameters: Dict[str, Any]) -> str:
    """
    Generate a short hash of parameters for filename deduplication.

    Parameters
    ----------
    parameters : dict
        Parameters to hash.

    Returns
    -------
    str
        8-character hash string.

    Example
    -------
    >>> hash_parameters({'region': 'pacific', 'year': 2024})
    'a1b2c3d4'
    """
    param_str = json.dumps(parameters, sort_keys=True, default=str)
    return hashlib.md5(param_str.encode()).hexdigest()[:8]
