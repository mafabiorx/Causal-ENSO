"""
Processing result tracking utilities for scientific data processing scripts.

Provides dataclasses and enums for tracking processing outcomes,
ensuring scientific traceability and proper exit codes.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging


class ProcessingStatus(Enum):
    """Status of a processing operation."""
    SUCCESS = auto()
    FAILED = auto()
    PARTIAL = auto()  # Some data processed, some failed


class MaskingStrategy(Enum):
    """Strategy used for spatial masking (critical for scientific validity)."""
    SEA_MASKED = auto()      # Standard: ocean-only data used
    UNMASKED = auto()        # Fallback: no masking applied
    LAND_MASKED = auto()     # Land-only data
    NONE = auto()            # No masking applicable


@dataclass
class ProcessingResult:
    """Result of a single processing operation."""
    name: str
    status: ProcessingStatus
    output_path: Optional[str] = None
    error_message: Optional[str] = None
    masking_strategy: MaskingStrategy = MaskingStrategy.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def is_success(self) -> bool:
        return self.status == ProcessingStatus.SUCCESS

    def get_versioned_filename(self, base_name: str, extension: str = '.nc') -> str:
        """
        Generate a versioned filename including timestamp.

        Parameters
        ----------
        base_name : str
            Base name for the file (without extension).
        extension : str, optional
            File extension including dot. Default is '.nc'.

        Returns
        -------
        str
            Filename with timestamp, e.g., 'output_20251214_120000.nc'
        """
        timestamp_str = self.timestamp.strftime('%Y%m%d_%H%M%S')
        return f"{base_name}_{timestamp_str}{extension}"

    def get_metadata_filename(self, base_name: str) -> str:
        """
        Generate a metadata filename for this result.

        Parameters
        ----------
        base_name : str
            Base name for the file (without extension).

        Returns
        -------
        str
            Metadata filename, e.g., 'output_20251214_120000_metadata.json'
        """
        timestamp_str = self.timestamp.strftime('%Y%m%d_%H%M%S')
        return f"{base_name}_{timestamp_str}_metadata.json"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'status': self.status.name,
            'output_path': self.output_path,
            'error_message': self.error_message,
            'masking_strategy': self.masking_strategy.name,
            'metadata': self.metadata,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ProcessingSummary:
    """Summary of multiple processing operations."""
    script_name: str
    results: List[ProcessingResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def add_result(self, result: ProcessingResult) -> None:
        """Add a processing result."""
        self.results.append(result)

    @property
    def success_count(self) -> int:
        return sum(1 for r in self.results if r.status == ProcessingStatus.SUCCESS)

    @property
    def failure_count(self) -> int:
        return sum(1 for r in self.results if r.status == ProcessingStatus.FAILED)

    @property
    def all_successful(self) -> bool:
        return all(r.status == ProcessingStatus.SUCCESS for r in self.results)

    @property
    def has_failures(self) -> bool:
        return any(r.status == ProcessingStatus.FAILED for r in self.results)

    def finalize(self) -> None:
        """Mark processing as complete."""
        self.end_time = datetime.now()

    def log_summary(self, logger: logging.Logger) -> None:
        """Log processing summary."""
        self.finalize()

        logger.info("=" * 60)
        logger.info(f"PROCESSING SUMMARY: {self.script_name}")
        logger.info("=" * 60)
        logger.info(f"Total operations: {len(self.results)}")
        logger.info(f"Successful: {self.success_count}")
        logger.info(f"Failed: {self.failure_count}")

        if self.has_failures:
            logger.warning("Failed operations:")
            for r in self.results:
                if r.status == ProcessingStatus.FAILED:
                    logger.warning(f"  - {r.name}: {r.error_message}")

        # Log any masking fallbacks (scientifically important)
        fallbacks = [r for r in self.results
                     if r.masking_strategy == MaskingStrategy.UNMASKED
                     and r.status == ProcessingStatus.SUCCESS]
        if fallbacks:
            logger.warning("SCIENTIFIC NOTE: The following used unmasked (fallback) processing:")
            for r in fallbacks:
                logger.warning(f"  - {r.name}")

        duration = (self.end_time - self.start_time).total_seconds()
        logger.info(f"Total duration: {duration:.1f} seconds")
        logger.info("=" * 60)

    def get_exit_code(self) -> int:
        """Return appropriate exit code based on results."""
        if self.all_successful:
            return 0
        elif self.success_count > 0:
            return 2  # Partial success
        else:
            return 1  # Complete failure
