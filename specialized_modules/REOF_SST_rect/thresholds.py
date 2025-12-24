# thresholds.py
from dataclasses import dataclass

@dataclass(frozen=True)
class RobustThresholds:
    """All numbers that define when an edge is called ‘robust’."""
    min_persistence: float = 0.75    # ≥ 3 out of 4 alpha runs
    max_avg_p: float = 0.05
    min_direction: float = 0.90    # 90 % same sign
    min_effect: float = 0.15    # absolute effect size
