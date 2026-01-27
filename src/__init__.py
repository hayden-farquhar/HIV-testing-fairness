"""
HIV Testing Fairness Analysis Package

Modules:
    data_utils: BRFSS data loading and preprocessing
    fairness_metrics: Fairness metric calculations
    mitigation: Fairness mitigation wrappers
    visualization: Figure generation utilities
"""

from . import data_utils
from . import fairness_metrics
from . import mitigation
from . import visualization

__version__ = "1.0.0"
__author__ = "HIV Testing Fairness Research Team"
