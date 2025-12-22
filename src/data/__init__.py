"""Data loading, cleaning, and validation modules"""

from .loaders import TravecomDataLoader
from .cleaners import TravecomDataCleaner
from .validators import DataValidator
from .aggregators import DataAggregator

# Aliases for backward compatibility with pipeline imports
DataLoader = TravecomDataLoader
DataCleaner = TravecomDataCleaner

__all__ = [
    'TravecomDataLoader',
    'TravecomDataCleaner',
    'DataValidator',
    'DataAggregator',
    'DataLoader',  # Alias
    'DataCleaner'  # Alias
]
