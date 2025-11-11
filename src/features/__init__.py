"""Feature engineering modules"""

from .engineering import TravecomFeatureEngine

# Alias for backward compatibility
FeatureEngine = TravecomFeatureEngine

__all__ = ['TravecomFeatureEngine', 'FeatureEngine']
