"""levlab - Levenshtein distance with a measured banded cutoff matcher."""
from .core import (
    BAND_MIN_LENGTH,
    distance_at_most,
    levenshtein_distance,
    within_distance,
)

__version__ = "1.0.0"
__all__ = [
    "BAND_MIN_LENGTH",
    "distance_at_most",
    "levenshtein_distance",
    "within_distance",
]
