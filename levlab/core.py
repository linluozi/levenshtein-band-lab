"""Levenshtein distance: exact two-row DP and cutoff-aware banded matching.

Public API
----------
``levenshtein_distance(a, b)``
    Exact edit distance. O(m*n) time, O(min(m, n)) auxiliary space.

``distance_at_most(a, b, k)``
    Exact distance when it is <= k, otherwise ``None``. Uses a diagonal band
    plus row-minimum early termination, with two measured guards that keep it
    from being slower than the plain two-row scan.

``within_distance(a, b, k)``
    Boolean wrapper around ``distance_at_most``.

Semantics
---------
* Case-sensitive.
* Operates on Python Unicode code points, not grapheme clusters.
* Insertion, deletion and substitution each cost 1. Transposition is not a
  single operation (that is Damerau-Levenshtein, a different metric).
"""
from __future__ import annotations

from typing import Optional

__all__ = [
    "levenshtein_distance",
    "distance_at_most",
    "within_distance",
    "BAND_MIN_LENGTH",
]

# ---------------------------------------------------------------------------
# Measured crossover lengths
# ---------------------------------------------------------------------------
# Below these lengths the banded kernel is slower than the plain two-row scan.
# In CPython each band cell costs roughly 3-4x a cell of the compact two-row
# inner loop (extra `lo`/`hi` arithmetic, two boundary resets, `row_min`
# maintenance), so on short strings the saved cells do not pay for that.
#
# These are measured, not guessed: `benchmarks/crossover.py` interleaves the two
# kernels over 31 rounds per point, each sample spanning >= 30 ms, and requires
# median < 0.90 AND p90 < 1.00 before declaring the band a win. That strict rule
# matters - an earlier, noisier pass suggested k=2 crossed at n=6, but with 31
# rounds the honest answer is n=8.
#
# Measured on CPython 3.13.12 / Windows x64 (2026-10-17).
# Re-measure on your runtime before trusting the absolute values; the shape is
# stable across versions but the crossover moves with interpreter and CPU.
BAND_MIN_LENGTH = {0: 3, 1: 5, 2: 8, 3: 9}

# Beyond the measured table each extra unit of cutoff widens the band by one
# cell per side, so the break-even length keeps growing. Slope fitted to the
# measured points (3, 5, 8, 9): ~2 extra characters per unit of k.
_BAND_SLOPE = 2.0


def _band_min_length(max_distance: int) -> int:
    """Smallest shorter-string length at which the band is worth using."""
    table_hit = BAND_MIN_LENGTH.get(max_distance)
    if table_hit is not None:
        return table_hit
    return int(BAND_MIN_LENGTH[3] + _BAND_SLOPE * (max_distance - 3))


def _validate_cutoff(max_distance: int) -> None:
    if isinstance(max_distance, bool) or not isinstance(max_distance, int):
        raise TypeError("max_distance must be an integer")
    if max_distance < 0:
        raise ValueError("max_distance must be >= 0")


def _validate_strings(a: str, b: str) -> None:
    if not isinstance(a, str) or not isinstance(b, str):
        raise TypeError("a and b must be strings")


def levenshtein_distance(a: str, b: str) -> int:
    """Return the exact Levenshtein distance.

    Time O(len(a) * len(b)); auxiliary space O(min(len(a), len(b))).

    Note that the two-row form is a *memory* optimisation only. Measured over
    n = 16..1024 it is within noise of a full-matrix implementation in time
    (0.87-1.06x), because the full matrix is also O(mn) cells. Use this to save
    memory, not to save time.
    """
    _validate_strings(a, b)
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Keep the shorter string on the columns to minimise the row buffers.
    if len(a) < len(b):
        a, b = b, a

    width = len(b)
    previous = list(range(width + 1))
    for i, char_a in enumerate(a, 1):
        current = [i] + [0] * width
        for j, char_b in enumerate(b, 1):
            current[j] = min(
                previous[j] + 1,                 # deletion
                current[j - 1] + 1,              # insertion
                previous[j - 1] + (char_a != char_b),
            )
        previous = current
    return previous[width]


def _two_row_at_most(a: str, b: str, max_distance: int) -> Optional[int]:
    """Exact distance via two rows; returns None when it exceeds the cutoff.

    No band, no early exit. This is the reference kernel that the banded path
    falls back to when the band would not pay for itself.
    """
    if len(a) < len(b):
        a, b = b, a

    width = len(b)
    previous = list(range(width + 1))
    for i, char_a in enumerate(a, 1):
        current = [i] + [0] * width
        for j, char_b in enumerate(b, 1):
            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (char_a != char_b),
            )
        previous = current
    result = previous[width]
    return result if result <= max_distance else None


def distance_at_most(a: str, b: str, max_distance: int) -> Optional[int]:
    """Return the exact distance when it is <= max_distance, otherwise None.

    Uses a diagonal band of at most ``2k + 1`` cells per row plus row-minimum
    early termination, giving O(k * max(m, n)) time in bounded cases.

    Two guards keep this from being a pessimisation:

    1. Length gap. If ``abs(len(a) - len(b)) > k`` the answer is ``None``
       immediately - insertions/deletions alone already exceed the budget.
    2. Band coverage. The band only helps when it is actually narrower than the
       matrix. If ``k >= min(len(a), len(b))`` the band spans the full width, so
       the plain two-row scan is used instead (measured 1.09-1.15x faster).
    3. Short strings. Below the measured break-even length (see
       ``BAND_MIN_LENGTH``) the per-cell band overhead dominates, so the plain
       two-row scan is used. Without this, Chinese 2-4 character search terms
       are 1.07-1.39x *slower* than no band at all.
    """
    _validate_strings(a, b)
    _validate_cutoff(max_distance)

    if a == b:
        return 0
    if abs(len(a) - len(b)) > max_distance:
        return None
    if not a:
        return len(b) if len(b) <= max_distance else None
    if not b:
        return len(a) if len(a) <= max_distance else None

    shorter = min(len(a), len(b))

    # Guard 2: the band would span the whole matrix; drop the band machinery.
    # Guard 3: too short for the band to pay for its own bookkeeping.
    if max_distance >= shorter or shorter < _band_min_length(max_distance):
        return _two_row_at_most(a, b, max_distance)

    # Put the shorter string on the columns, keeping the row storage compact.
    if len(a) < len(b):
        a, b = b, a

    n = len(a)  # rows; longer string
    m = len(b)  # columns; shorter string
    inf = max_distance + 1

    previous = [inf] * (m + 1)
    for j in range(min(m, max_distance) + 1):
        previous[j] = j
    current = [inf] * (m + 1)

    for i in range(1, n + 1):
        lo = max(1, i - max_distance)
        hi = min(m, i + max_distance)

        # Cells just outside this row's band must not retain stale values from
        # a previous use of the recycled row buffer.
        if lo > 1:
            current[lo - 1] = inf
        current[0] = i if i <= max_distance else inf
        if hi < m:
            current[hi + 1] = inf

        row_min = current[0] if i <= max_distance else inf
        char_a = a[i - 1]

        for j in range(lo, hi + 1):
            value = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (char_a != b[j - 1]),
            )
            current[j] = value
            if value < row_min:
                row_min = value

        # Every path through this row already exceeds the permitted budget.
        if row_min > max_distance:
            return None

        previous, current = current, previous

    result = previous[m]
    return result if result <= max_distance else None


def within_distance(a: str, b: str, max_distance: int) -> bool:
    """Return whether the edit distance is at most ``max_distance``."""
    return distance_at_most(a, b, max_distance) is not None
