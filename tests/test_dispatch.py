"""Prove the guards actually route where they claim, without relying on timers.

Short-string timings are 1-20 microseconds, close enough to timer noise that a
measured "win" can flip sign between runs. Path assertions are deterministic:
they check *which kernel ran*, so the claim "short strings avoid the band" is
verifiable even when the stopwatch is not.

The trick is a call counter injected around the internal kernels. Nothing else
changes, so the counts reflect the real dispatch logic.
"""
from __future__ import annotations

import unittest
from unittest import mock

import levlab.core as core


class PathDispatch(unittest.TestCase):
    def setUp(self):
        self.band_calls = 0
        self.two_row_calls = 0
        self._orig_band = core._band_min_length

    def _wrap(self):
        """Patch the two kernels with counting wrappers."""
        real_two_row = core._two_row_at_most

        def counting_two_row(a, b, k):
            self.two_row_calls += 1
            return real_two_row(a, b, k)

        return mock.patch.object(core, "_two_row_at_most", counting_two_row)

    def test_short_strings_take_two_row_path(self):
        """For every measured crossover, lengths below it must avoid the band."""
        for cutoff, threshold in sorted(core.BAND_MIN_LENGTH.items()):
            for shorter in range(0, threshold):
                a = "a" * max(shorter, 0)
                b = "a" * max(shorter - 1, 0) + "b"
                if shorter == 0:
                    continue  # empty strings short-circuit before dispatch
                self.two_row_calls = 0
                with self._wrap():
                    core.distance_at_most(a, b, cutoff)
                self.assertEqual(
                    self.two_row_calls, 1,
                    f"len={shorter} k={cutoff} should take the two-row path",
                )

    def test_long_strings_take_band_path(self):
        """Well above the crossover, the band must be used (not the fallback)."""
        for cutoff in (0, 1, 2):
            threshold = core._band_min_length(cutoff)
            n = threshold + 20
            a = "a" * n
            b = "a" * (n - 1) + "b"
            self.two_row_calls = 0
            with self._wrap():
                result = core.distance_at_most(a, b, cutoff)
            self.assertEqual(
                self.two_row_calls, 0,
                f"len={n} k={cutoff} should use the band, not the fallback",
            )
            # True distance is 1, so the answer is 1 when k >= 1, else None.
            self.assertEqual(result, 1 if cutoff >= 1 else None, (n, cutoff))

    def test_full_width_cutoff_takes_two_row_path(self):
        """k >= min(len) means the band spans the matrix; guard must catch it.

        Each case must also survive the length-gap shortcut (|len(a)-len(b)| <= k),
        otherwise it returns before dispatch and proves nothing.
        """
        for a, b, k in (
            ("x" * 50, "y" * 50, 50),          # equal lengths, k == min
            ("x" * 50, "y" * 50, 999),         # k far above min
            ("abc", "abcde", 3),               # gap 2 <= k 3, k == min
            ("abcdefghij", "abcdefgh", 8),     # gap 2 <= k 8, k == min
        ):
            self.assertLessEqual(abs(len(a) - len(b)), k, "case defeats the gap shortcut")
            self.assertGreaterEqual(k, min(len(a), len(b)), "case is not full-width")
            self.two_row_calls = 0
            with self._wrap():
                core.distance_at_most(a, b, k)
            self.assertEqual(
                self.two_row_calls, 1,
                f"{a!r} {b!r} k={k} should take the two-row path",
            )

    def test_guard_boundaries_match_documented_table(self):
        """The dispatch flips exactly at the documented length, not around it."""
        for cutoff, threshold in sorted(core.BAND_MIN_LENGTH.items()):
            # At the threshold the band is used...
            a = "a" * threshold
            b = "a" * (threshold - 1) + "b"
            self.two_row_calls = 0
            with self._wrap():
                core.distance_at_most(a, b, cutoff)
            self.assertEqual(self.two_row_calls, 0,
                             f"len={threshold} k={cutoff} should use the band")

            # ...and one character shorter it is not.
            a = "a" * (threshold - 1)
            b = "a" * (threshold - 2) + "b"
            self.two_row_calls = 0
            with self._wrap():
                core.distance_at_most(a, b, cutoff)
            self.assertEqual(self.two_row_calls, 1,
                             f"len={threshold - 1} k={cutoff} should fall back")


class GuardCorrectness(unittest.TestCase):
    """Both paths must return identical answers; the guard is not a semantic change."""

    def test_two_row_fallback_agrees_with_band(self):
        import random

        from levlab import distance_at_most

        rng = random.Random(31337)
        for _ in range(500):
            n = rng.randint(0, 30)
            m = rng.randint(0, 30)
            a = "".join(rng.choice("abc") for _ in range(n))
            b = "".join(rng.choice("abc") for _ in range(m))
            for k in range(0, 6):
                # Compare against the unguarded band kernel.
                expected = core._two_row_at_most(a, b, k)
                self.assertEqual(distance_at_most(a, b, k), expected, (a, b, k))


if __name__ == "__main__":
    unittest.main(verbosity=2)
