"""Correctness tests for levlab.

The oracle is written deliberately differently from the implementation (full
matrix, explicit bounds, no rolling rows) so that a shared bug cannot make both
agree.
"""
from __future__ import annotations

import itertools
import os
import random
import unittest

from levlab import (
    BAND_MIN_LENGTH,
    distance_at_most,
    levenshtein_distance,
    within_distance,
)
from levlab.core import _band_min_length, _two_row_at_most


def oracle(a: str, b: str) -> int:
    """Full-matrix reference, O(m*n) time and space."""
    rows, cols = len(a) + 1, len(b) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
            )
    return dp[-1][-1]


class KnownValues(unittest.TestCase):
    def test_textbook_examples(self):
        self.assertEqual(levenshtein_distance("kitten", "sitting"), 3)
        self.assertEqual(levenshtein_distance("saturday", "sunday"), 3)
        self.assertEqual(levenshtein_distance("", "abc"), 3)
        self.assertEqual(levenshtein_distance("abc", ""), 3)
        self.assertEqual(levenshtein_distance("same", "same"), 0)
        self.assertEqual(levenshtein_distance("café", "cafe"), 1)
        self.assertEqual(levenshtein_distance("你好", "你們好"), 1)

    def test_cutoff_boundaries(self):
        self.assertEqual(distance_at_most("kitten", "sitting", 3), 3)
        self.assertIsNone(distance_at_most("kitten", "sitting", 2))
        self.assertTrue(within_distance("book", "back", 2))
        self.assertFalse(within_distance("book", "back", 1))
        self.assertEqual(distance_at_most("", "abc", 3), 3)
        self.assertIsNone(distance_at_most("", "abc", 2))
        self.assertEqual(distance_at_most("abc", "abc", 0), 0)
        self.assertTrue(within_distance("abc", "abc", 0))

    def test_zero_distance_is_not_falsy_bug(self):
        # within_distance must use `is not None`, not truthiness.
        self.assertTrue(within_distance("abc", "abc", 0))
        self.assertTrue(within_distance("", "", 0))


class CutoffValidation(unittest.TestCase):
    def test_negative_rejected(self):
        with self.assertRaises(ValueError):
            within_distance("a", "b", -1)

    def test_non_integer_rejected(self):
        with self.assertRaises(TypeError):
            within_distance("a", "b", 1.5)
        with self.assertRaises(TypeError):
            within_distance("a", "b", "2")

    def test_bool_rejected(self):
        # bool is an int subclass; True would silently mean k=1.
        with self.assertRaises(TypeError):
            within_distance("a", "b", True)

    def test_non_string_rejected(self):
        with self.assertRaises(TypeError):
            levenshtein_distance(b"a", "b")
        with self.assertRaises(TypeError):
            distance_at_most("a", None, 1)


class ExhaustiveOracle(unittest.TestCase):
    """Every pair of strings up to length 5 over a 4-char alphabet.

    This is the slowest test by far (~2 min). It runs by default, but CI sets
    ``LEVLAB_FAST=1`` to skip it, where the 19-test suite drops to ~2 s.
    """

    ALPHABET = "abcé"

    @unittest.skipIf(
        os.environ.get("LEVLAB_FAST") == "1",
        "LEVLAB_FAST=1: skipping exhaustive sweep",
    )
    def test_all_pairs_all_cutoffs(self):
        strings = [""]
        for length in range(1, 6):
            strings.extend(
                "".join(chars) for chars in itertools.product(self.ALPHABET, repeat=length)
            )
        for a in strings:
            for b in strings:
                expected = oracle(a, b)
                self.assertEqual(levenshtein_distance(a, b), expected, (a, b))
                for cutoff in range(7):
                    self.assertEqual(
                        distance_at_most(a, b, cutoff),
                        expected if expected <= cutoff else None,
                        (a, b, cutoff, expected),
                    )

    def test_sampled_sweep(self):
        """Always-on reduced version of the sweep, so CI still covers the DP."""
        strings = [""]
        for length in range(1, 4):
            strings.extend(
                "".join(chars) for chars in itertools.product(self.ALPHABET, repeat=length)
            )
        for a in strings:
            for b in strings:
                expected = oracle(a, b)
                self.assertEqual(levenshtein_distance(a, b), expected, (a, b))
                for cutoff in range(5):
                    self.assertEqual(
                        distance_at_most(a, b, cutoff),
                        expected if expected <= cutoff else None,
                        (a, b, cutoff),
                    )


class BandPaths(unittest.TestCase):
    """Target the code paths the exhaustive sweep cannot reach."""

    def test_narrow_band_on_long_strings(self):
        # Long enough that the band is genuinely narrower than the matrix, so
        # both lo > 1 and hi < m are active in the same row.
        rng = random.Random(20261017)
        for _ in range(300):
            n = rng.randint(20, 120)
            a = "".join(rng.choice("abcdefgh") for _ in range(n))
            b = "".join(rng.choice("abcdefgh") for _ in range(n))
            expected = oracle(a, b)
            for cutoff in (0, 1, 2, 3, 5, 9):
                self.assertEqual(
                    distance_at_most(a, b, cutoff),
                    expected if expected <= cutoff else None,
                    (a, b, cutoff),
                )

    def test_constructed_near_misses(self):
        rng = random.Random(7)
        for base_len in (10, 40, 120):
            for distance in range(0, 11):
                if distance > base_len:
                    continue
                base = "".join(rng.choice("abcdef") for _ in range(base_len))
                variant = list(base)
                for pos in range(distance):
                    variant[pos] = "z" if variant[pos] != "z" else "y"
                variant = "".join(variant)
                expected = oracle(base, variant)
                for cutoff in range(0, 14):
                    self.assertEqual(
                        distance_at_most(base, variant, cutoff),
                        expected if expected <= cutoff else None,
                        (base_len, distance, cutoff),
                    )

    def test_cutoff_wider_than_string(self):
        # Guard 2: k >= min(len) takes the full-scan path. Must not IndexError.
        for a, b in (("x" * 40, "y" * 40), ("abc", "abcde"), ("", "abc")):
            expected = oracle(a, b)
            for cutoff in (len(a), len(b), 50, 10 ** 6):
                self.assertEqual(
                    distance_at_most(a, b, cutoff),
                    expected if expected <= cutoff else None,
                    (a, b, cutoff),
                )

    def test_short_string_guard_is_correct(self):
        # Guard 3 routes short strings away from the band. Verify both sides of
        # the threshold agree with the oracle.
        for cutoff in range(0, 6):
            threshold = _band_min_length(cutoff)
            for n in range(0, threshold + 4):
                a = "a" * n
                b = "a" * max(0, n - 1) + "b"
                expected = oracle(a, b)
                self.assertEqual(
                    distance_at_most(a, b, cutoff),
                    expected if expected <= cutoff else None,
                    (n, cutoff),
                )

    def test_band_min_length_table(self):
        self.assertEqual(BAND_MIN_LENGTH, {0: 3, 1: 5, 2: 8, 3: 9})
        # Extrapolation must stay monotonic and keep growing with k.
        values = [_band_min_length(k) for k in range(0, 20)]
        self.assertEqual(values, sorted(values))
        self.assertGreater(values[-1], values[3])

    def test_large_length_gap_short_circuits(self):
        self.assertFalse(within_distance("x" * 100, "x" * 110, 3))
        self.assertTrue(within_distance("x" * 100, "x" * 103, 3))


class Unicode(unittest.TestCase):
    def test_combining_marks_are_separate_code_points(self):
        # "a" + U+0301 is two code points; "á" is one. Distance is 2, not 0.
        self.assertEqual(levenshtein_distance("a\u0301", "\u00e1"), 2)
        self.assertEqual(levenshtein_distance("\u00e1", "\u00e1"), 0)

    def test_edge_characters(self):
        pairs = [
            ("\x00", "\x00"),
            ("\x00", "a"),
            ("🙂", "🙂🙂"),
            ("🙂", "🙃"),
            ("日本語", "日本語テキスト"),
        ]
        for a, b in pairs:
            expected = oracle(a, b)
            self.assertEqual(levenshtein_distance(a, b), expected, (a, b))
            for cutoff in range(0, 6):
                self.assertEqual(
                    distance_at_most(a, b, cutoff),
                    expected if expected <= cutoff else None,
                    (a, b, cutoff),
                )


class Properties(unittest.TestCase):
    def test_symmetry(self):
        rng = random.Random(99)
        for _ in range(400):
            a = "".join(rng.choice("abcde") for _ in range(rng.randint(0, 40)))
            b = "".join(rng.choice("abcde") for _ in range(rng.randint(0, 40)))
            self.assertEqual(levenshtein_distance(a, b), levenshtein_distance(b, a))
            self.assertEqual(
                distance_at_most(a, b, 3), distance_at_most(b, a, 3), (a, b)
            )

    def test_triangle_inequality(self):
        rng = random.Random(1234)
        for _ in range(200):
            a = "".join(rng.choice("abc") for _ in range(rng.randint(0, 20)))
            b = "".join(rng.choice("abc") for _ in range(rng.randint(0, 20)))
            c = "".join(rng.choice("abc") for _ in range(rng.randint(0, 20)))
            ab = levenshtein_distance(a, b)
            bc = levenshtein_distance(b, c)
            ac = levenshtein_distance(a, c)
            self.assertLessEqual(ac, ab + bc, (a, b, c))

    def test_fallback_matches_public_exact(self):
        # The two-row fallback and the public exact function must agree.
        rng = random.Random(555)
        for _ in range(200):
            a = "".join(rng.choice("abcdef") for _ in range(rng.randint(0, 60)))
            b = "".join(rng.choice("abcdef") for _ in range(rng.randint(0, 60)))
            self.assertEqual(
                _two_row_at_most(a, b, 10 ** 9), levenshtein_distance(a, b), (a, b)
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
