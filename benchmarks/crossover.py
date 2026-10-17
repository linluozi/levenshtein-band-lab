"""Boundary-region crossover, high-repetition.

The first pass showed max/min spread up to 2x on single rounds, which is too
wide to set a library threshold on. This version only sweeps the boundary
region (n=2..14) and raises the sample count so the per-round ratio settles:

  * 31 interleaved rounds per point,
  * each sample spans >= 30 ms (so timer resolution is ~0.1%),
  * report the median ratio and the 10th/90th percentile, not the extremes.

Decision rule: the crossover for cutoff k is the smallest n where the median
ratio is below 0.90 *and* the 90th percentile is below 1.00. Both conditions
must hold, because a guard that flips too early costs performance on exactly
the strings it was meant to help.
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.run_all import _band_core, arm_B_two_row_full  # noqa: E402


def band_kernel(a, b, k):
    return _band_core(a, b, k, gap_shortcut=False, early_exit=True) is not None


def sample(fn, args, min_time=0.03):
    reps = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        dt = time.perf_counter() - t0
        if dt >= min_time:
            return dt / reps
        reps *= 2


def pct(values, q):
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def measure(a, b, k, rounds=31):
    band, two = [], []
    for _ in range(rounds):
        band.append(sample(band_kernel, (a, b, k)))
        two.append(sample(arm_B_two_row_full, (a, b, k)))
    ratios = [x / y for x, y in zip(band, two)]
    return {
        "band": min(band), "two": min(two),
        "median": statistics.median(ratios),
        "p10": pct(ratios, 0.10), "p90": pct(ratios, 0.90),
        "ratio_of_bests": min(band) / min(two),
    }


print("Boundary crossover | 31 interleaved rounds, >=30 ms per sample")
print(f"{'n':>3} {'k':>3} {'band µs':>9} {'two µs':>9} {'median':>8} {'p10':>7} "
      f"{'p90':>7} {'best':>7}  verdict")
print("-" * 80)

stable = {}
clear = {}
for k in (0, 1, 2, 3, 4):
    for n in range(2, 15):
        if k > n:
            continue
        a = "a" * n
        b = "a" * (n - 1) + "b"
        r = measure(a, b, k)
        is_stable = r["median"] < 0.90 and r["p90"] < 1.00
        is_clear = r["median"] < 0.90
        if is_stable and k not in stable:
            stable[k] = n
        if is_clear and k not in clear:
            clear[k] = n
        if r["median"] < 0.90:
            verdict = "band wins"
        elif r["median"] > 1.10:
            verdict = "band loses"
        else:
            verdict = "tie"
        print(f"{n:3} {k:3} {r['band']*1e6:9.3f} {r['two']*1e6:9.3f} "
              f"{r['median']:7.3f}x {r['p10']:6.3f}x {r['p90']:6.3f}x "
              f"{r['ratio_of_bests']:6.3f}x  {verdict}")
    print()

print("Crossover, strict (median < 0.90 AND p90 < 1.00):")
for k in (0, 1, 2, 3, 4):
    print(f"  k={k}: n >= {stable.get(k, 'not found')}")
print("\nCrossover, median-only (median < 0.90):")
for k in (0, 1, 2, 3, 4):
    print(f"  k={k}: n >= {clear.get(k, 'not found')}")

print("\nRecommended BAND_MIN_LENGTH (strict, extrapolating beyond k=4):")
print("  ", {k: stable.get(k) for k in (0, 1, 2, 3, 4)})
