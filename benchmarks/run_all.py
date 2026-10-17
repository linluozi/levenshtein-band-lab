"""Benchmark harness for levlab.

Design rules, all of them learned the hard way:

1. **Four arms, adjacent pairs.** Every speedup in the report must be traceable
   to the two neighbouring arms that isolate it. An "optimized vs original"
   comparison hides how many changes the optimized version actually made.
2. **Ratios, not absolute milliseconds.** Absolute wall-clock drifts by 3x
   across cases inside one process (allocator state accumulates). Only the
   per-round ratio is stable, so every headline number is a ratio.
3. **Interleave the arms within a round.** Running arm A to completion and then
   arm B lets CPU frequency and background load land entirely on one side.
4. **Separate processes for the headline table.** ``--isolated`` re-executes
   each arm in its own interpreter so nothing leaks between measurements.
5. **best-of-N for time, min-of-N for tracemalloc.** Interference only ever
   makes a single run slower, so the minimum is the least contaminated sample.

Arms
----
A  full_matrix      full DP matrix, reallocated per call (the naive baseline)
B  two_row_full     rolling two rows, all cells, no band, no early exit
C  band_core        band + row-min early exit, length-gap shortcut disabled
D  levlab           the shipped ``distance_at_most`` (band + both guards)

Read the report as:

    B/A   value of dropping the matrix allocation        (memory, not time)
    C/B   value of the band + early exit                 (the real algorithm win)
    D/C   value of the guards (short-string + full-width)

Usage
-----
    python benchmarks/run_all.py            # full sweep, writes results.json
    python benchmarks/run_all.py --quick    # smoke test, writes results-quick.json
    python benchmarks/run_all.py --isolated # headline table in separate processes
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from levlab import distance_at_most  # noqa: E402


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def arm_A_full_matrix(a: str, b: str, k: int) -> bool:
    """Naive baseline: full matrix, reallocated on every call."""
    dp = [list(range(len(b) + 1))]
    for i, ca in enumerate(a, 1):
        row = [i] + [0] * len(b)
        prev = dp[-1]
        for j, cb in enumerate(b, 1):
            row[j] = min(prev[j] + 1, row[j - 1] + 1, prev[j - 1] + (ca != cb))
        dp.append(row)
    return dp[-1][-1] <= k


def arm_B_two_row_full(a: str, b: str, k: int) -> bool:
    """Rolling two rows, every cell computed. No band, no early exit."""
    if len(a) < len(b):
        a, b = b, a
    width = len(b)
    previous = list(range(width + 1))
    for i, ca in enumerate(a, 1):
        current = [i] + [0] * width
        for j, cb in enumerate(b, 1):
            current[j] = min(previous[j] + 1, current[j - 1] + 1,
                             previous[j - 1] + (ca != cb))
        previous = current
    return previous[width] <= k


def _band_core(a: str, b: str, k: int, *, gap_shortcut: bool, early_exit: bool):
    """Bare band kernel, with the guards switchable so each can be isolated."""
    if a == b:
        return 0
    if gap_shortcut and abs(len(a) - len(b)) > k:
        return None
    if not a or not b:
        return len(a) + len(b) if len(a) + len(b) <= k else None
    if len(a) < len(b):
        a, b = b, a
    n, m = len(a), len(b)
    inf = k + 1
    previous = [inf] * (m + 1)
    for j in range(min(m, k) + 1):
        previous[j] = j
    current = [inf] * (m + 1)
    for i in range(1, n + 1):
        lo = max(1, i - k)
        hi = min(m, i + k)
        if lo > 1:
            current[lo - 1] = inf
        current[0] = i if i <= k else inf
        if hi < m:
            current[hi + 1] = inf
        row_min = current[0] if i <= k else inf
        ca = a[i - 1]
        for j in range(lo, hi + 1):
            v = min(previous[j] + 1, current[j - 1] + 1,
                    previous[j - 1] + (ca != b[j - 1]))
            current[j] = v
            if v < row_min:
                row_min = v
        if early_exit and row_min > k:
            return None
        previous, current = current, previous
    r = previous[m]
    return r if r <= k else None


def arm_C_band_core(a: str, b: str, k: int) -> bool:
    """Band + early exit, but no length-gap shortcut and no guards."""
    return _band_core(a, b, k, gap_shortcut=False, early_exit=True) is not None


def arm_D_levlab(a: str, b: str, k: int) -> bool:
    """The shipped implementation."""
    return distance_at_most(a, b, k) is not None


ARMS = {
    "A_full_matrix": arm_A_full_matrix,
    "B_two_row_full": arm_B_two_row_full,
    "C_band_core": arm_C_band_core,
    "D_levlab": arm_D_levlab,
}
ARM_ORDER = list(ARMS)


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------

def time_once(fn, args, min_time: float = 0.01) -> float:
    """Return the best per-call time over an adaptive repeat count."""
    reps = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(reps):
            fn(*args)
        elapsed = time.perf_counter() - t0
        if elapsed >= min_time:
            return elapsed / reps
        reps *= 2


def interleaved_ratios(args, rounds: int = 7, min_time: float = 0.01):
    """Run every arm once per round, return per-arm samples and ratio medians.

    Interleaving is what makes the ratios trustworthy: a slow drift affects all
    arms equally within a round, so the ratio cancels it.
    """
    samples = {name: [] for name in ARM_ORDER}
    for _ in range(rounds):
        for name in ARM_ORDER:
            samples[name].append(time_once(ARMS[name], args, min_time))

    def median_ratio(num: str, den: str) -> float:
        return statistics.median(
            [samples[num][i] / samples[den][i] for i in range(rounds)]
        )

    return {
        "samples": samples,
        "best": {name: min(v) for name, v in samples.items()},
        "median": {name: statistics.median(v) for name, v in samples.items()},
        "ratio_B_A": median_ratio("B_two_row_full", "A_full_matrix"),
        "ratio_C_B": median_ratio("C_band_core", "B_two_row_full"),
        "ratio_D_C": median_ratio("D_levlab", "C_band_core"),
        "ratio_D_A": median_ratio("D_levlab", "A_full_matrix"),
        "ratio_D_B": median_ratio("D_levlab", "B_two_row_full"),
    }


def peak_kib(fn, args, repeat: int = 5) -> float:
    """tracemalloc peak in KiB, min-of-N (first run pays lazy init)."""
    peaks = []
    for _ in range(repeat):
        tracemalloc.start()
        fn(*args)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peaks.append(peak)
    return min(peaks) / 1024


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def near_match(n: int) -> tuple[str, str]:
    return "a" * n, "a" * (n // 2) + "b" + "a" * (n - n // 2 - 1)


def early_reject(n: int) -> tuple[str, str]:
    return "a" * n, "b" * n


def length_gap(n: int) -> tuple[str, str]:
    return "x" * n, "x" * (n + 20)


def section_headline(results: dict) -> None:
    print("\n## Headline cases (n=320, k=2)")
    print("   Each ratio isolates one change. B/A is memory-only, C/B is the "
          "band, D/C is the guards.")
    header = (f"{'case':16} {'A ms':>9} {'B ms':>9} {'C ms':>9} {'D ms':>9} "
              f"{'B/A':>7} {'C/B':>7} {'D/C':>7} {'D/A':>8}")
    print(header)
    for name, (a, b) in (
        ("near-match", near_match(320)),
        ("early-reject", early_reject(320)),
        ("length-gap", length_gap(320)),
    ):
        r = interleaved_ratios((a, b, 2), rounds=5)
        m = r["median"]
        print(f"{name:16} {m['A_full_matrix']*1000:9.3f} {m['B_two_row_full']*1000:9.3f} "
              f"{m['C_band_core']*1000:9.3f} {m['D_levlab']*1000:9.3f} "
              f"{r['ratio_B_A']:6.2f}x {r['ratio_C_B']:6.2f}x {r['ratio_D_C']:6.2f}x "
              f"{r['ratio_D_A']:7.1f}x")
        results.setdefault("headline", {})[name] = {
            "median_ms": {k: v * 1000 for k, v in m.items()},
            "ratio_B_A": r["ratio_B_A"],
            "ratio_C_B": r["ratio_C_B"],
            "ratio_D_C": r["ratio_D_C"],
            "ratio_D_A": r["ratio_D_A"],
            "ratio_D_B": r["ratio_D_B"],
        }


def section_length_sweep(results: dict, lengths=(16, 32, 64, 128, 256, 512, 1024)) -> None:
    print("\n## Length sweep (near-match, k=2)")
    print(f"{'n':>6} {'B ms':>10} {'D ms':>10} {'D/B':>8}  {'band wins?':>12}")
    rows = []
    for n in lengths:
        a, b = near_match(n)
        r = interleaved_ratios((a, b, 2), rounds=5)
        ratio = r["ratio_D_B"]
        verdict = "yes" if ratio < 0.95 else ("tie" if ratio < 1.05 else "NO - slower")
        print(f"{n:6} {r['median']['B_two_row_full']*1000:10.3f} "
              f"{r['median']['D_levlab']*1000:10.3f} {ratio:7.3f}x  {verdict:>12}")
        rows.append({"n": n, "ratio_D_B": ratio,
                     "B_ms": r["median"]["B_two_row_full"] * 1000,
                     "D_ms": r["median"]["D_levlab"] * 1000})
    results["length_sweep"] = rows


def section_cutoff_sweep(results: dict, n: int = 512) -> None:
    print(f"\n## Cutoff sweep (near-match, n={n})")
    print("   The band spans the full width once k >= min(n, m); the guard should "
          "prevent the regression visible in the unguarded kernel.")
    print(f"{'k':>6} {'C ms':>10} {'D ms':>10} {'B ms':>10} {'D/B':>8}  note")
    rows = []
    a, b = near_match(n)
    for k in (0, 1, 2, 3, 5, 10, 20, 50, 100, 200, 300, 400, 512, 1024):
        r = interleaved_ratios((a, b, k), rounds=5)
        ratio = r["ratio_D_B"]
        note = "band wins" if ratio < 0.95 else ("tie" if ratio < 1.05 else "slower")
        print(f"{k:6} {r['median']['C_band_core']*1000:10.3f} "
              f"{r['median']['D_levlab']*1000:10.3f} "
              f"{r['median']['B_two_row_full']*1000:10.3f} {ratio:7.3f}x  {note}")
        rows.append({"k": k, "ratio_D_B": ratio, "ratio_C_B": r["ratio_C_B"]})
    results["cutoff_sweep"] = rows


def section_short_strings(results: dict) -> None:
    """Isolate the short-string guard: bare band kernel vs bare two-row scan.

    Both arms are called the same way (no closure, no validation) so the only
    difference is the kernel. The shipped function is reported separately
    because its argument validation is a visible fixed cost at this scale.
    """
    print("\n## Short strings: where the guard earns its keep")
    print("   'band' and 'two-row' are the bare kernels, called identically.")
    print("   'shipped' is distance_at_most, which also pays validation + guards.")
    print(f"{'n':>4} {'k':>3} {'band µs':>9} {'two-row µs':>11} {'band/two':>9} "
          f"{'shipped µs':>11}  verdict")

    def band_kernel(a, b, k):
        return _band_core(a, b, k, gap_shortcut=False, early_exit=True) is not None

    def two_row_kernel(a, b, k):
        return arm_B_two_row_full(a, b, k)

    rows = []
    for n in (2, 3, 4, 5, 6, 8, 10, 12, 16):
        for k in (0, 1, 2, 3):
            if k > n:
                continue
            a = "a" * n
            b = "a" * (n - 1) + "b"
            # Interleave the two kernels so drift cancels, min-of-9 each.
            t_band, t_two = [], []
            for _ in range(9):
                t_band.append(time_once(band_kernel, (a, b, k), min_time=0.004))
                t_two.append(time_once(two_row_kernel, (a, b, k), min_time=0.004))
            t_band, t_two = min(t_band), min(t_two)
            t_shipped = min(time_once(arm_D_levlab, (a, b, k), min_time=0.004)
                            for _ in range(9))
            ratio = t_band / t_two
            if ratio < 0.95:
                verdict = "band wins"
            elif ratio > 1.05:
                verdict = "guard needed"
            else:
                verdict = "tie"
            print(f"{n:4} {k:3} {t_band*1e6:9.3f} {t_two*1e6:11.3f} {ratio:8.2f}x "
                  f"{t_shipped*1e6:11.3f}  {verdict}")
            rows.append({"n": n, "k": k, "band_us": t_band * 1e6,
                         "two_row_us": t_two * 1e6, "shipped_us": t_shipped * 1e6,
                         "ratio_band_vs_two_row": ratio})
    results["short_strings"] = rows

    # Derive the observed crossover: first n at which band < 0.95 * two_row.
    print("\n   observed crossover (first n where band/two-row < 0.95):")
    for k in (0, 1, 2, 3):
        hits = [r["n"] for r in rows if r["k"] == k and r["ratio_band_vs_two_row"] < 0.95]
        print(f"     k={k}: n >= {hits[0] if hits else 'never in 2..16'}")
        results.setdefault("crossover", {})[k] = hits[0] if hits else None


def section_chinese(results: dict) -> None:
    print("\n## Real Chinese search terms")
    print("   'band' vs 'two-row' bare kernels; the shipped function adds guards "
          "and validation on top.")
    pairs = [
        ("苹果", "平果"), ("香蕉", "香焦"), ("你好世界", "你好世届"),
        ("数据库", "数据裤"), ("机器学习", "机器学习算法"),
        ("北京", "背景"), ("搜索", "素索"),
    ]

    def band_kernel(a, b, k):
        return _band_core(a, b, k, gap_shortcut=False, early_exit=True) is not None

    print(f"{'a':>10} {'b':>14} {'k':>3} {'band µs':>9} {'two-row µs':>11} "
          f"{'band/two':>9} {'shipped µs':>11}")
    rows = []
    for a, b in pairs:
        k = 1 if len(a) == len(b) else abs(len(a) - len(b)) + 1
        t_band, t_two = [], []
        for _ in range(9):
            t_band.append(time_once(band_kernel, (a, b, k), min_time=0.004))
            t_two.append(time_once(arm_B_two_row_full, (a, b, k), min_time=0.004))
        t_band, t_two = min(t_band), min(t_two)
        t_shipped = min(time_once(arm_D_levlab, (a, b, k), min_time=0.004)
                        for _ in range(9))
        print(f"{a:>10} {b:>14} {k:3} {t_band*1e6:9.3f} {t_two*1e6:11.3f} "
              f"{t_band/t_two:8.2f}x {t_shipped*1e6:11.3f}")
        rows.append({"a": a, "b": b, "k": k, "band_us": t_band * 1e6,
                     "two_row_us": t_two * 1e6, "shipped_us": t_shipped * 1e6,
                     "ratio_band_vs_two_row": t_band / t_two})
    results["chinese"] = rows


def section_memory(results: dict, lengths=(64, 256, 1024, 2048)) -> None:
    print("\n## Peak memory (tracemalloc, min-of-5, KiB)")
    print(f"{'n':>6} {'A full matrix':>15} {'B two rows':>12} {'D band k=2':>12} {'exact':>10}")
    rows = []
    for n in lengths:
        a, b = near_match(n)
        pa = peak_kib(arm_A_full_matrix, (a, b, 2))
        pb = peak_kib(arm_B_two_row_full, (a, b, 2))
        pd = peak_kib(arm_D_levlab, (a, b, 2))
        pe = peak_kib(lambda x, y: distance_at_most(x, y, 10 ** 9), (a, b))
        print(f"{n:6} {pa:15.1f} {pb:12.1f} {pd:12.1f} {pe:10.1f}")
        rows.append({"n": n, "A_KiB": pa, "B_KiB": pb, "D_KiB": pd, "exact_KiB": pe})
    results["memory"] = rows


def section_isolated(results: dict) -> None:
    """Headline cases, each arm in its own interpreter."""
    print("\n## Isolated processes (one interpreter per arm)")
    print("   Removes allocator state accumulated by earlier cases in the same process.")
    script = str(Path(__file__).resolve())
    for name, (a, b) in (
        ("near-match", near_match(320)),
        ("early-reject", early_reject(320)),
        ("length-gap", length_gap(320)),
    ):
        best = {}
        for arm in ARM_ORDER:
            out = subprocess.run(
                [sys.executable, script, "--worker", arm, a, b, "2"],
                capture_output=True, text=True, timeout=300,
            )
            if out.returncode != 0:
                print(f"  worker {arm} failed: {out.stderr.strip()[:200]}")
                best[arm] = float("nan")
            else:
                best[arm] = float(out.stdout.strip())
        b_a = best["B_two_row_full"] / best["A_full_matrix"]
        c_b = best["C_band_core"] / best["B_two_row_full"]
        d_c = best["D_levlab"] / best["C_band_core"]
        d_a = best["D_levlab"] / best["A_full_matrix"]
        print(f"  {name:14} A={best['A_full_matrix']*1000:8.3f} "
              f"B={best['B_two_row_full']*1000:8.3f} "
              f"C={best['C_band_core']*1000:8.3f} "
              f"D={best['D_levlab']*1000:8.3f}  "
              f"B/A={b_a:.2f}x C/B={c_b:.3f}x D/C={d_c:.2f}x D/A={d_a:.0f}x")
        results.setdefault("isolated", {})[name] = {
            "best_ms": {k: v * 1000 for k, v in best.items()},
            "ratio_B_A": b_a, "ratio_C_B": c_b, "ratio_D_C": d_c, "ratio_D_A": d_a,
        }


def worker(arm: str, a: str, b: str, k: str) -> None:
    """Entry point for --worker: print the best per-call seconds."""
    fn = ARMS[arm]
    args = (a, b, int(k))
    best = min(time_once(fn, args, min_time=0.05) for _ in range(7))
    print(repr(best))


def main() -> None:
    # Windows consoles and CI runners default stdout to cp1252/cp936, which
    # cannot encode the Chinese search terms printed below. Reconfigure to UTF-8
    # with replacement so the benchmark never dies on a print call.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # already UTF-8, or a stream that cannot be reconfigured

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="smoke run: fewer lengths, writes results-quick.json")
    parser.add_argument("--ci", action="store_true",
                        help="minimal run for CI: exercises every section once, "
                             "writes results-ci.json")
    parser.add_argument("--isolated", action="store_true",
                        help="also run the headline table in separate processes")
    parser.add_argument("--worker", nargs=4, metavar=("ARM", "A", "B", "K"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        worker(*args.worker)
        return

    print(f"levlab benchmark | Python {sys.version.split()[0]} | {sys.platform}")
    print("All headline numbers are ratios of adjacent arms, not raw milliseconds.")

    results = {"python": sys.version.split()[0], "platform": sys.platform,
               "arm_order": ARM_ORDER}

    if args.ci:
        # Just enough to prove every code path runs and stays correct.
        section_headline(results)
        section_short_strings(results)
        section_chinese(results)
        out_name = "results-ci.json"
    else:
        section_headline(results)
        section_length_sweep(results, lengths=(16, 64, 256, 1024) if args.quick
                             else (16, 32, 64, 128, 256, 512, 1024))
        section_cutoff_sweep(results)
        section_short_strings(results)
        section_chinese(results)
        section_memory(results, lengths=(64, 1024) if args.quick else (64, 256, 1024, 2048))
        if args.isolated:
            section_isolated(results)
        out_name = "results-quick.json" if args.quick else "results.json"

    out_path = Path(__file__).resolve().parent / out_name
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
