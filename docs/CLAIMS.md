# Claims and evidence

Every performance claim in this repo, with the measurement that supports it and
the conditions under which it stops being true.

Measured on CPython 3.13.12 / Windows x64, 2026-10-17. Re-measure before quoting.

| # | Claim | Evidence | Fails when |
|---|---|---|---|
| 1 | Band + early exit is ~39x faster than a full matrix at n=320, k=2 | `C/B = 0.026x`, isolated processes | — |
| 2 | Rolling two rows saves **memory**, not time | `B/A = 0.98x` across n=16..1024 | — |
| 3 | Banded cutoff is 125x faster at n=1024, k=2 | `D/B = 0.008x`, length sweep | — |
| 4 | Band is **slower** on short strings without a guard | crossover sweep: 1.1-1.4x slower below the table in `BAND_MIN_LENGTH` | guarded since 1.0.0 |
| 5 | Band is **slower** when `k >= min(len)` | `D/B = 1.10x` unguarded at n=512, k=512 | guarded since 1.0.0 |
| 6 | Peak memory drops 130 MiB → 32 KiB at n=2048, k=2 | `tracemalloc` min-of-5 | — |
| 7 | Length-gap rejection is O(1), not a banded speedup | `D/C = 0.00x`; one `abs()` comparison | — |
| 8 | The band's advantage decays to 1x by k≈300 at n=512 | cutoff sweep, 80x at k=0 → 1x at k=300 | — |

## Claims deliberately NOT made

- **No universal speedup multiplier.** The number is a function of both `n` and
  `k`. Quoting one figure without both is meaningless.
- **No "O(k·n) always".** That bound holds only while the band is narrower than
  the matrix, i.e. `k < min(len(a), len(b))`. Beyond that the guard sends the
  call to the plain scan and the bound is `O(n·m)`.
- **No comparison against a compiled implementation.** This is pure Python. A
  C/Rust implementation of the same algorithm would beat all arms here; the
  ratios describe CPython's cost model, not the algorithm's asymptotic rank.
- **No claim that two-row is faster.** It is not. It saves memory.

## Why the numbers differ from other write-ups

Two structural differences, both of which inflate other projects' reported
speedups:

1. **Baseline choice.** Comparing against a full matrix that reallocates per call
   mixes "no matrix allocation" into the speedup. That part is worth 0.98x
   (nothing) in time — it only matters for memory. This repo isolates it as arm
   B.
2. **Short-circuit accounting.** A length-gap check is one integer comparison.
   Counting it as a banded win produces five-digit multipliers that have nothing
   to do with the band.

## Measurement caveats

- **Absolute milliseconds drift up to 3x** across cases inside one process. The
  first draft of this benchmark measured arm A at 29.70 ms in a process that had
  already run other cases, and 9.17 ms in a clean one. GC counters showed zero
  collections, so it was allocator state, not GC. Only within-round ratios are
  reported.
- **Short-string timings (1-20 µs) are near the noise floor.** That is why the
  guards are verified by path assertions (`tests/test_dispatch.py`) rather than
  by stopwatch.
- **Crossover points move with interpreter version.** The `BAND_MIN_LENGTH` table
  is a measured snapshot, not a constant of nature. Re-run
  `benchmarks/crossover.py` on your runtime if the difference matters.
