#!/usr/bin/env python
"""Generate the figures in assets/ from benchmarks/results.json.

Rule followed here: plot *ratios*, never absolute milliseconds. Absolute
wall-clock drifts by up to 3x across cases inside a single process, so a
millisecond axis produces curves whose shape is measurement artefact.

Run with the interpreter that has matplotlib installed:
    python assets/make_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = json.loads((ROOT / "benchmarks" / "results.json").read_text(encoding="utf-8"))
OUT = ROOT / "assets"

# Warm editorial palette: espresso brown / amber gold / terracotta.
INK = "#2b2119"
PAPER = "#faf6f0"
AMBER = "#c8892b"
TERRACOTTA = "#b4552f"
OLIVE = "#6b6b3a"
MUTED = "#8a7a68"

plt.rcParams.update({
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "font.size": 10,
    "axes.grid": True,
    "grid.color": "#ddd2c4",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def fig_speedup_vs_length() -> None:
    """Band vs two-row as a function of string length."""
    rows = RESULTS["length_sweep"]
    ns = [r["n"] for r in rows]
    ratios = [r["ratio_D_B"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(ns, ratios, "o-", color=TERRACOTTA, linewidth=2.2, markersize=6,
            label="banded / two-row")
    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.2,
               label="break-even (no gain)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("shorter string length n")
    ax.set_ylabel("time ratio (lower = band is faster)")
    ax.set_title("Banded cutoff vs plain two-row scan, k=2", loc="left",
                 fontsize=12, fontweight="bold")
    ax.set_ylim(0.004, 2.0)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0))

    # Label only a few points, alternating above/below to avoid collisions.
    for idx, (x, y) in enumerate(zip(ns, ratios)):
        if idx % 2 and idx != len(ns) - 1:
            continue
        offset = (0, -16) if idx % 4 == 0 else (0, -16)
        ax.annotate(f"{y:.3f}x", (x, y), textcoords="offset points",
                    xytext=offset, ha="center", fontsize=8, color=TERRACOTTA)
    ax.annotate("below this line the band pays for itself",
                xy=(ns[2], 0.62), fontsize=8, color=MUTED, style="italic")
    fig.tight_layout()
    fig.savefig(OUT / "speedup-vs-length.png", dpi=180)
    plt.close(fig)


def fig_cutoff_sweep() -> None:
    """How the band's advantage decays as the cutoff widens.

    Plotted as speedup (1/ratio) rather than ratio: the story is "350x at k=0
    decays to 1x by k~300", and a speedup axis makes the decay readable in one
    glance. Note that at large k the guarded and unguarded curves converge,
    because the guard's job there is only to avoid a regression, not to win.
    """
    rows = RESULTS["cutoff_sweep"]
    ks = [r["k"] for r in rows]
    # speedup of the band over the plain two-row scan
    d_b = [1.0 / r["ratio_D_B"] for r in rows]
    c_b = [1.0 / r["ratio_C_B"] for r in rows]

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.plot(ks, c_b, "s--", color=OLIVE, linewidth=1.5, markersize=5,
            label="unguarded band (C)")
    ax.plot(ks, d_b, "o-", color=TERRACOTTA, linewidth=2.3, markersize=6,
            label="shipped, with guards (D)")
    ax.axhline(1.0, color=MUTED, linestyle="--", linewidth=1.2,
               label="break-even")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("cutoff k")
    ax.set_ylabel("speedup vs two-row scan (higher = band is better)")
    ax.set_title("The band's advantage decays as the cutoff widens (n=512)",
                 loc="left", fontsize=12, fontweight="bold")
    ax.set_ylim(0.5, 800)
    ax.legend(frameon=False, loc="lower left")

    # Shade the region where the band no longer helps.
    ax.axvspan(300, ks[-1], color=AMBER, alpha=0.10)
    ax.annotate("band no longer\npays for itself",
                xy=(430, 60), fontsize=8, color=AMBER, ha="center",
                style="italic")
    ax.axvline(512, color=AMBER, linestyle=":", linewidth=1.5)
    ax.annotate("k = n: band spans\nthe whole matrix", xy=(512, 3.0),
                xytext=(-104, 34), textcoords="offset points",
                fontsize=8, color=AMBER,
                arrowprops=dict(arrowstyle="->", color=AMBER, lw=1.2))

    # Headline numbers at both ends.
    ax.annotate(f"{d_b[0]:.0f}x", (ks[0], d_b[0]), textcoords="offset points",
                xytext=(2, 10), fontsize=9, color=TERRACOTTA, fontweight="bold")
    ax.annotate("no gain", (ks[-1], d_b[-1]), textcoords="offset points",
                xytext=(-6, -16), ha="right", fontsize=8, color=MUTED)

    fig.tight_layout()
    fig.savefig(OUT / "cutoff-sweep.png", dpi=180)
    plt.close(fig)


def fig_attribution() -> None:
    """The four arms, showing where the speedup actually comes from."""
    iso = RESULTS["isolated"]["near-match"]
    best = iso["best_ms"]
    arms = ["A_full_matrix", "B_two_row_full", "C_band_core", "D_levlab"]
    labels = ["A\nfull matrix", "B\ntwo rows", "C\nband+early", "D\nshipped"]
    values = [best[a] for a in arms]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 4.0),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    colors = [MUTED, OLIVE, AMBER, TERRACOTTA]
    bars = ax1.bar(labels, values, color=colors, width=0.62)
    ax1.set_yscale("log")
    ax1.set_ylabel("ms per call (log scale)")
    ax1.set_title("Absolute cost per arm", loc="left", fontsize=11, fontweight="bold")
    for bar, v in zip(bars, values):
        ax1.annotate(f"{v:.3f} ms", (bar.get_x() + bar.get_width() / 2, v),
                     textcoords="offset points", xytext=(0, 5), ha="center", fontsize=8)

    steps = ["B/A\ntwo rows", "C/B\nband+early", "D/C\nguards"]
    ratios = [iso["ratio_B_A"], iso["ratio_C_B"], iso["ratio_D_C"]]
    # Plot speedup (1/ratio) so a longer bar always means a bigger win. Plotting
    # the ratio itself on a log axis makes 0.98x look like a long bar, which
    # reads as "big improvement" when it is actually "no change at all".
    speedups = [1.0 / r for r in ratios]
    colors2 = [OLIVE, TERRACOTTA, AMBER]
    ax2.barh(steps, speedups, color=colors2, height=0.55)
    ax2.axvline(1.0, color=MUTED, linestyle="--", linewidth=1.2)
    ax2.set_xscale("log")
    ax2.set_xlabel("speedup vs previous arm (higher = that change helped)")
    ax2.set_title("Which change actually bought the time", loc="left",
                  fontsize=11, fontweight="bold")
    for i, (s, r) in enumerate(zip(speedups, ratios)):
        if s < 1.05:
            label = "no gain (memory only)" if i == 0 else "no change"
        elif s < 10:
            label = f"{s:.1f}x faster"
        else:
            label = f"{s:.0f}x faster"
        ax2.annotate(label, (s, i), textcoords="offset points",
                     xytext=(6, 0), va="center", fontsize=8)

    fig.suptitle("Four-arm attribution: n=320, k=2, isolated processes",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "four-arms.png", dpi=180)
    plt.close(fig)


def fig_memory() -> None:
    """Peak memory across implementations, log scale."""
    rows = RESULTS["memory"]
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.plot(ns, [r["A_KiB"] for r in rows], "o-", color=MUTED, linewidth=2,
            label="full matrix (A)")
    ax.plot(ns, [r["B_KiB"] for r in rows], "s-", color=OLIVE, linewidth=2,
            label="two rows (B)")
    ax.plot(ns, [r["D_KiB"] for r in rows], "^-", color=TERRACOTTA, linewidth=2.2,
            label="banded, k=2 (D)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("shorter string length n")
    ax.set_ylabel("peak memory, KiB (log scale)")
    ax.set_title("Auxiliary memory: the one unambiguous win", loc="left",
                 fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    last = rows[-1]
    ax.annotate(f"{last['A_KiB']/1024:.0f} MiB", (last["n"], last["A_KiB"]),
                textcoords="offset points", xytext=(-8, 6), ha="right", fontsize=8)
    ax.annotate(f"{last['D_KiB']:.0f} KiB", (last["n"], last["D_KiB"]),
                textcoords="offset points", xytext=(-8, -14), ha="right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "memory.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_speedup_vs_length()
    fig_cutoff_sweep()
    fig_attribution()
    fig_memory()
    for p in sorted(OUT.glob("*.png")):
        print(f"wrote {p.relative_to(ROOT)}  ({p.stat().st_size // 1024} KiB)")
