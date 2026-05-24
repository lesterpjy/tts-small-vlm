#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.7",
#   "seaborn>=0.13",
#   "numpy>=1.24",
# ]
# ///
"""Generate the final-report figures (matplotlib/seaborn).

Produces:
  - scaling_v2.pdf    (Figure 1): accuracy vs compute + temperature sweep
  - languages_v2.pdf  (Figure 2): per-language accuracy bar chart

Numbers inlined from main.tex (tab:rq1, tab:rq1b, tab:rq2, tab:rq3lang).
To regenerate:

    python figures/plot_figures.py
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% score interval for a binomial proportion."""
    denom = 1.0 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


# ---- Inlined paper data ----
Q35_BASELINES = [
    ("ZS",            1,  57.11),
    ("CoT",           1,  69.66),
    ("SC-N=8 (1k)",   8,  77.81),
    ("SC-N=8 (2k)",   8,  81.49),
    ("SC-N=16 (2k)", 16,  81.64),
]
Q25_BASELINES = [
    ("ZS",            1,  56.83),
    ("CoT",           1,  63.10),
    ("SC-N=8",        8,  66.42),
]
Q25_SEARCH_DEV = [
    ("DTR N=2,M=2",    6,   55.0),
    ("DTR + PRM-BAS", 13.4, 60.5),
    ("DTR N=4,M=4",   20,   61.5),
]
TEMP_SWEEP = [(0.3, 80.84), (0.5, 81.01), (0.7, 81.49), (0.9, 81.04)]
N_VAL_FULL = 4651

# tab:rq3lang -- per-language accuracy on full validation (n=4,651).
# Columns: name, n, Q2.5 SC-N=8, Q3.5 SC-N=8 (1k-tok), Q3.5 SC-N=8 (2k-tok)
LANGUAGES = [
    ("Arabic",    517, 59.8, 71.8, 74.5),
    ("Bulgarian", 400, 81.0, 92.0, 93.5),
    ("Chinese",   600, 69.0, 66.2, 74.7),
    ("Croatian",  585, 70.4, 84.6, 86.0),
    ("English",   347, 30.0, 56.2, 73.5),
    ("French",    224, 81.3, 92.0, 92.0),
    ("German",    279, 69.2, 84.9, 86.4),
    ("Hungarian", 535, 59.3, 77.9, 80.0),
    ("Italian",   562, 69.4, 81.9, 83.3),
    ("Polish",    100, 52.0, 55.0, 57.0),
    ("Serbian",   502, 70.5, 83.3, 84.7),
]
OVERALL_BEST = 81.5
DELTA_THRESHOLD_PP = 5.0


# ---- Style ----
sns.set_theme(style="ticks", context="paper")
LABEL_C = "#333"

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "axes.titleweight": "normal",
    "axes.labelcolor": LABEL_C,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#666",
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.color": LABEL_C,
    "ytick.color": LABEL_C,
    "pdf.fonttype": 42,  # TrueType embedding (editable in Illustrator)
    "ps.fonttype": 42,
})

# Wong (2011) colorblind-friendly palette.
# Greens vs vermillions stay distinguishable under deuteranopia, so DTR
# uses green instead of the orange-vermillion pair that would collide
# with the annotation arrows.
Q35_C = "#0173B2"        # blue            (Q3.5, 2k-tok)
Q35_LIGHT_C = "#56B4E9"  # sky blue        (Q3.5, 1k-tok -- paired shade)
Q25_C = "#949494"        # grey            (Q2.5)
DTR_C = "#029E73"        # green           (DTR / PRM-BAS scatter)
ARROW_C = "#D55E00"      # vermillion      (annotations, best-line)
HIGHLIGHT_C = "#D55E00"  # vermillion      (best value in panel b)


# =====================================================================
def plot_scaling(out_path: Path) -> None:
    """Figure 1: accuracy vs compute (a) and temperature sweep (b)."""
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.0),
                             gridspec_kw={"wspace": 0.26})

    # ---------- Panel (a) ----------
    ax = axes[0]

    xs_q35 = [x for _, x, _ in Q35_BASELINES]
    ys_q35 = [y for _, _, y in Q35_BASELINES]
    xs_q25 = [x for _, x, _ in Q25_BASELINES]
    ys_q25 = [y for _, _, y in Q25_BASELINES]
    xs_dtr = [x for _, x, _ in Q25_SEARCH_DEV]
    ys_dtr = [y for _, _, y in Q25_SEARCH_DEV]

    ax.plot(xs_q35, ys_q35,
            color=Q35_C, linewidth=1.2,
            marker="o", markersize=5.5,
            markerfacecolor=Q35_C, markeredgecolor="white",
            markeredgewidth=0.8,
            label="Qwen3.5-4B (val-full)", zorder=5)

    ax.plot(xs_q25, ys_q25,
            color=Q25_C, linewidth=1.0,
            linestyle="--",
            marker="s", markersize=4.5,
            markerfacecolor="white", markeredgecolor=Q25_C,
            markeredgewidth=1.0,
            label="Qwen2.5-VL-7B (val-full)", zorder=4)

    ax.scatter(xs_dtr, ys_dtr,
               s=42, c=DTR_C, marker="^",
               edgecolors="white", linewidth=0.7,
               label="DTR / PRM-BAS (Q2.5, dev-200)",
               zorder=4)

    # Two-axis scaling guides
    ax.plot([8, 8], [77.81, 81.49],
            color=ARROW_C, linewidth=1.1, linestyle=":",
            zorder=3, alpha=0.95)
    ax.plot([8, 16], [81.49, 81.64],
            color=ARROW_C, linewidth=1.1, linestyle=":",
            zorder=3, alpha=0.95)

    ax.annotate("$+3.7$ pp\n(token budget)",
                xy=(8, 79.65), xytext=(10, 75.5),
                fontsize=7, color=ARROW_C, fontweight="bold",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=ARROW_C, lw=0.9,
                                shrinkA=0, shrinkB=3))

    ax.annotate("$+0.15$ pp\n(chain count)",
                xy=(11.3, 81.57), xytext=(11.3, 86),
                fontsize=7, color=ARROW_C, fontweight="bold",
                ha="center", va="center",
                arrowprops=dict(arrowstyle="->", color=ARROW_C, lw=0.9,
                                shrinkA=0, shrinkB=3))

    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 8, 16, 20])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "20"])
    ax.minorticks_off()
    ax.set_xlim(0.85, 25)
    ax.set_ylim(50, 90)
    ax.set_xlabel("VLM calls per question (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(a) Accuracy vs. compute")

    ax.legend(loc="upper left", frameon=True, framealpha=0.92,
              edgecolor="#CCC", fancybox=False,
              borderpad=0.3, labelspacing=0.22,
              handlelength=1.4, handletextpad=0.4,
              fontsize=6.5)

    ax.grid(True, axis="y", linestyle="-", linewidth=0.4,
            color="#DDD", alpha=0.8)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)

    # ---------- Panel (b) ----------
    ax = axes[1]
    ts = [t for t, _ in TEMP_SWEEP]
    accs = [a for _, a in TEMP_SWEEP]
    err_lo, err_hi = [], []
    for _, a in TEMP_SWEEP:
        lo, hi = wilson_ci(a / 100, N_VAL_FULL)
        err_lo.append(a - lo * 100)
        err_hi.append(hi * 100 - a)

    ax.errorbar(ts, accs,
                yerr=[err_lo, err_hi],
                color=Q35_C, linewidth=1.2,
                marker="o", markersize=6,
                markerfacecolor=Q35_C, markeredgecolor="white",
                markeredgewidth=0.8,
                capsize=3, capthick=1.0, elinewidth=1.0,
                ecolor=Q35_C, zorder=5)

    best_t = max(TEMP_SWEEP, key=lambda x: x[1])[0]
    for t, a in TEMP_SWEEP:
        is_best = (t == best_t)
        xy_offset = (8, -6) if t == 0.3 else (8, 0)
        ax.annotate(f"{a:.2f}",
                    xy=(t, a), xytext=xy_offset, textcoords="offset points",
                    ha="left", va="center",
                    fontsize=7,
                    color=HIGHLIGHT_C if is_best else LABEL_C,
                    fontweight="bold" if is_best else "normal")

    ax.axvline(best_t, color="#BBB", linewidth=0.6, linestyle=":",
               zorder=1, alpha=0.7)

    ax.set_xticks([0.3, 0.5, 0.7, 0.9])
    ax.set_xlim(0.22, 1.02)
    ax.set_ylim(79, 83.5)
    ax.set_xlabel("Sampling temperature $T$")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(b) Temperature sweep (Q3.5 SC-$N{=}8$, 2k-tok)")

    ax.grid(True, axis="y", linestyle="-", linewidth=0.4,
            color="#DDD", alpha=0.8)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
def plot_languages(out_path: Path) -> None:
    """Figure 2: per-language accuracy across Q2.5 SC, Q3.5 1k, Q3.5 2k."""
    names = [d[0] for d in LANGUAGES]
    q25_sc = np.array([d[2] for d in LANGUAGES])
    q35_1k = np.array([d[3] for d in LANGUAGES])
    q35_2k = np.array([d[4] for d in LANGUAGES])
    deltas = q35_2k - q35_1k

    fig, ax = plt.subplots(figsize=(7.5, 2.9))

    x = np.arange(len(names))
    width = 0.27

    # Best-line drawn first so it sits behind the bars.
    ax.axhline(OVERALL_BEST, linestyle="--", color=ARROW_C,
               linewidth=0.9, alpha=0.85, zorder=1,
               label=f"Overall best ({OVERALL_BEST}%)")

    ax.bar(x - width, q25_sc, width, color=Q25_C,
           edgecolor="white", linewidth=0.3,
           label=r"Q2.5 SC-$N{=}8$", zorder=2)
    ax.bar(x, q35_1k, width, color=Q35_LIGHT_C,
           edgecolor="white", linewidth=0.3,
           label=r"Q3.5 SC-$N{=}8$ (1k-tok)", zorder=2)
    ax.bar(x + width, q35_2k, width, color=Q35_C,
           edgecolor="white", linewidth=0.3,
           label=r"Q3.5 SC-$N{=}8$ (2k-tok)", zorder=2)

    # Vermillion delta annotations for languages with >= +5pp budget gain.
    for i, d in enumerate(deltas):
        if d > DELTA_THRESHOLD_PP:
            ax.annotate(
                f"$+{int(round(d))}$",
                xy=(x[i] + width, q35_2k[i]), xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=7.5, color=ARROW_C, fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=28, ha="right",
                       rotation_mode="anchor")
    ax.set_ylim(20, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.tick_params(axis="x", pad=1)

    ax.legend(loc="lower left", frameon=True, framealpha=0.94,
              edgecolor="#CCC", fancybox=False,
              ncol=2, borderpad=0.35, columnspacing=1.0,
              labelspacing=0.22, handlelength=1.4,
              handletextpad=0.4, fontsize=6.5)

    ax.grid(True, axis="y", linestyle="-", linewidth=0.4,
            color="#DDD", alpha=0.8)
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved: {out_path}")


# =====================================================================
if __name__ == "__main__":
    here = Path(__file__).parent
    plot_scaling(here / "scaling_v2.pdf")
    plot_languages(here / "languages_v2.pdf")
