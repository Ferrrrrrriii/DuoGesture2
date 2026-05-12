"""
plot_beat_vs_semantic_frequency.py
─────────────────────────────────────────────────────────────────────────────
Publication-quality 3-panel figure arguing why individual-component frequency
prediction belongs in the *beat* pathway, not the *semantic* pathway.

Panel A  – Oscillation cycles per window
           Beat windows contain 5+ cycles; semantic windows contain only 1-2.
           Below ~3 cycles Welch PSD cannot produce reliable peak estimates.

Panel B  – Duration × frequency scatter with iso-cycle contours
           Most semantic windows fall in the unreliable zone (< 3 cycles).
           Beat windows span a large range of durations with predictable freq.

Panel C  – Phase-Locking Value (shoulder ↔ forearm coupling)
           Semantic gestures are strongly coupled → no independent component
           frequencies to predict.  Beat gestures are moderately coupled →
           each component follows its own rhythm.

Usage
─────
python utils/plot_beat_vs_semantic_frequency.py \\
    --input outputs/window_records_test.json \\
    --output outputs/beat_vs_semantic_frequency.pdf

Additional optional flags
  --summary_json outputs/gt_semantic_beat_validation_testsplit_full.json
      (adds bootstrap CIs to Panel C from the aggregated run)
  --no_strip      suppress individual data-point strips in violins
  --dpi 300       output DPI for raster formats (PNG)
"""

import argparse
import json
import math
import os
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde

warnings.filterwarnings("ignore")

# ─────────────────────────── constants ────────────────────────────────────

BEAT_COLOR   = "#2166ac"   # sturdy blue
SEM_COLOR    = "#d6604d"   # warm orange-red
BEAT_LIGHT   = "#a6cee3"
SEM_LIGHT    = "#f4a582"
UNRELIABLE_THRESH_CYCLES = 3   # < 3 cycles → Welch PSD peak is unreliable

# ─────────────────────────── helpers ──────────────────────────────────────

def _clean(arr):
    """Remove None / NaN from a list, return numpy float64 array."""
    return np.array([x for x in arr if x is not None and math.isfinite(x)], dtype=np.float64)


def _violin_parts(ax, data, position, color, width=0.35, show_median=True):
    """Draw a clean half-horizontal violin at *position* on the y-axis."""
    data = _clean(data)
    if len(data) < 10:
        return
    kde = gaussian_kde(data, bw_method="scott")
    x_range = np.linspace(max(data.min(), 0), np.percentile(data, 99), 500)
    density = kde(x_range)
    density = density / density.max() * width
    ax.fill_betweenx(x_range, position - density, position + density,
                     color=color, alpha=0.55, linewidth=0)
    ax.plot(position - density, x_range, color=color, lw=0.7, alpha=0.6)
    ax.plot(position + density, x_range, color=color, lw=0.7, alpha=0.6)
    q25, q50, q75 = np.percentile(data, [25, 50, 75])
    ax.plot([position - width * 0.55, position + width * 0.55], [q50, q50],
            color="white", lw=2.5, solid_capstyle="round", zorder=4)
    ax.plot([position - width * 0.45, position + width * 0.45], [q25, q25],
            color=color, lw=1.2, alpha=0.9, zorder=3)
    ax.plot([position - width * 0.45, position + width * 0.45], [q75, q75],
            color=color, lw=1.2, alpha=0.9, zorder=3)
    ax.vlines(position, q25, q75, color=color, lw=1.5, zorder=3)


def _strip(ax, data, position, color, rng, alpha=0.25, jitter=0.12):
    """Jittered dot strip over the violin."""
    data = _clean(data)
    if len(data) == 0:
        return
    # clip for readability (p99)
    cap = np.percentile(data, 99)
    data = data[data <= cap]
    xs = position + rng.uniform(-jitter, jitter, len(data))
    ax.scatter(xs, data, s=3, color=color, alpha=alpha, linewidths=0, zorder=2)


# ─────────────────────────── panel A ─────────────────────────────────────

def draw_panel_A(ax, beat_recs, sem_recs, rng, show_strip):
    """Cycles-per-window violin (log-y).  Main reliability argument."""

    beat_cycles = _clean([r["cycles"] for r in beat_recs])
    sem_cycles  = _clean([r["cycles"] for r in sem_recs])

    # Log-transform for violin (fit in log space, display in log space)
    def log_violin(ax, data, pos, color, width=0.32):
        data = data[data > 0]
        log_data = np.log10(data)
        kde = gaussian_kde(log_data, bw_method="scott")
        x_range = np.linspace(log_data.min(), np.percentile(log_data, 99), 500)
        density = kde(x_range)
        density = density / density.max() * width
        ax.fill_betweenx(10 ** x_range, pos - density, pos + density,
                         color=color, alpha=0.55, linewidth=0)
        ax.plot(pos - density, 10 ** x_range, color=color, lw=0.8, alpha=0.7)
        ax.plot(pos + density, 10 ** x_range, color=color, lw=0.8, alpha=0.7)
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        ax.plot([pos - width * 0.6, pos + width * 0.6], [q50, q50],
                color="white", lw=2.5, solid_capstyle="round", zorder=4)
        ax.vlines(pos, q25, q75, color=color, lw=1.8, zorder=3)
        return np.percentile(data, 50)

    m_beat = log_violin(ax, beat_cycles, 1, BEAT_COLOR)
    m_sem  = log_violin(ax, sem_cycles,  2, SEM_COLOR)

    if show_strip:
        for data, pos, col in [(beat_cycles, 1, BEAT_COLOR), (sem_cycles, 2, SEM_COLOR)]:
            cap = np.percentile(data[data > 0], 98)
            d2 = data[(data > 0) & (data <= cap)]
            xs = pos + rng.uniform(-0.1, 0.1, len(d2))
            ax.scatter(xs, d2, s=2.5, color=col, alpha=0.20, linewidths=0, zorder=2)

    # unreliable zone
    ax.axhline(UNRELIABLE_THRESH_CYCLES, color="#555555", lw=1.2, ls="--", zorder=5)
    ax.text(2.6, UNRELIABLE_THRESH_CYCLES * 1.18, f"≥{UNRELIABLE_THRESH_CYCLES} cycles\n(reliable)",
            fontsize=7.5, color="#333333", va="bottom", ha="right")
    ax.text(2.6, UNRELIABLE_THRESH_CYCLES * 0.78, "< 3 cycles\n(unreliable)",
            fontsize=7.5, color="#cc2222", va="top", ha="right")

    # median labels — placed safely within axis limits
    ax.text(0.6, m_beat * 2.4,
            f"median\n{m_beat:.1f} cycles",
            fontsize=8, color=BEAT_COLOR, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=BEAT_COLOR, alpha=0.85))
    ax.annotate("", xy=(1.0, m_beat), xytext=(0.75, m_beat * 2.2),
                arrowprops=dict(arrowstyle="->", color=BEAT_COLOR, lw=0.9))

    ax.text(2.42, max(m_sem * 0.55, 0.65),
            f"median\n{m_sem:.1f} cycles",
            fontsize=8, color=SEM_COLOR, ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=SEM_COLOR, alpha=0.85))
    ax.annotate("", xy=(2.0, m_sem), xytext=(2.25, max(m_sem * 0.65, 0.8)),
                arrowprops=dict(arrowstyle="->", color=SEM_COLOR, lw=0.9))

    ax.set_yscale("log")
    ax.set_ylim(0.5, 120)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["Beat\nwindows", "Semantic\nwindows"], fontsize=9)
    ax.set_xlim(0.45, 2.55)
    ax.set_ylabel("Oscillation cycles per window", fontsize=9)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda val, _: f"{val:.0f}" if val >= 1 else f"{val:.1f}"))
    ax.set_title("(a)  Cycle count per window", fontsize=10, fontweight="bold", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)


# ─────────────────────────── panel B ─────────────────────────────────────

def draw_panel_B(ax, beat_recs, sem_recs):
    """Duration × frequency scatter with iso-cycle contours."""

    def scatter_data(recs):
        dur = _clean([r["duration_s"]    for r in recs])
        hz  = _clean([r["shoulder_hz"]   for r in recs])
        # align
        mask = np.array([
            r["duration_s"] is not None and math.isfinite(r["duration_s"]) and
            r["shoulder_hz"] is not None and math.isfinite(r["shoulder_hz"])
            for r in recs
        ])
        dur = np.array([r["duration_s"]  for r in recs], dtype=object)[mask].astype(float)
        hz  = np.array([r["shoulder_hz"] for r in recs], dtype=object)[mask].astype(float)
        return dur, hz

    b_dur, b_hz = scatter_data(beat_recs)
    s_dur, s_hz = scatter_data(sem_recs)

    # Scatter (sub-sampled to ≤800 pts each for readability)
    rng_local = np.random.RandomState(7)
    def subsample(arr, n=800):
        if len(arr) <= n:
            return arr
        idx = rng_local.choice(len(arr), n, replace=False)
        return arr[idx]

    ax.scatter(subsample(b_dur), subsample(b_hz), s=6, color=BEAT_COLOR,
               alpha=0.18, linewidths=0, label="Beat", rasterized=True, zorder=2)
    ax.scatter(subsample(s_dur), subsample(s_hz), s=6, color=SEM_COLOR,
               alpha=0.35, linewidths=0, label="Semantic", rasterized=True, zorder=3)

    # 2D KDE contours in log(dur) × hz space
    def add_kde_contours(ax, dur, hz, color, levels=3, zorder=5):
        log_dur = np.log10(np.clip(dur, 1e-3, None))
        try:
            from scipy.stats import gaussian_kde as gkde
            xy = np.vstack([log_dur, hz])
            kde = gkde(xy, bw_method=0.25)
            ld_grid = np.linspace(log_dur.min() - 0.1, log_dur.max() + 0.1, 60)
            hz_grid = np.linspace(0.25, 5.4, 50)
            LD, HZ = np.meshgrid(ld_grid, hz_grid)
            Z = kde(np.vstack([LD.ravel(), HZ.ravel()])).reshape(LD.shape)
            ax.contour(10 ** LD, HZ, Z,
                       levels=levels, colors=[color], linewidths=[0.9, 1.3, 1.8],
                       alpha=0.75, zorder=zorder)
        except Exception:
            pass

    add_kde_contours(ax, b_dur, b_hz, BEAT_COLOR, zorder=5)
    add_kde_contours(ax, s_dur, s_hz, SEM_COLOR,  zorder=6)

    # Iso-cycle curves:  freq = N / duration
    dur_grid = np.linspace(0.35, 55, 400)
    for n_cyc, ls, lw_val in [(1, ":", 0.9), (2, "--", 1.0), (3, "-", 1.2), (5, "-", 0.8)]:
        hz_curve = n_cyc / dur_grid
        mask_vis = (hz_curve >= 0.25) & (hz_curve <= 5.5)
        ax.plot(dur_grid[mask_vis], hz_curve[mask_vis],
                color="#444444", lw=lw_val, ls=ls, alpha=0.75, zorder=1)
        # Place label at the bottom-right end of each curve (low hz, long duration)
        vis_dur = dur_grid[mask_vis]
        vis_hz  = hz_curve[mask_vis]
        if len(vis_dur) > 0:
            # rightmost visible point
            x_lab = vis_dur[-1]
            y_lab = vis_hz[-1]
            if 0.35 <= x_lab and 0.27 <= y_lab <= 5.2:
                ax.text(x_lab * 1.03, y_lab, f"n={n_cyc}",
                        fontsize=6.5, color="#444444", va="center", ha="left",
                        clip_on=True)

    # shade unreliable zone (< 3 cycles)
    dur_shade = np.linspace(0.35, 55, 400)
    hz_3 = 3.0 / dur_shade
    hz_3_vis = np.clip(hz_3, 0.25, 5.4)
    ax.fill_between(dur_shade, hz_3_vis, 5.6, color="#f0c0c0", alpha=0.18, zorder=0)
    # put the label safely in the upper LEFT where the iso-cycle labels are NOT
    ax.text(1.2, 4.9, "unreliable\nzone  (< 3 cycles)", fontsize=7, color="#aa3333", va="top", ha="left")

    ax.set_xscale("log")
    ax.set_xlim(0.35, 60)
    ax.set_ylim(0.25, 5.5)
    ax.set_xlabel("Window duration  (s, log scale)", fontsize=9)
    ax.set_ylabel("Shoulder peak frequency  (Hz)", fontsize=9)
    ax.set_title("(b)  Duration vs. frequency  (iso-cycle contours)", fontsize=10, fontweight="bold", pad=6)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda val, _: f"{val:.0f}" if val >= 1 else f"{val:.1f}"))
    ax.legend(fontsize=8, markerscale=3, frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)

    # Annotation box
    b_pct95_cyc = np.percentile(b_dur * b_hz, 50)
    s_pct50_cyc = np.median(s_dur * s_hz)
    ax.text(0.03, 0.04,
            f"Beat median ≈{b_pct95_cyc:.1f} cycles\nSemantic median ≈{s_pct50_cyc:.1f} cycles",
            transform=ax.transAxes, fontsize=7.5, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))


# ─────────────────────────── panel C ─────────────────────────────────────

def draw_panel_C(ax, beat_recs, sem_recs, summary_json=None):
    """Phase-Locking Value violin + optional bootstrap CIs from summary."""

    beat_plv = _clean([r["plv"] for r in beat_recs])
    sem_plv  = _clean([r["plv"] for r in sem_recs])

    # --- violin fill
    def h_violin(ax, data, ypos, color, width=0.28):
        kde = gaussian_kde(data, bw_method="scott")
        x_range = np.linspace(data.min(), data.max(), 400)
        density = kde(x_range)
        density = density / density.max() * width
        ax.fill_between(x_range, ypos - density, ypos + density,
                        color=color, alpha=0.55, linewidth=0)
        ax.plot(x_range, ypos - density, color=color, lw=0.8, alpha=0.7)
        ax.plot(x_range, ypos + density, color=color, lw=0.8, alpha=0.7)
        q25, q50, q75 = np.percentile(data, [25, 50, 75])
        ax.plot([q25, q75], [ypos, ypos], color=color, lw=2.0, alpha=0.9, zorder=3)
        ax.plot([q50, q50], [ypos - width * 0.65, ypos + width * 0.65],
                color="white", lw=2.5, solid_capstyle="round", zorder=4)
        return q50

    m_beat = h_violin(ax, beat_plv, 1, BEAT_COLOR)
    m_sem  = h_violin(ax, sem_plv,  2, SEM_COLOR)

    # Bootstrap CIs from summary JSON
    if summary_json:
        try:
            with open(summary_json) as f:
                summ = json.load(f)
            cr = summ.get("category_reports", {})
            for ypos, cat_key, color in [(1, "beat", BEAT_COLOR), (2, "semantic", SEM_COLOR)]:
                plv_data = cr.get(cat_key, {}).get("metrics", {}).get("plv", {})
                bs = plv_data.get("bootstrap_window_mean", {})
                ci_lo = bs.get("ci_low")
                ci_hi = bs.get("ci_high")
                if ci_lo is not None and ci_hi is not None:
                    ax.barh(ypos, ci_hi - ci_lo, left=ci_lo, height=0.08,
                            color=color, alpha=0.85, zorder=5)
        except Exception:
            pass  # no CIs if anything goes wrong

    # Vertical line at 0.5 (strong coupling threshold)
    ax.axvline(0.5, color="#555555", lw=1.0, ls="--", alpha=0.7)
    ax.text(0.505, 2.55, "high coupling\n(PLV > 0.5)", fontsize=7, color="#555555", va="top")

    # Median labels for beat and semantic — placed on the violins' right tails
    b_median_plv = float(np.median(beat_plv))
    s_median_plv = float(np.median(sem_plv))
    ax.text(0.97, 1.0, f"median {b_median_plv:.2f}",
            fontsize=8, color=BEAT_COLOR, ha="right", va="center", style="italic")
    ax.text(0.97, 2.0, f"median {s_median_plv:.2f}",
            fontsize=8, color=SEM_COLOR, ha="right", va="center", style="italic")

    # Gap annotation above both violins — clear white space around y=2.5
    gap_y = 2.5
    ax.annotate("", xy=(m_sem, gap_y), xytext=(m_beat, gap_y),
                arrowprops=dict(arrowstyle="<->", color="#333333", lw=1.4))
    ax.text((m_beat + m_sem) / 2, gap_y + 0.06, f"Δ PLV = {m_sem - m_beat:.2f}",
            fontsize=8, ha="center", va="bottom", color="#333333",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#999999", alpha=0.85))

    ax.set_xlim(0.0, 1.02)
    ax.set_yticks([1, 2])
    ax.set_yticklabels(["Beat\nwindows", "Semantic\nwindows"], fontsize=9)
    ax.set_ylim(0.35, 2.85)
    ax.set_xlabel("Phase-Locking Value  (shoulder ↔ forearm)", fontsize=9)
    ax.set_title("(c)  Inter-joint coupling (PLV)", fontsize=10, fontweight="bold", pad=6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)

    # Explanation text
    ax.text(0.02, 0.07,
            "High PLV → joints move as one unit\n→ no individual component rhythm to predict",
            transform=ax.transAxes, fontsize=7, va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.85))


# ─────────────────────────── main ─────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",        type=str, default="outputs/window_records_test.json")
    parser.add_argument("--output",       type=str, default="outputs/beat_vs_semantic_frequency.pdf")
    parser.add_argument("--summary_json", type=str, default="outputs/gt_semantic_beat_validation_testsplit_full.json")
    parser.add_argument("--no_strip",     action="store_true")
    parser.add_argument("--dpi",          type=int, default=300)
    args = parser.parse_args()

    with open(args.input) as f:
        records = json.load(f)

    beat_recs = [r for r in records if r["category"] == "beat"]
    sem_recs  = [r for r in records if r["category"] == "semantic"]
    rng = np.random.RandomState(42)

    # ── Figure layout ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.6),
                             gridspec_kw={"wspace": 0.38, "left": 0.06, "right": 0.98,
                                          "top": 0.88, "bottom": 0.14})

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
    })

    draw_panel_A(axes[0], beat_recs, sem_recs, rng, show_strip=not args.no_strip)
    draw_panel_B(axes[1], beat_recs, sem_recs)
    draw_panel_C(axes[2], beat_recs, sem_recs,
                 summary_json=args.summary_json if os.path.exists(args.summary_json) else None)

    # ── Global legend / title ─────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(color=BEAT_COLOR, alpha=0.75, label="Beat windows"),
        mpatches.Patch(color=SEM_COLOR,  alpha=0.75, label="Semantic windows"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=2,
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.975),
               columnspacing=2.0)

    n_beat = len(beat_recs)
    n_sem  = len(sem_recs)
    fig.text(0.5, 0.01,
             f"BEAT2 English test split  ·  {n_beat} beat windows  ·  {n_sem} semantic windows  "
             f"·  min window ≥ 15 frames  ·  all 25 speakers",
             ha="center", fontsize=7, color="#666666")

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    # Also save PNG next to it
    png_path = os.path.splitext(args.output)[0] + ".png"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    main()
