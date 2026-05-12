#!/usr/bin/env python3
"""
DuoGesture – Ablation Study Table (publication-ready)
Rows: model variants with MoCLIP / S-VIB / Physics toggles
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────
# columns: Model | MoCLIP | S-VIB | Physics | FGD↓ | BC↑ | L1 Vel↓
rows = [
    # (label, moclip, svib, phys, fgd, bc, div)
    ("DuoGesture (baseline)",            False, False, False, 0.4380, None,  None ),
    ("+ MoCLIP",                        True,  False, False, 0.4263, None,  None ),
    ("+ MoCLIP + S-VIB",                True,  True,  False, None,   None,  None ),  # not yet run
    ("+ MoCLIP + Phys. (arms only)",    True,  False, True,  None,   None,  None ),  # not yet run
    ("+ MoCLIP + S-VIB + Phys. (arm)", True,  True,  True,  0.4180, None,  None ),
    ("+ MoCLIP + S-VIB + Phys. (full)",True,  True,  True,  0.4137, 0.746, 12.46),  # ★ best
]

# ─── Best-value masks for bold formatting ───────────────────────────────────
fgd_vals = [r[4] for r in rows if r[4] is not None]
bc_vals  = [r[5] for r in rows if r[5] is not None]
div_vals = [r[6] for r in rows if r[6] is not None]
best_fgd = min(fgd_vals)
best_bc  = max(bc_vals)
best_div = min(div_vals)   # L1 Vel ↓ = lower is better

# ── Layout ───────────────────────────────────────────────────────────────────
N   = len(rows)
fig_w, fig_h = 13, 3.8
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, fig_w)
ax.set_ylim(0, fig_h)
ax.axis('off')
fig.patch.set_facecolor('#FAFCFE')

# ── Colour palette ───────────────────────────────────────────────────────────
COL_HEAD  = '#1C3548'   # dark navy  – header
COL_ALT   = '#EEF3F8'   # light blue – alternating row BG
COL_BEST  = '#D5F0E3'   # pale green – best row
COL_PEND  = '#FDF6E3'   # pale amber – pending (no results yet)
COL_CHECK = '#1A8A4A'   # green tick
COL_CROSS = '#C0392B'   # red cross
COL_BOLD  = '#1A5276'   # bold metric value
GRAY      = '#7F8C8D'
FG        = '#1C2833'

# Column definitions: (label, x_center, width_fraction, align)
cols = [
    ("Model Variant",       2.2,  None,  'left'  ),
    ("MoCLIP",              5.3,  None,  'center'),
    ("S-VIB  ψ",            6.45, None,  'center'),
    ("Physics",             7.55, None,  'center'),
    ("FGD ↓",               9.0,  None,  'center'),
    ("BC ↑",               10.6,  None,  'center'),
    ("L1 Vel ↓",           12.1,  None,  'center'),
]

row_h    = (fig_h - 1.3) / N         # height per data row
head_y   = fig_h - 0.55              # header y (top of header band)
head_h   = 0.65

# ── Header band ──────────────────────────────────────────────────────────────
hdr = FancyBboxPatch((0.22, head_y - head_h), fig_w - 0.44, head_h,
                     boxstyle='round,pad=0.06',
                     facecolor=COL_HEAD, edgecolor='none', zorder=2)
ax.add_patch(hdr)
for label, xc, _, align in cols:
    ha = 'left' if align == 'left' else 'center'
    ox = -1.0 if align == 'left' else 0
    ax.text(xc + ox, head_y - head_h/2, label,
            ha=ha, va='center', fontsize=9.5, fontweight='bold',
            color='white', zorder=3)

# Sub-header: "Component flags" spanning the three check columns
ax.text(6.45, head_y - 0.10, 'Component Flags',
        ha='center', va='top', fontsize=7.5, color='#B0C4D8', zorder=3,
        fontstyle='italic')
ax.text(9.9, head_y - 0.10, 'Metrics',
        ha='center', va='top', fontsize=7.5, color='#B0C4D8', zorder=3,
        fontstyle='italic')

# Thin separator lines between metric groups (in header)
for xsep in [8.2, 11.4]:
    ax.plot([xsep, xsep], [head_y - head_h + 0.06, head_y - 0.03],
            color='#4A6278', lw=0.8, zorder=3)

# ── Title ────────────────────────────────────────────────────────────────────
ax.text(fig_w / 2, fig_h - 0.18,
        'Table 1 — Ablation Study: Effect of MoCLIP, S-VIB Gate, and Physics Smoother',
        ha='center', va='top', fontsize=11, fontweight='bold', color=FG)

# ─── Helper formatters ───────────────────────────────────────────────────────
def fmt_metric(val, best, is_best_fn):
    if val is None:
        return '—', False
    s = f'{val:.4f}' if val < 10 else f'{val:.2f}'
    return s, is_best_fn(val, best)

def check(flag):
    return ('✓', COL_CHECK) if flag else ('✗', COL_CROSS)


# ── Data rows ────────────────────────────────────────────────────────────────
for i, (label, moclip, svib, phys, fgd, bc, div) in enumerate(rows):
    y_bot = head_y - head_h - (i + 1) * row_h
    y_mid = y_bot + row_h / 2

    pending = fgd is None  # no results yet
    is_best_row = (fgd == best_fgd and bc is not None)

    # Row background
    if is_best_row:
        bg_col = COL_BEST
    elif pending:
        bg_col = COL_PEND
    elif i % 2 == 0:
        bg_col = COL_ALT
    else:
        bg_col = 'white'

    bg = FancyBboxPatch((0.22, y_bot + 0.03), fig_w - 0.44, row_h - 0.04,
                        boxstyle='round,pad=0.04',
                        facecolor=bg_col, edgecolor='none', zorder=1)
    ax.add_patch(bg)

    # ── Model label ──────────────────────────────────────────────────────────
    fw = 'bold' if is_best_row else 'normal'
    ax.text(1.2, y_mid, label,
            ha='left', va='center', fontsize=9, fontweight=fw,
            color=FG, zorder=3)

    # ── Check / cross marks ──────────────────────────────────────────────────
    for flag, xc in [(moclip, 5.3), (svib, 6.45), (phys, 7.55)]:
        sym, col = check(flag)
        ax.text(xc, y_mid, sym, ha='center', va='center',
                fontsize=12, color=col, fontweight='bold', zorder=3)

    # ── FGD ──────────────────────────────────────────────────────────────────
    fgd_s, fgd_best = fmt_metric(fgd, best_fgd, lambda v, b: v == b)
    ax.text(9.0, y_mid, fgd_s,
            ha='center', va='center', fontsize=9,
            fontweight='bold' if fgd_best else 'normal',
            color=COL_BOLD if fgd_best else (GRAY if fgd is None else FG),
            zorder=3)
    if fgd_best:
        ax.text(9.62, y_mid, '★', ha='left', va='center',
                fontsize=8, color=COL_CHECK, zorder=3)

    # ── BC ───────────────────────────────────────────────────────────────────
    bc_s, bc_best = fmt_metric(bc, best_bc, lambda v, b: v == b)
    ax.text(10.6, y_mid, bc_s,
            ha='center', va='center', fontsize=9,
            fontweight='bold' if bc_best else 'normal',
            color=COL_BOLD if bc_best else (GRAY if bc is None else FG),
            zorder=3)

    # ── L1 Vel ───────────────────────────────────────────────────────────────
    div_s, div_best = fmt_metric(div, best_div, lambda v, b: v == b)
    ax.text(12.1, y_mid, div_s,
            ha='center', va='center', fontsize=9,
            fontweight='bold' if div_best else 'normal',
            color=COL_BOLD if div_best else (GRAY if div is None else FG),
            zorder=3)

    # ── Pending label ─────────────────────────────────────────────────────────
    if pending:
        ax.text(11.0, y_mid, 'pending',
                ha='center', va='center', fontsize=7.5,
                color='#B8860B', fontstyle='italic', zorder=3)

    # ── Separator line ────────────────────────────────────────────────────────
    ax.plot([0.3, fig_w - 0.3], [y_bot + 0.03, y_bot + 0.03],
            color='#D5DCE3', lw=0.6, zorder=2)

# ── Vertical separator between flags and metrics ─────────────────────────────
y_top = head_y - head_h
y_bot_last = head_y - head_h - N * row_h
for xsep in [8.2, 11.4]:
    ax.plot([xsep, xsep], [y_top, y_bot_last + 0.06],
            color='#B0BEC5', lw=0.8, zorder=2, linestyle='--')

# ── Legend / footnote ────────────────────────────────────────────────────────
ax.text(0.3, 0.10,
        '✓/✗ = component enabled/disabled    '
        '★ = best result    '
        '— = metric not yet evaluated    '
        'amber rows = training run pending',
        ha='left', va='bottom', fontsize=7.5, color=GRAY, fontstyle='italic')

plt.tight_layout(pad=0.2)
out_base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'duogesture_ablation_table')
plt.savefig(out_base + '.png', dpi=240, bbox_inches='tight',
            facecolor='#FAFCFE')
plt.savefig(out_base + '.pdf', bbox_inches='tight', facecolor='#FAFCFE')
print(f'Saved:\n  {out_base}.png\n  {out_base}.pdf')
