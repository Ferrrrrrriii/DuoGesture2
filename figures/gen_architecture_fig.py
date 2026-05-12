#!/usr/bin/env python3
"""
DuoGesture Architecture Figure
Highlights: MoCLIP Conditioning | S-VIB Gate | Physics Smoother
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

# ── Palette ──────────────────────────────────────────────────────────────────
CI = '#E08B00'   # amber   – inputs
CF = '#7D8E99'   # slate   – frozen base
CM = '#1E8449'   # green   – MoCLIP (novel)
CV = '#1A5276'   # navy    – S-VIB   (novel)
CP = '#A04000'   # rust    – Physics (novel)
CR = '#6C3483'   # purple  – RVQ-VAE
CO = '#0E6655'   # teal    – output
BG = '#F4F6F8'
FG = '#1C2833'

# ── Figure ───────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(24, 11))
ax.set_xlim(0, 24)
ax.set_ylim(0, 11)
ax.axis('off')
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)


# ── Helpers ──────────────────────────────────────────────────────────────────
def box(x, y, w, h, color, label, sub=None, fs=8.5, alpha=0.90,
        lw=1.8, zorder=3):
    """Rounded filled box."""
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.10',
                       facecolor=color, edgecolor='white',
                       linewidth=lw, alpha=alpha, zorder=zorder)
    ax.add_patch(p)
    ty = y + h * (0.65 if sub else 0.5)
    ax.text(x + w / 2, ty, label,
            ha='center', va='center', fontsize=fs,
            fontweight='bold', color='white', zorder=zorder + 1,
            multialignment='center')
    if sub:
        ax.text(x + w / 2, y + h * 0.27, sub,
                ha='center', va='center', fontsize=fs - 1.5,
                color='white', alpha=0.92, zorder=zorder + 1,
                multialignment='center')


def zone(x, y, w, h, color, title, above=True, fs=9.5, alpha=0.11, lw=2.2):
    """Coloured highlight zone."""
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle='round,pad=0.22',
                       facecolor=color, edgecolor=color,
                       linewidth=lw, alpha=alpha, zorder=1)
    ax.add_patch(p)
    ty = y + h + 0.28 if above else y - 0.28
    va = 'bottom' if above else 'top'
    ax.text(x + w / 2, ty, title,
            ha='center', va=va, fontsize=fs,
            fontweight='bold', color=color, zorder=5)


def arr(x1, y1, x2, y2, color=FG, lw=1.6,
        cs='arc3,rad=0.0', lbl=None, style='->',
        ls='solid'):
    kw = dict(arrowstyle=style, color=color, lw=lw,
              connectionstyle=cs, linestyle=ls)
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=kw, zorder=2)
    if lbl:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my + 0.18, lbl, ha='center', va='bottom',
                fontsize=6.5, color=color, zorder=6,
                bbox=dict(facecolor=BG, edgecolor='none',
                          alpha=0.7, pad=1))


def txt(x, y, s, fs=9, color=FG, ha='center', bold=False, italic=False):
    style = 'italic' if italic else 'normal'
    fw = 'bold' if bold else 'normal'
    ax.text(x, y, s, ha=ha, va='center', fontsize=fs,
            color=color, fontstyle=style, fontweight=fw)


# ═══════════════════════════════════════════════════════════════════════════
#  TITLE
# ═══════════════════════════════════════════════════════════════════════════
txt(12, 10.65,
    'DuoGesture: Two-Stage Holistic Co-Speech Motion Generation',
    fs=15, bold=True)
txt(12, 10.22,
    '★ Novel:   MoCLIP Semantic Conditioning   ·   S-VIB Stochastic Gate ψ   ·'
    '   Physics Smoother',
    fs=10, italic=True, color='#555')

# ═══════════════════════════════════════════════════════════════════════════
#  SECTION A – INPUTS  (x: 0.2 – 2.2)
# ═══════════════════════════════════════════════════════════════════════════
zone(0.2, 1.2, 2.0, 7.5, CI, 'Inputs', above=False, fs=9, alpha=0.08)

box(0.4, 8.0, 1.6, 0.68, CI, 'Speech Audio', '.wav')
box(0.4, 6.9, 1.6, 0.68, CF, 'HuBERT', 'frozen encoder', fs=8)
arr(1.2, 8.0, 1.2, 7.58)

box(0.4, 5.6, 1.6, 0.68, CI, 'Speaker ID', '+ Embedding')
box(0.4, 4.3, 1.6, 0.68, CI, 'Speech', 'Transcripts')
box(0.4, 3.0, 1.6, 0.68, CI, 'Seed Pose', '4 frames')

# ═══════════════════════════════════════════════════════════════════════════
#  SECTION B – STAGE 1: BASE MOTION (frozen)  (x: 2.5 – 6.3)
# ═══════════════════════════════════════════════════════════════════════════
zone(2.5, 6.2, 3.6, 2.7, CF, 'Stage 1: Base Motion  ❄  (frozen)',
     fs=9, alpha=0.10)

box(2.7, 8.0, 1.3, 0.68, CF, 'Self-Attn', 'audio', fs=8)
box(4.2, 8.0, 1.3, 0.68, CF, 'Cross-Attn', 'speaker+seed', fs=8)
box(3.3, 6.9, 1.5, 0.68, CF, '× 8 Layers', 'motion encoder', fs=8)
box(3.3, 6.0, 1.5, 0.68, CF, 'q_b', 'base latent [B,T/4,256]', fs=7.5)

# wiring inside Stage 1
arr(2.0, 7.34, 2.7 + 0.65, 8.0, color=CF)                  # HuBERT → SA
arr(2.0, 5.94, 4.55, 8.0, color=CF, cs='arc3,rad=-0.12')    # Speaker → CA
arr(2.0, 3.34, 4.55, 8.0, color=CF, cs='arc3,rad=-0.2')     # Seed pose → CA
arr(4.0, 8.34, 4.2, 8.34, color=CF)                         # SA → CA
arr(4.55, 8.0, 4.0, 7.58, color=CF)                         # CA → ×8
arr(4.05, 6.9, 4.05, 6.68, color=CF)                        # ×8 → q_b

# q_b → right hand side
arr(4.8, 6.34, 6.8, 6.34, color=CF, lw=2.0, lbl='q_b')

# ═══════════════════════════════════════════════════════════════════════════
#  SECTION C – MoCLIP CONDITIONING  (x: 6.5 – 10.5)
# ═══════════════════════════════════════════════════════════════════════════
zone(6.5, 7.2, 3.8, 1.9, CM, 'MoCLIP  Semantic Conditioning  ★',
     fs=9.5, alpha=0.12)

box(6.7, 7.4, 1.5, 0.68, CM, 'Text Encoder', 'TMR / MoCLIP', fs=8)
box(8.4, 7.4, 1.7, 0.68, CM, 'Semantic Embeds', 'Z^s  [B, T, 256]', fs=8)
arr(2.0, 4.64, 6.7 + 0.75, 7.4, color=CM,
    cs='arc3,rad=-0.2')    # transcripts → TE
ax.text(3.5, 5.9, 'transcripts', ha='center', va='center',
        fontsize=6.5, color=CM, rotation=28,
        bbox=dict(facecolor=BG, edgecolor='none', alpha=0.75, pad=1))
arr(8.2, 7.74, 8.4, 7.74, color=CM)           # TE → embeds

# Dual-stage conditioning arrows (dashed)
arr(9.25, 7.4, 11.5, 7.9, color=CM, lw=1.4,
    cs='arc3,rad=-0.08', ls='dashed', lbl='early cond  ×0.20')
arr(9.25, 7.4, 11.5, 6.6, color=CM, lw=1.4,
    cs='arc3,rad=0.12', ls='dashed', lbl='mid cond  ×0.15')

# ═══════════════════════════════════════════════════════════════════════════
#  SECTION D – S-VIB GATE  (x: 6.5 – 10.5)
# ═══════════════════════════════════════════════════════════════════════════
zone(6.5, 1.4, 3.8, 5.5, CV, 'S-VIB  Stochastic Gate  ψ  ★',
     fs=9.5, alpha=0.12)

box(6.7, 6.0, 1.5, 0.68, CV, 'Semantic', 'Decoder (8L Xf)', fs=8)
box(8.4, 6.0, 1.7, 1.0, CV, 'SemanticVIB',
    'Stream A: sem 256d\nStream B: timing 64d\nbottleneck z = 16d', fs=7)
box(6.7, 4.7, 1.3, 0.68, CV, 'ψ  gate', 'softmax[0,1]', fs=8)
box(8.4, 4.7, 1.7, 0.68, CV, 'KL Loss',
    'β · KL  (free-bits 0.5/dim)', fs=7.5)
box(6.7, 3.5, 1.3, 0.68, CV, 'Gated Sem.', 'q_s  × ψ', fs=8)
box(8.4, 3.5, 1.7, 0.68, CV, 'Latent Bridge',
    'α·q_b + (1−α)·q_s', fs=7.5)
box(7.0, 2.3, 2.7, 0.68, CV, 'q_m  =  MLP( ψ·q_s + (1−ψ)·q_b )',
    fs=8, alpha=0.94)

# HuBERT → VIB timing stream
arr(2.0, 7.34, 7.4, 6.68, color=CV,
    cs='arc3,rad=-0.25', lbl='Z^a timing')
# MoCLIP embeds → Semantic Decoder
arr(9.25, 7.4, 7.4, 6.68, color=CM, cs='arc3,rad=0.15')
# Sem Decoder → VIB
arr(7.4 + 0.75, 6.0, 8.4 + 0.85, 7.0, color=CV)
# VIB → ψ
arr(8.4 + 0.85, 6.0, 7.35, 5.38, color=CV)
# VIB → KL
arr(8.4 + 0.85, 6.0, 8.4 + 0.85, 5.38, color=CV)
# ψ → gated semantic
arr(7.35, 4.7, 7.35, 4.18, color=CV)
# ψ → fused latent
arr(7.35, 4.7, 7.85, 2.98, color=CV, cs='arc3,rad=0.2')
# gated sem → q_m
arr(7.35, 3.5, 7.85, 2.98, color=CV)
# latent bridge → q_m
arr(8.4 + 0.85, 3.5, 8.55, 2.98, color=CV)
# q_b → latent bridge (dashed – from frozen stage)
arr(6.5, 6.34, 8.4 + 0.85, 4.04, color=CF,
    cs='arc3,rad=0.25', ls='dashed')

# q_m exits right
arr(9.7, 2.64, 11.45, 2.64, color=CV, lw=2.0, lbl='q_m')

# ψ long arrow to Physics (drawn later – placeholder)
# (drawn at end to avoid z-order issues)

# ═══════════════════════════════════════════════════════════════════════════
#  SECTION E – SPARSE SEMANTIC MODEL + RVQ  (x: 11.4 – 15.8)
# ═══════════════════════════════════════════════════════════════════════════
zone(11.4, 1.4, 4.2, 6.9, '#2C3E50',
     'Stage 2: Sparse Semantic  +  RVQ-VAE Decoder',
     fs=9, alpha=0.09)

box(11.6, 7.7, 1.6, 0.68, '#374151', 'Word-Hints', 'Decoder  8L', fs=8)
box(11.6, 6.6, 1.6, 0.68, '#374151', 'Upper Body', 'Decoder', fs=8)
box(13.4, 6.6, 1.6, 0.68, '#374151', 'Hands', 'Decoder', fs=8)
box(11.6, 5.5, 1.6, 0.68, '#374151', 'Lower Body', 'Decoder', fs=8)

box(13.4, 5.2, 1.8, 1.25, CR, 'RVQ-VAE', '6-level codebook\ndecoder  (frozen)', fs=8)
box(12.0, 3.8, 2.4, 0.68, CR, 'Motion Tokens', 'quantised [B, T/4, 256]', fs=8)
box(12.0, 2.8, 2.4, 0.68, CR, 'SMPLX Motion', 'rot6d  [B, T, 337]', fs=8)

# wiring inside sparse decoder
arr(11.6 + 0.8, 7.7, 12.4, 7.28, color='#374151')  # word-hints → upper
arr(13.2, 6.94, 13.4, 6.94, color='#374151')        # upper ↔ hands (cross-attn)
arr(13.4, 6.6, 12.2, 6.6, color='#374151',
    cs='arc3,rad=0.3')                               # hands → upper back
arr(12.4, 6.6, 12.4, 6.18, color='#374151')         # upper → lower
arr(13.2, 5.84, 13.4, 5.84, color='#374151')        # lower → RVQ
arr(14.3, 6.6, 14.3, 6.45, color='#374151')         # hands → RVQ
arr(13.4 + 0.9, 5.2, 13.2, 4.14, color=CR)          # RVQ → tokens
arr(12.4 + 0.8, 3.8, 12.4 + 0.8, 3.48, color=CR)   # tokens → SMPLX

# q_m into lower decoder
arr(11.45, 2.64, 12.4, 5.5, color=CV,
    cs='arc3,rad=-0.15')

# early/mid cond hits word-hints and upper
arr(11.5, 7.9, 11.6 + 0.8, 8.04, color=CM, lw=1.2,
    cs='arc3,rad=0.0')
arr(11.5, 6.6, 11.6 + 0.4, 6.94, color=CM, lw=1.2)

# SMPLX exits right
arr(14.4, 3.14, 16.5, 3.14, color=CR, lw=2.0)

# ═══════════════════════════════════════════════════════════════════════════
#  SECTION F – PHYSICS SMOOTHER  (x: 16.5 – 21.5)
# ═══════════════════════════════════════════════════════════════════════════
zone(16.4, 1.4, 5.8, 6.5, CP, 'Physics Smoother  ★',
     fs=9.5, alpha=0.11)

box(16.6, 6.8, 2.0, 0.68, CP, 'EMA Filter', 'τ(ψ) per joint group', fs=8)
box(19.0, 6.8, 2.2, 0.68, CP, 'Gate-Modulated τ',
    'τ = τ_base·(1−ψ) + τ_floor·ψ', fs=7.5)

box(16.6, 5.6, 2.0, 0.68, CP, 'De Leva Prior',
    'joint mass conditioning', fs=7.5)
box(19.0, 5.6, 2.2, 0.68, CP, 'Jerk Loss',
    'λ·‖Δ²rot‖  (training only)', fs=7.5)

box(16.6, 4.3, 2.0, 0.68, CP, 'Smoothed', 'rot6d  [B, T, 337]', fs=8)

arr(17.6, 6.8, 19.0, 7.14, color=CP)    # EMA → gate-mod τ
arr(17.6, 6.8, 17.6, 6.28, color=CP)    # EMA → De Leva
arr(19.0 + 1.1, 6.8, 19.0 + 1.1, 6.28, color=CP)  # gate-mod → jerk
arr(17.6, 5.6, 17.6, 4.98, color=CP)    # De Leva → smoothed

# SMPLX → Physics
arr(16.5, 3.14, 17.6, 4.3, color=CR, cs='arc3,rad=-0.1')

# ψ long arrow to Physics smoother
arr(7.35, 4.7, 17.6, 6.8, color=CV, lw=1.6,
    cs='arc3,rad=-0.35', lbl='ψ  (gate signal)')

# ═══════════════════════════════════════════════════════════════════════════
#  OUTPUT
# ═══════════════════════════════════════════════════════════════════════════
box(19.1, 2.4, 4.5, 1.5, CO,
    'Generated Gesture  Ĝ',
    'SMPLX rot6d · 337-dim\nFace · Upper · Hands · Lower · Transl',
    fs=9.5, alpha=0.92)

arr(17.6, 4.3, 21.35, 3.9, color=CP, lw=2.0,
    cs='arc3,rad=0.1')

# Silhouette placeholder (simple stick figures)
for sx, sy, scale, a in [(22.7, 2.9, 1.0, 0.5), (23.5, 3.1, 0.75, 0.32)]:
    # head
    head = plt.Circle((sx, sy + 0.55 * scale), 0.18 * scale,
                       color=CO, alpha=a, zorder=4)
    ax.add_patch(head)
    # body
    ax.plot([sx, sx], [sy - 0.35 * scale, sy + 0.37 * scale],
            color=CO, lw=2.5 * scale, alpha=a, zorder=4)
    # arms
    ax.plot([sx - 0.32 * scale, sx + 0.32 * scale],
            [sy + 0.1 * scale, sy + 0.1 * scale],
            color=CO, lw=2 * scale, alpha=a, zorder=4)
    # legs
    ax.plot([sx, sx - 0.22 * scale], [sy - 0.35 * scale, sy - 0.75 * scale],
            color=CO, lw=2 * scale, alpha=a, zorder=4)
    ax.plot([sx, sx + 0.22 * scale], [sy - 0.35 * scale, sy - 0.75 * scale],
            color=CO, lw=2 * scale, alpha=a, zorder=4)

# ── Z labels alongside main arrows ──────────────────────────────────────────
ax.text(5.65, 6.55, r'$\mathbf{q_b}$', ha='center', fontsize=9,
        color=CF, fontweight='bold')
ax.text(10.55, 2.82, r'$\mathbf{q_m}$', ha='center', fontsize=9,
        color=CV, fontweight='bold')

# ═══════════════════════════════════════════════════════════════════════════
#  FROZEN BADGE (small ❄ stamp on Stage 1 zone)
# ═══════════════════════════════════════════════════════════════════════════
ax.text(2.72, 8.82, '❄ Frozen', ha='left', fontsize=7.5,
        color=CF, fontstyle='italic')

# ═══════════════════════════════════════════════════════════════════════════
#  LEGEND
# ═══════════════════════════════════════════════════════════════════════════
patches = [
    mpatches.Patch(facecolor=CI,  label='Input signals',             alpha=0.88),
    mpatches.Patch(facecolor=CF,  label='Stage 1: Base  ❄',         alpha=0.88),
    mpatches.Patch(facecolor=CM,  label='MoCLIP Conditioning  ★',   alpha=0.88),
    mpatches.Patch(facecolor=CV,  label='S-VIB Gate  ★',            alpha=0.88),
    mpatches.Patch(facecolor=CR,  label='RVQ-VAE Decoder',           alpha=0.88),
    mpatches.Patch(facecolor=CP,  label='Physics Smoother  ★',      alpha=0.88),
    mpatches.Patch(facecolor=CO,  label='Output  Ĝ',                 alpha=0.88),
]
ax.legend(handles=patches, loc='lower center', ncol=7,
          fontsize=9, framealpha=0.93,
          bbox_to_anchor=(0.5, -0.04), edgecolor='#ccc')

plt.tight_layout(pad=0.3)

out_base = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'duogesture_architecture')
plt.savefig(out_base + '.png', dpi=240, bbox_inches='tight', facecolor=BG)
plt.savefig(out_base + '.pdf', bbox_inches='tight', facecolor=BG)
print(f'Saved:\n  {out_base}.png\n  {out_base}.pdf')
