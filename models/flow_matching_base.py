"""
GestureLSM-style Flow Matching Base Motion Generator for DuoGesture
================================================================

Replaces DuoGesture's Coarse2Fine Cross-Attention base motion module with:
  1. Spatial-Temporal Transformer (from GestureLSM, ICCV 2025)
  2. Flow Matching generative process (instead of regression/diffusion)
  3. Beta Distribution timestep sampling (α=2, β=1.2)

Architecture overview:
    Condition Encoding (kept from DuoGesture):
        HuBERT + Audio beats + Seed pose → rhythmic condition  c
    ↓
    Spatial-Temporal Transformer (GestureLSM-style):
        • Spatial Attention  — models Face/Hands/Upper/Lower body interactions
        • Temporal Attention  — temporal dynamics + cross-attn with condition c
    ↓
    Flow Matching — learns OT vector field  x₀ → x₁  with  t ∼ Beta(2, 1.2)
        v_t = f_θ(x_t, t, c)
        x_t = (1 − (1−σ_min)·t) · x₀  +  t · x₁
    ↓
    Output: per-body-part latent codes  q_b ∈ ℝ^{T'×256}  (T'=16 for T=64)

Compatible with DuoGesture Adaptive Fusion:
    q_m = MLP(ψ · q_s + (1−ψ) · q_b)

References:
    [1] DuoGesture  (ICCV 2025) — Semantic-aware base/sparse gesture generation
    [2] GestureLSM (ICCV 2025) — Flow matching for co-speech gesture synthesis
"""

import sys
import pathlib
THIS_FILE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- reuse DuoGesture building blocks ---
try:
    from .motion_encoder import VQEncoderV6
    from .duogesture import (
        MLP,
        PeriodicPositionalEncoding,
        predict_residual_zq,
        RhythmicIdentificationLoss,
    )
except ImportError:
    from models.motion_encoder import VQEncoderV6
    from models.duogesture import (
        MLP,
        PeriodicPositionalEncoding,
        predict_residual_zq,
        RhythmicIdentificationLoss,
    )


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    FLOW  MATCHING  PRIMITIVES                      ║
# ╚══════════════════════════════════════════════════════════════════════╝

class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal embedding for the continuous flow timestep  t ∈ [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [B]  (scalar timestep per sample)
        Returns:
            emb: [B, dim]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 2:
            t = t.squeeze(-1)
        half = self.dim // 2
        freq = torch.exp(
            -math.log(10_000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
        )
        args = t.unsqueeze(-1) * freq.unsqueeze(0)          # [B, half]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # [B, dim]
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TimestepMLP(nn.Module):
    """
    Timestep conditioning:
        sinusoidal → MLP → [scale, shift]   for Adaptive LayerNorm.
    """

    def __init__(self, dim: int, hidden_mult: int = 4):
        super().__init__()
        self.sinusoidal = SinusoidalTimestepEmbedding(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * hidden_mult),
            nn.SiLU(),
            nn.Linear(dim * hidden_mult, dim * 2),     # → (scale, shift)
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Returns [B, dim*2] — first-half=scale, second-half=shift."""
        return self.mlp(self.sinusoidal(t))


class AdaptiveLayerNorm(nn.Module):
    """Adaptive LN conditioned on timestep:  y = (1+s)·LN(x) + b."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:        [*, D]
            time_emb: [B, D*2]   (from TimestepMLP)
        """
        scale, shift = time_emb.chunk(2, dim=-1)         # each [B, D]
        # broadcast to match x shape
        while scale.dim() < x.dim():
            scale = scale.unsqueeze(1)
            shift = shift.unsqueeze(1)
        return self.norm(x) * (1.0 + scale) + shift


# ╔══════════════════════════════════════════════════════════════════════╗
# ║              SPATIAL-TEMPORAL  TRANSFORMER  (GestureLSM)           ║
# ╚══════════════════════════════════════════════════════════════════════╝

class SpatialAttentionBlock(nn.Module):
    """
    Models interactions between body parts at every time-step.
    Input  [B, T, P, D]  →  reshape to  [B·T, P, D]  →  self-attention  →  back.

    P = 4  (face, upper, hands, lower)
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, P, D = x.shape
        x = x.reshape(B * T, P, D)
        # pre-norm self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        # pre-norm FFN
        h = self.norm2(x)
        x = x + self.ff(h)
        return x.reshape(B, T, P, D)


class TemporalCrossAttentionBlock(nn.Module):
    """
    Models temporal dynamics within each body part and fuses rhythmic condition.
    Input  [B, T, P, D]  →  reshape to  [B·P, T, D]
        1. temporal self-attention
        2. cross-attention (Q=motion, K/V=condition)
        3. feed-forward

    cond shape:  [B, T_c, D]
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # temporal self-attn
        self.norm_self  = nn.LayerNorm(dim)
        self.self_attn  = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        # temporal cross-attn with condition
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        # ffn
        self.norm_ff    = nn.LayerNorm(dim)
        self.ff         = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    [B, T, P, D]
            cond: [B, T_c, D]
        """
        B, T, P, D = x.shape
        T_c = cond.shape[1]

        x = x.permute(0, 2, 1, 3).reshape(B * P, T, D)          # [B·P, T,  D]
        cond_exp = cond.unsqueeze(1).expand(-1, P, -1, -1)       # [B, P, T_c, D]
        cond_exp = cond_exp.reshape(B * P, T_c, D)               # [B·P, T_c, D]

        # 1) temporal self-attention
        h = self.norm_self(x)
        h, _ = self.self_attn(h, h, h)
        x = x + h
        # 2) cross-attention with rhythmic condition
        h = self.norm_cross(x)
        h, _ = self.cross_attn(h, cond_exp, cond_exp)
        x = x + h
        # 3) feed-forward
        h = self.norm_ff(x)
        x = x + self.ff(h)

        return x.reshape(B, P, T, D).permute(0, 2, 1, 3)        # [B, T, P, D]


class SpatialTemporalBlock(nn.Module):
    """
    One block:  Spatial Attention → Adaptive-LN → Temporal Cross-Attention.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.adaln    = AdaptiveLayerNorm(dim)
        self.spatial  = SpatialAttentionBlock(dim, num_heads, dropout)
        self.temporal = TemporalCrossAttentionBlock(dim, num_heads, dropout)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:        [B, T, P, D]   noisy motion latent
            cond:     [B, T_c, D]    rhythmic condition
            time_emb: [B, D*2]       timestep conditioning
        """
        B, T, P, D = x.shape
        # adaptive layer-norm conditioned on flow timestep
        x_flat = x.reshape(B, T * P, D)
        x_flat = self.adaln(x_flat, time_emb)
        x = x_flat.reshape(B, T, P, D)
        # spatial → temporal
        x = self.spatial(x)
        x = self.temporal(x, cond)
        return x


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                VELOCITY  FIELD  PREDICTOR   f_θ                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

class FlowMatchingVelocityNet(nn.Module):
    """
    Predicts the velocity field  v_t = f_θ(x_t, t, c)  for the
    Optimal-Transport Conditional Flow Matching (OT-CFM) objective.

    x_t : noisy latent for *all* body parts   [B, T', 4, D]
    t   : flow timestep ∈ [0, 1]              [B]
    c   : rhythmic condition                   [B, T_c, D]
    """

    def __init__(
        self,
        latent_dim: int = 256,
        num_parts: int = 4,       # face, upper, hands, lower
        num_heads: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_parts  = num_parts

        # learnable per-part type embeddings
        self.part_embed = nn.Embedding(num_parts, latent_dim)

        # timestep MLP  (output: scale+shift for AdaptiveLN)
        self.time_mlp = TimestepMLP(latent_dim)

        # input projection
        self.input_proj = nn.Linear(latent_dim, latent_dim)

        # stack of Spatial-Temporal transformer blocks
        self.blocks = nn.ModuleList([
            SpatialTemporalBlock(latent_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # output head
        self.out_norm = nn.LayerNorm(latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t:  [B, T', P, D]
            t:    [B]                  timestep  ∈ [0, 1]
            cond: [B, T_c, D]         rhythmic condition
        Returns:
            v_t:  [B, T', P, D]       predicted velocity
        """
        B, T, P, D = x_t.shape

        # --- add part-type embeddings ---
        part_ids = torch.arange(P, device=x_t.device)
        part_emb = self.part_embed(part_ids)                    # [P, D]
        x = self.input_proj(x_t) + part_emb.unsqueeze(0).unsqueeze(0)  # broadcast

        # --- timestep conditioning ---
        time_emb = self.time_mlp(t)                              # [B, D*2]

        # --- spatial-temporal blocks ---
        for block in self.blocks:
            x = block(x, cond, time_emb)

        # --- output ---
        x = self.out_norm(x.reshape(B, T * P, D)).reshape(B, T, P, D)
        v_t = self.out_proj(x)
        return v_t


# ╔══════════════════════════════════════════════════════════════════════╗
# ║            FLOW  MATCHING  UTILITIES  (loss / sampling)            ║
# ╚══════════════════════════════════════════════════════════════════════╝

def sample_beta_timesteps(
    batch_size: int,
    alpha: float = 2.0,
    beta: float = 1.2,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Sample timesteps from Beta(α, β) — biased towards harder mid-trajectory
    regions as proposed in GestureLSM.
    Returns: [B]  values in (0, 1).
    """
    dist = torch.distributions.Beta(alpha, beta)
    t = dist.sample((batch_size,))
    if device is not None:
        t = t.to(device)
    return t.clamp(1e-5, 1.0 - 1e-5)


def ot_conditional_flow(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    t: torch.Tensor,
    sigma_min: float = 1e-4,
):
    """
    Optimal-Transport Conditional Flow Matching interpolation and target.

    Interpolation path:
        x_t = (1 − (1−σ_min)·t) · x_0  +  t · x_1

    Target velocity:
        u_t = x_1 − (1 − σ_min) · x_0

    Args:
        x_0: [B, ...]  noise samples       ~ N(0, I)
        x_1: [B, ...]  data samples        (ground-truth latent)
        t:   [B]        timestep ∈ (0, 1)

    Returns:
        x_t:      interpolated sample
        target_v: target velocity field
    """
    # reshape t to broadcast with spatial dims
    t_view = t
    for _ in range(x_0.dim() - 1):
        t_view = t_view.unsqueeze(-1)

    x_t = (1.0 - (1.0 - sigma_min) * t_view) * x_0 + t_view * x_1
    target_v = x_1 - (1.0 - sigma_min) * x_0
    return x_t, target_v


def flow_matching_loss(
    v_pred: torch.Tensor,
    target_v: torch.Tensor,
) -> torch.Tensor:
    """
    Flow Matching objective:
        L_FM = E_{t, x₁, x₀}  || v_t  −  (x₁ − (1 − σ_min)·x₀) ||²

    Args:
        v_pred:   [B, T', P, D]   predicted velocity from the network
        target_v: [B, T', P, D]   ground-truth velocity from OT path

    Returns:
        scalar loss (mean over all dims)
    """
    return F.mse_loss(v_pred, target_v)


@torch.no_grad()
def ode_euler_sample(
    velocity_net: nn.Module,
    cond: torch.Tensor,
    shape: tuple,
    num_steps: int = 50,
    sigma_min: float = 1e-4,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Generate motion latent by numerically integrating the learned ODE
    using Euler's method:   x_{k+1} = x_k + Δt · v_θ(x_k, t_k, c)

    Args:
        velocity_net: the FlowMatchingVelocityNet module
        cond:         [B, T_c, D]  rhythmic condition
        shape:        target shape  (B, T', P, D)
        num_steps:    integration steps (higher → better quality)
        sigma_min:    minimal noise level
        device:       target device

    Returns:
        x_1: [B, T', P, D]  generated latent
    """
    B = shape[0]
    if device is None:
        device = cond.device

    dt = 1.0 / num_steps
    x = torch.randn(shape, device=device)                       # x_0 ~ N(0, I)

    for k in range(num_steps):
        t_k = torch.full((B,), k * dt, device=device)
        v = velocity_net(x, t_k, cond)
        x = x + v * dt

    return x


# ╔══════════════════════════════════════════════════════════════════════╗
# ║          MAIN  MODULE  —  GestureLSMBaseMotion                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

class GestureLSMBaseMotion(nn.Module):
    """
    Drop-in replacement for ``duogesture_base`` using GestureLSM Flow Matching.

    Inputs  (identical to duogesture_base):
        in_audio : [B, T, 3]    — beat features (onset + amplitude)
        hubert   : [B, T, 1024] — HuBERT audio representations
        in_motion: [B, T, 337]  — motion (6D-rot + trans + contact)
        mask     : [B, T, 337]  — first ``pre_frames`` frames = 0 (seed)
        in_id    : [B, T, 1]    — speaker id
        in_word  : [B, T]       — (accepted but NOT used — text is
                                    reserved for the Sparse branch)

    Outputs  (same dict keys as duogesture_base):
        Training:
            fm_loss        — Flow Matching vector-field loss
            hubert_cons_loss / beat_cons_loss — contrastive losses
            rec_face / rec_upper / rec_lower / rec_hands
            cls_face / cls_upper / cls_lower / cls_hands

        Inference via ``forward_latent``:
            face_latent, upper_latent, lower_latent, hands_latent
            (each  [B, T'=16, 256])
    """

    # body-part order used throughout the module
    PART_FACE  = 0
    PART_UPPER = 1
    PART_HANDS = 2
    PART_LOWER = 3
    NUM_PARTS  = 4

    def __init__(self, args):
        super().__init__()
        self.args = args
        latent_dim = 256                # = args.motion_f = args.audio_f

        # ────────── Flow Matching hyper-parameters ──────────
        self.sigma_min           = float(getattr(args, "fm_sigma_min", 1e-4))
        self.beta_alpha          = float(getattr(args, "fm_beta_alpha", 2.0))
        self.beta_beta           = float(getattr(args, "fm_beta_beta", 1.2))
        self.num_inference_steps = int(getattr(args, "fm_num_inference_steps", 50))
        self.cfg_drop_prob       = float(getattr(args, "fm_cfg_drop_prob", 0.1))

        # ────────── Condition encoders (HuBERT + audio beats) ──────────
        def _hubert_enc():
            return nn.Sequential(
                nn.Conv1d(1024, latent_dim, 3, 1, 1, bias=False),
                nn.BatchNorm1d(latent_dim),
                nn.GELU(),
                nn.Conv1d(latent_dim, latent_dim, 3, 1, 1, bias=False),
            )

        self.hubert_encoder      = _hubert_enc()        # for face
        self.hubert_encoder_body = _hubert_enc()        # for body

        self.audio_pre_encoder_face = MLP(3, args.hidden_size, latent_dim)
        self.audio_pre_encoder_body = MLP(3, args.hidden_size, latent_dim)

        # gated attention fusion (audio ↔ HuBERT) — same as DuoGesture
        self.at_attn_face = nn.Linear(latent_dim * 2, latent_dim * 2)
        self.at_attn_body = nn.Linear(latent_dim * 2, latent_dim * 2)

        # ────────── Seed pose / masked motion encoder ──────────
        args_enc = copy.deepcopy(args)
        args_enc.vae_layer    = 3
        args_enc.vae_length   = args.motion_f        # 256
        args_enc.vae_test_dim = args.pose_dims + 3 + 4  # 337
        self.motion_encoder = VQEncoderV6(args_enc)   # [B,T,337]→[B,T,256] (no temporal down)

        self.bodyhints_face = MLP(args.motion_f, args.hidden_size, args.motion_f)
        self.bodyhints_body = MLP(args.motion_f, args.hidden_size, args.motion_f)

        self.mask_embeddings = nn.Parameter(torch.zeros(1, 1, args.pose_dims + 3 + 4))
        nn.init.normal_(self.mask_embeddings, 0, args.hidden_size ** -0.5)

        # ────────── Speaker embeddings (256-d to match latent) ──────────
        self.speaker_encoder_face = nn.Embedding(25, latent_dim)
        self.speaker_encoder_body = nn.Embedding(25, latent_dim)

        # ────────── Condition fusion & projection ──────────
        # Merge face-cond [T,256] + body-cond [T,256] → [T,256]
        self.cond_merge = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # ────────── CORE: Spatial-Temporal Velocity Network ──────────
        fm_layers = int(getattr(args, "fm_num_layers", 6))
        fm_heads  = int(getattr(args, "fm_num_heads", 8))
        fm_drop   = float(getattr(args, "fm_dropout", 0.1))

        self.velocity_net = FlowMatchingVelocityNet(
            latent_dim=latent_dim,
            num_parts=self.NUM_PARTS,
            num_heads=fm_heads,
            num_layers=fm_layers,
            dropout=fm_drop,
        )

        # ────────── Contrastive losses (rhythmic identification) ──────────
        self.hubert_face_cons_loss = RhythmicIdentificationLoss(temperature=0.1)
        self.beat_cons_loss        = RhythmicIdentificationLoss(temperature=0.1)

        # ────────── RVQ prediction heads (levels 1-5, kept from DuoGesture) ──────────
        self.predict_res_face  = predict_residual_zq(latent_dim, 8, 1024, 0.1)
        self.predict_res_upper = predict_residual_zq(latent_dim, 8, 1024, 0.1)
        self.predict_res_hands = predict_residual_zq(latent_dim, 8, 1024, 0.1)
        self.predict_res_lower = predict_residual_zq(latent_dim, 8, 1024, 0.1)

        # level-0 classifiers
        self.face_classifier  = MLP(latent_dim, args.hidden_size, latent_dim)
        self.upper_classifier = MLP(latent_dim, args.hidden_size, latent_dim)
        self.hands_classifier = MLP(latent_dim, args.hidden_size, latent_dim)
        self.lower_classifier = MLP(latent_dim, args.hidden_size, latent_dim)

        # ────────── Condition downsampling for contrastive losses ──────────
        # face_cond [T,256] → [T//4, 256]  to match latent temporal res
        self.cond_face_down = nn.Conv1d(latent_dim, latent_dim, 4, 4)
        self.cond_body_down = nn.Conv1d(latent_dim, latent_dim, 4, 4)

    # ------------------------------------------------------------------
    #  helpers
    # ------------------------------------------------------------------

    def _encode_conditions(
        self,
        in_audio: torch.Tensor,        # [B, T, 3]
        hubert: torch.Tensor,           # [B, T, 1024]
        in_motion: torch.Tensor,        # [B, T, 337]
        mask: torch.Tensor,             # [B, T, 337]
        in_id: torch.Tensor,            # [B, T, 1]
    ):
        """Encode all rhythmic conditions; return a dict of intermediate features."""
        bs, t, _ = hubert.shape

        # --- HuBERT → 256 ---
        in_word_face = self.hubert_encoder(hubert.permute(0, 2, 1)).permute(0, 2, 1)      # [B,T,256]
        in_word_body = self.hubert_encoder_body(hubert.permute(0, 2, 1)).permute(0, 2, 1)  # [B,T,256]

        # --- Audio beats → 256 ---
        in_audio_face = self.audio_pre_encoder_face(in_audio)   # [B,T,256]
        in_audio_body = self.audio_pre_encoder_body(in_audio)   # [B,T,256]

        # --- Gated attention fusion ---
        c = in_word_face.shape[-1]

        alpha_face = torch.cat([in_word_face, in_audio_face], dim=-1)       # [B,T,512]
        alpha_face = self.at_attn_face(alpha_face).reshape(bs, t, c, 2)
        alpha_face = alpha_face.softmax(dim=-1)
        fusion_face = (in_word_face * alpha_face[..., 1]
                       + in_audio_face * alpha_face[..., 0])                # [B,T,256]

        alpha_body = torch.cat([in_word_body, in_audio_body], dim=-1)
        alpha_body = self.at_attn_body(alpha_body).reshape(bs, t, c, 2)
        alpha_body = alpha_body.softmax(dim=-1)
        fusion_body = (in_word_body * alpha_body[..., 1]
                       + in_audio_body * alpha_body[..., 0])                # [B,T,256]

        # --- Seed-pose / masked motion encoding ---
        masked_emb = self.mask_embeddings.expand_as(in_motion)
        masked_motion = torch.where(mask == 1, masked_emb, in_motion)
        body_hint = self.motion_encoder(masked_motion)                      # [B,T,256]

        body_hint_face = self.bodyhints_face(body_hint)                     # [B,T,256]
        body_hint_body = self.bodyhints_body(body_hint)                     # [B,T,256]

        # --- Add seed-pose hints to rhythmic fusion ---
        face_cond = fusion_face + body_hint_face                            # [B,T,256]
        body_cond = fusion_body + body_hint_body                            # [B,T,256]

        # --- Merge into a single condition sequence ---
        full_cond = self.cond_merge(
            torch.cat([face_cond, body_cond], dim=-1)                       # [B,T,512]
        )                                                                    # [B,T,256]

        # --- Speaker embedding (added to noisy-latent, not condition) ---
        spk_face = self.speaker_encoder_face(in_id).squeeze(2)              # [B,T,256]
        spk_body = self.speaker_encoder_body(in_id).squeeze(2)              # [B,T,256]

        # --- Downsampled for contrastive losses ---
        face_cond_down = self.cond_face_down(
            face_cond.permute(0, 2, 1)
        ).permute(0, 2, 1)                                                  # [B,T//4,256]
        body_cond_down = self.cond_body_down(
            body_cond.permute(0, 2, 1)
        ).permute(0, 2, 1)                                                  # [B,T//4,256]

        return {
            "full_cond":      full_cond,        # [B, T,    256]
            "fusion_face":    fusion_face,       # [B, T,    256]  (for RVQ cross-attn)
            "fusion_body":    fusion_body,       # [B, T,    256]  (for RVQ cross-attn)
            "in_word_face":   in_word_face,      # [B, T,    256]  (for contrastive)
            "in_word_body":   in_word_body,      # [B, T,    256]  (for contrastive)
            "face_cond_down": face_cond_down,    # [B, T//4, 256]
            "body_cond_down": body_cond_down,    # [B, T//4, 256]
            "spk_face":       spk_face,          # [B, T,    256]
            "spk_body":       spk_body,          # [B, T,    256]
        }

    def _add_speaker_to_latent(self, x: torch.Tensor, spk_face, spk_body):
        """
        Add speaker embeddings to the body-part latent tensor.
        x: [B, T', 4, D]
        spk_face / spk_body: [B, T, D]  (full resolution — we pool to T')
        """
        T_prime = x.shape[1]
        # adaptive average pool from T → T'
        sf = F.adaptive_avg_pool1d(spk_face.permute(0, 2, 1), T_prime).permute(0, 2, 1)  # [B,T',D]
        sb = F.adaptive_avg_pool1d(spk_body.permute(0, 2, 1), T_prime).permute(0, 2, 1)  # [B,T',D]

        x = x.clone()
        x[:, :, self.PART_FACE,  :] = x[:, :, self.PART_FACE,  :] + sf
        x[:, :, self.PART_UPPER, :] = x[:, :, self.PART_UPPER, :] + sb
        x[:, :, self.PART_HANDS, :] = x[:, :, self.PART_HANDS, :] + sb
        x[:, :, self.PART_LOWER, :] = x[:, :, self.PART_LOWER, :] + sb
        return x

    def _run_rvq_heads(self, face_lat, upper_lat, hands_lat, lower_lat, cond_dict):
        """
        Given level-0 latents for each part, run classifiers + predict_residual_zq
        to produce RVQ predictions at all 6 levels.

        Returns:
            rec_*  : [B, 6, 1, T', 256]   — latent reconstructions per level
            cls_*  : [B, T', 256, 6]       — classification logits per level
        """
        fusion_face = cond_dict["fusion_face"]
        fusion_body = cond_dict["fusion_body"]

        # ---- level-0 classifiers ----
        cls0_face  = self.face_classifier(face_lat)
        cls0_upper = self.upper_classifier(upper_lat)
        cls0_hands = self.hands_classifier(hands_lat)
        cls0_lower = self.lower_classifier(lower_lat)

        # ---- levels 1-5 via predict_residual_zq ----
        zq1_f, zq2_f, zq3_f, zq4_f, zq5_f, ci1_f, ci2_f, ci3_f, ci4_f, ci5_f = \
            self.predict_res_face(face_lat, fusion_face)
        zq1_u, zq2_u, zq3_u, zq4_u, zq5_u, ci1_u, ci2_u, ci3_u, ci4_u, ci5_u = \
            self.predict_res_upper(upper_lat, fusion_body)
        zq1_h, zq2_h, zq3_h, zq4_h, zq5_h, ci1_h, ci2_h, ci3_h, ci4_h, ci5_h = \
            self.predict_res_hands(hands_lat, fusion_body)
        zq1_l, zq2_l, zq3_l, zq4_l, zq5_l, ci1_l, ci2_l, ci3_l, ci4_l, ci5_l = \
            self.predict_res_lower(lower_lat, fusion_body)

        # ---- stack level-0..5 ----
        rec_face = torch.stack([face_lat,  zq1_f, zq2_f, zq3_f, zq4_f, zq5_f], dim=1).unsqueeze(2)
        rec_upper = torch.stack([upper_lat, zq1_u, zq2_u, zq3_u, zq4_u, zq5_u], dim=1).unsqueeze(2)
        rec_hands = torch.stack([hands_lat, zq1_h, zq2_h, zq3_h, zq4_h, zq5_h], dim=1).unsqueeze(2)
        rec_lower = torch.stack([lower_lat, zq1_l, zq2_l, zq3_l, zq4_l, zq5_l], dim=1).unsqueeze(2)

        cls_face  = torch.stack([cls0_face,  ci1_f, ci2_f, ci3_f, ci4_f, ci5_f], dim=-1)
        cls_upper = torch.stack([cls0_upper, ci1_u, ci2_u, ci3_u, ci4_u, ci5_u], dim=-1)
        cls_hands = torch.stack([cls0_hands, ci1_h, ci2_h, ci3_h, ci4_h, ci5_h], dim=-1)
        cls_lower = torch.stack([cls0_lower, ci1_l, ci2_l, ci3_l, ci4_l, ci5_l], dim=-1)

        return {
            "rec_face": rec_face,   "rec_upper": rec_upper,
            "rec_hands": rec_hands, "rec_lower": rec_lower,
            "cls_face": cls_face,   "cls_upper": cls_upper,
            "cls_hands": cls_hands, "cls_lower": cls_lower,
        }

    # ------------------------------------------------------------------
    #  forward  (training)
    # ------------------------------------------------------------------

    def forward(
        self,
        in_audio=None,
        in_word=None,           # accepted but NOT used (text → Sparse branch)
        mask=None,
        is_train=False,
        in_motion=None,
        use_attentions=True,
        use_word=True,          # ignored (no text in base)
        in_id=None,
        hubert=None,
        # ── NEW: Flow-Matching-specific inputs ──
        target_latents=None,    # dict: face/upper/hands/lower  each [B,T',256]
    ):
        """
        Training forward.

        ``target_latents`` contains the VQ level-0 ground-truth latent for
        each body part (extracted from ``loaded_data["zq_*"][:, 0, 0]`` in the
        trainer).  These serve as  x₁  in the Flow Matching objective.

        Returns dict identical to ``duogesture_base.forward()`` plus ``fm_loss``.
        """
        # ---- encode rhythmic conditions ----
        cond_dict = self._encode_conditions(in_audio, hubert, in_motion, mask, in_id)
        full_cond = cond_dict["full_cond"]                       # [B, T, 256]

        # ---- contrastive losses ----
        #   (computed on downsampled condition features as proxy for latent)
        face_cond_down = cond_dict["face_cond_down"]             # [B, T//4, 256]
        body_cond_down = cond_dict["body_cond_down"]             # [B, T//4, 256]
        hubert_cons_loss = self.hubert_face_cons_loss(face_cond_down, cond_dict["in_word_face"])
        beat_cons_loss   = self.beat_cons_loss(body_cond_down, cond_dict["in_word_body"])

        # ---- Flow Matching training ----
        if target_latents is not None:
            # Stack body-part targets into  x₁  [B, T', 4, 256]
            x_1 = torch.stack([
                target_latents["face"],
                target_latents["upper"],
                target_latents["hands"],
                target_latents["lower"],
            ], dim=2)                                            # [B, T', 4, 256]

            B, T_prime, P, D = x_1.shape

            # Sample noise  x₀ ∼ N(0, I)
            x_0 = torch.randn_like(x_1)

            # Sample timestep  t ∼ Beta(α, β)
            t = sample_beta_timesteps(B, self.beta_alpha, self.beta_beta, device=x_1.device)

            # OT interpolation  x_t  and target velocity  u_t
            x_t, target_v = ot_conditional_flow(x_0, x_1, t, self.sigma_min)

            # Add speaker embeddings to noisy latent
            x_t = self._add_speaker_to_latent(x_t, cond_dict["spk_face"], cond_dict["spk_body"])

            # Classifier-free guidance dropout: with probability p, zero the condition
            if self.training and self.cfg_drop_prob > 0:
                drop_mask = (torch.rand(B, 1, 1, device=full_cond.device) < self.cfg_drop_prob).float()
                full_cond_dropped = full_cond * (1.0 - drop_mask)
            else:
                full_cond_dropped = full_cond

            # Predict velocity field  v_t = f_θ(x_t, t, c)
            v_pred = self.velocity_net(x_t, t, full_cond_dropped)

            # Flow Matching loss:  || v_pred − u_t ||²
            fm_loss = flow_matching_loss(v_pred, target_v)

            # --- Teacher-forced RVQ heads (use ground-truth as level-0) ---
            face_lat  = target_latents["face"]
            upper_lat = target_latents["upper"]
            hands_lat = target_latents["hands"]
            lower_lat = target_latents["lower"]
        else:
            # No targets (inference path) — generate via ODE
            T_prime = in_audio.shape[1] // 4   # 64 → 16
            shape = (in_audio.shape[0], T_prime, self.NUM_PARTS, 256)
            gen = ode_euler_sample(
                self.velocity_net, full_cond, shape,
                num_steps=self.num_inference_steps,
                sigma_min=self.sigma_min,
                device=in_audio.device,
            )
            face_lat  = gen[:, :, self.PART_FACE]
            upper_lat = gen[:, :, self.PART_UPPER]
            hands_lat = gen[:, :, self.PART_HANDS]
            lower_lat = gen[:, :, self.PART_LOWER]
            fm_loss   = torch.tensor(0.0, device=in_audio.device)

        # ---- RVQ heads → rec + cls at all 6 levels ----
        rvq_out = self._run_rvq_heads(face_lat, upper_lat, hands_lat, lower_lat, cond_dict)

        return {
            "fm_loss":           fm_loss,
            "hubert_cons_loss":  hubert_cons_loss,
            "beat_cons_loss":    beat_cons_loss,
            **rvq_out,
        }

    # ------------------------------------------------------------------
    #  forward_latent  (used by Sparse module for fusion)
    # ------------------------------------------------------------------

    def forward_latent(
        self,
        in_audio=None,
        in_word=None,
        mask=None,
        is_test=None,
        in_motion=None,
        use_attentions=True,
        use_word=True,
        in_id=None,
        hubert=None,
    ):
        """
        Generate base-motion latents via ODE integration.

        Output dict is compatible with ``duogesture_sparse``'s ``latent`` argument
        for the adaptive fusion:
            q_m = MLP(ψ · q_s + (1−ψ) · q_b)

        Returns:
            face_latent  : [B, T', 256]
            upper_latent : [B, T', 256]
            lower_latent : [B, T', 256]
            hands_latent : [B, T', 256]
        """
        cond_dict = self._encode_conditions(in_audio, hubert, in_motion, mask, in_id)
        full_cond = cond_dict["full_cond"]

        B  = in_audio.shape[0]
        T  = in_audio.shape[1]
        T_prime = T // 4                                        # 64 → 16
        shape   = (B, T_prime, self.NUM_PARTS, 256)

        gen = ode_euler_sample(
            self.velocity_net,
            full_cond,
            shape,
            num_steps=self.num_inference_steps,
            sigma_min=self.sigma_min,
            device=in_audio.device,
        )

        return {
            "face_latent":  gen[:, :, self.PART_FACE],
            "upper_latent": gen[:, :, self.PART_UPPER],
            "lower_latent": gen[:, :, self.PART_LOWER],
            "hands_latent": gen[:, :, self.PART_HANDS],
        }


# ╔══════════════════════════════════════════════════════════════════════╗
# ║                          SMOKE  TEST                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    from utils import config
    args = config.parse_args()

    # ensure required attrs exist
    for attr, val in [
        ("hidden_size", 768), ("audio_f", 256), ("motion_f", 256),
        ("pose_dims", 330), ("pose_length", 64), ("pre_frames", 4),
        ("vae_codebook_size", 256), ("vae_layer", 4), ("vae_length", 240),
    ]:
        if not hasattr(args, attr):
            setattr(args, attr, val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GestureLSMBaseMotion(args).to(device)
    print(f"[smoke test] model params: {sum(p.numel() for p in model.parameters()):,}")

    B, T = 2, 64
    in_audio  = torch.randn(B, T, 3, device=device)
    in_word   = torch.randint(0, 10, (B, T), device=device)
    mask      = torch.ones(B, T, 337, device=device)
    mask[:, :4, :] = 0.0
    in_motion = torch.randn(B, T, 337, device=device)
    in_id     = torch.randint(0, 25, (B, T, 1), device=device)
    hubert    = torch.randn(B, T, 1024, device=device)

    # dummy VQ targets (level-0 only)
    T_prime = T // 4  # 16
    target_latents = {
        "face":  torch.randn(B, T_prime, 256, device=device),
        "upper": torch.randn(B, T_prime, 256, device=device),
        "hands": torch.randn(B, T_prime, 256, device=device),
        "lower": torch.randn(B, T_prime, 256, device=device),
    }

    # --- training forward ---
    model.train()
    out = model(
        in_audio, in_word, mask,
        is_train=True, in_motion=in_motion, in_id=in_id, hubert=hubert,
        target_latents=target_latents,
    )
    print(f"[smoke test] fm_loss       = {out['fm_loss'].item():.4f}")
    print(f"[smoke test] rec_face      = {out['rec_face'].shape}")    # [B, 6, 1, 16, 256]
    print(f"[smoke test] cls_upper     = {out['cls_upper'].shape}")   # [B, 16, 256, 6]

    # --- inference: forward_latent ---
    model.eval()
    lat = model.forward_latent(
        in_audio, in_word, mask,
        in_motion=in_motion, in_id=in_id, hubert=hubert,
    )
    print(f"[smoke test] face_latent   = {lat['face_latent'].shape}")   # [B, 16, 256]
    print(f"[smoke test] upper_latent  = {lat['upper_latent'].shape}")
    print(f"[smoke test] hands_latent  = {lat['hands_latent'].shape}")
    print(f"[smoke test] lower_latent  = {lat['lower_latent'].shape}")

    print("[smoke test] PASSED ✓")
