"""
PhysicsSmootherSMPLX — Gate-modulated per-joint physics smoothing for DuoGesture.

Design principle (from S-VIB gate-as-muscle-force analogy):

    τ_j  =  τ_base  ·  max( sqrt(m_j / m_max), τ_floor )  ·  (1 − ψ)  ·  (1 + α · σ²)

Where:
    τ_j      : per-joint smoothing coefficient
    τ_base   : base smoothing strength (hyperparameter, default 0.50)
    m_j      : normalised mass fraction for joint j (De Leva 1996)
    τ_floor  : minimum shape factor (default 0.10) — prevents fingers from
               getting zero smoothing; justified by the empirical finding that
               fingers peak at 0.51 Hz (dragged appendages), not 1.14 Hz
               (free pendulums)
    ψ        : S-VIB gate activation  (1 = semantic,  0 = beat)
    σ²       : VIB posterior variance  (uncertainty)

The previous formula (τ_base × m_j) used *linear* mass scaling which collapsed
τ to near-zero for arms (0.009), wrists (0.002), and fingers (0.0001). The
sqrt compression reduces the 4-decade mass range to 2 decades, and the floor
ensures no joint is left unsmoothed.

Interpretation:
    ψ ≈ 1  (semantic gesture) → τ_j → 0  →  trust the prediction  (muscle-driven)
    ψ ≈ 0  (beat gesture)     → τ_j large →  apply physics EMA     (inertia)
    σ² high (uncertain)       →  more smoothing (conservative)
    Heavy joints              →  more smoothing (Newton's 2nd law)

Applied as exponential moving average (EMA):
    θ_t  =  (1 − τ_j) · θ_pred_t  +  τ_j · θ_{t−1}

Two modes of operation:
    1. **Training**:  latent-space jerk loss on VQ latents, weighted by body-part
       mass × beat strength.  Cheap surrogate for full per-joint physics.
    2. **Inference**: per-joint EMA smoothing on decoded rot6d poses [B, T, 55, 6].
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────────────────────
# SMPL-X joint order  (beat_smplx_full — see dataloaders/data_tools.py)
# ──────────────────────────────────────────────────────────────────────────────
#  0: pelvis         1: left_hip       2: right_hip       3: spine1
#  4: left_knee      5: right_knee     6: spine2          7: left_ankle
#  8: right_ankle    9: spine3        10: left_foot      11: right_foot
# 12: neck          13: left_collar   14: right_collar   15: head
# 16: left_shoulder 17: right_shoulder 18: left_elbow    19: right_elbow
# 20: left_wrist    21: right_wrist   22: jaw            23: left_eye
# 24: right_eye     25–39: left-hand fingers (15)  40–54: right-hand fingers (15)
# ──────────────────────────────────────────────────────────────────────────────

# De Leva (1996) body-segment mass fractions (male, 75 kg reference body)
# mapped to SMPL-X joints by anatomical correspondence.
_SMPLX_MASS_FRACTIONS = [
    0.1117,   #  0: pelvis       (lower trunk)
    0.1416,   #  1: left_hip     (thigh)
    0.1416,   #  2: right_hip    (thigh)
    0.1633,   #  3: spine1       (abdomen / mid trunk)
    0.0433,   #  4: left_knee    (shank)
    0.0433,   #  5: right_knee   (shank)
    0.1596,   #  6: spine2       (upper trunk / thorax)
    0.0137,   #  7: left_ankle   (foot)
    0.0137,   #  8: right_ankle  (foot)
    0.1596,   #  9: spine3       (upper trunk)
    0.0137,   # 10: left_foot
    0.0137,   # 11: right_foot
    0.0350,   # 12: neck
    0.0080,   # 13: left_collar  (clavicle region)
    0.0080,   # 14: right_collar
    0.0694,   # 15: head
    0.0271,   # 16: left_shoulder  (upper arm)
    0.0271,   # 17: right_shoulder
    0.0162,   # 18: left_elbow   (forearm)
    0.0162,   # 19: right_elbow
    0.0061,   # 20: left_wrist   (hand)
    0.0061,   # 21: right_wrist
    0.0070,   # 22: jaw          (fraction of head mass)
    0.0010,   # 23: left_eye
    0.0010,   # 24: right_eye
] + [0.0004] * 30  # 25-54: finger joints  (~hand_mass / 15 joints)

# Body-part → joint indices (matching beat_smplx_upper/lower/hands)
_UPPER_JOINT_IDX = [3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]  # 13 joints
_HANDS_JOINT_IDX = list(range(25, 55))                                   # 30 joints
_LOWER_JOINT_IDX = [0, 1, 2, 3, 4, 5, 6, 7, 8]                         #  9 joints

# Public aliases for use by trainer and eval code
UPPER_JOINT_IDX = _UPPER_JOINT_IDX
HANDS_JOINT_IDX = _HANDS_JOINT_IDX
LOWER_JOINT_IDX = _LOWER_JOINT_IDX


class PhysicsSmootherSMPLX(nn.Module):
    """
    Gate-modulated per-joint physics smoother for SMPL-X gesture generation.

    Two operational modes:
        1. ``compute_latent_jerk_loss()``  — training regularisation on VQ latents
        2. ``forward_inference()``         — inference-time EMA on decoded rot6d poses
    """

    def __init__(
        self,
        num_joints: int = 55,
        tau_base: float = 0.50,
        tau_floor: float = 0.10,
        alpha: float = 1.0,
        pose_fps: int = 30,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.tau_base = tau_base
        self.tau_floor = tau_floor
        self.alpha = alpha
        self.pose_fps = pose_fps

        # Per-joint mass fractions, normalised so max = 1.0
        raw = torch.tensor(
            _SMPLX_MASS_FRACTIONS[:num_joints], dtype=torch.float32
        )
        raw = raw / raw.max()
        self.register_buffer("mass_fractions", raw)  # [J]

        # Sqrt-compressed shape factors with floor clamp.
        # sqrt reduces the 4-decade mass range to 2 decades;
        # floor ensures fingers (m_j ≈ 0.002) still get τ ≥ τ_base × τ_floor.
        shape = torch.sqrt(raw)                        # sqrt(m_j / m_max) since raw is already normalised to max=1
        shape = torch.clamp(shape, min=tau_floor)      # [J]
        self.register_buffer("mass_shape", shape)      # [J]  — used by compute_tau

        # Pre-compute body-part average masses for latent-space jerk loss
        self.register_buffer(
            "mass_upper", raw[_UPPER_JOINT_IDX].mean().unsqueeze(0)
        )
        self.register_buffer(
            "mass_hands", raw[_HANDS_JOINT_IDX].mean().unsqueeze(0)
        )
        self.register_buffer(
            "mass_lower", raw[_LOWER_JOINT_IDX].mean().unsqueeze(0)
        )

    # ── Core τ computation ──────────────────────────────────────────

    def compute_tau(self, gate_psi, logvar=None):
        """
        Per-joint smoothing coefficient (sqrt-compressed + floor):

            shape_j = max( sqrt(m_j / m_max), τ_floor )
            τ_j     = τ_base · shape_j · (1 − ψ) · (1 + α · σ²)

        The sqrt compression and floor clamp are precomputed in __init__
        and stored as ``self.mass_shape``  [J].

        Args:
            gate_psi:  [B, T', 1]  semantic gate probability
            logvar:    [B, T', z]  VIB posterior log-variance (optional)

        Returns:
            tau:       [B, T', J]
        """
        shape = self.mass_shape[None, None, :]             # [1, 1, J]
        tau = self.tau_base * shape * (1.0 - gate_psi)     # [B, T', J]

        if logvar is not None:
            sigma_sq = logvar.exp().mean(dim=-1, keepdim=True)  # [B, T', 1]
            tau = tau * (1.0 + self.alpha * sigma_sq)

        return tau.clamp(0.0, 0.95)

    # ── Inference: per-joint EMA smoothing ──────────────────────────

    def smooth_poses(self, poses_rot6d, tau, initial_state=None):
        """
        Differentiable per-joint EMA:
            θ_t = (1 − τ_j) · θ_pred_t  +  τ_j · θ_{t−1}

        Args:
            poses_rot6d:  [B, T, J, 6]
            tau:          [B, T, J]   (full temporal resolution)
            initial_state: [B, J, 6]  optional seed from previous chunk

        Returns:
            smoothed:     [B, T, J, 6]
        """
        B, T, J, D = poses_rot6d.shape
        if initial_state is not None:
            prev = initial_state
        else:
            prev = poses_rot6d[:, 0]
        out = []

        for t in range(T):
            if t == 0 and initial_state is None:
                out.append(poses_rot6d[:, 0])
            else:
                tau_t = tau[:, t, :, None]                      # [B, J, 1]
                frame = (1.0 - tau_t) * poses_rot6d[:, t] + tau_t * prev
                out.append(frame)
            prev = out[-1]

        return torch.stack(out, dim=1)                      # [B, T, J, 6]

    def forward_inference(self, poses_rot6d, gate_psi, logvar=None, initial_state=None):
        """
        Full inference smoothing pipeline.

        Args:
            poses_rot6d:  [B, T, J, 6]  decoded poses
            gate_psi:     [B, T']        semantic gate probability (soft)
            logvar:       [B, T', z]     VIB posterior log-variance
            initial_state: [B, J, 6]     optional EMA seed from previous chunk

        Returns:
            smoothed:     [B, T, J, 6]
        """
        B, T, J, D = poses_rot6d.shape
        T_prime = gate_psi.shape[1]

        # Ensure gate_psi is [B, T', 1]
        if gate_psi.dim() == 2:
            gate_psi = gate_psi.unsqueeze(-1)

        # Compute τ at gate resolution [B, T', J]
        tau_low = self.compute_tau(gate_psi, logvar)

        # Upsample τ to full temporal resolution [B, T, J]
        ratio = T // T_prime
        if ratio > 1:
            tau = (
                tau_low.unsqueeze(2)
                .expand(-1, -1, ratio, -1)
                .reshape(B, T_prime * ratio, J)
            )
            # Handle remainder frames if T not divisible by T'
            if tau.shape[1] < T:
                pad = tau[:, -1:].expand(-1, T - tau.shape[1], -1)
                tau = torch.cat([tau, pad], dim=1)
            tau = tau[:, :T]
        else:
            tau = tau_low

        return self.smooth_poses(poses_rot6d, tau, initial_state=initial_state)

    # ── Training: latent-space jerk loss ────────────────────────────

    @staticmethod
    def _jerk(x):
        """3rd finite difference  (jerk).   x: [B, T, C] → [B, T-3, C]"""
        vel  = x[:, 1:] - x[:, :-1]       # Δ¹
        acc  = vel[:, 1:] - vel[:, :-1]    # Δ²
        jerk = acc[:, 1:] - acc[:, :-1]    # Δ³
        return jerk

    @staticmethod
    def _accel(x):
        """2nd finite difference  (acceleration).  x: [B, T, C] → [B, T-2, C]"""
        vel = x[:, 1:] - x[:, :-1]        # Δ¹
        acc = vel[:, 1:] - vel[:, :-1]    # Δ²
        return acc

    def compute_latent_jerk_loss(
        self,
        latent_upper,   # [B, T', C]
        latent_hands,   # [B, T', C]
        latent_lower,   # [B, T', C]
        gate_psi,       # [B, T', 1]   ← detached by caller
        logvar=None,    # [B, T', z]   ← detached by caller
    ):
        """
        Physics-aware jerk regularisation on VQ latents.

        Weight = body_mass · (1 − ψ) · (1 + α · σ²)

        Heavier body parts + beat segments → higher jerk penalty.
        Semantic segments + light parts   → nearly unconstrained.
        """
        # Common temporal weight from gate & uncertainty
        beat_weight = 1.0 - gate_psi                                   # [B, T', 1]
        if logvar is not None:
            sigma_sq = logvar.exp().mean(dim=-1, keepdim=True)         # [B, T', 1]
            modifier = beat_weight * (1.0 + self.alpha * sigma_sq)     # [B, T', 1]
        else:
            modifier = beat_weight

        # Per-body-part jerk magnitude  (squared L2 over latent dims)
        jerk_upper = self._jerk(latent_upper).pow(2).mean(dim=-1, keepdim=True)  # [B, T'-3, 1]
        jerk_hands = self._jerk(latent_hands).pow(2).mean(dim=-1, keepdim=True)
        jerk_lower = self._jerk(latent_lower).pow(2).mean(dim=-1, keepdim=True)

        # Align temporal modifier to jerk length  (trim first 3 frames)
        mod = modifier[:, 3:]                                          # [B, T'-3, 1]

        # Weighted sum with body-part mass coefficients
        loss = (
            self.mass_upper * (jerk_upper * mod).mean()
            + self.mass_hands * (jerk_hands * mod).mean()
            + self.mass_lower * (jerk_lower * mod).mean()
        )
        return loss

    # ── Convenience: compute jerk loss on full rot6d poses ──────────

    def compute_pose_jerk_loss(self, poses_rot6d, tau=None):
        """
        Optional: jerk on decoded poses [B, T, J, 6]  (more expensive).
        Weighted by τ if provided, otherwise by raw mass fractions.
        """
        jerk = self._jerk(poses_rot6d)                                 # [B, T-3, J, 6]
        jerk_mag = jerk.pow(2).sum(dim=-1)                             # [B, T-3, J]

        if tau is not None:
            weight = tau[:, 3:]                                        # [B, T-3, J]
        else:
            weight = self.mass_fractions[None, None, :]                # [1, 1, J]

        return (jerk_mag * weight).mean()

    def compute_pose_accel_loss(self, poses_rot6d, tau=None):
        """
        Softer alternative to jerk: squared acceleration on decoded poses [B, T, J, 6].
        Allows 1 Hz beat swings while penalising unnatural high-freq noise.
        Weighted by τ if provided, otherwise by raw mass fractions.
        """
        acc = self._accel(poses_rot6d)                                 # [B, T-2, J, 6]
        acc_mag = acc.pow(2).sum(dim=-1)                               # [B, T-2, J]

        if tau is not None:
            weight = tau[:, 2:]                                        # [B, T-2, J]
        else:
            weight = self.mass_fractions[None, None, :]                # [1, 1, J]

        return (acc_mag * weight).mean()

    def compute_pose_inertia_residual_loss(self, poses_rot6d, tau=None):
        """
        Inertia-residual (constant-velocity) loss on decoded poses [B, T, J, 6].

        For each frame t≥2 we extrapolate from constant-velocity:
            θ̂_t  = θ_{t-1} + (θ_{t-1} − θ_{t-2})
        and penalise the squared prediction error ||θ_t − θ̂_t||².

        Motivation (BEAT2 evidence, 200 clips, N_beat=666k, N_sem=56k):
            Arms:  beat residual 0.0106 rad  vs  semantic 0.0165 rad  (ratio 1.56×, p≈0)
            Hands: beat 0.0152  vs  semantic 0.0208  (ratio 1.37×, p≈0)
            Torso: beat 0.0021  vs  semantic 0.0030  (ratio 1.44×, p≈0)

        Beat gestures are naturally more inertially compliant: lower τ (beat weight)
        → higher penalty, which trains the model to generate smooth beat motion.

        tau: [B, T, J] — physics weight per frame/joint (beat frames have LOW τ,
             so (1 - τ) gives high weight to beat; we follow the same convention as
             the other physics methods and let the caller decide the sign / scale).
        """
        # θ̂_t = θ_{t-1} + v_{t-1},  v_{t-1} = θ_{t-1} − θ_{t-2}
        vel = poses_rot6d[:, 1:] - poses_rot6d[:, :-1]        # [B, T-1, J, 6]
        pred = poses_rot6d[:, 1:-1] + vel[:, :-1]             # [B, T-2, J, 6]  (CV prediction)
        residual = (poses_rot6d[:, 2:] - pred).pow(2).sum(dim=-1)  # [B, T-2, J]

        if tau is not None:
            weight = tau[:, 2:]                                # [B, T-2, J]
        else:
            weight = self.mass_fractions[None, None, :]        # [1, 1, J]

        return (residual * weight).mean()

    def compute_adaptive_inertia_loss(self, poses_rot6d, tau=None, eps=1e-4):
        """
        Velocity-adaptive inertia-residual loss on decoded poses [B, T, J, 6].

        Like compute_pose_inertia_residual_loss but normalised by speed²:

            L = Σ_t  τ_t  ·  ||θ_t − θ̂_t||²  /  (||v_{t-1}||² + ε)

        This makes the loss *dynamic*:
          - Fast limb + big deviation  →  small penalty  (semantic redirect OK)
          - Slow limb + any deviation  →  large penalty  (beat phases must be smooth)
          - ε controls the scale and prevents division by zero

        The motivation is that beats follow momentum (our BEAT2 data shows 1.56×
        lower inertia residual for arms), but a static penalty doesn't know whether
        the limb is mid-stroke or at rest.  Normalising by speed makes the prior
        adapt to the motion state.
        """
        vel = poses_rot6d[:, 1:] - poses_rot6d[:, :-1]            # [B, T-1, J, 6]
        pred = poses_rot6d[:, 1:-1] + vel[:, :-1]                 # [B, T-2, J, 6]
        residual_sq = (poses_rot6d[:, 2:] - pred).pow(2).sum(dim=-1)  # [B, T-2, J]
        speed_sq = vel[:, :-1].pow(2).sum(dim=-1)                 # [B, T-2, J]

        normalised = residual_sq / (speed_sq + eps)                # [B, T-2, J]

        if tau is not None:
            weight = tau[:, 2:]                                    # [B, T-2, J]
        else:
            weight = self.mass_fractions[None, None, :]            # [1, 1, J]

        return (normalised * weight).mean()
