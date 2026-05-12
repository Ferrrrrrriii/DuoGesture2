"""
Fréchet Gesture Distance (FGD) metric for evaluating co-speech gesture generation.

Formula:
    FGD(g, ĝ) = ||μ_r - μ_g||² + Tr(Σ_r + Σ_g - 2*(Σ_r Σ_g)^(1/2))

where μ_r, Σ_r are the mean and covariance of real gesture latent features,
and μ_g, Σ_g are those of the generated gesture latent features.

Usage (CLI):
    # Quick sanity check with dummy mean-pool encoder:
    python utils/fgd.py --real demo/2_scott_0_1_1.npz --gen demo/2_scott_0_1_1_test_moclip.npz

    # With the pretrained upper-body RVQVAE encoder:
    python utils/fgd.py \\
        --real demo/2_scott_0_1_1.npz \\
        --gen  demo/2_scott_0_1_1_test_moclip.npz \\
        --encoder_type rvqvae_upper

    # All body-part encoders (upper, hands, lower) concatenated:
    python utils/fgd.py \\
        --real demo/2_scott_0_1_1.npz \\
        --gen  demo/2_scott_0_1_1_test_moclip.npz \\
        --encoder_type rvqvae_all
"""

import os
import sys
import warnings
import argparse
import types
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import sqrtm

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path when this script is run directly
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Pose layout for the DuoGesture / BEAT2 smplxflame_30 representation.
# All slices index into the flat 165-dim axis-angle vector (55 joints × 3).
# The RVQVAE encoders were trained on rot6d, so the axis-angle slice is
# converted to rot6d before encoding (doubles the dimension).
#
# Lower-body note: the lower RVQVAE was trained on a 61-dim input:
#   rot6d(9 lower joints) = 54  +  global_trans = 3  +  foot_contact = 4
# See aelower_trainer.py for the original assembly.
# ---------------------------------------------------------------------------
JOINT_SLICES: Dict[str, Tuple[int, int]] = {
    # joints 0-12 → 13 × 3 = 39 axis-angle dims → rot6d 78 dims
    "upper": (0, 39),
    # joints 25-54 → 30 × 3 = 90 axis-angle dims → rot6d 180 dims
    "hands": (75, 165),
    # joints 0-8 → 9 × 3 = 27 axis-angle dims → rot6d 54 dims  (+7 appended)
    "lower": (0, 27),
}

# RVQVAE config per body part (matches the yaml configs and pretrained weights)
RVQVAE_CONFIGS: Dict[str, dict] = {
    "upper": dict(vae_test_dim=78,  vae_length=256, vae_layer=2,
                  ckpt="weights/pretrained_vq/rvq_upper_500.bin"),
    "hands": dict(vae_test_dim=180, vae_length=256, vae_layer=2,
                  ckpt="weights/pretrained_vq/rvq_hands_500.bin"),
    # vae_test_dim=61: rot6d(9 joints)=54 + trans=3 + foot_contact=4
    "lower": dict(vae_test_dim=61,  vae_length=256, vae_layer=4,
                  ckpt="weights/pretrained_vq/rvq_lower_600.bin"),
    "face":  dict(vae_test_dim=100, vae_length=256, vae_layer=2,
                  ckpt="weights/pretrained_vq/rvq_face_600.bin"),
}

WINDOW_SIZE   = 64   # frames per clip fed to the RVQVAE encoder
WINDOW_STRIDE = 32   # hop size between consecutive clips


# ===========================================================================
# 1.  Data Loading
# ===========================================================================

def load_sequences(
    paths: Union[str, List[str]],
    key: str = "poses",
) -> torch.Tensor:
    """
    Load gesture frame sequences from .npz or .pt files.

    Args:
        paths:  A single path or list of paths (.npz / .pt).
        key:    Array key when reading .npz files (default: "poses").

    Returns:
        Tensor of shape (N_frames, D) or (N_seq, T, D); always float32.
        Multiple files are concatenated along dim=0.
    """
    if isinstance(paths, str):
        paths = [paths]

    chunks: List[torch.Tensor] = []
    for p in paths:
        ext = os.path.splitext(p)[-1].lower()
        if ext == ".npz":
            data = np.load(p, allow_pickle=True)
            if key not in data:
                raise KeyError(
                    f"Key '{key}' not found in '{p}'. Available: {list(data.files)}"
                )
            chunks.append(torch.from_numpy(data[key].copy()).float())
        elif ext == ".pt":
            obj = torch.load(p, map_location="cpu", weights_only=False)
            if isinstance(obj, torch.Tensor):
                chunks.append(obj.float())
            elif isinstance(obj, dict):
                for candidate in ("poses", "motion", "joints", "data"):
                    if candidate in obj:
                        chunks.append(obj[candidate].float())
                        break
                else:
                    raise KeyError(
                        f"Cannot find motion tensor in .pt dict at '{p}'. "
                        f"Keys: {list(obj.keys())}"
                    )
            else:
                raise TypeError(f"Unsupported type {type(obj)} loaded from '{p}'.")
        else:
            raise ValueError(f"Unsupported extension '{ext}'. Use .npz or .pt.")

    return torch.cat(chunks, dim=0)


# ===========================================================================
# 2.  Feature Extraction
# ===========================================================================

class _DummyEncoder(nn.Module):
    """
    Dummy feature extractor (no learned weights).

    Accepts:
        (N, D)       – returned as-is (cast to float32).
        (N, T, D)    – mean-pooled over the time axis → (N, D).
    """
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            return x.mean(dim=1)
        raise ValueError(f"Expected 2-D or 3-D tensor, got shape {tuple(x.shape)}")


def _make_rvqvae_args(cfg: dict) -> types.SimpleNamespace:
    """Build a minimal args namespace accepted by the RVQVAE constructor."""
    return types.SimpleNamespace(
        vae_test_dim=cfg["vae_test_dim"],
        vae_length=cfg["vae_length"],
        vae_layer=cfg["vae_layer"],
        vae_codebook_size=256,
        vae_grow=[1, 1, 2, 1],
        variational=False,
    )


def _load_rvqvae(cfg: dict, device: str = "cpu") -> nn.Module:
    """
    Instantiate and load a pretrained RVQVAE.

    Args:
        cfg:    One entry from RVQVAE_CONFIGS.
        device: Torch device string.

    Returns:
        RVQVAE in eval mode on the requested device.
    """
    from models.rvq import RVQVAE
    args = _make_rvqvae_args(cfg)
    model = RVQVAE(args).to(device).eval()

    ckpt_path = cfg["ckpt"]
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"RVQVAE checkpoint not found at '{ckpt_path}'. "
            "Run download_weights.py or pass a custom --encoder_ckpt path."
        )
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    # DuoGesture checkpoints are saved as {"model_state": OrderedDict(...)}
    state_dict = state.get("model_state", state)
    # Strip 'module.' prefix when saved from DataParallel / DDP
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    return model


def _axis_angle_to_rot6d(poses: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle rotations to 6D rotation representation.

    Args:
        poses: (..., J*3) axis-angle tensor.

    Returns:
        (..., J*6) rot6d tensor.
    """
    from utils.rotation_conversions import axis_angle_to_matrix, matrix_to_rotation_6d
    *batch_dims, flat_dim = poses.shape
    j = flat_dim // 3
    aa  = poses.reshape(*batch_dims, j, 3)
    mat = axis_angle_to_matrix(aa)       # (..., J, 3, 3)
    r6d = matrix_to_rotation_6d(mat)     # (..., J, 6)
    return r6d.reshape(*batch_dims, j * 6)


def _chunk_frames(frames: torch.Tensor, window: int, stride: int) -> torch.Tensor:
    """
    Slide a fixed window over a flat (N_frames, D) sequence.

    Returns:
        (N_windows, window, D).  Tail frames are dropped if < window.
        If the sequence is shorter than one window it is zero-padded.
    """
    N, D = frames.shape
    indices = list(range(0, N - window + 1, stride))
    if not indices:
        pad = torch.zeros(window - N, D, device=frames.device, dtype=frames.dtype)
        return torch.cat([frames, pad], dim=0).unsqueeze(0)  # (1, window, D)
    return torch.stack([frames[i: i + window] for i in indices], dim=0)


class RVQVAEEncoderWrapper(nn.Module):
    """
    Wraps a pretrained RVQVAE as a frame-level feature extractor for FGD.

    Pipeline for flat (N_frames, 165) input:
        1. Select the joints for this body part (axis-angle slice).
        2. Convert axis-angle → rot6d (doubles the feature dimension).
        3. Slide a fixed window  →  (N_windows, window_size, D_part).
        4. RVQVAE.map2zp()       →  (N_windows, code_dim, T_enc).
        5. Mean-pool over T_enc  →  (N_windows, code_dim).

    FGD statistics are computed over the N_windows dimension.
    """

    def __init__(
        self,
        rvqvae: nn.Module,
        joint_slice: Tuple[int, int],
        use_rot6d: bool = True,
        window: int = WINDOW_SIZE,
        stride: int = WINDOW_STRIDE,
    ):
        super().__init__()
        self.rvqvae = rvqvae
        self.j_start, self.j_end = joint_slice
        self.use_rot6d = use_rot6d
        self.window = window
        self.stride = stride

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N_frames, 165) flat axis-angle sequence, OR
               (N_seq, T, 165)  batched sequences (flattened to frames first).

        Returns:
            features: (N_windows, code_dim)
        """
        x = x.float()

        # Flatten batched sequences into one long frame sequence
        if x.ndim == 3:
            N, T, D = x.shape
            x = x.reshape(N * T, D)

        # Select relevant joint dimensions
        x = x[:, self.j_start: self.j_end]            # (N_frames, D_part)

        # Convert axis-angle → rot6d
        if self.use_rot6d:
            x = _axis_angle_to_rot6d(x)               # (N_frames, D_part * 2)

        # Slide window into clips
        clips = _chunk_frames(x, self.window, self.stride)   # (W, window, D)

        # Encode in mini-batches to avoid OOM
        feats = []
        enc_device = next(self.rvqvae.parameters()).device
        for i in range(0, len(clips), 64):
            z = self.rvqvae.map2zp(clips[i: i + 64].to(enc_device))  # (b, code_dim, T_enc)
            feats.append(z.mean(dim=-1))                              # (b, code_dim)
        return torch.cat(feats, dim=0)                                # (N_windows, code_dim)


class LowerBodyEncoderWrapper(nn.Module):
    """
    Specialised encoder wrapper for the lower-body RVQVAE.

    The checkpoint was trained on a 61-dim input assembled as:
        rot6d( joints 0-8 ) = 54 dims
        global_trans         =  3 dims  (loaded from npz 'trans' key if available,
                                         otherwise zero-padded)
        foot_contact         =  4 dims  (always zero-padded at inference)

    Args:
        rvqvae:      pretrained RVQVAE(vae_test_dim=61) instance.
        trans:       Optional (N_frames, 3) tensor of global translations.
                     If provided and frame counts match, it is appended to the
                     rot6d features.  Otherwise zeros are used.
        window / stride: same windowing convention as RVQVAEEncoderWrapper.
    """

    def __init__(
        self,
        rvqvae: nn.Module,
        trans: Optional[torch.Tensor] = None,
        window: int = WINDOW_SIZE,
        stride: int = WINDOW_STRIDE,
    ):
        super().__init__()
        self.rvqvae = rvqvae
        self.trans = trans   # (N_frames, 3) or None
        self.window = window
        self.stride = stride

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N_frames, 165) flat axis-angle sequence, OR
               (N_seq, T, 165) batched (flattened to frames first).

        Returns:
            features: (N_windows, 256)
        """
        x = x.float()
        if x.ndim == 3:
            N, T, D = x.shape
            x = x.reshape(N * T, D)

        n_frames = x.shape[0]

        # 1. Select lower-body joints (0-26 → 9 joints × 3 axis-angle)
        aa = x[:, 0:27]                             # (N_frames, 27)

        # 2. Convert axis-angle → rot6d
        r6d = _axis_angle_to_rot6d(aa)              # (N_frames, 54)

        # 3. Global translation (3 dims)
        if self.trans is not None and self.trans.shape[0] == n_frames:
            tr = self.trans.to(x.device).float()    # (N_frames, 3)
        else:
            tr = torch.zeros(n_frames, 3, device=x.device)

        # 4. Foot contact (4 dims) — zero at inference
        fc = torch.zeros(n_frames, 4, device=x.device)

        # 5. Assemble 61-dim input
        inp = torch.cat([r6d, tr, fc], dim=-1)      # (N_frames, 61)

        # 6. Slide window & encode
        clips = _chunk_frames(inp, self.window, self.stride)  # (W, window, 61)
        feats = []
        enc_device = next(self.rvqvae.parameters()).device
        for i in range(0, len(clips), 64):
            z = self.rvqvae.map2zp(clips[i: i + 64].to(enc_device))  # (b, 256, T_enc)
            feats.append(z.mean(dim=-1))                              # (b, 256)
        return torch.cat(feats, dim=0)                                # (N_windows, 256)


class MultiPartRVQVAEEncoder(nn.Module):
    """
    Runs one RVQVAEEncoderWrapper per body part and concatenates their
    features, giving a richer full-body representation for FGD.

    Supported parts: "upper", "hands", "lower".
    ("face" needs a separate expression array and is not wired in here.)

    When loading .npz files that have a 'trans' key, pass the translation
    array via the `real_trans` / `gen_trans` arguments of compute_fgd_from_files
    for a more accurate lower-body encoding.
    """

    def __init__(
        self,
        parts: List[str] = ("upper", "hands", "lower"),
        device: str = "cpu",
        trans: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict()
        for part in parts:
            cfg = RVQVAE_CONFIGS[part]
            rvqvae = _load_rvqvae(cfg, device)
            if part == "lower":
                self.encoders[part] = LowerBodyEncoderWrapper(rvqvae, trans=trans)
            else:
                self.encoders[part] = RVQVAEEncoderWrapper(rvqvae, JOINT_SLICES[part])
        print(f"[INFO] MultiPartRVQVAEEncoder loaded parts: {list(parts)}")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N_frames, 165) or (N_seq, T, 165).

        Returns:
            (N_windows, sum_of_code_dims) concatenated features.
        """
        return torch.cat([enc(x) for enc in self.encoders.values()], dim=-1)


# ===========================================================================
# 3.  Statistical Computation
# ===========================================================================

def compute_statistics(
    features: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute empirical mean and covariance of latent features.

    Args:
        features: (N, latent_dim)  – at least 2 samples required.

    Returns:
        mu:    (latent_dim,)             float64 mean vector.
        sigma: (latent_dim, latent_dim)  float64 covariance matrix.
    """
    if features.ndim != 2:
        raise ValueError(
            f"Expected 2-D feature tensor (N, latent_dim), got shape {tuple(features.shape)}."
        )
    if features.shape[0] < 2:
        raise ValueError(
            f"Need at least 2 samples to compute covariance, got {features.shape[0]}."
        )
    feats = features.cpu().numpy().astype(np.float64)
    mu    = np.mean(feats, axis=0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma


# ===========================================================================
# 4.  Fréchet Distance Calculation
# ===========================================================================

def calculate_fgd(
    real_features: torch.Tensor,
    generated_features: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """
    Compute the Fréchet Gesture Distance (FGD).

    FGD = ||μ_r - μ_g||²  +  Tr(Σ_r + Σ_g - 2·(Σ_r Σ_g)^½)

    Args:
        real_features:      (N_r, latent_dim) features from GT gestures.
        generated_features: (N_g, latent_dim) features from generated gestures.
        eps:                Ridge regularisation added to both covariance
                            matrices before the matrix square root.

    Returns:
        fgd_score: Scalar FGD value (float).
    """
    mu_r, sigma_r = compute_statistics(real_features)
    mu_g, sigma_g = compute_statistics(generated_features)

    # Regularise covariances
    eye = np.eye(sigma_r.shape[0], dtype=np.float64)
    sigma_r = sigma_r + eps * eye
    sigma_g = sigma_g + eps * eye

    # Squared Euclidean distance between means
    diff = mu_r - mu_g
    mean_sq_dist = float(diff @ diff)

    # Matrix square root of Σ_r · Σ_g  (complex in general due to float precision)
    sqrt_product, _ = sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(sqrt_product):
        max_imag = np.max(np.abs(sqrt_product.imag))
        if max_imag > 1e-3:
            warnings.warn(
                f"Large imaginary component in sqrtm ({max_imag:.4e}). "
                "Covariance matrices may be ill-conditioned.",
                RuntimeWarning,
                stacklevel=2,
            )
        sqrt_product = sqrt_product.real

    # Trace term
    trace_term = float(np.trace(sigma_r + sigma_g - 2.0 * sqrt_product))

    return mean_sq_dist + trace_term


# ===========================================================================
# Convenience wrapper
# ===========================================================================

def compute_fgd_from_files(
    real_paths: Union[str, List[str]],
    gen_paths: Union[str, List[str]],
    encoder: nn.Module = None,
    device: str = "cpu",
    npz_key: str = "poses",
) -> float:
    """
    Full pipeline: load files → extract features → compute FGD.

    Args:
        real_paths: Path(s) to real / GT .npz or .pt files.
        gen_paths:  Path(s) to generated .npz or .pt files.
        encoder:    Any nn.Module mapping (N, D) or (N, T, D) → (M, latent_dim).
                    Defaults to _DummyEncoder if None.
        device:     Torch device.
        npz_key:    Array key for .npz files.

    Returns:
        Scalar FGD score.
    """
    if encoder is None:
        encoder = _DummyEncoder()
    encoder.eval().to(device)

    real_seq = load_sequences(real_paths, key=npz_key).to(device)
    gen_seq  = load_sequences(gen_paths,  key=npz_key).to(device)

    with torch.no_grad():
        real_feats = encoder(real_seq)
        gen_feats  = encoder(gen_seq)

    return calculate_fgd(real_feats, gen_feats)


# ===========================================================================
# __main__ entry point
# ===========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute Fréchet Gesture Distance (FGD).",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--real", nargs="+", required=True,
                        help="Path(s) to GT gesture files (.npz or .pt).")
    parser.add_argument("--gen",  nargs="+", required=True,
                        help="Path(s) to generated gesture files (.npz or .pt).")
    parser.add_argument("--npz_key", default="poses",
                        help="Array key in .npz files (default: 'poses').")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device (default: auto).")
    parser.add_argument(
        "--encoder_type",
        default="dummy",
        choices=["dummy", "rvqvae_upper", "rvqvae_hands", "rvqvae_lower",
                 "rvqvae_face", "rvqvae_all"],
        help=(
            "Feature extractor to use:\n"
            "  dummy        — mean-pool over feature dim (no learned weights).\n"
            "  rvqvae_upper — pretrained upper-body RVQVAE  (weights/pretrained_vq/rvq_upper_500.bin).\n"
            "  rvqvae_hands — pretrained hands RVQVAE       (weights/pretrained_vq/rvq_hands_500.bin).\n"
            "  rvqvae_lower — pretrained lower-body RVQVAE  (weights/pretrained_vq/rvq_lower_600.bin).\n"
            "  rvqvae_face  — pretrained face RVQVAE        (weights/pretrained_vq/rvq_face_600.bin).\n"
            "  rvqvae_all   — upper + hands + lower features concatenated.\n"
        ),
    )
    parser.add_argument(
        "--encoder_ckpt", default=None,
        help="Override the checkpoint path for single-part RVQVAE encoders.",
    )
    parser.add_argument("--window", type=int, default=WINDOW_SIZE,
                        help=f"Clip window size in frames (default: {WINDOW_SIZE}).")
    parser.add_argument("--stride", type=int, default=WINDOW_STRIDE,
                        help=f"Clip window stride in frames (default: {WINDOW_STRIDE}).")
    args = parser.parse_args()

    device = args.device

    # -----------------------------------------------------------------------
    # Build encoder
    # -----------------------------------------------------------------------
    if args.encoder_type == "dummy":
        print("[INFO] Using dummy mean-pool encoder (no learned weights).")
        encoder = _DummyEncoder()

    elif args.encoder_type == "rvqvae_all":
        print("[INFO] Loading pretrained RVQVAE for all body parts (upper + hands + lower).")
        # Load trans from first real file if available (improves lower-body encoding)
        _trans = None
        try:
            _d = np.load(args.real[0], allow_pickle=True)
            if "trans" in _d.files:
                _trans = torch.from_numpy(_d["trans"].copy()).float()
        except Exception:
            pass
        encoder = MultiPartRVQVAEEncoder(
            parts=["upper", "hands", "lower"], device=device, trans=_trans
        )

    else:
        # Single body-part RVQVAE  (e.g. "rvqvae_upper" → part = "upper")
        part = args.encoder_type.replace("rvqvae_", "")
        cfg  = dict(RVQVAE_CONFIGS[part])

        if args.encoder_ckpt:
            cfg["ckpt"] = args.encoder_ckpt

        print(f"[INFO] Loading pretrained RVQVAE — body part : '{part}'")
        print(f"[INFO]                           — checkpoint: '{cfg['ckpt']}'")
        rvqvae  = _load_rvqvae(cfg, device)

        if part == "lower":
            # Load trans from npz for more accurate lower-body encoding
            _trans = None
            try:
                _d = np.load(args.real[0], allow_pickle=True)
                if "trans" in _d.files:
                    _trans = torch.from_numpy(_d["trans"].copy()).float()
            except Exception:
                pass
            encoder = LowerBodyEncoderWrapper(
                rvqvae, trans=_trans,
                window=args.window, stride=args.stride,
            )
        else:
            encoder = RVQVAEEncoderWrapper(
                rvqvae,
                JOINT_SLICES[part],
                use_rot6d=True,
                window=args.window,
                stride=args.stride,
            )

    encoder.eval().to(device)

    # -----------------------------------------------------------------------
    # Load sequences
    # -----------------------------------------------------------------------
    print(f"\n[INFO] Loading real sequences:      {args.real}")
    print(f"[INFO] Loading generated sequences: {args.gen}")

    real_seq = load_sequences(args.real, key=args.npz_key).to(device)
    gen_seq  = load_sequences(args.gen,  key=args.npz_key).to(device)

    print(f"[INFO] Real sequences shape:      {tuple(real_seq.shape)}")
    print(f"[INFO] Generated sequences shape: {tuple(gen_seq.shape)}")

    # -----------------------------------------------------------------------
    # Extract latent features
    # -----------------------------------------------------------------------
    with torch.no_grad():
        real_feats = encoder(real_seq)
        gen_feats  = encoder(gen_seq)

    print(f"[INFO] Real feature matrix shape:      {tuple(real_feats.shape)}")
    print(f"[INFO] Generated feature matrix shape: {tuple(gen_feats.shape)}")

    # -----------------------------------------------------------------------
    # Compute FGD
    # -----------------------------------------------------------------------
    mu_r, sigma_r = compute_statistics(real_feats)
    mu_g, sigma_g = compute_statistics(gen_feats)

    fgd = calculate_fgd(real_feats, gen_feats)

    print("\n" + "=" * 54)
    print(f"  Mean (real)       : {mu_r[:4].round(4)} ...")
    print(f"  Mean (generated)  : {mu_g[:4].round(4)} ...")
    print(f"  Cov trace (real)      : {np.trace(sigma_r):.4f}")
    print(f"  Cov trace (generated) : {np.trace(sigma_g):.4f}")
    print(f"\n  >>> FGD Score: {fgd:.6f}")
    print("=" * 54)
