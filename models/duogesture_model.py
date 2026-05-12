import sys, pathlib
THIS_FILE = pathlib.Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]  # .../DuoGesture
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import copy
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
from dataloaders.build_vocab import Vocab
# from .utils.layer import BasicBlock
try:
    from .motion_encoder import *
except ImportError:
    from models.motion_encoder import *


# De Leva mass fractions mapped to SMPL-X 55-joint layout.
SMPLX_MASS_FRACTIONS_55 = [
    0.1117, 0.1416, 0.1416, 0.1633, 0.0433, 0.0433, 0.1596,
    0.0137, 0.0137, 0.1596, 0.0137, 0.0137, 0.0350, 0.0080,
    0.0080, 0.0694, 0.0271, 0.0271, 0.0162, 0.0162, 0.0061,
    0.0061, 0.0070, 0.0010, 0.0010,
] + [0.0004] * 30

MASS_GROUPS = {
    "shoulder": [16, 17],
    "forearm": [18, 19, 20, 21],
    "arm_combined": [16, 17, 18, 19, 20, 21],
    "core": [0, 3, 6, 9, 12],
}


# from .transformer import CAG
class WavEncoder(nn.Module):
    def __init__(self, out_dim, audio_in=1):
        super().__init__() 
        self.out_dim = out_dim
        self.feat_extractor = nn.Sequential( 
                BasicBlock(audio_in, out_dim//4, 15, 5, first_dilation=1600, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//4, out_dim//4, 15, 1, first_dilation=7, ),
                BasicBlock(out_dim//4, out_dim//2, 15, 6, first_dilation=0, downsample=True),
                BasicBlock(out_dim//2, out_dim//2, 15, 1, first_dilation=7),
                BasicBlock(out_dim//2, out_dim, 15, 3,  first_dilation=0,downsample=True),     
            )
    def forward(self, wav_data):
        # print("WavEncoder input:",wav_data.shape)
        if wav_data.dim() == 2:
            wav_data = wav_data.unsqueeze(1) 
            # print("WavEncoder input dim==2:",wav_data.shape)
        else:
            wav_data = wav_data.transpose(1, 2)
            # print("WavEncoder input dim!=2:",wav_data.shape)
        out = self.feat_extractor(wav_data)
        # exit()
        # print("WavEncoder output:",out.shape) 
        # exit()
        return out.transpose(1, 2) # bs, t, 256

    
class MLP(nn.Module):
    def __init__(self, in_dim, hidden_size, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, out_dim)
        )
    def forward(self, inputs):
        out = self.mlp(inputs)
        return out


class PeriodicPositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, period=15, max_seq_len=65): 
        super(PeriodicPositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(period, d_model)
        position = torch.arange(0, period, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, period, d_model)
        repeat_num = (max_seq_len//period) + 1
        pe = pe.repeat(1, repeat_num, 1) # (1, repeat_num, period, d_model)
        self.register_buffer('pe', pe)
    def forward(self, x):
        # print(self.pe.shape, x.shape)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class SemanticVIB(nn.Module):
    """
    Two-stream Semantic Variational Information Bottleneck (S-VIB).
    Replaces the hard gate (nn.Linear(256, 2)).

    Stream A (semantic):  body_semantic_down  [bs, T, 256]  — WHAT content
    Stream B (timing):    in_word_body_down   [bs, T, 256]  — WHEN onset (low-cap)

    The timing stream is deliberately bottlenecked to prevent full HuBERT
    rhythm from leaking into the semantic gate decision.
    """
    def __init__(self, sem_dim=256, timing_dim=64, z_dim=16):
        super(SemanticVIB, self).__init__()
        self.z_dim = z_dim
        self.timing_dim = timing_dim

        # Low-capacity timing projection — bottleneck HuBERT to onset-only
        self.timing_proj = nn.Sequential(
            nn.Linear(sem_dim, timing_dim),
            nn.LayerNorm(timing_dim),
            nn.GELU(),
        )

        in_dim = sem_dim + timing_dim  # 256 + 64 = 320

        # VIB encoder heads
        self.fc_mu     = nn.Linear(in_dim, z_dim)
        self.fc_logvar = nn.Linear(in_dim, z_dim)

        # Classifier: z -> gate logits
        self.classifier = nn.Sequential(
            nn.Linear(z_dim, z_dim * 2),
            nn.GELU(),
            nn.Linear(z_dim * 2, 2),
        )

    def reparameterize(self, mu, logvar):
        """Reparameterisation trick — paper Eq. (svib_q).

        During training we sample z = mu + sigma * eps, eps ~ N(0,I).
        This keeps gradients flowing through mu and logvar while
        injecting stochasticity that regularises the gate manifold.
        At inference we use the posterior mean (no sampling noise),
        so the gate is deterministic and reproducible.

        Args:
            mu:     [bs, T, z_dim]  posterior mean
            logvar: [bs, T, z_dim]  log-variance (not variance)
        Returns:
            z:      [bs, T, z_dim]  sampled (train) or mean (eval)
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def forward(self, body_semantic_down, in_word_body_down):
        """
        Args:
            body_semantic_down: [bs, T, 256] — semantic content stream
            in_word_body_down:  [bs, T, 256] — HuBERT-derived timing stream
        Returns:
            gate_logits: [bs, T, 2]
            mu:          [bs, T, z_dim]
            logvar:      [bs, T, z_dim]
        """
        timing = self.timing_proj(in_word_body_down)           # [bs, T, 64]
        x = torch.cat([body_semantic_down, timing], dim=-1)   # [bs, T, 320]

        mu     = self.fc_mu(x)                                 # [bs, T, z_dim]
        logvar = self.fc_logvar(x)
        logvar = torch.clamp(logvar, -10.0, 2.0)              # stability

        z = self.reparameterize(mu, logvar)                    # [bs, T, z_dim]
        gate_logits = self.classifier(z)                       # [bs, T, 2]

        return gate_logits, mu, logvar


class predict_residual_zq(nn.Module):
    def __init__(self,latent_dim=256,num_head = 8,ffn_dim = 1024, dropout = 0.1, n_tokens = 5):
        super(predict_residual_zq, self).__init__()
        self.cross_layer = nn.TransformerDecoderLayer(
            d_model=latent_dim, nhead=num_head, dim_feedforward=ffn_dim, dropout=dropout
        )
        self.cross_attn_1 = nn.TransformerDecoder(self.cross_layer, num_layers=3)
        self.cross_attn_2 = nn.TransformerDecoder(self.cross_layer, num_layers=3)
        self.cross_attn_3 = nn.TransformerDecoder(self.cross_layer, num_layers=3)
        self.cross_attn_4 = nn.TransformerDecoder(self.cross_layer, num_layers=3)
        self.cross_attn_5 = nn.TransformerDecoder(self.cross_layer, num_layers=3)
        self.map_zq2 = nn.Linear(latent_dim*2, latent_dim)
        self.map_zq3 = nn.Linear(latent_dim*3, latent_dim)
        self.map_zq4 = nn.Linear(latent_dim*4, latent_dim)
        self.map_zq5 = nn.Linear(latent_dim*5, latent_dim)
        self.classfier_1 = MLP(latent_dim, latent_dim, latent_dim)
        self.classfier_2 = MLP(latent_dim, latent_dim, latent_dim)
        self.classfier_3 = MLP(latent_dim, latent_dim, latent_dim)
        self.classfier_4 = MLP(latent_dim, latent_dim, latent_dim)
        self.classfier_5 = MLP(latent_dim, latent_dim, latent_dim)
    def forward(self, pre_index, cond):
        pre_index = pre_index.permute(1, 0, 2)
        cond = cond.permute(1, 0, 2)
        zq_1 = self.cross_attn_1(tgt=pre_index, memory=cond)
        zq_1 = zq_1 + pre_index
        zq_index_1 = self.classfier_1(zq_1.permute(1,0,2))
        
        pre_zq2 = torch.cat([pre_index, zq_1],dim=-1)
        # pre_zq2 = pre_zq2.permute(1,0,2)
        pre_zq2 = self.map_zq2(pre_zq2)
        # pre_zq2 = pre_zq2.permute(1,0,2)

        zq_2 = self.cross_attn_2(tgt=pre_zq2, memory=cond)
        zq_2 = zq_2 + pre_zq2

        zq_index_2 = self.classfier_2(zq_2.permute(1,0,2))
        
        pre_zq3 = torch.cat([pre_index, zq_1,zq_2],dim=-1)
        # pre_zq3 = pre_zq3.permute(1,0,2)
        pre_zq3 = self.map_zq3(pre_zq3)
        # pre_zq3 = pre_zq3.permute(1,0,2)
        zq_3 = self.cross_attn_3(tgt=pre_zq3, memory=cond)
        zq_3 = zq_3 + pre_zq3
        zq_index_3 = self.classfier_3(zq_3.permute(1,0,2))

        pre_zq4 = torch.cat([pre_index, zq_1,zq_2,zq_3],dim=-1)
        # pre_zq4 = pre_zq4.permute(1,0,2)
        pre_zq4 = self.map_zq4(pre_zq4)
        # pre_zq4 = pre_zq4.permute(1,0,2)
        zq_4 = self.cross_attn_4(tgt=pre_zq4, memory=cond)
        zq_4 = zq_4 + pre_zq4
        zq_index_4 = self.classfier_4(zq_4.permute(1,0,2))

        pre_zq5 = torch.cat([pre_index, zq_1,zq_2,zq_3,zq_4],dim=-1)
        # pre_zq5 = pre_zq5.permute(1,0,2)
        pre_zq5 = self.map_zq5(pre_zq5)
        # pre_zq5 = pre_zq5.permute(1,0,2)
        zq_5 = self.cross_attn_5(tgt=pre_zq5, memory=cond)
        zq_5 = zq_5 + pre_zq5
        zq_index_5 = self.classfier_5(zq_5.permute(1,0,2))

        zq_1 = zq_1.permute(1,0,2)
        zq_2 = zq_2.permute(1,0,2)
        zq_3 = zq_3.permute(1,0,2)
        zq_4 = zq_4.permute(1,0,2)
        zq_5 = zq_5.permute(1,0,2)
        return zq_1, zq_2, zq_3, zq_4, zq_5, zq_index_1, zq_index_2, zq_index_3, zq_index_4, zq_index_5
    
class RhythmicIdentificationLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(RhythmicIdentificationLoss, self).__init__()
        self.temperature = temperature

    def forward(self, facial_features, audio_features):
        """
        facial_features: Tensor of shape (batch_size, num_frames, feature_dim)
        audio_features: Tensor of shape (batch_size, num_frames, feature_dim)
        """
        # Normalize the features to compute cosine similarity
        # bs, t, c = facial_features.shape
        audio_features = F.avg_pool1d(audio_features.permute(0, 2, 1), kernel_size=4).permute(0, 2, 1)
        facial_features = F.normalize(facial_features, p=2, dim=-1)
        audio_features = F.normalize(audio_features, p=2, dim=-1)
       
        # Compute cosine similarity between each pair of facial and audio features
        similarity_matrix = torch.matmul(facial_features, audio_features.transpose(-1, -2)) / self.temperature
       
        # Create labels: each frame should correspond to itself (diagonal alignment)
        batch_size, num_frames, _ = facial_features.shape
        labels = torch.arange(num_frames).unsqueeze(0).repeat(batch_size, 1).to(facial_features.device)
       
        # Compute the InfoNCE loss
        loss = F.cross_entropy(similarity_matrix, labels)
       
        return loss

# class RhythmicIdentificationLoss(nn.Module):
#     """
#     支持三种时间对齐方式：
#       - mode="interp":  把 audio 插值到 facial 的长度 T（推荐、简单）
#       - mode="pool":    两路一起 pool 到同一长度（stride=kernel）
#       - mode="local":   局部对齐 InfoNCE，不需等长
#     """
#     def __init__(self, temperature=0.2, mode="interp",
#                  pool_kernel=4, local_window=3):
#         super().__init__()
#         assert mode in ["interp", "pool", "local"]
#         self.temperature = temperature
#         self.mode = mode
#         self.pool_kernel = pool_kernel
#         self.local_window = local_window  # 仅用于 local 模式

#     @staticmethod
#     def _norm(x):
#         # x: (B, T, C)
#         return F.normalize(x, p=2, dim=-1)

#     def _pool_both(self, facial, audio):
#         """
#         输入:
#           facial: (B, T,  C)
#           audio:  (B, Ta, C)
#         输出:
#           对齐后相同长度的 (facial_p, audio_p)，形状均为 (B, Tp, C)
#         """
#         k = self.pool_kernel
#         # (B,T,C) -> (B,C,T) 方便 1d pooling
#         facial_p = F.avg_pool1d(facial.transpose(1, 2), kernel_size=k, stride=k).transpose(1, 2)
#         audio_p  = F.avg_pool1d(audio.transpose(1, 2),  kernel_size=k, stride=k).transpose(1, 2)
#         Tp = min(facial_p.size(1), audio_p.size(1))
#         return facial_p[:, :Tp], audio_p[:, :Tp]

#     def forward(self, facial_features, audio_features):
#         """
#         facial_features: (B, T,  C)
#         audio_features:  (B, Ta, C)
#         """
#         facial_features = facial_features.transpose(1, 2)
#         assert facial_features.dim() == 3 and audio_features.dim() == 3, "输入必须是三维 (B,T,C)"
#         B, T,  C  = facial_features.shape
#         Ba, Ta, Ca = audio_features.shape
#         assert B == Ba and C == Ca, f"batch 或通道不匹配: {(B,C)} vs {(Ba,Ca)}"

#         if self.mode == "interp":
#             # 把 audio 插值到长度 T
#             audio = audio_features.transpose(1, 2)                               # (B, C, Ta)
#             audio = F.interpolate(audio, size=T, mode="linear", align_corners=False)
#             audio = audio.transpose(1, 2).detach()                               # (B, T, C) 目标端停梯度
#             facial = facial_features

#             # 归一化
#             facial = self._norm(facial)                                          # (B,T,C)
#             audio  = self._norm(audio)                                           # (B,T,C)

#             # 相似度 (B, T, T)
#             sim = torch.matmul(facial, audio.transpose(-1, -2)) / self.temperature
#             # 每个时间步 t 的正确类别是 t（对角线）
#             labels = torch.arange(T, device=sim.device).unsqueeze(0).expand(B, T)  # (B,T)
#             loss = F.cross_entropy(sim.reshape(-1, T), labels.reshape(-1))
#             return loss

#         elif self.mode == "pool":
#             # 同步下采样并裁切到相同的 Tp
#             facial_p, audio_p = self._pool_both(facial_features, audio_features.detach())  # (B,Tp,C)
#             B2, Tp, C2 = facial_p.shape
#             assert B2 == B and C2 == C
#             facial_p = self._norm(facial_p)
#             audio_p  = self._norm(audio_p)
#             sim = torch.matmul(facial_p, audio_p.transpose(-1, -2)) / self.temperature  # (B, Tp, Tp)
#             labels = torch.arange(Tp, device=sim.device).unsqueeze(0).expand(B, Tp)     # (B,Tp)
#             loss = F.cross_entropy(sim.reshape(-1, Tp), labels.reshape(-1))
#             return loss

#         else:  # self.mode == "local"
#             # 局部对齐：对每个 t，仅在 audio 的 [t'-w, t'+w] 中做分类
#             facial = self._norm(facial_features)               # (B,T,C)
#             audio  = self._norm(audio_features.detach())       # (B,Ta,C)

#             # 全对相似度 (B, T, Ta)
#             sim = torch.matmul(facial, audio.transpose(-1, -2)) / self.temperature

#             w = self.local_window
#             # 线性速率映射：t -> t'≈ t * Ta / T
#             t_idx = torch.arange(T, device=sim.device, dtype=torch.float32) * (Ta / float(T))
#             t_idx = t_idx.round().long().clamp(0, Ta - 1)      # (T,)

#             # 构造 mask：窗口内=0，窗口外=-inf
#             mask = torch.full_like(sim, fill_value=float("-inf"))  # (B,T,Ta)
#             for t in range(T):
#                 center = t_idx[t].item()
#                 lo = max(0, int(center) - w)
#                 hi = min(Ta - 1, int(center) + w)
#                 mask[:, t, lo:hi+1] = 0.0

#             sim_masked = sim + mask  # (B,T,Ta)
#             labels = t_idx.unsqueeze(0).expand(B, T)           # (B,T)
#             loss = F.cross_entropy(sim_masked.reshape(B * T, Ta), labels.reshape(B * T))
#             return loss
     
class Recycle_loss(nn.Module):
    def __init__(self,face_dim=256, audio_dim=256):
        super(Recycle_loss, self).__init__()
        self.mlp_face = MLP(face_dim, 256, audio_dim)
        # self.mlp_audio = MLP(audio_dim, 256, face_dim)
        # self.audio_dim = audio_dim
        self.loss_func = nn.MSELoss()
    def forward(self, facial_features, audio_features):
        """
        facial_features: Tensor of shape (batch_size, num_frames, feature_dim)
        audio_features: Tensor of shape (batch_size, num_frames, feature_dim)
        """
        # Normalize the features to compute cosine similarity
        # bs, t, c = facial_features.shape
        audio_features = F.avg_pool1d(audio_features.permute(0, 2, 1), kernel_size=4).permute(0, 2, 1)
        # N, T, C = audio_features.shape
        pred_audio = self.mlp_face(facial_features)
       
        # Compute the InfoNCE loss
        # loss = F.cross_entropy(pred_audio.reshape(-1,C),)
        loss = self.loss_func(pred_audio, audio_features)
        loss = loss/0.05
        return loss

# --------- 轻量模块：数据集级归一化、通道Dropout、时间遮挡、对齐工具 ----------

class DatasetNorm1D(nn.Module):
    """
    对 B,T,C 的 HuBERT 特征做通道级数据集归一化：
      y = (x - mean) / std
    - mean/std 形状 (C,)
    - 未设置统计量时，自动回退到 LayerNorm(C)（等价“每个时间步单样本归一化”）
    """
    def __init__(self, C: int, eps: float = 1e-5, fallback_layernorm: bool = True):
        super().__init__()
        self.register_buffer("mean", torch.zeros(C), persistent=True)
        self.register_buffer("std", torch.ones(C), persistent=True)
        self.has_stats = False
        self.eps = eps
        self.fallback_ln = nn.LayerNorm(C) if fallback_layernorm else None

    @torch.no_grad()
    def set_stats(self, mean_1024: torch.Tensor, std_1024: torch.Tensor):
        std_1024 = std_1024.clamp(min=1e-6)
        self.mean.copy_(mean_1024)
        self.std.copy_(std_1024)
        self.has_stats = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, C]
        if self.has_stats:
            return (x - self.mean.view(1, 1, -1)) / (self.std.view(1, 1, -1) + self.eps)
        if self.fallback_ln is not None:
            return self.fallback_ln(x)
        return x


class FeatureDropout(nn.Module):
    """
    通道级 dropout（对 B,T,C 的 C 维作“整通道”mask，等价 1D 特征版 Dropout2d）
    仅在训练时生效；测试自动跳过。
    """
    def __init__(self, p: float = 0.05):
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0:
            return x
        B, T, C = x.shape
        mask = x.new_empty(B, 1, C).bernoulli_(1 - self.p) / (1 - self.p)  # 保期望不变
        return x * mask


class TimeMask(nn.Module):
    """
    简单的时间遮挡（SpecAugment 风格）：
      在时间维随机选择 num_mask 段，每段宽度 [1, max_width]，把这段置零。
    仅在训练时生效；测试自动跳过。
    """
    def __init__(self, max_width: int = 4, num_mask: int = 1):
        super().__init__()
        self.max_width = int(max_width)
        self.num_mask = int(num_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.max_width <= 0 or self.num_mask <= 0:
            return x
        B, T, C = x.shape
        for _ in range(self.num_mask):
            w = int(torch.randint(1, self.max_width + 1, (1,)).item())
            if w >= T:
                continue
            s = int(torch.randint(0, T - w + 1, (1,)).item())
            x[:, s:s+w, :] = 0.0
        return x


def match_time_to(x: torch.Tensor, T_target: int) -> torch.Tensor:
    """
    把 x 的时间长度对齐到 T_target。
    - x: [B, T, C]（或 [B, C, T] 均可，自动检测）
    - 使用线性插值
    """
    if x.dim() != 3:
        raise ValueError(f"match_time_to expects 3D tensor, got {x.shape}")
    # 兼容 [B, C, T] 输入
    channel_first = False
    if x.size(1) != x.size(-1) and x.size(2) != x.size(-1):  # 粗略判断
        # 假定当前是 [B, T, C]
        pass
    elif x.shape[1] < x.shape[2]:  # 多数是 [B, T, C]
        pass
    else:
        # 看起来像 [B, C, T]
        x = x.transpose(1, 2)  # -> [B, T, C]
        channel_first = True

    B, T, C = x.shape
    if T == T_target:
        return x if not channel_first else x.transpose(1, 2)
    # 插值需要 [B, C, T]
    x_cf = x.transpose(1, 2)
    x_resized = F.interpolate(x_cf, size=T_target, mode="linear", align_corners=True)
    x_out = x_resized.transpose(1, 2)  # -> [B, T_target, C]
    return x_out if not channel_first else x_out.transpose(1, 2)

class duogesture_base(nn.Module):
    def __init__(self, args=None):
        super(duogesture_base, self).__init__()
        self.args = args

        # Optional weight-informed conditioning for base/beat pathway.
        # none | arm_combined_core | split_arm_core | all_joints_core
        self._base_mass_cond_mode = str(getattr(args, 'base_mass_cond_mode', 'none')).strip().lower()
        if self._base_mass_cond_mode not in {
            'none', 'arm_combined_core', 'split_arm_core', 'all_joints_core'
        }:
            self._base_mass_cond_mode = 'none'
        self._base_mass_cond_scale = float(getattr(args, 'base_mass_cond_scale', 0.30))
        self._base_mass_cond_enabled = self._base_mass_cond_mode != 'none'
        # Learnable scalar gate (sigmoid) lets the model suppress mass priors when they conflict.
        self.base_mass_cond_gate = nn.Parameter(
            torch.tensor(float(getattr(args, 'base_mass_cond_gate_init', -4.0)))
        )

        base_mass_vec = self._build_mass_condition_vector(self._base_mass_cond_mode)
        self.register_buffer('base_mass_cond_vector', base_mass_vec)
        self.base_mass_cond_proj = nn.Sequential(
            nn.Linear(int(base_mass_vec.numel()), 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        self.base_mass_cond_hidden_proj = nn.Sequential(
            nn.Linear(256, self.args.hidden_size),
            nn.LayerNorm(self.args.hidden_size),
        )

        ######## constractive loss ########
        self.hubert_cons_loss = RhythmicIdentificationLoss(temperature=0.1)
        self.hubert_face_cons_loss = RhythmicIdentificationLoss(temperature=0.1)
        self.beat_cons_loss = RhythmicIdentificationLoss(temperature=0.1)
        self.beat_face_cons_loss = RhythmicIdentificationLoss(temperature=0.1)
        self.face_cons_mlp = MLP(256, args.hidden_size, 256)
        self.hands_cons_mlp = MLP(256, args.hidden_size, 256)
        

         # --------- HuBERT 特征预处理：数据集级归一化 + 轻扰动 ----------
        # self.hubert_dataset_norm = DatasetNorm1D(C=1024, fallback_layernorm=True)
        # self.feature_noise_sigma = float(getattr(args, "hubert_noise_sigma", 0.02))
        # self.feat_dropout = FeatureDropout(p=float(getattr(args, "hubert_featdrop_p", 0.05)))
        # self.time_mask = TimeMask(
        #     max_width=int(getattr(args, "hubert_timemask_w", 4)),
        #     num_mask=int(getattr(args, "hubert_timemask_n", 1))
        # )

        # --------- HuBERT 编码到 256 (Face/Body 各一份)，用 GN 替代 BN ----------
        def hubert_encoder_block():
            return nn.Sequential(
                nn.Conv1d(1024, 256, 3, 1, 1, bias=False),
                # nn.GroupNorm(32, 256),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Conv1d(256, 256, 3, 1, 1, bias=False)
            )
        self.hubert_encoder = hubert_encoder_block()
        self.hubert_encoder_body = hubert_encoder_block()
        # 融合阶段 dropout
        # self.fuse_dropout = nn.Dropout(p=float(getattr(args, "fuse_dropout_p", 0.05)))

        ##### predict_residual #####
        self.predict_res_face = predict_residual_zq(latent_dim=256,num_head = 8,ffn_dim = 1024, dropout = 0.1, n_tokens = 5)
        self.predict_res_hands = predict_residual_zq(latent_dim=256,num_head = 8,ffn_dim = 1024, dropout = 0.1, n_tokens = 5)
        self.predict_res_upper = predict_residual_zq(latent_dim=256,num_head = 8,ffn_dim = 1024, dropout = 0.1, n_tokens = 5)
        self.predict_res_lower = predict_residual_zq(latent_dim=256,num_head = 8,ffn_dim = 1024, dropout = 0.1, n_tokens = 5)

        ##### con1d #####
        self.face1d = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.face1d_2 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.body1d = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=2, stride=2)
        self.upper1d = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.hands1d = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.lower1d = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.spearker_encoder_body1d = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=2, stride=2)
        self.face_latent_mlp = MLP(256, args.hidden_size, 256)
        
        self.audio_pre_encoder_face = MLP(3, args.hidden_size, 256)
        self.audio_pre_encoder_body = MLP(3, args.hidden_size, 256)
        self.at_attn_face = nn.Linear(args.audio_f*2, args.audio_f*2)
        self.at_attn_body = nn.Linear(args.audio_f*2, args.audio_f*2)
        args_top = copy.deepcopy(self.args)
        args_top.vae_layer = 3
        args_top.vae_length = args.motion_f  # args.motion_f 256
        args_top.vae_test_dim = args.pose_dims+3+4 # 330 + 3 + 4 = 337
        self.motion_encoder = VQEncoderV6(args_top) # masked motion to latent bs t 337 to bs t 256
        
        # face decoder  hidden_size:768, audio_f:256, vae_codebook_size:256
        self.feature2face = nn.Linear(args.audio_f*2, args.hidden_size) # 256*2 to 768
        self.face2latent = nn.Linear(args.hidden_size, args.vae_codebook_size) # vae_codebook_size:256
        self.transformer_de_layer = nn.TransformerDecoderLayer(
            d_model=self.args.hidden_size,
            nhead=4,
            dim_feedforward=self.args.hidden_size*2,
            batch_first=True
            )
        self.transformer_de_fu_layer = nn.TransformerDecoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=256*2,
            batch_first=True
            )
        self.hands_face_decoder = nn.TransformerDecoder(self.transformer_de_fu_layer, num_layers=1)
        self.face_hands_decoder = nn.TransformerDecoder(self.transformer_de_fu_layer, num_layers=1)

        self.face_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=4)
        # pose_length = 64
        self.position_embeddings = PeriodicPositionalEncoding(self.args.hidden_size, period=self.args.pose_length, max_seq_len=self.args.pose_length)
        
        # motion decoder
        self.transformer_en_layer = nn.TransformerEncoderLayer(
            d_model=self.args.hidden_size,
            nhead=4,
            dim_feedforward=self.args.hidden_size*2,
            batch_first=True
            )
        self.motion_self_encoder = nn.TransformerEncoder(self.transformer_en_layer, num_layers=1)
        self.audio_feature2motion = nn.Linear(args.audio_f, args.hidden_size) # 256 to 768
        self.feature2motion = nn.Linear(args.motion_f, args.hidden_size) # 256 to 768

        self.bodyhints_face = MLP(args.motion_f, args.hidden_size, args.motion_f) # 256 to 256
        self.bodyhints_body = MLP(args.motion_f, args.hidden_size, args.motion_f) # 256 to 256
        self.motion2latent_upper = MLP(args.hidden_size, args.hidden_size, self.args.hidden_size)
        self.motion2latent_hands = MLP(args.hidden_size, args.hidden_size, self.args.hidden_size)
        self.motion2latent_lower = MLP(args.hidden_size, args.hidden_size, self.args.hidden_size)
        self.wordhints_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=8)
        
        self.upper_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=1)
        self.hands_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=1)
        self.lower_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=1)

        self.upper_hands_decoder = nn.TransformerDecoder(self.transformer_de_fu_layer, num_layers=1)
        self.lower_hands_decoder = nn.TransformerDecoder(self.transformer_de_fu_layer, num_layers=1)

        self.face_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)
        self.upper_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)
        self.hands_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)
        self.lower_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)

        self.mask_embeddings = nn.Parameter(torch.zeros(1, 1, self.args.pose_dims+3+4)) # [1, 1, 337]
        self.motion_down_upper = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_hands = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_lower = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self._reset_parameters()


        self.spearker_encoder_body = nn.Embedding(25, args.hidden_size)
        self.spearker_encoder_face = nn.Embedding(25, args.hidden_size)

    def _reset_parameters(self):
        nn.init.normal_(self.mask_embeddings, 0, self.args.hidden_size ** -0.5)

    def _build_mass_condition_vector(self, mode):
        if mode == 'none':
            return torch.zeros(1, dtype=torch.float32)

        mass = torch.tensor(SMPLX_MASS_FRACTIONS_55[:55], dtype=torch.float32)
        mass = mass / mass.sum().clamp(min=1e-8)

        shoulder = mass[MASS_GROUPS['shoulder']].mean()
        forearm = mass[MASS_GROUPS['forearm']].mean()
        arm_combined = mass[MASS_GROUPS['arm_combined']].mean()
        core = mass[MASS_GROUPS['core']].mean()

        if mode == 'arm_combined_core':
            return torch.stack([arm_combined, core], dim=0)
        if mode == 'split_arm_core':
            return torch.stack([shoulder, forearm, core], dim=0)
        if mode == 'all_joints_core':
            return torch.cat([mass, core.unsqueeze(0)], dim=0)
        return torch.zeros(1, dtype=torch.float32)

    def _apply_base_mass_prior(self, hidden_seq):
        if not self._base_mass_cond_enabled:
            return hidden_seq
        bs, t, _ = hidden_seq.shape
        mass_feat = self.base_mass_cond_vector.view(1, 1, -1).expand(bs, t, -1)
        mass_prior = self.base_mass_cond_proj(mass_feat)
        # Keep mass prior projection decoupled from audio projection to avoid
        # perturbing learned audio-to-motion alignment during finetuning.
        mass_prior = self.base_mass_cond_hidden_proj(mass_prior)
        gate = torch.sigmoid(self.base_mass_cond_gate)
        scale = self._base_mass_cond_scale * gate
        return hidden_seq + scale * mass_prior
    
    # 外部可在训练开始前注入 HuBERT 统计量（推荐）
    # @torch.no_grad()
    # def set_hubert_dataset_stats(self, mean_1024: torch.Tensor, std_1024: torch.Tensor):
    #     """
    #     mean/std: shape (1024,)
    #     """
    #     self.hubert_dataset_norm.set_stats(mean_1024.to(self.mask_embeddings.device),
    #                                        std_1024.to(self.mask_embeddings.device))
        
    def forward(self, in_audio=None, in_word=None, mask=None, is_train=False, in_motion=None, use_attentions=True, use_word=True, in_id = None, hubert=None):
         # ---- 1) HuBERT 预处理：标准化 + 训练期扰动 ----
        # hubert = self.hubert_dataset_norm(hubert)                 # 数据集级归一化/或 LN 回退
        # if is_train and self.feature_noise_sigma > 0:
        #     hubert = hubert + torch.randn_like(hubert) * self.feature_noise_sigma
        # if is_train:
        #     hubert = self.feat_dropout(hubert)                        # 通道dropout（仅训练）
        #     hubert = self.time_mask(hubert)                           # 时间遮挡（仅训练）

        in_word_face = self.hubert_encoder(hubert.permute(0, 2, 1)).permute(0, 2, 1)
        in_word_body = self.hubert_encoder_body(hubert.permute(0, 2, 1)).permute(0, 2, 1)
        bs, t, c = in_word_face.shape
        in_audio_face = self.audio_pre_encoder_face(in_audio) # [bs, t, 256]
        in_audio_body = self.audio_pre_encoder_body(in_audio) # [bs, t, 256]

        if use_attentions:           
            alpha_at_face = torch.cat([in_word_face, in_audio_face], dim=-1).reshape(bs, t, c*2)
            alpha_at_face = self.at_attn_face(alpha_at_face).reshape(bs, t, c, 2) # bs, t, c, 2
            alpha_at_face = alpha_at_face.softmax(dim=-1)
            fusion_face = in_word_face * alpha_at_face[:,:,:,1] + in_audio_face * alpha_at_face[:,:,:,0]
            alpha_at_body = torch.cat([in_word_body, in_audio_body], dim=-1).reshape(bs, t, c*2)
            alpha_at_body = self.at_attn_body(alpha_at_body).reshape(bs, t, c, 2)
            alpha_at_body = alpha_at_body.softmax(dim=-1)
            fusion_body = in_word_body * alpha_at_body[:,:,:,1] + in_audio_body * alpha_at_body[:,:,:,0]
        else:
            fusion_face = in_word_face + in_audio_face
            fusion_body = in_word_body + in_audio_body
        # print('fusion_face:', fusion_face.shape)
        # if is_train:
        #     fusion_face = self.fuse_dropout(fusion_face)
        #     fusion_body = self.fuse_dropout(fusion_body)
        masked_embeddings = self.mask_embeddings.expand_as(in_motion) # in_motion [bs, t, 337]
        # mask [bs, t, 337], 前四帧为0，后面为1
        masked_motion = torch.where(mask == 1, masked_embeddings, in_motion) 
        body_hint = self.motion_encoder(masked_motion) # bs t 256
        # print('id:', in_id.shape)
        
        speaker_embedding_face = self.spearker_encoder_face(in_id).squeeze(2) # bs, t, 768
        speaker_embedding_body = self.spearker_encoder_body(in_id).squeeze(2)

        # decode face
        use_body_hints = True
        if use_body_hints:
            body_hint_face = self.bodyhints_face(body_hint)
            fusion_face_a = torch.cat([fusion_face, body_hint_face], dim=2)
        a2g_face = self.feature2face(fusion_face_a)
        face_embeddings = speaker_embedding_face
        face_embeddings = self.position_embeddings(face_embeddings)
        decoded_face = self.face_decoder(tgt=face_embeddings, memory=a2g_face)
        face_latent = self.face2latent(decoded_face)
        face_latent = face_latent.permute(0, 2, 1)
        face_latent = self.face1d(face_latent)
        face_latent = face_latent.permute(0, 2, 1)
        face_latent = self.face_latent_mlp(face_latent)
        face_latent = face_latent.permute(0, 2, 1)
        face_latent = self.face1d_2(face_latent)

        face_latent_prezq = face_latent.permute(0, 2, 1)
        ######### hubert cons loss ###########
        hubert_cons_loss = self.hubert_face_cons_loss(face_latent_prezq, in_word_face)
        # ---- HuBERT 对比损失（对齐 + detach 音频端）----
        # in_word_face_for_loss = in_word_face.detach()
        # in_word_face_for_loss = match_time_to(in_word_face_for_loss, face_latent_prezq.shape[-1]).permute(0, 2, 1)  # -> (B,T,256)
        # hubert_cons_loss = self.hubert_face_cons_loss(face_latent_prezq.permute(0, 2, 1), in_word_face_for_loss)

        body_hint_body = self.bodyhints_body(body_hint) # MLP 256 to 256
        motion_embeddings = self.feature2motion(body_hint_body) # linear 256 to 768
        motion_embeddings = speaker_embedding_body + motion_embeddings # bs, t, 768
        motion_embeddings = self.position_embeddings(motion_embeddings) # bs, t, 768

        # bi-directional self-attention
        motion_refined_embeddings = self.motion_self_encoder(motion_embeddings) 
        
        # audio to gesture cross-modal attention
        if use_word:
            a2g_motion = self.audio_feature2motion(fusion_body) # linear 256 to 768
            motion_refined_embeddings_in = motion_refined_embeddings + speaker_embedding_body # bs, t, 768
            motion_refined_embeddings_in = self.position_embeddings(motion_refined_embeddings)
            word_hints = self.wordhints_decoder(tgt=motion_refined_embeddings_in, memory=a2g_motion)
            motion_refined_embeddings = motion_refined_embeddings + word_hints

        # Inject mass prior at decoder-latent stage (after audio fusion) with learnable scalar gate.
        motion_refined_embeddings = self._apply_base_mass_prior(motion_refined_embeddings)
        
        # feedforward
        motion_refined_embeddings = motion_refined_embeddings.permute(0, 2, 1)
        motion_refined_embeddings = self.body1d(motion_refined_embeddings)
        motion_refined_embeddings = motion_refined_embeddings.permute(0, 2, 1)
        speaker_embedding_body = speaker_embedding_body.permute(0, 2, 1)
        speaker_embedding_body = self.spearker_encoder_body1d(speaker_embedding_body)
        speaker_embedding_body = speaker_embedding_body.permute(0, 2, 1)

        upper_latent = self.motion2latent_upper(motion_refined_embeddings)
        hands_latent = self.motion2latent_hands(motion_refined_embeddings)
        lower_latent = self.motion2latent_lower(motion_refined_embeddings)

        upper_latent_in = upper_latent + speaker_embedding_body
        upper_latent_in = self.position_embeddings(upper_latent_in)
        hands_latent_in = hands_latent + speaker_embedding_body
        hands_latent_in = self.position_embeddings(hands_latent_in)
        lower_latent_in = lower_latent + speaker_embedding_body
        lower_latent_in = self.position_embeddings(lower_latent_in)

        # transformer decoder
        motion_upper = self.upper_decoder(tgt=upper_latent_in, memory=hands_latent+lower_latent)
        motion_hands = self.hands_decoder(tgt=hands_latent_in, memory=upper_latent+lower_latent)
        motion_lower = self.lower_decoder(tgt=lower_latent_in, memory=upper_latent+hands_latent)
        upper_latent = self.motion_down_upper(motion_upper+upper_latent) # linear 768 to 256
        hands_latent = self.motion_down_hands(motion_hands+hands_latent)
        lower_latent = self.motion_down_lower(motion_lower+lower_latent)

        upper_latent = upper_latent.permute(0, 2, 1)
        upper_latent = self.upper1d(upper_latent)
        upper_latent_prezq = upper_latent.permute(0, 2, 1)
       
        hands_latent = hands_latent.permute(0, 2, 1)
        hands_latent = self.hands1d(hands_latent)
        hands_latent_prezq = hands_latent.permute(0, 2, 1)

        ########## beat cons loss ###########
        beat_cons_loss = self.beat_cons_loss(hands_latent_prezq, in_word_body)
        # ---- Beat 对比损失（对齐 + detach 音频端）----
        # in_word_body_for_loss = in_word_body.detach()
        # in_word_body_for_loss = match_time_to(in_word_body_for_loss, hands_latent_prezq.shape[-1]).permute(0, 2, 1)
        # beat_cons_loss = self.beat_cons_loss(hands_latent_prezq.permute(0, 2, 1), in_word_body_for_loss)

        lower_latent = lower_latent.permute(0, 2, 1)
        lower_latent = self.lower1d(lower_latent)
        lower_latent_prezq = lower_latent.permute(0, 2, 1)

        hands_latent = self.hands_face_decoder(tgt=hands_latent_prezq, memory=face_latent_prezq)
        face_latent = self.face_hands_decoder(tgt=face_latent_prezq, memory=hands_latent_prezq)
        # face_latent = face_latent_prezq
        upper_latent = self.upper_hands_decoder(tgt = upper_latent_prezq, memory = hands_latent+lower_latent_prezq)
        lower_latent = self.lower_hands_decoder(tgt = lower_latent_prezq, memory = upper_latent_prezq+hands_latent)
        
        

        zq_index0_lower = self.lower_classifier(lower_latent)
        zq_index0_face = self.face_classifier(face_latent) # bs, t, 256

        zq1_face, zq2_face, zq3_face, zq4_face, zq5_face, zq_index1_face, zq_index2_face, zq_index3_face, zq_index4_face, zq_index5_face = self.predict_res_face(face_latent, fusion_face)
        # motion spatial encoder
        zq1_lower, zq2_lower, zq3_lower, zq4_lower, zq5_lower, zq_index1_lower, zq_index2_lower, zq_index3_lower, zq_index4_lower, zq_index5_lower = self.predict_res_lower(lower_latent, fusion_body)
        
        zq_index0_upper = self.upper_classifier(upper_latent)
        zq1_upper, zq2_upper, zq3_upper, zq4_upper, zq5_upper, zq_index1_upper, zq_index2_upper, zq_index3_upper, zq_index4_upper, zq_index5_upper = self.predict_res_upper(upper_latent, fusion_body)
        
        zq_index0_hands = self.hands_classifier(hands_latent)
        zq1_hands, zq2_hands, zq3_hands, zq4_hands, zq5_hands, zq_index1_hands, zq_index2_hands, zq_index3_hands, zq_index4_hands, zq_index5_hands = self.predict_res_hands(hands_latent, fusion_body)
        
        cls_face = torch.stack([zq_index0_face, zq_index1_face, zq_index2_face, zq_index3_face, zq_index4_face, zq_index5_face], dim=-1)
        cls_upper = torch.stack([zq_index0_upper, zq_index1_upper, zq_index2_upper, zq_index3_upper, zq_index4_upper, zq_index5_upper], dim=-1)
        cls_lower = torch.stack([zq_index0_lower, zq_index1_lower, zq_index2_lower, zq_index3_lower, zq_index4_lower, zq_index5_lower], dim=-1)
        cls_hands = torch.stack([zq_index0_hands, zq_index1_hands, zq_index2_hands, zq_index3_hands, zq_index4_hands, zq_index5_hands], dim=-1)

        rec_face = torch.stack([face_latent, zq1_face, zq2_face, zq3_face, zq4_face, zq5_face], dim=1).unsqueeze(2)
        rec_upper = torch.stack([upper_latent, zq1_upper, zq2_upper, zq3_upper, zq4_upper, zq5_upper], dim=1).unsqueeze(2)
        rec_lower = torch.stack([lower_latent, zq1_lower, zq2_lower, zq3_lower, zq4_lower, zq5_lower], dim=1).unsqueeze(2)
        rec_hands = torch.stack([hands_latent, zq1_hands, zq2_hands, zq3_hands, zq4_hands, zq5_hands], dim=1).unsqueeze(2)

        return {
            'hubert_cons_loss': hubert_cons_loss,
            'beat_cons_loss': beat_cons_loss,
            "rec_face":rec_face,
            "rec_upper":rec_upper,
            "rec_lower":rec_lower,
            "rec_hands":rec_hands,
            # "rec_face":face_latent,
            # "rec_upper":upper_latent,
            # "rec_lower":lower_latent,
            # "rec_hands":hands_latent,
            "cls_face":cls_face,
            "cls_upper":cls_upper,
            "cls_lower":cls_lower,
            "cls_hands":cls_hands,
            }
    
    def forward_latent(self, in_audio=None, in_word=None, mask=None, is_test=None, in_motion=None, use_attentions=True, use_word=True, in_id = None, hubert=None):
        in_word_face = self.hubert_encoder(hubert.permute(0, 2, 1)).permute(0, 2, 1)
        in_word_body = self.hubert_encoder_body(hubert.permute(0, 2, 1)).permute(0, 2, 1)
        # in_word_body = self.text_encoder_body(in_word_body)
        bs, t, c = in_word_face.shape
        in_audio_face = self.audio_pre_encoder_face(in_audio) # [bs, t, 256]
        in_audio_body = self.audio_pre_encoder_body(in_audio) # [bs, t, 256]
        

        if use_attentions:           
            alpha_at_face = torch.cat([in_word_face, in_audio_face], dim=-1).reshape(bs, t, c*2)
            alpha_at_face = self.at_attn_face(alpha_at_face).reshape(bs, t, c, 2) # bs, t, c, 2
            alpha_at_face = alpha_at_face.softmax(dim=-1)
            fusion_face = in_word_face * alpha_at_face[:,:,:,1] + in_audio_face * alpha_at_face[:,:,:,0]
            alpha_at_body = torch.cat([in_word_body, in_audio_body], dim=-1).reshape(bs, t, c*2)
            alpha_at_body = self.at_attn_body(alpha_at_body).reshape(bs, t, c, 2)
            alpha_at_body = alpha_at_body.softmax(dim=-1)
            fusion_body = in_word_body * alpha_at_body[:,:,:,1] + in_audio_body * alpha_at_body[:,:,:,0]
        else:
            fusion_face = in_word_face + in_audio_face
            fusion_body = in_word_body + in_audio_body
        # print('fusion_face:', fusion_face.shape)

        masked_embeddings = self.mask_embeddings.expand_as(in_motion) # in_motion [bs, t, 337]
        # mask [bs, t, 337], 前四帧为0，后面为1
        masked_motion = torch.where(mask == 1, masked_embeddings, in_motion) 
        body_hint = self.motion_encoder(masked_motion) # bs t 256
        # print('id:', in_id.shape)
        speaker_embedding_face = self.spearker_encoder_face(in_id).squeeze(2) # bs, t, 768
        speaker_embedding_body = self.spearker_encoder_body(in_id).squeeze(2)

        # decode face
        use_body_hints = True
        if use_body_hints:
            body_hint_face = self.bodyhints_face(body_hint)
            fusion_face_a = torch.cat([fusion_face, body_hint_face], dim=2)
        a2g_face = self.feature2face(fusion_face_a)
        face_embeddings = speaker_embedding_face
        face_embeddings = self.position_embeddings(face_embeddings)
        # print('face_embeddings:', face_embeddings.shape)
        decoded_face = self.face_decoder(tgt=face_embeddings, memory=a2g_face)
        face_latent = self.face2latent(decoded_face)
        face_latent = face_latent.permute(0, 2, 1)
        face_latent = self.face1d(face_latent)
        face_latent = face_latent.permute(0, 2, 1)
        face_latent = self.face_latent_mlp(face_latent)
        face_latent = face_latent.permute(0, 2, 1)
        face_latent = self.face1d_2(face_latent)
        face_latent_prezq = face_latent.permute(0, 2, 1)
        ######### hubert cons loss ###########
        hubert_cons_loss = self.hubert_face_cons_loss(face_latent_prezq, in_word_face)
       
        body_hint_body = self.bodyhints_body(body_hint) # MLP 256 to 256
        motion_embeddings = self.feature2motion(body_hint_body) # linear 256 to 768
        motion_embeddings = speaker_embedding_body + motion_embeddings # bs, t, 768
        motion_embeddings = self.position_embeddings(motion_embeddings) # bs, t, 768

        # bi-directional self-attention
        motion_refined_embeddings = self.motion_self_encoder(motion_embeddings) 
        
        # audio to gesture cross-modal attention
        if use_word:
            a2g_motion = self.audio_feature2motion(fusion_body) # linear 256 to 768
            motion_refined_embeddings_in = motion_refined_embeddings + speaker_embedding_body # bs, t, 768
            motion_refined_embeddings_in = self.position_embeddings(motion_refined_embeddings)
            word_hints = self.wordhints_decoder(tgt=motion_refined_embeddings_in, memory=a2g_motion)
            motion_refined_embeddings = motion_refined_embeddings + word_hints

        # Keep inference path aligned with training path.
        motion_refined_embeddings = self._apply_base_mass_prior(motion_refined_embeddings)
        
        # feedforward
        motion_refined_embeddings = motion_refined_embeddings.permute(0, 2, 1)
        motion_refined_embeddings = self.body1d(motion_refined_embeddings)
        motion_refined_embeddings = motion_refined_embeddings.permute(0, 2, 1)
        speaker_embedding_body = speaker_embedding_body.permute(0, 2, 1)
        speaker_embedding_body = self.spearker_encoder_body1d(speaker_embedding_body)
        speaker_embedding_body = speaker_embedding_body.permute(0, 2, 1)

        upper_latent = self.motion2latent_upper(motion_refined_embeddings)
        hands_latent = self.motion2latent_hands(motion_refined_embeddings)
        lower_latent = self.motion2latent_lower(motion_refined_embeddings)

        upper_latent_in = upper_latent + speaker_embedding_body
        upper_latent_in = self.position_embeddings(upper_latent_in)
        hands_latent_in = hands_latent + speaker_embedding_body
        hands_latent_in = self.position_embeddings(hands_latent_in)
        lower_latent_in = lower_latent + speaker_embedding_body
        lower_latent_in = self.position_embeddings(lower_latent_in)

        # transformer decoder
        motion_upper = self.upper_decoder(tgt=upper_latent_in, memory=hands_latent+lower_latent)
        motion_hands = self.hands_decoder(tgt=hands_latent_in, memory=upper_latent+lower_latent)
        motion_lower = self.lower_decoder(tgt=lower_latent_in, memory=upper_latent+hands_latent)
        upper_latent = self.motion_down_upper(motion_upper+upper_latent) # linear 768 to 256
        hands_latent = self.motion_down_hands(motion_hands+hands_latent)
        lower_latent = self.motion_down_lower(motion_lower+lower_latent)

        upper_latent = upper_latent.permute(0, 2, 1)
        upper_latent = self.upper1d(upper_latent)
        upper_latent_prezq = upper_latent.permute(0, 2, 1)
       
        hands_latent = hands_latent.permute(0, 2, 1)
        hands_latent = self.hands1d(hands_latent)
        hands_latent_prezq = hands_latent.permute(0, 2, 1)

        ########## beat cons loss ###########
        beat_cons_loss = self.beat_cons_loss(hands_latent_prezq, in_word_body)

    
        lower_latent = lower_latent.permute(0, 2, 1)
        lower_latent = self.lower1d(lower_latent)
        lower_latent_prezq = lower_latent.permute(0, 2, 1)

        hands_latent = self.hands_face_decoder(tgt=hands_latent_prezq, memory=face_latent_prezq)
        face_latent = self.face_hands_decoder(tgt=face_latent_prezq, memory=hands_latent_prezq)
        # face_latent = face_latent_prezq
        upper_latent = self.upper_hands_decoder(tgt = upper_latent_prezq, memory = hands_latent+lower_latent_prezq)
        lower_latent = self.lower_hands_decoder(tgt = lower_latent_prezq, memory = upper_latent_prezq+hands_latent)
        return {
            'face_latent':face_latent,
            'upper_latent':upper_latent,
            'lower_latent':lower_latent,
            'hands_latent':hands_latent,
            }

class duogesture_sparse(nn.Module):
    def __init__(self, args=None):
        super(duogesture_sparse, self).__init__()
        self.args = args   
        # Optional bridge from base latents to sparse latents.
        # When enabled, sparse can keep base rhythm on low-semantic frames
        # while still diverging on high-semantic frames.
        self._latent_bridge_enabled = bool(getattr(args, 'latent_bridge_enabled', False))
        self._latent_bridge_alpha = float(getattr(args, 'latent_bridge_alpha', 0.35))
        self._latent_bridge_alpha = max(0.0, min(1.0, self._latent_bridge_alpha))
        self._dual_stage_cond_enabled = bool(getattr(args, 'dual_stage_cond_enabled', True))
        self._early_cond_scale = float(getattr(args, 'early_cond_scale', 0.20))
        self._mid_cond_scale = float(getattr(args, 'mid_cond_scale', 0.15))

        # Weight-informed conditioning ablation.
        # none | arm_combined_core | split_arm_core | all_joints_core
        self._mass_cond_mode = str(getattr(args, 'mass_cond_mode', 'none')).strip().lower()
        if self._mass_cond_mode not in {
            'none', 'arm_combined_core', 'split_arm_core', 'all_joints_core'
        }:
            self._mass_cond_mode = 'none'
        self._mass_cond_scale = float(getattr(args, 'mass_cond_scale', 0.30))
        self._mass_cond_enabled = self._mass_cond_mode != 'none'

        mass_vec = self._build_mass_condition_vector(self._mass_cond_mode)
        self.register_buffer('mass_cond_vector', mass_vec)
        self.mass_cond_proj = nn.Sequential(
            nn.Linear(int(mass_vec.numel()), 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Linear(256, 256),
        )
        with open(f"{args.data_path}weights/vocab.pkl", 'rb') as f:
            self.lang_model = pickle.load(f)
            pre_trained_embedding = self.lang_model.word_embedding_weights
        self.hubert_encoder_body = nn.Sequential(*[
                nn.Conv1d(1024, 256, 3, 1, 1, bias=False),
                nn.BatchNorm1d(256),
                nn.GELU(),
                nn.Conv1d(256, 256, 3, 1, 1, bias=False)
            ])
        self.audio_pre_encoder_body = MLP(3, args.hidden_size, 256)
        self.at_attn_bert = nn.Linear(args.audio_f*2, args.audio_f*2)
        # ── S-VIB gate (replaces old nn.Linear(256, 2)) ──
        _vib_enabled = getattr(args, 'vib_enabled', True)
        self._vib_enabled = _vib_enabled
        self._gate_floor = float(getattr(args, 'gate_floor', 0.0))
        if _vib_enabled:
            _vib_z   = getattr(args, 'vib_z_dim', 16)
            _vib_t   = getattr(args, 'vib_timing_dim', 64)
            self.gate = SemanticVIB(sem_dim=256, timing_dim=_vib_t, z_dim=_vib_z)
        else:
            self.gate = nn.Linear(256, 2)
        ######## clip / moclip embedding #########
        clip_dim = getattr(args, 'clip_dim', 512)  # 512 for ViT-B/32, 768 for ViT-L/14
        self.clip_embedding = nn.Linear(clip_dim, 256)
        self.emotion_embedding = nn.Linear(clip_dim, 256)
        self.at_attn_face_semantic = nn.Linear(512, 512)
        self.at_attn_body_semantic = nn.Linear(512, 512)
        
        self.gate_face = nn.Linear(256, 1)
        self.gate_body = nn.Linear(256, 1)

        ######## constractive loss ########
        # self.hubert_cons_loss = RhythmicIdentificationLoss(temperature=0.1)
        self.beat_cons_loss = RhythmicIdentificationLoss(temperature=0.1)
        self.face_cons_mlp = MLP(256, args.hidden_size, 256)
        self.hands_cons_mlp = MLP(256, args.hidden_size, 256)
        

        ##### predict_residual #####
        # self.predict_res_face = predict_residual_zq(latent_dim=256,num_head = 8,ffn_dim = 1024, dropout = 0.1, n_tokens = 5)
        self.predict_res_hands = predict_residual_zq(latent_dim=256,num_head = 8,ffn_dim = 1024, dropout = 0.1, n_tokens = 5)
        self.predict_res_upper = predict_residual_zq(latent_dim=256,num_head = 8,ffn_dim = 1024, dropout = 0.1, n_tokens = 5)
        self.predict_res_lower = predict_residual_zq(latent_dim=256,num_head = 8,ffn_dim = 1024, dropout = 0.1, n_tokens = 5)

        ##### con1d #####
        self.body1d = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=2, stride=2)
        self.upper1d = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.hands1d = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.lower1d = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.spearker_encoder_body1d = nn.Conv1d(in_channels=768, out_channels=768, kernel_size=2, stride=2)
        self.text_pre_encoder_body = nn.Embedding.from_pretrained(torch.FloatTensor(pre_trained_embedding),freeze=args.t_fix_pre)
        self.text_encoder_body = nn.Linear(300, args.audio_f) # audio_f = 256
        self.text_encoder_body = nn.Linear(300, args.audio_f) # audio_f = 256
        args_top = copy.deepcopy(self.args)
        args_top.vae_layer = 3
        args_top.vae_length = args.motion_f  # args.motion_f 256
        args_top.vae_test_dim = args.pose_dims+3+4 # 330 + 3 + 4 = 337
        self.motion_encoder = VQEncoderV6(args_top) # masked motion to latent bs t 337 to bs t 256
        
        self.transformer_de_layer = nn.TransformerDecoderLayer(
            d_model=self.args.hidden_size,
            nhead=4,
            dim_feedforward=self.args.hidden_size*2,
            batch_first=True
            )
        self.transformer_de_fu_layer = nn.TransformerDecoderLayer(
            d_model=256,
            nhead=4,
            dim_feedforward=256*2,
            batch_first=True
            )
        
        self.semantic_body_decoder = nn.TransformerDecoder(self.transformer_de_fu_layer, num_layers=1)
        self.body_semantic_mlp = MLP(256, 256, 256)

        self.hands_face_decoder = nn.TransformerDecoder(self.transformer_de_fu_layer, num_layers=1)
        
        # pose_length = 64
        self.position_embeddings = PeriodicPositionalEncoding(self.args.hidden_size, period=self.args.pose_length, max_seq_len=self.args.pose_length)
        self.semantic_position_embeddings = PeriodicPositionalEncoding(256, period=self.args.pose_length, max_seq_len=self.args.pose_length)
        
        # motion decoder
        self.transformer_en_layer = nn.TransformerEncoderLayer(
            d_model=self.args.hidden_size,
            nhead=4,
            dim_feedforward=self.args.hidden_size*2,
            batch_first=True
            )
        self.motion_self_encoder = nn.TransformerEncoder(self.transformer_en_layer, num_layers=1)
        self.audio_feature2motion = nn.Linear(args.audio_f, args.hidden_size) # 256 to 768
        self.feature2motion = nn.Linear(args.motion_f, args.hidden_size) # 256 to 768

        self.bodyhints_body = MLP(args.motion_f, args.hidden_size, args.motion_f) # 256 to 256
        self.motion2latent_upper = MLP(args.hidden_size, args.hidden_size, self.args.hidden_size)
        self.motion2latent_hands = MLP(args.hidden_size, args.hidden_size, self.args.hidden_size)
        self.motion2latent_lower = MLP(args.hidden_size, args.hidden_size, self.args.hidden_size)
        self.wordhints_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=8)
        
        self.upper_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=1)
        self.hands_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=1)
        self.lower_decoder = nn.TransformerDecoder(self.transformer_de_layer, num_layers=1)

        self.upper_hands_decoder = nn.TransformerDecoder(self.transformer_de_fu_layer, num_layers=1)
        self.lower_hands_decoder = nn.TransformerDecoder(self.transformer_de_fu_layer, num_layers=1)

        self.upper_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)
        self.hands_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)
        self.lower_classifier = MLP(self.args.vae_codebook_size, args.hidden_size, self.args.vae_codebook_size)

        self.mask_embeddings = nn.Parameter(torch.zeros(1, 1, self.args.pose_dims+3+4)) # [1, 1, 337]
        self.motion_down_upper = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_hands = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_lower = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_upper = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_hands = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self.motion_down_lower = nn.Linear(args.hidden_size, self.args.vae_codebook_size)
        self._reset_parameters()


        self.spearker_encoder_body = nn.Embedding(25, args.hidden_size)

    def _reset_parameters(self):
        nn.init.normal_(self.mask_embeddings, 0, self.args.hidden_size ** -0.5)

    def _build_mass_condition_vector(self, mode):
        if mode == 'none':
            return torch.zeros(1, dtype=torch.float32)

        mass = torch.tensor(SMPLX_MASS_FRACTIONS_55[:55], dtype=torch.float32)
        mass = mass / mass.sum().clamp(min=1e-8)

        shoulder = mass[MASS_GROUPS['shoulder']].mean()
        forearm = mass[MASS_GROUPS['forearm']].mean()
        arm_combined = mass[MASS_GROUPS['arm_combined']].mean()
        core = mass[MASS_GROUPS['core']].mean()

        if mode == 'arm_combined_core':
            return torch.stack([arm_combined, core], dim=0)
        if mode == 'split_arm_core':
            return torch.stack([shoulder, forearm, core], dim=0)
        if mode == 'all_joints_core':
            return torch.cat([mass, core.unsqueeze(0)], dim=0)
        return torch.zeros(1, dtype=torch.float32)
    
    def forward(self, in_word=None, feat_clip_text=None, emotion=None, mask=None, is_test=None, epoch=121, use_attentions=True, use_word=True, in_id = None, in_motion=None,hubert=None,latent=None):

        in_bert_body = self.text_pre_encoder_body(in_word)
        in_bert_body = self.text_encoder_body(in_bert_body)

        in_word_body = self.hubert_encoder_body(hubert.permute(0, 2, 1)).permute(0, 2, 1)
        bs, t, c = in_bert_body.shape

        feat_clip_text = feat_clip_text.unsqueeze(1).repeat(1, t, 1)
        feat_clip_text_body = self.clip_embedding(feat_clip_text)

        
        emotion_clip_text = emotion.unsqueeze(1).repeat(1, t, 1)
        emotion_clip_text_body = self.emotion_embedding(emotion_clip_text)
        
        alpha_at_body_semantic = torch.cat([feat_clip_text_body, emotion_clip_text_body], dim=-1).reshape(bs, t, c*2)
        alpha_at_body_semantic = self.at_attn_body_semantic(alpha_at_body_semantic).reshape(bs, t, c, 2)
        alpha_at_body_semantic = alpha_at_body_semantic.softmax(dim=-1)
        fusion_body_semantic = feat_clip_text_body * alpha_at_body_semantic[:,:,:,1] + emotion_clip_text_body * alpha_at_body_semantic[:,:,:,0]
        bert_body_pre = self.semantic_position_embeddings(in_bert_body)

        body_semantic = self.semantic_body_decoder(tgt=bert_body_pre, memory=fusion_body_semantic)
        # ── Keep pure semantic for downstream (no fusion_bert leakage) ──
        body_semantic_pure = body_semantic  # [bs, t, 256] — unmixed

        # Add physics-inspired mass priors in embedding space for ablation.
        if self._mass_cond_enabled:
            mass_feat = self.mass_cond_vector.view(1, 1, -1).expand(bs, t, -1)
            mass_prior = self.mass_cond_proj(mass_feat)
            body_semantic_pure = body_semantic_pure + self._mass_cond_scale * mass_prior

        alpha_at_bert = torch.cat([body_semantic_pure, in_word_body], dim=-1).reshape(bs, t, c*2)
        alpha_at_bert = self.at_attn_bert(alpha_at_bert).reshape(bs, t, c, 2)
        alpha_at_bert = alpha_at_bert.softmax(dim=-1)
        fusion_bert = body_semantic_pure * alpha_at_bert[:,:,:,1] + in_word_body * alpha_at_bert[:,:,:,0]

        # ── S-VIB gate (two-stream) ──
        if self._vib_enabled:
            body_semantic_down = body_semantic_pure.reshape(bs, t//4, 4, c).mean(dim=2)  # [bs, t//4, 256]
            in_word_body_down  = in_word_body.reshape(bs, t//4, 4, c).mean(dim=2)        # [bs, t//4, 256]
            gate, gate_mu, gate_logvar = self.gate(body_semantic_down, in_word_body_down)
        else:
            fusion_bert_down = fusion_bert.reshape(bs, t//4, 4, c).mean(dim=2)
            gate = self.gate(fusion_bert_down)
            gate_mu = None
            gate_logvar = None

        # ── Use pure body_semantic through MLP (NOT fusion_bert) ──
        body_semantic = self.body_semantic_mlp(body_semantic_pure)
        gate_pred = torch.softmax(gate, dim=-1)

        body_gate = gate_pred[:,:,1].unsqueeze(2).repeat(1, 1, 4).reshape(bs, t, 1)
        if self._gate_floor > 0.0:
            # Soft floor: ψ_eff = floor + (1-floor)*ψ.  ψ=0 → floor (not zero),
            # so the decoder always receives some semantic signal.  This breaks
            # the ψ≡1 collapse caused by multiplicative gating.
            body_gate = self._gate_floor + (1.0 - self._gate_floor) * body_gate
        body_semantic = body_semantic * body_gate
        masked_embeddings = self.mask_embeddings.expand_as(in_motion) # in_motion [bs, t, 337]
        # mask [bs, t, 337], 前四帧为0，后面为1
        masked_motion = torch.where(mask == 1, masked_embeddings, in_motion) 
        body_hint = self.motion_encoder(masked_motion) # bs t 256
        speaker_embedding_body = self.spearker_encoder_body(in_id).squeeze(2)

        # decode face
        body_hint_body = self.bodyhints_body(body_hint) # MLP 256 to 256
        motion_embeddings = self.feature2motion(body_hint_body) # linear 256 to 768
        motion_embeddings = speaker_embedding_body + motion_embeddings # bs, t, 768
        motion_embeddings = self.position_embeddings(motion_embeddings) # bs, t, 768
        if self._dual_stage_cond_enabled:
            # Early conditioning: set trajectory manifold before motion self-encoding.
            early_prior = self.audio_feature2motion(body_semantic)
            motion_embeddings = motion_embeddings + self._early_cond_scale * early_prior

        # bi-directional self-attention
        motion_refined_embeddings = self.motion_self_encoder(motion_embeddings) 
        
        # audio to gesture cross-modal attention
        if use_word:
            a2g_motion = self.audio_feature2motion(body_semantic) # linear 256 to 768
            motion_refined_embeddings_in = motion_refined_embeddings + speaker_embedding_body # bs, t, 768
            motion_refined_embeddings_in = self.position_embeddings(motion_refined_embeddings_in)
            word_hints = self.wordhints_decoder(tgt=motion_refined_embeddings_in, memory=a2g_motion)
            motion_refined_embeddings = motion_refined_embeddings + word_hints
        
        # feedforward
        motion_refined_embeddings = motion_refined_embeddings.permute(0, 2, 1)
        motion_refined_embeddings = self.body1d(motion_refined_embeddings)
        motion_refined_embeddings = motion_refined_embeddings.permute(0, 2, 1)
        speaker_embedding_body = speaker_embedding_body.permute(0, 2, 1)
        speaker_embedding_body = self.spearker_encoder_body1d(speaker_embedding_body)
        speaker_embedding_body = speaker_embedding_body.permute(0, 2, 1)

        upper_latent = self.motion2latent_upper(motion_refined_embeddings)
        hands_latent = self.motion2latent_hands(motion_refined_embeddings)
        lower_latent = self.motion2latent_lower(motion_refined_embeddings)

        if self._dual_stage_cond_enabled:
            # Mid conditioning: inject semantic/body priors before cross-part decoding.
            t_mid = motion_refined_embeddings.shape[1]
            body_semantic_mid = F.adaptive_avg_pool1d(
                body_semantic.permute(0, 2, 1), t_mid
            ).permute(0, 2, 1)
            body_gate_mid = F.adaptive_avg_pool1d(
                body_gate.permute(0, 2, 1), t_mid
            ).permute(0, 2, 1)
            mid_prior = self.audio_feature2motion(body_semantic_mid) * body_gate_mid
            upper_latent = upper_latent + self._mid_cond_scale * mid_prior
            hands_latent = hands_latent + self._mid_cond_scale * mid_prior
            lower_latent = lower_latent + self._mid_cond_scale * mid_prior

        upper_latent_in = upper_latent + speaker_embedding_body
        upper_latent_in = self.position_embeddings(upper_latent_in)
        hands_latent_in = hands_latent + speaker_embedding_body
        hands_latent_in = self.position_embeddings(hands_latent_in)
        lower_latent_in = lower_latent + speaker_embedding_body
        lower_latent_in = self.position_embeddings(lower_latent_in)

        # transformer decoder
        motion_upper = self.upper_decoder(tgt=upper_latent_in, memory=hands_latent+lower_latent)
        motion_hands = self.hands_decoder(tgt=hands_latent_in, memory=upper_latent+lower_latent)
        motion_lower = self.lower_decoder(tgt=lower_latent_in, memory=upper_latent+hands_latent)
        upper_latent = self.motion_down_upper(motion_upper+upper_latent) # linear 768 to 256
        hands_latent = self.motion_down_hands(motion_hands+hands_latent)
        lower_latent = self.motion_down_lower(motion_lower+lower_latent)

        upper_latent = upper_latent.permute(0, 2, 1)
        upper_latent = self.upper1d(upper_latent)
        upper_latent_prezq = upper_latent.permute(0, 2, 1)
       
        hands_latent = hands_latent.permute(0, 2, 1)
        hands_latent = self.hands1d(hands_latent)
        hands_latent_prezq = hands_latent.permute(0, 2, 1)

        ########## beat cons loss ###########
        # beat_cons_loss = self.beat_cons_loss(hands_latent_prezq, fusion_body_semantic)
        # hands_latent_prezq = self.hands_cons_mlp(hands_latent_prezq)

        

        lower_latent = lower_latent.permute(0, 2, 1)
        lower_latent = self.lower1d(lower_latent)
        lower_latent_prezq = lower_latent.permute(0, 2, 1)

        hands_latent = self.hands_face_decoder(tgt=hands_latent_prezq, memory= upper_latent_prezq + lower_latent_prezq)
        upper_latent = self.upper_hands_decoder(tgt = upper_latent_prezq, memory = hands_latent+lower_latent_prezq)
        lower_latent = self.lower_hands_decoder(tgt = lower_latent_prezq, memory = upper_latent_prezq+hands_latent)

        body_gate = body_gate.reshape(bs, t//4, 4, 1)
        body_gate = body_gate.mean(dim=2)

        # Base-to-sparse latent bridge (bounded bold change).
        # mix = alpha * (1 - semantic_gate): strong base guidance on non-semantic chunks.
        if self._latent_bridge_enabled and latent is not None:
            mm_upper_latent = latent.get('upper_latent', None)
            mm_hands_latent = latent.get('hands_latent', None)
            mm_lower_latent = latent.get('lower_latent', None)
            if (
                mm_upper_latent is not None and mm_hands_latent is not None and mm_lower_latent is not None
                and mm_upper_latent.shape == upper_latent.shape
                and mm_hands_latent.shape == hands_latent.shape
                and mm_lower_latent.shape == lower_latent.shape
            ):
                bridge_mix = (1.0 - body_gate) * self._latent_bridge_alpha
                upper_latent = upper_latent * (1.0 - bridge_mix) + mm_upper_latent * bridge_mix
                hands_latent = hands_latent * (1.0 - bridge_mix) + mm_hands_latent * bridge_mix
                lower_latent = lower_latent * (1.0 - bridge_mix) + mm_lower_latent * bridge_mix


        zq_index0_lower = self.lower_classifier(lower_latent)
        
        # motion spatial encoder
        zq1_lower, zq2_lower, zq3_lower, zq4_lower, zq5_lower, zq_index1_lower, zq_index2_lower, zq_index3_lower, zq_index4_lower, zq_index5_lower = self.predict_res_lower(lower_latent, body_semantic)
        
        zq_index0_upper = self.upper_classifier(upper_latent)
        zq1_upper, zq2_upper, zq3_upper, zq4_upper, zq5_upper, zq_index1_upper, zq_index2_upper, zq_index3_upper, zq_index4_upper, zq_index5_upper = self.predict_res_upper(upper_latent, body_semantic)
        
        zq_index0_hands = self.hands_classifier(hands_latent)
        zq1_hands, zq2_hands, zq3_hands, zq4_hands, zq5_hands, zq_index1_hands, zq_index2_hands, zq_index3_hands, zq_index4_hands, zq_index5_hands = self.predict_res_hands(hands_latent, body_semantic)

        cls_upper = torch.stack([zq_index0_upper, zq_index1_upper, zq_index2_upper, zq_index3_upper, zq_index4_upper, zq_index5_upper], dim=-1)
        cls_lower = torch.stack([zq_index0_lower, zq_index1_lower, zq_index2_lower, zq_index3_lower, zq_index4_lower, zq_index5_lower], dim=-1)
        cls_hands = torch.stack([zq_index0_hands, zq_index1_hands, zq_index2_hands, zq_index3_hands, zq_index4_hands, zq_index5_hands], dim=-1)

        rec_upper = torch.stack([upper_latent, zq1_upper, zq2_upper, zq3_upper, zq4_upper, zq5_upper], dim=1).unsqueeze(2)
        rec_lower = torch.stack([lower_latent, zq1_lower, zq2_lower, zq3_lower, zq4_lower, zq5_lower], dim=1).unsqueeze(2)
        rec_hands = torch.stack([hands_latent, zq1_hands, zq2_hands, zq3_hands, zq4_hands, zq5_hands], dim=1).unsqueeze(2)
        return {
            "gate":gate,
            "gate_mu":gate_mu,
            "gate_logvar":gate_logvar,
            "rec_upper":rec_upper,
            "rec_lower":rec_lower,
            "rec_hands":rec_hands,
            "cls_upper":cls_upper,
            "cls_lower":cls_lower,
            "cls_hands":cls_hands,
            }

# ── Named aliases so configs can use DuoGesture names ────────────────────────
duogesture_base = duogesture_base
duogesture_sparse = duogesture_sparse
duogesture_svib_phys = duogesture_sparse

# Backward compatibility for existing checkpoints/configs.
duogesture_svib_phys = duogesture_sparse


if __name__ == "__main__":
    from utils import config
    args = config.parse_args()
    in_audio = torch.randn(2, 64, 3)
    mask = torch.ones(2, 64, 337).bool()
    in_word = torch.randint(0, 10, (2, 64))

    in_motion = torch.randn(2, 64, 337)
    in_id = torch.randint(0, 25, (2, 64, 1))
    hubert = torch.randn(2, 64, 1024)
    model = duogesture_base(args)
    out = model(in_audio, in_word, mask, in_motion=in_motion, in_id=in_id, hubert=hubert)
    print(out['rec_face'].shape)

    sparse_model = duogesture_sparse(args)
    clip_dim = getattr(args, 'clip_dim', 512)
    out_sparse = sparse_model(in_word=in_word, feat_clip_text=torch.randn(2, clip_dim), emotion=torch.randn(2, clip_dim), mask=mask, in_motion=in_motion, in_id=in_id, hubert=hubert)
    print(out_sparse['rec_upper'].shape)