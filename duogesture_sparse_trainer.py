import train
import os
import time
import csv
import sys
import warnings
import random
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
import numpy as np
import time
import pprint
from loguru import logger
from utils import rotation_conversions as rc
import smplx
from utils import config, logger_tools, other_tools, metric, data_transfer
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from dataloaders.data_tools import joints_list
import librosa
import pickle
import clip
from src.utils.moclip_utils import load_tmr_text_encoder, load_moclip_text_encoder
from models.physics_smoother import PhysicsSmootherSMPLX, UPPER_JOINT_IDX, HANDS_JOINT_IDX, LOWER_JOINT_IDX


def _resolve_legacy_checkpoint_path(path: str) -> str:
    if not path:
        return path
    if os.path.exists(path):
        return path
    if "duogesture" in path:
        legacy_path = path.replace("duogesture", "duogesture")
        if os.path.exists(legacy_path):
            logger.warning(f"Checkpoint {path} not found; falling back to legacy path {legacy_path}")
            return legacy_path
    return path

# ── Flow-Matching base model (optional) ──
try:
    from models.flow_matching_base import GestureLSMBaseMotion
    _FM_AVAILABLE = True
except ImportError:
    _FM_AVAILABLE = False

class CustomTrainer(train.BaseTrainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.now_epoch = 0
        self.best_fid = float("inf")
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]
       
        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        self.joints = 55
        for joint_name in self.tar_joint_list_face:
            self.joint_mask_face[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1

        self.tracker = other_tools.EpochTracker([ "reg","reg_s","reg_w","gate","gate_self","gate_word","sem","sem_self","sem_word", "hubert","beat","hubert_sem","beat_sem", "acc_face", "acc_hands", "acc_upper", "acc_lower", "fid", "l1div", "bc", "rec", "trans", "vel", "transv", 'dis', 'gen', 'acc', 'transa', 'exp', 'lvd', 'mse', "cls", "rec_face", "latent", "cls_full", "cls_self", "cls_word", "latent_word","latent_self", "kl_val","kl_self","kl_word","vib_beta","gate_mu_norm","phys_jerk","phys_beta","jerk_fused","gate_smooth","contrast","gate_mean"], [False,True,True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,False,False,False,  False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False,False, False,False,False,False,False,False,False,False,False,False,False])
        ##### vq_model #####
        vq_model_module = __import__(f"models.motion_representation", fromlist=["something"])
        rvq_model_module = __import__(f"models.rvq", fromlist=["something"])
        base_model_module = __import__(f"models.duogesture", fromlist=["something"])

        # ── Pluggable base-motion module ──
        # Set  use_flow_matching: true  in the YAML config to swap in the
        # GestureLSM Flow-Matching base instead of the original duogesture_base.
        self.use_flow_matching = bool(getattr(self.args, "use_flow_matching", False))

        if self.use_flow_matching:
            assert _FM_AVAILABLE, (
                "use_flow_matching=True but models/flow_matching_base.py could not be imported. "
                "Check that the file exists and has no import errors."
            )
            logger.info("[sparse_trainer] Using GestureLSMBaseMotion (Flow-Matching) as base module")
            self.duogesture_base = GestureLSMBaseMotion(self.args).to(self.rank)
            _fm_ckpt = getattr(self.args, "fm_base_ckpt", None) or getattr(self.args, "base_ckpt", None)
            if _fm_ckpt:
                try:
                    other_tools.load_checkpoints(self.duogesture_base, _fm_ckpt, 'GestureLSMBaseMotion')
                except Exception as e:
                    logger.warning(f"Failed to load FM base checkpoint: {e}. Starting with uninit FM model.")
        else:
            logger.info("[sparse_trainer] Using original duogesture_base as base module")
            self.duogesture_base = getattr(base_model_module, "duogesture_base")(self.args).to(self.rank)
            try:
                other_tools.load_checkpoints(self.duogesture_base, _resolve_legacy_checkpoint_path(self.args.base_ckpt), 'duogesture_base')
            except Exception as e:
                logger.warning(f"Failed to load base model checkpoint: {e}. Starting with an uninitialized model.")

        logger.info(
            f"[sparse_trainer] mass_cond_mode={getattr(self.args, 'mass_cond_mode', 'none')} "
            f"mass_cond_scale={getattr(self.args, 'mass_cond_scale', 0.30)}"
        )
        logger.info(
            f"[sparse_trainer] base_mass_cond_mode={getattr(self.args, 'base_mass_cond_mode', 'none')} "
            f"base_mass_cond_scale={getattr(self.args, 'base_mass_cond_scale', 0.30)} "
            f"joint_train_base_in_sparse={getattr(self.args, 'joint_train_base_in_sparse', False)}"
        )
        
        
        self.args.vae_layer = 2
        self.args.vae_length = 256
        self.args.vae_test_dim = 106 # face
        self.vq_model_face = getattr(rvq_model_module, "RVQVAE")(self.args).to(self.rank)

        other_tools.load_checkpoints(self.vq_model_face, "./weights/pretrained_vq/rvq_face_600.bin", args.e_name)

        self.args.vae_test_dim = 78 # upper body
        self.vq_model_upper = getattr(rvq_model_module, "RVQVAE")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_upper, "./weights/pretrained_vq/rvq_upper_500.bin", args.e_name)

        self.args.vae_test_dim = 180 # hands
        self.vq_model_hands = getattr(rvq_model_module, "RVQVAE")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_hands, "./weights/pretrained_vq/rvq_hands_500.bin", args.e_name)
        
        self.args.vae_test_dim = 61 # lower body
        self.args.vae_layer = 4
        self.vq_model_lower = getattr(rvq_model_module, "RVQVAE")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_lower, "./weights/pretrained_vq/rvq_lower_600.bin", args.e_name)

        self.args.vae_test_dim = 61 #global motion
        self.args.vae_layer = 4
        self.global_motion = getattr(vq_model_module, "VAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.global_motion, "./weights/pretrained_vq/last_1700_foot.bin", args.e_name)
        print(f"[rank {self.rank}] all VQ checkpoints loaded — GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)

        self.args.vae_test_dim = 330
        self.args.vae_layer = 4
        self.args.vae_length = 240
        self.vq_model_face.eval()
        self.vq_model_upper.eval()
        self.vq_model_hands.eval()
        self.vq_model_lower.eval()
        self.global_motion.eval()

        self._joint_train_base = bool(getattr(self.args, "joint_train_base_in_sparse", False))
        if self._joint_train_base:
            self.duogesture_base.train()
            for p in self.duogesture_base.parameters():
                p.requires_grad = True
            self.opt.add_param_group({"params": self.duogesture_base.parameters()})
            logger.info("[sparse_trainer] Base module is trainable and added to optimizer")
        else:
            self.duogesture_base.eval()
            for p in self.duogesture_base.parameters():
                p.requires_grad = False
            logger.info("[sparse_trainer] Base module frozen (eval)")

        self.cls_loss = nn.NLLLoss(reduction='none').to(self.rank)
        self.reclatent_loss = nn.MSELoss(reduction='none').to(self.rank)
        self.vel_loss = torch.nn.L1Loss(reduction='mean').to(self.rank)
        self.rec_loss = get_loss_func("GeodesicLoss").to(self.rank) 
        self.log_softmax = nn.LogSoftmax(dim=2).to(self.rank)
        self.reclatent_loss_test = nn.MSELoss().to(self.rank)
        # Gate CE loss — optionally class-weighted to counter 10:1 non-sem:sem imbalance
        gate_class_weight = getattr(args, 'gate_class_weight', 1.0)
        if gate_class_weight != 1.0:
            w = torch.tensor([1.0, float(gate_class_weight)]).to(self.rank)
            self.gate_fuc = nn.CrossEntropyLoss(weight=w).to(self.rank)
        else:
            self.gate_fuc = nn.CrossEntropyLoss().to(self.rank)

        # ── S-VIB KL hyper-parameters ──
        self._vib_enabled = getattr(args, 'vib_enabled', True)
        self._vib_beta_target  = getattr(args, 'vib_beta_target',  0.001)
        self._vib_warmup_start = getattr(args, 'vib_warmup_start', 20)
        self._vib_warmup_end   = getattr(args, 'vib_warmup_end',   100)
        self._vib_free_bits    = getattr(args, 'vib_free_bits',     0.5)

        # ── Latent loss gating mode ──
        # "gt_mask"   (default, legacy): multiply latent loss by GT sem_mean → zero on 90% of frames
        # "pred_gate": weight by detached predicted gate: sem_boost*ψ + (1-ψ), trains on ALL frames
        # "uniform":   no masking, latent loss on all frames equally
        self._latent_loss_mode = getattr(args, 'latent_loss_mode', 'gt_mask')
        self._sem_boost = getattr(args, 'sem_boost', 3.0)  # how much more weight on semantic frames

        # ── Contrastive margin loss ──
        self._contrast_enabled = getattr(args, 'contrast_enabled', False)
        self._contrast_weight  = getattr(args, 'contrast_weight', 0.1)
        self._contrast_margin  = getattr(args, 'contrast_margin', 0.5)

        # ── Physics Smoother (gate-modulated per-joint EMA) ──
        self._phys_enabled = getattr(args, 'phys_enabled', True)
        self._phys_upper_only = getattr(args, 'phys_upper_only', False)
        self._phys_train_enabled = getattr(args, 'phys_train_enabled', True)
        self._phys_lambda       = getattr(args, 'phys_lambda',       0.01)
        self._phys_hands_lambda = getattr(args, 'phys_hands_lambda', 1.0)
        self._phys_warmup_start = getattr(args, 'phys_warmup_start', 30)
        self._phys_warmup_end   = getattr(args, 'phys_warmup_end',   80)
        # Option 1: fused latent jerk (gate-blended sparse+base latent, no decoder pass)
        self._phys_fused_latent = getattr(args, 'phys_fused_latent', False)
        self._phys_fused_lambda = getattr(args, 'phys_fused_lambda', 0.10)
        # Gate-complementary physics: weight jerk by (1-ψ), gradient flows through ψ
        self._phys_gate_grad = getattr(args, 'phys_gate_grad', False)
        # Adaptive inertia: normalise residual by speed² (dynamic, motion-state-aware)
        self._phys_adaptive = getattr(args, 'phys_adaptive_inertia', False)
        # Option 3: gate temporal smoothing (penalises abrupt ψ transitions)
        self._gate_smooth_weight = getattr(args, 'gate_smooth_weight', 0.0)
        self._gate_mean_weight = getattr(args, 'gate_mean_weight', 0.0)
        if self._phys_enabled:
            self.physics_smoother = PhysicsSmootherSMPLX(
                num_joints=self.joints,
                tau_base=getattr(args, 'phys_tau_base', 0.50),
                tau_floor=getattr(args, 'phys_tau_floor', 0.10),
                alpha=getattr(args, 'phys_alpha', 1.0),
                pose_fps=self.args.pose_fps,
            ).to(self.rank)
            logger.info(
                f"[sparse_trainer] PhysicsSmootherSMPLX enabled  "
                f"(τ_base={self.physics_smoother.tau_base}, "
                f"τ_floor={self.physics_smoother.tau_floor}, "
                f"α={self.physics_smoother.alpha}, "
                f"λ={self._phys_lambda})"
            )

    # ── S-VIB helpers ────────────────────────────────────────────────
    def _compute_vib_beta(self, epoch):
        """Sigmoid warmup: 0 → beta_target over [warmup_start, warmup_end]."""
        if not self._vib_enabled:
            return 0.0
        if epoch < self._vib_warmup_start:
            return 0.0
        t = (epoch - self._vib_warmup_start) / max(self._vib_warmup_end - self._vib_warmup_start, 1)
        t = min(t, 1.0)
        # sigmoid-shaped ramp: 6*(t-0.5) gives ~0 at t=0 and ~1 at t=1
        import math
        sig = 1.0 / (1.0 + math.exp(-6.0 * (t - 0.5)))
        return self._vib_beta_target * sig

    @staticmethod
    def _compute_kl_loss(mu, logvar, free_bits=0.5):
        """Per-dimension KL divergence with free-bits clamp — paper Eq. (svib_kl).

        KL formula (per dim):  0.5 * (mu^2 + sigma^2 - log(sigma^2) - 1)

        The free-bits clamp sets a *floor* on the per-dimension KL:
          - Dimensions already below `free_bits` nats receive NO KL gradient.
            This prevents the optimiser from over-regularising dimensions that
            are already close to the prior (collapsed / uninformative dims).
          - Dimensions above the floor are penalised normally.
        Net effect: the bottleneck spends its capacity budget only on
        dimensions that actually carry information about the semantic flag,
        preventing the gate from collapsing to a constant.

        Args:
            mu:        [bs, T, z_dim]  posterior mean
            logvar:    [bs, T, z_dim]  posterior log-variance
            free_bits: float  minimum KL allowed per dimension (default 0.5 nats,
                              overridden by vib_free_bits=0.30 in m1 config)
        Returns:
            scalar mean KL (after clamping)
        """
        # KL per dimension: 0.5 * (mu^2 + var - logvar - 1)
        kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
        # Free-bits: clamp per-dim KL to at least `free_bits`
        kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        return kl_per_dim.mean()

    def _compute_phys_beta(self, epoch):
        """Sigmoid warmup for physics jerk loss weight: 0 → phys_lambda."""
        if not self._phys_enabled:
            return 0.0
        if epoch < self._phys_warmup_start:
            return 0.0
        t = (epoch - self._phys_warmup_start) / max(
            self._phys_warmup_end - self._phys_warmup_start, 1
        )
        t = min(t, 1.0)
        import math
        sig = 1.0 / (1.0 + math.exp(-6.0 * (t - 0.5)))
        return self._phys_lambda * sig

    def _compute_latent_weight(self, sem_mean, gate_class_pred):
        """Per-frame weight for the latent reconstruction loss — paper Eq. (lat_loss).

        The weight w_t scales the MSE between predicted and target latent codes:

            L_lat = (1/L) sum_t  w_t * ||Z^q_t - Z_hat_t||^2

        Three modes:

          gt_mask   (legacy DuoGesture default)
                    w_t = s_t  (binary ground-truth semantic flag)
                    Trains f_sem on ~10% of frames only; starves the semantic branch.

          pred_gate (used in DuoGesture / m1_full_tuned)
                    w_t = rho * sg[psi_t] + (1 - sg[psi_t])
                    where psi_t = predicted semantic probability, sg = stop-gradient.
                    rho = sem_boost (default 3.0).
                    - Non-semantic frames: w_t ≈ 1.0  (always trained)
                    - Semantic frames:     w_t ≈ rho  (boosted by 3x)
                    Trains on ALL frames; soft emphasis where the model itself
                    believes the frame is semantic. Gate uses its own predictions
                    as a curriculum without a hard dependency on the GT flag.

          uniform   w_t = 1  everywhere (ablation baseline)

        Args:
            sem_mean:        [B]     mean GT semantic flag per sample (used in gt_mask mode)
            gate_class_pred: [B, T', 2]  gate logits (softmax gives psi)
        Returns:
            weight tensor broadcastable to the latent loss shape
        """
        mode = self._latent_loss_mode
        if mode == 'gt_mask':
            return sem_mean.unsqueeze(1).unsqueeze(2).unsqueeze(-1)
        elif mode == 'pred_gate':
            psi = gate_class_pred[:, :, 1].detach()  # [B, T'] soft semantic prob
            w = self._sem_boost * psi + (1.0 - psi)  # semantic frames boosted, rest = 1
            return w.unsqueeze(1).unsqueeze(2).unsqueeze(-1)
        elif mode == 'uniform':
            return 1.0
        else:
            raise ValueError(f"Unknown latent_loss_mode: {mode}")

    def _compute_cls_weight(self, sem_mean, gate_class_pred):
        """Compute per-frame weight for classification loss.

        Same modes as _compute_latent_weight but returns [B, T'] shape.
        """
        mode = self._latent_loss_mode
        if mode == 'gt_mask':
            return sem_mean
        elif mode == 'pred_gate':
            psi = gate_class_pred[:, :, 1].detach()
            return self._sem_boost * psi + (1.0 - psi)
        elif mode == 'uniform':
            return 1.0
        else:
            raise ValueError(f"Unknown latent_loss_mode: {mode}")

    def inverse_selection(self, filtered_t, selection_array, n):
        original_shape_t = np.zeros((n, selection_array.size))
        selected_indices = np.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        selected_indices = torch.where(selection_array == 1)[0]
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
        return original_shape_t
    
    def _load_data(self, dict_data):
        loaded_data = self._move_to_device(dict_data, self.device)
        return loaded_data
    
    def _move_to_device(self, obj, device):
        if torch.is_tensor(obj):
            # Keep original dtype unless casting is explicitly required.
            return obj.to(device, non_blocking=True)
        if isinstance(obj, np.ndarray):
            # Skip object/text arrays.
            if obj.dtype == np.object_:
                return obj
            return torch.from_numpy(obj).to(device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move_to_device(v, device) for v in obj)
        return obj  # 其余类型原样返回
  
    def _g_training(self, loaded_data, use_adv, mode="train", epoch=0):
        vib_beta = self._compute_vib_beta(epoch)
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints
        # print('train data n:', n) 
        # ------ full generatation task ------ #
        mask_val = torch.ones(bs, n, self.args.pose_dims+3+4).float().cuda() # same shape as latent_all
        mask_val[:, :self.args.pre_frames, :] = 0.0  # first frames are seed poses (paper Sec. 5.2.1)
        latent_val = self.duogesture_base.forward_latent(loaded_data['beat'].cuda(), loaded_data['in_word'].cuda(), mask=mask_val, in_id = loaded_data['tar_id'].cuda(), in_motion = loaded_data['latent_all'].cuda(), use_attentions = True,hubert=loaded_data['hubert'].cuda())
        
        net_out_val  = self.model(
            loaded_data['in_word'], loaded_data['feat_clip_text'], loaded_data['emo_clip_text'], mask=mask_val,
            in_id = loaded_data['tar_id'], in_motion = loaded_data['latent_all'],
            use_attentions = True, hubert=loaded_data['hubert'], latent = latent_val, epoch=epoch)

        g_loss_final = 0
        
        loss_latent_lower = self.reclatent_loss(net_out_val["rec_lower"], loaded_data["zq_lower"])
        loss_latent_hands = self.reclatent_loss(net_out_val["rec_hands"], loaded_data["zq_hands"])
        loss_latent_upper = self.reclatent_loss(net_out_val["rec_upper"], loaded_data["zq_upper"])

        sem_mean = loaded_data['sem_mean'].cuda()
        sem_label = (sem_mean>0).long().cuda()
        sem_label = sem_label.reshape(-1)
        total_sem_label = sem_label.numel()
        gate_val = net_out_val["gate"]
        
        loss_gate_val = self.gate_fuc(gate_val.reshape(-1, 2), sem_label)
        # ── S-VIB KL for gate_val ──
        if self._vib_enabled:
            kl_val = self._compute_kl_loss(net_out_val["gate_mu"], net_out_val["gate_logvar"], self._vib_free_bits)
            loss_gate_val = loss_gate_val + vib_beta * kl_val
            self.tracker.update_meter("kl_val", "train", kl_val.item())
            self.tracker.update_meter("vib_beta", "train", vib_beta)
            # Diagnostic: ‖μ‖ tracks posterior collapse (→0 = collapsed)
            mu_norm = net_out_val["gate_mu"].detach().norm(dim=-1).mean().item()
            self.tracker.update_meter("gate_mu_norm", "train", mu_norm)
        self.tracker.update_meter("sem", "train", loss_gate_val.item())
        gate_class_pred_val = torch.softmax(gate_val, dim=-1)
        gate_class_1 = torch.argmax(gate_class_pred_val, dim=-1)
        gate_class_val = gate_class_1.reshape(-1)

        correct_gate_val = (gate_class_val == sem_label).sum().item()
        acc_gate_val = correct_gate_val / total_sem_label
        self.tracker.update_meter("gate", "train", acc_gate_val)
        gate_expanded = self._compute_latent_weight(sem_mean, gate_class_pred_val)
        cls_weight = self._compute_cls_weight(sem_mean, gate_class_pred_val)
        
        loss_latent_lower = loss_latent_lower * gate_expanded
        loss_latent_hands = loss_latent_hands * gate_expanded
        loss_latent_upper = loss_latent_upper * gate_expanded
        loss_latent_lower = loss_latent_lower.mean()
        loss_latent_hands = loss_latent_hands.mean()
        loss_latent_upper = loss_latent_upper.mean()
        # if epoch > -1:
        #     gate_expanded_val_detach = gate_expanded_val.detach()
        #     loss_latent_lower = loss_latent_lower *(2*gate_expanded_val_detach + (1-gate_expanded_val_detach))
        #     loss_latent_hands = loss_latent_hands * (2*gate_expanded_val_detach + (1-gate_expanded_val_detach))
        #     loss_latent_upper = loss_latent_upper * (2*gate_expanded_val_detach + (1-gate_expanded_val_detach))
        # loss_latent_lower = loss_latent_lower.mean()
        # loss_latent_hands = loss_latent_hands.mean()
        # loss_latent_upper = loss_latent_upper.mean()

        loss_latent = self.args.ll*loss_latent_lower + self.args.lh*loss_latent_hands + self.args.lu*loss_latent_upper
        self.tracker.update_meter("latent", "train", loss_latent.item())
        g_loss_final += loss_latent/6

        # ── Contrastive margin loss: push sparse ≠ base on semantic frames ──
        if self._contrast_enabled:
            # Sparse 0th-level continuous latents  [B, T', C]
            sp_upper = net_out_val["rec_upper"][:, 0, 0]
            sp_hands = net_out_val["rec_hands"][:, 0, 0]
            sp_lower = net_out_val["rec_lower"][:, 0, 0]
            # Base latents (detached — we don't back-prop into the frozen base model)
            ba_upper = latent_val['upper_latent'].detach()
            ba_hands = latent_val['hands_latent'].detach()
            ba_lower = latent_val['lower_latent'].detach()
            # Per-timestep L2 distance between sparse and base
            delta = (
                (sp_upper - ba_upper).pow(2).mean(dim=-1)
                + (sp_hands - ba_hands).pow(2).mean(dim=-1)
                + (sp_lower - ba_lower).pow(2).mean(dim=-1)
            ) / 3.0  # [B, T']
            sem_t = (sem_mean > 0).float()  # [B, T']
            margin = self._contrast_margin
            loss_contrast = (
                sem_t * torch.relu(margin - delta)       # semantic: push apart
                + (1.0 - sem_t) * delta                  # non-semantic: stay close
            ).mean()
            g_loss_final = g_loss_final + self._contrast_weight * loss_contrast
            self.tracker.update_meter("contrast", "train", loss_contrast.item())

        # ── Physics jerk loss on decoded rot6d poses (P2) ──
        # Gradient path: sparse_model → upper_latent → frozen_VQ_decoder → decoded_rot6d → jerk_loss
        # Frozen decoder params are not updated, but ∂loss/∂upper_latent flows correctly.
        phys_beta = self._compute_phys_beta(epoch)
        if self._phys_enabled and self._phys_train_enabled and phys_beta > 0:
            # 0th-level continuous latent predictions from sparse model  [B, T', C_latent]
            upper_lat = net_out_val["rec_upper"][:, 0, 0]   # [B, T', C]
            hands_lat = net_out_val["rec_hands"][:, 0, 0]   # [B, T', C]
            lower_lat = net_out_val["rec_lower"][:, 0, 0]   # [B, T', C]

            # Decode through frozen VQ decoders — gradient flows to upper/hands/lower_lat
            # Decoder.forward() takes [B, C, T'] channels-first, then internally does
            # x.permute(0,2,1), so it already returns [B, T_dec, pose_dim] — no extra permute needed.
            dec_upper = self.vq_model_upper.decoder(upper_lat.permute(0,2,1))  # [B, T_dec, 78]
            dec_hands = self.vq_model_hands.decoder(hands_lat.permute(0,2,1))  # [B, T_dec, 180]
            dec_lower = self.vq_model_lower.decoder(lower_lat.permute(0,2,1))  # [B, T_dec, 61]

            B_d, T_dec, _ = dec_upper.shape

            # Determine joint counts from decoder output dynamically and validate
            pose_dim_upper = dec_upper.shape[2]
            if pose_dim_upper % 6 != 0:
                logger.error("upper decoder output dim %d not divisible by 6", pose_dim_upper)
                raise RuntimeError(f"upper decoder output dim {pose_dim_upper} not divisible by 6")
            n_joints_upper = pose_dim_upper // 6
            dec_upper_6d = dec_upper.reshape(B_d, T_dec, n_joints_upper, 6)

            pose_dim_hands = dec_hands.shape[2]
            if pose_dim_hands % 6 != 0:
                logger.error("hands decoder output dim %d not divisible by 6", pose_dim_hands)
                raise RuntimeError(f"hands decoder output dim {pose_dim_hands} not divisible by 6")
            n_joints_hands = pose_dim_hands // 6
            dec_hands_6d = dec_hands.reshape(B_d, T_dec, n_joints_hands, 6)

            # Lower decoder includes extra translation/contact dims (61 total); take exactly
            # len(LOWER_JOINT_IDX) full 6d joints (= 9 × 6 = 54 dims), matching tau_lower.
            n_joints_lower = len(LOWER_JOINT_IDX)   # 9
            dec_lower_6d = dec_lower[:, :, :(n_joints_lower * 6)].reshape(B_d, T_dec, n_joints_lower, 6)

            if self._phys_gate_grad:
                # ── Gate-complementary physics (Experiment C) ─────────────────
                # Weight = (1 − ψ): jerk penalty is strong on beat frames (ψ↓)
                # and weak on semantic frames (ψ↑).  Gradient flows through ψ so
                # the gate learns to lower ψ on frames where semantic would be jerky.
                gate_psi_w = gate_class_pred_val[:, :, 1:2]          # [B, T', 1] — NOT detached
                beat_w_tp = 1.0 - gate_psi_w                          # [B, T', 1]
                if T_dec != beat_w_tp.shape[1]:
                    beat_w_td = torch.nn.functional.interpolate(
                        beat_w_tp.permute(0, 2, 1), size=T_dec, mode='nearest'
                    ).permute(0, 2, 1)                                 # [B, T_dec, 1]
                else:
                    beat_w_td = beat_w_tp                              # [B, T', 1]
                tau_upper = beat_w_td.expand(-1, -1, n_joints_upper)  # [B, T_dec, 13]
                tau_hands = beat_w_td.expand(-1, -1, n_joints_hands)  # [B, T_dec, 30]
                tau_lower = beat_w_td.expand(-1, -1, n_joints_lower)  # [B, T_dec, 9]
            else:
                # ── Original τ-weighted physics (detached gate) ───────────────
                # Gate ψ — detached so physics loss does NOT pull gate toward semantic
                gate_psi_det = gate_class_pred_val[:, :, 1:2].detach()   # [B, T', 1]
                logvar_det = net_out_val["gate_logvar"].detach() if self._vib_enabled else None

                # τ for all 55 joints at T' resolution  →  [B, T', 55]
                tau_all = self.physics_smoother.compute_tau(gate_psi_det, logvar_det)

                # Select per-part τ and upsample from T' → T_dec via nearest-neighbour
                tau_upper_tp = tau_all[:, :, UPPER_JOINT_IDX]  # [B, T', 13]
                tau_hands_tp = tau_all[:, :, HANDS_JOINT_IDX]  # [B, T', 30]
                tau_lower_tp = tau_all[:, :, LOWER_JOINT_IDX]  # [B, T', 9]

                if T_dec != tau_all.shape[1]:
                    # F.interpolate expects [B, C, T]; we have [B, T', J]
                    tau_upper = torch.nn.functional.interpolate(
                        tau_upper_tp.permute(0,2,1), size=T_dec, mode='nearest').permute(0,2,1)  # [B, T_dec, 13]
                    tau_hands = torch.nn.functional.interpolate(
                        tau_hands_tp.permute(0,2,1), size=T_dec, mode='nearest').permute(0,2,1)  # [B, T_dec, 30]
                    tau_lower = torch.nn.functional.interpolate(
                        tau_lower_tp.permute(0,2,1), size=T_dec, mode='nearest').permute(0,2,1)  # [B, T_dec, 9]
                else:
                    tau_upper, tau_hands, tau_lower = tau_upper_tp, tau_hands_tp, tau_lower_tp

            # Inertia loss — two modes:
            #   static:   ||θ_t − θ̂_t||²  (raw CV residual)
            #   adaptive: ||θ_t − θ̂_t||² / (||v_{t-1}||² + ε)  (normalised by speed)
            # Both weighted by τ (beat frames penalised more via phys_gate_grad).
            _inertia_fn = (self.physics_smoother.compute_adaptive_inertia_loss
                           if self._phys_adaptive else
                           self.physics_smoother.compute_pose_inertia_residual_loss)
            phys_jerk = _inertia_fn(dec_upper_6d, tau_upper)
            if not self._phys_upper_only:
                phys_jerk_hands = _inertia_fn(dec_hands_6d, tau_hands)
                phys_jerk = phys_jerk + (
                    self._phys_hands_lambda * phys_jerk_hands
                    + _inertia_fn(dec_lower_6d, tau_lower)
                )
            g_loss_final = g_loss_final + phys_beta * phys_jerk
            self.tracker.update_meter("phys_jerk", "train", phys_jerk.item())
            self.tracker.update_meter("phys_beta", "train", phys_beta)

        # ── Option 1: Fused latent jerk ──────────────────────────────────────
        # Penalises jerk of (ψ·sparse + (1−ψ)·base) latent — covers ALL frames.
        # No decoder pass needed; gradient flows through sparse prediction + gate.
        # Base latent is detached so we only train the sparse model.
        phys_beta = self._compute_phys_beta(epoch)  # reuse same warmup schedule
        if self._phys_fused_latent and phys_beta > 0:
            sp_f_upper = net_out_val["rec_upper"][:, 0, 0]           # [B, T', C]
            sp_f_hands = net_out_val["rec_hands"][:, 0, 0]
            sp_f_lower = net_out_val["rec_lower"][:, 0, 0]
            ba_f_upper = latent_val['upper_latent'].detach()          # [B, T', C]
            ba_f_hands = latent_val['hands_latent'].detach()
            ba_f_lower = latent_val['lower_latent'].detach()
            gate_psi = gate_class_pred_val[:, :, 1:2]                 # [B, T', 1] — NOT detached; gradient into gate too
            fused_upper = gate_psi * sp_f_upper + (1.0 - gate_psi) * ba_f_upper
            fused_hands = gate_psi * sp_f_hands + (1.0 - gate_psi) * ba_f_hands
            fused_lower = gate_psi * sp_f_lower + (1.0 - gate_psi) * ba_f_lower
            # 2nd finite difference ≈ acceleration; mean over C and B
            def _latent_jerk(x):
                return (x[:, 2:] - 2.0 * x[:, 1:-1] + x[:, :-2]).pow(2).mean()
            jerk_fused = (_latent_jerk(fused_upper) + _latent_jerk(fused_hands) + _latent_jerk(fused_lower)) / 3.0
            g_loss_final = g_loss_final + phys_beta * self._phys_fused_lambda * jerk_fused
            self.tracker.update_meter("jerk_fused", "train", jerk_fused.item())

        # ── Option 3: Gate temporal smoothing ────────────────────────────────
        # Penalises abrupt ψ transitions between frames: avoids motion pops at
        # beat→semantic→beat boundaries.  Gradient flows through gate directly.
        if self._gate_smooth_weight > 0.0:
            gate_psi_seq = gate_class_pred_val[:, :, 1]               # [B, T']
            gate_smooth = (gate_psi_seq[:, 1:] - gate_psi_seq[:, :-1]).pow(2).mean()
            g_loss_final = g_loss_final + self._gate_smooth_weight * gate_smooth
            self.tracker.update_meter("gate_smooth", "train", gate_smooth.item())

        # ── Gate mean penalty ─────────────────────────────────────────────────
        # Directly penalises mean(ψ): always-on gate (ψ≈1 everywhere) incurs a
        # large constant cost, forcing selective activation on semantic frames.
        # Gradient flows through softmax → gate predictor weights.
        if self._gate_mean_weight > 0.0:
            gate_mean_val = gate_class_pred_val[:, :, 1].mean()
            g_loss_final = g_loss_final + self._gate_mean_weight * gate_mean_val
            self.tracker.update_meter("gate_mean", "train", gate_mean_val.item())

        self.now_epoch += 1
       
        loss_cls = 0
        
        tar_index_value_upper_top = loaded_data["tar_index_value_upper_top"]
        tar_index_value_lower_top = loaded_data["tar_index_value_lower_top"]
        tar_index_value_hands_top = loaded_data["tar_index_value_hands_top"]
       
        for i in range(6):
            rec_index_upper_val = self.log_softmax(net_out_val["cls_upper"][:,:,:,i])
            rec_index_lower_val = self.log_softmax(net_out_val["cls_lower"][:,:,:,i])
            rec_index_hands_val = self.log_softmax(net_out_val["cls_hands"][:,:,:,i])
            loss_cls_i_upper = ((self.args.cu*self.cls_loss(rec_index_upper_val.transpose(1, 2), tar_index_value_upper_top[:, :, i]))*cls_weight).mean()
            loss_cls_i_lower =  ((self.args.cl*self.cls_loss(rec_index_lower_val.transpose(1, 2), tar_index_value_lower_top[:, :, i]))*cls_weight).mean()
            loss_cls_i_hands = ((self.args.ch*self.cls_loss(rec_index_hands_val.transpose(1, 2), tar_index_value_hands_top[:, :,i]))*cls_weight).mean()
            # if epoch > -1:
            #     loss_cls_i_upper = ((self.args.cu*self.cls_loss(rec_index_upper_val.transpose(1, 2), tar_index_value_upper_top[:, :, i]))*gate_class_1).mean()
            #     loss_cls_i_lower =  ((self.args.cl*self.cls_loss(rec_index_lower_val.transpose(1, 2), tar_index_value_lower_top[:, :, i]))*gate_class_1).mean()
            #     loss_cls_i_hands = ((self.args.ch*self.cls_loss(rec_index_hands_val.transpose(1, 2), tar_index_value_hands_top[:, :,i]))*gate_class_1).mean()
            #     # gate_class_1_detach = gate_class_1.detach()
            #     # loss_cls_i_upper = ((self.args.cu*self.cls_loss(rec_index_upper_val.transpose(1, 2), tar_index_value_upper_top[:, :, i]))*(2*gate_class_1_detach + (1-gate_class_1_detach))).mean()
            #     # loss_cls_i_lower =  ((self.args.cl*self.cls_loss(rec_index_lower_val.transpose(1, 2), tar_index_value_lower_top[:, :, i]))*(2*gate_class_1_detach + (1-gate_class_1_detach))).mean()
            #     # loss_cls_i_hands = ((self.args.ch*self.cls_loss(rec_index_hands_val.transpose(1, 2), tar_index_value_hands_top[:, :,i]))*(2*gate_class_1_detach + (1-gate_class_1_detach))).mean()
            # else:
            #     loss_cls_i_upper = ((self.args.cu*self.cls_loss(rec_index_upper_val.transpose(1, 2), tar_index_value_upper_top[:, :, i]))).mean()
            #     loss_cls_i_lower =  ((self.args.cl*self.cls_loss(rec_index_lower_val.transpose(1, 2), tar_index_value_lower_top[:, :, i]))).mean()
            #     loss_cls_i_hands = ((self.args.ch*self.cls_loss(rec_index_hands_val.transpose(1, 2), tar_index_value_hands_top[:, :,i]))).mean()

            loss_cls_i = loss_cls_i_upper + loss_cls_i_lower + loss_cls_i_hands
            loss_cls = loss_cls + loss_cls_i/(i+1)

        self.tracker.update_meter("cls_full", "train", loss_cls.item())
        g_loss_final += loss_cls 
        
        if mode == 'train':
        #     # ------ masked gesture moderling------ #
            if epoch < 130:
                mask_ratio = (epoch / 400) * 0.95 + 0.05
            else:
                mask_ratio = 0.35875
            
            mask = torch.rand(bs, n, self.args.pose_dims+3+4) < mask_ratio
            mask = mask.float().cuda()
            
            latent_self = self.duogesture_base.forward_latent(loaded_data['beat'].cuda(), loaded_data['in_word'].cuda(), mask=mask, in_id = loaded_data['tar_id'].cuda(), in_motion = loaded_data['latent_all'].cuda(), use_attentions = True,use_word=False,hubert=loaded_data['hubert'].cuda())

            net_out_self  = self.model(
                loaded_data['in_word'], loaded_data['feat_clip_text'], loaded_data['emo_clip_text'], mask=mask,
                in_id = loaded_data['tar_id'], in_motion = loaded_data['latent_all'],
                use_attentions = True, use_word=False,hubert=loaded_data['hubert'],latent = latent_self,epoch=epoch)
            gate_self = net_out_self["gate"]
            loss_gate_self = self.gate_fuc(gate_self.reshape(-1, 2), sem_label)
            # ── S-VIB KL for gate_self ──
            if self._vib_enabled:
                kl_self = self._compute_kl_loss(net_out_self["gate_mu"], net_out_self["gate_logvar"], self._vib_free_bits)
                loss_gate_self = loss_gate_self + vib_beta * kl_self
                self.tracker.update_meter("kl_self", "train", kl_self.item())
            self.tracker.update_meter("sem_self", "train", loss_gate_self.item())
            gate_class_self_pre = torch.softmax(gate_self, dim=-1)
            gate_class_2 = torch.argmax(gate_class_self_pre, dim=-1)

            gate_class_self = gate_class_2.reshape(-1)
            correct_gate_self = (gate_class_self == sem_label).sum().item()
            acc_gate_self = correct_gate_self / total_sem_label
            self.tracker.update_meter("gate_self", "train", acc_gate_self)
            
            g_loss_final += loss_gate_self
            loss_latent_lower_self = self.reclatent_loss(net_out_self["rec_lower"], loaded_data["zq_lower"])
            loss_latent_hands_self = self.reclatent_loss(net_out_self["rec_hands"], loaded_data["zq_hands"])
            loss_latent_upper_self = self.reclatent_loss(net_out_self["rec_upper"], loaded_data["zq_upper"])
            # gate_expanded_self = gate_class_2.unsqueeze(1).unsqueeze(2).unsqueeze(-1)
            # if epoch > -1:
            #     gate_expanded_self_detach = gate_expanded_self.detach()
            #     loss_latent_lower_self = loss_latent_lower_self * (2*gate_expanded_self_detach+(1-gate_expanded_self_detach))
            #     loss_latent_hands_self = loss_latent_hands_self * (2*gate_expanded_self_detach+(1-gate_expanded_self_detach))
            #     loss_latent_upper_self = loss_latent_upper_self * (2*gate_expanded_self_detach+(1-gate_expanded_self_detach))
            gate_expanded_self = self._compute_latent_weight(sem_mean, gate_class_self_pre)
            loss_latent_lower_self = loss_latent_lower_self * gate_expanded_self
            loss_latent_hands_self = loss_latent_hands_self * gate_expanded_self
            loss_latent_upper_self = loss_latent_upper_self * gate_expanded_self
            loss_latent_lower_self = loss_latent_lower_self.mean()
            loss_latent_hands_self = loss_latent_hands_self.mean()
            loss_latent_upper_self = loss_latent_upper_self.mean()
            loss_latent_self = self.args.ll*loss_latent_lower_self + self.args.lh*loss_latent_hands_self + self.args.lu*loss_latent_upper_self
            self.tracker.update_meter("latent_self", "train", loss_latent_self.item())
            g_loss_final += loss_latent_self/6
            index_loss_top_self = 0 
            for i in range(6):
                rec_index_upper_self = self.log_softmax(net_out_self["cls_upper"][:,:,:,i])
                rec_index_lower_self = self.log_softmax(net_out_self["cls_lower"][:,:,:,i])
                rec_index_hands_self = self.log_softmax(net_out_self["cls_hands"][:,:,:,i])
                
                cls_weight_self = self._compute_cls_weight(sem_mean, gate_class_self_pre)
                index_loss_top_self_i_upper = ((self.args.cu*self.cls_loss(rec_index_upper_self.transpose(1, 2), tar_index_value_upper_top[:,:,i]))*cls_weight_self).mean()
                index_loss_top_self_i_lower =  ((self.args.cl*self.cls_loss(rec_index_lower_self.transpose(1, 2), tar_index_value_lower_top[:,:,i]))*cls_weight_self).mean()
                index_loss_top_self_i_hands = ((self.args.ch*self.cls_loss(rec_index_hands_self.transpose(1, 2), tar_index_value_hands_top[:,:,i]))*cls_weight_self).mean()
                # if epoch > -1:
                #     index_loss_top_self_i_upper = ((self.args.cu*self.cls_loss(rec_index_upper_self.transpose(1, 2), tar_index_value_upper_top[:,:,i]))*gate_class_2).mean()
                #     index_loss_top_self_i_lower =  ((self.args.cl*self.cls_loss(rec_index_lower_self.transpose(1, 2), tar_index_value_lower_top[:,:,i]))*gate_class_2).mean()
                #     index_loss_top_self_i_hands = ((self.args.ch*self.cls_loss(rec_index_hands_self.transpose(1, 2), tar_index_value_hands_top[:,:,i]))*gate_class_2).mean()
                #     # gate_class_2_detach = gate_class_2.detach()
                #     # index_loss_top_self_i_upper = ((self.args.cu*self.cls_loss(rec_index_upper_self.transpose(1, 2), tar_index_value_upper_top[:,:,i])) * (2*gate_class_2_detach + (1-gate_class_2_detach))).mean()
                #     # index_loss_top_self_i_lower =  ((self.args.cl*self.cls_loss(rec_index_lower_self.transpose(1, 2), tar_index_value_lower_top[:,:,i])) * (2*gate_class_2_detach + (1-gate_class_2_detach))).mean()
                #     # index_loss_top_self_i_hands = ((self.args.ch*self.cls_loss(rec_index_hands_self.transpose(1, 2), tar_index_value_hands_top[:,:,i])) * (2*gate_class_2_detach + (1-gate_class_2_detach))).mean()
                # else:
                #     index_loss_top_self_i_upper = ((self.args.cu*self.cls_loss(rec_index_upper_self.transpose(1, 2), tar_index_value_upper_top[:,:,i])).mean())
                #     index_loss_top_self_i_lower =  ((self.args.cl*self.cls_loss(rec_index_lower_self.transpose(1, 2), tar_index_value_lower_top[:,:,i])).mean())
                #     index_loss_top_self_i_hands = ((self.args.ch*self.cls_loss(rec_index_hands_self.transpose(1, 2), tar_index_value_hands_top[:,:,i])).mean())
                # loss_cls = loss_cls + loss_cls_i/(6-i)
                # loss_cls = loss_cls + loss_cls_i/6
                index_loss_top_self_i = index_loss_top_self_i_upper + index_loss_top_self_i_lower + index_loss_top_self_i_hands
                index_loss_top_self = index_loss_top_self + index_loss_top_self_i/(i+1)

            self.tracker.update_meter("cls_self", "train", index_loss_top_self.item())
            g_loss_final += index_loss_top_self
            
            # ------ masked audio gesture moderling ------ #
            
            latent_word = self.duogesture_base.forward_latent(loaded_data['beat'].cuda(), loaded_data['in_word'].cuda(), mask=mask_val, in_id = loaded_data['tar_id'].cuda(), in_motion = loaded_data['latent_all'].cuda(), use_word=True, use_attentions = True,hubert=loaded_data['hubert'].cuda())

            net_out_word  = self.model(
                loaded_data['in_word'], loaded_data['feat_clip_text'], loaded_data['emo_clip_text'], mask=mask,
                in_id = loaded_data['tar_id'], in_motion = loaded_data['latent_all'],
                use_attentions = True, use_word=True,hubert=loaded_data['hubert'],latent = latent_word,epoch=epoch)
            gate_word = net_out_word["gate"]
            loss_gate_word = self.gate_fuc(gate_word.reshape(-1, 2), sem_label)
            # ── S-VIB KL for gate_word ──
            if self._vib_enabled:
                kl_word = self._compute_kl_loss(net_out_word["gate_mu"], net_out_word["gate_logvar"], self._vib_free_bits)
                loss_gate_word = loss_gate_word + vib_beta * kl_word
                self.tracker.update_meter("kl_word", "train", kl_word.item())
            self.tracker.update_meter("sem_word", "train", loss_gate_word.item())
            gate_class_word_pre = torch.softmax(gate_word, dim=-1)
            gate_class_3= torch.argmax(gate_class_word_pre, dim=-1)
            gate_class_word = gate_class_3.reshape(-1)

            correct_gate_word = (gate_class_word == sem_label).sum().item()
            acc_gate_word = correct_gate_word / total_sem_label
            self.tracker.update_meter("gate_word", "train", acc_gate_word)
            
            loss_latent_lower_word = self.reclatent_loss(net_out_word["rec_lower"], loaded_data["zq_lower"])
            loss_latent_hands_word = self.reclatent_loss(net_out_word["rec_hands"], loaded_data["zq_hands"])
            loss_latent_upper_word = self.reclatent_loss(net_out_word["rec_upper"], loaded_data["zq_upper"])
            # gate_expanded_word= gate_class_3.unsqueeze(1).unsqueeze(2).unsqueeze(-1)
            # gate_expanded_word = gate_class_word_pre[:,:,0].unsqueeze(1).unsqueeze(2).unsqueeze(-1)
            # if epoch > -1:
            #     gate_expanded_word_detach = gate_expanded_word.detach()
            #     loss_latent_lower_word = loss_latent_lower_word * (2*gate_expanded_word_detach+(1-gate_expanded_word_detach))
            #     loss_latent_hands_word = loss_latent_hands_word * (2*gate_expanded_word_detach+(1-gate_expanded_word_detach))
            #     loss_latent_upper_word = loss_latent_upper_word * (2*gate_expanded_word_detach+(1-gate_expanded_word_detach))
            gate_expanded_word = self._compute_latent_weight(sem_mean, gate_class_word_pre)
            loss_latent_lower_word = loss_latent_lower_word * gate_expanded_word
            loss_latent_hands_word = loss_latent_hands_word * gate_expanded_word
            loss_latent_upper_word = loss_latent_upper_word * gate_expanded_word
            loss_latent_lower_word = loss_latent_lower_word.mean()
            loss_latent_hands_word = loss_latent_hands_word.mean()
            loss_latent_upper_word = loss_latent_upper_word.mean()
            loss_latent_word = self.args.ll*loss_latent_lower_word + self.args.lh*loss_latent_hands_word + self.args.lu*loss_latent_upper_word
            self.tracker.update_meter("latent_word", "train", loss_latent_word.item())
            g_loss_final += loss_latent_word/6
            index_loss_top_word = 0
            for i in range(6):
                rec_index_upper_word = self.log_softmax(net_out_word["cls_upper"][:,:,:,i])
                rec_index_lower_word = self.log_softmax(net_out_word["cls_lower"][:,:,:,i])
                rec_index_hands_word = self.log_softmax(net_out_word["cls_hands"][:,:,:,i])
                cls_weight_word = self._compute_cls_weight(sem_mean, gate_class_word_pre)
                index_loss_top_word_i_upper = ((self.args.cu*self.cls_loss(rec_index_upper_word.transpose(1, 2), tar_index_value_upper_top[:,:,i]))*cls_weight_word).mean()
                index_loss_top_word_i_lower =  ((self.args.cl*self.cls_loss(rec_index_lower_word.transpose(1, 2), tar_index_value_lower_top[:,:,i]))*cls_weight_word).mean()
                index_loss_top_word_i_hands = ((self.args.ch*self.cls_loss(rec_index_hands_word.transpose(1, 2), tar_index_value_hands_top[:,:,i]))*cls_weight_word).mean()
                # if epoch > -1:
                #     index_loss_top_word_i_upper = ((self.args.cu*self.cls_loss(rec_index_upper_word.transpose(1, 2), tar_index_value_upper_top[:,:,i]))*gate_class_3).mean()
                #     index_loss_top_word_i_lower =  ((self.args.cl*self.cls_loss(rec_index_lower_word.transpose(1, 2), tar_index_value_lower_top[:,:,i]))*gate_class_3).mean()
                #     index_loss_top_word_i_hands = ((self.args.ch*self.cls_loss(rec_index_hands_word.transpose(1, 2), tar_index_value_hands_top[:,:,i]))*gate_class_3).mean()
                #     # gate_class_3_detach = gate_class_3.detach()
                #     # index_loss_top_word_i_upper = ((self.args.cu*self.cls_loss(rec_index_upper_word.transpose(1, 2), tar_index_value_upper_top[:,:,i]))*(2*gate_class_3_detach+(1-gate_class_3_detach))).mean()
                #     # index_loss_top_word_i_lower =  ((self.args.cl*self.cls_loss(rec_index_lower_word.transpose(1, 2), tar_index_value_lower_top[:,:,i]))*(2*gate_class_3_detach+(1-gate_class_3_detach))).mean()
                #     # index_loss_top_word_i_hands = ((self.args.ch*self.cls_loss(rec_index_hands_word.transpose(1, 2), tar_index_value_hands_top[:,:,i]))*(2*gate_class_3_detach+(1-gate_class_3_detach))).mean()
                # else:
                #     index_loss_top_word_i_upper = ((self.args.cu*self.cls_loss(rec_index_upper_word.transpose(1, 2), tar_index_value_upper_top[:,:,i])).mean())
                #     index_loss_top_word_i_lower =  ((self.args.cl*self.cls_loss(rec_index_lower_word.transpose(1, 2), tar_index_value_lower_top[:,:,i])).mean())
                #     index_loss_top_word_i_hands = ((self.args.ch*self.cls_loss(rec_index_hands_word.transpose(1, 2), tar_index_value_hands_top[:,:,i])).mean())
                # loss_cls = loss_cls + loss_cls_i/(6-i)
                # loss_cls = loss_cls + loss_cls_i/6
                index_loss_top_word_i = index_loss_top_word_i_upper + index_loss_top_word_i_lower + index_loss_top_word_i_hands
                index_loss_top_word = index_loss_top_word + index_loss_top_word_i/(i+1)

            self.tracker.update_meter("cls_word", "train", index_loss_top_word.item())
            g_loss_final += index_loss_top_word

        if mode == 'train':
            return g_loss_final
        else:
            raise NotImplementedError("The training function should not be called in test mode.")
    
    def _g_test(self, loaded_data):
        mode = 'test'
        bs, n, j = loaded_data["tar_pose"].shape[0], loaded_data["tar_pose"].shape[1], self.joints 
        tar_pose = loaded_data["tar_pose"].cuda()
        tar_beta = loaded_data["tar_beta"].cuda()
        in_word = loaded_data["in_word"].cuda()
        tar_exps = loaded_data["tar_exps"].cuda()
        tar_contact = loaded_data["tar_contact"].cuda()
        tar_trans = loaded_data["tar_trans"]
        hubert = loaded_data["hubert"].cuda()
        beat = loaded_data["beat"].cuda()
        emo_clip_text = loaded_data["emo_clip_text"].cuda()
        feat_clip_text = loaded_data["feat_clip_text"].cuda()
        remain = n%8
        
        if remain != 0:
            tar_pose = tar_pose[:, :-remain, :]
            tar_beta = tar_beta[:, :-remain, :]
            tar_trans = tar_trans[:, :-remain, :]
            in_word = in_word[:, :-remain]
            tar_exps = tar_exps[:, :-remain, :]
            tar_contact = tar_contact[:, :-remain, :]
            n = n - remain

        tar_pose_jaw = tar_pose[:, :, 66:69]
        tar_pose_jaw = rc.axis_angle_to_matrix(tar_pose_jaw.reshape(bs, n, 1, 3))
        tar_pose_jaw = rc.matrix_to_rotation_6d(tar_pose_jaw).reshape(bs, n, 1*6)
        tar_pose_face = torch.cat([tar_pose_jaw, tar_exps], dim=2)

        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)
        
        tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, 55, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 55*6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)
        
        rec_index_all_face = []
        rec_index_all_upper = []
        rec_index_all_lower = []
        rec_index_all_hands = []
        sem_score = []
        # Physics smoother: collect soft gate & logvar for inference-time EMA
        gate_psi_all = []
        gate_logvar_all = []
        
        roundt = (n - self.args.pre_frames) // (self.args.pose_length - self.args.pre_frames)
        remain = (n - self.args.pre_frames) % (self.args.pose_length - self.args.pre_frames)
        round_l = self.args.pose_length - self.args.pre_frames
        


        for i in range(0, roundt):
            in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            feat_clip_text_tmp = feat_clip_text[:, i]
            emo_clip_text_tmp = emo_clip_text[:, i]
            in_beat_tmp = beat[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            in_id_tmp = loaded_data['tar_id'][:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            hubert_tmp = hubert[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+4).float().cuda()
            mask_val[:, :self.args.pre_frames, :] = 0.0
            if i == 0:
                latent_all_tmp = latent_all[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :]
            else:
                latent_all_tmp = latent_all[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :]
                latent_all_tmp[:, :self.args.pre_frames, :] = latent_last[:, -self.args.pre_frames:, :]
            
            duogesture_base_out = self.duogesture_base(
                in_audio = in_beat_tmp,
                in_word=in_word_tmp,
                mask=mask_val,
                in_motion = latent_all_tmp,
                in_id = in_id_tmp,
                hubert=hubert_tmp,
                use_attentions=True,)

            latent_tmp = self.duogesture_base.forward_latent(
                in_audio=in_beat_tmp,
                in_word=in_word_tmp,
                mask=mask_val,
                in_motion=latent_all_tmp,
                in_id=in_id_tmp,
                hubert=hubert_tmp,
                use_attentions=True,
            )
            
            net_out_val = self.model(
                in_word=in_word_tmp,
                feat_clip_text=feat_clip_text_tmp,
                emotion=emo_clip_text_tmp,
                mask=mask_val,
                in_motion = latent_all_tmp,
                in_id = in_id_tmp,
                use_attentions=True,
                hubert=hubert_tmp,
                latent=latent_tmp,
            )
            
            gate = net_out_val["gate"]
            gate_score = torch.softmax(gate, dim=-1)
            gate_psi_chunk = gate_score[:, :, 1:2]              # [B, T', 1] soft semantic prob
            gate_logvar_chunk = net_out_val.get("gate_logvar")   # [B, T', z] or None
            gate_hard = torch.argmax(gate_score, dim=-1).unsqueeze(-1)
            gate_sem = gate_score[:, :, 1:2].unsqueeze(-1)      # [B, T', 1, 1]

            # Soft blend at logit level avoids discontinuity from hard switching.
            cls_upper_mix = duogesture_base_out["cls_upper"] * (1.0 - gate_sem) + net_out_val["cls_upper"] * gate_sem
            cls_lower_mix = duogesture_base_out["cls_lower"] * (1.0 - gate_sem) + net_out_val["cls_lower"] * gate_sem
            cls_hands_mix = duogesture_base_out["cls_hands"] * (1.0 - gate_sem) + net_out_val["cls_hands"] * gate_sem

            rec_index_upper = torch.argmax(self.log_softmax(cls_upper_mix), dim=2)
            rec_index_lower = torch.argmax(self.log_softmax(cls_lower_mix), dim=2)
            rec_index_hands = torch.argmax(self.log_softmax(cls_hands_mix), dim=2)
            rec_index_face = torch.argmax(self.log_softmax(duogesture_base_out["cls_face"]), dim=2)

            if i == 0:
                rec_index_all_face.append(rec_index_face)
                rec_index_all_upper.append(rec_index_upper)
                rec_index_all_lower.append(rec_index_lower)
                rec_index_all_hands.append(rec_index_hands)
                sem_score.append(gate_hard)
                gate_psi_all.append(gate_psi_chunk)
                if gate_logvar_chunk is not None:
                    gate_logvar_all.append(gate_logvar_chunk)
            else:
                rec_index_all_face.append(rec_index_face[:, 1:])
                rec_index_all_upper.append(rec_index_upper[:, 1:])
                rec_index_all_lower.append(rec_index_lower[:, 1:])
                rec_index_all_hands.append(rec_index_hands[:, 1:])
                sem_score.append(gate_hard[:, 1:])
                gate_psi_all.append(gate_psi_chunk[:, 1:])
                if gate_logvar_chunk is not None:
                    gate_logvar_all.append(gate_logvar_chunk[:, 1:])


            rec_upper_last = self.vq_model_upper.decode(rec_index_upper)
            rec_lower_last = self.vq_model_lower.decode(rec_index_lower)
            rec_hands_last = self.vq_model_hands.decode(rec_index_hands)
            
            rec_pose_legs = rec_lower_last[:, :, :54]
            bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
            rec_pose_upper = rec_upper_last.reshape(bs, n, 13, 6)
            rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
            rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 13*3)
            rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
            rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
            rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
            rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
            rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
            rec_pose_hands = rec_hands_last.reshape(bs, n, 30, 6)
            rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
            rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 30*3)
            rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
            rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
            rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs, n, j, 3))
            rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
            rec_trans_v_s = rec_lower_last[:, :, 54:57]
            rec_x_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
            rec_z_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
            rec_y_trans = rec_trans_v_s[:,:,1:2]
            rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
            latent_last = torch.cat([rec_pose, rec_trans, rec_lower_last[:, :, 57:61]], dim=-1)

        rec_index_face = torch.cat(rec_index_all_face, dim=1)
        rec_index_upper = torch.cat(rec_index_all_upper, dim=1)
        rec_index_lower = torch.cat(rec_index_all_lower, dim=1)
        rec_index_hands = torch.cat(rec_index_all_hands, dim=1)

        sem_score = torch.cat(sem_score, dim=1)

        rec_upper = self.vq_model_upper.decode(rec_index_upper)
        rec_lower = self.vq_model_lower.decode(rec_index_lower)
        rec_hands = self.vq_model_hands.decode(rec_index_hands)
        rec_face = self.vq_model_face.decode(rec_index_face)
        

        rec_exps = rec_face[:, :, 6:]
        rec_pose_jaw = rec_face[:, :, :6]
        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_jaw.shape[0], rec_pose_jaw.shape[1]
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 13*3)
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = rc.matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9*6)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 30*3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
        rec_pose_jaw = rec_pose_jaw.reshape(bs*n, 6)
        rec_pose_jaw = rc.rotation_6d_to_matrix(rec_pose_jaw)
        rec_pose_jaw = rc.matrix_to_axis_angle(rec_pose_jaw).reshape(bs*n, 1*3)
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
        rec_pose[:, 66:69] = rec_pose_jaw

        to_global = rec_lower
        to_global[:, :, 54:57] = 0.0
        to_global[:, :, :54] = rec_lower2global
        rec_global = self.global_motion(to_global)

        rec_trans_v_s = rec_global["rec_pose"][:, :, 54:57]
        rec_x_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
        rec_z_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
        rec_y_trans = rec_trans_v_s[:,:,1:2]
        rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]

        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs*n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
        tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs*n, j, 3))
        tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)

        # ── Physics smoother: per-joint EMA on decoded rot6d ──
        if self._phys_enabled and len(gate_psi_all) > 0:
            gate_psi_cat = torch.cat(gate_psi_all, dim=1)           # [B, total_T', 1]
            gate_logvar_cat = (
                torch.cat(gate_logvar_all, dim=1)
                if len(gate_logvar_all) > 0 else None
            )
            # Log τ statistics at inference for analysis
            with torch.no_grad():
                _tau_dbg = self.physics_smoother.compute_tau(gate_psi_cat, gate_logvar_cat)
                _psi_mean = gate_psi_cat.mean().item()
                _tau_mean = _tau_dbg.mean().item()
                _tau_max  = _tau_dbg.max().item()
                _sem_frac = (gate_psi_cat.squeeze(-1) > 0.5).float().mean().item()
                if not hasattr(self, '_tau_log_count'):
                    self._tau_log_count = 0
                if self._tau_log_count % 50 == 0:
                    logger.info(
                        f"[phys-infer] ψ_mean={_psi_mean:.3f}  sem_frac={_sem_frac:.3f}  "
                        f"τ_mean={_tau_mean:.4f}  τ_max={_tau_max:.4f}"
                    )
                self._tau_log_count += 1
            rec_pose_6d = rec_pose.reshape(bs, n, j, 6)
            rec_pose_6d = self.physics_smoother.forward_inference(
                rec_pose_6d,
                gate_psi_cat.squeeze(-1),               # [B, total_T']
                gate_logvar_cat,
            )
            rec_pose = rec_pose_6d.reshape(bs, n, j * 6)
        
        # Collect gate_psi for downstream visualisation
        gate_psi_np = None
        if len(gate_psi_all) > 0:
            gate_psi_np = torch.cat(gate_psi_all, dim=1).squeeze(-1).detach().cpu().numpy()

        return {
            'rec_pose': rec_pose,
            'rec_trans': rec_trans,
            'tar_pose': tar_pose,
            'tar_exps': tar_exps,
            'tar_beta': tar_beta,
            'tar_trans': tar_trans,
            'rec_exps': rec_exps,
            'gate_psi': gate_psi_np,
        }
    
    def train(self, epoch):
        #torch.autograd.set_detect_anomaly(True)
        use_adv = bool(epoch>=self.args.no_adv_epoch)
        self.model.train()
        # self.d_model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, batch_data in enumerate(self.train_loader):
            loaded_data = self._load_data(batch_data)
            t_data = time.time() - t_start
            # print('load data time:', t_data)
            self.opt.zero_grad()
            g_loss_final = 0
            g_loss_final += self._g_training(loaded_data, use_adv, 'train', epoch)
            # print('g_training time:', time.time()-t_start)
            #with torch.autograd.detect_anomaly():
            g_loss_final.backward()
            if self.args.grad_norm != 0: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            # for name, param in self.model.named_parameters():
            #     if param.grad is None:
            #         print(name)
            self.opt.step()
            
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            # lr_d = self.opt_d.param_groups[0]['lr']
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            # print('train time:', t_train)

            if its % self.args.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)   
            if self.args.debug:
                if its == 1: break
        self.opt_s.step(epoch)
        # self.opt_d_s.step(epoch) 
   
    def test(self, epoch):
        
        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if os.path.exists(results_save_path):
            logger.warning(
                f"[test] Skip epoch {epoch}: results already exist at {results_save_path}"
            )
            return None
        os.makedirs(results_save_path, exist_ok=True)
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        align = 0 
        latent_out = []
        latent_ori = []
        l2_all = 0 
        lvel = 0
        self.model.eval()
        self.smplx.eval()
        self.eval_copy.eval()
        sem_talk_list = []
        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)
                net_out = self._g_test(loaded_data)
                tar_pose = net_out['tar_pose']
                rec_pose = net_out['rec_pose']
                tar_exps = net_out['tar_exps']
                tar_beta = net_out['tar_beta']
                rec_trans = net_out['rec_trans']
                tar_trans = net_out['tar_trans']
                rec_exps = net_out['rec_exps']
                # print(rec_pose.shape, tar_pose.shape)
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                
                if (30/self.args.pose_fps) != 1:
                    assert 30%self.args.pose_fps == 0
                    n *= int(30/self.args.pose_fps)
                    tar_pose = torch.nn.functional.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                    rec_pose = torch.nn.functional.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30/self.args.pose_fps, mode='linear').permute(0,2,1)
                
                # print(rec_pose.shape, tar_pose.shape)
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
                remain = n%self.args.vae_test_len
                latent_out.append(self.eval_copy.map2latent(rec_pose[:, :n-remain]).reshape(-1, self.args.vae_length).detach().cpu().numpy()) # bs * n/8, 240
                latent_ori.append(self.eval_copy.map2latent(tar_pose[:, :n-remain]).reshape(-1, self.args.vae_length).detach().cpu().numpy())
                
                rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(bs*n, j, 6))
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs*n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                
                vertices_rec = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
                        expression=tar_exps.reshape(bs*n, 100)-tar_exps.reshape(bs*n, 100),
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3], 
                        left_hand_pose=rec_pose[:,25*3:40*3], 
                        right_hand_pose=rec_pose[:,40*3:55*3], 
                        return_joints=True, 
                        leye_pose=rec_pose[:, 69:72], 
                        reye_pose=rec_pose[:, 72:75],
                    )
                vertices_rec_face = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
                        expression=rec_exps.reshape(bs*n, 100), 
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3]-rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3]-rec_pose[:,3:21*3+3],
                        left_hand_pose=rec_pose[:,25*3:40*3]-rec_pose[:,25*3:40*3],
                        right_hand_pose=rec_pose[:,40*3:55*3]-rec_pose[:,40*3:55*3],
                        return_verts=True, 
                        return_joints=True,
                        leye_pose=rec_pose[:, 69:72]-rec_pose[:, 69:72],
                        reye_pose=rec_pose[:, 72:75]-rec_pose[:, 72:75],
                    )
                vertices_tar_face = self.smplx(
                    betas=tar_beta.reshape(bs*n, 300), 
                    transl=tar_trans.reshape(bs*n, 3)-tar_trans.reshape(bs*n, 3), 
                    expression=tar_exps.reshape(bs*n, 100), 
                    jaw_pose=tar_pose[:, 66:69], 
                    global_orient=tar_pose[:,:3]-tar_pose[:,:3],
                    body_pose=tar_pose[:,3:21*3+3]-tar_pose[:,3:21*3+3], 
                    left_hand_pose=tar_pose[:,25*3:40*3]-tar_pose[:,25*3:40*3],
                    right_hand_pose=tar_pose[:,40*3:55*3]-tar_pose[:,40*3:55*3],
                    return_verts=True, 
                    return_joints=True,
                    leye_pose=tar_pose[:, 69:72]-tar_pose[:, 69:72],
                    reye_pose=tar_pose[:, 72:75]-tar_pose[:, 72:75],
                )  
                joints_rec = vertices_rec["joints"].detach().cpu().numpy().reshape(1, n, 127*3)[0, :n, :55*3]
                sem_talk_list.append(joints_rec)
                facial_rec = vertices_rec_face['vertices'].reshape(1, n, -1)[0, :n]
                facial_tar = vertices_tar_face['vertices'].reshape(1, n, -1)[0, :n]
                face_vel_loss = self.vel_loss(facial_rec[1:, :] - facial_tar[:-1, :], facial_tar[1:, :] - facial_tar[:-1, :])
               
                l2 = self.reclatent_loss_test(facial_rec, facial_tar)
                l2_all += l2.item() * n
                lvel += face_vel_loss.item() * n
                
                _ = self.l1_calculator.run(joints_rec)
                if self.alignmenter is not None:
                    in_audio_eval, sr = librosa.load(self.args.data_path+"wave16k/"+test_seq_list.iloc[its]['id']+".wav")
                    in_audio_eval = librosa.resample(in_audio_eval, orig_sr=sr, target_sr=self.args.audio_sr)
                    a_offset = int(self.align_mask * (self.args.audio_sr / self.args.pose_fps))
                    onset_bt = self.alignmenter.load_audio(in_audio_eval[:int(self.args.audio_sr / self.args.pose_fps*n)], a_offset, len(in_audio_eval)-a_offset, True)
                    beat_vel = self.alignmenter.load_pose(joints_rec, self.align_mask, n-self.align_mask, 30, True)
                    align += (self.alignmenter.calculate_align(onset_bt, beat_vel, 30) * (n-2*self.align_mask))
               
                tar_pose_np = tar_pose.detach().cpu().numpy()
                rec_pose_np = rec_pose.detach().cpu().numpy()
                rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs*n, 3)
                rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs*n, 100) 
                tar_exp_np = tar_exps.detach().cpu().numpy().reshape(bs*n, 100)
                tar_trans_np = tar_trans.detach().cpu().numpy().reshape(bs*n, 3)
                gt_npz = np.load(self.args.data_path+self.args.pose_rep +"/"+test_seq_list.iloc[its]['id']+".npz", allow_pickle=True)
                
                np.savez(results_save_path+"gt_"+test_seq_list.iloc[its]['id']+'.npz',
                    betas=gt_npz["betas"],
                    poses=tar_pose_np,
                    expressions=tar_exp_np,
                    trans=tar_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 ,
                )
                save_kwargs = dict(
                    betas=gt_npz["betas"],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate=30,
                )
                if net_out.get('gate_psi') is not None:
                    save_kwargs['gate_psi'] = net_out['gate_psi']
                np.savez(results_save_path+"res_"+test_seq_list.iloc[its]['id']+'.npz', **save_kwargs)
                total_length += n
        
        logger.info(f"l2 loss: {l2_all/total_length}")
        logger.info(f"lvel loss: {lvel/total_length}")

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)
        fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fid score: {fid}")
        self.test_recording("fid", fid, epoch) 
        
        align_avg = align/(total_length-2*len(self.test_loader)*self.align_mask)
        logger.info(f"align score: {align_avg}")
        self.test_recording("bc", align_avg, epoch)

        l1div = self.l1_calculator.avg()
        logger.info(f"l1div score: {l1div}")
        self.test_recording("l1div", l1div, epoch)

        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")
        return fid

    
    def inference(self, audio_path, out_name=None):
        mode = 'inference'
        # test_seq_list = self.test_data.selected_file
        from utils import audio_to_frame_tokens

        in_word, in_sentence = audio_to_frame_tokens.get_word_sentence(audio_path, self.args)
        beat, hubert  = audio_to_frame_tokens.get_hubert(audio_path, self.args)
        beat = beat.cuda()
        hubert = hubert.cuda()
        print("frame:",hubert.shape, in_word.shape, beat.shape, len(in_sentence))
        frames = len(in_sentence) * (self.args.pose_length - self.args.pre_frames) + self.args.pre_frames
        bs, j = 1, self.joints
        in_emo_sep = audio_to_frame_tokens.get_emo(audio_path, frames)

        # --- Load text encoder: TMR / MoCLIP / vanilla CLIP ---
        semantic_encoder = getattr(self.args, 'semantic_encoder', 'clip')
        moclip_ckpt = getattr(self.args, 'moclip_ckpt', None)
        self._use_tmr = False
        if semantic_encoder == 'tmr' and moclip_ckpt and os.path.isfile(moclip_ckpt):
            distilbert_path = getattr(self.args, 'distilbert_path', 'distilbert-base-uncased')
            print(f"[Inference] Loading TMR text encoder from {moclip_ckpt}")
            self.tmr_encoder = load_tmr_text_encoder(
                text_encoder_path=moclip_ckpt,
                distilbert_path=distilbert_path,
                device=str(self.device),
            )
            self._use_tmr = True
        elif semantic_encoder == 'moclip' and moclip_ckpt and os.path.isfile(moclip_ckpt):
            clip_version = getattr(self.args, 'clip_version', 'ViT-B/32')
            print(f"[Inference] Loading MoCLIP text encoder from {moclip_ckpt}")
            self.clip_model = load_moclip_text_encoder(moclip_ckpt, device=str(self.device), clip_version=clip_version)
        else:
            self.clip_model = self.load_and_freeze_clip(device=self.device)
        feat_clip_text_list = []
        emo_clip_text_list = []
        for i in range(len(in_sentence)):
            feat_clip_text_i = self.encode_text(in_sentence[i], self.device)
            emo_clip_text_i  = self.encode_text(in_emo_sep[i], self.device)
            feat_clip_text_list.append(feat_clip_text_i)
            emo_clip_text_list.append(emo_clip_text_i)
        feat_clip_text = torch.stack(feat_clip_text_list, dim=1)
        emo_clip_text  = torch.stack(emo_clip_text_list, dim=1)

                # 从 dataloader 中取出第一个 batch
        batch_data = next(iter(self.test_loader))
        loaded_data = self._load_data(batch_data)
        tar_pose = loaded_data["tar_pose"].cuda()
        tar_beta = loaded_data["tar_beta"].cuda()
        tar_exps = loaded_data["tar_exps"].cuda()
        tar_contact = loaded_data["tar_contact"].cuda()
        tar_trans = loaded_data["tar_trans"].cuda()
        pos_len = tar_pose.shape[1]
        remain = pos_len%8
        
        if remain != 0:
            tar_pose = tar_pose[:, :pos_len-remain, :]
            tar_beta = tar_beta[:, :pos_len-remain, :]
            tar_trans = tar_trans[:, :pos_len-remain, :]
            tar_exps = tar_exps[:, :pos_len-remain, :]
            tar_contact = tar_contact[:, :pos_len-remain, :]
            pos_len = pos_len - remain
        n = pos_len
        tar_pose_jaw = tar_pose[:, :, 66:69]
        tar_pose_jaw = rc.axis_angle_to_matrix(tar_pose_jaw.reshape(bs, n, 1, 3))
        tar_pose_jaw = rc.matrix_to_rotation_6d(tar_pose_jaw).reshape(bs, n, 1*6)
        tar_pose_face = torch.cat([tar_pose_jaw, tar_exps], dim=2)

        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30*6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13*6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9*6)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)
        
        tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, 55, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 55*6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)
        
        rec_index_all_face = []
        rec_index_all_upper = []
        rec_index_all_lower = []
        rec_index_all_hands = []
        sem_score = []
        # Physics smoother: collect soft gate & logvar for inference-time EMA
        gate_psi_all_inf = []
        gate_logvar_all_inf = []
        
        roundt = len(in_sentence)
        round_l = self.args.pose_length - self.args.pre_frames
        
        for i in range(0, roundt):
            in_word_tmp = in_word[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            feat_clip_text_tmp = feat_clip_text[:, i]
            emo_clip_text_tmp = emo_clip_text[:, i]
            in_beat_tmp = beat[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            in_id_tmp = loaded_data['tar_id'][:, : round_l+self.args.pre_frames]
            hubert_tmp = hubert[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames]
            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims+3+4).float().cuda()
            mask_val[:, :self.args.pre_frames, :] = 0.0
            if i == 0:
                latent_all_tmp = latent_all[:, i*(round_l):(i+1)*(round_l)+self.args.pre_frames, :]
            else:
                latent_all_tmp = torch.zeros_like(latent_last[:, :round_l+self.args.pre_frames, :])
                latent_all_tmp[:, :self.args.pre_frames, :] = latent_last[:, -self.args.pre_frames:, :]
            
            duogesture_base_out = self.duogesture_base(
                in_audio = in_beat_tmp,
                in_word=in_word_tmp,
                mask=mask_val,
                in_motion = latent_all_tmp,
                in_id = in_id_tmp,
                hubert=hubert_tmp,
                use_attentions=True,)

            latent_tmp = self.duogesture_base.forward_latent(
                in_audio=in_beat_tmp,
                in_word=in_word_tmp,
                mask=mask_val,
                in_motion=latent_all_tmp,
                in_id=in_id_tmp,
                hubert=hubert_tmp,
                use_attentions=True,
            )
            
            net_out_val = self.model(
                in_word=in_word_tmp,
                feat_clip_text=feat_clip_text_tmp,
                emotion=emo_clip_text_tmp,
                mask=mask_val,
                in_motion = latent_all_tmp,
                in_id = in_id_tmp,
                use_attentions=True,
                hubert=hubert_tmp,
                latent=latent_tmp,
            )
            
            gate = net_out_val["gate"]
            gate_score = torch.softmax(gate, dim=-1)
            gate_psi_chunk = gate_score[:, :, 1:2]              # [B, T', 1] soft semantic prob
            gate_logvar_chunk = net_out_val.get("gate_logvar")   # [B, T', z] or None
            gate_hard = torch.argmax(gate_score, dim=-1).unsqueeze(-1)
            gate_sem = gate_score[:, :, 1:2].unsqueeze(-1)      # [B, T', 1, 1]

            cls_upper_mix = duogesture_base_out["cls_upper"] * (1.0 - gate_sem) + net_out_val["cls_upper"] * gate_sem
            cls_lower_mix = duogesture_base_out["cls_lower"] * (1.0 - gate_sem) + net_out_val["cls_lower"] * gate_sem
            cls_hands_mix = duogesture_base_out["cls_hands"] * (1.0 - gate_sem) + net_out_val["cls_hands"] * gate_sem

            rec_index_upper = torch.argmax(self.log_softmax(cls_upper_mix), dim=2)
            rec_index_lower = torch.argmax(self.log_softmax(cls_lower_mix), dim=2)
            rec_index_hands = torch.argmax(self.log_softmax(cls_hands_mix), dim=2)
            rec_index_face = torch.argmax(self.log_softmax(duogesture_base_out["cls_face"]), dim=2)

            if i == 0:
                rec_index_all_face.append(rec_index_face)
                rec_index_all_upper.append(rec_index_upper)
                rec_index_all_lower.append(rec_index_lower)
                rec_index_all_hands.append(rec_index_hands)
                sem_score.append(gate_hard)
                gate_psi_all_inf.append(gate_psi_chunk)
                if gate_logvar_chunk is not None:
                    gate_logvar_all_inf.append(gate_logvar_chunk)
            else:
                rec_index_all_face.append(rec_index_face[:, 1:])
                rec_index_all_upper.append(rec_index_upper[:, 1:])
                rec_index_all_lower.append(rec_index_lower[:, 1:])
                rec_index_all_hands.append(rec_index_hands[:, 1:])
                sem_score.append(gate_hard[:, 1:])
                gate_psi_all_inf.append(gate_psi_chunk[:, 1:])
                if gate_logvar_chunk is not None:
                    gate_logvar_all_inf.append(gate_logvar_chunk[:, 1:])


            rec_upper_last = self.vq_model_upper.decode(rec_index_upper)
            rec_lower_last = self.vq_model_lower.decode(rec_index_lower)
            rec_hands_last = self.vq_model_hands.decode(rec_index_hands)
            
            rec_pose_legs = rec_lower_last[:, :, :54]
            bs, n = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
            rec_pose_upper = rec_upper_last.reshape(bs, n, 13, 6)
            rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
            rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 13*3)
            rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
            rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
            rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
            rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
            rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
            rec_pose_hands = rec_hands_last.reshape(bs, n, 30, 6)
            rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
            rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 30*3)
            rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
            rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
            rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs, n, j, 3))
            rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j*6)
            rec_trans_v_s = rec_lower_last[:, :, 54:57]
            rec_x_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
            rec_z_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
            rec_y_trans = rec_trans_v_s[:,:,1:2]
            rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
            latent_last = torch.cat([rec_pose, rec_trans, rec_lower_last[:, :, 57:61]], dim=-1)

        rec_index_face = torch.cat(rec_index_all_face, dim=1)
        rec_index_upper = torch.cat(rec_index_all_upper, dim=1)
        rec_index_lower = torch.cat(rec_index_all_lower, dim=1)
        rec_index_hands = torch.cat(rec_index_all_hands, dim=1)

        sem_score = torch.cat(sem_score, dim=1)

        rec_upper = self.vq_model_upper.decode(rec_index_upper)
        rec_lower = self.vq_model_lower.decode(rec_index_lower)
        rec_hands = self.vq_model_hands.decode(rec_index_hands)
        rec_face = self.vq_model_face.decode(rec_index_face)
        

        rec_exps = rec_face[:, :, 6:]
        rec_pose_jaw = rec_face[:, :, :6]
        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_jaw.shape[0], rec_pose_jaw.shape[1]
        rec_pose_upper = rec_upper.reshape(bs, n, 13, 6)
        rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)#
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 13*3)
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs*n)
        rec_pose_lower = rec_pose_legs.reshape(bs, n, 9, 6)
        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
        rec_lower2global = rc.matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9*6)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs*n)
        rec_pose_hands = rec_hands.reshape(bs, n, 30, 6)
        rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 30*3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs*n)
        rec_pose_jaw = rec_pose_jaw.reshape(bs*n, 6)
        rec_pose_jaw = rc.rotation_6d_to_matrix(rec_pose_jaw)
        rec_pose_jaw = rc.matrix_to_axis_angle(rec_pose_jaw).reshape(bs*n, 1*3)
        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover 
        rec_pose[:, 66:69] = rec_pose_jaw

        to_global = rec_lower
        to_global[:, :, 54:57] = 0.0
        to_global[:, :, :54] = rec_lower2global
        rec_global = self.global_motion(to_global)

        rec_trans_v_s = rec_global["rec_pose"][:, :, 54:57]
        rec_x_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 0:1], 1/self.args.pose_fps, tar_trans[:, 0, 0:1])
        rec_z_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 2:3], 1/self.args.pose_fps, tar_trans[:, 0, 2:3])
        rec_y_trans = rec_trans_v_s[:,:,1:2]
        rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
        tar_pose = tar_pose[:, :n, :]
        tar_exps = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta = tar_beta[:, :n, :]

        # ── Physics smoother: per-joint EMA on inference poses ──
        if self._phys_enabled and len(gate_psi_all_inf) > 0:
            gate_psi_cat = torch.cat(gate_psi_all_inf, dim=1)
            gate_logvar_cat = (
                torch.cat(gate_logvar_all_inf, dim=1)
                if len(gate_logvar_all_inf) > 0 else None
            )
            rec_pose_6d_inf = rc.axis_angle_to_matrix(rec_pose.reshape(bs, n, j, 3))
            rec_pose_6d_inf = rc.matrix_to_rotation_6d(rec_pose_6d_inf)  # [bs, n, 55, 6]
            rec_pose_6d_inf = self.physics_smoother.forward_inference(
                rec_pose_6d_inf, gate_psi_cat.squeeze(-1), gate_logvar_cat,
            )
            rec_pose_6d_inf = rc.rotation_6d_to_matrix(rec_pose_6d_inf)
            rec_pose = rc.matrix_to_axis_angle(rec_pose_6d_inf).reshape(bs * n, j * 3)
      
        rec_pose_np = rec_pose.detach().cpu().numpy()
        rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs*n, 3)
        rec_exp_np = rec_exps.detach().cpu().numpy().reshape(bs*n, 100)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if out_name is not None:
            # Use caller-specified name but always append timestamp to avoid overwrite.
            out_dir, out_file = os.path.split(out_name)
            base_name, ext = os.path.splitext(out_file)
            if ext == "":
                ext = ".npz"
            stamped_name = f"{base_name}_{timestamp}{ext}"
            if out_dir:
                save_path = os.path.join(out_dir, stamped_name)
            else:
                save_path = os.path.join("./demo", stamped_name)
        else:
            base = os.path.basename(audio_path)
            filename = os.path.splitext(base)[0] + f"_{timestamp}.npz"
            save_path = os.path.join("./demo", filename)
        gt_npz = np.load("demo/2_scott_0_1_1.npz", allow_pickle=True)
        save_kwargs = dict(
            betas=gt_npz["betas"],
            poses=rec_pose_np,
            expressions=rec_exp_np,
            trans=rec_trans_np,
            model='smplx2020',
            gender='neutral',
            mocap_frame_rate=30,
        )
        if len(gate_psi_all_inf) > 0:
            save_kwargs['gate_psi'] = torch.cat(gate_psi_all_inf, dim=1).squeeze(-1).detach().cpu().numpy()
        np.savez(save_path, **save_kwargs)
        print("result saved to ", save_path)
    
    def encode_text(self, raw_text, device):
        if getattr(self, '_use_tmr', False):
            if isinstance(raw_text, str):
                raw_text = [raw_text]
            return self.tmr_encoder.encode(raw_text)  # [B, 256]
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text
    
    def load_and_freeze_clip(self, clip_version='ViT-B/32', device='cuda:0'):
        clip_model, clip_preprocess = clip.load(clip_version, device=device,
                                                jit=False)  # Must set jit=False for training
        # Cannot run on cpu
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model
        

