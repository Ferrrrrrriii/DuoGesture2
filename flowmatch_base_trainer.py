"""
Flow Matching Base-Motion Trainer for GestureLSM × DuoGesture
==========================================================

Trains the ``GestureLSMBaseMotion`` model using:
    1. Flow Matching velocity-field loss  (L_FM)
    2. RVQ latent reconstruction loss     (L_rec)  — teacher-forced
    3. RVQ classification loss            (L_cls)  — NLL on codebook indices
    4. Contrastive losses                 (L_con)  — rhythmic identification

Total loss =
    w_fm · L_FM
  + (1/6) · L_rec
  + L_cls
  + L_hubert_con + L_beat_con
  + [ masked-gesture-modeling variants when training ]

Usage:
    python train.py --config configs/duogesture_flowmatch_base.yaml
"""

import train
import os
import time
import csv
import sys
import warnings
import random
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


class CustomTrainer(train.BaseTrainer):
    """
    Drop-in replacement for the original ``duogesture_base_trainer.CustomTrainer``.

    Key differences:
        - Extracts VQ level-0 targets and passes them to the model as
          ``target_latents`` for the Flow Matching objective.
        - Adds ``fm_loss`` tracking.
        - Test-time generation uses ODE integration (``forward_latent``).
    """

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.now_epoch = 0
        self.best_fid = float("inf")

        # ---- joint masks (same as original base trainer) ----
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.tar_joint_list_face  = joints_list["beat_smplx_face"]
        self.tar_joint_list_upper = joints_list["beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["beat_smplx_lower"]

        self.joint_mask_face = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        self.joints = 55
        for jn in self.tar_joint_list_face:
            self.joint_mask_face[self.ori_joint_list[jn][1] - self.ori_joint_list[jn][0]:self.ori_joint_list[jn][1]] = 1
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for jn in self.tar_joint_list_upper:
            self.joint_mask_upper[self.ori_joint_list[jn][1] - self.ori_joint_list[jn][0]:self.ori_joint_list[jn][1]] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for jn in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[jn][1] - self.ori_joint_list[jn][0]:self.ori_joint_list[jn][1]] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys())) * 3)
        for jn in self.tar_joint_list_lower:
            self.joint_mask_lower[self.ori_joint_list[jn][1] - self.ori_joint_list[jn][0]:self.ori_joint_list[jn][1]] = 1

        # ---- metrics tracker ----
        self.tracker = other_tools.EpochTracker(
            ["fm", "hubert_cons", "beat_cons",
             "acc_face", "acc_hands", "acc_upper", "acc_lower",
             "fid", "l1div", "bc", "rec", "trans", "vel", "transv",
             "dis", "gen", "acc", "transa", "exp", "lvd", "mse",
             "cls", "rec_face", "latent",
             "cls_full", "cls_self", "cls_word",
             "latent_word", "latent_self"],
            [False, False, True,
             True, False, False, False,
             False, False, False, False, False, False, False,
             False, False, False, False, False, False, False,
             False, False, False,
             False, False, False,
             False, False]
        )

        # ---- pre-trained VQ decoders (frozen) ----
        vq_model_module  = __import__("models.motion_representation", fromlist=["something"])
        rvq_model_module = __import__("models.rvq", fromlist=["something"])

        self.args.vae_layer  = 2
        self.args.vae_length = 256
        self.args.vae_test_dim = 106  # face
        self.vq_model_face = getattr(rvq_model_module, "RVQVAE")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_face, "./weights/pretrained_vq/rvq_face_600.bin", args.e_name)

        self.args.vae_test_dim = 78  # upper body
        self.vq_model_upper = getattr(rvq_model_module, "RVQVAE")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_upper, "./weights/pretrained_vq/rvq_upper_500.bin", args.e_name)

        self.args.vae_test_dim = 180  # hands
        self.vq_model_hands = getattr(rvq_model_module, "RVQVAE")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_hands, "./weights/pretrained_vq/rvq_hands_500.bin", args.e_name)

        self.args.vae_test_dim = 61  # lower body
        self.args.vae_layer = 4
        self.vq_model_lower = getattr(rvq_model_module, "RVQVAE")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.vq_model_lower, "./weights/pretrained_vq/rvq_lower_600.bin", args.e_name)

        self.args.vae_test_dim = 61  # global motion
        self.args.vae_layer = 4
        self.global_motion = getattr(vq_model_module, "VAEConvZero")(self.args).to(self.rank)
        other_tools.load_checkpoints(self.global_motion, "./weights/pretrained_vq/last_1700_foot.bin", args.e_name)

        # restore args to defaults
        self.args.vae_test_dim = 330
        self.args.vae_layer = 4
        self.args.vae_length = 240

        self.vq_model_face.eval()
        self.vq_model_upper.eval()
        self.vq_model_hands.eval()
        self.vq_model_lower.eval()
        self.global_motion.eval()

        # ---- loss functions ----
        self.cls_loss      = nn.NLLLoss().to(self.rank)
        self.reclatent_loss = nn.MSELoss().to(self.rank)
        self.vel_loss       = nn.L1Loss(reduction="mean").to(self.rank)
        self.rec_loss       = get_loss_func("GeodesicLoss").to(self.rank)
        self.log_softmax    = nn.LogSoftmax(dim=2).to(self.rank)

        # flow matching loss weight
        self.fm_loss_weight = float(getattr(args, "fm_loss_weight", 1.0))

    # ------------------------------------------------------------------
    # data helpers
    # ------------------------------------------------------------------

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
        return self._move_to_device(dict_data, self.device)

    def _move_to_device(self, obj, device):
        if torch.is_tensor(obj):
            return obj.to(device, non_blocking=True)
        if isinstance(obj, np.ndarray):
            if obj.dtype == np.object_:
                return obj
            return torch.from_numpy(obj).to(device, non_blocking=True)
        if isinstance(obj, dict):
            return {k: self._move_to_device(v, device) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move_to_device(v, device) for v in obj)
        return obj

    # ------------------------------------------------------------------
    # extract level-0 VQ targets for Flow Matching
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_level0_targets(loaded_data):
        """
        Extract VQ level-0 latent codes from the pre-computed data.

        loaded_data["zq_*"] shape:  [B, 6, 1, T', 256]
            dim-1 = 6 RVQ levels,  dim-2 = 1 (singleton)

        Returns:
            dict with face/upper/hands/lower each [B, T', 256]
        """
        return {
            "face":  loaded_data["zq_face"][:, 0, 0],    # [B, T', 256]
            "upper": loaded_data["zq_upper"][:, 0, 0],
            "hands": loaded_data["zq_hands"][:, 0, 0],
            "lower": loaded_data["zq_lower"][:, 0, 0],
        }

    # ------------------------------------------------------------------
    # training step
    # ------------------------------------------------------------------

    def _g_training(self, loaded_data, use_adv, mode="train", epoch=0):
        bs, n, j = (loaded_data["tar_pose"].shape[0],
                    loaded_data["tar_pose"].shape[1],
                    self.joints)

        # ---- seed-pose mask (first pre_frames unmasked) ----
        mask_val = torch.ones(bs, n, self.args.pose_dims + 3 + 4).float().cuda()
        mask_val[:, :self.args.pre_frames, :] = 0.0

        # ---- extract FM targets ----
        target_latents = self._extract_level0_targets(loaded_data)

        # ---- forward ----
        net_out = self.model(
            loaded_data["beat"],
            loaded_data["in_word"],
            mask=mask_val,
            in_id=loaded_data["tar_id"],
            in_motion=loaded_data["latent_all"],
            use_attentions=True,
            hubert=loaded_data["hubert"],
            is_train=True,
            target_latents=target_latents,
        )

        g_loss_final = 0.0

        # ═══════  1. Flow Matching loss  ═══════
        fm_loss = net_out["fm_loss"] * self.fm_loss_weight
        self.tracker.update_meter("fm", "train", fm_loss.item())
        g_loss_final += fm_loss

        # ═══════  2. RVQ latent reconstruction loss (teacher-forced)  ═══════
        loss_latent_face  = self.reclatent_loss(net_out["rec_face"],  loaded_data["zq_face"])
        loss_latent_lower = self.reclatent_loss(net_out["rec_lower"], loaded_data["zq_lower"])
        loss_latent_hands = self.reclatent_loss(net_out["rec_hands"], loaded_data["zq_hands"])
        loss_latent_upper = self.reclatent_loss(net_out["rec_upper"], loaded_data["zq_upper"])
        loss_latent = (self.args.lf * loss_latent_face
                       + self.args.ll * loss_latent_lower
                       + self.args.lh * loss_latent_hands
                       + self.args.lu * loss_latent_upper)
        self.tracker.update_meter("latent", "train", loss_latent.item())
        g_loss_final += loss_latent / 6

        # ═══════  3. Contrastive losses  ═══════
        g_loss_final += net_out["hubert_cons_loss"] + net_out["beat_cons_loss"]
        self.tracker.update_meter("hubert_cons", "train", net_out["hubert_cons_loss"].item())
        self.tracker.update_meter("beat_cons", "train", net_out["beat_cons_loss"].item())

        self.now_epoch += 1

        # ═══════  4. RVQ classification loss  ═══════
        loss_cls = 0
        tar_face_top  = loaded_data["tar_index_value_face_top"].reshape(-1, 6)
        tar_upper_top = loaded_data["tar_index_value_upper_top"].reshape(-1, 6)
        tar_lower_top = loaded_data["tar_index_value_lower_top"].reshape(-1, 6)
        tar_hands_top = loaded_data["tar_index_value_hands_top"].reshape(-1, 6)

        for i in range(6):
            idx_f = self.log_softmax(net_out["cls_face"][:, :, :, i]).reshape(-1, self.args.vae_codebook_size)
            idx_u = self.log_softmax(net_out["cls_upper"][:, :, :, i]).reshape(-1, self.args.vae_codebook_size)
            idx_l = self.log_softmax(net_out["cls_lower"][:, :, :, i]).reshape(-1, self.args.vae_codebook_size)
            idx_h = self.log_softmax(net_out["cls_hands"][:, :, :, i]).reshape(-1, self.args.vae_codebook_size)
            loss_cls_i = (self.args.cf * self.cls_loss(idx_f, tar_face_top[:, i])
                          + self.args.cu * self.cls_loss(idx_u, tar_upper_top[:, i])
                          + self.args.cl * self.cls_loss(idx_l, tar_lower_top[:, i])
                          + self.args.ch * self.cls_loss(idx_h, tar_hands_top[:, i]))
            loss_cls += loss_cls_i / (i + 1)

        self.tracker.update_meter("cls_full", "train", loss_cls.item())
        g_loss_final += loss_cls

        # ═══════  5. Masked gesture modeling (self-supervised) ═══════
        if mode == "train":
            if epoch < 130:
                mask_ratio = (epoch / 400) * 0.95 + 0.05
            else:
                mask_ratio = 0.35875

            mask = torch.rand(bs, n, self.args.pose_dims + 3 + 4) < mask_ratio
            mask = mask.float().cuda()

            net_out_self = self.model(
                loaded_data["beat"],
                loaded_data["in_word"],
                mask=mask,
                in_id=loaded_data["tar_id"],
                in_motion=loaded_data["latent_all"],
                use_attentions=True,
                use_word=False,
                hubert=loaded_data["hubert"],
                is_train=True,
                target_latents=target_latents,
            )

            # FM loss (self-supervised masked variant)
            g_loss_final += net_out_self["fm_loss"] * self.fm_loss_weight

            # latent rec
            loss_lat_self = (
                self.args.lf * self.reclatent_loss(net_out_self["rec_face"], loaded_data["zq_face"])
                + self.args.ll * self.reclatent_loss(net_out_self["rec_lower"], loaded_data["zq_lower"])
                + self.args.lh * self.reclatent_loss(net_out_self["rec_hands"], loaded_data["zq_hands"])
                + self.args.lu * self.reclatent_loss(net_out_self["rec_upper"], loaded_data["zq_upper"])
            )
            self.tracker.update_meter("latent_self", "train", loss_lat_self.item())
            g_loss_final += loss_lat_self / 6

            # cls loss
            idx_loss_self = 0
            for j_idx in range(6):
                idx_f_s = self.log_softmax(net_out_self["cls_face"][:, :, :, j_idx]).reshape(-1, self.args.vae_codebook_size)
                idx_u_s = self.log_softmax(net_out_self["cls_upper"][:, :, :, j_idx]).reshape(-1, self.args.vae_codebook_size)
                idx_l_s = self.log_softmax(net_out_self["cls_lower"][:, :, :, j_idx]).reshape(-1, self.args.vae_codebook_size)
                idx_h_s = self.log_softmax(net_out_self["cls_hands"][:, :, :, j_idx]).reshape(-1, self.args.vae_codebook_size)
                idx_loss_self_i = (
                    self.cls_loss(idx_f_s, tar_face_top[:, j_idx])
                    + self.cls_loss(idx_u_s, tar_upper_top[:, j_idx])
                    + self.cls_loss(idx_l_s, tar_lower_top[:, j_idx])
                    + self.cls_loss(idx_h_s, tar_hands_top[:, j_idx])
                )
                idx_loss_self += idx_loss_self_i / (j_idx + 1)
            self.tracker.update_meter("cls_self", "train", idx_loss_self.item())
            g_loss_final += idx_loss_self

            # ---- word-augmented masked variant ----
            net_out_word = self.model(
                loaded_data["beat"],
                loaded_data["in_word"],
                mask=mask,
                in_id=loaded_data["tar_id"],
                in_motion=loaded_data["latent_all"],
                use_attentions=True,
                use_word=True,
                hubert=loaded_data["hubert"],
                is_train=True,
                target_latents=target_latents,
            )
            g_loss_final += net_out_word["fm_loss"] * self.fm_loss_weight

            loss_lat_word = (
                self.args.lf * self.reclatent_loss(net_out_word["rec_face"], loaded_data["zq_face"])
                + self.args.ll * self.reclatent_loss(net_out_word["rec_lower"], loaded_data["zq_lower"])
                + self.args.lh * self.reclatent_loss(net_out_word["rec_hands"], loaded_data["zq_hands"])
                + self.args.lu * self.reclatent_loss(net_out_word["rec_upper"], loaded_data["zq_upper"])
            )
            self.tracker.update_meter("latent_word", "train", loss_lat_word.item())
            g_loss_final += loss_lat_word / 6

            idx_loss_word = 0
            for i in range(6):
                idx_f_w = self.log_softmax(net_out_word["cls_face"][:, :, :, i]).reshape(-1, self.args.vae_codebook_size)
                idx_u_w = self.log_softmax(net_out_word["cls_upper"][:, :, :, i]).reshape(-1, self.args.vae_codebook_size)
                idx_l_w = self.log_softmax(net_out_word["cls_lower"][:, :, :, i]).reshape(-1, self.args.vae_codebook_size)
                idx_h_w = self.log_softmax(net_out_word["cls_hands"][:, :, :, i]).reshape(-1, self.args.vae_codebook_size)
                idx_loss_word_i = (
                    self.cls_loss(idx_f_w, tar_face_top[:, i])
                    + self.cls_loss(idx_u_w, tar_upper_top[:, i])
                    + self.cls_loss(idx_l_w, tar_lower_top[:, i])
                    + self.cls_loss(idx_h_w, tar_hands_top[:, i])
                )
                idx_loss_word += idx_loss_word_i / (i + 1)
            self.tracker.update_meter("cls_word", "train", idx_loss_word.item())
            g_loss_final += idx_loss_word

        if mode == "train":
            return g_loss_final
        else:
            raise NotImplementedError("Only 'train' mode supported here.")

    # ------------------------------------------------------------------
    # test step  (ODE-based generation)
    # ------------------------------------------------------------------

    def _g_test(self, loaded_data):
        bs, n, j = (loaded_data["tar_pose"].shape[0],
                    loaded_data["tar_pose"].shape[1],
                    self.joints)
        tar_pose    = loaded_data["tar_pose"]
        tar_beta    = loaded_data["tar_beta"]
        in_word     = loaded_data["in_word"]
        tar_exps    = loaded_data["tar_exps"]
        tar_contact = loaded_data["tar_contact"]
        in_audio    = loaded_data["in_audio"]
        tar_trans   = loaded_data["tar_trans"]
        hubert      = loaded_data["hubert"]
        beat        = loaded_data["beat"]

        remain = n % 8
        if remain != 0:
            tar_pose    = tar_pose[:, :-remain, :]
            tar_beta    = tar_beta[:, :-remain, :]
            tar_trans   = tar_trans[:, :-remain, :]
            in_word     = in_word[:, :-remain]
            tar_exps    = tar_exps[:, :-remain, :]
            tar_contact = tar_contact[:, :-remain, :]
            n -= remain

        # --- prepare rot6d targets (same as original) ---
        tar_pose_jaw = rc.axis_angle_to_matrix(tar_pose[:, :, 66:69].reshape(bs, n, 1, 3))
        tar_pose_jaw = rc.matrix_to_rotation_6d(tar_pose_jaw).reshape(bs, n, 1 * 6)
        tar_pose_face = torch.cat([tar_pose_jaw, tar_exps], dim=2)

        tar_pose_hands = tar_pose[:, :, 25*3:55*3]
        tar_pose_hands = rc.axis_angle_to_matrix(tar_pose_hands.reshape(bs, n, 30, 3))
        tar_pose_hands = rc.matrix_to_rotation_6d(tar_pose_hands).reshape(bs, n, 30 * 6)

        tar_pose_upper = tar_pose[:, :, self.joint_mask_upper.astype(bool)]
        tar_pose_upper = rc.axis_angle_to_matrix(tar_pose_upper.reshape(bs, n, 13, 3))
        tar_pose_upper = rc.matrix_to_rotation_6d(tar_pose_upper).reshape(bs, n, 13 * 6)

        tar_pose_leg = tar_pose[:, :, self.joint_mask_lower.astype(bool)]
        tar_pose_leg = rc.axis_angle_to_matrix(tar_pose_leg.reshape(bs, n, 9, 3))
        tar_pose_leg = rc.matrix_to_rotation_6d(tar_pose_leg).reshape(bs, n, 9 * 6)
        tar_pose_lower = torch.cat([tar_pose_leg, tar_trans, tar_contact], dim=2)

        tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, 55, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, 55 * 6)
        latent_all = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)

        rec_index_all_face  = []
        rec_index_all_upper = []
        rec_index_all_lower = []
        rec_index_all_hands = []

        roundt = (n - self.args.pre_frames) // (self.args.pose_length - self.args.pre_frames)
        round_l = self.args.pose_length - self.args.pre_frames

        for i in range(roundt):
            in_word_tmp = in_word[:, i * round_l:(i + 1) * round_l + self.args.pre_frames]
            in_beat_tmp = beat[:, i * round_l:(i + 1) * round_l + self.args.pre_frames]
            in_id_tmp   = loaded_data["tar_id"][:, i * round_l:(i + 1) * round_l + self.args.pre_frames]
            hubert_tmp  = hubert[:, i * round_l:(i + 1) * round_l + self.args.pre_frames]

            mask_val = torch.ones(bs, self.args.pose_length, self.args.pose_dims + 3 + 4).float().cuda()
            mask_val[:, :self.args.pre_frames, :] = 0.0

            if i == 0:
                latent_all_tmp = latent_all[:, i * round_l:(i + 1) * round_l + self.args.pre_frames, :]
            else:
                latent_all_tmp = latent_all[:, i * round_l:(i + 1) * round_l + self.args.pre_frames, :]
                latent_all_tmp[:, :self.args.pre_frames, :] = latent_last[:, -self.args.pre_frames:, :]

            # --- generation via forward (no targets → uses ODE internally) ---
            net_out = self.model(
                in_audio=in_beat_tmp,
                in_word=in_word_tmp,
                mask=mask_val,
                in_motion=latent_all_tmp,
                in_id=in_id_tmp,
                hubert=hubert_tmp,
                use_attentions=True,
                is_train=False,
                target_latents=None,   # triggers ODE generation
            )

            # decode VQ indices via argmax on cls logits
            rec_index_upper = self.log_softmax(net_out["cls_upper"]).reshape(-1, self.args.vae_codebook_size, 6)
            _, rec_index_upper = torch.max(rec_index_upper.reshape(-1, 16, self.args.vae_codebook_size, 6), dim=2)

            rec_index_lower = self.log_softmax(net_out["cls_lower"]).reshape(-1, self.args.vae_codebook_size, 6)
            _, rec_index_lower = torch.max(rec_index_lower.reshape(-1, 16, self.args.vae_codebook_size, 6), dim=2)

            rec_index_hands = self.log_softmax(net_out["cls_hands"]).reshape(-1, self.args.vae_codebook_size, 6)
            _, rec_index_hands = torch.max(rec_index_hands.reshape(-1, 16, self.args.vae_codebook_size, 6), dim=2)

            rec_index_face = self.log_softmax(net_out["cls_face"]).reshape(-1, self.args.vae_codebook_size, 6)
            _, rec_index_face = torch.max(rec_index_face.reshape(-1, 16, self.args.vae_codebook_size, 6), dim=2)

            if i == 0:
                rec_index_all_face.append(rec_index_face)
                rec_index_all_upper.append(rec_index_upper)
                rec_index_all_lower.append(rec_index_lower)
                rec_index_all_hands.append(rec_index_hands)
            else:
                rec_index_all_face.append(rec_index_face[:, 1:])
                rec_index_all_upper.append(rec_index_upper[:, 1:])
                rec_index_all_lower.append(rec_index_lower[:, 1:])
                rec_index_all_hands.append(rec_index_hands[:, 1:])

            # decode for autoregressive seed update
            rec_upper_last = self.vq_model_upper.decode(rec_index_upper)
            rec_lower_last = self.vq_model_lower.decode(rec_index_lower)
            rec_hands_last = self.vq_model_hands.decode(rec_index_hands)

            rec_pose_legs = rec_lower_last[:, :, :54]
            bs_l, n_l = rec_pose_legs.shape[0], rec_pose_legs.shape[1]
            rec_pose_upper_6d = rec_upper_last.reshape(bs_l, n_l, 13, 6)
            rec_pose_upper_m  = rc.rotation_6d_to_matrix(rec_pose_upper_6d)
            rec_pose_upper_aa = rc.matrix_to_axis_angle(rec_pose_upper_m).reshape(bs_l * n_l, 13 * 3)
            rec_pose_upper_r  = self.inverse_selection_tensor(rec_pose_upper_aa, self.joint_mask_upper, bs_l * n_l)

            rec_pose_lower_6d = rec_pose_legs.reshape(bs_l, n_l, 9, 6)
            rec_pose_lower_m  = rc.rotation_6d_to_matrix(rec_pose_lower_6d)
            rec_pose_lower_aa = rc.matrix_to_axis_angle(rec_pose_lower_m).reshape(bs_l * n_l, 9 * 3)
            rec_pose_lower_r  = self.inverse_selection_tensor(rec_pose_lower_aa, self.joint_mask_lower, bs_l * n_l)

            rec_pose_hands_6d = rec_hands_last.reshape(bs_l, n_l, 30, 6)
            rec_pose_hands_m  = rc.rotation_6d_to_matrix(rec_pose_hands_6d)
            rec_pose_hands_aa = rc.matrix_to_axis_angle(rec_pose_hands_m).reshape(bs_l * n_l, 30 * 3)
            rec_pose_hands_r  = self.inverse_selection_tensor(rec_pose_hands_aa, self.joint_mask_hands, bs_l * n_l)

            rec_pose_comb = rec_pose_upper_r + rec_pose_lower_r + rec_pose_hands_r
            rec_pose_comb = rc.axis_angle_to_matrix(rec_pose_comb.reshape(bs_l, n_l, j, 3))
            rec_pose_comb = rc.matrix_to_rotation_6d(rec_pose_comb).reshape(bs_l, n_l, j * 6)

            rec_trans_v_s = rec_lower_last[:, :, 54:57]
            rec_x_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 0:1], 1 / self.args.pose_fps, tar_trans[:, 0, 0:1])
            rec_z_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 2:3], 1 / self.args.pose_fps, tar_trans[:, 0, 2:3])
            rec_y_trans = rec_trans_v_s[:, :, 1:2]
            rec_trans   = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
            latent_last = torch.cat([rec_pose_comb, rec_trans, rec_lower_last[:, :, 57:61]], dim=-1)

        # ---- final decode ----
        rec_index_face  = torch.cat(rec_index_all_face, dim=1)
        rec_index_upper = torch.cat(rec_index_all_upper, dim=1)
        rec_index_lower = torch.cat(rec_index_all_lower, dim=1)
        rec_index_hands = torch.cat(rec_index_all_hands, dim=1)

        rec_upper = self.vq_model_upper.decode(rec_index_upper)
        rec_lower = self.vq_model_lower.decode(rec_index_lower)
        rec_hands = self.vq_model_hands.decode(rec_index_hands)
        rec_face  = self.vq_model_face.decode(rec_index_face)

        rec_exps     = rec_face[:, :, 6:]
        rec_pose_jaw = rec_face[:, :, :6]
        rec_pose_legs = rec_lower[:, :, :54]
        bs, n = rec_pose_jaw.shape[0], rec_pose_jaw.shape[1]

        rec_pose_upper = rc.rotation_6d_to_matrix(rec_upper.reshape(bs, n, 13, 6))
        rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs * n, 13 * 3)
        rec_pose_upper_recover = self.inverse_selection_tensor(rec_pose_upper, self.joint_mask_upper, bs * n)

        rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_legs.reshape(bs, n, 9, 6))
        rec_lower2global = rc.matrix_to_rotation_6d(rec_pose_lower.clone()).reshape(bs, n, 9 * 6)
        rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs * n, 9 * 3)
        rec_pose_lower_recover = self.inverse_selection_tensor(rec_pose_lower, self.joint_mask_lower, bs * n)

        rec_pose_hands = rc.rotation_6d_to_matrix(rec_hands.reshape(bs, n, 30, 6))
        rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs * n, 30 * 3)
        rec_pose_hands_recover = self.inverse_selection_tensor(rec_pose_hands, self.joint_mask_hands, bs * n)

        rec_pose_jaw_aa = rc.matrix_to_axis_angle(
            rc.rotation_6d_to_matrix(rec_pose_jaw.reshape(bs * n, 6))
        ).reshape(bs * n, 1 * 3)

        rec_pose = rec_pose_upper_recover + rec_pose_lower_recover + rec_pose_hands_recover
        rec_pose[:, 66:69] = rec_pose_jaw_aa

        to_global = rec_lower.clone()
        to_global[:, :, 54:57] = 0.0
        to_global[:, :, :54] = rec_lower2global
        rec_global = self.global_motion(to_global)

        rec_trans_v_s = rec_global["rec_pose"][:, :, 54:57]
        rec_x_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 0:1], 1 / self.args.pose_fps, tar_trans[:, 0, 0:1])
        rec_z_trans = other_tools.velocity2position(rec_trans_v_s[:, :, 2:3], 1 / self.args.pose_fps, tar_trans[:, 0, 2:3])
        rec_y_trans = rec_trans_v_s[:, :, 1:2]
        rec_trans = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)

        tar_pose  = tar_pose[:, :n, :]
        tar_exps  = tar_exps[:, :n, :]
        tar_trans = tar_trans[:, :n, :]
        tar_beta  = tar_beta[:, :n, :]

        rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs * n, j, 3))
        rec_pose = rc.matrix_to_rotation_6d(rec_pose).reshape(bs, n, j * 6)
        tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs * n, j, 3))
        tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j * 6)

        return {
            "rec_pose":  rec_pose,
            "rec_trans": rec_trans,
            "tar_pose":  tar_pose,
            "tar_exps":  tar_exps,
            "tar_beta":  tar_beta,
            "tar_trans": tar_trans,
            "rec_exps":  rec_exps,
        }

    # ------------------------------------------------------------------
    # train / test loops  (same structure as original base trainer)
    # ------------------------------------------------------------------

    def train(self, epoch):
        use_adv = bool(epoch >= self.args.no_adv_epoch) if hasattr(self.args, "no_adv_epoch") else False
        self.model.train()
        t_start = time.time()
        self.tracker.reset()

        for its, batch_data in enumerate(self.train_loader):
            loaded_data = self._load_data(batch_data)
            t_data = time.time() - t_start
            self.opt.zero_grad()

            g_loss_final = self._g_training(loaded_data, use_adv, "train", epoch)
            g_loss_final.backward()

            if self.args.grad_norm != 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()

            mem_cost = torch.cuda.memory_cached() / 1e9
            lr_g = self.opt.param_groups[0]["lr"]
            t_train = time.time() - t_start - t_data
            t_start = time.time()

            if its % self.args.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)
            if self.args.debug:
                if its == 1:
                    break

        self.opt_s.step(epoch)

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

        with torch.no_grad():
            for its, batch_data in enumerate(self.test_loader):
                loaded_data = self._load_data(batch_data)
                net_out = self._g_test(loaded_data)

                tar_pose  = net_out["tar_pose"]
                rec_pose  = net_out["rec_pose"]
                tar_exps  = net_out["tar_exps"]
                tar_beta  = net_out["tar_beta"]
                rec_trans = net_out["rec_trans"]
                tar_trans = net_out["tar_trans"]
                rec_exps  = net_out["rec_exps"]
                bs, n, j_dim = tar_pose.shape[0], tar_pose.shape[1], self.joints

                if (30 / self.args.pose_fps) != 1:
                    assert 30 % self.args.pose_fps == 0
                    n *= int(30 / self.args.pose_fps)
                    tar_pose = F.interpolate(tar_pose.permute(0, 2, 1), scale_factor=30 / self.args.pose_fps, mode="linear").permute(0, 2, 1)
                    rec_pose = F.interpolate(rec_pose.permute(0, 2, 1), scale_factor=30 / self.args.pose_fps, mode="linear").permute(0, 2, 1)

                rec_pose_m = rc.rotation_6d_to_matrix(rec_pose.reshape(bs * n, j_dim, 6))
                rec_pose   = rc.matrix_to_rotation_6d(rec_pose_m).reshape(bs, n, j_dim * 6)
                tar_pose_m = rc.rotation_6d_to_matrix(tar_pose.reshape(bs * n, j_dim, 6))
                tar_pose   = rc.matrix_to_rotation_6d(tar_pose_m).reshape(bs, n, j_dim * 6)

                remain = n % self.args.vae_test_len
                latent_out.append(self.eval_copy.map2latent(rec_pose[:, :n - remain]).reshape(-1, self.args.vae_length).detach().cpu().numpy())
                latent_ori.append(self.eval_copy.map2latent(tar_pose[:, :n - remain]).reshape(-1, self.args.vae_length).detach().cpu().numpy())

                rec_pose_aa = rc.matrix_to_axis_angle(rc.rotation_6d_to_matrix(rec_pose.reshape(bs * n, j_dim, 6))).reshape(bs * n, j_dim * 3)
                tar_pose_aa = rc.matrix_to_axis_angle(rc.rotation_6d_to_matrix(tar_pose.reshape(bs * n, j_dim, 6))).reshape(bs * n, j_dim * 3)

                vertices_rec = self.smplx(
                    betas=tar_beta.reshape(bs * n, 300),
                    transl=rec_trans.reshape(bs * n, 3) - rec_trans.reshape(bs * n, 3),
                    expression=tar_exps.reshape(bs * n, 100) - tar_exps.reshape(bs * n, 100),
                    jaw_pose=rec_pose_aa[:, 66:69],
                    global_orient=rec_pose_aa[:, :3],
                    body_pose=rec_pose_aa[:, 3:21 * 3 + 3],
                    left_hand_pose=rec_pose_aa[:, 25 * 3:40 * 3],
                    right_hand_pose=rec_pose_aa[:, 40 * 3:55 * 3],
                    return_joints=True,
                    leye_pose=rec_pose_aa[:, 69:72],
                    reye_pose=rec_pose_aa[:, 72:75],
                )

                vertices_rec_face = self.smplx(
                    betas=tar_beta.reshape(bs * n, 300),
                    transl=rec_trans.reshape(bs * n, 3) - rec_trans.reshape(bs * n, 3),
                    expression=rec_exps.reshape(bs * n, 100),
                    jaw_pose=rec_pose_aa[:, 66:69],
                    global_orient=rec_pose_aa[:, :3] - rec_pose_aa[:, :3],
                    body_pose=rec_pose_aa[:, 3:21 * 3 + 3] - rec_pose_aa[:, 3:21 * 3 + 3],
                    left_hand_pose=rec_pose_aa[:, 25 * 3:40 * 3] - rec_pose_aa[:, 25 * 3:40 * 3],
                    right_hand_pose=rec_pose_aa[:, 40 * 3:55 * 3] - rec_pose_aa[:, 40 * 3:55 * 3],
                    return_verts=True,
                    return_joints=True,
                    leye_pose=rec_pose_aa[:, 69:72] - rec_pose_aa[:, 69:72],
                    reye_pose=rec_pose_aa[:, 72:75] - rec_pose_aa[:, 72:75],
                )
                vertices_tar_face = self.smplx(
                    betas=tar_beta.reshape(bs * n, 300),
                    transl=tar_trans.reshape(bs * n, 3) - tar_trans.reshape(bs * n, 3),
                    expression=tar_exps.reshape(bs * n, 100),
                    jaw_pose=tar_pose_aa[:, 66:69],
                    global_orient=tar_pose_aa[:, :3] - tar_pose_aa[:, :3],
                    body_pose=tar_pose_aa[:, 3:21 * 3 + 3] - tar_pose_aa[:, 3:21 * 3 + 3],
                    left_hand_pose=tar_pose_aa[:, 25 * 3:40 * 3] - tar_pose_aa[:, 25 * 3:40 * 3],
                    right_hand_pose=tar_pose_aa[:, 40 * 3:55 * 3] - tar_pose_aa[:, 40 * 3:55 * 3],
                    return_verts=True,
                    return_joints=True,
                    leye_pose=tar_pose_aa[:, 69:72] - tar_pose_aa[:, 69:72],
                    reye_pose=tar_pose_aa[:, 72:75] - tar_pose_aa[:, 72:75],
                )

                joints_rec  = vertices_rec["joints"].detach().cpu().numpy().reshape(1, n, 127 * 3)[0, :n, :55 * 3]
                facial_rec  = vertices_rec_face["vertices"].reshape(1, n, -1)[0, :n]
                facial_tar  = vertices_tar_face["vertices"].reshape(1, n, -1)[0, :n]
                face_vel_loss = self.vel_loss(facial_rec[1:, :] - facial_tar[:-1, :], facial_tar[1:, :] - facial_tar[:-1, :])
                l2 = self.reclatent_loss(facial_rec, facial_tar)
                l2_all += l2.item() * n
                lvel += face_vel_loss.item() * n

                _ = self.l1_calculator.run(joints_rec)
                if self.alignmenter is not None:
                    in_audio_eval, sr = librosa.load(self.args.data_path + "wave16k/" + test_seq_list.iloc[its]["id"] + ".wav")
                    in_audio_eval = librosa.resample(in_audio_eval, orig_sr=sr, target_sr=self.args.audio_sr)
                    a_offset = int(self.align_mask * (self.args.audio_sr / self.args.pose_fps))
                    onset_bt = self.alignmenter.load_audio(in_audio_eval[:int(self.args.audio_sr / self.args.pose_fps * n)], a_offset, len(in_audio_eval) - a_offset, True)
                    beat_vel = self.alignmenter.load_pose(joints_rec, self.align_mask, n - self.align_mask, 30, True)
                    align += self.alignmenter.calculate_align(onset_bt, beat_vel, 30) * (n - 2 * self.align_mask)

                tar_pose_np  = tar_pose_aa.detach().cpu().numpy()
                rec_pose_np  = rec_pose_aa.detach().cpu().numpy()
                rec_trans_np = rec_trans.detach().cpu().numpy().reshape(bs * n, 3)
                rec_exp_np   = rec_exps.detach().cpu().numpy().reshape(bs * n, 100)
                tar_exp_np   = tar_exps.detach().cpu().numpy().reshape(bs * n, 100)
                tar_trans_np = tar_trans.detach().cpu().numpy().reshape(bs * n, 3)

                gt_npz = np.load(self.args.data_path + self.args.pose_rep + "/" + test_seq_list.iloc[its]["id"] + ".npz", allow_pickle=True)
                np.savez(
                    results_save_path + "gt_" + test_seq_list.iloc[its]["id"] + ".npz",
                    betas=gt_npz["betas"], poses=tar_pose_np, expressions=tar_exp_np,
                    trans=tar_trans_np, model="smplx2020", gender="neutral", mocap_frame_rate=30,
                )
                np.savez(
                    results_save_path + "res_" + test_seq_list.iloc[its]["id"] + ".npz",
                    betas=gt_npz["betas"], poses=rec_pose_np, expressions=rec_exp_np,
                    trans=rec_trans_np, model="smplx2020", gender="neutral", mocap_frame_rate=30,
                )
                total_length += n

        logger.info(f"l2 loss: {l2_all / total_length}")
        logger.info(f"lvel loss: {lvel / total_length}")

        latent_out_all = np.concatenate(latent_out, axis=0)
        latent_ori_all = np.concatenate(latent_ori, axis=0)
        fid = data_tools.FIDCalculator.frechet_distance(latent_out_all, latent_ori_all)
        logger.info(f"fid score: {fid}")
        self.test_recording("fid", fid, epoch)

        align_avg = align / (total_length - 2 * len(self.test_loader) * self.align_mask)
        logger.info(f"align score: {align_avg}")
        self.test_recording("bc", align_avg, epoch)

        l1div = self.l1_calculator.avg()
        logger.info(f"l1div score: {l1div}")
        self.test_recording("l1div", l1div, epoch)

        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length / self.args.pose_fps)} s motion")
        return fid
