import librosa
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from matplotlib.pyplot import figure
import math
from scipy.signal import argrelextrema, hilbert


class L1div(object):
    def __init__(self):
        self.counter = 0
        self.sum = 0
    def run(self, results):
        self.counter += results.shape[0]
        mean = np.mean(results, 0)
        for i in range(results.shape[0]):
            results[i, :] = abs(results[i, :] - mean)
        sum_l1 = np.sum(results)
        self.sum += sum_l1
    def avg(self):
        return self.sum/self.counter
    def reset(self):
        self.counter = 0
        self.sum = 0
        

class SRGR(object):
    def __init__(self, threshold=0.1, joints=47):
        self.threshold = threshold
        self.pose_dimes = joints
        self.counter = 0
        self.sum = 0
        
    def run(self, results, targets, semantic):
        results = results.reshape(-1, self.pose_dimes, 3)
        targets = targets.reshape(-1, self.pose_dimes, 3)
        semantic = semantic.reshape(-1)
        diff = np.sum(abs(results-targets),2)
        success = np.where(diff<self.threshold, 1.0, 0.0)
        for i in range(success.shape[0]):
            # srgr == 0.165 when all success, scale range to [0, 1]
            success[i, :] *= semantic[i] * (1/0.165) 
        rate = np.sum(success)/(success.shape[0]*success.shape[1])
        self.counter += success.shape[0]
        self.sum += (rate*success.shape[0])
        return rate
    
    def avg(self):
        return self.sum/self.counter

class alignment(object):
    def __init__(self, sigma, order, mmae=None, upper_body=[3,6,9,12,13,14,15,16,17,18,19,20,21]):
        self.sigma = sigma
        self.order = order
        self.upper_body= upper_body
        # self.times = self.oenv = self.S = self.rms = None
        self.pose_data = []
        self.mmae = mmae
        self.threshold = 0.3
    
    def load_audio(self, wave, t_start=None, t_end=None, without_file=False, sr_audio=16000):
        hop_length = 512
        if without_file:
            y = wave
            sr = sr_audio
        else: y, sr = librosa.load(wave)
        if t_start is None:
            short_y = y
        else:
            short_y = y[t_start:t_end]
        # print(short_y.shape)
        onset_t = librosa.onset.onset_detect(y=short_y, sr=sr_audio, hop_length=hop_length, units='time')
        return onset_t

    def load_pose(self, pose, t_start, t_end, pose_fps, without_file=False):
        data_each_file = []
        if without_file:
            for line_data_np in pose: #,args.pre_frames, args.pose_length
                data_each_file.append(line_data_np)
                    #data_each_file.append(np.concatenate([line_data_np[9:18], line_data_np[75:84], ],0))
        else: 
            with open(pose, "r") as f:
                for i, line_data in enumerate(f.readlines()):
                    if i < 432: continue
                    line_data_np = np.fromstring(line_data, sep=" ",)
                    if pose_fps == 15:
                        if i % 2 == 0:
                            continue
                    data_each_file.append(np.concatenate([line_data_np[30:39], line_data_np[112:121], ],0))
                    
        data_each_file = np.array(data_each_file)
        #print(data_each_file.shape)
        
        joints = data_each_file.transpose(1, 0)
        dt = 1/pose_fps
        # first steps is forward diff (t+1 - t) / dt
        init_vel = (joints[:, 1:2] - joints[:, :1]) / dt
        # middle steps are second order (t+1 - t-1) / 2dt
        middle_vel = (joints[:, 2:] - joints[:, 0:-2]) / (2 * dt)
        # last step is backward diff (t - t-1) / dt
        final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
        #print(joints.shape, init_vel.shape, middle_vel.shape, final_vel.shape)
        vel = np.concatenate([init_vel, middle_vel, final_vel], 1).transpose(1, 0).reshape(data_each_file.shape[0], -1, 3)
        #print(vel.shape)
        #vel = data_each_file.reshape(data_each_file.shape[0], -1, 3)[1:] - data_each_file.reshape(data_each_file.shape[0], -1, 3)[:-1]
        vel = np.linalg.norm(vel, axis=2) / self.mmae
        
        beat_vel_all = []
        for i in range(vel.shape[1]):
            vel_mask = np.where(vel[:, i]>self.threshold)
            #print(vel.shape)
            #t_end = 80
            #vel[::2, :] -= 0.000001
            #print(vel[t_start:t_end, i], vel[t_start:t_end, i].shape)
            beat_vel = argrelextrema(vel[t_start:t_end, i], np.less, order=self.order) # n*47
            #print(beat_vel, t_start, t_end)
            beat_vel_list = []
            for j in beat_vel[0]:
                if j in vel_mask[0]:
                    beat_vel_list.append(j)
            beat_vel = np.array(beat_vel_list)
            beat_vel_all.append(beat_vel)
        #print(beat_vel_all)
        return beat_vel_all #beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist
    
    
    def load_data(self, wave, pose, t_start, t_end, pose_fps):
        onset_raw, onset_bt, onset_bt_rms = self.load_audio(wave, t_start, t_end)
        beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.load_pose(pose, t_start, t_end, pose_fps)
        return onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist 

    def eval_random_pose(self, wave, pose, t_start, t_end, pose_fps, num_random=60):
        onset_raw, onset_bt, onset_bt_rms = self.load_audio(wave, t_start, t_end)
        dur = t_end - t_start
        for i in range(num_random):
            beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist = self.load_pose(pose, i, i+dur, pose_fps)
            dis_all_b2a= self.calculate_align(onset_raw, onset_bt, onset_bt_rms, beat_right_arm, beat_right_shoulder, beat_right_wrist, beat_left_arm, beat_left_shoulder, beat_left_wrist)
            print(f"{i}s: ",dis_all_b2a)


    @staticmethod
    def plot_onsets(audio, sr, onset_times_1, onset_times_2):
        import librosa
        import librosa.display
        import matplotlib.pyplot as plt
        # Plot audio waveform
        fig, axarr = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Plot audio waveform in both subplots
        librosa.display.waveshow(audio, sr=sr, alpha=0.7, ax=axarr[0])
        librosa.display.waveshow(audio, sr=sr, alpha=0.7, ax=axarr[1])
        
        # Plot onsets from first method on the first subplot
        for onset in onset_times_1:
            axarr[0].axvline(onset, color='r', linestyle='--', alpha=0.9, label='Onset Method 1')
        axarr[0].legend()
        axarr[0].set(title='Onset Method 1', xlabel='', ylabel='Amplitude')
        
        # Plot onsets from second method on the second subplot
        for onset in onset_times_2:
            axarr[1].axvline(onset, color='b', linestyle='-', alpha=0.7, label='Onset Method 2')
        axarr[1].legend()
        axarr[1].set(title='Onset Method 2', xlabel='Time (s)', ylabel='Amplitude')
    
        
        # Add legend (eliminate duplicate labels)
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        
        # Show plot
        plt.title("Audio waveform with Onsets")
        plt.savefig("./onset.png", dpi=500)
    
    def audio_beat_vis(self, onset_raw, onset_bt, onset_bt_rms):
        figure(figsize=(24, 6), dpi=80)
        fig, ax = plt.subplots(nrows=4, sharex=True)
        librosa.display.specshow(librosa.amplitude_to_db(self.S, ref=np.max),
                                y_axis='log', x_axis='time', ax=ax[0])
        ax[0].label_outer()
        ax[1].plot(self.times, self.oenv, label='Onset strength')
        ax[1].vlines(librosa.frames_to_time(onset_raw), 0, self.oenv.max(), label='Raw onsets', color='r')
        ax[1].legend()
        ax[1].label_outer()

        ax[2].plot(self.times, self.oenv, label='Onset strength')
        ax[2].vlines(librosa.frames_to_time(onset_bt), 0, self.oenv.max(), label='Backtracked', color='r')
        ax[2].legend()
        ax[2].label_outer()

        ax[3].plot(self.times, self.rms[0], label='RMS')
        ax[3].vlines(librosa.frames_to_time(onset_bt_rms), 0, self.oenv.max(), label='Backtracked (RMS)', color='r')
        ax[3].legend()
        fig.savefig("./onset.png", dpi=500)
    
    @staticmethod
    def motion_frames2time(vel, offset, pose_fps):
        time_vel = vel/pose_fps + offset 
        return time_vel    
    
    @staticmethod
    def GAHR(a, b, sigma):
        dis_all_a2b = 0
        dis_all_b2a = 0
        for b_each in b:
            l2_min = np.inf
            for a_each in a:
                l2_dis = abs(a_each - b_each)
                if l2_dis < l2_min:
                    l2_min = l2_dis
            dis_all_b2a += math.exp(-(l2_min**2)/(2*sigma**2))
        dis_all_b2a /= len(b)
        return dis_all_b2a 
    
    @staticmethod
    def fix_directed_GAHR(a, b, sigma):
        a = alignment.motion_frames2time(a, 0, 30)
        b = alignment.motion_frames2time(b, 0, 30)
        t = len(a)/30
        a = [0] + a + [t]
        b = [0] + b + [t]
        dis_a2b = alignment.GAHR(a, b, sigma)
        return dis_a2b

    def calculate_align(self, onset_bt_rms, beat_vel, pose_fps=30):
        audio_bt = onset_bt_rms
        avg_dis_all_b2a_list = []
        for its, beat_vel_each in enumerate(beat_vel):
            if its not in self.upper_body:
                continue
            #print(beat_vel_each)
            #print(audio_bt.shape, beat_vel_each.shape)
            pose_bt = self.motion_frames2time(beat_vel_each, 0, pose_fps)
            #print(pose_bt)
            avg_dis_all_b2a_list.append(self.GAHR(pose_bt, audio_bt, self.sigma))
        # avg_dis_all_b2a = max(avg_dis_all_b2a_list)
        avg_dis_all_b2a = sum(avg_dis_all_b2a_list)/len(avg_dis_all_b2a_list) #max(avg_dis_all_b2a_list)
        #print(avg_dis_all_b2a, sum(avg_dis_all_b2a_list)/47)
        return avg_dis_all_b2a  


class CoupledSwingAnalyzer(object):
    """
    Coupled limb-dynamics analysis utilities for SMPL-X joint trajectories.

    Expected joints input shape: [T, J, 3] in world coordinates.
    """

    def __init__(self, fps=30, joint_map=None):
        self.fps = float(fps)
        self.dt = 1.0 / max(self.fps, 1.0)
        self.joint_map = joint_map or {
            "pelvis": 0,
            "neck": 12,
            "l_shoulder": 16,
            "r_shoulder": 17,
            "l_elbow": 18,
            "r_elbow": 19,
            "l_wrist": 20,
            "r_wrist": 21,
        }

    @staticmethod
    def _safe_norm(x, axis=-1, keepdims=False, eps=1e-8):
        n = np.linalg.norm(x, axis=axis, keepdims=keepdims)
        if np.isscalar(n):
            return max(n, eps)
        return np.maximum(n, eps)

    @staticmethod
    def _normalize(x, axis=-1, eps=1e-8):
        return x / CoupledSwingAnalyzer._safe_norm(x, axis=axis, keepdims=True, eps=eps)

    @staticmethod
    def _moving_average(x, win):
        win = int(max(1, win))
        if win <= 1:
            return x.copy()
        kernel = np.ones(win, dtype=np.float64) / float(win)
        return np.convolve(x, kernel, mode="same")

    @staticmethod
    def _mutual_information_1d(x, y, bins=16):
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        n = min(len(x), len(y))
        if n < 4:
            return 0.0
        x = x[:n]
        y = y[:n]
        c_xy, _, _ = np.histogram2d(x, y, bins=bins)
        p_xy = c_xy / max(np.sum(c_xy), 1.0)
        p_x = np.sum(p_xy, axis=1, keepdims=True)
        p_y = np.sum(p_xy, axis=0, keepdims=True)
        denom = np.maximum(p_x @ p_y, 1e-12)
        nz = p_xy > 0
        mi = np.sum(p_xy[nz] * np.log(np.maximum(p_xy[nz], 1e-12) / denom[nz]))
        return float(mi)

    @staticmethod
    def _r2(y_true, y_pred):
        y_true = np.asarray(y_true).reshape(-1)
        y_pred = np.asarray(y_pred).reshape(-1)
        n = min(len(y_true), len(y_pred))
        if n < 2:
            return 0.0
        y_true = y_true[:n]
        y_pred = y_pred[:n]
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        if ss_tot <= 1e-12:
            return 0.0
        return 1.0 - ss_res / ss_tot

    @staticmethod
    def _fit_linear(X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim == 1:
            X = X[:, None]
        Xb = np.concatenate([X, np.ones((X.shape[0], 1), dtype=np.float64)], axis=1)
        w, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
        y_hat = Xb @ w
        return w, y_hat

    def _local_frames(self, joints):
        joints = np.asarray(joints)
        pelvis = joints[:, self.joint_map["pelvis"]]
        neck = joints[:, self.joint_map["neck"]]
        l_sh = joints[:, self.joint_map["l_shoulder"]]
        r_sh = joints[:, self.joint_map["r_shoulder"]]

        x_axis = self._normalize(r_sh - l_sh)
        y_axis = self._normalize(neck - pelvis)
        z_axis = self._normalize(np.cross(x_axis, y_axis))
        y_axis = self._normalize(np.cross(z_axis, x_axis))
        return np.stack([x_axis, y_axis, z_axis], axis=-1)  # [T,3,3]

    def _segment_dir_local(self, joints, parent_idx, child_idx):
        joints = np.asarray(joints)
        R = self._local_frames(joints)
        seg = joints[:, child_idx] - joints[:, parent_idx]
        seg = self._normalize(seg)
        # world -> local (R columns are local axes in world coords)
        seg_local = np.einsum("tij,tj->ti", np.transpose(R, (0, 2, 1)), seg)
        return self._normalize(seg_local)

    def segment_omega_local(self, joints, parent_idx, child_idx):
        d = self._segment_dir_local(joints, parent_idx, child_idx)
        if d.shape[0] < 2:
            return np.zeros((d.shape[0], 3), dtype=np.float64)
        omega = np.cross(d[:-1], d[1:]) / self.dt
        omega = np.concatenate([omega[:1], omega], axis=0)
        return omega

    def segment_angular_velocity_coherence(self, joints, side="right", max_lag_s=0.6):
        if side == "right":
            sh, el, wr = self.joint_map["r_shoulder"], self.joint_map["r_elbow"], self.joint_map["r_wrist"]
        else:
            sh, el, wr = self.joint_map["l_shoulder"], self.joint_map["l_elbow"], self.joint_map["l_wrist"]

        omg_upper = self.segment_omega_local(joints, sh, el)
        omg_fore = self.segment_omega_local(joints, el, wr)
        upper_mag = np.linalg.norm(omg_upper, axis=-1)
        fore_mag = np.linalg.norm(omg_fore, axis=-1)

        plv, lag_frames, lag_sec, lag_corr = self.phase_locking_and_lag(
            upper_mag, fore_mag, max_lag_s=max_lag_s
        )
        return {
            "plv": plv,
            "lead_lag_frames": lag_frames,
            "lead_lag_seconds": lag_sec,
            "lead_lag_corr": lag_corr,
            "upper_omega": upper_mag,
            "forearm_omega": fore_mag,
        }

    def phase_locking_and_lag(self, x, y, max_lag_s=0.6):
        x = np.asarray(x).reshape(-1)
        y = np.asarray(y).reshape(-1)
        n = min(len(x), len(y))
        if n < 8:
            return 0.0, 0, 0.0, 0.0
        x = x[:n] - np.mean(x[:n])
        y = y[:n] - np.mean(y[:n])

        phx = np.angle(hilbert(x))
        phy = np.angle(hilbert(y))
        plv = float(np.abs(np.mean(np.exp(1j * (phx - phy)))))

        max_lag = int(max(1, round(max_lag_s * self.fps)))
        xc = np.correlate(x, y, mode="full")
        lags = np.arange(-n + 1, n)
        mask = (lags >= -max_lag) & (lags <= max_lag)
        xc_sub = xc[mask]
        lags_sub = lags[mask]
        best = int(np.argmax(xc_sub))
        lag_frames = int(lags_sub[best])
        lag_sec = float(lag_frames * self.dt)
        denom = self._safe_norm(x) * self._safe_norm(y)
        lag_corr = float(xc_sub[best] / max(denom, 1e-12))
        return plv, lag_frames, lag_sec, lag_corr

    def _segment_angle_series(self, joints, parent_idx, child_idx):
        d_local = self._segment_dir_local(joints, parent_idx, child_idx)
        # angle to local +Y axis
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        cosv = np.clip(np.sum(d_local * y_axis[None, :], axis=-1), -1.0, 1.0)
        return np.arccos(cosv)

    def coupled_oscillator_fit(self, joints, side="right"):
        if side == "right":
            sh, el, wr = self.joint_map["r_shoulder"], self.joint_map["r_elbow"], self.joint_map["r_wrist"]
        else:
            sh, el, wr = self.joint_map["l_shoulder"], self.joint_map["l_elbow"], self.joint_map["l_wrist"]

        x1 = self._segment_angle_series(joints, sh, el)
        x2 = self._segment_angle_series(joints, el, wr)
        if len(x1) < 8:
            return {
                "r2_upper": 0.0,
                "r2_forearm": 0.0,
                "k_upper_to_forearm": 0.0,
                "k_forearm_to_upper": 0.0,
                "coupling_asymmetry": 0.0,
            }

        dx1 = np.gradient(x1, self.dt)
        dx2 = np.gradient(x2, self.dt)
        ddx1 = np.gradient(dx1, self.dt)
        ddx2 = np.gradient(dx2, self.dt)

        X1 = np.stack([dx1, x1, (x1 - x2)], axis=1)
        y1 = -ddx1
        w1, y1_hat = self._fit_linear(X1, y1)

        X2 = np.stack([dx2, x2, (x2 - x1)], axis=1)
        y2 = -ddx2
        w2, y2_hat = self._fit_linear(X2, y2)

        k12 = float(w1[2])
        k21 = float(w2[2])
        return {
            "r2_upper": self._r2(y1, y1_hat),
            "r2_forearm": self._r2(y2, y2_hat),
            "k_upper_to_forearm": k12,
            "k_forearm_to_upper": k21,
            "coupling_asymmetry": float(abs(k12 - k21)),
        }

    def cross_frequency_coupling(self, joints, side="right", lf_window=21, hf_window=5):
        if side == "right":
            sh, el, wr = self.joint_map["r_shoulder"], self.joint_map["r_elbow"], self.joint_map["r_wrist"]
        else:
            sh, el, wr = self.joint_map["l_shoulder"], self.joint_map["l_elbow"], self.joint_map["l_wrist"]

        shoulder = np.linalg.norm(self.segment_omega_local(joints, sh, el), axis=-1)
        forearm = np.linalg.norm(self.segment_omega_local(joints, el, wr), axis=-1)

        lf_shoulder = self._moving_average(shoulder, lf_window)
        hf_forearm = forearm - self._moving_average(forearm, hf_window)
        hf_env = np.abs(hilbert(hf_forearm))

        n = min(len(lf_shoulder), len(hf_env))
        if n < 8:
            return {
                "corr": 0.0,
                "mutual_info": 0.0,
            }
        x = lf_shoulder[:n]
        y = hf_env[:n]
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
        return {
            "corr": corr,
            "mutual_info": self._mutual_information_1d(x, y, bins=16),
        }

    def extract_predictor_signals(self, joints, side="right"):
        if side == "right":
            sh, el, wr = self.joint_map["r_shoulder"], self.joint_map["r_elbow"], self.joint_map["r_wrist"]
        else:
            sh, el, wr = self.joint_map["l_shoulder"], self.joint_map["l_elbow"], self.joint_map["l_wrist"]
        pelvis = self.joint_map["pelvis"]
        neck = self.joint_map["neck"]

        shoulder = np.linalg.norm(self.segment_omega_local(joints, sh, el), axis=-1)
        forearm = np.linalg.norm(self.segment_omega_local(joints, el, wr), axis=-1)
        torso = np.linalg.norm(self.segment_omega_local(joints, pelvis, neck), axis=-1)
        hand = forearm.copy()
        return {
            "shoulder": shoulder,
            "forearm": forearm,
            "torso": torso,
            "hand": hand,
        }

    @staticmethod
    def _make_lag_features(signals_dict, keys, max_lag):
        arrays = [np.asarray(signals_dict[k]).reshape(-1) for k in keys if k in signals_dict and signals_dict[k] is not None]
        if len(arrays) == 0:
            return None, None
        n = min([len(a) for a in arrays])
        if n <= max_lag + 2:
            return None, None
        feats = []
        for k in keys:
            if k not in signals_dict or signals_dict[k] is None:
                continue
            s = np.asarray(signals_dict[k]).reshape(-1)[:n]
            for lag in range(max_lag + 1):
                feats.append(s[max_lag - lag:n - lag])
        X = np.stack(feats, axis=1)
        y = np.asarray(signals_dict["hand"]).reshape(-1)[:n][max_lag:]
        return X, y

    def predictor_analysis(self, samples, max_lag=4):
        """
        samples: list of dicts with keys:
            speaker_id, shoulder, forearm, torso, hand, optional audio, optional gate
        """
        groups = {}
        for s in samples:
            sid = s.get("speaker_id", -1)
            groups.setdefault(sid, []).append(s)

        model_defs = {
            "shoulder_only": ["shoulder"],
            "forearm_only": ["forearm"],
            "shoulder_forearm_torso": ["shoulder", "forearm", "torso"],
            "plus_audio_gate": ["shoulder", "forearm", "torso", "audio", "gate"],
        }

        out = {}
        for model_name, keys in model_defs.items():
            fold_r2 = []
            fold_mi = []
            for sid in groups.keys():
                train = []
                test = []
                for sid2, seqs in groups.items():
                    if sid2 == sid:
                        test.extend(seqs)
                    else:
                        train.extend(seqs)

                Xtr_list, ytr_list = [], []
                Xte_list, yte_list = [], []

                for seq in train:
                    X, y = self._make_lag_features(seq, keys, max_lag)
                    if X is not None:
                        Xtr_list.append(X)
                        ytr_list.append(y)
                for seq in test:
                    X, y = self._make_lag_features(seq, keys, max_lag)
                    if X is not None:
                        Xte_list.append(X)
                        yte_list.append(y)

                if len(Xtr_list) == 0 or len(Xte_list) == 0:
                    continue

                Xtr = np.concatenate(Xtr_list, axis=0)
                ytr = np.concatenate(ytr_list, axis=0)
                Xte = np.concatenate(Xte_list, axis=0)
                yte = np.concatenate(yte_list, axis=0)

                w, _ = self._fit_linear(Xtr, ytr)
                Xteb = np.concatenate([Xte, np.ones((Xte.shape[0], 1), dtype=np.float64)], axis=1)
                yhat = Xteb @ w
                fold_r2.append(self._r2(yte, yhat))

                # MI proxy between aggregated predictor and target
                mi_proxy = self._mutual_information_1d(np.mean(Xte, axis=1), yte, bins=16)
                fold_mi.append(mi_proxy)

            out[model_name] = {
                "mean_r2": float(np.mean(fold_r2)) if len(fold_r2) > 0 else 0.0,
                "mean_mutual_info": float(np.mean(fold_mi)) if len(fold_mi) > 0 else 0.0,
                "num_folds": int(len(fold_r2)),
            }

        base_r2 = out.get("shoulder_only", {}).get("mean_r2", 0.0)
        for model_name in out.keys():
            out[model_name]["delta_r2_vs_shoulder"] = float(out[model_name]["mean_r2"] - base_r2)
        return out

    def audio_conditioned_causality(self, samples, max_lag=8):
        """
        Granger-style test with lagged linear models:
            restricted: hand ~ lag(shoulder, forearm)
            full:       hand ~ lag(audio, shoulder, forearm)
        """
        Xr_list, Xf_list, y_list = [], [], []
        for s in samples:
            if s.get("audio", None) is None:
                continue
            base = {
                "shoulder": s.get("shoulder", None),
                "forearm": s.get("forearm", None),
                "hand": s.get("hand", None),
            }
            full = {
                "audio": s.get("audio", None),
                "shoulder": s.get("shoulder", None),
                "forearm": s.get("forearm", None),
                "hand": s.get("hand", None),
            }
            Xr, y = self._make_lag_features(base, ["shoulder", "forearm"], max_lag)
            Xf, y2 = self._make_lag_features(full, ["audio", "shoulder", "forearm"], max_lag)
            if Xr is None or Xf is None:
                continue
            n = min(len(y), len(y2), Xr.shape[0], Xf.shape[0])
            Xr_list.append(Xr[:n])
            Xf_list.append(Xf[:n])
            y_list.append(y[:n])

        if len(Xr_list) == 0:
            return {
                "delta_r2_audio": 0.0,
                "f_stat": 0.0,
                "top_audio_lag_frames": 0,
                "top_audio_lag_seconds": 0.0,
            }

        Xr = np.concatenate(Xr_list, axis=0)
        Xf = np.concatenate(Xf_list, axis=0)
        y = np.concatenate(y_list, axis=0)

        wr, yhat_r = self._fit_linear(Xr, y)
        wf, yhat_f = self._fit_linear(Xf, y)

        r2_r = self._r2(y, yhat_r)
        r2_f = self._r2(y, yhat_f)
        delta_r2 = float(r2_f - r2_r)

        rss_r = float(np.sum((y - yhat_r) ** 2))
        rss_f = float(np.sum((y - yhat_f) ** 2))
        df1 = max(Xf.shape[1] - Xr.shape[1], 1)
        df2 = max(len(y) - Xf.shape[1] - 1, 1)
        f_stat = float(((rss_r - rss_f) / df1) / max(rss_f / df2, 1e-12))

        # Audio-lag attribution from full model coefficients.
        # First (max_lag+1) columns correspond to audio lags.
        audio_coef = np.abs(wf[:(max_lag + 1)])
        top_lag = int(np.argmax(audio_coef))
        return {
            "delta_r2_audio": delta_r2,
            "f_stat": f_stat,
            "top_audio_lag_frames": top_lag,
            "top_audio_lag_seconds": float(top_lag * self.dt),
        }

    def side_report(self, joints, side="right"):
        coh = self.segment_angular_velocity_coherence(joints, side=side)
        osc = self.coupled_oscillator_fit(joints, side=side)
        cfc = self.cross_frequency_coupling(joints, side=side)
        return {
            "coherence": {
                "plv": coh["plv"],
                "lead_lag_frames": coh["lead_lag_frames"],
                "lead_lag_seconds": coh["lead_lag_seconds"],
                "lead_lag_corr": coh["lead_lag_corr"],
            },
            "oscillator": osc,
            "cross_frequency": cfc,
        }

    def bilateral_oscillator_asymmetry(self, joints):
        left = self.coupled_oscillator_fit(joints, side="left")
        right = self.coupled_oscillator_fit(joints, side="right")
        return {
            "left": left,
            "right": right,
            "k12_asymmetry_lr": float(abs(left["k_upper_to_forearm"] - right["k_upper_to_forearm"])),
            "k21_asymmetry_lr": float(abs(left["k_forearm_to_upper"] - right["k_forearm_to_upper"])),
            "r2_gap_upper": float(abs(left["r2_upper"] - right["r2_upper"])),
            "r2_gap_forearm": float(abs(left["r2_forearm"] - right["r2_forearm"])),
        }