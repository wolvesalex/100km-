# train_gnn_imager.py
# -*- coding: utf-8 -*-
"""
Train UNet-based multi-stage refiner for D-region imaging on regular grid.

本版改进要点（针对：val早停过早、β漂移、远离路径区域不稳、样本间R²方差大）：
1) 将验证指标拆分为：
   - val_pred：只衡量预测误差（near-path混合 + global），不含任何正则/辅助项
   - val_total：训练总目标（包含 correction / prior 正则等）
   默认用 val_pred 做 best/scheduler/early-stopping，避免“正则项主导选择”。
2) 增加 EMA（Exponential Moving Average）权重：默认用 EMA 做验证与保存 best，提高泛化稳定性。
3) 增加 obs_coverage(wsum) 平滑（coverage_blur_ksize）：把 near-path 约束从“细线”变成“带状”，减少漂移与失败样本。
4) 修复 argparse：原 correction_nearpath_only 的 store_true + default=True 使其无法关闭。
   改为 --correction-scope {nearpath,all}。
5) 增加 AMP（autocast+GradScaler）与 grad-accum 选项，便于用更大有效batch。
6) 数据增强新增：每条路径的幅度/相位 bias（模拟标定偏置），提升鲁棒性。
"""

from __future__ import annotations

import os
import argparse
import json
import math
import random
from contextlib import contextmanager
import numpy as np
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C
from gnn_imager_models import DRegionImagerUNetRefiner, tv_loss_2d


# -------------------------
# Reproducibility
# -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------------
# Preprocess helpers
# -------------------------

def param_norm_stats_from_range(vmin: float, vmax: float) -> tuple[float, float]:
    mid = 0.5 * (vmin + vmax)
    half = 0.5 * (vmax - vmin) + 1e-6
    return float(mid), float(half)


def phase_to_sincos_deg(phase_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rad = phase_deg.astype(np.float32) * (math.pi / 180.0)
    return np.sin(rad).astype(np.float32), np.cos(rad).astype(np.float32)


def wrap_phase_delta_deg(delta_deg: np.ndarray) -> np.ndarray:
    dd = ((delta_deg + 180.0) % 360.0) - 180.0
    return dd.astype(np.float32)


def wrap_phase_deg(phase_deg: np.ndarray) -> np.ndarray:
    # wrap to (-180, 180]
    return (((phase_deg + 180.0) % 360.0) - 180.0).astype(np.float32)


def unwrap_phase_deg_to_rad(phase_deg: np.ndarray) -> np.ndarray:
    rad = np.deg2rad(phase_deg.astype(np.float32))
    return np.unwrap(rad, axis=1).astype(np.float32)


def transform_path_static(path_static_raw: np.ndarray, log_transform: bool) -> np.ndarray:
    """
    path_static_raw: (...,6) = [tx_lat, tx_lon, rx_lat, rx_lon, freq_hz, power_kw]
    对 freq/power 做 log10 变换可显著改善分布形态（即使后续标准化，也常常更稳）。
    """
    ps = np.array(path_static_raw, dtype=np.float32, copy=True)
    if log_transform:
        # freq_hz, power_kw should be positive
        ps[..., 4] = np.log10(np.maximum(ps[..., 4], 1e-6))
        ps[..., 5] = np.log10(np.maximum(ps[..., 5], 1e-6))
    return ps


def build_path_seq_features(
    amp_x_obs, pha_x_obs, amp_y_obs, pha_y_obs, amp_z_obs, pha_z_obs,
    dist_km: np.ndarray,
    amp_scale_db: float = 50.0,
    # optional prior-sim fields (same shape)
    amp_x_prior_sim=None, pha_x_prior_sim=None,
    amp_y_prior_sim=None, pha_y_prior_sim=None,
    amp_z_prior_sim=None, pha_z_prior_sim=None,
    # optional prior params along path (P,L)
    hp_prior_path=None, be_prior_path=None,
    h_mid: float = 0.0, h_half: float = 1.0,
    b_mid: float = 0.0, b_half: float = 1.0,
):
    """
    输出: (P,L,C)
    """
    amp_x_obs = amp_x_obs.astype(np.float32)
    amp_y_obs = amp_y_obs.astype(np.float32)
    amp_z_obs = amp_z_obs.astype(np.float32)
    pha_x_obs = pha_x_obs.astype(np.float32)
    pha_y_obs = pha_y_obs.astype(np.float32)
    pha_z_obs = pha_z_obs.astype(np.float32)

    P, L = amp_x_obs.shape
    dist_km = dist_km.astype(np.float32)
    dist_norm = dist_km / (dist_km[:, -1:] + 1e-6)

    def amp_pack(amp_db):
        a = amp_db / float(amp_scale_db)
        ac = a - a.mean(axis=1, keepdims=True)
        da = np.diff(a, axis=1, prepend=a[:, :1])
        return a.astype(np.float32), ac.astype(np.float32), da.astype(np.float32)

    def phase_pack(pha_deg):
        s, c = phase_to_sincos_deg(pha_deg)
        p = unwrap_phase_deg_to_rad(pha_deg)
        dp = np.diff(p, axis=1, prepend=p[:, :1])
        dp = np.clip(dp, -3.0, 3.0) / 3.0
        return s, c, dp.astype(np.float32)

    ax, axc, dax = amp_pack(amp_x_obs)
    ay, ayc, day = amp_pack(amp_y_obs)
    az, azc, daz = amp_pack(amp_z_obs)

    sx, cx, dpx = phase_pack(pha_x_obs)
    sy, cy, dpy = phase_pack(pha_y_obs)
    sz, cz, dpz = phase_pack(pha_z_obs)

    feats = [
        # obs absolute
        ax, axc, dax, sx, cx, dpx,
        ay, ayc, day, sy, cy, dpy,
        az, azc, daz, sz, cz, dpz,
        dist_norm.astype(np.float32),
    ]

    # residual features if prior-sim exists
    has_prior_sim = (amp_x_prior_sim is not None) and (pha_x_prior_sim is not None)
    if has_prior_sim:
        amp_x_prior_sim = amp_x_prior_sim.astype(np.float32)
        amp_y_prior_sim = amp_y_prior_sim.astype(np.float32)
        amp_z_prior_sim = amp_z_prior_sim.astype(np.float32)
        pha_x_prior_sim = pha_x_prior_sim.astype(np.float32)
        pha_y_prior_sim = pha_y_prior_sim.astype(np.float32)
        pha_z_prior_sim = pha_z_prior_sim.astype(np.float32)

        rx = (amp_x_obs - amp_x_prior_sim) / float(amp_scale_db)
        ry = (amp_y_obs - amp_y_prior_sim) / float(amp_scale_db)
        rz = (amp_z_obs - amp_z_prior_sim) / float(amp_scale_db)

        drx = np.diff(rx, axis=1, prepend=rx[:, :1])
        dry = np.diff(ry, axis=1, prepend=ry[:, :1])
        drz = np.diff(rz, axis=1, prepend=rz[:, :1])

        dphx = wrap_phase_delta_deg(pha_x_obs - pha_x_prior_sim)
        dphy = wrap_phase_delta_deg(pha_y_obs - pha_y_prior_sim)
        dphz = wrap_phase_delta_deg(pha_z_obs - pha_z_prior_sim)

        sdx, cdx = phase_to_sincos_deg(dphx)
        sdy, cdy = phase_to_sincos_deg(dphy)
        sdz, cdz = phase_to_sincos_deg(dphz)

        pxo = unwrap_phase_deg_to_rad(pha_x_obs)
        pxp = unwrap_phase_deg_to_rad(pha_x_prior_sim)
        rpx = pxo - pxp
        drpx = np.diff(rpx, axis=1, prepend=rpx[:, :1])
        drpx = np.clip(drpx, -3.0, 3.0) / 3.0

        pyo = unwrap_phase_deg_to_rad(pha_y_obs)
        pyp = unwrap_phase_deg_to_rad(pha_y_prior_sim)
        rpy = pyo - pyp
        drpy = np.diff(rpy, axis=1, prepend=rpy[:, :1])
        drpy = np.clip(drpy, -3.0, 3.0) / 3.0

        pzo = unwrap_phase_deg_to_rad(pha_z_obs)
        pzp = unwrap_phase_deg_to_rad(pha_z_prior_sim)
        rpz = pzo - pzp
        drpz = np.diff(rpz, axis=1, prepend=rpz[:, :1])
        drpz = np.clip(drpz, -3.0, 3.0) / 3.0

        feats += [
            rx.astype(np.float32), drx.astype(np.float32),
            ry.astype(np.float32), dry.astype(np.float32),
            rz.astype(np.float32), drz.astype(np.float32),
            sdx.astype(np.float32), cdx.astype(np.float32), drpx.astype(np.float32),
            sdy.astype(np.float32), cdy.astype(np.float32), drpy.astype(np.float32),
            sdz.astype(np.float32), cdz.astype(np.float32), drpz.astype(np.float32),
        ]

    # prior params along path
    if (hp_prior_path is not None) and (be_prior_path is not None):
        hp_prior_path = hp_prior_path.astype(np.float32)
        be_prior_path = be_prior_path.astype(np.float32)
        hp_n = (hp_prior_path - float(h_mid)) / (float(h_half) + 1e-6)
        be_n = (be_prior_path - float(b_mid)) / (float(b_half) + 1e-6)
        dhp = np.diff(hp_n, axis=1, prepend=hp_n[:, :1])
        dbe = np.diff(be_n, axis=1, prepend=be_n[:, :1])
        feats += [hp_n.astype(np.float32), dhp.astype(np.float32), be_n.astype(np.float32), dbe.astype(np.float32)]

    feats = np.stack(feats, axis=-1).astype(np.float32)
    return feats


# -------------------------
# EMA helper
# -------------------------

class EMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow = {}
        self._init_from(model)

    def _init_from(self, model: nn.Module):
        self.shadow = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        d = self.decay
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = p.detach().clone()
            else:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    def state_dict(self) -> dict:
        return {k: v.detach().cpu() for k, v in self.shadow.items()}

    def load_state_dict(self, sd: dict, device: torch.device):
        self.shadow = {k: v.to(device=device) for k, v in sd.items()}

    @contextmanager
    def apply_to(self, model: nn.Module):
        backup = {}
        try:
            for name, p in model.named_parameters():
                if p.requires_grad and name in self.shadow:
                    backup[name] = p.detach().clone()
                    p.data.copy_(self.shadow[name].data)
            yield
        finally:
            for name, p in model.named_parameters():
                if name in backup:
                    p.data.copy_(backup[name].data)


# -------------------------
# Dataset loader
# -------------------------

class VLFH5GridDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        h5_path: str,
        split: str,
        train_frac: float = 0.9,
        seed: int = 42,
        amp_scale_db: float = 50.0,
        amp_noise_db: float = 0.0,
        phase_noise_deg: float = 0.0,
        path_drop_prob: float = 0.0,
        amp_bias_db: float = 0.0,
        phase_bias_deg: float = 0.0,
        static_log_transform: bool = False,
    ):
        super().__init__()
        self.h5_path = h5_path
        self.split = split
        self.amp_scale_db = float(amp_scale_db)

        self.amp_noise_db = float(amp_noise_db) if split == "train" else 0.0
        self.phase_noise_deg = float(phase_noise_deg) if split == "train" else 0.0
        self.path_drop_prob = float(path_drop_prob) if split == "train" else 0.0

        # new: per-path bias augmentation (train only)
        self.amp_bias_db = float(amp_bias_db) if split == "train" else 0.0
        self.phase_bias_deg = float(phase_bias_deg) if split == "train" else 0.0

        self.static_log_transform = bool(static_log_transform)

        self._rng = np.random.default_rng(seed + (0 if split == "train" else 9991))
        self._f = None  # lazy open (safe for num_workers=0; for >0 each worker will open its own)

        with h5py.File(self.h5_path, "r") as f:
            self.num_samples = int(f.attrs["num_samples"])
            self.num_paths = int(f.attrs["num_paths"])
            self.path_segments = int(f.attrs["path_segments"])
            H, W = f["grid_shape"][:].astype(np.int32).tolist()
            self.grid_shape = (int(H), int(W))

            self.x_grid = f["x_grid"][:].astype(np.float32)
            self.y_grid = f["y_grid"][:].astype(np.float32)
            self.grid_valid_mask = f.get("grid_valid_mask", None)
            if self.grid_valid_mask is not None:
                self.grid_valid_mask = self.grid_valid_mask[:].astype(np.uint8)
            else:
                self.grid_valid_mask = np.ones(self.grid_shape, dtype=np.uint8)

            self.path_seg_dist = f["path_segment_distances_km"][:].astype(np.float32)  # (P,L)

            # geometry mapping (shared)
            self.path_point_grid_idx = f["path_point_grid_idx"][:].astype(np.int64)  # (P,L,4)
            self.path_point_grid_w = f["path_point_grid_w"][:].astype(np.float32)    # (P,L,4)
            self.path_point_valid = f["path_point_valid"][:].astype(np.uint8)        # (P,L)

            # path static
            path_static = []
            pg = f["paths"]
            for pi in range(self.num_paths):
                g = pg[f"path_{pi:02d}"]
                tx_lat = float(g.attrs["tx_lat"])
                tx_lon = float(g.attrs["tx_lon"])
                rx_lat = float(g.attrs["rx_lat"])
                rx_lon = float(g.attrs["rx_lon"])
                freq_hz = float(g.attrs["frequency_hz"])
                power_kw = float(g.attrs["power_kw"])
                path_static.append([tx_lat, tx_lon, rx_lat, rx_lon, freq_hz, power_kw])
            path_static = np.asarray(path_static, dtype=np.float32)  # (P,6)
            self.path_static = transform_path_static(path_static, log_transform=self.static_log_transform)

            self.has_prior_sim = ("amplitude_x_prior_sim" in f)
            self.has_path_success = ("path_success_obs" in f)

        self.static_mean = self.path_static.mean(axis=0, keepdims=True)
        self.static_std = self.path_static.std(axis=0, keepdims=True) + 1e-6

        # split indices (deterministic)
        rng = np.random.default_rng(seed)
        idx = np.arange(self.num_samples)
        rng.shuffle(idx)
        n_train = int(round(self.num_samples * train_frac))
        self.train_idx = idx[:n_train]
        self.val_idx = idx[n_train:]
        self.indices = self.train_idx if split == "train" else self.val_idx

        # grid coord norm maps
        H, W = self.grid_shape
        X, Y = np.meshgrid(self.x_grid, self.y_grid)  # (H,W)
        xm, xs = float(X.mean()), float(X.std() + 1e-6)
        ym, ys = float(Y.mean()), float(Y.std() + 1e-6)
        x_norm = ((X - xm) / xs).astype(np.float32)
        y_norm = ((Y - ym) / ys).astype(np.float32)
        self.grid_xy_norm = np.stack([x_norm, y_norm], axis=0)  # (2,H,W)

        # param normalization stats
        self.h_mid, self.h_half = param_norm_stats_from_range(*C.HPRIME_RANGE_DAY)
        self.b_mid, self.b_half = param_norm_stats_from_range(*C.BETA_RANGE_DAY)

        print(f"Dataset[{split}] samples={len(self.indices)}, grid={self.grid_shape}, paths={self.num_paths}, L={self.path_segments}")
        print(f"  seq amp_scale_db={self.amp_scale_db}")
        print(f"  grid_valid_mask sum={int(self.grid_valid_mask.sum())}")
        print(f"  has_prior_sim_fields={bool(self.has_prior_sim)}  has_path_success={bool(self.has_path_success)}")
        print(f"  path_static: log_transform={self.static_log_transform}")
        if split == "train":
            print(f"  aug: amp_noise_db={self.amp_noise_db} phase_noise_deg={self.phase_noise_deg} path_drop_prob={self.path_drop_prob}")
            print(f"       amp_bias_db={self.amp_bias_db} phase_bias_deg={self.phase_bias_deg}")

    def _get_file(self):
        if self._f is None:
            self._f = h5py.File(self.h5_path, "r")
        return self._f

    def __del__(self):
        try:
            if self._f is not None:
                self._f.close()
        except Exception:
            pass

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        sample_idx = int(self.indices[j])
        f = self._get_file()

        hp_prior = f["hprime_prior_grid"][sample_idx].astype(np.float32)  # (H,W)
        be_prior = f["beta_prior_grid"][sample_idx].astype(np.float32)
        hp_true = f["hprime_true_grid"][sample_idx].astype(np.float32)
        be_true = f["beta_true_grid"][sample_idx].astype(np.float32)

        amp_x = f["amplitude_x"][sample_idx].astype(np.float32)  # (P,L)
        pha_x = f["phase_x"][sample_idx].astype(np.float32)
        amp_y = f["amplitude_y"][sample_idx].astype(np.float32)
        pha_y = f["phase_y"][sample_idx].astype(np.float32)
        amp_z = f["amplitude_z"][sample_idx].astype(np.float32)
        pha_z = f["phase_z"][sample_idx].astype(np.float32)

        if self.has_prior_sim:
            amp_x_p = f["amplitude_x_prior_sim"][sample_idx].astype(np.float32)
            pha_x_p = f["phase_x_prior_sim"][sample_idx].astype(np.float32)
            amp_y_p = f["amplitude_y_prior_sim"][sample_idx].astype(np.float32)
            pha_y_p = f["phase_y_prior_sim"][sample_idx].astype(np.float32)
            amp_z_p = f["amplitude_z_prior_sim"][sample_idx].astype(np.float32)
            pha_z_p = f["phase_z_prior_sim"][sample_idx].astype(np.float32)
        else:
            amp_x_p = pha_x_p = amp_y_p = pha_y_p = amp_z_p = pha_z_p = None

        if self.has_path_success:
            path_success_obs = f["path_success_obs"][sample_idx].astype(np.uint8)  # (P,)
        else:
            path_success_obs = np.ones((self.num_paths,), dtype=np.uint8)

        # --------- augmentation on obs only (train) ----------
        P, L = amp_x.shape

        # per-path calibration-like bias (shared across x/y/z)
        if self.amp_bias_db > 0:
            bias = self._rng.normal(0.0, self.amp_bias_db, size=(P, 1)).astype(np.float32)
            amp_x = amp_x + bias
            amp_y = amp_y + bias
            amp_z = amp_z + bias

        if self.phase_bias_deg > 0:
            pb = self._rng.normal(0.0, self.phase_bias_deg, size=(P, 1)).astype(np.float32)
            pha_x = wrap_phase_deg(pha_x + pb)
            pha_y = wrap_phase_deg(pha_y + pb)
            pha_z = wrap_phase_deg(pha_z + pb)

        # point-wise noise
        if self.amp_noise_db > 0:
            amp_x = amp_x + self._rng.normal(0.0, self.amp_noise_db, size=amp_x.shape).astype(np.float32)
            amp_y = amp_y + self._rng.normal(0.0, self.amp_noise_db, size=amp_y.shape).astype(np.float32)
            amp_z = amp_z + self._rng.normal(0.0, self.amp_noise_db, size=amp_z.shape).astype(np.float32)

        if self.phase_noise_deg > 0:
            pha_x = wrap_phase_deg(pha_x + self._rng.normal(0.0, self.phase_noise_deg, size=pha_x.shape).astype(np.float32))
            pha_y = wrap_phase_deg(pha_y + self._rng.normal(0.0, self.phase_noise_deg, size=pha_y.shape).astype(np.float32))
            pha_z = wrap_phase_deg(pha_z + self._rng.normal(0.0, self.phase_noise_deg, size=pha_z.shape).astype(np.float32))

        # sample prior params along path segments (P,L)
        H, W = self.grid_shape
        hp_flat = hp_prior.reshape(-1)
        be_flat = be_prior.reshape(-1)
        idxp = self.path_point_grid_idx  # (P,L,4)
        wp = self.path_point_grid_w      # (P,L,4)
        hp_prior_path = np.sum(hp_flat[idxp] * wp, axis=-1).astype(np.float32)
        be_prior_path = np.sum(be_flat[idxp] * wp, axis=-1).astype(np.float32)

        seq = build_path_seq_features(
            amp_x, pha_x, amp_y, pha_y, amp_z, pha_z,
            dist_km=self.path_seg_dist,
            amp_scale_db=self.amp_scale_db,
            amp_x_prior_sim=amp_x_p, pha_x_prior_sim=pha_x_p,
            amp_y_prior_sim=amp_y_p, pha_y_prior_sim=pha_y_p,
            amp_z_prior_sim=amp_z_p, pha_z_prior_sim=pha_z_p,
            hp_prior_path=hp_prior_path,
            be_prior_path=be_prior_path,
            h_mid=self.h_mid, h_half=self.h_half,
            b_mid=self.b_mid, b_half=self.b_half,
        )  # (P,L,C)

        ps = ((self.path_static - self.static_mean) / self.static_std).astype(np.float32)

        # 屏蔽失败路径：将对应 path_point_valid 置 0，避免 splat 污染
        path_point_valid_eff = self.path_point_valid.copy()  # (P,L)
        fail = (path_success_obs <= 0)
        if np.any(fail):
            path_point_valid_eff[fail, :] = 0

        # 训练时随机丢弃部分路径（增强鲁棒性）
        if self.path_drop_prob > 0:
            drop = self._rng.random(self.num_paths) < self.path_drop_prob
            if np.any(drop):
                path_point_valid_eff[drop, :] = 0

        return {
            "sample_idx": np.int64(sample_idx),
            "hp_prior": hp_prior,
            "be_prior": be_prior,
            "hp_true": hp_true,
            "be_true": be_true,
            "grid_xy_norm": self.grid_xy_norm,          # (2,H,W)
            "grid_valid_mask": self.grid_valid_mask,    # (H,W) uint8
            "path_seq": seq,                            # (P,L,C)
            "path_static": ps,                          # (P,6)
            "path_point_grid_idx": self.path_point_grid_idx,   # (P,L,4)
            "path_point_grid_w": self.path_point_grid_w,       # (P,L,4)
            "path_point_valid": path_point_valid_eff,          # (P,L)
        }


# -------------------------
# Loss helpers
# -------------------------

def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    pred/target: (B,H,W)
    mask: (B,H,W) bool
    """
    if mask.sum().item() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    pred_m = pred[mask]
    targ_m = target[mask]
    return F.smooth_l1_loss(pred_m, targ_m)


def weighted_smooth_l1(pred: torch.Tensor, targ: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
    """
    pred/targ: (B,H,W)
    weight_map: (B,H,W) non-negative
    """
    wsum = weight_map.sum().clamp_min(1e-6)
    diff = pred - targ
    absdiff = diff.abs()
    beta = 1.0
    l = torch.where(absdiff < beta, 0.5 * (diff ** 2) / beta, absdiff - 0.5 * beta)
    return (l * weight_map).sum() / wsum


def ramp(target: float, epoch: int, ramp_epochs: int) -> float:
    if ramp_epochs <= 0:
        return float(target)
    t = min(1.0, float(epoch) / float(ramp_epochs))
    return float(target) * t


def make_coverage_weights(
    obs_wsum: torch.Tensor,  # (B,1,H,W)
    mask_f: torch.Tensor,    # (B,H,W) float (0/1)
    near_alpha: float,
    near_gamma: float,
    far_gamma: float,
    blur_ksize: int = 1,
):
    """
    返回：
      wsum_norm_blur: (B,H,W) in [0,1]
      near_w: (B,H,W)
      far_w:  (B,H,W)
    """
    wsum = obs_wsum.squeeze(1)  # (B,H,W)
    wsum_norm = wsum / (wsum.amax(dim=(1, 2), keepdim=True) + 1e-6)
    if blur_ksize and int(blur_ksize) > 1:
        k = int(blur_ksize)
        pad = k // 2
        wsum_norm = F.avg_pool2d(wsum_norm.unsqueeze(1), kernel_size=k, stride=1, padding=pad).squeeze(1)
        wsum_norm = wsum_norm.clamp(0.0, 1.0)

    ng = float(near_gamma)
    fg = float(far_gamma)
    near_term = wsum_norm.clamp(0.0, 1.0).pow(ng)
    far_term = (1.0 - wsum_norm).clamp(0.0, 1.0).pow(fg)

    near_w = (1.0 + float(near_alpha) * near_term) * mask_f
    far_w = far_term * mask_f
    return wsum_norm, near_w, far_w


# -------------------------
# Training
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Train multi-stage UNet refiner (regular grid, coarse->fine)")

    ap.add_argument("--data", type=str, default=C.FINAL_DATASET_FILE)
    ap.add_argument("--out", type=str, default=os.path.join(C.MODEL_DIR, "unet_refiner.pt"))
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)

    # perf
    ap.add_argument("--amp", action="store_true", default=False, help="Use torch autocast + GradScaler (CUDA only)")
    ap.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps")

    # dataset / scaling
    ap.add_argument("--amp-scale-db", type=float, default=50.0)
    ap.add_argument("--static-log-transform", action="store_true", default=False, help="log10 on freq/power in path_static before standardization")

    # augmentation (train only)
    ap.add_argument("--amp-noise-db", type=float, default=0.0)
    ap.add_argument("--phase-noise-deg", type=float, default=0.0)
    ap.add_argument("--path-drop-prob", type=float, default=0.0)
    ap.add_argument("--amp-bias-db", type=float, default=0.0, help="Per-path constant amp bias (dB)")
    ap.add_argument("--phase-bias-deg", type=float, default=0.0, help="Per-path constant phase bias (deg)")

    # model refinement
    ap.add_argument("--refine-steps", type=int, default=3)
    ap.add_argument("--coarse-factor", type=int, default=4)
    ap.add_argument("--coarse-steps", type=int, default=1)
    ap.add_argument("--delta-h-scale", type=float, default=2.0)
    ap.add_argument("--delta-beta-scale", type=float, default=0.03)

    # unet
    ap.add_argument("--unet-base", type=int, default=64)
    ap.add_argument("--unet-depth", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.05)

    # loss
    ap.add_argument("--lambda-tv", type=float, default=0.005)
    ap.add_argument("--lambda-intermediate", type=float, default=0.3)
    ap.add_argument("--lambda-coarse", type=float, default=0.3)
    ap.add_argument("--lambda-nearpath", type=float, default=0.5)  # 0~1
    ap.add_argument("--beta-loss-weight", type=float, default=1.0)  # 建议 2~5

    # near/far weighting shaping
    ap.add_argument("--nearpath-alpha", type=float, default=3.0, help="near_w = 1 + alpha * wsum_norm^gamma")
    ap.add_argument("--nearpath-gamma", type=float, default=1.0)
    ap.add_argument("--farpath-gamma", type=float, default=1.0, help="far_w = (1-wsum_norm)^gamma")
    ap.add_argument("--coverage-blur-ksize", type=int, default=1, help="AvgPool blur kernel for wsum_norm (odd int). 1 disables.")

    # coverage-aware prior regularization
    ap.add_argument("--lambda-beta-prior", type=float, default=0.0)
    ap.add_argument("--lambda-h-prior", type=float, default=0.0)

    # correction loss
    ap.add_argument("--lambda-correction", type=float, default=0.15)
    ap.add_argument("--correction-scope", type=str, default="nearpath", choices=["nearpath", "all"])

    # optional ramp for regularizers
    ap.add_argument("--ramp-epochs", type=int, default=0, help="Linearly ramp correction/prior lambdas from 0 to target in first N epochs")

    ap.add_argument("--clip-grad", type=float, default=1.0)

    # early stopping / scheduler
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--lr-factor", type=float, default=0.5)
    ap.add_argument("--lr-patience", type=int, default=3)

    # metric selection
    ap.add_argument("--select-metric", type=str, default="pred", choices=["pred", "total"],
                    help="Use val_pred (no regs) or val_total (with regs) for best/scheduler/early-stop")

    # EMA
    ap.add_argument("--ema-decay", type=float, default=0.0, help="EMA decay. 0 disables EMA.")
    ap.add_argument("--ema-start-epoch", type=int, default=1, help="Start EMA update from this epoch (1-based)")
    ap.add_argument("--ema-eval", action="store_true", default=False, help="Use EMA weights for validation & best selection")

    args = ap.parse_args()

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    use_amp = bool(args.amp) and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # datasets
    train_ds = VLFH5GridDataset(
        args.data,
        split="train",
        seed=args.seed,
        amp_scale_db=args.amp_scale_db,
        amp_noise_db=args.amp_noise_db,
        phase_noise_deg=args.phase_noise_deg,
        path_drop_prob=args.path_drop_prob,
        amp_bias_db=args.amp_bias_db,
        phase_bias_deg=args.phase_bias_deg,
        static_log_transform=args.static_log_transform,
    )
    val_ds = VLFH5GridDataset(
        args.data,
        split="val",
        seed=args.seed,
        amp_scale_db=args.amp_scale_db,
        static_log_transform=args.static_log_transform,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Windows + HDF5：建议 0
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=max(1, min(args.batch_size, 4)),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # infer dims
    sample = train_ds[0]
    path_seq_in_ch = sample["path_seq"].shape[-1]
    path_static_dim = sample["path_static"].shape[-1]

    # param norm stats
    h_mid, h_half = param_norm_stats_from_range(*C.HPRIME_RANGE_DAY)
    b_mid, b_half = param_norm_stats_from_range(*C.BETA_RANGE_DAY)

    model_hparams = {
        "path_seq_in_ch": int(path_seq_in_ch),
        "path_static_dim": int(path_static_dim),
        "h_mid": float(h_mid),
        "h_half": float(h_half),
        "b_mid": float(b_mid),
        "b_half": float(b_half),
        "delta_h_scale": float(args.delta_h_scale),
        "delta_beta_scale": float(args.delta_beta_scale),
        "obs_point_dim": 64,
        "obs_grid_dim": 64,
        "unet_base": int(args.unet_base),
        "refine_steps": int(args.refine_steps),
        "coarse_factor": int(args.coarse_factor),
        "coarse_steps": int(args.coarse_steps),
        "unet_depth": int(args.unet_depth),
        "dropout": float(args.dropout),
    }

    print(f"Creating model: {model_hparams}")
    model = DRegionImagerUNetRefiner(**model_hparams).to(device)
    print(f"Params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=float(args.weight_decay))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(args.lr_factor),
        patience=int(args.lr_patience),
        min_lr=1e-6,
        threshold=float(args.min_delta),
        threshold_mode="abs",
    )

    ema = None
    if float(args.ema_decay) > 0:
        ema = EMA(model, decay=float(args.ema_decay))
        print(f"EMA enabled: decay={args.ema_decay}, start_epoch={args.ema_start_epoch}, ema_eval={bool(args.ema_eval)}")

    # selection
    best_metric = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []

    feature_scaling = {
        "amp_scale_db": float(args.amp_scale_db),
        "delta_h_scale": float(args.delta_h_scale),
        "delta_beta_scale": float(args.delta_beta_scale),
        "hprime_range": [float(C.HPRIME_RANGE_DAY[0]), float(C.HPRIME_RANGE_DAY[1])],
        "beta_range": [float(C.BETA_RANGE_DAY[0]), float(C.BETA_RANGE_DAY[1])],
        "refine_steps": int(args.refine_steps),
        "coarse_factor": int(args.coarse_factor),
        "coarse_steps": int(args.coarse_steps),
        "seed": int(args.seed),
        "amp_noise_db": float(args.amp_noise_db),
        "phase_noise_deg": float(args.phase_noise_deg),
        "path_drop_prob": float(args.path_drop_prob),
        "amp_bias_db": float(args.amp_bias_db),
        "phase_bias_deg": float(args.phase_bias_deg),
        "beta_loss_weight": float(args.beta_loss_weight),
        "lambda_beta_prior": float(args.lambda_beta_prior),
        "lambda_h_prior": float(args.lambda_h_prior),
        "lambda_correction": float(args.lambda_correction),
        "correction_scope": str(args.correction_scope),
        "nearpath_alpha": float(args.nearpath_alpha),
        "nearpath_gamma": float(args.nearpath_gamma),
        "farpath_gamma": float(args.farpath_gamma),
        "coverage_blur_ksize": int(args.coverage_blur_ksize),
        "select_metric": str(args.select_metric),
        "ema_decay": float(args.ema_decay),
        "ema_eval": bool(args.ema_eval),
        "static_log_transform": bool(args.static_log_transform),
    }

    # precompute for normalized loss
    h_mid_t = torch.tensor(h_mid, device=device, dtype=torch.float32)
    h_half_t = torch.tensor(h_half, device=device, dtype=torch.float32)
    b_mid_t = torch.tensor(b_mid, device=device, dtype=torch.float32)
    b_half_t = torch.tensor(b_half, device=device, dtype=torch.float32)

    def norm_hp(x): return (x - h_mid_t) / (h_half_t + 1e-6)
    def norm_be(x): return (x - b_mid_t) / (b_half_t + 1e-6)

    def corr_hp(pred, prior): return (pred - prior) / (h_half_t + 1e-6)
    def corr_be(pred, prior): return (pred - prior) / (b_half_t + 1e-6)

    grad_accum = max(1, int(args.grad_accum))

    for epoch in range(1, args.epochs + 1):
        # ramp regularizers if asked
        lam_corr = ramp(args.lambda_correction, epoch, args.ramp_epochs)
        lam_bprior = ramp(args.lambda_beta_prior, epoch, args.ramp_epochs)
        lam_hprior = ramp(args.lambda_h_prior, epoch, args.ramp_epochs)

        # ---------------- train ----------------
        model.train()
        train_total_list = []
        train_pred_list = []
        train_h = []
        train_b = []
        train_corr = []
        train_hprior = []
        train_bprior = []

        optimizer.zero_grad(set_to_none=True)

        for it, b in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")):
            hp_prior = b["hp_prior"].to(device=device, dtype=torch.float32)
            be_prior = b["be_prior"].to(device=device, dtype=torch.float32)
            hp_true = b["hp_true"].to(device=device, dtype=torch.float32)
            be_true = b["be_true"].to(device=device, dtype=torch.float32)

            grid_xy_norm = b["grid_xy_norm"].to(device=device, dtype=torch.float32)
            mask = b["grid_valid_mask"].to(device=device).bool()  # (B,H,W)
            mask_f = mask.float()

            path_seq = b["path_seq"].to(device=device, dtype=torch.float32)
            path_static = b["path_static"].to(device=device, dtype=torch.float32)

            idx = b["path_point_grid_idx"].to(device=device, dtype=torch.long)
            w = b["path_point_grid_w"].to(device=device, dtype=torch.float32)
            v = b["path_point_valid"].to(device=device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                out = model(
                    hp_prior=hp_prior,
                    be_prior=be_prior,
                    grid_xy_norm=grid_xy_norm,
                    grid_valid_mask=b["grid_valid_mask"].to(device=device),
                    path_seq_feats=path_seq,
                    path_static_feats=path_static,
                    path_point_grid_idx=idx,
                    path_point_grid_w=w,
                    path_point_valid=v,
                )

                hp_pred = out["hprime_pred"]
                be_pred = out["beta_pred"]

                # normalized targets
                hp_pred_n = norm_hp(hp_pred)
                hp_true_n = norm_hp(hp_true)
                be_pred_n = norm_be(be_pred)
                be_true_n = norm_be(be_true)

                # global pred loss (分项)
                loss_h_global = masked_smooth_l1(hp_pred_n, hp_true_n, mask)
                loss_b_global = masked_smooth_l1(be_pred_n, be_true_n, mask)
                loss_global = loss_h_global + float(args.beta_loss_weight) * loss_b_global

                # near-path mixing (pred-only objective)
                obs_wsum = out.get("obs_wsum", None)  # (B,1,H,W)
                if obs_wsum is not None:
                    _, near_w, far_w = make_coverage_weights(
                        obs_wsum=obs_wsum,
                        mask_f=mask_f,
                        near_alpha=args.nearpath_alpha,
                        near_gamma=args.nearpath_gamma,
                        far_gamma=args.farpath_gamma,
                        blur_ksize=args.coverage_blur_ksize,
                    )
                    loss_h_near = weighted_smooth_l1(hp_pred_n, hp_true_n, near_w)
                    loss_b_near = weighted_smooth_l1(be_pred_n, be_true_n, near_w)
                    loss_near = loss_h_near + float(args.beta_loss_weight) * loss_b_near
                    loss_pred = (1.0 - args.lambda_nearpath) * loss_global + args.lambda_nearpath * loss_near
                else:
                    far_w = mask_f
                    loss_pred = loss_global

                # correction loss
                loss_corr = torch.zeros((), device=device, dtype=torch.float32)
                if lam_corr > 0:
                    hp_corr_p = corr_hp(hp_pred, hp_prior)
                    hp_corr_t = corr_hp(hp_true, hp_prior)
                    be_corr_p = corr_be(be_pred, be_prior)
                    be_corr_t = corr_be(be_true, be_prior)

                    if (args.correction_scope == "nearpath") and (obs_wsum is not None):
                        w_corr = near_w
                    else:
                        w_corr = mask_f

                    lh_c = weighted_smooth_l1(hp_corr_p, hp_corr_t, w_corr)
                    lb_c = weighted_smooth_l1(be_corr_p, be_corr_t, w_corr)
                    loss_corr = lh_c + float(args.beta_loss_weight) * lb_c

                # prior regularization (far from paths => stronger)
                loss_beta_prior = torch.zeros((), device=device, dtype=torch.float32)
                if lam_bprior > 0:
                    be_prior_n = norm_be(be_prior)
                    loss_beta_prior = weighted_smooth_l1(be_pred_n, be_prior_n, far_w) if (obs_wsum is not None) else masked_smooth_l1(be_pred_n, be_prior_n, mask)

                loss_h_prior = torch.zeros((), device=device, dtype=torch.float32)
                if lam_hprior > 0:
                    hp_prior_n = norm_hp(hp_prior)
                    loss_h_prior = weighted_smooth_l1(hp_pred_n, hp_prior_n, far_w) if (obs_wsum is not None) else masked_smooth_l1(hp_pred_n, hp_prior_n, mask)

                # coarse supervision
                loss_coarse = torch.zeros((), device=device, dtype=torch.float32)
                if args.lambda_coarse > 0 and ("hprime_coarse_up" in out):
                    hp_c = norm_hp(out["hprime_coarse_up"])
                    be_c = norm_be(out["beta_coarse_up"])
                    loss_h_c = masked_smooth_l1(hp_c, hp_true_n, mask)
                    loss_b_c = masked_smooth_l1(be_c, be_true_n, mask)
                    loss_coarse = loss_h_c + float(args.beta_loss_weight) * loss_b_c

                # intermediate fine steps
                loss_inter = torch.zeros((), device=device, dtype=torch.float32)
                if args.lambda_intermediate > 0:
                    hp_steps = out.get("hprime_steps", [])
                    be_steps = out.get("beta_steps", [])
                    if len(hp_steps) > 1:
                        wsum_t = 0.0
                        acc = 0.0
                        for t in range(len(hp_steps) - 1):
                            wt = float(t + 1) / float(len(hp_steps))
                            hp_t = norm_hp(hp_steps[t])
                            be_t = norm_be(be_steps[t])
                            lh = masked_smooth_l1(hp_t, hp_true_n, mask)
                            lb = masked_smooth_l1(be_t, be_true_n, mask)
                            acc = acc + wt * (lh + float(args.beta_loss_weight) * lb)
                            wsum_t += wt
                        loss_inter = acc / max(1e-6, wsum_t)

                # TV loss (on normalized outputs)
                loss_tv = torch.zeros((), device=device, dtype=torch.float32)
                if args.lambda_tv > 0:
                    pred_stack = torch.stack([hp_pred_n, be_pred_n], dim=1)  # (B,2,H,W)
                    loss_tv = tv_loss_2d(pred_stack, mask=mask)

                loss_total = (
                    loss_pred
                    + float(lam_corr) * loss_corr
                    + float(lam_hprior) * loss_h_prior
                    + float(lam_bprior) * loss_beta_prior
                    + float(args.lambda_coarse) * loss_coarse
                    + float(args.lambda_intermediate) * loss_inter
                    + float(args.lambda_tv) * loss_tv
                )

                loss_to_backprop = loss_total / float(grad_accum)

            if use_amp:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()

            do_step = ((it + 1) % grad_accum == 0) or (it + 1 == len(train_loader))
            if do_step:
                if args.clip_grad and args.clip_grad > 0:
                    if use_amp:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.clip_grad)

                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)

            # EMA update
            if ema is not None and epoch >= int(args.ema_start_epoch) and do_step:
                ema.update(model)

            train_total_list.append(float(loss_total.detach().item()))
            train_pred_list.append(float(loss_pred.detach().item()))
            train_h.append(float(loss_h_global.detach().item()))
            train_b.append(float(loss_b_global.detach().item()))
            train_corr.append(float(loss_corr.detach().item()))
            train_hprior.append(float(loss_h_prior.detach().item()))
            train_bprior.append(float(loss_beta_prior.detach().item()))

        # ---------------- val ----------------
        model.eval()

        val_pred_list = []
        val_total_list = []
        val_h_list = []
        val_b_list = []
        val_corr_list = []
        val_hprior_list = []
        val_bprior_list = []

        @contextmanager
        def maybe_ema():
            if ema is not None and bool(args.ema_eval):
                with ema.apply_to(model):
                    yield
            else:
                yield

        with maybe_ema():
            with torch.no_grad():
                for b in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                    hp_prior = b["hp_prior"].to(device=device, dtype=torch.float32)
                    be_prior = b["be_prior"].to(device=device, dtype=torch.float32)
                    hp_true = b["hp_true"].to(device=device, dtype=torch.float32)
                    be_true = b["be_true"].to(device=device, dtype=torch.float32)

                    grid_xy_norm = b["grid_xy_norm"].to(device=device, dtype=torch.float32)
                    mask = b["grid_valid_mask"].to(device=device).bool()
                    mask_f = mask.float()

                    path_seq = b["path_seq"].to(device=device, dtype=torch.float32)
                    path_static = b["path_static"].to(device=device, dtype=torch.float32)

                    idx = b["path_point_grid_idx"].to(device=device, dtype=torch.long)
                    w = b["path_point_grid_w"].to(device=device, dtype=torch.float32)
                    v = b["path_point_valid"].to(device=device)

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        out = model(
                            hp_prior=hp_prior,
                            be_prior=be_prior,
                            grid_xy_norm=grid_xy_norm,
                            grid_valid_mask=b["grid_valid_mask"].to(device=device),
                            path_seq_feats=path_seq,
                            path_static_feats=path_static,
                            path_point_grid_idx=idx,
                            path_point_grid_w=w,
                            path_point_valid=v,
                        )

                        hp_pred = out["hprime_pred"]
                        be_pred = out["beta_pred"]

                        hp_pred_n = norm_hp(hp_pred)
                        hp_true_n = norm_hp(hp_true)
                        be_pred_n = norm_be(be_pred)
                        be_true_n = norm_be(be_true)

                        loss_h = masked_smooth_l1(hp_pred_n, hp_true_n, mask)
                        loss_b = masked_smooth_l1(be_pred_n, be_true_n, mask)
                        val_global = loss_h + float(args.beta_loss_weight) * loss_b

                        obs_wsum = out.get("obs_wsum", None)
                        if obs_wsum is not None:
                            _, near_w, far_w = make_coverage_weights(
                                obs_wsum=obs_wsum,
                                mask_f=mask_f,
                                near_alpha=args.nearpath_alpha,
                                near_gamma=args.nearpath_gamma,
                                far_gamma=args.farpath_gamma,
                                blur_ksize=args.coverage_blur_ksize,
                            )
                            loss_h_near = weighted_smooth_l1(hp_pred_n, hp_true_n, near_w)
                            loss_b_near = weighted_smooth_l1(be_pred_n, be_true_n, near_w)
                            val_near = loss_h_near + float(args.beta_loss_weight) * loss_b_near
                            val_pred = (1.0 - args.lambda_nearpath) * val_global + args.lambda_nearpath * val_near
                        else:
                            far_w = mask_f
                            val_pred = val_global

                        # total = pred + regs (same weights as train for fair tracking)
                        val_corr = torch.zeros((), device=device, dtype=torch.float32)
                        if lam_corr > 0:
                            hp_corr_p = corr_hp(hp_pred, hp_prior)
                            hp_corr_t = corr_hp(hp_true, hp_prior)
                            be_corr_p = corr_be(be_pred, be_prior)
                            be_corr_t = corr_be(be_true, be_prior)

                            if (args.correction_scope == "nearpath") and (obs_wsum is not None):
                                w_corr = near_w
                            else:
                                w_corr = mask_f

                            lh_c = weighted_smooth_l1(hp_corr_p, hp_corr_t, w_corr)
                            lb_c = weighted_smooth_l1(be_corr_p, be_corr_t, w_corr)
                            val_corr = lh_c + float(args.beta_loss_weight) * lb_c

                        val_bprior = torch.zeros((), device=device, dtype=torch.float32)
                        if lam_bprior > 0:
                            be_prior_n = norm_be(be_prior)
                            val_bprior = weighted_smooth_l1(be_pred_n, be_prior_n, far_w) if (obs_wsum is not None) else masked_smooth_l1(be_pred_n, be_prior_n, mask)

                        val_hprior = torch.zeros((), device=device, dtype=torch.float32)
                        if lam_hprior > 0:
                            hp_prior_n = norm_hp(hp_prior)
                            val_hprior = weighted_smooth_l1(hp_pred_n, hp_prior_n, far_w) if (obs_wsum is not None) else masked_smooth_l1(hp_pred_n, hp_prior_n, mask)

                        val_total = val_pred + float(lam_corr) * val_corr + float(lam_hprior) * val_hprior + float(lam_bprior) * val_bprior

                    val_pred_list.append(float(val_pred.detach().item()))
                    val_total_list.append(float(val_total.detach().item()))
                    val_h_list.append(float(loss_h.detach().item()))
                    val_b_list.append(float(loss_b.detach().item()))
                    val_corr_list.append(float(val_corr.detach().item()))
                    val_hprior_list.append(float(val_hprior.detach().item()))
                    val_bprior_list.append(float(val_bprior.detach().item()))

        tr_total = float(np.mean(train_total_list)) if train_total_list else float("nan")
        tr_pred = float(np.mean(train_pred_list)) if train_pred_list else float("nan")
        tr_h = float(np.mean(train_h)) if train_h else float("nan")
        tr_b = float(np.mean(train_b)) if train_b else float("nan")
        tr_corr = float(np.mean(train_corr)) if train_corr else float("nan")
        tr_hprior = float(np.mean(train_hprior)) if train_hprior else float("nan")
        tr_bprior = float(np.mean(train_bprior)) if train_bprior else float("nan")

        va_pred = float(np.mean(val_pred_list)) if val_pred_list else float("nan")
        va_total = float(np.mean(val_total_list)) if val_total_list else float("nan")
        va_h = float(np.mean(val_h_list)) if val_h_list else float("nan")
        va_b = float(np.mean(val_b_list)) if val_b_list else float("nan")
        va_corr = float(np.mean(val_corr_list)) if val_corr_list else float("nan")
        va_hprior = float(np.mean(val_hprior_list)) if val_hprior_list else float("nan")
        va_bprior = float(np.mean(val_bprior_list)) if val_bprior_list else float("nan")

        # choose metric for scheduler / best
        metric = va_pred if args.select_metric == "pred" else va_total

        lr_before = float(optimizer.param_groups[0]["lr"])
        scheduler.step(metric)
        lr_after = float(optimizer.param_groups[0]["lr"])

        history.append({
            "epoch": epoch,
            "train_total": tr_total,
            "train_pred": tr_pred,
            "train_h_global": tr_h,
            "train_b_global": tr_b,
            "train_correction": tr_corr,
            "train_h_prior_reg": tr_hprior,
            "train_beta_prior_reg": tr_bprior,
            "val_pred": va_pred,
            "val_total": va_total,
            "val_h_global": va_h,
            "val_b_global": va_b,
            "val_correction": va_corr,
            "val_h_prior_reg": va_hprior,
            "val_beta_prior_reg": va_bprior,
            "metric_used": float(metric),
            "lr": lr_after,
            "lam_corr": float(lam_corr),
            "lam_hprior": float(lam_hprior),
            "lam_bprior": float(lam_bprior),
        })

        print(
            f"Epoch {epoch:03d}: "
            f"TrainTotal={tr_total:.6f} (Pred={tr_pred:.6f}, h={tr_h:.6f}, b={tr_b:.6f}, corr={tr_corr:.6f}, hprior={tr_hprior:.6f}, bprior={tr_bprior:.6f}) | "
            f"ValPred={va_pred:.6f} ValTotal={va_total:.6f} (h={va_h:.6f}, b={va_b:.6f}, corr={va_corr:.6f}, hprior={va_hprior:.6f}, bprior={va_bprior:.6f}) | "
            f"Metric[{args.select_metric}]={metric:.6f} | LR={lr_after:.2e}"
        )

        # ---- save best by selected metric ----
        improved = (metric + args.min_delta) < best_metric
        if improved:
            best_metric = float(metric)
            best_epoch = epoch
            bad_epochs = 0

            ckpt = {
                "epoch": epoch,
                "best_metric": best_metric,
                "metric_name": str(args.select_metric),
                "val_pred": va_pred,
                "val_total": va_total,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": (ema.state_dict() if ema is not None else None),
                "model_hparams": model_hparams,
                "feature_scaling": feature_scaling,
                "history": history,
                "args": vars(args),
            }
            torch.save(ckpt, args.out)
            print(f"  ✓ Saved best to {args.out} (metric={best_metric:.6f}, val_pred={va_pred:.6f}, val_total={va_total:.6f})")
        else:
            bad_epochs += 1

        # reset early-stopping counter if LR reduced
        if lr_after < lr_before - 1e-12:
            print(f"  ↘ LR reduced: {lr_before:.2e} -> {lr_after:.2e}, reset early-stopping counter.")
            bad_epochs = 0

        if args.patience > 0 and bad_epochs >= args.patience:
            print(f"Early stopping: no improvement for {bad_epochs} epochs. Best epoch={best_epoch} best_metric={best_metric:.6f}")
            break

    hist_path = args.out.replace(".pt", "_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    print(f"History saved to {hist_path}")
    print(f"Done. Best metric[{args.select_metric}]={best_metric:.6f} at epoch {best_epoch}")


if __name__ == "__main__":
    main()