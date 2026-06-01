# enhanced_train.py
# -*- coding: utf-8 -*-
"""
Complete standalone training script for enhanced D-region imaging model.
No dependencies on old code.
"""

from __future__ import annotations
import os
import argparse
import json
import math
import random
import numpy as np
import h5py
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# Import from our new model file
from gnn_imager_models import EnhancedDRegionImager, tv_loss_2d


# -------------------------
# Configuration
# -------------------------

class Config:
    """Configuration class matching your original config"""
    HPRIME_RANGE_DAY = (55.0, 80.0)
    BETA_RANGE_DAY = (0.3, 0.8)
    FINAL_DATASET_FILE = "data/north_america_vlf_dataset.h5"
    MODEL_DIR = "models"
    PLOT_DIR = "plots"
    DATA_DIR = "data"


C = Config()


# -------------------------
# Reproducibility
# -------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# Dataset Utilities
# -------------------------

def param_norm_stats_from_range(vmin: float, vmax: float) -> Tuple[float, float]:
    mid = 0.5 * (vmin + vmax)
    half = 0.5 * (vmax - vmin) + 1e-6
    return float(mid), float(half)


def phase_to_sincos_deg(phase_deg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rad = phase_deg.astype(np.float32) * (math.pi / 180.0)
    return np.sin(rad).astype(np.float32), np.cos(rad).astype(np.float32)


def wrap_phase_delta_deg(delta_deg: np.ndarray) -> np.ndarray:
    dd = ((delta_deg + 180.0) % 360.0) - 180.0
    return dd.astype(np.float32)


def wrap_phase_deg(phase_deg: np.ndarray) -> np.ndarray:
    return (((phase_deg + 180.0) % 360.0) - 180.0).astype(np.float32)


def unwrap_phase_deg_to_rad(phase_deg: np.ndarray) -> np.ndarray:
    rad = np.deg2rad(phase_deg.astype(np.float32))
    return np.unwrap(rad, axis=1).astype(np.float32)


def build_path_seq_features(
    amp_x_obs, pha_x_obs, amp_y_obs, pha_y_obs, amp_z_obs, pha_z_obs,
    dist_km: np.ndarray,
    amp_scale_db: float = 50.0,
    amp_x_prior_sim=None, pha_x_prior_sim=None,
    amp_y_prior_sim=None, pha_y_prior_sim=None,
    amp_z_prior_sim=None, pha_z_prior_sim=None,
    hp_prior_path=None, be_prior_path=None,
    h_mid: float = 0.0, h_half: float = 1.0,
    b_mid: float = 0.0, b_half: float = 1.0,
):
    """
    Build path sequence features
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
        ax, axc, dax, sx, cx, dpx,
        ay, ayc, day, sy, cy, dpy,
        az, azc, daz, sz, cz, dpz,
        dist_norm.astype(np.float32),
    ]

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
# Dataset Class
# -------------------------

class EnhancedVLFDataset(torch.utils.data.Dataset):
    """Enhanced dataset for VLF imaging"""
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
    ):
        super().__init__()
        self.h5_path = h5_path
        self.split = split
        self.amp_scale_db = float(amp_scale_db)

        self.amp_noise_db = float(amp_noise_db) if split == "train" else 0.0
        self.phase_noise_deg = float(phase_noise_deg) if split == "train" else 0.0
        self.path_drop_prob = float(path_drop_prob) if split == "train" else 0.0

        self._rng = np.random.default_rng(seed + (0 if split == "train" else 9991))

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

            self.path_seg_dist = f["path_segment_distances_km"][:].astype(np.float32)

            self.path_point_grid_idx = f["path_point_grid_idx"][:].astype(np.int64)
            self.path_point_grid_w = f["path_point_grid_w"][:].astype(np.float32)
            self.path_point_valid = f["path_point_valid"][:].astype(np.uint8)

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
            self.path_static = np.asarray(path_static, dtype=np.float32)

            self.has_prior_sim = ("amplitude_x_prior_sim" in f)
            self.has_path_success = ("path_success_obs" in f)

        self.static_mean = self.path_static.mean(axis=0, keepdims=True)
        self.static_std = self.path_static.std(axis=0, keepdims=True) + 1e-6

        rng = np.random.default_rng(seed)
        idx = np.arange(self.num_samples)
        rng.shuffle(idx)
        n_train = int(round(self.num_samples * train_frac))
        self.train_idx = idx[:n_train]
        self.val_idx = idx[n_train:]
        self.indices = self.train_idx if split == "train" else self.val_idx

        H, W = self.grid_shape
        X, Y = np.meshgrid(self.x_grid, self.y_grid)
        xm, xs = float(X.mean()), float(X.std() + 1e-6)
        ym, ys = float(Y.mean()), float(Y.std() + 1e-6)
        x_norm = ((X - xm) / xs).astype(np.float32)
        y_norm = ((Y - ym) / ys).astype(np.float32)
        self.grid_xy_norm = np.stack([x_norm, y_norm], axis=0)

        self.h_mid, self.h_half = param_norm_stats_from_range(*C.HPRIME_RANGE_DAY)
        self.b_mid, self.b_half = param_norm_stats_from_range(*C.BETA_RANGE_DAY)

        print(f"Dataset[{split}] samples={len(self.indices)}, grid={self.grid_shape}, paths={self.num_paths}")
        print(f"  amp_scale_db={self.amp_scale_db}")
        print(f"  grid_valid_mask sum={int(self.grid_valid_mask.sum())}")
        if split == "train":
            print(f"  aug: amp_noise_db={self.amp_noise_db} phase_noise_deg={self.phase_noise_deg}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, j):
        sample_idx = int(self.indices[j])
        with h5py.File(self.h5_path, "r") as f:
            hp_prior = f["hprime_prior_grid"][sample_idx].astype(np.float32)
            be_prior = f["beta_prior_grid"][sample_idx].astype(np.float32)
            hp_true = f["hprime_true_grid"][sample_idx].astype(np.float32)
            be_true = f["beta_true_grid"][sample_idx].astype(np.float32)

            amp_x = f["amplitude_x"][sample_idx].astype(np.float32)
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
                path_success_obs = f["path_success_obs"][sample_idx].astype(np.uint8)
            else:
                path_success_obs = np.ones((self.num_paths,), dtype=np.uint8)

        if self.amp_noise_db > 0:
            n = self._rng.normal(0.0, self.amp_noise_db, size=amp_x.shape).astype(np.float32)
            amp_x = amp_x + n
            amp_y = amp_y + self._rng.normal(0.0, self.amp_noise_db, size=amp_y.shape).astype(np.float32)
            amp_z = amp_z + self._rng.normal(0.0, self.amp_noise_db, size=amp_z.shape).astype(np.float32)

        if self.phase_noise_deg > 0:
            pha_x = wrap_phase_deg(pha_x + self._rng.normal(0.0, self.phase_noise_deg, size=pha_x.shape).astype(np.float32))
            pha_y = wrap_phase_deg(pha_y + self._rng.normal(0.0, self.phase_noise_deg, size=pha_y.shape).astype(np.float32))
            pha_z = wrap_phase_deg(pha_z + self._rng.normal(0.0, self.phase_noise_deg, size=pha_z.shape).astype(np.float32))

        H, W = self.grid_shape
        hp_flat = hp_prior.reshape(-1)
        be_flat = be_prior.reshape(-1)
        idxp = self.path_point_grid_idx
        wp = self.path_point_grid_w
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
        )

        ps = ((self.path_static - self.static_mean) / self.static_std).astype(np.float32)

        path_point_valid_eff = self.path_point_valid.copy()
        fail = (path_success_obs <= 0)
        if np.any(fail):
            path_point_valid_eff[fail, :] = 0

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
            "grid_xy_norm": self.grid_xy_norm,
            "grid_valid_mask": self.grid_valid_mask,
            "path_seq": seq,
            "path_static": ps,
            "path_point_grid_idx": self.path_point_grid_idx,
            "path_point_grid_w": self.path_point_grid_w,
            "path_point_valid": path_point_valid_eff,
        }


# -------------------------
# Loss Functions
# -------------------------

def masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked smooth L1 loss"""
    if mask.sum().item() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    pred_m = pred[mask]
    targ_m = target[mask]
    return F.smooth_l1_loss(pred_m, targ_m)


def weighted_smooth_l1(pred: torch.Tensor, targ: torch.Tensor, weight_map: torch.Tensor) -> torch.Tensor:
    """Weighted smooth L1 loss"""
    wsum = weight_map.sum().clamp_min(1e-6)
    diff = pred - targ
    absdiff = diff.abs()
    beta = 1.0
    l = torch.where(absdiff < beta, 0.5 * (diff ** 2) / beta, absdiff - 0.5 * beta)
    return (l * weight_map).sum() / wsum


class MixedLoss(nn.Module):
    """混合损失函数"""
    def __init__(self, alpha: float = 0.7, beta: float = 0.2, gamma: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
    def ssim_loss(self, x, y, window_size=11):
        """结构相似性损失"""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.avg_pool2d(x, window_size, 1, window_size//2)
        mu_y = F.avg_pool2d(y, window_size, 1, window_size//2)
        
        sigma_x = F.avg_pool2d(x**2, window_size, 1, window_size//2) - mu_x**2
        sigma_y = F.avg_pool2d(y**2, window_size, 1, window_size//2) - mu_y**2
        sigma_xy = F.avg_pool2d(x*y, window_size, 1, window_size//2) - mu_x*mu_y
        
        ssim = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) / \
               ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
        
        return 1 - ssim.mean()
        
    def forward(self, pred, target, mask=None):
        if mask is not None:
            pred = pred * mask
            target = target * mask
            
        l1_loss = F.l1_loss(pred, target)
        l2_loss = F.mse_loss(pred, target)
        ssim_loss = self.ssim_loss(pred.unsqueeze(1), target.unsqueeze(1))
        
        return self.alpha * l1_loss + self.beta * l2_loss + self.gamma * ssim_loss


# -------------------------
# Gradient Accumulation
# -------------------------

class GradientAccumulator:
    """梯度累积器"""
    def __init__(self, accumulation_steps: int = 4):
        self.accumulation_steps = accumulation_steps
        self.step_count = 0
        
    def step(self, model, optimizer, scaler, loss):
        loss = loss / self.accumulation_steps
        scaler.scale(loss).backward()
        
        self.step_count += 1
        if self.step_count % self.accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
    def is_update_step(self):
        return self.step_count % self.accumulation_steps == 0


# -------------------------
# Main Training Function
# -------------------------

def main():
    parser = argparse.ArgumentParser(description="Train enhanced D-region imaging model")
    parser.add_argument("--data", type=str, default=C.FINAL_DATASET_FILE)
    parser.add_argument("--out", type=str, default=os.path.join(C.MODEL_DIR, "enhanced_model.pt"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)

    # 模型架构
    parser.add_argument("--unet-base", type=int, default=64)
    parser.add_argument("--unet-depth", type=int, default=3)
    parser.add_argument("--refine-steps", type=int, default=3)
    parser.add_argument("--coarse-factor", type=int, default=4)
    parser.add_argument("--coarse-steps", type=int, default=1)

    # 学习率
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--t0", type=int, default=10)
    parser.add_argument("--t-mult", type=int, default=2)

    # 损失权重
    parser.add_argument("--lambda-tv", type=float, default=0.01)
    parser.add_argument("--lambda-intermediate", type=float, default=0.3)
    parser.add_argument("--lambda-coarse", type=float, default=0.3)
    parser.add_argument("--lambda-nearpath", type=float, default=0.6)
    parser.add_argument("--beta-loss-weight", type=float, default=5.0)
    parser.add_argument("--h-loss-weight", type=float, default=1.0)
    parser.add_argument("--lambda-ssim", type=float, default=0.1)

    # 正则化
    parser.add_argument("--lambda-beta-prior", type=float, default=0.02)
    parser.add_argument("--lambda-h-prior", type=float, default=0.01)
    parser.add_argument("--lambda-correction", type=float, default=0.25)

    # 数据增强
    parser.add_argument("--amp-noise-db", type=float, default=0.2)
    parser.add_argument("--phase-noise-deg", type=float, default=2.0)
    parser.add_argument("--path-drop-prob", type=float, default=0.15)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)

    # 训练
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(f"Configuration: {args}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # 数据集
    train_ds = EnhancedVLFDataset(
        args.data,
        split="train",
        seed=args.seed,
        amp_scale_db=50.0,
        amp_noise_db=args.amp_noise_db,
        phase_noise_deg=args.phase_noise_deg,
        path_drop_prob=args.path_drop_prob,
    )
    
    val_ds = EnhancedVLFDataset(
        args.data,
        split="val",
        seed=args.seed,
        amp_scale_db=50.0,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=max(1, min(args.batch_size, 4)),
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    # 获取输入维度
    sample = train_ds[0]
    path_seq_in_ch = sample["path_seq"].shape[-1]
    path_static_dim = sample["path_static"].shape[-1]

    # 参数标准化
    h_mid, h_half = param_norm_stats_from_range(*C.HPRIME_RANGE_DAY)
    b_mid, b_half = param_norm_stats_from_range(*C.BETA_RANGE_DAY)

    # 模型参数
    model_hparams = {
        "path_seq_in_ch": int(path_seq_in_ch),
        "path_static_dim": int(path_static_dim),
        "h_mid": float(h_mid),
        "h_half": float(h_half),
        "b_mid": float(b_mid),
        "b_half": float(b_half),
        "delta_h_scale": 2.0,
        "delta_beta_scale": 0.06,
        "obs_point_dim": 96,
        "obs_grid_dim": 96,
        "unet_base": int(args.unet_base),
        "refine_steps": int(args.refine_steps),
        "coarse_factor": int(args.coarse_factor),
        "coarse_steps": int(args.coarse_steps),
        "unet_depth": int(args.unet_depth),
        "dropout": 0.1,
    }

    print(f"Creating model: {model_hparams}")
    model = EnhancedDRegionImager(**model_hparams).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-8)

    # 学习率调度
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=args.t0,
        T_mult=args.t_mult,
        eta_min=args.min_lr
    )

    # 混合损失
    mixed_loss_fn = MixedLoss(alpha=0.7, beta=0.2, gamma=0.1)

    # 梯度累积
    grad_accum = GradientAccumulator(args.grad_accum)
    scaler = GradScaler(enabled=(device.type == "cuda"))

    # 训练历史
    history = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    # 训练循环
    for epoch in range(1, args.epochs + 1):
        # 预热学习率
        if epoch <= args.warmup_epochs:
            for param_group in optimizer.param_groups:
                base_lr = args.lr
                param_group['lr'] = base_lr * (epoch / args.warmup_epochs)

        # 训练阶段
        model.train()
        train_losses = []
        train_h_losses = []
        train_b_losses = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for batch_idx, b in enumerate(pbar):
            hp_prior = b["hp_prior"].to(device, dtype=torch.float32)
            be_prior = b["be_prior"].to(device, dtype=torch.float32)
            hp_true = b["hp_true"].to(device, dtype=torch.float32)
            be_true = b["be_true"].to(device, dtype=torch.float32)
            
            grid_xy_norm = b["grid_xy_norm"].to(device, dtype=torch.float32)
            mask = b["grid_valid_mask"].to(device).bool()
            
            path_seq = b["path_seq"].to(device, dtype=torch.float32)
            path_static = b["path_static"].to(device, dtype=torch.float32)
            
            idx = b["path_point_grid_idx"].to(device, dtype=torch.long)
            w = b["path_point_grid_w"].to(device, dtype=torch.float32)
            v = b["path_point_valid"].to(device)

            # MixUp增强
            if args.mixup_alpha > 0 and random.random() < 0.5:
                lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                batch_size = hp_prior.size(0)
                index = torch.randperm(batch_size).to(device)
                
                hp_prior = lam * hp_prior + (1 - lam) * hp_prior[index]
                be_prior = lam * be_prior + (1 - lam) * be_prior[index]
                hp_true = lam * hp_true + (1 - lam) * hp_true[index]
                be_true = lam * be_true + (1 - lam) * be_true[index]
                path_seq = lam * path_seq + (1 - lam) * path_seq[index]
                path_static = lam * path_static + (1 - lam) * path_static[index]

            # 前向传播
            with autocast(enabled=(device.type == "cuda")):
                out = model(
                    hp_prior=hp_prior,
                    be_prior=be_prior,
                    grid_xy_norm=grid_xy_norm,
                    grid_valid_mask=b["grid_valid_mask"].to(device),
                    path_seq_feats=path_seq,
                    path_static_feats=path_static,
                    path_point_grid_idx=idx,
                    path_point_grid_w=w,
                    path_point_valid=v,
                )

                hp_pred = out["hprime_pred"]
                be_pred = out["beta_pred"]

                # 归一化
                hp_pred_n = (hp_pred - model.h_mid) / (model.h_half + 1e-6)
                hp_true_n = (hp_true - model.h_mid) / (model.h_half + 1e-6)
                be_pred_n = (be_pred - model.b_mid) / (model.b_half + 1e-6)
                be_true_n = (be_true - model.b_mid) / (model.b_half + 1e-6)

                # 主损失
                loss_h = mixed_loss_fn(hp_pred_n, hp_true_n, mask.float())
                loss_b = mixed_loss_fn(be_pred_n, be_true_n, mask.float())
                loss_main = args.h_loss_weight * loss_h + args.beta_loss_weight * loss_b

                # 近路径加权
                obs_wsum = out.get("obs_wsum", None)
                if obs_wsum is not None:
                    wsum = obs_wsum.squeeze(1)
                    wsum_norm = wsum / (wsum.amax(dim=(1, 2), keepdim=True) + 1e-6)
                    near_w = (1.0 + 3.0 * wsum_norm) * mask.float()
                    
                    loss_h_near = weighted_smooth_l1(hp_pred_n, hp_true_n, near_w)
                    loss_b_near = weighted_smooth_l1(be_pred_n, be_true_n, near_w)
                    loss_near = args.h_loss_weight * loss_h_near + args.beta_loss_weight * loss_b_near
                    
                    loss_main = (1.0 - args.lambda_nearpath) * loss_main + args.lambda_nearpath * loss_near

                # 修正损失
                if args.lambda_correction > 0:
                    hp_corr_p = (hp_pred - hp_prior) / (model.h_half + 1e-6)
                    hp_corr_t = (hp_true - hp_prior) / (model.h_half + 1e-6)
                    be_corr_p = (be_pred - be_prior) / (model.b_half + 1e-6)
                    be_corr_t = (be_true - be_prior) / (model.b_half + 1e-6)
                    
                    loss_corr = (F.l1_loss(hp_corr_p, hp_corr_t) + 
                               args.beta_loss_weight * F.l1_loss(be_corr_p, be_corr_t))
                    loss_main = loss_main + args.lambda_correction * loss_corr

                # 先验正则化
                if args.lambda_beta_prior > 0:
                    loss_beta_prior = F.l1_loss(be_pred_n, (be_prior - model.b_mid) / (model.b_half + 1e-6))
                    loss_main = loss_main + args.lambda_beta_prior * loss_beta_prior
                
                if args.lambda_h_prior > 0:
                    loss_h_prior = F.l1_loss(hp_pred_n, (hp_prior - model.h_mid) / (model.h_half + 1e-6))
                    loss_main = loss_main + args.lambda_h_prior * loss_h_prior

                # 中间监督
                loss_inter = 0.0
                if args.lambda_intermediate > 0:
                    hp_steps = out.get("hprime_steps", [])
                    be_steps = out.get("beta_steps", [])
                    if len(hp_steps) > 1:
                        for t in range(len(hp_steps) - 1):
                            wt = float(t + 1) / float(len(hp_steps))
                            loss_inter += wt * (
                                F.l1_loss(hp_steps[t], hp_true) + 
                                args.beta_loss_weight * F.l1_loss(be_steps[t], be_true)
                            )

                # TV损失
                loss_tv = tv_loss_2d(torch.stack([hp_pred_n, be_pred_n], dim=1), mask=mask)

                # 总损失
                loss = (loss_main + 
                       args.lambda_intermediate * loss_inter + 
                       args.lambda_tv * loss_tv)

            # 梯度累积
            grad_accum.step(model, optimizer, scaler, loss)

            # 记录
            train_losses.append(loss.item())
            train_h_losses.append(loss_h.item())
            train_b_losses.append(loss_b.item())

            pbar.set_postfix({
                'loss': f"{np.mean(train_losses):.4f}",
                'h_loss': f"{np.mean(train_h_losses):.4f}",
                'b_loss': f"{np.mean(train_b_losses):.4f}"
            })

        # 学习率调度
        scheduler.step()

        # 验证阶段
        model.eval()
        val_losses = []
        val_h_losses = []
        val_b_losses = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]")
            for b in pbar:
                hp_prior = b["hp_prior"].to(device, dtype=torch.float32)
                be_prior = b["be_prior"].to(device, dtype=torch.float32)
                hp_true = b["hp_true"].to(device, dtype=torch.float32)
                be_true = b["be_true"].to(device, dtype=torch.float32)
                
                grid_xy_norm = b["grid_xy_norm"].to(device, dtype=torch.float32)
                mask = b["grid_valid_mask"].to(device).bool()
                
                path_seq = b["path_seq"].to(device, dtype=torch.float32)
                path_static = b["path_static"].to(device, dtype=torch.float32)
                
                idx = b["path_point_grid_idx"].to(device, dtype=torch.long)
                w = b["path_point_grid_w"].to(device, dtype=torch.float32)
                v = b["path_point_valid"].to(device)

                with autocast(enabled=(device.type == "cuda")):
                    out = model(
                        hp_prior=hp_prior,
                        be_prior=be_prior,
                        grid_xy_norm=grid_xy_norm,
                        grid_valid_mask=b["grid_valid_mask"].to(device),
                        path_seq_feats=path_seq,
                        path_static_feats=path_static,
                        path_point_grid_idx=idx,
                        path_point_grid_w=w,
                        path_point_valid=v,
                    )

                    hp_pred = out["hprime_pred"]
                    be_pred = out["beta_pred"]

                    loss_h = F.l1_loss(hp_pred * mask.float(), hp_true * mask.float())
                    loss_b = F.l1_loss(be_pred * mask.float(), be_true * mask.float())
                    loss = args.h_loss_weight * loss_h + args.beta_loss_weight * loss_b

                    val_losses.append(loss.item())
                    val_h_losses.append(loss_h.item())
                    val_b_losses.append(loss_b.item())

                pbar.set_postfix({
                    'val_loss': f"{np.mean(val_losses):.4f}",
                    'val_h': f"{np.mean(val_h_losses):.4f}",
                    'val_b': f"{np.mean(val_b_losses):.4f}"
                })

        # 计算平均损失
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_val_h_loss = np.mean(val_h_losses)
        avg_val_b_loss = np.mean(val_b_losses)
        
        current_lr = optimizer.param_groups[0]['lr']

        # 保存历史
        history.append({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'val_h_loss': avg_val_h_loss,
            'val_b_loss': avg_val_b_loss,
            'lr': current_lr
        })

        # 打印结果
        print(f"\nEpoch {epoch:03d}: "
              f"Train Loss: {avg_train_loss:.6f} | "
              f"Val Loss: {avg_val_loss:.6f} (h: {avg_val_h_loss:.6f}, b: {avg_val_b_loss:.6f}) | "
              f"LR: {current_lr:.2e}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss - args.min_delta:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': avg_val_loss,
                'model_hparams': model_hparams,
                'args': vars(args),
                'history': history,
            }
            
            torch.save(checkpoint, args.out)
            print(f"  ✓ Saved best model to {args.out}")
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement.")
            print(f"Best epoch: {best_epoch}, Best val loss: {best_val_loss:.6f}")
            break

        # 定期保存历史
        if epoch % 10 == 0:
            hist_path = args.out.replace('.pt', f'_epoch_{epoch}_history.json')
            with open(hist_path, 'w') as f:
                json.dump(history, f, indent=2)
            print(f"  Saved history to {hist_path}")

    # 训练完成
    print(f"\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
    
    # 保存最终历史
    final_hist_path = args.out.replace('.pt', '_final_history.json')
    with open(final_hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Final history saved to {final_hist_path}")


if __name__ == "__main__":
    main()