# predict_gnn_imager.py
# -*- coding: utf-8 -*-
"""
Inference + visualization for multi-stage UNet refiner on regular grid.

本版改动：
1) 支持加载 EMA 权重（如果 checkpoint 内包含 ema_state_dict，默认优先用 EMA）。
2) path_static 与训练保持一致：可根据 feature_scaling.static_log_transform 决定是否对 freq/power 做 log10。
3) 计算 NormMAE 时优先使用 checkpoint 内 feature_scaling 的 hprime_range/beta_range，避免 config 不一致。
"""

from __future__ import annotations

import os
import sys
import argparse
import math
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import warnings
warnings.filterwarnings("ignore")

import torch
from scipy import stats

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

import config as C
from gnn_imager_models import DRegionImagerUNetRefiner


# ---------- feature helpers (must match training) ----------

def phase_to_sincos_deg(phase_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rad = phase_deg.astype(np.float32) * (math.pi / 180.0)
    return np.sin(rad).astype(np.float32), np.cos(rad).astype(np.float32)


def wrap_phase_delta_deg(delta_deg: np.ndarray) -> np.ndarray:
    dd = ((delta_deg + 180.0) % 360.0) - 180.0
    return dd.astype(np.float32)


def unwrap_phase_deg_to_rad(phase_deg: np.ndarray) -> np.ndarray:
    rad = np.deg2rad(phase_deg.astype(np.float32))
    return np.unwrap(rad, axis=1).astype(np.float32)


def transform_path_static(path_static_raw: np.ndarray, log_transform: bool) -> np.ndarray:
    ps = np.array(path_static_raw, dtype=np.float32, copy=True)
    if log_transform:
        ps[..., 4] = np.log10(np.maximum(ps[..., 4], 1e-6))
        ps[..., 5] = np.log10(np.maximum(ps[..., 5], 1e-6))
    return ps


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


# ---------- load model ----------

def load_model(ckpt_path: str, device: torch.device, use_ema: bool = True):
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    hparams = ckpt.get("model_hparams", None)
    if not hparams:
        raise ValueError("Checkpoint missing model_hparams")

    model = DRegionImagerUNetRefiner(**hparams).to(device)

    ema_sd = ckpt.get("ema_state_dict", None)
    if use_ema and (ema_sd is not None):
        print("Using EMA weights from checkpoint.")
        model.load_state_dict(ema_sd, strict=False)
    else:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)

    model.eval()

    feature_scaling = ckpt.get("feature_scaling", {})
    print(f"Loaded model. Params={sum(p.numel() for p in model.parameters()):,}")
    if "epoch" in ckpt:
        print(f"Checkpoint epoch: {ckpt['epoch']}")
    if "val_pred" in ckpt:
        print(f"Checkpoint val_pred: {ckpt['val_pred']:.6f}")
    if "val_total" in ckpt:
        print(f"Checkpoint val_total: {ckpt['val_total']:.6f}")
    if "best_metric" in ckpt and "metric_name" in ckpt:
        print(f"Best metric[{ckpt['metric_name']}]: {ckpt['best_metric']:.6f}")

    return model, feature_scaling


# ---------- dataset meta for prediction ----------

def load_dataset_meta(h5_path: str, split="val", train_frac=0.9, seed=42, static_log_transform: bool = False):
    with h5py.File(h5_path, "r") as f:
        num_samples = int(f.attrs["num_samples"])
        num_paths = int(f.attrs["num_paths"])
        path_segments = int(f.attrs["path_segments"])
        H, W = f["grid_shape"][:].astype(np.int32).tolist()

        rng = np.random.default_rng(seed)
        idx = np.arange(num_samples)
        rng.shuffle(idx)
        n_train = int(round(num_samples * train_frac))
        indices = idx[:n_train] if split == "train" else idx[n_train:]

        x_grid = f["x_grid"][:].astype(np.float32)
        y_grid = f["y_grid"][:].astype(np.float32)
        grid_valid_mask = f.get("grid_valid_mask", None)
        if grid_valid_mask is not None:
            grid_valid_mask = grid_valid_mask[:].astype(np.uint8)
        else:
            grid_valid_mask = np.ones((H, W), dtype=np.uint8)

        path_seg_dist = f["path_segment_distances_km"][:].astype(np.float32)  # (P,L)

        path_point_grid_idx = f["path_point_grid_idx"][:].astype(np.int64)
        path_point_grid_w = f["path_point_grid_w"][:].astype(np.float32)
        path_point_valid = f["path_point_valid"][:].astype(np.uint8)

        has_prior_sim = ("amplitude_x_prior_sim" in f)
        has_path_success = ("path_success_obs" in f)

        # path static
        path_static = []
        path_info = []
        pg = f["paths"]
        for pi in range(num_paths):
            g = pg[f"path_{pi:02d}"]
            tx_name = g.attrs.get("tx_name", "")
            rx_name = g.attrs.get("rx_name", "")
            tx_lat = float(g.attrs["tx_lat"])
            tx_lon = float(g.attrs["tx_lon"])
            rx_lat = float(g.attrs["rx_lat"])
            rx_lon = float(g.attrs["rx_lon"])
            freq_hz = float(g.attrs["frequency_hz"])
            power_kw = float(g.attrs["power_kw"])
            path_static.append([tx_lat, tx_lon, rx_lat, rx_lon, freq_hz, power_kw])
            path_info.append({
                "tx_name": tx_name.decode() if isinstance(tx_name, (bytes, np.bytes_)) else str(tx_name),
                "rx_name": rx_name.decode() if isinstance(rx_name, (bytes, np.bytes_)) else str(rx_name),
                "tx_lat": tx_lat, "tx_lon": tx_lon,
                "rx_lat": rx_lat, "rx_lon": rx_lon,
                "freq_hz": freq_hz,
                "power_kw": power_kw
            })
        path_static = np.asarray(path_static, dtype=np.float32)
        path_static = transform_path_static(path_static, log_transform=static_log_transform)

        static_mean = path_static.mean(axis=0, keepdims=True)
        static_std = path_static.std(axis=0, keepdims=True) + 1e-6

    # grid coord norm
    X, Y = np.meshgrid(x_grid, y_grid)
    xm, xs = float(X.mean()), float(X.std() + 1e-6)
    ym, ys = float(Y.mean()), float(Y.std() + 1e-6)
    grid_xy_norm = np.stack([((X - xm) / xs).astype(np.float32), ((Y - ym) / ys).astype(np.float32)], axis=0)

    return {
        "h5_path": h5_path,
        "indices": indices,
        "num_paths": num_paths,
        "path_segments": path_segments,
        "grid_shape": (H, W),
        "x_grid": x_grid,
        "y_grid": y_grid,
        "grid_valid_mask": grid_valid_mask,
        "grid_xy_norm": grid_xy_norm,
        "path_seg_dist": path_seg_dist,
        "path_point_grid_idx": path_point_grid_idx,
        "path_point_grid_w": path_point_grid_w,
        "path_point_valid": path_point_valid,
        "path_static": path_static,
        "path_info": path_info,
        "static_mean": static_mean,
        "static_std": static_std,
        "has_prior_sim": bool(has_prior_sim),
        "has_path_success": bool(has_path_success),
        "static_log_transform": bool(static_log_transform),
    }


def calculate_r2_score(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 1.0
    return 1.0 - (ss_res / ss_tot)


def predict_single_sample(model, ds, sample_idx: int, device: torch.device, feature_scaling: dict):
    with h5py.File(ds["h5_path"], "r") as f:
        hp_prior = f["hprime_prior_grid"][sample_idx].astype(np.float32)  # (H,W)
        be_prior = f["beta_prior_grid"][sample_idx].astype(np.float32)
        hp_true = f["hprime_true_grid"][sample_idx].astype(np.float32)
        be_true = f["beta_true_grid"][sample_idx].astype(np.float32)

        amp_x = f["amplitude_x"][sample_idx].astype(np.float32)
        pha_x = f["phase_x"][sample_idx].astype(np.float32)
        amp_y = f["amplitude_y"][sample_idx].astype(np.float32)
        pha_y = f["phase_y"][sample_idx].astype(np.float32)
        amp_z = f["amplitude_z"][sample_idx].astype(np.float32)
        pha_z = f["phase_z"][sample_idx].astype(np.float32)

        if ds.get("has_prior_sim", False) and ("amplitude_x_prior_sim" in f):
            amp_x_p = f["amplitude_x_prior_sim"][sample_idx].astype(np.float32)
            pha_x_p = f["phase_x_prior_sim"][sample_idx].astype(np.float32)
            amp_y_p = f["amplitude_y_prior_sim"][sample_idx].astype(np.float32)
            pha_y_p = f["phase_y_prior_sim"][sample_idx].astype(np.float32)
            amp_z_p = f["amplitude_z_prior_sim"][sample_idx].astype(np.float32)
            pha_z_p = f["phase_z_prior_sim"][sample_idx].astype(np.float32)
        else:
            amp_x_p = pha_x_p = amp_y_p = pha_y_p = amp_z_p = pha_z_p = None

        if ds.get("has_path_success", False) and ("path_success_obs" in f):
            path_success_obs = f["path_success_obs"][sample_idx].astype(np.uint8)
        else:
            path_success_obs = np.ones((ds["num_paths"],), dtype=np.uint8)

    amp_scale_db = float(feature_scaling.get("amp_scale_db", 50.0))

    # prior params along path
    H, W = ds["grid_shape"]
    hp_flat = hp_prior.reshape(-1)
    be_flat = be_prior.reshape(-1)
    idxp = ds["path_point_grid_idx"]
    wp = ds["path_point_grid_w"]
    hp_prior_path = np.sum(hp_flat[idxp] * wp, axis=-1).astype(np.float32)
    be_prior_path = np.sum(be_flat[idxp] * wp, axis=-1).astype(np.float32)

    # 使用 checkpoint 模型内的 mid/half（避免 config 不一致）
    h_mid = float(model.h_mid.detach().cpu().reshape(-1)[0].item())
    h_half = float(model.h_half.detach().cpu().reshape(-1)[0].item())
    b_mid = float(model.b_mid.detach().cpu().reshape(-1)[0].item())
    b_half = float(model.b_half.detach().cpu().reshape(-1)[0].item())

    seq = build_path_seq_features(
        amp_x, pha_x, amp_y, pha_y, amp_z, pha_z,
        dist_km=ds["path_seg_dist"],
        amp_scale_db=amp_scale_db,
        amp_x_prior_sim=amp_x_p, pha_x_prior_sim=pha_x_p,
        amp_y_prior_sim=amp_y_p, pha_y_prior_sim=pha_y_p,
        amp_z_prior_sim=amp_z_p, pha_z_prior_sim=pha_z_p,
        hp_prior_path=hp_prior_path,
        be_prior_path=be_prior_path,
        h_mid=h_mid, h_half=h_half,
        b_mid=b_mid, b_half=b_half,
    )

    ps = ((ds["path_static"] - ds["static_mean"]) / ds["static_std"]).astype(np.float32)

    # mask failed paths by setting point_valid=0
    path_point_valid = ds["path_point_valid"].copy()
    fail = (path_success_obs <= 0)
    if np.any(fail):
        path_point_valid[fail, :] = 0

    hp_prior_t = torch.from_numpy(hp_prior).to(device)
    be_prior_t = torch.from_numpy(be_prior).to(device)
    grid_xy_norm_t = torch.from_numpy(ds["grid_xy_norm"]).to(device)
    grid_valid_mask_t = torch.from_numpy(ds["grid_valid_mask"]).to(device)
    path_seq_t = torch.from_numpy(seq).to(device)
    path_static_t = torch.from_numpy(ps).to(device)
    idx_t = torch.from_numpy(ds["path_point_grid_idx"]).to(device)
    w_t = torch.from_numpy(ds["path_point_grid_w"]).to(device)
    v_t = torch.from_numpy(path_point_valid).to(device)

    with torch.no_grad():
        out = model(
            hp_prior=hp_prior_t,
            be_prior=be_prior_t,
            grid_xy_norm=grid_xy_norm_t,
            grid_valid_mask=grid_valid_mask_t,
            path_seq_feats=path_seq_t,
            path_static_feats=path_static_t,
            path_point_grid_idx=idx_t,
            path_point_grid_w=w_t,
            path_point_valid=v_t,
        )
        hp_pred = out["hprime_pred"].detach().cpu().numpy().astype(np.float32).squeeze(0)
        be_pred = out["beta_pred"].detach().cpu().numpy().astype(np.float32).squeeze(0)
        hp_coarse = out.get("hprime_coarse_up", None)
        be_coarse = out.get("beta_coarse_up", None)
        obs_wsum = out.get("obs_wsum", None)

        if hp_coarse is not None:
            hp_coarse = hp_coarse.detach().cpu().numpy().astype(np.float32).squeeze(0)
            be_coarse = be_coarse.detach().cpu().numpy().astype(np.float32).squeeze(0)
        if obs_wsum is not None:
            obs_wsum = obs_wsum.detach().cpu().numpy().astype(np.float32)

    # residuals
    hp_residual_prior = hp_true - hp_prior
    be_residual_prior = be_true - be_prior
    hp_residual_pred = hp_true - hp_pred
    be_residual_pred = be_true - be_pred

    mask = ds["grid_valid_mask"].astype(bool)
    h_mae = float(np.mean(np.abs(hp_pred[mask] - hp_true[mask])))
    b_mae = float(np.mean(np.abs(be_pred[mask] - be_true[mask])))

    h_prior_mae = float(np.mean(np.abs(hp_prior[mask] - hp_true[mask])))
    b_prior_mae = float(np.mean(np.abs(be_prior[mask] - be_true[mask])))

    near_mask = None
    if obs_wsum is not None:
        wmap = obs_wsum.squeeze()
        if wmap.ndim == 3:
            wmap = wmap[0]
        thr = 0.15 * float(np.max(wmap) + 1e-6)
        near_mask = (wmap > thr) & mask

    if near_mask is not None and np.any(near_mask):
        h_mae_near = float(np.mean(np.abs(hp_pred[near_mask] - hp_true[near_mask])))
        b_mae_near = float(np.mean(np.abs(be_pred[near_mask] - be_true[near_mask])))
        h_prior_mae_near = float(np.mean(np.abs(hp_prior[near_mask] - hp_true[near_mask])))
        b_prior_mae_near = float(np.mean(np.abs(be_prior[near_mask] - be_true[near_mask])))
    else:
        h_mae_near = b_mae_near = h_prior_mae_near = b_prior_mae_near = float("nan")

    # prefer checkpoint ranges if provided
    hr = feature_scaling.get("hprime_range", [float(C.HPRIME_RANGE_DAY[0]), float(C.HPRIME_RANGE_DAY[1])])
    br = feature_scaling.get("beta_range", [float(C.BETA_RANGE_DAY[0]), float(C.BETA_RANGE_DAY[1])])
    h_range = float(hr[1] - hr[0])
    b_range = float(br[1] - br[0])

    h_rel = h_mae / max(1e-6, h_range) * 100.0
    b_rel = b_mae / max(1e-6, b_range) * 100.0
    norm_mae = 0.5 * (h_rel + b_rel)

    h_r2 = float(calculate_r2_score(hp_true[mask], hp_pred[mask]))
    b_r2 = float(calculate_r2_score(be_true[mask], be_pred[mask]))

    return {
        "sample_idx": int(sample_idx),
        "hprime_prior": hp_prior,
        "beta_prior": be_prior,
        "hprime_true": hp_true,
        "beta_true": be_true,
        "hprime_pred": hp_pred,
        "beta_pred": be_pred,
        "hprime_coarse_up": hp_coarse,
        "beta_coarse_up": be_coarse,
        "obs_wsum": obs_wsum,
        "hprime_residual_prior": hp_residual_prior,
        "beta_residual_prior": be_residual_prior,
        "hprime_residual_pred": hp_residual_pred,
        "beta_residual_pred": be_residual_pred,
        "hprime_mae": h_mae,
        "beta_mae": b_mae,
        "hprime_prior_mae": h_prior_mae,
        "beta_prior_mae": b_prior_mae,
        "hprime_mae_near": h_mae_near,
        "beta_mae_near": b_mae_near,
        "hprime_prior_mae_near": h_prior_mae_near,
        "beta_prior_mae_near": b_prior_mae_near,
        "hprime_r2": h_r2,
        "beta_r2": b_r2,
        "normalized_mae": float(norm_mae),
        "x_grid": ds["x_grid"],
        "y_grid": ds["y_grid"],
        "grid_valid_mask": ds["grid_valid_mask"],
        "path_info": ds["path_info"],
    }


# ---------- visualization ----------

def plot_maps(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    x = result["x_grid"]
    y = result["y_grid"]
    X, Y = np.meshgrid(x, y)

    mask = result["grid_valid_mask"].astype(bool)

    def apply_mask(a):
        aa = a.copy()
        aa[~mask] = np.nan
        return aa

    hp_t = apply_mask(result["hprime_true"])
    hp_p = apply_mask(result["hprime_pred"])
    hp_pr = apply_mask(result["hprime_prior"])
    be_t = apply_mask(result["beta_true"])
    be_p = apply_mask(result["beta_pred"])
    be_pr = apply_mask(result["beta_prior"])

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    im = axes[0, 0].pcolormesh(X, Y, hp_t, shading="auto", cmap="viridis",
                               vmin=C.HPRIME_RANGE_DAY[0], vmax=C.HPRIME_RANGE_DAY[1])
    axes[0, 0].set_title("True h'")
    plt.colorbar(im, ax=axes[0, 0], label="km")

    im = axes[0, 1].pcolormesh(X, Y, hp_pr, shading="auto", cmap="viridis",
                               vmin=C.HPRIME_RANGE_DAY[0], vmax=C.HPRIME_RANGE_DAY[1])
    axes[0, 1].set_title(f"Prior h' (MAE={result['hprime_prior_mae']:.3f} km)")
    plt.colorbar(im, ax=axes[0, 1], label="km")

    im = axes[0, 2].pcolormesh(X, Y, hp_p, shading="auto", cmap="viridis",
                               vmin=C.HPRIME_RANGE_DAY[0], vmax=C.HPRIME_RANGE_DAY[1])
    axes[0, 2].set_title(f"Pred h' (MAE={result['hprime_mae']:.3f} km, R²={result['hprime_r2']:.3f})")
    plt.colorbar(im, ax=axes[0, 2], label="km")

    im = axes[1, 0].pcolormesh(X, Y, be_t, shading="auto", cmap="plasma",
                               vmin=C.BETA_RANGE_DAY[0], vmax=C.BETA_RANGE_DAY[1])
    axes[1, 0].set_title("True β")
    plt.colorbar(im, ax=axes[1, 0], label="km^-1")

    im = axes[1, 1].pcolormesh(X, Y, be_pr, shading="auto", cmap="plasma",
                               vmin=C.BETA_RANGE_DAY[0], vmax=C.BETA_RANGE_DAY[1])
    axes[1, 1].set_title(f"Prior β (MAE={result['beta_prior_mae']:.4f} km^-1)")
    plt.colorbar(im, ax=axes[1, 1], label="km^-1")

    im = axes[1, 2].pcolormesh(X, Y, be_p, shading="auto", cmap="plasma",
                               vmin=C.BETA_RANGE_DAY[0], vmax=C.BETA_RANGE_DAY[1])
    axes[1, 2].set_title(f"Pred β (MAE={result['beta_mae']:.4f} km^-1, R²={result['beta_r2']:.3f})")
    plt.colorbar(im, ax=axes[1, 2], label="km^-1")

    plt.suptitle(
        f"Sample {result['sample_idx']}: NormMAE={result['normalized_mae']:.4f}\n"
        f"Prior MAE: h'={result['hprime_prior_mae']:.3f} km, β={result['beta_prior_mae']:.4f} km^-1\n"
        f"Near-path MAE: prior h'={result['hprime_prior_mae_near']:.3f}, pred h'={result['hprime_mae_near']:.3f} | "
        f"prior β={result['beta_prior_mae_near']:.4f}, pred β={result['beta_mae_near']:.4f}",
        y=1.02
    )
    plt.tight_layout()

    out = os.path.join(out_dir, f"sample_{result['sample_idx']:04d}_maps.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def plot_residual_maps(result, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    x = result["x_grid"]
    y = result["y_grid"]
    X, Y = np.meshgrid(x, y)

    mask = result["grid_valid_mask"].astype(bool)

    def apply_mask(a):
        aa = a.copy()
        aa[~mask] = np.nan
        return aa

    hp_res_prior = apply_mask(result["hprime_residual_prior"])
    be_res_prior = apply_mask(result["beta_residual_prior"])
    hp_res_pred = apply_mask(result["hprime_residual_pred"])
    be_res_pred = apply_mask(result["beta_residual_pred"])

    hp_res_prior_stats = {"mean": float(np.nanmean(hp_res_prior)), "std": float(np.nanstd(hp_res_prior)), "max": float(np.nanmax(np.abs(hp_res_prior)))}
    be_res_prior_stats = {"mean": float(np.nanmean(be_res_prior)), "std": float(np.nanstd(be_res_prior)), "max": float(np.nanmax(np.abs(be_res_prior)))}
    hp_res_pred_stats = {"mean": float(np.nanmean(hp_res_pred)), "std": float(np.nanstd(hp_res_pred)), "max": float(np.nanmax(np.abs(hp_res_pred)))}
    be_res_pred_stats = {"mean": float(np.nanmean(be_res_pred)), "std": float(np.nanstd(be_res_pred)), "max": float(np.nanmax(np.abs(be_res_pred)))}

    h_res_max = max(hp_res_prior_stats["max"], hp_res_pred_stats["max"])
    b_res_max = max(be_res_prior_stats["max"], be_res_pred_stats["max"])
    h_vmin, h_vmax = (-h_res_max, h_res_max) if h_res_max != 0 else (-1, 1)
    b_vmin, b_vmax = (-b_res_max, b_res_max) if b_res_max != 0 else (-0.1, 0.1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    cmap_residual = "RdBu_r"

    im1 = axes[0, 0].pcolormesh(X, Y, hp_res_prior, shading="auto", cmap=cmap_residual, vmin=h_vmin, vmax=h_vmax)
    axes[0, 0].set_title(f"h' Residual (True - Prior)\nMean={hp_res_prior_stats['mean']:.3f}, Std={hp_res_prior_stats['std']:.3f}")
    plt.colorbar(im1, ax=axes[0, 0], label="km")

    im2 = axes[0, 1].pcolormesh(X, Y, hp_res_pred, shading="auto", cmap=cmap_residual, vmin=h_vmin, vmax=h_vmax)
    axes[0, 1].set_title(f"h' Residual (True - Pred)\nMean={hp_res_pred_stats['mean']:.3f}, Std={hp_res_pred_stats['std']:.3f}")
    plt.colorbar(im2, ax=axes[0, 1], label="km")

    im3 = axes[1, 0].pcolormesh(X, Y, be_res_prior, shading="auto", cmap=cmap_residual, vmin=b_vmin, vmax=b_vmax)
    axes[1, 0].set_title(f"β Residual (True - Prior)\nMean={be_res_prior_stats['mean']:.5f}, Std={be_res_prior_stats['std']:.5f}")
    plt.colorbar(im3, ax=axes[1, 0], label="km^-1")

    im4 = axes[1, 1].pcolormesh(X, Y, be_res_pred, shading="auto", cmap=cmap_residual, vmin=b_vmin, vmax=b_vmax)
    axes[1, 1].set_title(f"β Residual (True - Pred)\nMean={be_res_pred_stats['mean']:.5f}, Std={be_res_pred_stats['std']:.5f}")
    plt.colorbar(im4, ax=axes[1, 1], label="km^-1")

    plt.suptitle(
        f"Sample {result['sample_idx']} - Residual Maps\n"
        f"Improvement: h' MAE reduced by {result['hprime_prior_mae'] - result['hprime_mae']:.3f} km "
        f"({((result['hprime_prior_mae'] - result['hprime_mae'])/result['hprime_prior_mae']*100):.1f}%)\n"
        f"β MAE reduced by {result['beta_prior_mae'] - result['beta_mae']:.5f} km^-1 "
        f"({((result['beta_prior_mae'] - result['beta_mae'])/result['beta_prior_mae']*100):.1f}%)",
        y=1.02
    )
    plt.tight_layout()

    out = os.path.join(out_dir, f"sample_{result['sample_idx']:04d}_residuals.png")
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved residual maps: {out}")

    residual_stats = {
        "sample_idx": result["sample_idx"],
        "hprime": {
            "prior_residual": hp_res_prior_stats,
            "pred_residual": hp_res_pred_stats,
            "mae_improvement": float(result["hprime_prior_mae"] - result["hprime_mae"]),
            "mae_improvement_percent": float(((result["hprime_prior_mae"] - result["hprime_mae"]) / result["hprime_prior_mae"] * 100))
        },
        "beta": {
            "prior_residual": be_res_prior_stats,
            "pred_residual": be_res_pred_stats,
            "mae_improvement": float(result["beta_prior_mae"] - result["beta_mae"]),
            "mae_improvement_percent": float(((result["beta_prior_mae"] - result["beta_mae"]) / result["beta_prior_mae"] * 100))
        }
    }

    stats_file = os.path.join(out_dir, f"sample_{result['sample_idx']:04d}_residual_stats.json")
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(residual_stats, f, ensure_ascii=False, indent=2)
    print(f"Saved residual stats: {stats_file}")

    return out, residual_stats


def plot_r2_scatter(results_dict, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    all_hp_true, all_hp_pred = [], []
    all_be_true, all_be_pred = [], []

    for result in results_dict.values():
        mask = result["grid_valid_mask"].astype(bool)
        all_hp_true.extend(result["hprime_true"][mask].flatten())
        all_hp_pred.extend(result["hprime_pred"][mask].flatten())
        all_be_true.extend(result["beta_true"][mask].flatten())
        all_be_pred.extend(result["beta_pred"][mask].flatten())

    all_hp_true = np.array(all_hp_true)
    all_hp_pred = np.array(all_hp_pred)
    all_be_true = np.array(all_be_true)
    all_be_pred = np.array(all_be_pred)

    hp_r2 = float(calculate_r2_score(all_hp_true, all_hp_pred))
    be_r2 = float(calculate_r2_score(all_be_true, all_be_pred))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(all_hp_true, all_hp_pred, alpha=0.6, s=10, edgecolors="none")
    min_val = float(min(np.min(all_hp_true), np.min(all_hp_pred)))
    max_val = float(max(np.max(all_hp_true), np.max(all_hp_pred)))
    axes[0].plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, label="y=x")
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_hp_true, all_hp_pred)
    x_fit = np.array([min_val, max_val])
    y_fit = slope * x_fit + intercept
    axes[0].plot(x_fit, y_fit, "g-", alpha=0.8, label=f"Fit: y={slope:.3f}x+{intercept:.3f}")
    axes[0].set_xlabel("True h' (km)")
    axes[0].set_ylabel("Predicted h' (km)")
    axes[0].set_title(f"h' R²: {hp_r2:.3f}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(all_be_true, all_be_pred, alpha=0.6, s=10, edgecolors="none")
    min_val = float(min(np.min(all_be_true), np.min(all_be_pred)))
    max_val = float(max(np.max(all_be_true), np.max(all_be_pred)))
    axes[1].plot([min_val, max_val], [min_val, max_val], "r--", alpha=0.8, label="y=x")
    slope, intercept, r_value, p_value, std_err = stats.linregress(all_be_true, all_be_pred)
    x_fit = np.array([min_val, max_val])
    y_fit = slope * x_fit + intercept
    axes[1].plot(x_fit, y_fit, "g-", alpha=0.8, label=f"Fit: y={slope:.3f}x+{intercept:.3f}")
    axes[1].set_xlabel("True β (km⁻¹)")
    axes[1].set_ylabel("Predicted β (km⁻¹)")
    axes[1].set_title(f"β R²: {be_r2:.3f}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "r2_scores.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved R² plot: {out_path}")

    return hp_r2, be_r2


def convert_to_serializable(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def main():
    ap = argparse.ArgumentParser(description="Predict with multi-stage UNet refiner (regular grid)")
    ap.add_argument("--data", type=str, default=C.FINAL_DATASET_FILE)
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--sample", type=int, default=-1)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--num-samples", type=int, default=-1)
    ap.add_argument("--train-frac", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plot", action="store_true", default=False)
    ap.add_argument("--plot-residuals", action="store_true", default=False, help="Plot residual maps")
    ap.add_argument("--plot-r2", action="store_true", default=False)
    ap.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True, help="Use EMA weights if available in checkpoint")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    model, feature_scaling = load_model(args.ckpt, device, use_ema=bool(args.use_ema))
    static_log_transform = bool(feature_scaling.get("static_log_transform", False))

    ds = load_dataset_meta(args.data, split="val", train_frac=args.train_frac, seed=args.seed, static_log_transform=static_log_transform)
    val_indices = ds["indices"]

    if args.sample >= 0:
        sid = int(args.sample)
        if sid not in val_indices:
            print(f"Warning: sample {sid} not in val split, using {int(val_indices[0])}")
            sid = int(val_indices[0])

        res = predict_single_sample(model, ds, sid, device, feature_scaling)
        print(
            f"Sample {sid}: "
            f"h' MAE={res['hprime_mae']:.3f} km (prior {res['hprime_prior_mae']:.3f}) | "
            f"β MAE={res['beta_mae']:.4f} km^-1 (prior {res['beta_prior_mae']:.4f}) | "
            f"Near-path h' MAE={res['hprime_mae_near']:.3f} (prior {res['hprime_prior_mae_near']:.3f}) | "
            f"Near-path β MAE={res['beta_mae_near']:.4f} (prior {res['beta_prior_mae_near']:.4f}) | "
            f"NormMAE={res['normalized_mae']:.4f} | h' R²={res['hprime_r2']:.3f} | β R²={res['beta_r2']:.3f}"
        )

        h_improvement = ((res["hprime_prior_mae"] - res["hprime_mae"]) / res["hprime_prior_mae"] * 100) if res["hprime_prior_mae"] > 0 else 0
        b_improvement = ((res["beta_prior_mae"] - res["beta_mae"]) / res["beta_prior_mae"] * 100) if res["beta_prior_mae"] > 0 else 0
        print(f"Improvement: h' {h_improvement:.1f}%, β {b_improvement:.1f}%")

        out_dir = args.out or os.path.join(C.PLOT_DIR, f"sample_{sid:04d}_unet_refiner")
        os.makedirs(out_dir, exist_ok=True)
        if args.plot:
            plot_maps(res, out_dir)
        if args.plot_residuals:
            plot_residual_maps(res, out_dir)
        return

    if args.num_samples > 0:
        val_indices = val_indices[:args.num_samples]

    results = {}
    maes = []
    hp_r2_list = []
    be_r2_list = []
    improvements_h = []
    improvements_b = []

    for i, sid in enumerate(val_indices):
        sid = int(sid)
        r = predict_single_sample(model, ds, sid, device, feature_scaling)
        results[sid] = r
        maes.append(r["normalized_mae"])
        hp_r2_list.append(r["hprime_r2"])
        be_r2_list.append(r["beta_r2"])

        if r["hprime_prior_mae"] > 0:
            improvements_h.append(((r["hprime_prior_mae"] - r["hprime_mae"]) / r["hprime_prior_mae"] * 100))
        if r["beta_prior_mae"] > 0:
            improvements_b.append(((r["beta_prior_mae"] - r["beta_mae"]) / r["beta_prior_mae"] * 100))

        print(f"{i+1}/{len(val_indices)} sid={sid} NormMAE={r['normalized_mae']:.4f} h'R²={r['hprime_r2']:.3f} βR²={r['beta_r2']:.3f}", end="\r")
    print()

    best_sid = int(val_indices[int(np.argmin(maes))])
    best_result = results[best_sid]
    print(f"Best sample by Normalized MAE: {best_sid} (={best_result['normalized_mae']:.4f})")
    print(f"Average h' R²: {np.mean(hp_r2_list):.3f} ± {np.std(hp_r2_list):.3f}")
    print(f"Average β R²: {np.mean(be_r2_list):.3f} ± {np.std(be_r2_list):.3f}")
    if improvements_h:
        print(f"Average h' improvement: {np.mean(improvements_h):.1f}% ± {np.std(improvements_h):.1f}%")
    if improvements_b:
        print(f"Average β improvement: {np.mean(improvements_b):.1f}% ± {np.std(improvements_b):.1f}%")

    report = {
        "checkpoint": args.ckpt,
        "use_ema": bool(args.use_ema),
        "n_samples": int(len(val_indices)),
        "best_sample_idx": best_sid,
        "best_normalized_mae": float(best_result["normalized_mae"]),
        "avg_normalized_mae": float(np.mean(maes)),
        "std_normalized_mae": float(np.std(maes)),
        "avg_hprime_r2": float(np.mean(hp_r2_list)),
        "std_hprime_r2": float(np.std(hp_r2_list)),
        "avg_beta_r2": float(np.mean(be_r2_list)),
        "std_beta_r2": float(np.std(be_r2_list)),
        "avg_hprime_improvement_percent": float(np.mean(improvements_h)) if improvements_h else 0.0,
        "avg_beta_improvement_percent": float(np.mean(improvements_b)) if improvements_b else 0.0,
        "feature_scaling": convert_to_serializable(feature_scaling),
    }

    if args.plot_r2:
        overall_hp_r2, overall_be_r2 = plot_r2_scatter(results, C.PLOT_DIR)
        report["overall_hprime_r2"] = float(overall_hp_r2)
        report["overall_beta_r2"] = float(overall_be_r2)
        print(f"Overall h' R²: {overall_hp_r2:.3f}")
        print(f"Overall β R²: {overall_be_r2:.3f}")

    report_path = os.path.join(C.DATA_DIR, "unet_refiner_val_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(convert_to_serializable(report), f, ensure_ascii=False, indent=2)
    print(f"Saved report: {report_path}")

    if args.plot or args.plot_residuals:
        out_dir = args.out or os.path.join(C.PLOT_DIR, f"sample_{best_sid:04d}_unet_refiner")
        os.makedirs(out_dir, exist_ok=True)
        if args.plot:
            plot_maps(best_result, out_dir)
        if args.plot_residuals:
            plot_residual_maps(best_result, out_dir)

    print("\n" + "=" * 80)
    print(f"BEST SAMPLE {best_sid} DETAILS:")
    print(f"  h' Prior MAE: {best_result['hprime_prior_mae']:.3f} km")
    print(f"  h' Pred MAE:  {best_result['hprime_mae']:.3f} km")
    print(f"  h' Improvement: {best_result['hprime_prior_mae'] - best_result['hprime_mae']:.3f} km ({((best_result['hprime_prior_mae'] - best_result['hprime_mae']) / best_result['hprime_prior_mae'] * 100):.1f}%)")
    print(f"  β Prior MAE: {best_result['beta_prior_mae']:.4f} km^-1")
    print(f"  β Pred MAE:  {best_result['beta_mae']:.4f} km^-1")
    print(f"  β Improvement: {best_result['beta_prior_mae'] - best_result['beta_mae']:.5f} km^-1 ({((best_result['beta_prior_mae'] - best_result['beta_mae']) / best_result['beta_prior_mae'] * 100):.1f}%)")
    print(f"  Normalized MAE: {best_result['normalized_mae']:.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()