# ionosphere_models.py
# -*- coding: utf-8 -*-
"""
电离层模型
- 先验：Ferguson 经验公式（显式依赖 χ、θ、月份）
- 真实：先验 + 空间相关扰动（多尺度 + h'/β 相关 + 可选局地 blob）

关键改动(为U-Net适配):
- 生成/返回 h'/beta 为规则栅格 (H,W)
- save_ionosphere_states 保存为 (num_samples,H,W)，并保存 grid_valid_mask

本次增强(提升可反演性 / 提升 per-sample R² 稳定性 / 改善 β 学习):
1) 将原先 O(N^3) 的全协方差 multivariate_normal 改为
   “白噪声(网格) + 高斯滤波”近似生成相关随机场（快很多，适合 2000 样本）
2) 采用多尺度叠加：大尺度 + 中尺度
3) 引入 dh 与 db 的样本级相关（通常取负相关）
4) 可选加入少量局地 blob 异常，提高样本内方差与可观测结构
"""

import numpy as np
from datetime import datetime
import config as C
from utils_geo import calculate_solar_zenith_angle
from scipy.ndimage import gaussian_filter


def get_rng(seed_offset=0):
    """获取随机数生成器（保证可复现）"""
    return np.random.default_rng(C.SEED + int(seed_offset))


def ferguson_hprime_beta_at(lat_deg, lon_deg, dt):
    """
    Ferguson经验公式计算单点 h′ 与 β
    返回:
        hprime (km), beta (km^-1)
    """
    m = int(dt.month)
    term = 2.0 * np.pi * (m - 0.5) / 12.0

    chi = float(calculate_solar_zenith_angle(lat_deg, lon_deg, dt))
    chi = np.clip(chi, 0.0, np.pi)

    theta = np.radians(lat_deg)

    hprime = 74.37 - 8.097 * np.cos(theta) - 5.772 * np.cos(chi) - 1.213 * np.cos(term)

    beta = (
        0.3849
        - 0.08584 * np.cos(term)
        + 0.1296 * np.sin(term)
        - 0.1658 * np.cos(chi)
        + 0.15
    )

    return float(hprime), float(beta)


# -----------------------------
# Fast correlated random fields
# -----------------------------

def _normalize_field(field: np.ndarray, valid_mask: np.ndarray | None):
    """Zero-mean, unit-std on valid region."""
    if valid_mask is None:
        mu = float(np.mean(field))
        sd = float(np.std(field) + 1e-6)
        return (field - mu) / sd

    m = valid_mask.astype(bool)
    if np.any(m):
        mu = float(np.mean(field[m]))
        sd = float(np.std(field[m]) + 1e-6)
        out = field.copy()
        out[m] = (out[m] - mu) / sd
        # keep invalid as 0 (so it won't affect later)
        out[~m] = 0.0
        return out
    else:
        # no valid -> return zeros
        return np.zeros_like(field, dtype=np.float32)


def generate_correlated_field_on_grid(
    H: int,
    W: int,
    rng: np.random.Generator,
    length_scale_km: float,
    grid_spacing_km: float,
    valid_mask: np.ndarray | None = None,
):
    """
    生成近似空间相关随机场：
      white noise -> Gaussian filter
    length_scale_km 越大越平滑。
    """
    z = rng.normal(0.0, 1.0, size=(H, W)).astype(np.float32)

    # 将“相关长度”映射为滤波 sigma(网格点)
    # 经验设定：sigma_points = length_scale / (2 * grid_spacing)
    # - 1200km, spacing=100 -> sigma=6 (较平滑)
    # - 500km -> sigma=2.5 (中尺度结构)
    sigma_points = max(0.8, float(length_scale_km) / (2.0 * float(grid_spacing_km) + 1e-6))

    f = gaussian_filter(z, sigma=sigma_points, mode="reflect").astype(np.float32)
    f = _normalize_field(f, valid_mask)
    return f.astype(np.float32)


def generate_multiscale_field_on_grid(
    H: int,
    W: int,
    rng: np.random.Generator,
    length_scales_km: list[float],
    weights: list[float],
    grid_spacing_km: float,
    valid_mask: np.ndarray | None = None,
):
    """多尺度叠加的相关随机场（最终再归一化为 unit-std）。"""
    ls = [float(x) for x in length_scales_km]
    w = np.asarray(weights, dtype=np.float32)
    if w.size != len(ls) or w.size == 0:
        raise ValueError("length_scales_km and weights must have same non-zero length")
    w = w / (float(np.sum(w)) + 1e-6)

    acc = np.zeros((H, W), dtype=np.float32)
    for li, wi in zip(ls, w):
        acc += float(wi) * generate_correlated_field_on_grid(
            H, W, rng, length_scale_km=li, grid_spacing_km=grid_spacing_km, valid_mask=valid_mask
        )

    acc = _normalize_field(acc, valid_mask)
    return acc.astype(np.float32)


def add_gaussian_blobs(
    base_field: np.ndarray,
    X_km: np.ndarray,
    Y_km: np.ndarray,
    rng: np.random.Generator,
    valid_mask: np.ndarray | None,
    n_blobs: int,
    sigma_km_range: tuple[float, float],
    amp_range: tuple[float, float],
):
    """在场上叠加若干二维高斯 blob。"""
    H, W = base_field.shape
    out = base_field.copy().astype(np.float32)

    if n_blobs <= 0:
        return out

    if valid_mask is None:
        valid_mask = np.ones((H, W), dtype=np.uint8)

    m = valid_mask.astype(bool)
    valid_idx = np.argwhere(m)
    if valid_idx.size == 0:
        return out

    for _ in range(int(n_blobs)):
        # pick center among valid points
        ii = valid_idx[int(rng.integers(0, len(valid_idx)))]
        cy, cx = int(ii[0]), int(ii[1])
        x0 = float(X_km[cy, cx])
        y0 = float(Y_km[cy, cx])

        sigma = float(rng.uniform(float(sigma_km_range[0]), float(sigma_km_range[1])))
        amp = float(rng.uniform(float(amp_range[0]), float(amp_range[1])))

        d2 = (X_km - x0) ** 2 + (Y_km - y0) ** 2
        blob = np.exp(-0.5 * d2 / (sigma ** 2 + 1e-6)).astype(np.float32)

        out[m] = out[m] + amp * blob[m]

    return out.astype(np.float32)


def generate_ionosphere_state(grid_data, sample_idx, rng):
    """
    为单个样本生成：
    - prior: Ferguson 网格 (H,W)
    - true: prior + 多尺度空间相关扰动 + (h',beta)相关 + blob (H,W)
    """
    print(f"生成样本 {sample_idx} 的电离层状态...")

    year = int(rng.integers(2020, 2026))
    month = int(rng.choice(C.DAYTIME_MONTHS))
    day = int(rng.choice(C.DAYTIME_DAYS))
    hour = int(rng.choice(C.DAYTIME_HOURS))
    dt = datetime(year, month, day, hour, 0, 0)

    grid_latlon = np.asarray(grid_data['latlon_coords'], dtype=np.float32)  # (N,2)
    H, W = map(int, grid_data['grid_shape'])
    N = H * W
    assert grid_latlon.shape[0] == N

    valid_mask = grid_data.get('valid_mask', None)
    if valid_mask is None:
        valid_mask = np.ones((H, W), dtype=np.uint8)
    valid_mask = valid_mask.astype(np.uint8)

    # --- prior from Ferguson (point-wise) ---
    hprime_prior_1d = np.zeros(N, dtype=np.float32)
    beta_prior_1d = np.zeros(N, dtype=np.float32)
    for i in range(N):
        lat, lon = float(grid_latlon[i, 0]), float(grid_latlon[i, 1])
        hp, be = ferguson_hprime_beta_at(lat, lon, dt)
        hprime_prior_1d[i] = hp
        beta_prior_1d[i] = be

    hprime_prior = hprime_prior_1d.reshape(H, W).astype(np.float32)
    beta_prior = beta_prior_1d.reshape(H, W).astype(np.float32)

    # --- prepare coordinate grids (km) for blobs ---
    x_grid = np.asarray(grid_data['x_grid'], dtype=np.float32)  # (W,)
    y_grid = np.asarray(grid_data['y_grid'], dtype=np.float32)  # (H,)
    X_km, Y_km = np.meshgrid(x_grid, y_grid)  # (H,W)

    # --- correlated perturbations (multi-scale) ---
    length_scales = getattr(C, "PERTURB_LENGTH_SCALES_KM", [1200.0])
    weights = getattr(C, "PERTURB_SCALE_WEIGHTS", [1.0])

    dh_unit = generate_multiscale_field_on_grid(
        H, W, rng,
        length_scales_km=list(length_scales),
        weights=list(weights),
        grid_spacing_km=float(C.GRID_SPACING_KM),
        valid_mask=valid_mask,
    )

    # independent field for beta mixing
    e_unit = generate_multiscale_field_on_grid(
        H, W, rng,
        length_scales_km=list(length_scales),
        weights=list(weights),
        grid_spacing_km=float(C.GRID_SPACING_KM),
        valid_mask=valid_mask,
    )

    rho_lo, rho_hi = getattr(C, "HB_PERTURB_CORR_RANGE", (-0.7, -0.4))
    rho = float(rng.uniform(float(rho_lo), float(rho_hi)))
    rho = float(np.clip(rho, -0.98, 0.98))

    db_unit = rho * dh_unit + float(np.sqrt(1.0 - rho ** 2)) * e_unit
    db_unit = _normalize_field(db_unit, valid_mask).astype(np.float32)

    # --- optional blobs (increase sample-internal variance / local structure) ---
    bmin, bmax = getattr(C, "BLOB_COUNT_RANGE", (0, 0))
    n_blobs = int(rng.integers(int(bmin), int(bmax) + 1)) if bmax >= bmin else 0

    if n_blobs > 0:
        dh_unit = add_gaussian_blobs(
            dh_unit, X_km, Y_km, rng, valid_mask,
            n_blobs=n_blobs,
            sigma_km_range=getattr(C, "BLOB_SIGMA_KM_RANGE", (300.0, 600.0)),
            amp_range=(-1.0, 1.0),  # blob on unit field; final amplitude handled below
        )
        # beta blobs correlated with h blobs but still allow sign variability via amp ranges below
        db_unit = add_gaussian_blobs(
            db_unit, X_km, Y_km, rng, valid_mask,
            n_blobs=n_blobs,
            sigma_km_range=getattr(C, "BLOB_SIGMA_KM_RANGE", (300.0, 600.0)),
            amp_range=(-1.0, 1.0),
        )
        dh_unit = _normalize_field(dh_unit, valid_mask).astype(np.float32)
        db_unit = _normalize_field(db_unit, valid_mask).astype(np.float32)

    # --- sample amplitudes ---
    h_amp_lo, h_amp_hi = getattr(C, "HPRIME_PERTURB_AMPLITUDE_KM", (3.0, 6.0))
    b_amp_lo, b_amp_hi = getattr(C, "BETA_PERTURB_AMPLITUDE_KM_INV", (0.03, 0.09))
    hprime_perturb_amplitude = float(rng.uniform(float(h_amp_lo), float(h_amp_hi)))
    beta_perturb_amplitude = float(rng.uniform(float(b_amp_lo), float(b_amp_hi)))

    # --- apply perturbations ---
    hprime_true = hprime_prior + dh_unit * hprime_perturb_amplitude
    beta_true = beta_prior + db_unit * beta_perturb_amplitude

    # --- apply blob amplitudes in physical units (optional, separate from unit-field blobs) ---
    if n_blobs > 0:
        hp_blob_lo, hp_blob_hi = getattr(C, "BLOB_HPRIME_AMP_KM_RANGE", (-2.0, 2.0))
        be_blob_lo, be_blob_hi = getattr(C, "BLOB_BETA_AMP_KM_INV_RANGE", (-0.02, 0.02))
        # Use separate draws so that blob sign/amplitude differs from global perturb amplitude
        hp_blob_amp = float(rng.uniform(float(hp_blob_lo), float(hp_blob_hi)))
        be_blob_amp = float(rng.uniform(float(be_blob_lo), float(be_blob_hi)))
        # Reuse normalized blobs embedded in dh_unit/db_unit shape by adding low-rank-ish components:
        # Here we create another smooth blob field to avoid “double normalize” complexity.
        blob_field = generate_correlated_field_on_grid(
            H, W, rng,
            length_scale_km=float(np.mean(getattr(C, "BLOB_SIGMA_KM_RANGE", (300.0, 600.0)))),
            grid_spacing_km=float(C.GRID_SPACING_KM),
            valid_mask=valid_mask
        )
        hprime_true = hprime_true + hp_blob_amp * blob_field
        beta_true = beta_true + be_blob_amp * blob_field

    # --- clip to physical ranges ---
    hprime_prior = np.clip(hprime_prior, C.HPRIME_RANGE_DAY[0], C.HPRIME_RANGE_DAY[1]).astype(np.float32)
    beta_prior = np.clip(beta_prior, C.BETA_RANGE_DAY[0], C.BETA_RANGE_DAY[1]).astype(np.float32)
    hprime_true = np.clip(hprime_true, C.HPRIME_RANGE_DAY[0], C.HPRIME_RANGE_DAY[1]).astype(np.float32)
    beta_true = np.clip(beta_true, C.BETA_RANGE_DAY[0], C.BETA_RANGE_DAY[1]).astype(np.float32)

    metadata = {
        'sample_idx': int(sample_idx),
        'datetime': dt.strftime('%Y-%m-%d %H:%M:%S'),
        'year': year, 'month': month, 'day': day, 'hour': hour,

        'hprime_prior_mean': float(np.mean(hprime_prior[valid_mask.astype(bool)])),
        'hprime_prior_std': float(np.std(hprime_prior[valid_mask.astype(bool)])),
        'beta_prior_mean': float(np.mean(beta_prior[valid_mask.astype(bool)])),
        'beta_prior_std': float(np.std(beta_prior[valid_mask.astype(bool)])),

        'hprime_true_mean': float(np.mean(hprime_true[valid_mask.astype(bool)])),
        'hprime_true_std': float(np.std(hprime_true[valid_mask.astype(bool)])),
        'beta_true_mean': float(np.mean(beta_true[valid_mask.astype(bool)])),
        'beta_true_std': float(np.std(beta_true[valid_mask.astype(bool)])),

        'perturb_method': 'multiscale_gaussian_filter',
        'length_scales_km': str(list(map(float, length_scales))),
        'length_scale_weights': str(list(map(float, weights))),
        'hb_corr_rho': float(rho),
        'hprime_perturb_amplitude_km': float(hprime_perturb_amplitude),
        'beta_perturb_amplitude_km_inv': float(beta_perturb_amplitude),
        'n_blobs': int(n_blobs),
    }

    print(
        f"  时间: {dt}, prior h'均值: {metadata['hprime_prior_mean']:.2f} km, "
        f"true h'均值: {metadata['hprime_true_mean']:.2f} km, "
        f"prior β均值: {metadata['beta_prior_mean']:.3f}, true β均值: {metadata['beta_true_mean']:.3f}, "
        f"rho(h,b)={rho:.2f}, blobs={n_blobs}"
    )

    return hprime_prior, beta_prior, hprime_true, beta_true, metadata


def generate_all_ionosphere_states(grid_data, num_samples):
    """生成所有样本的电离层状态（prior & true），输出 (S,H,W)"""
    print(f"开始生成 {num_samples} 个电离层状态样本...")
    print("扰动方法: 多尺度相关随机场（white noise + gaussian filter） + h'/β 相关 + 可选 blob")
    print(f"length_scales_km={getattr(C, 'PERTURB_LENGTH_SCALES_KM', [1200.0])}")

    H, W = map(int, grid_data['grid_shape'])

    all_hprime_prior = np.zeros((num_samples, H, W), dtype=np.float32)
    all_beta_prior = np.zeros((num_samples, H, W), dtype=np.float32)
    all_hprime_true = np.zeros((num_samples, H, W), dtype=np.float32)
    all_beta_true = np.zeros((num_samples, H, W), dtype=np.float32)

    all_metadata = []

    for sample_idx in range(num_samples):
        rng = get_rng(sample_idx * 1000)
        hp_p, be_p, hp_t, be_t, meta = generate_ionosphere_state(grid_data, sample_idx, rng)
        all_hprime_prior[sample_idx] = hp_p
        all_beta_prior[sample_idx] = be_p
        all_hprime_true[sample_idx] = hp_t
        all_beta_true[sample_idx] = be_t
        all_metadata.append(meta)

        if (sample_idx + 1) % 10 == 0:
            print(f"已生成 {sample_idx + 1}/{num_samples} 个样本")

    print("所有电离层状态生成完成!")
    return all_hprime_prior, all_beta_prior, all_hprime_true, all_beta_true, all_metadata


def save_ionosphere_states(
    grid_data,
    all_hprime_prior,
    all_beta_prior,
    all_hprime_true,
    all_beta_true,
    all_metadata,
    filename
):
    """保存电离层状态到HDF5文件（规则栅格版本）"""
    import h5py

    print(f"保存电离层状态到 {filename}...")

    H, W = map(int, grid_data['grid_shape'])
    valid_mask = grid_data.get('valid_mask', None)
    if valid_mask is None:
        valid_mask = np.ones((H, W), dtype=np.uint8)

    with h5py.File(filename, 'w') as f:
        f.create_dataset('grid_proj_coords', data=grid_data['proj_coords'], compression='gzip')
        f.create_dataset('grid_latlon_coords', data=grid_data['latlon_coords'], compression='gzip')
        f.create_dataset('grid_shape', data=np.array([H, W], dtype=np.int32))
        f.create_dataset('x_grid', data=grid_data['x_grid'])
        f.create_dataset('y_grid', data=grid_data['y_grid'])
        f.create_dataset('grid_valid_mask', data=valid_mask.astype(np.uint8), compression='gzip')

        f.create_dataset('hprime_prior_grid', data=all_hprime_prior, compression='gzip')
        f.create_dataset('beta_prior_grid', data=all_beta_prior, compression='gzip')
        f.create_dataset('hprime_true_grid', data=all_hprime_true, compression='gzip')
        f.create_dataset('beta_true_grid', data=all_beta_true, compression='gzip')

        metadata_group = f.create_group('metadata')
        for i, meta in enumerate(all_metadata):
            sg = metadata_group.create_group(f'sample_{i:04d}')
            for key, value in meta.items():
                sg.attrs[key] = value

        f.attrs['num_samples'] = int(len(all_hprime_true))
        f.attrs['num_grid_points'] = int(H * W)
        f.attrs['grid_shape_h'] = int(H)
        f.attrs['grid_shape_w'] = int(W)
        f.attrs['grid_spacing_km'] = float(C.GRID_SPACING_KM)
        f.attrs['projection'] = 'North_America_Equidistant_Conic'
        f.attrs['perturb_method'] = 'multiscale_gaussian_filter'
        f.attrs['note'] = 'Saved both prior (Ferguson) and true (prior+multiscale correlated perturbation) on regular grid'

    print(f"电离层状态已保存到 {filename}")