# compute_vlf_paths.py
# -*- coding: utf-8 -*-
"""
计算VLF路径的电场数据，并保存"先验(prior) + 真实(true) + 观测(电场)"的监督学习数据集

关键改动(为U-Net + 迭代细化适配):
1) 电离层参数 h'/β 采用规则栅格 (S,H,W) 存储
2) 预计算并保存:
   - path_segment_proj_coords_km: (P,L,2)
   - path_point_grid_idx: (P,L,4)   双线性邻点flatten索引
   - path_point_grid_w:   (P,L,4)   双线性权重
   - path_point_valid:    (P,L)     是否落在栅格内
3) 生成LWPC输入时仍沿路径100点(或你配置的 PATH_SEGMENTS)，不改变LWPC流程
4) 从网格采样到路径时优先使用预计算双线性采样(更快、更一致)

新增(为“可学的反演”显著增强):
5) 同时计算并保存“先验电离层(prior)驱动的LWPC电场”（prior-sim），用于构造观测残差特征 (obs - prior-sim)
6) 保存每条路径计算成功掩码，训练时可将失败路径点 valid=0，避免污染
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import h5py
from tqdm import tqdm
import config as C
from grid_utils import create_path_segments, latlon_to_projection, build_bilinear_index_weights
from superlwpc_utils import run_lwpc_for_path

ADD_MEASUREMENT_NOISE = True
SIGMA_AMP_DB = 0.1
SIGMA_PHASE_DEG = 1.0

# 新增：是否额外计算 prior-sim 电场（建议 True）
COMPUTE_PRIOR_SIM_FIELDS = True


def load_ionosphere_states(filename):
    """加载电离层状态数据（prior & true）"""
    print(f"加载电离层状态数据: {filename}")

    with h5py.File(filename, 'r') as f:
        grid_proj_coords = f['grid_proj_coords'][:].astype(np.float32)     # (H*W,2)
        grid_latlon_coords = f['grid_latlon_coords'][:].astype(np.float32)
        grid_shape = tuple(f['grid_shape'][:].astype(np.int32).tolist())
        x_grid = f['x_grid'][:].astype(np.float32)
        y_grid = f['y_grid'][:].astype(np.float32)

        if 'grid_valid_mask' in f:
            grid_valid_mask = f['grid_valid_mask'][:].astype(np.uint8)     # (H,W)
        else:
            H, W = grid_shape
            grid_valid_mask = np.ones((H, W), dtype=np.uint8)

        if 'hprime_prior_grid' in f:
            hprime_prior = f['hprime_prior_grid'][:].astype(np.float32)  # (S,H,W)
            beta_prior = f['beta_prior_grid'][:].astype(np.float32)
            hprime_true = f['hprime_true_grid'][:].astype(np.float32)
            beta_true = f['beta_true_grid'][:].astype(np.float32)
        else:
            # 兼容旧格式
            hprime_true = f['hprime_grid'][:].astype(np.float32)
            beta_true = f['beta_grid'][:].astype(np.float32)
            hprime_prior = None
            beta_prior = None
            print("警告：输入 ionosphere_states.h5 缺少 prior，将无法计算 prior-sim 电场。")

        metadata = []
        metadata_group = f['metadata']
        for sample_name in sorted(metadata_group.keys()):
            sg = metadata_group[sample_name]
            meta = {key: sg.attrs[key] for key in sg.attrs}
            metadata.append(meta)

    grid_data = {
        'proj_coords': grid_proj_coords,
        'latlon_coords': grid_latlon_coords,
        'grid_shape': grid_shape,
        'x_grid': x_grid,
        'y_grid': y_grid,
        'valid_mask': grid_valid_mask,
    }

    return grid_data, hprime_prior, beta_prior, hprime_true, beta_true, metadata


def compute_all_vlf_paths():
    """计算所有VLF路径的电场数据"""
    print("=" * 60)
    print("北美VLF数据集 - VLF路径计算（基于 true 电离层生成观测；可选 prior-sim）")
    print("=" * 60)

    grid_data, hp_prior, be_prior, hp_true, be_true, all_metadata = load_ionosphere_states(
        C.IONOSPHERE_STATES_FILE
    )

    num_samples = int(hp_true.shape[0]) if hp_true is not None else int(be_true.shape[0])
    num_paths = len(C.ALL_PATHS)
    H, W = map(int, grid_data['grid_shape'])

    print(f"\n开始计算 {num_samples} 个样本 × {num_paths} 条路径 = {num_samples * num_paths} 次计算")
    print(f"发射机: {list(C.TRANSMITTERS.keys())}")
    print(f"接收机: {list(C.RECEIVERS.keys())}")
    print(f"LWPC工作目录: {C.SUPERLWPC_WORK_DIR}")
    print(f"LWPC可执行文件: {C.SUPERLWPC_EXECUTABLE}")
    print(f"规则栅格: H={H}, W={W}, N={H*W}")
    print(f"COMPUTE_PRIOR_SIM_FIELDS={bool(COMPUTE_PRIOR_SIM_FIELDS)}")

    base_work_dir = os.path.join(C.WORK_DIR, "lwpc_calculations")
    os.makedirs(base_work_dir, exist_ok=True)

    work_dir_true = os.path.join(base_work_dir, "true_obs")
    os.makedirs(work_dir_true, exist_ok=True)

    work_dir_prior = os.path.join(base_work_dir, "prior_sim")
    os.makedirs(work_dir_prior, exist_ok=True)

    # 预先计算路径分段(所有样本共用)
    path_segment_latlon = np.zeros((num_paths, C.PATH_SEGMENTS, 2), dtype=np.float32)
    path_segment_distances = np.zeros((num_paths, C.PATH_SEGMENTS), dtype=np.float32)
    path_segment_proj = np.zeros((num_paths, C.PATH_SEGMENTS, 2), dtype=np.float32)

    for path_idx, path_info in enumerate(C.ALL_PATHS):
        seg_pts, seg_dist = create_path_segments(
            path_info['tx_lat'], path_info['tx_lon'],
            path_info['rx_lat'], path_info['rx_lon'],
            C.PATH_SEGMENTS
        )
        path_segment_latlon[path_idx] = seg_pts.astype(np.float32)
        path_segment_distances[path_idx] = seg_dist.astype(np.float32)

        for i in range(C.PATH_SEGMENTS):
            lat, lon = float(seg_pts[i, 0]), float(seg_pts[i, 1])
            x, y = latlon_to_projection(lat, lon)
            path_segment_proj[path_idx, i, 0] = x
            path_segment_proj[path_idx, i, 1] = y

    # 预计算 双线性 采样映射: (P,L,4)
    idx4, w4, valid = build_bilinear_index_weights(
        path_segment_proj.reshape(-1, 2),
        grid_data['x_grid'], grid_data['y_grid']
    )
    path_point_grid_idx = idx4.reshape(num_paths, C.PATH_SEGMENTS, 4).astype(np.int64)
    path_point_grid_w = w4.reshape(num_paths, C.PATH_SEGMENTS, 4).astype(np.float32)
    path_point_valid = valid.reshape(num_paths, C.PATH_SEGMENTS).astype(np.uint8)

    # 输出数组：每样本×每路径×每段（true_obs）
    amplitude_x = np.zeros((num_samples, num_paths, C.PATH_SEGMENTS), dtype=np.float32)
    phase_x = np.zeros((num_samples, num_paths, C.PATH_SEGMENTS), dtype=np.float32)
    amplitude_y = np.zeros((num_samples, num_paths, C.PATH_SEGMENTS), dtype=np.float32)
    phase_y = np.zeros((num_samples, num_paths, C.PATH_SEGMENTS), dtype=np.float32)
    amplitude_z = np.zeros((num_samples, num_paths, C.PATH_SEGMENTS), dtype=np.float32)
    phase_z = np.zeros((num_samples, num_paths, C.PATH_SEGMENTS), dtype=np.float32)

    # 新增：prior-sim 电场（与 true_obs 同形状）
    amplitude_x_prior_sim = np.zeros_like(amplitude_x)
    phase_x_prior_sim = np.zeros_like(phase_x)
    amplitude_y_prior_sim = np.zeros_like(amplitude_y)
    phase_y_prior_sim = np.zeros_like(phase_y)
    amplitude_z_prior_sim = np.zeros_like(amplitude_z)
    phase_z_prior_sim = np.zeros_like(phase_z)

    # 新增：路径计算成功掩码
    path_success_obs = np.zeros((num_samples, num_paths), dtype=np.uint8)
    path_success_prior_sim = np.zeros((num_samples, num_paths), dtype=np.uint8)

    success_count = 0
    total_calculations = num_samples * num_paths

    for sample_idx in tqdm(range(num_samples), desc="计算样本"):
        hp_grid_true = hp_true[sample_idx].astype(np.float32)  # (H,W)
        be_grid_true = be_true[sample_idx].astype(np.float32)

        has_prior = (hp_prior is not None) and (be_prior is not None)
        if has_prior:
            hp_grid_prior = hp_prior[sample_idx].astype(np.float32)
            be_grid_prior = be_prior[sample_idx].astype(np.float32)
        else:
            hp_grid_prior = None
            be_grid_prior = None

        rng = np.random.default_rng(C.SEED + 10_000 + sample_idx)
        sample_success = 0

        # flatten once for fast sampling
        hp_true_flat = hp_grid_true.reshape(-1)
        be_true_flat = be_grid_true.reshape(-1)

        if has_prior:
            hp_prior_flat = hp_grid_prior.reshape(-1)
            be_prior_flat = be_grid_prior.reshape(-1)
        else:
            hp_prior_flat = None
            be_prior_flat = None

        for path_idx, path_info in enumerate(tqdm(C.ALL_PATHS, desc=f"样本{sample_idx}路径", leave=False)):
            segment_distances = path_segment_distances[path_idx]

            # 双线性从网格采样到路径（true）
            idxp = path_point_grid_idx[path_idx]  # (L,4)
            wp = path_point_grid_w[path_idx]      # (L,4)
            hprime_path_true = np.sum(hp_true_flat[idxp] * wp, axis=-1).astype(np.float32)
            beta_path_true = np.sum(be_true_flat[idxp] * wp, axis=-1).astype(np.float32)

            # prior path（若存在）
            if has_prior:
                hprime_path_prior = np.sum(hp_prior_flat[idxp] * wp, axis=-1).astype(np.float32)
                beta_path_prior = np.sum(be_prior_flat[idxp] * wp, axis=-1).astype(np.float32)
            else:
                hprime_path_prior = None
                beta_path_prior = None

            tx_name = path_info['tx_name']
            tx_info = C.TRANSMITTERS[tx_name]
            rx_name = path_info['rx_name']
            rx_info = C.RECEIVERS[rx_name]

            # --- true -> obs（加噪） ---
            success, results = run_lwpc_for_path(
                sample_idx, path_idx,
                hprime_path_true, beta_path_true, segment_distances,
                tx_info, rx_info,
                work_dir_true
            )

            if success and results:
                def add_noise(amp, pha):
                    if not ADD_MEASUREMENT_NOISE:
                        return amp, pha
                    amp_n = amp + rng.normal(0.0, SIGMA_AMP_DB, size=amp.shape).astype(np.float32)
                    pha_n = pha + rng.normal(0.0, SIGMA_PHASE_DEG, size=pha.shape).astype(np.float32)
                    pha_n = ((pha_n + 180.0) % 360.0) - 180.0
                    return amp_n, pha_n

                if 'x' in results:
                    amp, pha = results['x']
                    amp, pha = add_noise(amp, pha)
                    amplitude_x[sample_idx, path_idx] = amp
                    phase_x[sample_idx, path_idx] = pha

                if 'y' in results:
                    amp, pha = results['y']
                    amp, pha = add_noise(amp, pha)
                    amplitude_y[sample_idx, path_idx] = amp
                    phase_y[sample_idx, path_idx] = pha

                if 'z' in results:
                    amp, pha = results['z']
                    amp, pha = add_noise(amp, pha)
                    amplitude_z[sample_idx, path_idx] = amp
                    phase_z[sample_idx, path_idx] = pha

                path_success_obs[sample_idx, path_idx] = 1
                sample_success += 1
                success_count += 1
            else:
                # keep zeros, success mask stays 0
                pass

            # --- prior-sim（不加噪；用于构造 obs-residual） ---
            if COMPUTE_PRIOR_SIM_FIELDS and has_prior and (hprime_path_prior is not None):
                suc_p, res_p = run_lwpc_for_path(
                    sample_idx, path_idx,
                    hprime_path_prior, beta_path_prior, segment_distances,
                    tx_info, rx_info,
                    work_dir_prior
                )
                if suc_p and res_p:
                    if 'x' in res_p:
                        amp, pha = res_p['x']
                        amplitude_x_prior_sim[sample_idx, path_idx] = amp
                        phase_x_prior_sim[sample_idx, path_idx] = pha
                    if 'y' in res_p:
                        amp, pha = res_p['y']
                        amplitude_y_prior_sim[sample_idx, path_idx] = amp
                        phase_y_prior_sim[sample_idx, path_idx] = pha
                    if 'z' in res_p:
                        amp, pha = res_p['z']
                        amplitude_z_prior_sim[sample_idx, path_idx] = amp
                        phase_z_prior_sim[sample_idx, path_idx] = pha
                    path_success_prior_sim[sample_idx, path_idx] = 1
                else:
                    # keep zeros
                    pass

        print(f"样本 {sample_idx}: obs成功 {sample_success}/{num_paths} 条路径")

    success_rate = success_count / total_calculations * 100.0
    print("\n计算完成!")
    print(f"总计算次数: {total_calculations}")
    print(f"obs成功次数: {success_count}")
    print(f"obs成功率: {success_rate:.2f}%")

    save_final_dataset(
        grid_data,
        hp_prior, be_prior,
        hp_true, be_true,
        all_metadata,
        path_segment_latlon, path_segment_distances,
        path_segment_proj,
        path_point_grid_idx, path_point_grid_w, path_point_valid,
        amplitude_x, phase_x,
        amplitude_y, phase_y,
        amplitude_z, phase_z,
        amplitude_x_prior_sim, phase_x_prior_sim,
        amplitude_y_prior_sim, phase_y_prior_sim,
        amplitude_z_prior_sim, phase_z_prior_sim,
        path_success_obs, path_success_prior_sim,
    )
    return True


def save_final_dataset(
    grid_data,
    hp_prior, be_prior,
    hp_true, be_true,
    all_metadata,
    path_segment_latlon, path_segment_distances,
    path_segment_proj,
    path_point_grid_idx, path_point_grid_w, path_point_valid,
    amp_x, pha_x, amp_y, pha_y, amp_z, pha_z,
    amp_x_prior_sim, pha_x_prior_sim,
    amp_y_prior_sim, pha_y_prior_sim,
    amp_z_prior_sim, pha_z_prior_sim,
    path_success_obs, path_success_prior_sim,
):
    """保存最终数据集到HDF5文件（规则栅格版本）"""
    print(f"\n保存最终数据集到 {C.FINAL_DATASET_FILE}...")

    H, W = map(int, grid_data['grid_shape'])

    with h5py.File(C.FINAL_DATASET_FILE, 'w') as f:
        f.create_dataset('grid_proj_coords', data=grid_data['proj_coords'], compression='gzip')
        f.create_dataset('grid_latlon_coords', data=grid_data['latlon_coords'], compression='gzip')
        f.create_dataset('grid_shape', data=np.array([H, W], dtype=np.int32))
        f.create_dataset('x_grid', data=grid_data['x_grid'])
        f.create_dataset('y_grid', data=grid_data['y_grid'])
        f.create_dataset('grid_valid_mask', data=grid_data['valid_mask'].astype(np.uint8), compression='gzip')

        if hp_prior is not None and be_prior is not None:
            f.create_dataset('hprime_prior_grid', data=hp_prior, compression='gzip')  # (S,H,W)
            f.create_dataset('beta_prior_grid', data=be_prior, compression='gzip')
        f.create_dataset('hprime_true_grid', data=hp_true, compression='gzip')
        f.create_dataset('beta_true_grid', data=be_true, compression='gzip')

        # 路径分段几何（所有样本共用）
        f.create_dataset('path_segment_latlon', data=path_segment_latlon, compression='gzip')
        f.create_dataset('path_segment_proj_coords_km', data=path_segment_proj, compression='gzip')
        f.create_dataset('path_segment_distances_km', data=path_segment_distances, compression='gzip')

        # 双线性映射（所有样本共用）
        f.create_dataset('path_point_grid_idx', data=path_point_grid_idx.astype(np.int64), compression='gzip')  # (P,L,4)
        f.create_dataset('path_point_grid_w', data=path_point_grid_w.astype(np.float32), compression='gzip')    # (P,L,4)
        f.create_dataset('path_point_valid', data=path_point_valid.astype(np.uint8), compression='gzip')        # (P,L)

        # 电场：true->obs（含噪）
        f.create_dataset('amplitude_x', data=amp_x, compression='gzip')
        f.create_dataset('phase_x', data=pha_x, compression='gzip')
        f.create_dataset('amplitude_y', data=amp_y, compression='gzip')
        f.create_dataset('phase_y', data=pha_y, compression='gzip')
        f.create_dataset('amplitude_z', data=amp_z, compression='gzip')
        f.create_dataset('phase_z', data=pha_z, compression='gzip')

        # 新增：prior-sim（不加噪，用于构造 residual 特征）
        f.create_dataset('amplitude_x_prior_sim', data=amp_x_prior_sim, compression='gzip')
        f.create_dataset('phase_x_prior_sim', data=pha_x_prior_sim, compression='gzip')
        f.create_dataset('amplitude_y_prior_sim', data=amp_y_prior_sim, compression='gzip')
        f.create_dataset('phase_y_prior_sim', data=pha_y_prior_sim, compression='gzip')
        f.create_dataset('amplitude_z_prior_sim', data=amp_z_prior_sim, compression='gzip')
        f.create_dataset('phase_z_prior_sim', data=pha_z_prior_sim, compression='gzip')

        # 新增：路径成功掩码
        f.create_dataset('path_success_obs', data=path_success_obs.astype(np.uint8), compression='gzip')              # (S,P)
        f.create_dataset('path_success_prior_sim', data=path_success_prior_sim.astype(np.uint8), compression='gzip')  # (S,P)

        # 路径信息
        path_group = f.create_group('paths')
        for i, path_info in enumerate(C.ALL_PATHS):
            sg = path_group.create_group(f'path_{i:02d}')
            for key, value in path_info.items():
                sg.attrs[key] = value

        # 元数据
        metadata_group = f.create_group('metadata')
        for i, meta in enumerate(all_metadata):
            sg = metadata_group.create_group(f'sample_{i:04d}')
            for key, value in meta.items():
                sg.attrs[key] = value

        f.attrs['num_samples'] = int(len(hp_true))
        f.attrs['num_grid_points'] = int(H * W)
        f.attrs['grid_shape_h'] = int(H)
        f.attrs['grid_shape_w'] = int(W)
        f.attrs['num_paths'] = int(len(C.ALL_PATHS))
        f.attrs['path_segments'] = int(C.PATH_SEGMENTS)
        f.attrs['grid_spacing_km'] = float(C.GRID_SPACING_KM)
        f.attrs['projection'] = 'North_America_Equidistant_Conic'
        f.attrs['measurement_noise_added'] = bool(ADD_MEASUREMENT_NOISE)
        f.attrs['sigma_amp_db'] = float(SIGMA_AMP_DB)
        f.attrs['sigma_phase_deg'] = float(SIGMA_PHASE_DEG)
        f.attrs['compute_prior_sim_fields'] = bool(COMPUTE_PRIOR_SIM_FIELDS)
        f.attrs['description'] = (
            "North America VLF Dataset (regular grid): prior(Ferguson) + true(prior+correlated perturbation) "
            "+ LWPC fields along paths. Includes (1) obs fields (true->LWPC+noise), (2) prior-sim fields (prior->LWPC), "
            "(3) bilinear path->grid mapping, (4) path success masks."
        )

    print(f"最终数据集已保存到: {C.FINAL_DATASET_FILE}")
    print("\n数据集统计信息:")
    print(f"  样本数量: {len(hp_true)}")
    print(f"  网格尺寸: H={H}, W={W}, N={H*W}")
    print(f"  路径数量: {len(C.ALL_PATHS)}")
    print(f"  每路径分段数: {C.PATH_SEGMENTS}")
    print(f"  数据总量: {os.path.getsize(C.FINAL_DATASET_FILE) / (1024**3):.2f} GB")


def main():
    """主函数"""
    if not os.path.exists(C.SUPERLWPC_WORK_DIR):
        print(f"错误: LWPC工作目录不存在: {C.SUPERLWPC_WORK_DIR}")
        return False

    if not os.path.exists(C.SUPERLWPC_EXECUTABLE):
        print(f"错误: LWPC可执行文件不存在: {C.SUPERLWPC_EXECUTABLE}")
        return False

    if not os.path.exists(C.IONOSPHERE_STATES_FILE):
        print(f"错误: 电离层状态文件不存在: {C.IONOSPHERE_STATES_FILE}")
        print("请先运行 generate_ionosphere_states.py")
        return False

    print(f"LWPC工作目录: {C.SUPERLWPC_WORK_DIR}")
    print(f"LWPC可执行文件: {C.SUPERLWPC_EXECUTABLE}")

    try:
        test_file = os.path.join(C.SUPERLWPC_WORK_DIR, "test_permission.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write("test")
        os.remove(test_file)
        print("LWPC工作目录可写: 是")
    except Exception as e:
        print(f"LWPC工作目录可写: 否 - {e}")
        return False

    success = compute_all_vlf_paths()

    if success:
        print("\n项目完成!")
        print(f"最终数据集: {C.FINAL_DATASET_FILE}")
        print("数据集包含:")
        print(f"  1. {C.NUM_SAMPLES} 个网格化电离层状态（prior & true，规则栅格）")
        print("  2. 18条VLF传播路径的电场数据（沿路径100点，true->obs含噪）")
        print("  3. prior->LWPC 的 prior-sim 电场（用于残差特征）")
        print("  4. 预计算路径点到网格的双线性映射 + 路径计算成功掩码（用于训练时屏蔽失败路径）")

    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)