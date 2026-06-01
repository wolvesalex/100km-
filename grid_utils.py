# grid_utils.py
# -*- coding: utf-8 -*-
"""
等距圆锥投影网格化工具（规则栅格版本）
用于北美区域的网格化处理

关键改动(为U-Net适配):
- 输出规则的 (H,W) 栅格：proj_coords/latlon_coords 不再筛掉点导致不规则
- 提供 grid_valid_mask：标记哪些栅格点落在 NORTH_AMERICA_BOUNDS 的 lat/lon 范围内
- 提供路径点到栅格的双线性索引/权重预计算（用于“观测特征图”splat / 以及从网格采样到路径）

新增(为多尺度/级联稳定性):
- x_grid / y_grid 自动补齐到给定 multiple 的长度（默认 4），避免 H=31 这类奇数导致级联/下采样不整齐
"""

import numpy as np
from math import radians, degrees, sin, cos, tan, exp
import math
import config as C

try:
    from pyproj import Proj
    PYPROJ_AVAILABLE = True
except ImportError:
    PYPROJ_AVAILABLE = False
    print("警告: pyproj 未安装，将使用简化投影函数")


def latlon_to_projection(lat_deg, lon_deg):
    """
    将经纬度转换为等距圆锥投影坐标（km）
    使用准确的ESRI:102010投影
    """
    if PYPROJ_AVAILABLE:
        proj_string = (
            f"+proj=eqdc "
            f"+lat_1={C.PROJECTION_PARAMS['standard_parallel_1']} "
            f"+lat_2={C.PROJECTION_PARAMS['standard_parallel_2']} "
            f"+lon_0={C.PROJECTION_PARAMS['central_meridian']} "
            f"+lat_0={C.PROJECTION_PARAMS['latitude_of_origin']} "
            f"+x_0=0.0 "
            f"+y_0=0.0 "
            f"+datum=WGS84 "
            f"+units=m "
            f"+no_defs"
        )
        proj = Proj(proj_string)
        x_m, y_m = proj(lon_deg, lat_deg)  # (lon,lat)
        return x_m / 1000.0, y_m / 1000.0
    else:
        # 简化投影（保留你原来的逻辑）
        lat1 = radians(C.PROJECTION_PARAMS['standard_parallel_1'])
        lat2 = radians(C.PROJECTION_PARAMS['standard_parallel_2'])
        lat0 = radians(C.PROJECTION_PARAMS['latitude_of_origin'])
        lon0 = radians(C.PROJECTION_PARAMS['central_meridian'])

        phi = radians(lat_deg)
        lam = radians(lon_deg)

        n = (cos(lat1) - cos(lat2)) / (lat2 - lat1) if lat1 != lat2 else sin(lat1)

        if n == 0:
            F = cos(lat1)
        else:
            F = (cos(lat1) * (1 / n + tan(lat1))) / (1 / n + tan(lat0))

        if n >= 0:
            rho = C.EARTH_RADIUS_KM * F * exp(-n * (phi - lat0))
        else:
            rho = C.EARTH_RADIUS_KM * F / exp(n * (phi - lat0))

        rho0 = C.EARTH_RADIUS_KM * F
        theta = n * (lam - lon0)

        x = rho * sin(theta)
        y = rho0 - rho * cos(theta)
        return x, y


def projection_to_latlon(x, y):
    """将等距圆锥投影坐标（km）转换回经纬度"""
    if PYPROJ_AVAILABLE:
        proj_string = (
            f"+proj=eqdc "
            f"+lat_1={C.PROJECTION_PARAMS['standard_parallel_1']} "
            f"+lat_2={C.PROJECTION_PARAMS['standard_parallel_2']} "
            f"+lon_0={C.PROJECTION_PARAMS['central_meridian']} "
            f"+lat_0={C.PROJECTION_PARAMS['latitude_of_origin']} "
            f"+x_0=0.0 "
            f"+y_0=0.0 "
            f"+datum=WGS84 "
            f"+units=m "
            f"+no_defs"
        )
        proj = Proj(proj_string)
        lon_deg, lat_deg = proj(x * 1000.0, y * 1000.0, inverse=True)
        return float(lat_deg), float(lon_deg)
    else:
        # 简化逆投影（保留你原来的逻辑）
        from math import log
        lat1 = radians(C.PROJECTION_PARAMS['standard_parallel_1'])
        lat2 = radians(C.PROJECTION_PARAMS['standard_parallel_2'])
        lat0 = radians(C.PROJECTION_PARAMS['latitude_of_origin'])
        lon0 = radians(C.PROJECTION_PARAMS['central_meridian'])

        n = (cos(lat1) - cos(lat2)) / (lat2 - lat1) if lat1 != lat2 else sin(lat1)

        if n == 0:
            F = cos(lat1)
        else:
            F = (cos(lat1) * (1 / n + tan(lat1))) / (1 / n + tan(lat0))

        rho0 = C.EARTH_RADIUS_KM * F

        rho = np.sqrt(x**2 + (rho0 - y)**2)
        theta = np.arctan2(x, rho0 - y)

        if n > 0:
            phi = lat0 - (1 / n) * np.log(rho / (C.EARTH_RADIUS_KM * F))
        elif n < 0:
            phi = lat0 + (1 / abs(n)) * np.log(rho / (C.EARTH_RADIUS_KM * F))
        else:
            phi = lat0

        lam = lon0 + theta / n if n != 0 else lon0

        lat_deg = degrees(phi)
        lon_deg = degrees(lam)
        lon_deg = (lon_deg + 180) % 360 - 180
        return float(lat_deg), float(lon_deg)


def _pad_1d_to_multiple(arr: np.ndarray, multiple: int, step: float) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    n = int(arr.size)
    if multiple <= 1:
        return arr
    target = int(math.ceil(n / multiple) * multiple)
    if target == n:
        return arr
    k = target - n
    last = float(arr[-1])
    extra = last + step * np.arange(1, k + 1, dtype=np.float32)
    out = np.concatenate([arr, extra], axis=0).astype(np.float32)
    return out


def create_north_america_grid():
    """
    创建北美区域的等距圆锥投影“规则栅格”
    返回:
      - proj_coords: (H*W,2) km
      - latlon_coords: (H*W,2)
      - grid_shape: (H,W) with H=len(y_grid), W=len(x_grid)
      - x_grid, y_grid
      - valid_mask: (H,W) uint8, 1 表示对应 lat/lon 在 NORTH_AMERICA_BOUNDS 内
    """
    print("创建北美区域规则栅格...")

    min_lon, max_lon = C.NORTH_AMERICA_BOUNDS['min_lon'], C.NORTH_AMERICA_BOUNDS['max_lon']
    min_lat, max_lat = C.NORTH_AMERICA_BOUNDS['min_lat'], C.NORTH_AMERICA_BOUNDS['max_lat']

    corners = [
        (min_lat, min_lon),
        (min_lat, max_lon),
        (max_lat, min_lon),
        (max_lat, max_lon),
    ]

    proj_corners = [latlon_to_projection(lat, lon) for lat, lon in corners]
    proj_x = [p[0] for p in proj_corners]
    proj_y = [p[1] for p in proj_corners]

    min_x, max_x = float(min(proj_x)), float(max(proj_x))
    min_y, max_y = float(min(proj_y)), float(max(proj_y))

    print(f"投影坐标范围(由角点估计): x[{min_x:.1f}, {max_x:.1f}], y[{min_y:.1f}, {max_y:.1f}] km")
    dx = float(C.GRID_SPACING_KM)

    x_grid = np.arange(min_x, max_x + dx, dx, dtype=np.float32)
    y_grid = np.arange(min_y, max_y + dx, dx, dtype=np.float32)

    # --- 新增：补齐到 multiple（默认 4；你当前会把 H=31 补到 32） ---
    pad_multiple = int(getattr(C, "GRID_PAD_TO_MULTIPLE", 4))
    x_grid = _pad_1d_to_multiple(x_grid, pad_multiple, step=dx)
    y_grid = _pad_1d_to_multiple(y_grid, pad_multiple, step=dx)

    H, W = len(y_grid), len(x_grid)
    print(f"规则栅格尺寸: H={H}, W={W}, N={H*W} (pad_multiple={pad_multiple})")

    X, Y = np.meshgrid(x_grid, y_grid)  # (H,W)
    proj_coords = np.stack([X.reshape(-1), Y.reshape(-1)], axis=-1).astype(np.float32)

    # 计算每个栅格点的 lat/lon（用于先验模型 + valid_mask）
    latlon_coords = np.zeros((H * W, 2), dtype=np.float32)
    valid_mask = np.zeros((H, W), dtype=np.uint8)

    for iy in range(H):
        for ix in range(W):
            lat, lon = projection_to_latlon(float(X[iy, ix]), float(Y[iy, ix]))
            latlon_coords[iy * W + ix, 0] = lat
            latlon_coords[iy * W + ix, 1] = lon
            if (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat):
                valid_mask[iy, ix] = 1

    print(f"valid_mask: {int(valid_mask.sum())}/{H*W} 栅格点在lat/lon边界内")

    return {
        'proj_coords': proj_coords,
        'latlon_coords': latlon_coords,
        'grid_shape': (H, W),
        'x_grid': x_grid,
        'y_grid': y_grid,
        'valid_mask': valid_mask,
    }


def build_bilinear_index_weights(points_xy_km: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray):
    """
    对任意投影坐标点，计算其落在规则栅格上的双线性插值索引与权重。

    points_xy_km: (N,2)
    x_grid: (W,), y_grid: (H,)
    返回:
      idx4: (N,4) int64  -> flatten index in [0, H*W)
      w4:   (N,4) float32
      valid:(N,)  uint8  -> 1 表示在网格范围内(可插值)
    """
    pts = np.asarray(points_xy_km, dtype=np.float32)
    xg = np.asarray(x_grid, dtype=np.float32)
    yg = np.asarray(y_grid, dtype=np.float32)

    H = yg.size
    W = xg.size
    dx = float(xg[1] - xg[0]) if W > 1 else float(C.GRID_SPACING_KM)
    dy = float(yg[1] - yg[0]) if H > 1 else float(C.GRID_SPACING_KM)

    x0 = float(xg[0])
    y0 = float(yg[0])

    fx = (pts[:, 0] - x0) / (dx + 1e-12)
    fy = (pts[:, 1] - y0) / (dy + 1e-12)

    ix0 = np.floor(fx).astype(np.int64)
    iy0 = np.floor(fy).astype(np.int64)

    # valid要求: ix0 in [0, W-2], iy0 in [0, H-2]
    valid = (ix0 >= 0) & (ix0 < W - 1) & (iy0 >= 0) & (iy0 < H - 1)
    valid_u8 = valid.astype(np.uint8)

    ix0c = np.clip(ix0, 0, W - 2)
    iy0c = np.clip(iy0, 0, H - 2)
    ix1c = ix0c + 1
    iy1c = iy0c + 1

    tx = (fx - ix0c).astype(np.float32)
    ty = (fy - iy0c).astype(np.float32)
    tx = np.clip(tx, 0.0, 1.0)
    ty = np.clip(ty, 0.0, 1.0)

    w00 = (1 - tx) * (1 - ty)
    w10 = tx * (1 - ty)
    w01 = (1 - tx) * ty
    w11 = tx * ty

    # flatten index = iy*W + ix
    idx00 = iy0c * W + ix0c
    idx10 = iy0c * W + ix1c
    idx01 = iy1c * W + ix0c
    idx11 = iy1c * W + ix1c

    idx4 = np.stack([idx00, idx10, idx01, idx11], axis=-1).astype(np.int64)
    w4 = np.stack([w00, w10, w01, w11], axis=-1).astype(np.float32)

    # 对于 invalid 点，把权重置0，避免污染
    w4[~valid] = 0.0
    return idx4, w4, valid_u8


def sample_grid_bilinear(grid_2d: np.ndarray, idx4: np.ndarray, w4: np.ndarray):
    """
    使用预计算的 idx4/w4 从 grid_2d (H,W) 双线性采样到点值。
    返回: (N,) float32
    """
    g = np.asarray(grid_2d, dtype=np.float32)
    flat = g.reshape(-1)
    idx = np.asarray(idx4, dtype=np.int64)
    w = np.asarray(w4, dtype=np.float32)
    vals = flat[idx]  # (N,4)
    out = np.sum(vals * w, axis=-1)
    return out.astype(np.float32)


def create_path_segments(tx_lat, tx_lon, rx_lat, rx_lon, num_segments):
    """
    创建发射机到接收机之间的路径分段点
    【保持原有代码不变】
    """
    from utils_geo import destination_point

    total_distance = haversine_distance(tx_lat, tx_lon, rx_lat, rx_lon)
    bearing = calculate_bearing(tx_lat, tx_lon, rx_lat, rx_lon)

    segment_distances = np.linspace(0.0, total_distance, num_segments, dtype=float)
    segment_points = []

    for dist in segment_distances:
        lat, lon = destination_point(tx_lat, tx_lon, bearing, float(dist))
        segment_points.append([lat, lon])

    segment_points = np.asarray(segment_points, dtype=float)

    segment_points[-1, 0] = rx_lat
    segment_points[-1, 1] = rx_lon
    segment_distances[-1] = total_distance

    return segment_points, segment_distances


def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两点间的大圆距离（km）"""
    from math import radians, sin, cos, sqrt, atan2

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return C.EARTH_RADIUS_KM * c


def calculate_bearing(lat1, lon1, lat2, lon2):
    """计算从点1到点2的方位角（度）"""
    from math import radians, degrees, sin, cos, atan2

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)

    initial_bearing = atan2(x, y)
    initial_bearing = degrees(initial_bearing)
    bearing = (initial_bearing + 360) % 360
    return bearing