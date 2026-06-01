# config.py
# -*- coding: utf-8 -*-
"""
North America VLF Dataset Generation Configuration
Contains all parameters for gridded ionosphere states and VLF path calculations
"""

import os
import numpy as np

# ==================== Project Path Configuration ====================

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# LWPC root directory - Modify to your LWPC installation directory
LWPC_ROOT = "E:/LWPC-main/LWPCv21/build"
# Random seed
SEED = 42

# ==================== Grid Configuration ====================

# Modified North America region bounds (based on path coverage analysis)
# Original: X: [-2072.9, 2127.1] km, Y: [-471.1, 928.9] km, 513 points
# New: Reduced range to focus on path-covered area
# Based on analysis: Path X: [-1236.5, 1420.5], Path Y: [-553.7, 592.3]
# Added 300km margin around path points for better coverage
NORTH_AMERICA_BOUNDS = {
    'min_lon': -125.0,  # West longitude 125 degrees (adjusted from -130)
    'max_lon': -65.0,   # West longitude 65 degrees (adjusted from -60)
    'min_lat': 24.0,    # North latitude 24 degrees (adjusted from 25)
    'max_lat': 52.0     # North latitude 52 degrees (adjusted from 55)
}

# Grid parameters
GRID_SPACING_KM = 100.0  # Grid spacing (maintain for consistency)
EARTH_RADIUS_KM = 6371.0  # Earth radius

# Projection parameters (North America Equidistant Conic)
PROJECTION_PARAMS = {
    'standard_parallel_1': 20.0,  # Standard parallel 1
    'standard_parallel_2': 60.0,  # Standard parallel 2
    'central_meridian': -96.0,    # Central meridian
    'latitude_of_origin': 40.0    # Latitude of origin
}

# ==================== Transmitter Configuration ====================

TRANSMITTERS = {
    'NLK': {
        'name': 'NLK',
        'frequency_hz': 24_800.0,
        'power_kw': 130,
        'lat_deg': 48.20,
        'lon_deg': -121.917,
        'height_km': 0.0
    },
    'NML': {
        'name': 'NML',
        'frequency_hz': 25_200.0,
        'power_kw': 100,
        'lat_deg': 46.35,
        'lon_deg': -98.33,
        'height_km': 0.0
    },
    'NAA': {
        'name': 'NAA',
        'frequency_hz': 24_000.0,
        'power_kw': 1000,
        'lat_deg': 44.633,
        'lon_deg': -67.283,
        'height_km': 0.0
    }
}

# ==================== Receiver Configuration ====================

RECEIVERS = {
    'LP': {'lat_deg': 30.09, 'lon_deg': -97.17, 'height_km': 0.0},
    'BD': {'lat_deg': 37.32, 'lon_deg': -96.75, 'height_km': 0.0},
    'OX': {'lat_deg': 34.43, 'lon_deg': -89.39, 'height_km': 0.0},
    'BX': {'lat_deg': 31.88, 'lon_deg': -82.36, 'height_km': 0.0},
    'BW': {'lat_deg': 33.43, 'lon_deg': -82.58, 'height_km': 0.0},
    'DA': {'lat_deg': 39.28, 'lon_deg': -75.58, 'height_km': 0.0}
}

# All paths (3 transmitters × 6 receivers = 18 paths)
ALL_PATHS = []
for tx_name, tx_info in TRANSMITTERS.items():
    for rx_name, rx_info in RECEIVERS.items():
        ALL_PATHS.append({
            'tx_name': tx_name,
            'rx_name': rx_name,
            'tx_lat': tx_info['lat_deg'],
            'tx_lon': tx_info['lon_deg'],
            'rx_lat': rx_info['lat_deg'],
            'rx_lon': rx_info['lon_deg'],
            'frequency_hz': tx_info['frequency_hz'],
            'power_kw': tx_info['power_kw']
        })

# ==================== Dataset Parameters ====================

# Ionosphere state sample count
NUM_SAMPLES = 5000

# Time parameters (daytime 9:00-16:00)
DAYTIME_HOURS = list(range(9, 17))  # 9:00 to 16:00
DAYTIME_MONTHS = list(range(1, 13))  # All months
DAYTIME_DAYS = list(range(1, 29))  # 1-28 days, avoid month day differences

# Ionosphere parameter ranges (daytime)
HPRIME_RANGE_DAY = (55.0, 80.0)  # km
BETA_RANGE_DAY = (0.30, 0.80)    # km^-1

# Path segments count
PATH_SEGMENTS = 100

# ==================== Ionosphere Truth/Perturbation Model Parameters ====================
# 目标：让单样本空间方差更合理、让 β 与 h′ 有一定物理相关性、让任务更“可反演”。

# Multi-scale correlated perturbation length scales (km)
# 建议：一个大尺度（控制全局趋势）+ 一个中尺度（提高样本内方差，从而提升 per-sample R² 稳定性）
PERTURB_LENGTH_SCALES_KM = [1200.0, 500.0]
PERTURB_SCALE_WEIGHTS = [0.7, 0.3]  # should sum to 1 (will be normalized in code)

# Amplitude ranges of perturbations (applied after unit-std random field)
HPRIME_PERTURB_AMPLITUDE_KM = (3.0, 8.0)
BETA_PERTURB_AMPLITUDE_KM_INV = (0.05, 0.14)

# Correlation between h' perturbation field and beta perturbation field (sample-wise random)
# 多数情况下：h′降低（反射高度更低）往往伴随 β 增大（更陡峭），因此设负相关更合理
HB_PERTURB_CORR_RANGE = (-0.85, -0.35)

# Optional localized "blob" anomalies to increase observable local structure
BLOB_COUNT_RANGE = (0, 3)                 # number of blobs per sample
BLOB_SIGMA_KM_RANGE = (250.0, 700.0)      # spatial scale of blobs
BLOB_HPRIME_AMP_KM_RANGE = (-3.0, 3.0)    # blob amplitude added to h' (km)
BLOB_BETA_AMP_KM_INV_RANGE = (-0.03, 0.03)  # blob amplitude added to beta (km^-1)

# ==================== Ionosphere Model Parameters ====================

# Geomagnetic field parameters
BFIELD_T = 50e-6
BFIELD_THETA_RAD = np.pi / 2.0  # Horizontal
BFIELD_PHI_RAD = 0.0

# Ground parameters
GROUND_SIGMA = 10.0  # S/m
GROUND_EPS_R = 1e-4  # Relative permittivity

# ==================== LWPC Configuration ====================

SUPERLWPC_EXECUTABLE = os.path.join(LWPC_ROOT, "lwpc.bin.exe")
SUPERLWPC_WORK_DIR = LWPC_ROOT  # LWPC working directory

# ==================== Output Configuration ====================

# Data directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
WORK_DIR = os.path.join(PROJECT_ROOT, "work")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "templates")
PLOT_DIR = os.path.join(PROJECT_ROOT, "plots")

# Output files
GRID_DATA_FILE = os.path.join(DATA_DIR, "grid_data.npz")
IONOSPHERE_STATES_FILE = os.path.join(DATA_DIR, "ionosphere_states.h5")
FINAL_DATASET_FILE = os.path.join(DATA_DIR, "north_america_vlf_dataset.h5")

# ==================== Create Directories ====================

for d in [DATA_DIR, WORK_DIR, MODEL_DIR, TEMPLATE_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

# ==================== Physical Constants ====================

LIGHT_SPEED = 3e8  # m/s
VACUUM_PERMITTIVITY = 8.854e-12  # F/m
VACUUM_PERMEABILITY = 4e-7 * np.pi  # H/m