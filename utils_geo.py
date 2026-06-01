# utils_geo.py
# -*- coding: utf-8 -*-
"""
地理计算工具
包含太阳天顶角计算、大圆距离等
"""

import numpy as np
from math import pi, sin, cos, tan, asin, acos, radians, degrees, sqrt, atan2
from datetime import datetime, timedelta

def deg2rad(deg):
    """角度转弧度"""
    return deg * pi / 180.0

def rad2deg(rad):
    """弧度转角度"""
    return rad * 180.0 / pi

def calculate_solar_zenith_angle(lat_deg, lon_deg, dt):
    """
    准确计算太阳天顶角χ（单位：弧度）
    使用标准天文公式
    """
    # 转换为弧度
    lat_rad = deg2rad(lat_deg)
    lon_rad = deg2rad(lon_deg)
    
    # 计算儒略日
    year, month, day = dt.year, dt.month, dt.day
    hour = dt.hour + dt.minute/60.0 + dt.second/3600.0
    
    # 儒略日计算
    if month <= 2:
        year -= 1
        month += 12
    
    A = int(year / 100)
    B = 2 - A + int(A / 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5
    jd += hour / 24.0
    
    # 计算儒略世纪数
    T = (jd - 2451545.0) / 36525.0
    
    # 太阳几何平均经度（度）
    L0 = 280.46646 + 36000.76983 * T + 0.0003032 * T**2
    L0 = L0 % 360
    
    # 太阳几何平均近点角（度）
    M = 357.52911 + 35999.05029 * T - 0.0001537 * T**2
    M_rad = deg2rad(M)
    
    # 太阳方程中心
    C = (1.914602 - 0.004817 * T - 0.000014 * T**2) * sin(M_rad) + (0.019993 - 0.000101 * T) * sin(2 * M_rad) + 0.000289 * sin(3 * M_rad)
    
    # 太阳真经度
    sun_lon = L0 + C
    
    # 太阳真近点角
    sun_anomaly = M + C
    
    # 地球轨道偏心率
    e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T**2
    
    # 太阳视赤经
    sun_lon_rad = deg2rad(sun_lon)
    epsilon = 23.43929111 - 0.013004167 * T  # 黄赤交角
    epsilon_rad = deg2rad(epsilon)
    alpha = atan2(cos(epsilon_rad) * sin(sun_lon_rad), cos(sun_lon_rad))
    alpha_deg = rad2deg(alpha)
    
    if alpha_deg < 0:
        alpha_deg += 360
    
    # 太阳视赤纬
    delta = asin(sin(epsilon_rad) * sin(sun_lon_rad))
    
    # 计算时角
    # 格林尼治恒星时
    theta0 = 280.46061837 + 360.98564736629 * (jd - 2451545.0) + 0.000387933 * T**2 - T**3 / 38710000.0
    theta0 = theta0 % 360
    
    # 本地恒星时
    theta = theta0 + lon_deg
    theta_rad = deg2rad(theta)
    
    # 时角
    H = theta_rad - alpha
    
    if H < 0:
        H += 2 * pi
    if H > 2 * pi:
        H -= 2 * pi
    
    # 太阳高度角
    sin_altitude = sin(lat_rad) * sin(delta) + cos(lat_rad) * cos(delta) * cos(H)
    altitude = asin(max(min(sin_altitude, 1.0), -1.0))  # 防止数值误差
    
    # 太阳天顶角 = 90° - 太阳高度角
    zenith_angle = pi/2 - altitude
    
    return zenith_angle

def destination_point(lat_deg, lon_deg, bearing_deg, distance_km):
    """
    准确计算目标点坐标（使用WGS84椭球模型）
    """
    import config as C
    
    # WGS84椭球参数
    a = 6378.137  # 赤道半径 (km)
    f = 1/298.257223563  # 扁率
    b = a * (1 - f)  # 极半径
    
    lat1 = deg2rad(lat_deg)
    lon1 = deg2rad(lon_deg)
    bearing = deg2rad(bearing_deg)
    
    # 辅助量
    tanU1 = (1 - f) * tan(lat1)
    cosU1 = 1 / sqrt(1 + tanU1**2)
    sinU1 = tanU1 * cosU1
    
    sigma1 = atan2(tanU1, cos(bearing))
    sin_alpha = cosU1 * sin(bearing)
    cos2_alpha = 1 - sin_alpha**2
    
    u2 = cos2_alpha * (a**2 - b**2) / b**2
    A = 1 + u2/16384 * (4096 + u2*(-768 + u2*(320 - 175*u2)))
    B = u2/1024 * (256 + u2*(-128 + u2*(74 - 47*u2)))
    
    sigma = distance_km / (b * A)
    
    # 迭代计算
    for i in range(10):
        cos2sigma_m = cos(2*sigma1 + sigma)
        sin_sigma = sin(sigma)
        cos_sigma = cos(sigma)
        
        delta_sigma = B * sin_sigma * (cos2sigma_m + B/4 * (cos_sigma*(-1 + 2*cos2sigma_m**2) - 
                          B/6 * cos2sigma_m * (-3 + 4*sin_sigma**2) * (-3 + 4*cos2sigma_m**2)))
        
        sigma_new = distance_km / (b * A) + delta_sigma
        
        if abs(sigma_new - sigma) < 1e-12:
            break
        sigma = sigma_new
    
    # 计算目标点坐标
    sin_sigma = sin(sigma)
    cos_sigma = cos(sigma)
    cos2sigma_m = cos(2*sigma1 + sigma)
    
    lat2 = atan2(sinU1*cos_sigma + cosU1*sin_sigma*cos(bearing),
                (1 - f) * sqrt(sin_alpha**2 + (sinU1*sin_sigma - cosU1*cos_sigma*cos(bearing))**2))
    
    lambda_val = atan2(sin_sigma*sin(bearing), 
                      cosU1*cos_sigma - sinU1*sin_sigma*cos(bearing))
    
    C_val = f/16 * cos2_alpha * (4 + f*(4 - 3*cos2_alpha))
    L = lambda_val - (1 - C_val) * f * sin_alpha * (sigma + C_val*sin_sigma*(cos2sigma_m + C_val*cos_sigma*(-1 + 2*cos2sigma_m**2)))
    
    lon2 = lon1 + L
    lon2 = (lon2 + 3*pi) % (2*pi) - pi  # 标准化到[-π, π]
    
    return rad2deg(lat2), rad2deg(lon2)

def haversine_distance(lat1, lon1, lat2, lon2):
    """计算两点间的大圆距离（km）"""
    import config as C
    
    R = C.EARTH_RADIUS_KM
    dlat = deg2rad(lat2 - lat1)
    dlon = deg2rad(lon2 - lon1)
    
    a = sin(dlat/2)**2 + cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    return R * c

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    计算从点1到点2的方位角（度）
    """
    lat1, lon1, lat2, lon2 = map(deg2rad, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    x = sin(dlon) * cos(lat2)
    y = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dlon)
    
    initial_bearing = atan2(x, y)
    initial_bearing = rad2deg(initial_bearing)
    
    # 归一化到0-360度
    bearing = (initial_bearing + 360) % 360
    
    return bearing