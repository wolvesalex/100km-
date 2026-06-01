# superlwpc_utils.py
# -*- coding: utf-8 -*-
"""
SuperLWPC工具函数
用于生成NDX文件、运行LWPC、解析输出
【保持原有代码不变，只修改投影相关调用】
"""

import os
import subprocess
import numpy as np
import shutil
import config as C


def create_ndx_file_for_path(ndx_file_path, hprime_path, beta_path, distances):
    """
    为路径创建NDX文件
    【保持原有代码不变】
    """
    hprime_path = np.asarray(hprime_path).reshape(-1)
    beta_path = np.asarray(beta_path).reshape(-1)
    distances = np.asarray(distances).reshape(-1)

    if len(distances) == 0:
        raise ValueError("distances 为空，无法生成 NDX 文件")
    if not (len(hprime_path) == len(beta_path) == len(distances)):
        raise ValueError("hprime_path/beta_path/distances 长度不一致")

    with open(ndx_file_path, 'w') as f:
        for i in range(len(distances)):
            distance = float(distances[i])
            beta_val = float(beta_path[i])
            hprime_val = float(hprime_path[i])
            f.write(f"{distance:8.1f} {beta_val:8.4f} {hprime_val:8.3f}\n")


def create_inp_file_for_path(
    inp_file_path,
    ndx_file_path,
    tx_info,
    rx_info,
    sample_idx,
    path_idx,
    distances,
    output_dir
):
    """
    为路径创建LWPC输入文件
    使用 RECEIVERS 控制字符串指定接收机位置
    【保持原有代码不变】
    """
    ndx_basename = os.path.basename(ndx_file_path).replace('.ndx', '')

    # 发射机参数
    tx_id = tx_info['name']
    freq_khz = tx_info['frequency_hz'] / 1000.0  # kHz
    tx_lat = tx_info['lat_deg']
    tx_lon = tx_info['lon_deg']
    power_kw = tx_info['power_kw']
    height_km = tx_info.get('height_km', 0.0)

    # LWPC 经度约定：很多示例使用"西经为正"
    tx_lon_for_lwpc = -tx_lon if tx_lon < 0 else tx_lon

    # 接收机参数
    rx_lat = rx_info['lat_deg']
    rx_lon = rx_info['lon_deg']
    rx_lon_for_lwpc = -rx_lon if rx_lon < 0 else rx_lon

    # 最大距离
    if distances is not None and len(distances) > 0:
        max_range = float(distances[-1])
    else:
        max_range = 8000.0

    # 让输出距离点数尽量贴近你的 PATH_SEGMENTS（默认 100）
    if distances is not None and len(distances) > 1:
        lwf_step = max_range / max(1, (len(distances) - 1))
    else:
        lwf_step = 20.0

    with open(inp_file_path, 'w') as f:
        # 基本配置
        f.write(f"CASE-ID Sample_{sample_idx:04d}_Path_{path_idx:02d}\n")
        f.write(f"TX-NTR sample_{sample_idx:04d}_path_{path_idx:02d}\n")

        # 文件路径配置
        f.write("FILE-MDS ./\n")
        f.write("FILE-LWF ./\n")
        f.write("FILE-GRD ./\n")
        f.write("FILE-PRF ./\n")
        f.write("FILE-NDX ./\n")

        # 发射机参数
        f.write(
            "TX-DATA "
            f"{tx_id}\n"
        )

        # 电离层模型（引用 NDX 基名）
        f.write(f"IONOSPHERE RANGE EXPONENTIAL {ndx_basename}\n")

        # 接收机位置
        f.write(f"RECEIVERS {rx_lat:.6f} {rx_lon_for_lwpc:.6f}\n")

        # 接收机场分量配置
        f.write("RX-DATA HORIZONTAL 0.0\n")

        # 最大距离与输出设置
        f.write(f"RANGE-MAX {max_range:.1f}\n")
        f.write(f"LWF-VS-DIST {max_range:.1f} {lwf_step:.3f}\n")

        # 输出控制
        f.write("PRINT-SWG 0\n")
        f.write("PRINT-MDS 0\n")
        f.write("PRINT-MC 0\n")
        f.write("PRINT-LWF 1\n")
        f.write("PRINT-WF 0\n")

        # 开始计算
        f.write("START\n")
        f.write("QUIT\n")


def prepare_lwpc_input_files(
    sample_idx,
    path_idx,
    hprime_path,
    beta_path,
    distances,
    tx_info,
    rx_info,
    work_dir
):
    """
    准备LWPC输入文件，并复制到LWPC工作目录
    【保持原有代码不变】
    """
    temp_dir = os.path.join(work_dir, f"sample_{sample_idx:04d}_path_{path_idx:02d}")
    os.makedirs(temp_dir, exist_ok=True)

    ndx_file_temp = os.path.join(temp_dir, f"sample_{sample_idx:04d}_path_{path_idx:02d}.ndx")
    create_ndx_file_for_path(ndx_file_temp, hprime_path, beta_path, distances)

    inp_file_temp = os.path.join(temp_dir, f"sample_{sample_idx:04d}_path_{path_idx:02d}.inp")
    create_inp_file_for_path(
        inp_file_temp, ndx_file_temp, tx_info, rx_info,
        sample_idx, path_idx, distances, temp_dir
    )

    inp_basename = f"sample_{sample_idx:04d}_path_{path_idx:02d}"
    inp_file_lwpc = os.path.join(C.SUPERLWPC_WORK_DIR, f"{inp_basename}.inp")
    ndx_file_lwpc = os.path.join(C.SUPERLWPC_WORK_DIR, f"{inp_basename}.ndx")

    shutil.copy2(inp_file_temp, inp_file_lwpc)
    shutil.copy2(ndx_file_temp, ndx_file_lwpc)

    # 调试输出
    try:
        with open(inp_file_temp, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            print("    生成的输入文件内容:")
            for i, line in enumerate(lines):
                print(f"      {i+1}: {line.rstrip()}")
    except Exception as e:
        print(f"    读取输入文件失败: {e}")

    print(f"    已将输入文件复制到LWPC工作目录: {C.SUPERLWPC_WORK_DIR}")
    return inp_basename, inp_file_lwpc, ndx_file_lwpc


def cleanup_lwpc_files(inp_basename, work_dir=None):
    """
    清理LWPC工作目录中的临时文件，并将log文件转移到工作目录
    【保持原有代码不变】
    """
    try:
        lwpc_log_file = os.path.join(C.SUPERLWPC_WORK_DIR, f"{inp_basename}.log")
        if os.path.exists(lwpc_log_file) and work_dir:
            work_subdir = os.path.join(work_dir, inp_basename)
            os.makedirs(work_subdir, exist_ok=True)
            target_log = os.path.join(work_subdir, f"{inp_basename}.log")
            shutil.move(lwpc_log_file, target_log)
            print(f"    已转移log文件到: {target_log}")

        file_prefixes = [
            f"{inp_basename}.inp",
            f"{inp_basename}.ndx",
            f"{inp_basename}.log",
            f"{inp_basename}.lwf",
            f"{inp_basename}.mds",
            f"{inp_basename}.grd",
            f"{inp_basename}.",
        ]

        for file_name in os.listdir(C.SUPERLWPC_WORK_DIR):
            file_path = os.path.join(C.SUPERLWPC_WORK_DIR, file_name)
            if any(file_name == p or file_name.startswith(p) for p in file_prefixes):
                # 避免误删 lwpc.bin.exe 等
                if file_name.lower().endswith(('.exe', '.dll')):
                    continue
                try:
                    if os.path.exists(file_path) and os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"    已删除: {file_path}")
                except Exception as e:
                    print(f"    删除文件 {file_path} 时出错: {e}")

        if work_dir:
            for dir_name in os.listdir(work_dir):
                dir_path = os.path.join(work_dir, dir_name)
                if inp_basename in dir_name and os.path.isdir(dir_path):
                    if not os.path.exists(os.path.join(dir_path, f"{inp_basename}.log")):
                        try:
                            shutil.rmtree(dir_path)
                            print(f"    已清理工作目录: {dir_path}")
                        except Exception as e:
                            print(f"    清理工作目录时出错: {e}")

    except Exception as e:
        print(f"    清理文件时出错: {e}")


def run_lwpc_for_path(
    sample_idx,
    path_idx,
    hprime_path,
    beta_path,
    distances,
    tx_info,
    rx_info,
    work_dir
):
    """
    为单个路径运行LWPC计算
    【保持原有代码不变】
    """
    print(f"  样本 {sample_idx}, 路径 {path_idx}: 准备运行LWPC计算...")

    work_subdir = os.path.join(work_dir, f"sample_{sample_idx:04d}_path_{path_idx:02d}")
    os.makedirs(work_subdir, exist_ok=True)

    try:
        inp_basename, _, _ = prepare_lwpc_input_files(
            sample_idx, path_idx, hprime_path, beta_path, distances,
            tx_info, rx_info, work_dir
        )

        print(f"    在目录 {C.SUPERLWPC_WORK_DIR} 中运行LWPC: {C.SUPERLWPC_EXECUTABLE} {inp_basename}")
        cmd = [C.SUPERLWPC_EXECUTABLE, inp_basename]

        result = subprocess.run(
            cmd,
            cwd=C.SUPERLWPC_WORK_DIR,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='ignore',
            timeout=300
        )

        if result.returncode != 0:
            print(f"    LWPC执行失败，返回码: {result.returncode}")
            if result.stderr:
                print(f"    错误输出: {result.stderr[:800]}")

            error_log = os.path.join(work_subdir, "error.log")
            with open(error_log, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(f"返回码: {result.returncode}\n")
                f.write(f"标准输出:\n{result.stdout}\n")
                f.write(f"标准错误:\n{result.stderr}\n")

            cleanup_lwpc_files(inp_basename, work_dir)
            return False, None

        if "STOP Normal run complete" not in (result.stdout or ""):
            print("    LWPC可能未正常完成（未检测到 STOP Normal run complete）")
            print(f"    标准输出前800字符: {(result.stdout or '')[:800]}")
            warning_log = os.path.join(work_subdir, "warning.log")
            with open(warning_log, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(f"标准输出:\n{result.stdout}\n")

        log_file_lwpc = os.path.join(C.SUPERLWPC_WORK_DIR, f"{inp_basename}.log")
        lwf_file_lwpc = os.path.join(C.SUPERLWPC_WORK_DIR, f"{inp_basename}.lwf")

        output_files_found = []
        if os.path.exists(log_file_lwpc):
            output_files_found.append(log_file_lwpc)
        if os.path.exists(lwf_file_lwpc):
            output_files_found.append(lwf_file_lwpc)

        if not output_files_found:
            print("    未找到输出文件")
            print(f"    LWPC工作目录内容(前10): {os.listdir(C.SUPERLWPC_WORK_DIR)[:10]}")
            cleanup_lwpc_files(inp_basename, work_dir)
            return False, None

        print(f"    找到输出文件: {', '.join([os.path.basename(f) for f in output_files_found])}")

        # 优先解析 log
        if os.path.exists(log_file_lwpc):
            work_log_file = os.path.join(work_subdir, f"{inp_basename}.log")
            shutil.copy2(log_file_lwpc, work_log_file)
            results = parse_lwpc_log(work_log_file)
            if results:
                print("    成功解析输出，获得电场数据")
                cleanup_lwpc_files(inp_basename, work_dir)
                return True, results
            else:
                print("    解析log文件失败，尝试解析LWF文件")

        if os.path.exists(lwf_file_lwpc):
            work_lwf_file = os.path.join(work_subdir, f"{inp_basename}.lwf")
            shutil.copy2(lwf_file_lwpc, work_lwf_file)
            results = parse_lwpc_lwf(work_lwf_file)
            if results:
                print("    从LWF文件成功解析输出")
                cleanup_lwpc_files(inp_basename, work_dir)
                return True, results

        print("    解析输出文件失败")
        cleanup_lwpc_files(inp_basename, work_dir)
        return False, None

    except subprocess.TimeoutExpired:
        print("    LWPC计算超时")
        cleanup_lwpc_files(inp_basename, work_dir)
        return False, None
    except Exception as e:
        print(f"    LWPC执行异常: {e}")
        import traceback
        traceback.print_exc()
        cleanup_lwpc_files(inp_basename, work_dir)
        return False, None


def parse_lwpc_log(log_file_path):
    """
    解析LWPC日志文件，提取电场分量数据
    【保持原有代码不变】
    """
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.read().splitlines()

        component_data = {}
        current_component = None

        distances = []
        amplitudes = []
        phases = []

        def flush_component():
            nonlocal distances, amplitudes, phases, current_component, component_data
            if current_component and distances:
                amp_aligned, phase_aligned = align_lwpc_data(distances, amplitudes, phases)
                component_data[current_component] = (amp_aligned, phase_aligned)

        for raw in lines:
            line = raw.strip()
            low = line.lower()

            # 组件切换
            if "x component" in low:
                flush_component()
                current_component = 'x'
                distances, amplitudes, phases = [], [], []
                continue
            if "y component" in low:
                flush_component()
                current_component = 'y'
                distances, amplitudes, phases = [], [], []
                continue
            if "z component" in low:
                flush_component()
                current_component = 'z'
                distances, amplitudes, phases = [], [], []
                continue

            # 跳过表头/无关行
            if current_component:
                # 更稳健的表头识别
                if ("dist" in low and "amplitude" in low and "phase" in low):
                    continue

                # 数据行：可能一行多个三元组
                if line and (line[0].isdigit() or (len(line) > 1 and line[0] == '-' and line[1].isdigit())):
                    parts = line.split()
                    num_triplets = len(parts) // 3
                    for i in range(num_triplets):
                        try:
                            d = float(parts[i * 3])
                            a = float(parts[i * 3 + 1])
                            p = float(parts[i * 3 + 2])
                            distances.append(d)
                            amplitudes.append(a)
                            phases.append(p)
                        except Exception:
                            continue

        flush_component()
        return component_data

    except Exception as e:
        print(f"解析LWPC日志文件错误: {e}")
        return {}


def parse_lwpc_lwf(lwf_file_path):
    """
    解析LWPC LWF文件（备选方法）
    当前仍未完整实现
    【保持原有代码不变】
    """
    try:
        with open(lwf_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            _ = f.read()
        print("    警告：LWF文件解析功能未完全实现")
        return {}
    except Exception as e:
        print(f"解析LWPC LWF文件错误: {e}")
        return {}


def align_lwpc_data(distances, amplitudes, phases):
    """
    对齐LWPC输出数据到标准距离网格
    【保持原有代码不变】
    """
    distances = np.asarray(distances, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    phases = np.asarray(phases, dtype=float)

    n_out = int(getattr(C, "PATH_SEGMENTS", 100))

    if distances.size == 0:
        return (
            np.full(n_out, -99.0, dtype=np.float32),
            np.zeros(n_out, dtype=np.float32),
        )

    # 乱序/重复处理
    order = np.argsort(distances)
    d_sorted = distances[order]
    a_sorted = amplitudes[order]
    p_sorted = phases[order]

    uniq_d, uniq_idx = np.unique(d_sorted, return_index=True)
    a_uniq = a_sorted[uniq_idx]
    p_uniq = p_sorted[uniq_idx]

    max_dist = float(np.max(uniq_d)) if uniq_d.size else 8000.0
    expected = np.linspace(0.0, max_dist, n_out)

    # 相位：unwrap -> 插值 -> wrap
    p_rad = np.deg2rad(p_uniq)
    p_unwrap = np.unwrap(p_rad)
    p_unwrap_deg = np.rad2deg(p_unwrap)

    try:
        amp_aligned = np.interp(expected, uniq_d, a_uniq, left=-99.0, right=-99.0).astype(np.float32)
        pha_interp = np.interp(expected, uniq_d, p_unwrap_deg, left=p_unwrap_deg[0], right=p_unwrap_deg[-1])
        pha_wrapped = ((pha_interp + 180.0) % 360.0) - 180.0
        phase_aligned = pha_wrapped.astype(np.float32)
    except Exception as e:
        print(f"    距离对齐插值失败: {e}")
        amp_aligned = np.full(n_out, -99.0, dtype=np.float32)
        phase_aligned = np.zeros(n_out, dtype=np.float32)

    return amp_aligned, phase_aligned