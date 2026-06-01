# generate_ionosphere_states.py
# -*- coding: utf-8 -*-
"""
生成网格化的电离层状态（prior & true）
（规则栅格版本：保存 grid_valid_mask）
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import config as C
from grid_utils import create_north_america_grid
from ionosphere_models import generate_all_ionosphere_states, save_ionosphere_states


def main():
    """主函数：生成网格化的电离层状态"""
    print("=" * 60)
    print("北美VLF数据集 - 电离层状态生成（prior & true）")
    print("=" * 60)

    try:
        print("\n步骤1: 创建北美等距圆锥投影规则栅格...")
        grid_data = create_north_america_grid()

        np.savez(
            C.GRID_DATA_FILE,
            proj_coords=grid_data['proj_coords'],
            latlon_coords=grid_data['latlon_coords'],
            grid_shape=np.array(grid_data['grid_shape'], dtype=np.int32),
            x_grid=grid_data['x_grid'],
            y_grid=grid_data['y_grid'],
            valid_mask=grid_data.get('valid_mask', None),
        )
        print(f"网格数据已保存到: {C.GRID_DATA_FILE}")

        print(f"\n步骤2: 生成 {C.NUM_SAMPLES} 个电离层状态（prior & true）...")
        hp_prior, be_prior, hp_true, be_true, all_metadata = generate_all_ionosphere_states(
            grid_data, C.NUM_SAMPLES
        )

        print("\n步骤3: 保存电离层状态到HDF5文件...")
        save_ionosphere_states(
            grid_data,
            hp_prior, be_prior,
            hp_true, be_true,
            all_metadata,
            C.IONOSPHERE_STATES_FILE
        )

        print("\n统计信息:")
        print(f"  样本数量: {len(hp_true)}")
        H, W = grid_data['grid_shape']
        print(f"  网格尺寸: H={H}, W={W}, N={H*W}")
        print(f"  网格间距: {C.GRID_SPACING_KM} km")
        print(f"  prior h'平均值: {np.mean(hp_prior):.2f} km")
        print(f"  true  h'平均值: {np.mean(hp_true):.2f} km")
        print(f"  prior β平均值: {np.mean(be_prior):.3f} km^-1")
        print(f"  true  β平均值: {np.mean(be_true):.3f} km^-1")

        print("\n电离层状态生成完成!")
        print(f"输出文件: {C.IONOSPHERE_STATES_FILE}")
        return True

    except Exception as e:
        print(f"生成电离层状态时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)