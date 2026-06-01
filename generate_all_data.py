# generate_all_data.py
# -*- coding: utf-8 -*-
"""
主程序：生成完整的北美VLF数据集
"""

import sys
import os
import argparse
import time

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config as C

def main():
    parser = argparse.ArgumentParser(description='生成北美VLF数据集')
    parser.add_argument('--step', type=int, default=0, 
                       help='步骤: 0=全部, 1=生成电离层状态, 2=计算VLF路径')
    parser.add_argument('--samples', type=int, default=C.NUM_SAMPLES,
                       help='样本数量')
    parser.add_argument('--grid-spacing', type=float, default=C.GRID_SPACING_KM,
                       help='网格间距 (km)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("北美VLF数据集生成程序")
    print("="*70)
    print(f"网格间距: {args.grid_spacing} km")
    print(f"样本数量: {args.samples}")
    print(f"发射机数量: {len(C.TRANSMITTERS)}")
    print(f"接收机数量: {len(C.RECEIVERS)}")
    print(f"路径数量: {len(C.ALL_PATHS)}")
    print(f"每路径分段数: {C.PATH_SEGMENTS}")
    print("="*70)
    
    # 更新配置
    if args.samples != C.NUM_SAMPLES:
        C.NUM_SAMPLES = args.samples
        print(f"更新样本数量为: {C.NUM_SAMPLES}")
    
    if args.grid_spacing != C.GRID_SPACING_KM:
        C.GRID_SPACING_KM = args.grid_spacing
        print(f"更新网格间距为: {C.GRID_SPACING_KM} km")
    
    total_start_time = time.time()
    
    # 步骤1: 生成电离层状态
    if args.step in [0, 1]:
        print("\n" + "="*60)
        print("步骤1: 生成网格化电离层状态")
        print("="*60)
        
        step1_start = time.time()
        
        try:
            from generate_ionosphere_states import main as step1_main
            success = step1_main()
            
            if not success:
                print("步骤1失败，程序终止")
                return False
            
            step1_time = time.time() - step1_start
            print(f"步骤1完成，耗时: {step1_time:.2f} 秒")
            
        except Exception as e:
            print(f"步骤1执行过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    # 步骤2: 计算VLF路径
    if args.step in [0, 2]:
        print("\n" + "="*60)
        print("步骤2: 计算VLF路径电场数据")
        print("="*60)
        print("注意: 此步骤可能需要较长时间（数小时到数天）")
        print("取决于样本数量和计算资源")
        print("="*60)
        
        step2_start = time.time()
        
        try:
            from compute_vlf_paths import main as step2_main
            success = step2_main()
            
            if not success:
                print("步骤2失败，程序终止")
                return False
            
            step2_time = time.time() - step2_start
            print(f"步骤2完成，耗时: {step2_time:.2f} 秒")
            
        except Exception as e:
            print(f"步骤2执行过程中发生错误: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    total_time = time.time() - total_start_time
    
    print("\n" + "="*70)
    print("项目完成!")
    print("="*70)
    print(f"总耗时: {total_time:.2f} 秒 ({total_time/3600:.2f} 小时)")
    print(f"最终数据集: {C.FINAL_DATASET_FILE}")
    print("\n数据集包含:")
    print(f"  1. {C.NUM_SAMPLES} 个网格化电离层状态")
    print(f"  2. 每个状态包含 {len(C.ALL_PATHS)} 条VLF传播路径")
    print(f"  3. 每条路径包含 {C.PATH_SEGMENTS} 个分段")
    print(f"  4. 每个分段包含x,y,z三个分量的振幅和相位")
    print(f"  5. 网格间距: {C.GRID_SPACING_KM} km")
    print(f"  6. 投影方式: 北美等距圆锥投影")
    print("="*70)
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n用户中断程序执行")
        sys.exit(1)
    except Exception as e:
        print(f"程序执行过程中发生未预期错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)