#!/usr/bin/env python3
"""
将 .pkl 文件转换为 .npz 文件的工具脚本
支持多种常见的pkl格式（包括numpy数组、字典、列表等）
"""


# # 基本转换
# python pkl_to_npz.py data.pkl

# # 指定输出文件名
# python pkl_to_npz.py data.pkl -o my_data.npz

# # 不使用压缩（生成更大的文件，但加载可能更快）
# python pkl_to_npz.py data.pkl --no-compress

# # 禁止使用pickle（只支持纯numpy数组）
# python pkl_to_npz.py data.pkl --no-pickle

# # 显示详细信息
# python pkl_to_npz.py data.pkl -v

import pickle
import numpy as np
import argparse
import os
import sys
from pathlib import Path


def load_pkl(file_path):
    """
    加载.pkl文件
    
    Args:
        file_path: pkl文件路径
    
    Returns:
        加载的数据
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
    except Exception as e:
        print(f"Error loading pkl file: {e}")
        sys.exit(1)


def convert_to_npz(data, output_path, compressed=True, allow_pickle=True):
    """
    将数据转换为.npz格式
    
    Args:
        data: 要转换的数据
        output_path: 输出文件路径
        compressed: 是否使用压缩（.npz vs .npz）
        allow_pickle: 是否允许pickle序列化（用于非numpy对象）
    """
    if isinstance(data, dict):
        # 如果数据是字典，直接保存所有键值对
        npz_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                npz_data[key] = value
            elif isinstance(value, (list, tuple)):
                try:
                    npz_data[key] = np.array(value)
                except:
                    print(f"Warning: Key '{key}' contains non-numeric data, storing as object")
                    npz_data[key] = np.array(value, dtype=object)
            else:
                # 对于其他类型，尝试保存为对象数组
                print(f"Warning: Key '{key}' is not an array (type: {type(value)}), storing as object")
                npz_data[key] = np.array(value, dtype=object)
        
        if compressed:
            np.savez_compressed(output_path, **npz_data, allow_pickle=allow_pickle)
        else:
            np.savez(output_path, **npz_data, allow_pickle=allow_pickle)
            
    elif isinstance(data, (np.ndarray, list, tuple)):
        # 如果是单个数组/列表/元组，保存为 'data' 键
        if isinstance(data, list):
            data = np.array(data)
        if compressed:
            np.savez_compressed(output_path, data=data, allow_pickle=allow_pickle)
        else:
            np.savez(output_path, data=data, allow_pickle=allow_pickle)
            
    else:
        # 其他类型，尝试保存为对象
        print(f"Warning: Data is not a dict or array (type: {type(data)}), saving as object")
        obj_array = np.array(data, dtype=object)
        if compressed:
            np.savez_compressed(output_path, data=obj_array, allow_pickle=allow_pickle)
        else:
            np.savez(output_path, data=obj_array, allow_pickle=allow_pickle)


def main():
    parser = argparse.ArgumentParser(
        description="Convert .pkl files to .npz files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pkl_to_npz.py data.pkl
  python pkl_to_npz.py data.pkl -o output.npz
  python pkl_to_npz.py data.pkl --no-compress
  python pkl_to_npz.py data.pkl --no-pickle
        """
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Input .pkl file path'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output .npz file path (default: same name as input with .npz extension)'
    )
    
    parser.add_argument(
        '--no-compress',
        action='store_true',
        help='Disable compression (save as .npz instead of .npz_compressed)'
    )
    
    parser.add_argument(
        '--no-pickle',
        action='store_true',
        help='Disable pickle in npz (may fail for non-numpy objects)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed information'
    )
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file '{args.input}' not found")
        sys.exit(1)
    
    if not input_path.suffix.lower() == '.pkl':
        print(f"Warning: Input file does not have .pkl extension: {args.input}")
    
    # 确定输出路径
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.with_suffix('.npz')
    
    # 确认输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 打印信息
    if args.verbose:
        print(f"Input file: {input_path}")
        print(f"Output file: {output_path}")
        print(f"Compression: {'Disabled' if args.no_compress else 'Enabled'}")
        print(f"Allow pickle: {'Disabled' if args.no_pickle else 'Enabled'}")
    
    # 加载pkl文件
    if args.verbose:
        print("\nLoading pkl file...")
    data = load_pkl(input_path)
    if args.verbose:
        print(f"Loaded data type: {type(data)}")
        if isinstance(data, dict):
            print(f"Number of keys: {len(data)}")
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    print(f"  - {key}: array shape {value.shape}, dtype {value.dtype}")
                else:
                    print(f"  - {key}: {type(value)}")
        elif isinstance(data, np.ndarray):
            print(f"Array shape: {data.shape}, dtype: {data.dtype}")
        else:
            print(f"Data type: {type(data)}")
    
    # 转换为npz
    if args.verbose:
        print("\nConverting to npz...")
    try:
        convert_to_npz(
            data, 
            output_path, 
            compressed=not args.no_compress,
            allow_pickle=not args.no_pickle
        )
        
        file_size = output_path.stat().st_size
        print(f"\nSuccessfully converted to: {output_path}")
        print(f"File size: {file_size / 1024:.2f} KB")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
