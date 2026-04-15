#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
准备因子分析所需的CSV数据文件
将pkl格式的数据转换为C++程序可读的CSV格式
"""

import pickle
import pandas as pd
import sys

def prepare_factor_data(factor_file='factors_by_type/alpha101/WorldQuant_alpha014.pkl',
                       data_pkl='data.pkl', output_csv='factor_data.csv'):
    """
    准备因子分析数据
    
    参数:
        factor_file: 因子数据pkl文件路径
        data_pkl: 行情数据pkl文件路径
        output_csv: 输出的CSV文件路径
    """
    
    print(f"Loading data from {data_pkl}...")
    
    # 加载行情数据
    with open(data_pkl, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")
    
    print(f"\nLoading factor from {factor_file}...")
    factor_data = pd.read_pickle(factor_file)
    
    print(f"Factor shape: {factor_data.shape}")
    print(f"Factor columns: {list(factor_data.columns)}")
    
    # 合并数据
    print("\nMerging data...")
    
    # 确保索引一致
    data = data.set_index(['order_book_id', 'date'])
    
    # 提取因子值（假设因子只有一列）
    factor_name = factor_data.columns[0]
    data['factor_value'] = factor_data[factor_name]
    
    # 重置索引
    data = data.reset_index()
    
    # 选择需要的列
    output_columns = ['date', 'order_book_id', 'close', 'factor_value']
    for col in ['limit_up_flag', 'limit_down_flag', 'ST', 'suspended']:
        if col in data.columns:
            if col == 'limit_up_flag':
                data['is_limit_up'] = data[col].astype(int)
            elif col == 'limit_down_flag':
                data['is_limit_down'] = data[col].astype(int)
            elif col == 'ST':
                data['is_st'] = data[col].astype(int)
            elif col == 'suspended':
                data['is_suspended'] = data[col].astype(int)
    
    # 重命名factor_value列
    if 'factor_value' in data.columns:
        data = data.rename(columns={'factor_value': 'factor_value'})
    
    # 准备输出的列
    output_data = pd.DataFrame()
    output_data['date'] = data['date'].astype(str)
    output_data['order_book_id'] = data['order_book_id'].astype(str)
    output_data['close'] = data['close']
    
    # 添加因子值
    if 'factor_value' in data.columns:
        output_data['factor_value'] = data['factor_value']
    else:
        print("Warning: No factor_value found, using dummy values")
        output_data['factor_value'] = 0.0
    
    # 添加布尔字段
    output_data['is_st'] = data.get('is_st', 0)
    output_data['is_limit_up'] = data.get('is_limit_up', 0)
    output_data['is_limit_down'] = data.get('is_limit_down', 0)
    output_data['is_suspended'] = data.get('is_suspended', 0)
    
    # 处理缺失值
    output_data = output_data.dropna(subset=['close', 'factor_value'])
    
    print(f"\nOutput data shape: {output_data.shape}")
    print(f"Date range: {output_data['date'].min()} to {output_data['date'].max()}")
    
    # 保存为CSV
    print(f"\nSaving to {output_csv}...")
    output_data.to_csv(output_csv, index=False)
    
    print(f"Data prepared successfully!")
    print(f"Total records: {len(output_data):,}")
    
    return output_data

if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])
    # 检查命令行参数
    if len(sys.argv) > 1:
        factor_file = sys.argv[1]
        prepare_factor_data(factor_file=factor_file)
    else:
        print("Usage: python prepare_data.py <factor_file>")
        print("Example: python prepare_data.py factors_by_type/alpha101/WorldQuant_alpha014.pkl")
        print("\nUsing default factor...")
        prepare_factor_data()

