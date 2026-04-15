#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按类型批量下载A股因子数据 - 优先下载小类型
"""

import rqdatac as rq
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import pickle
import logging

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('factor_download_by_type.log'),
        logging.StreamHandler()
    ]
)

def init_rqdatac():
    """初始化rqdatac"""
    try:
        RICE_QUANT_API_TOKEN = "csgY6xjHzy_rqdMIqmDwsg_6XLXyQ5vrsxNKxkvri6s3mQ4sunRhPNPZ9pvMXG-xxuVZHODietCupsqhocF-1tWIwR6tvOyj03WiOuCrrbK5WipPw6uDEdI6b5dpxqvw9ia-56KkgUoa6bMJ1qvFVqcaw-2F9zrCmBOH0KGtCK4=DU7RS7YnfXgwGtbjDMuAurCxha2ewCRzweL9r4pM9Az-aSU3P6CufjIgtMsT99Os_tiBwMVcrji2R9HnXiTLZu8R9uxCqCDbbpLPzwc85eU-xbIB0jGqifyMqwCmBIZHXrdKaf339JJ2mU9X0En8Te369rE4I5DWGVyJoWe7orI="
        # RICE_QUANT_API_TOKEN = "OUAxlh6_oGHkXZTE9SFuLXmbAabdkJl7b56FkXfo6pxo4tTpNRgneD0Qhbd6YAadaZ2VWGwF5_INmA9IB3EE0jd_JKIh0doSG0hC1JSHFDnSS5ie66IcA_mS0uG1Ac0gJq_QpdPrwxgUanfN_vZs0doGxzK9hPwprFlN5xPuVR0=IfuVvS_P3IFvpPIGQfh8S7Ri7U4K-BFhJGxNh8H2TzquUeEnbeLhCBSuMasPFoK-khXZhSsYv-tM8LWulhCPpowQoOIp2Xw-7TRlALTNTIhsYJnisfrBHbt34XOmtd30xDRDtFzcIZfPTBIbk-K9pL40wv0JplN1YLaYO-TYmQc="
        rq.init("license", RICE_QUANT_API_TOKEN)
        # rq.init()
        logging.info("rqdatac初始化成功")
        return True
    except Exception as e:
        logging.error(f"rqdatac初始化失败: {e}")
        return False

def get_factor_names_by_type(factor_type):
    """按类型获取因子名称"""
    try:
        factor_names = rq.get_all_factor_names(type=factor_type, market='cn')
        logging.info(f"获取到{factor_type}类型因子{len(factor_names)}个")
        return factor_names
    except Exception as e:
        logging.error(f"获取{factor_type}类型因子失败: {e}")
        return []

def get_stock_list():
    """获取A股股票列表"""
    try:
        stocks = rq.all_instruments(type='CS')
        stock_ids = stocks.order_book_id.tolist()
        logging.info(f"获取到{len(stock_ids)}只A股股票")
        return stock_ids
    except Exception as e:
        logging.error(f"获取股票列表失败: {e}")
        return []

def download_factor_batch(stock_ids, factor_name, start_date, end_date):
    """一次性下载单个因子的所有股票数据"""
    try:
        logging.info(f"下载因子 {factor_name} - 一次性下载全部 {len(stock_ids)}只股票")
        
        factor_data = rq.get_factor(
            order_book_ids=stock_ids,
            factor=factor_name,
            start_date=start_date,
            end_date=end_date
        )
        
        if factor_data is not None and not factor_data.empty:
            logging.info(f"因子 {factor_name} 下载完成，数据量: {factor_data.shape}")
            return factor_data
        else:
            logging.warning(f"因子 {factor_name} 返回空数据")
            return None
                
    except Exception as e:
        logging.error(f"下载因子 {factor_name} 失败: {e}")
        return None

def save_factor_data(factor_data, factor_name, output_dir):
    """保存因子数据"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, f'{factor_name}.pkl')
    try:
        factor_data.to_pickle(filename)
        logging.info(f"因子 {factor_name} 数据已保存到 {filename}")
        return True
    except Exception as e:
        logging.error(f"保存因子 {factor_name} 失败: {e}")
        return False

def download_factors_by_type(factor_type, start_date='2010-01-01', end_date=None, 
                            output_dir='factors_by_type'):
    """按类型下载因子数据"""
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    logging.info(f"开始下载{factor_type}类型因子数据: {start_date} 到 {end_date}")
    
    # 获取因子列表
    factor_names = get_factor_names_by_type(factor_type)
    if not factor_names:
        logging.warning(f"没有找到{factor_type}类型的因子")
        return False
    
    # 获取股票列表
    stock_ids = get_stock_list()
    if not stock_ids:
        return False
    
    # 创建输出目录
    type_output_dir = os.path.join(output_dir, factor_type)
    if not os.path.exists(type_output_dir):
        os.makedirs(type_output_dir)
    
    # 统计信息
    success_count = 0
    failed_count = 0
    failed_factors = []
    
    # 如果是 alpha101 类型，从 alpha014 开始
    if factor_type == 'alpha101':
        # 找到 alpha014 的索引
        try:
            alpha014_idx = next(i for i, name in enumerate(factor_names) if 'alpha014' in name.lower())
            factor_names = factor_names[alpha014_idx:]
            logging.info(f"从 alpha014 开始下载，剩余 {len(factor_names)} 个因子")
        except StopIteration:
            logging.warning("未找到 alpha014，从头开始下载")
    
    # 下载每个因子
    for i, factor_name in enumerate(factor_names):
        logging.info(f"开始下载因子 {i+1}/{len(factor_names)}: {factor_name}")
        
        try:
            # 检查是否已经下载过
            filename = os.path.join(type_output_dir, f'{factor_name}.pkl')
            if os.path.exists(filename):
                logging.info(f"因子 {factor_name} 已存在，跳过")
                success_count += 1
                continue
            
            # 下载因子数据
            factor_data = download_factor_batch(
                stock_ids, factor_name, start_date, end_date
            )
            
            if factor_data is not None:
                # 保存数据
                if save_factor_data(factor_data, factor_name, type_output_dir):
                    success_count += 1
                else:
                    failed_count += 1
                    failed_factors.append(factor_name)
            else:
                failed_count += 1
                failed_factors.append(factor_name)
                
        except Exception as e:
            logging.error(f"下载因子 {factor_name} 时发生错误: {e}")
            failed_count += 1
            failed_factors.append(factor_name)
        
        # 每下载10个因子休息一下
        if (i + 1) % 10 == 0:
            logging.info(f"已处理 {i+1} 个因子，成功 {success_count} 个，失败 {failed_count} 个")
            time.sleep(1)
    
    # 输出最终统计
    logging.info("="*60)
    logging.info(f"{factor_type}类型因子下载完成!")
    logging.info(f"总因子数: {len(factor_names)}")
    logging.info(f"成功下载: {success_count}")
    logging.info(f"下载失败: {failed_count}")
    
    if failed_factors:
        logging.info(f"失败的因子: {failed_factors[:10]}...")  # 只显示前10个
    
    return True

def create_factor_summary_by_type(output_dir='factors_by_type'):
    """创建按类型分组的因子数据汇总"""
    if not os.path.exists(output_dir):
        logging.warning(f"输出目录 {output_dir} 不存在")
        return
    
    summary_data = []
    
    # 遍历每个类型目录
    for factor_type in os.listdir(output_dir):
        type_dir = os.path.join(output_dir, factor_type)
        if not os.path.isdir(type_dir):
            continue
            
        factor_files = [f for f in os.listdir(type_dir) if f.endswith('.pkl')]
        
        for factor_file in factor_files:
            factor_name = factor_file.replace('.pkl', '')
            filepath = os.path.join(type_dir, factor_file)
            
            try:
                factor_data = pd.read_pickle(filepath)
                summary_data.append({
                    'factor_type': factor_type,
                    'factor_name': factor_name,
                    'shape': factor_data.shape,
                    'rows': factor_data.shape[0],
                    'cols': factor_data.shape[1],
                    'columns': list(factor_data.columns),
                    'file_size_mb': os.path.getsize(filepath) / 1024 / 1024,
                    'has_data': not factor_data.empty
                })
            except Exception as e:
                logging.error(f"读取因子文件 {factor_file} 失败: {e}")
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'factor_summary_by_type.csv'), index=False)
    logging.info(f"因子汇总已保存到 {output_dir}/factor_summary_by_type.csv")
    
    return summary_df

if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])
    # 只下载指定的两种类型
    factor_types = [
        ('eod_indicator', 41),            # 估值指标 (41个)
        ('financial_indicator', 126),     # 财务指标 (126个)
        ('alpha101', 101),                # alpha101因子 (101个)
        # ('financial_indicator', 126),     # 财务指标 (126个)

    ]
    
    print("开始按类型下载因子数据...")
    print(f"将下载 {len(factor_types)} 种类型的因子")
    print("按因子数量从小到大排序，优先下载小类型")
    
    # 初始化
    if not init_rqdatac():
        print("初始化失败!")
        exit(1)
    
    # 逐个类型下载
    for i, (factor_type, expected_count) in enumerate(factor_types):
        print(f"\n开始下载第 {i+1}/{len(factor_types)} 种类型: {factor_type} (预期{expected_count}个因子)")
        
        success = download_factors_by_type(
            factor_type=factor_type,
            start_date='2010-01-01',
            end_date='2025-09-30',
            output_dir='factors_by_type'
        )
        
        if success:
            print(f"{factor_type} 类型下载完成!")
        else:
            print(f"{factor_type} 类型下载失败!")
        
        # 每下载完一个类型休息一下
        if i < len(factor_types) - 1:
            print("休息10秒后继续...")
            time.sleep(10)
    
    print("\n所有类型因子下载完成!")
    print("创建汇总文件...")
    
    summary = create_factor_summary_by_type('factors_by_type')
    if summary is not None:
        print(f"汇总完成! 共下载 {len(summary)} 个因子")
        print("\n按类型统计:")
        type_stats = summary.groupby('factor_type').size().sort_values(ascending=False)
        for factor_type, count in type_stats.items():
            print(f"  {factor_type}: {count} 个因子")
    else:
        print("汇总失败!")
