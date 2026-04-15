#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量下载A股所有因子数据
从2010年至今的所有股票的所有因子
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
        logging.FileHandler('factor_download.log'),
        logging.StreamHandler()
    ]
)

def init_rqdatac():
    """初始化rqdatac"""
    try:
        
        rq.init("license", RICE_QUANT_API_TOKEN)
        logging.info("rqdatac初始化成功")
        return True
    except Exception as e:
        logging.error(f"rqdatac初始化失败: {e}")
        return False

def get_factor_names():
    """获取所有因子名称"""
    try:
        factor_names = rq.get_all_factor_names(type=None, market='cn')
        logging.info(f"获取到{len(factor_names)}个因子")
        return factor_names
    except Exception as e:
        logging.error(f"获取因子名称失败: {e}")
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

def download_factor_batch(stock_ids, factor_name, start_date, end_date, batch_size=100):
    """分批下载单个因子的数据"""
    all_data = []
    total_stocks = len(stock_ids)
    
    for i in range(0, total_stocks, batch_size):
        batch_stocks = stock_ids[i:i+batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_stocks + batch_size - 1) // batch_size
        
        try:
            logging.info(f"下载因子 {factor_name} - 批次 {batch_num}/{total_batches} ({len(batch_stocks)}只股票)")
            
            factor_data = rq.get_factor(
                order_book_ids=batch_stocks,
                factor=factor_name,
                start_date=start_date,
                end_date=end_date
            )
            
            if factor_data is not None and not factor_data.empty:
                all_data.append(factor_data)
                logging.info(f"批次 {batch_num} 成功下载 {factor_data.shape[0]} 条数据")
            else:
                logging.warning(f"批次 {batch_num} 返回空数据")
                
        except Exception as e:
            logging.error(f"批次 {batch_num} 下载失败: {e}")
            continue
        
        # 避免请求过于频繁
        time.sleep(0.1)
    
    if all_data:
        combined_data = pd.concat(all_data, ignore_index=False)
        logging.info(f"因子 {factor_name} 下载完成，总数据量: {combined_data.shape}")
        return combined_data
    else:
        logging.warning(f"因子 {factor_name} 没有下载到任何数据")
        return None

def save_factor_data(factor_data, factor_name, output_dir='factors'):
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

def download_all_factors(start_date='2010-01-01', end_date=None, 
                        max_factors=None, batch_size=100, 
                        output_dir='factors'):
    """下载所有因子数据"""
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    logging.info(f"开始下载因子数据: {start_date} 到 {end_date}")
    
    # 初始化
    if not init_rqdatac():
        return False
    
    # 获取因子列表
    factor_names = get_factor_names()
    if not factor_names:
        return False
    
    # 限制因子数量（用于测试）
    if max_factors:
        factor_names = factor_names[:max_factors]
        logging.info(f"限制下载前{max_factors}个因子")
    
    # 获取股票列表
    stock_ids = get_stock_list()
    if not stock_ids:
        return False
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 统计信息
    success_count = 0
    failed_count = 0
    failed_factors = []
    
    # 下载每个因子
    for i, factor_name in enumerate(factor_names):
        logging.info(f"开始下载因子 {i+1}/{len(factor_names)}: {factor_name}")
        
        try:
            # 检查是否已经下载过
            filename = os.path.join(output_dir, f'{factor_name}.pkl')
            if os.path.exists(filename):
                logging.info(f"因子 {factor_name} 已存在，跳过")
                success_count += 1
                continue
            
            # 下载因子数据
            factor_data = download_factor_batch(
                stock_ids, factor_name, start_date, end_date, batch_size
            )
            
            if factor_data is not None:
                # 保存数据
                if save_factor_data(factor_data, factor_name, output_dir):
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
    logging.info("下载完成!")
    logging.info(f"总因子数: {len(factor_names)}")
    logging.info(f"成功下载: {success_count}")
    logging.info(f"下载失败: {failed_count}")
    
    if failed_factors:
        logging.info(f"失败的因子: {failed_factors[:10]}...")  # 只显示前10个
    
    return True

def create_factor_summary(output_dir='factors'):
    """创建因子数据汇总"""
    if not os.path.exists(output_dir):
        logging.warning(f"输出目录 {output_dir} 不存在")
        return
    
    factor_files = [f for f in os.listdir(output_dir) if f.endswith('.pkl')]
    
    summary_data = []
    for factor_file in factor_files:
        factor_name = factor_file.replace('.pkl', '')
        filepath = os.path.join(output_dir, factor_file)
        
        try:
            factor_data = pd.read_pickle(filepath)
            summary_data.append({
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
    summary_df.to_csv(os.path.join(output_dir, 'factor_summary.csv'), index=False)
    logging.info(f"因子汇总已保存到 {output_dir}/factor_summary.csv")
    
    return summary_df

if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])
    # 测试模式：只下载前5个因子
    # print("开始下载因子数据...")
    # print("注意：这是测试模式，只下载前5个因子")
    
    # success = download_all_factors(
    #     start_date='2010-01-01',
    #     end_date='2025-09-30', # 修改为当前日期
    #     batch_size=400,  # 减小批次大小
    #     output_dir='factors'
    # )
    
    # if success:
    #     print("创建因子汇总...")
    #     summary = create_factor_summary('factors')
    #     if summary is not None:
    #         print(f"因子汇总:\n{summary}")
    # else:
    #     print("下载失败!")
    init_rqdatac()
    factor_names = get_factor_names()
    print(factor_names)
