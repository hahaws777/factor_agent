#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch download all A-share factor data via rqdatac.
Covers all stocks from 2010 to the present for every available factor.
"""

import rqdatac as rq
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
import pickle
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('factor_download.log'),
        logging.StreamHandler()
    ]
)


def init_rqdatac():
    """Initialise rqdatac connection."""
    try:
        rq.init("license", RICE_QUANT_API_TOKEN)
        logging.info("rqdatac initialised successfully")
        return True
    except Exception as e:
        logging.error(f"rqdatac init failed: {e}")
        return False


def get_factor_names():
    """Return list of all available factor names."""
    try:
        factor_names = rq.get_all_factor_names(type=None, market='cn')
        logging.info(f"Found {len(factor_names)} factors")
        return factor_names
    except Exception as e:
        logging.error(f"get_all_factor_names failed: {e}")
        return []


def get_stock_list():
    """Return list of all A-share order_book_ids."""
    try:
        stocks = rq.all_instruments(type='CS')
        stock_ids = stocks.order_book_id.tolist()
        logging.info(f"Found {len(stock_ids)} A-share stocks")
        return stock_ids
    except Exception as e:
        logging.error(f"all_instruments failed: {e}")
        return []


def download_factor_batch(stock_ids, factor_name, start_date, end_date, batch_size=100):
    """Download one factor for all stocks in batches."""
    all_data = []
    total_stocks = len(stock_ids)

    for i in range(0, total_stocks, batch_size):
        batch_stocks = stock_ids[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_stocks + batch_size - 1) // batch_size

        try:
            logging.info(f"Factor {factor_name} — batch {batch_num}/{total_batches} ({len(batch_stocks)} stocks)")
            factor_data = rq.get_factor(
                order_book_ids=batch_stocks,
                factor=factor_name,
                start_date=start_date,
                end_date=end_date,
            )
            if factor_data is not None and not factor_data.empty:
                all_data.append(factor_data)
                logging.info(f"Batch {batch_num}: {factor_data.shape[0]} rows")
            else:
                logging.warning(f"Batch {batch_num}: empty response")
        except Exception as e:
            logging.error(f"Batch {batch_num} failed: {e}")
            continue

        time.sleep(0.1)

    if all_data:
        combined = pd.concat(all_data, ignore_index=False)
        logging.info(f"Factor {factor_name} complete: {combined.shape}")
        return combined
    else:
        logging.warning(f"Factor {factor_name}: no data downloaded")
        return None


def save_factor_data(factor_data, factor_name, output_dir='factors'):
    """Save factor DataFrame as pickle."""
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f'{factor_name}.pkl')
    try:
        factor_data.to_pickle(filename)
        logging.info(f"Saved {factor_name} → {filename}")
        return True
    except Exception as e:
        logging.error(f"Save {factor_name} failed: {e}")
        return False


def download_all_factors(start_date='2010-01-01', end_date=None,
                         max_factors=None, batch_size=100,
                         output_dir='factors'):
    """Download all factors to output_dir, skipping already-existing files."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    logging.info(f"Downloading factors: {start_date} → {end_date}")

    if not init_rqdatac():
        return False

    factor_names = get_factor_names()
    if not factor_names:
        return False

    if max_factors:
        factor_names = factor_names[:max_factors]
        logging.info(f"Limiting to first {max_factors} factors")

    stock_ids = get_stock_list()
    if not stock_ids:
        return False

    os.makedirs(output_dir, exist_ok=True)
    success_count = 0
    failed_count = 0
    failed_factors = []

    for i, factor_name in enumerate(factor_names):
        logging.info(f"Factor {i + 1}/{len(factor_names)}: {factor_name}")

        try:
            filename = os.path.join(output_dir, f'{factor_name}.pkl')
            if os.path.exists(filename):
                logging.info(f"Factor {factor_name} already exists — skipping")
                success_count += 1
                continue

            factor_data = download_factor_batch(stock_ids, factor_name, start_date, end_date, batch_size)

            if factor_data is not None:
                if save_factor_data(factor_data, factor_name, output_dir):
                    success_count += 1
                else:
                    failed_count += 1
                    failed_factors.append(factor_name)
            else:
                failed_count += 1
                failed_factors.append(factor_name)

        except Exception as e:
            logging.error(f"Error downloading {factor_name}: {e}")
            failed_count += 1
            failed_factors.append(factor_name)

        if (i + 1) % 10 == 0:
            logging.info(f"Progress: {i + 1} factors processed — success={success_count} failed={failed_count}")
            time.sleep(1)

    logging.info("=" * 60)
    logging.info("Download complete")
    logging.info(f"Total: {len(factor_names)}  Success: {success_count}  Failed: {failed_count}")
    if failed_factors:
        logging.info(f"Failed factors (first 10): {failed_factors[:10]}")

    return True


def create_factor_summary(output_dir='factors'):
    """Summarise downloaded factor files to a CSV."""
    if not os.path.exists(output_dir):
        logging.warning(f"Output directory {output_dir} not found")
        return None

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
                'has_data': not factor_data.empty,
            })
        except Exception as e:
            logging.error(f"Could not read {factor_file}: {e}")

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(output_dir, 'factor_summary.csv'), index=False)
    logging.info(f"Summary saved to {output_dir}/factor_summary.csv")
    return summary_df


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])
    init_rqdatac()
    factor_names = get_factor_names()
    print(factor_names)
