#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch download A-share factor data by type via rqdatac.
Downloads smaller factor groups first (priority ordering).

Reads RICE_QUANT_API_TOKEN from environment or .env file.
"""

import rqdatac as rq
import pandas as pd
import os
import time
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('factor_download_by_type.log'),
        logging.StreamHandler(),
    ]
)


def _load_token() -> str:
    token = os.environ.get("RICE_QUANT_API_TOKEN", "")
    if not token:
        from pathlib import Path
        env_path = Path(__file__).resolve().parents[2] / ".env"
        if env_path.is_file():
            for line in env_path.read_text().splitlines():
                if line.startswith("RICE_QUANT_API_TOKEN="):
                    token = line.split("=", 1)[1].strip()
    return token


def init_rqdatac():
    """Initialise rqdatac from RICE_QUANT_API_TOKEN env var."""
    try:
        token = _load_token()
        if not token:
            raise ValueError("RICE_QUANT_API_TOKEN not set — add it to .env or environment")
        rq.init("license", token)
        logging.info("rqdatac initialised successfully")
        return True
    except Exception as e:
        logging.error(f"rqdatac init failed: {e}")
        return False


def get_factor_names_by_type(factor_type):
    """Return factor names for a given type."""
    try:
        factor_names = rq.get_all_factor_names(type=factor_type, market='cn')
        logging.info(f"Type '{factor_type}': found {len(factor_names)} factors")
        return factor_names
    except Exception as e:
        logging.error(f"get_all_factor_names({factor_type}) failed: {e}")
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


def download_factor_batch(stock_ids, factor_name, start_date, end_date):
    """Download a single factor for all stocks in one call."""
    try:
        logging.info(f"Downloading {factor_name} for {len(stock_ids)} stocks")
        factor_data = rq.get_factor(
            order_book_ids=stock_ids,
            factor=factor_name,
            start_date=start_date,
            end_date=end_date,
        )
        if factor_data is not None and not factor_data.empty:
            logging.info(f"Factor {factor_name}: {factor_data.shape}")
            return factor_data
        else:
            logging.warning(f"Factor {factor_name}: empty response")
            return None
    except Exception as e:
        logging.error(f"Download {factor_name} failed: {e}")
        return None


def save_factor_data(factor_data, factor_name, output_dir):
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


def download_factors_by_type(factor_type, start_date='2010-01-01', end_date=None,
                              output_dir='factors_by_type'):
    """Download all factors of one type, skipping already-saved files."""
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')

    logging.info(f"Downloading '{factor_type}' factors: {start_date} → {end_date}")

    factor_names = get_factor_names_by_type(factor_type)
    if not factor_names:
        logging.warning(f"No factors found for type '{factor_type}'")
        return False

    stock_ids = get_stock_list()
    if not stock_ids:
        return False

    type_output_dir = os.path.join(output_dir, factor_type)
    os.makedirs(type_output_dir, exist_ok=True)

    # For alpha101, start from alpha014 (skip already-downloaded earlier ones)
    if factor_type == 'alpha101':
        try:
            start_idx = next(i for i, n in enumerate(factor_names) if 'alpha014' in n.lower())
            factor_names = factor_names[start_idx:]
            logging.info(f"Starting from alpha014 — {len(factor_names)} remaining")
        except StopIteration:
            logging.warning("alpha014 not found — downloading from beginning")

    success_count = 0
    failed_count = 0
    failed_factors = []

    for i, factor_name in enumerate(factor_names):
        logging.info(f"Factor {i + 1}/{len(factor_names)}: {factor_name}")
        try:
            filename = os.path.join(type_output_dir, f'{factor_name}.pkl')
            if os.path.exists(filename):
                logging.info(f"Factor {factor_name} already exists — skipping")
                success_count += 1
                continue

            factor_data = download_factor_batch(stock_ids, factor_name, start_date, end_date)

            if factor_data is not None:
                if save_factor_data(factor_data, factor_name, type_output_dir):
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
            logging.info(f"Progress: {i + 1}/{len(factor_names)}  success={success_count} failed={failed_count}")
            time.sleep(1)

    logging.info("=" * 60)
    logging.info(f"Type '{factor_type}' complete — total={len(factor_names)} success={success_count} failed={failed_count}")
    if failed_factors:
        logging.info(f"Failed factors (first 10): {failed_factors[:10]}")

    return True


def create_factor_summary_by_type(output_dir='factors_by_type'):
    """Summarise all downloaded factor files by type to a CSV."""
    if not os.path.exists(output_dir):
        logging.warning(f"Output directory {output_dir} not found")
        return None

    summary_data = []
    for factor_type in os.listdir(output_dir):
        type_dir = os.path.join(output_dir, factor_type)
        if not os.path.isdir(type_dir):
            continue
        for factor_file in os.listdir(type_dir):
            if not factor_file.endswith('.pkl'):
                continue
            factor_name = factor_file.replace('.pkl', '')
            filepath = os.path.join(type_dir, factor_file)
            try:
                factor_data = pd.read_pickle(filepath)
                summary_data.append({
                    'factor_type': factor_type,
                    'factor_name': factor_name,
                    'rows': factor_data.shape[0],
                    'cols': factor_data.shape[1],
                    'columns': list(factor_data.columns),
                    'file_size_mb': os.path.getsize(filepath) / 1024 / 1024,
                    'has_data': not factor_data.empty,
                })
            except Exception as e:
                logging.error(f"Could not read {factor_file}: {e}")

    summary_df = pd.DataFrame(summary_data)
    out_csv = os.path.join(output_dir, 'factor_summary_by_type.csv')
    summary_df.to_csv(out_csv, index=False)
    logging.info(f"Summary saved to {out_csv}")
    return summary_df


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])

    factor_types = [
        ('eod_indicator', 41),
        ('financial_indicator', 126),
        ('alpha101', 101),
    ]

    print("Downloading factors by type...")
    print(f"Types to download: {len(factor_types)}")

    if not init_rqdatac():
        print("Initialisation failed — check RICE_QUANT_API_TOKEN in .env")
        exit(1)

    for i, (factor_type, expected_count) in enumerate(factor_types):
        print(f"\nType {i + 1}/{len(factor_types)}: {factor_type} (expected {expected_count} factors)")
        success = download_factors_by_type(
            factor_type=factor_type,
            start_date='2010-01-01',
            end_date='2025-09-30',
            output_dir='factors_by_type',
        )
        print(f"{factor_type}: {'done' if success else 'FAILED'}")
        if i < len(factor_types) - 1:
            print("Sleeping 10s before next type...")
            time.sleep(10)

    print("\nAll types downloaded.")
    print("Creating summary file...")
    summary = create_factor_summary_by_type('factors_by_type')
    if summary is not None:
        print(f"Summary done — {len(summary)} factors")
        type_stats = summary.groupby('factor_type').size().sort_values(ascending=False)
        for factor_type, count in type_stats.items():
            print(f"  {factor_type}: {count} factors")
    else:
        print("Summary failed.")
