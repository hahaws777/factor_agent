#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare factor analysis CSV data.
Converts pkl-format data to a CSV readable by downstream programs.
"""

import pickle
import pandas as pd
import sys


def prepare_factor_data(factor_file='factors_by_type/alpha101/WorldQuant_alpha014.pkl',
                        data_pkl='data.pkl', output_csv='factor_data.csv'):
    """
    Merge market data with a factor pkl and write a flat CSV.

    Parameters
    ----------
    factor_file : factor .pkl path
    data_pkl    : market data .pkl path
    output_csv  : output CSV path
    """
    print(f"Loading data from {data_pkl}...")
    with open(data_pkl, 'rb') as f:
        data = pickle.load(f)
    print(f"Data shape: {data.shape}")
    print(f"Columns: {list(data.columns)}")

    print(f"\nLoading factor from {factor_file}...")
    factor_data = pd.read_pickle(factor_file)
    print(f"Factor shape: {factor_data.shape}")
    print(f"Factor columns: {list(factor_data.columns)}")

    print("\nMerging data...")
    data = data.set_index(['order_book_id', 'date'])
    factor_name = factor_data.columns[0]
    data['factor_value'] = factor_data[factor_name]
    data = data.reset_index()

    output_data = pd.DataFrame()
    output_data['date'] = data['date'].astype(str)
    output_data['order_book_id'] = data['order_book_id'].astype(str)
    output_data['close'] = data['close']

    if 'factor_value' in data.columns:
        output_data['factor_value'] = data['factor_value']
    else:
        print("Warning: No factor_value found, using dummy values")
        output_data['factor_value'] = 0.0

    # Use != True comparisons — limit_up/down flags are object dtype with NaN,
    # and NaN.astype(bool) == True which would misclassify non-limit rows.
    output_data['is_st'] = (data['ST'] == True).astype(int) if 'ST' in data.columns else 0
    output_data['is_limit_up'] = (data['limit_up_flag'] == True).astype(int) if 'limit_up_flag' in data.columns else 0
    output_data['is_limit_down'] = (data['limit_down_flag'] == True).astype(int) if 'limit_down_flag' in data.columns else 0
    output_data['is_suspended'] = (data['suspended'] == True).astype(int) if 'suspended' in data.columns else 0

    output_data = output_data.dropna(subset=['close', 'factor_value'])

    print(f"\nOutput data shape: {output_data.shape}")
    print(f"Date range: {output_data['date'].min()} to {output_data['date'].max()}")

    print(f"\nSaving to {output_csv}...")
    output_data.to_csv(output_csv, index=False)
    print(f"Done — {len(output_data):,} records")

    return output_data


if __name__ == "__main__":
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])
    if len(sys.argv) > 1:
        factor_file = sys.argv[1]
        prepare_factor_data(factor_file=factor_file)
    else:
        print("Usage: python prepare_data.py <factor_file>")
        print("Example: python prepare_data.py factors_by_type/alpha101/WorldQuant_alpha014.pkl")
        print("\nUsing default factor...")
        prepare_factor_data()
