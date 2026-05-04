#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Market data overview analysis — summarize structure and distributions of
the three main .pkl data files (allA_close, close, data).
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def load_data():
    """Load all data files."""
    print("Loading data...")
    with open('allA_close.pkl', 'rb') as f:
        allA_close = pickle.load(f)
    with open('close.pkl', 'rb') as f:
        close = pickle.load(f)
    with open('data.pkl', 'rb') as f:
        data = pickle.load(f)
    print("Data loaded.")
    return allA_close, close, data


def analyze_data_structure():
    """Print shape, memory, stock count, and date range for each file."""
    allA_close, close, data = load_data()

    print("\n" + "=" * 60)
    print("Data Overview")
    print("=" * 60)

    files_info = [
        ("allA_close.pkl", allA_close),
        ("close.pkl", close),
        ("data.pkl", data),
    ]

    for filename, df in files_info:
        print(f"\n{filename}:")
        print(f"  Shape:  {df.shape}")
        print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        if isinstance(df.index, pd.MultiIndex):
            print(f"  Stocks: {df.index.get_level_values(0).nunique()}")
            print(f"  Days:   {df.index.get_level_values(1).nunique()}")
            print(f"  Range:  {df.index.get_level_values(1).min()} → {df.index.get_level_values(1).max()}")
        else:
            if 'order_book_id' in df.columns:
                print(f"  Stocks: {df['order_book_id'].nunique()}")
            if 'date' in df.columns:
                print(f"  Days:   {df['date'].nunique()}")
                print(f"  Range:  {df['date'].min()} → {df['date'].max()}")


def create_visualizations():
    """Create 4-panel summary chart."""
    allA_close, close, data = load_data()

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Market Data Overview', fontsize=16, fontweight='bold')

    # 1. Record count comparison
    ax1 = axes[0, 0]
    files = ['allA_close.pkl', 'close.pkl', 'data.pkl']
    sizes = [13024795, 11948722, 20734623]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = ax1.bar(files, sizes, color=colors)
    ax1.set_title('Data Volume Comparison', fontweight='bold')
    ax1.set_ylabel('Number of Records')
    ax1.tick_params(axis='x', rotation=45)
    for bar, size in zip(bars, sizes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(sizes) * 0.01,
                 f'{size:,}', ha='center', va='bottom', fontweight='bold')

    # 2. Stock count comparison
    ax2 = axes[0, 1]
    stock_counts = [5397, 5389, 5481]
    bars2 = ax2.bar(files, stock_counts, color=colors)
    ax2.set_title('Number of Stocks Comparison', fontweight='bold')
    ax2.set_ylabel('Number of Stocks')
    ax2.tick_params(axis='x', rotation=45)
    for bar, count in zip(bars2, stock_counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(stock_counts) * 0.01,
                 f'{count:,}', ha='center', va='bottom', fontweight='bold')

    # 3. Time span
    ax3 = axes[1, 0]
    start_date = datetime(2010, 1, 4)
    end_date = datetime(2025, 7, 31)
    ax3.barh(['Time Range'], [(end_date - start_date).days], color='#96CEB4')
    ax3.set_title('Data Time Span', fontweight='bold')
    ax3.set_xlabel('Days')
    ax3.text((end_date - start_date).days / 2, 0,
             f'{start_date.strftime("%Y-%m-%d")} to {end_date.strftime("%Y-%m-%d")}',
             ha='center', va='center', fontweight='bold')

    # 4. Field count comparison
    ax4 = axes[1, 1]
    field_counts = [1, 1, 17]
    bars4 = ax4.bar(files, field_counts, color=colors)
    ax4.set_title('Number of Data Fields', fontweight='bold')
    ax4.set_ylabel('Number of Fields')
    ax4.tick_params(axis='x', rotation=45)
    for bar, count in zip(bars4, field_counts):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(field_counts) * 0.01,
                 f'{count}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('data_overview.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_price_data():
    """Print close-price statistics and save distribution plots."""
    allA_close, close, data = load_data()

    print("\n" + "=" * 60)
    print("Price Data Analysis")
    print("=" * 60)

    close_prices = data['close'].dropna()

    print(f"\nClose price stats:")
    print(f"  Valid records:  {len(close_prices):,}")
    print(f"  Range:          {close_prices.min():.2f} – {close_prices.max():.2f}")
    print(f"  Mean:           {close_prices.mean():.2f}")
    print(f"  Median:         {close_prices.median():.2f}")
    print(f"  Std:            {close_prices.std():.2f}")

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.hist(close_prices, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
    plt.title('Close Price Distribution', fontweight='bold')
    plt.xlabel('Close Price')
    plt.ylabel('Frequency')
    plt.yscale('log')

    plt.subplot(2, 2, 2)
    plt.hist(np.log10(close_prices), bins=100, alpha=0.7, color='lightcoral', edgecolor='black')
    plt.title('Log10 Close Price Distribution', fontweight='bold')
    plt.xlabel('Log10(Close Price)')
    plt.ylabel('Frequency')

    plt.subplot(2, 2, 3)
    data['year'] = data['date'].dt.year
    yearly_avg = data.groupby('year')['close'].mean()
    plt.plot(yearly_avg.index, yearly_avg.values, marker='o', linewidth=2, markersize=6)
    plt.title('Average Close Price by Year', fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Average Close Price')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    yearly_stocks = data.groupby('year')['order_book_id'].nunique()
    plt.plot(yearly_stocks.index, yearly_stocks.values, marker='s', linewidth=2, markersize=6, color='green')
    plt.title('Number of Stocks by Year', fontweight='bold')
    plt.xlabel('Year')
    plt.ylabel('Number of Stocks')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('price_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    import os
    from pathlib import Path
    os.chdir(Path(__file__).resolve().parents[2])
    print("Market Data Analysis")
    print("=" * 60)
    analyze_data_structure()
    create_visualizations()
    analyze_price_data()
    print("\nDone. Charts saved as data_overview.png and price_analysis.png")


if __name__ == "__main__":
    main()
