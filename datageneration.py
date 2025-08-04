#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datageneration.py - Generate images in original paper format

Original img_data/ structure:
- monthly_20d/20d_month_has_vb_[20]_ma_YYYY_images.dat (binary, uint8)
- monthly_20d/20d_month_has_vb_[20]_ma_YYYY_labels_w_delay.feather
- label_columns.txt (metadata)

Usage:
    python datageneration.py --image_days 20 --mode train
"""

from __init__ import *
import dataset as _D
import argparse
import os
import struct
from tqdm import tqdm
from joblib import Parallel, delayed
try:
    import pyarrow.feather as feather
except ImportError:
    import pandas as pd
    # Use pandas built-in feather
    feather = type('feather', (), {
        'write_feather': pd.DataFrame.to_feather,
        'read_feather': pd.read_feather
    })()

def create_original_format_images(win_size, mode, sample_rate=1.0, data_version='original'):
    """
    Generate images in original paper format
    
    Args:
        win_size (int): Window size (5, 20, 60)  
        mode (str): 'train' or 'test'
        sample_rate (float): Sampling rate
        data_version (str): Data version ('original' or 'filled')
    """
    
    print(f"Starting original format image generation")
    print(f"  Window size: {win_size} days")
    print(f"  Mode: {mode}")
    print(f"  Data version: {data_version}")
    print(f"  Sample rate: {sample_rate}")
    
    # Output directory setup (same naming convention as original)
    if win_size == 5:
        dir_name = "weekly_5d"
        filename_prefix = "5d_week_has_vb_[5]_ma"
    elif win_size == 20:
        dir_name = "monthly_20d"
        filename_prefix = "20d_month_has_vb_[20]_ma"
    else:  # 60
        dir_name = "quarterly_60d" 
        filename_prefix = "60d_quarter_has_vb_[60]_ma"
    
    # Add data version to output directory
    base_dir = "img_data_reconstructed" if data_version == 'original' else "img_data_reconstructed_filled"
    output_dir = f"{base_dir}/{dir_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset creation
    print(f"\nCreating dataset...")
    dataset = _D.ImageDataSet(
        win_size=win_size,
        mode=mode,
        label=f'RET{win_size}',  # ImageDataSet에서 지원하는 형식
        parallel_num=4,
        data_version=data_version
    )
    
    # Generate and save images by year
    years = range(1993, 2001) if mode == 'train' else range(2001, 2020)
    
    for year in years:
        print(f"\nProcessing {year} data...")
        
        # Filter data for current year
        year_start = int(f"{year}0101")
        year_end = int(f"{year}1231")
        year_df = dataset.df[
            (dataset.df['date'] >= year_start) & 
            (dataset.df['date'] <= year_end)
        ].copy()
        
        if len(year_df) == 0:
            print(f"  No data for {year}, skipping")
            continue
        
        print(f"  {year} records: {len(year_df):,}")
        
        # Generate images for current year
        year_images = []
        year_labels = []
        
        # Parallel processing for speed improvement
        symbol_groups = list(year_df.groupby('code'))
        print(f"  Parallel processing ({len(symbol_groups)} symbols)...")
        
        # Use image generation function
        symbol_results = Parallel(n_jobs=4)(
            delayed(_D.single_symbol_image)(
                group_df, 
                image_size=dataset.image_size,
                start_date=year_start,
                sample_rate=sample_rate,
                mode=mode
            ) for code, group_df in tqdm(symbol_groups, desc=f"{year} image generation")
        )
        
        print(f"  Generating labels...")
        for symbol_data in symbol_results:
            for entry in symbol_data:
                if len(entry) == 11:  # [image, label_5, label_20, label_60, ret5, ret20, ret60, date, code, market_cap, ewma_vol]
                    year_images.append(entry[0].astype(np.uint8))
                    
                    # Generate labels with actual data (direct use without search)
                    entry_date = entry[7]      # Date
                    entry_code = entry[8]      # Symbol code
                    market_cap = entry[9]      # Market cap
                    ewma_vol = entry[10]       # EWMA volatility
                    
                    year_labels.append({
                        'Date': pd.to_datetime(str(entry_date), format='%Y%m%d'),  # Actual date
                        'StockID': str(entry_code),  # Actual symbol code (PERMNO)
                        'MarketCap': market_cap / 1000,  # Convert to thousands of dollars
                        'Ret_5d': entry[4],   # actual_ret5 (decimal format)
                        'Ret_20d': entry[5],  # actual_ret20
                        'Ret_60d': entry[6],  # actual_ret60
                        'Ret_month': entry[5],  # Use 20-day for monthly returns
                        'EWMA_vol': ewma_vol  # Actual EWMA volatility
                    })
        
        if len(year_images) == 0:
            print(f"  No images generated for {year}")
            continue
        
        print(f"  Generated images: {len(year_images):,}")
        
        # 1. Save images as .dat file (binary format)
        images_filename = f"{filename_prefix}_{year}_images.dat"
        images_path = os.path.join(output_dir, images_filename)
        
        print(f"  Saving images: {images_filename}")
        images_array = np.array(year_images, dtype=np.uint8)
        
        # Save as binary (same as original)
        with open(images_path, 'wb') as f:
            f.write(images_array.tobytes())
        
        # 2. Save labels as .feather file
        labels_filename = f"{filename_prefix}_{year}_labels_w_delay.feather"
        labels_path = os.path.join(output_dir, labels_filename)
        
        print(f"  Saving labels: {labels_filename}")
        labels_df = pd.DataFrame(year_labels)
        
        # Set data types same as original
        labels_df['Date'] = pd.to_datetime(labels_df['Date'])
        labels_df['StockID'] = labels_df['StockID'].astype(str)
        labels_df['MarketCap'] = labels_df['MarketCap'].astype(np.float32)
        labels_df['Ret_5d'] = labels_df['Ret_5d'].astype(np.float64)
        labels_df['Ret_20d'] = labels_df['Ret_20d'].astype(np.float64) 
        labels_df['Ret_60d'] = labels_df['Ret_60d'].astype(np.float64)
        labels_df['Ret_month'] = labels_df['Ret_month'].astype(np.float64)
        labels_df['EWMA_vol'] = labels_df['EWMA_vol'].astype(np.float64)
        
        # Save in Feather format
        feather.write_feather(labels_df, labels_path)
        
        # Check file sizes
        img_size_mb = os.path.getsize(images_path) / (1024*1024)
        label_size_mb = os.path.getsize(labels_path) / (1024*1024)
        print(f"  File sizes: images {img_size_mb:.1f}MB, labels {label_size_mb:.1f}MB")
        
        # Memory cleanup
        del year_images, year_labels, images_array, labels_df
        import gc
        gc.collect()
    
    # 3. Generate label_columns.txt (same as original)
    label_columns_path = os.path.join("img_data_reconstructed", "label_columns.txt")
    with open(label_columns_path, 'w') as f:
        f.write("'Date': The last day of the {}-day rolling window for the chart.\n".format(win_size))
        f.write("'StockID': CRSP PERMNO that identifies the stock.\n")
        f.write("'MarketCap': Market capitalization in dollar, recorded in thousands.\n")
        f.write("'Ret_{t}d': t=5,20,60, next t-day holding period return.\n")
        f.write("'Ret_month': Holding period return for the next month, from the current monthend to the next monthend.\n")
        f.write("'EWMA_vol': Exponentially weighted volatility (square of daily returns) with alpha as 0.05. One day delay is included.\n")
    
    print(f"\nOriginal format image generation completed!")
    print(f"  Save path: {output_dir}")
    print(f"  Metadata: img_data_reconstructed/label_columns.txt")


def verify_original_format(output_dir, year=1993):
    """
    Verify that generated data has same format as original
    """
    print(f"\nVerifying original format...")
    
    # Check file existence
    images_file = f"20d_month_has_vb_[20]_ma_{year}_images.dat"
    labels_file = f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather"
    
    images_path = os.path.join(output_dir, images_file)
    labels_path = os.path.join(output_dir, labels_file)
    
    if not os.path.exists(images_path):
        print(f"Image file missing: {images_path}")
        return False
    
    if not os.path.exists(labels_path):
        print(f"Label file missing: {labels_path}")
        return False
    
    # Verify image file
    try:
        images = np.fromfile(images_path, dtype=np.uint8)
        num_images = len(images) // (64 * 60)
        images = images.reshape(num_images, 64, 60)
        print(f"Image file: {num_images:,} images ({images.shape})")
        print(f"   Pixel value range: {images.min()} ~ {images.max()}")
        print(f"   Binary verification: {set(np.unique(images)) <= {0, 255}}")
    except Exception as e:
        print(f"Failed to read image file: {e}")
        return False
    
    # Verify label file
    try:
        labels = feather.read_feather(labels_path)
        print(f"Label file: {len(labels):,} records")
        print(f"   Columns: {labels.columns.tolist()}")
        print(f"   Data types: {labels.dtypes.to_dict()}")
        
        # Compare columns with original
        expected_cols = ['Date', 'StockID', 'MarketCap', 'Ret_5d', 'Ret_20d', 'Ret_60d', 'Ret_month', 'EWMA_vol']
        missing_cols = set(expected_cols) - set(labels.columns)
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            return False
        else:
            print(f"All required columns exist")
        
    except Exception as e:
        print(f"Failed to read label file: {e}")
        return False
    
    print(f"Original format verification completed!")
    return True


def main():
    parser = argparse.ArgumentParser(description='Generate images in original paper format')
    parser.add_argument('--image_days', type=int, required=True,
                       choices=[5, 20, 60],
                       help='Image window size (days)')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'test'],
                       help='Dataset mode')
    parser.add_argument('--sample_rate', type=float, default=1.0,
                       help='Data sampling rate (default: 1.0)')
    parser.add_argument('--data_version', type=str, default='original',
                       choices=['original', 'filled'],
                       help='Data version: original (with missing values) or filled (missing values filled with previous close)')
    parser.add_argument('--verify', action='store_true',
                       help='Perform verification after generation')
    
    args = parser.parse_args()
    
    # Measure start time
    import time
    start_time = time.time()
    
    # Generate original format images
    create_original_format_images(
        win_size=args.image_days,
        mode=args.mode,
        sample_rate=args.sample_rate,
        data_version=args.data_version
    )
    
    # Perform verification
    if args.verify:
        if args.image_days == 20:  # Only 20-day verification implemented
            dir_name = "monthly_20d"
            output_dir = f"img_data_reconstructed/{dir_name}"
            verify_original_format(output_dir)
    
    # Completion time
    total_time = time.time() - start_time
    print(f"\nTotal time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")


if __name__ == '__main__':
    main()