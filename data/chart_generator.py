#!/usr/bin/env python3
"""
chart_generator.py - Main script for generating stock chart images

Based on trend_submit/Data/generate_chart.py
Uses chart_library.py for drawing and processes data from datageneration.ipynb
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from chart_library import DrawOHLC


def adjust_price(df):
    """
    Normalize stock prices using trend_submit's adjust_price method.
    First day close = 1.0, subsequent prices based on returns.
    
    Args:
        df (pd.DataFrame): Stock price data with OHLC columns
    
    Returns:
        pd.DataFrame: Price-adjusted data with first close = 1.0
    """
    if len(df) == 0:
        return None  # Empty DataFrame
    
    df = df.reset_index(drop=True)
    
    # Get first day closing price as normalization base
    fd_close = abs(df.at[0, "close"])
    if df.at[0, "close"] == 0.0 or np.isnan(df.at[0, "close"]):
        return None  # trend_submit처럼 에러코드 대신 None 반환

    pre_close = fd_close
    res_df = df.copy()
    
    # Normalize first day prices to base of 1.0
    res_df.at[0, "close"] = 1.0
    res_df.at[0, "open"] = abs(res_df.at[0, "open"]) / pre_close
    res_df.at[0, "high"] = abs(res_df.at[0, "high"]) / pre_close
    res_df.at[0, "low"] = abs(res_df.at[0, "low"]) / pre_close

    # Reconstruct prices using returns while maintaining intraday patterns
    pre_close = 1  # Previous close starts at normalized base
    for i in range(1, len(res_df)):
        # Get original price levels for this day
        today_closep = abs(res_df.at[i, "close"])
        today_openp = abs(res_df.at[i, "open"])
        today_highp = abs(res_df.at[i, "high"])
        today_lowp = abs(res_df.at[i, "low"])
        today_ret = np.float64(res_df.at[i, "ret"])  # Daily return

        # Calculate new close using return from previous close
        res_df.at[i, "close"] = (1 + today_ret) * pre_close
        
        # Scale other prices proportionally to maintain intraday patterns
        if today_closep != 0:
            res_df.at[i, "open"] = res_df.at[i, "close"] / today_closep * today_openp
            res_df.at[i, "high"] = res_df.at[i, "close"] / today_closep * today_highp
            res_df.at[i, "low"] = res_df.at[i, "close"] / today_closep * today_lowp

        # Update previous close for next iteration
        if not np.isnan(res_df.at[i, "close"]):
            pre_close = res_df.at[i, "close"]

    return res_df


def process_unified_data(df, image_size, mode):
    """
    Process unified data to generate chart images using calendar-based sampling.
    
    Args:
        df (pd.DataFrame): Unified stock data (all stocks)
        image_size (tuple): (height, width) of output images
        mode (str): 'train' or 'test'
    
    Returns:
        tuple: (images, labels, dates, stock_ids)
    """
    images = []
    labels = []
    dates = []
    stock_ids = []
    
    height, width = image_size
    lookback = width // 3
    
    # Date filtering based on mode
    if mode == 'train':
        df = df[(df['date'] >= 19930101) & (df['date'] <= 20001231)]
    elif mode == 'test':
        df = df[(df['date'] >= 20010101) & (df['date'] <= 20191231)]
    
    # Get all unique dates and sort them (통합된 거래일)
    all_dates = sorted(df['date'].unique())
    print(f"Total trading days: {len(all_dates)}")
    
    # Pre-group by stock for efficiency
    print("Grouping data by stock...")
    stock_groups = df.groupby('code')
    print(f"Processing {len(stock_groups)} stocks")
    
    # Calculate window indices
    window_indices = list(range(lookback-1, len(all_dates), lookback))
    print(f"Will process {len(window_indices)} time windows")
    
    # Process each stock first, then windows within each stock
    for stock_code, stock_df in tqdm(stock_groups, desc="Processing stocks"):
        stock_df = stock_df.sort_values('date').reset_index(drop=True)
        stock_dates = set(stock_df['date'].values)
        
        # Process each time window for this stock
        for start_idx in window_indices:
            # Get the window of dates
            window_dates = all_dates[start_idx-(lookback-1):start_idx+1]
            if len(window_dates) < lookback:
                continue
                
            # Check if this stock has data for all dates in the window
            if not all(date in stock_dates for date in window_dates):
                continue  # Skip if missing any dates in window
            
            # Get stock data for this window
            stock_window_data = stock_df[stock_df['date'].isin(window_dates)]
            if len(stock_window_data) < lookback:
                continue  # Skip stocks without complete data
                
            # Sort by date to ensure proper order
            stock_window_data = stock_window_data.sort_values('date').reset_index(drop=True)
            
            # Extract OHLC data for this window
            window_data = stock_window_data[['open', 'high', 'low', 'close', 'ret']].reset_index(drop=True)
            
            try:
                # Apply price adjustment (trend_submit methodology)
                adjusted_data = adjust_price(window_data)
                if adjusted_data is None:
                    continue  # 첫날 종가 문제로 스킵
                
                # Add moving average
                adjusted_data['ma'] = adjusted_data['close'].rolling(window=lookback, min_periods=1).mean()
                
                # Re-normalize to chart starting point (trend_submit line 242-244)
                start_close_price = adjusted_data['close'].iloc[0]
                if start_close_price <= 0 or np.isnan(start_close_price):
                    continue  # 시작 가격 문제로 스킵
                adjusted_data = adjusted_data / start_close_price
                
                # Generate chart using DrawOHLC
                chart_drawer = DrawOHLC(
                    df=adjusted_data,
                    image_width=width,
                    image_height=height,
                    has_volume_bar=False,  # Can be enabled later
                    ma_lags=None  # Can add MA later: [lookback]
                )
                
                # Draw the image
                image = chart_drawer.draw_image()
                
                if image is not None:
                    # Convert PIL image to numpy array
                    image_array = np.array(image)
                    
                    # Get labels from the target date row
                    target_row = stock_window_data.iloc[-1]  # Last row = target date
                    
                    # Extract labels and returns - trend_submit 방식: NaN은 특별값으로 처리
                    # 라벨: 1 if > 0 else 0 if <= 0 else 2 (NaN인 경우)
                    ret_5 = target_row['ret5'] if not np.isnan(target_row['ret5']) else float('nan')
                    ret_20 = target_row['ret20'] if not np.isnan(target_row['ret20']) else float('nan') 
                    ret_60 = target_row['ret60'] if not np.isnan(target_row['ret60']) else float('nan')
                    
                    # trend_submit 라벨 생성 방식
                    label_5 = 1 if ret_5 > 0 else 0 if ret_5 <= 0 else 2
                    label_20 = 1 if ret_20 > 0 else 0 if ret_20 <= 0 else 2  
                    label_60 = 1 if ret_60 > 0 else 0 if ret_60 <= 0 else 2
                    
                    # 수익률 스케일링 제거 (trend_submit은 원본 그대로 사용)
                    # ret_5, ret_20, ret_60 = 원본 값 그대로 사용
                    
                    # trend_submit은 결측치 있어도 저장함 (우리도 동일하게)
                    
                    # Store results
                    images.append(image_array)
                    labels.append({
                        'Ret_5d': ret_5,
                        'Ret_20d': ret_20,
                        'Ret_60d': ret_60,
                        'StockID': target_row['code'],
                        'Date': target_row['date'],
                        'MarketCap': target_row['mktcap'] if 'mktcap' in target_row else 0.0
                    })
                    dates.append(target_row['date'])
                    stock_ids.append(target_row['code'])
                    
            except Exception as e:
                target_date = window_dates[-1] if window_dates else "unknown"
                print(f"Error processing {stock_code} on {target_date}: {e}")
                continue
    
    return images, labels, dates, stock_ids


def save_images_and_labels(images, labels, year, output_dir, prefix, image_size):
    """
    Save images and labels in original format (.dat + .feather).
    
    Args:
        images (list): List of image arrays
        labels (list): List of label dictionaries
        year (int): Year being processed
        output_dir (str): Output directory path
        prefix (str): File prefix
        image_size (tuple): (height, width) of images
    """
    height, width = image_size
    
    # Save images (.dat format - binary)
    images_file = f"{prefix}_{year}_images.dat"
    images_path = os.path.join(output_dir, images_file)
    
    with open(images_path, 'wb') as f:
        for img in images:
            # Ensure correct shape and data type
            if img.shape != (height, width):
                print(f"Warning: Image shape {img.shape} doesn't match expected {(height, width)}")
                continue
            # Convert to uint8 and write as bytes
            img_bytes = img.astype(np.uint8).tobytes()
            f.write(img_bytes)
    
    # Save labels (.feather format)
    labels_file = f"{prefix}_{year}_labels_w_delay.feather"
    labels_path = os.path.join(output_dir, labels_file)
    
    labels_df = pd.DataFrame(labels)
    labels_df.to_feather(labels_path)
    
    print(f"Saved {len(images)} images and labels for {year}")
    print(f"  Images: {images_path}")
    print(f"  Labels: {labels_path}")


def main():
    """Main function for chart generation."""
    parser = argparse.ArgumentParser(description="Generate stock chart images using trend_submit methodology")
    parser.add_argument("--image_days", type=int, choices=[5, 20, 60], required=True,
                        help="Number of days in each image")
    parser.add_argument("--mode", choices=['train', 'test'], required=True,
                        help="Generation mode (train: 1993-2000, test: 2001-2019)")
    
    args = parser.parse_args()
    
    # Load unified data from datageneration.ipynb
    data_file = "data_1992_2019_unified.parquet"
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found")
        print("Please run datageneration.ipynb first to create the unified dataset")
        return
    
    print(f"Loading data from {data_file}...")
    df = pd.read_parquet(data_file)
    
    # Convert pandas nullable types to regular numpy types like trend_submit
    print("Converting data types...")
    for col in ['label_5', 'label_20', 'label_60', 'ret5', 'ret20', 'ret60', 'open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('float64')
    
    # !!!!!!! 중요: 결측치 제거 하지 말고 trend_submit처럼 보존 !!!!!!!
    # df = df.dropna() 같은 것들 모두 제거
    
    # Convert date format to YYYYMMDD integer
    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y%m%d').astype(int)
    
    # Set image size and output paths based on days
    if args.image_days == 5:
        image_size = (32, 15)
        output_dir = "img_data_generated/weekly_5d"
        prefix = "5d_week_has_vb_[5]_ma"
    elif args.image_days == 20:
        image_size = (64, 60)
        output_dir = "img_data_generated/monthly_20d"
        prefix = "20d_month_has_vb_[20]_ma"
    else:  # 60
        image_size = (96, 180)
        output_dir = "img_data_generated/quarterly_60d"
        prefix = "60d_quarter_has_vb_[60]_ma"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating {args.image_days}-day images for {args.mode} mode using unified calendar sampling...")
    print(f"Image size: {image_size}")
    
    # Process all data at once with unified calendar
    images, labels, dates, stock_ids = process_unified_data(df, image_size, args.mode)
    
    total_images = len(images)
    print(f"Generated {total_images} total images")
    
    # Group by year for saving (to match original format)
    if args.mode == 'train':
        years = range(1993, 2001)
    else:  # test
        years = range(2001, 2020)
    
    # Convert dates to pandas datetime for easier grouping
    dates_df = pd.DataFrame({
        'date': dates,
        'image_idx': range(len(dates)),
        'label': labels,
        'stock_id': stock_ids
    })
    dates_df['year'] = pd.to_datetime(dates_df['date'], format='%Y%m%d').dt.year
    
    # Save by year
    for year in years:
        # Check if files already exist
        images_file = f"{prefix}_{year}_images.dat"
        labels_file = f"{prefix}_{year}_labels_w_delay.feather"
        images_path = os.path.join(output_dir, images_file)
        labels_path = os.path.join(output_dir, labels_file)
        
        if os.path.exists(images_path) and os.path.exists(labels_path):
            print(f"Year {year} already exists, skipping...")
            continue
            
        # Get data for this year
        year_data = dates_df[dates_df['year'] == year]
        
        if len(year_data) == 0:
            print(f"No data for year {year}")
            continue
            
        print(f"\nSaving year {year}: {len(year_data)} images...")
        
        # Extract images and labels for this year
        year_images = [images[idx] for idx in year_data['image_idx']]
        year_labels = [labels[idx] for idx in year_data['image_idx']]
        
        # Save results for this year
        if year_images:
            save_images_and_labels(year_images, year_labels, year, output_dir, prefix, image_size)
        else:
            print(f"No images generated for year {year}")
    
    print(f"\n🎉 Chart generation completed!")
    print(f"Total images generated: {total_images}")
    print(f"Mode: {args.mode}, Window: {args.image_days} days")


if __name__ == "__main__":
    main()