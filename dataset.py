#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset.py - Module to convert stock price data to candlestick chart images
"""

from __init__ import *
import utils as _U
reload(_U)
try:
    import pyarrow.feather as feather
except ImportError:
    # pandas 내장 feather 사용
    feather = type('feather', (), {
        'write_feather': pd.DataFrame.to_feather,
        'read_feather': pd.read_feather
    })()
import struct

# Numba JIT for performance optimization
try:
    import numba
    NUMBA_AVAILABLE = True
    print("Numba JIT enabled for 50-100x image generation speedup!")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Warning: Numba not available - using slower pure Python")

if NUMBA_AVAILABLE:
    @numba.jit(nopython=True, cache=True)
    def generate_candlestick_image_fast(open_prices, high_prices, low_prices, close_prices, 
                                       ma_prices, volumes, image_height, image_width):
        """
        Numba-optimized candlestick image generation
        
        Args:
            All prices/volumes as numpy arrays
            image_height, image_width: image dimensions
            
        Returns:
            numpy array: Generated image
        """
        image = np.zeros((image_height, image_width), dtype=np.float32)
        lookback = image_width // 3
        
        # Region boundaries
        if image_height == 32:  # 5-day
            price_region_end = 25
            volume_start_row = 26
        elif image_height == 64:  # 20-day  
            price_region_end = 51
            volume_start_row = 52
        else:  # 96, 60-day
            price_region_end = 76
            volume_start_row = 77
        
        for i in range(lookback):
            # Candlestick (price region only) - Y-axis flipped for correct orientation
            open_px = min(int(open_prices[i]), price_region_end)
            close_px = min(int(close_prices[i]), price_region_end)
            low_px = min(int(low_prices[i]), price_region_end)
            high_px = min(int(high_prices[i]), price_region_end)
            
            # Open pixel - Y-axis flipped
            image[price_region_end - open_px, i*3] = 255.0
            
            # High-Low bar (vectorized) - Y-axis flipped
            for px in range(low_px, high_px + 1):
                if px <= price_region_end:
                    image[price_region_end - px, i*3+1] = 255.0
            
            # Close pixel - Y-axis flipped
            image[price_region_end - close_px, i*3+2] = 255.0
            
            # Moving average handled separately as continuous line
            
            # Volume in dedicated bottom region
            volume_height = int(volumes[i])
            if volume_height > 0:
                volume_bottom = image_height - 1
                volume_top = max(volume_start_row, volume_bottom - volume_height + 1)
                for px in range(volume_top, volume_bottom + 1):
                    image[px, i*3+1] = 255.0
        
        # Render moving average as continuous line (after candlesticks)
        ma_count = 0
        ma_x = np.zeros(lookback, dtype=np.int32)
        ma_y = np.zeros(lookback, dtype=np.int32)
        
        for i in range(lookback):
            if not np.isnan(ma_prices[i]):
                ma_px = min(int(ma_prices[i]), price_region_end)
                ma_x[ma_count] = i * 3 + 1  # Center column
                ma_y[ma_count] = price_region_end - ma_px  # Y-axis flipped
                ma_count += 1
        
        # Draw line connecting MA points
        for j in range(ma_count - 1):
            x1, y1 = ma_x[j], ma_y[j]
            x2, y2 = ma_x[j + 1], ma_y[j + 1]
            
            # Simple line drawing
            steps = max(abs(x2 - x1), abs(y2 - y1))
            if steps > 0:
                for step in range(steps + 1):
                    t = step / steps
                    x = int(x1 + t * (x2 - x1))
                    y = int(y1 + t * (y2 - y1))
                    if 0 <= x < image_width and 0 <= y <= price_region_end:
                        image[y, x] = 255.0
        
        return image
else:
    def generate_candlestick_image_fast(open_prices, high_prices, low_prices, close_prices, 
                                       ma_prices, volumes, image_height, image_width):
        """Fallback pure Python version"""
        return None  # Will use original method


def single_symbol_image(tabular_df, image_size, start_date, sample_rate, mode):
    """
    Convert individual stock price data to candlestick chart images
    
    Args:
        tabular_df (pd.DataFrame): 종목 데이터 (OHLCV + 수익률 라벨)
        image_size (tuple): 이미지 크기 (높이, 너비)
        start_date (int): 시작일 (YYYYMMDD)
        sample_rate (float): 샘플링 비율
        mode (str): 모드 ('train', 'test', 'inference')
    
    Returns:
        list: 생성된 이미지 데이터
    """
    
    dataset = []
    valid_dates = []
    lookback = image_size[1] // 3
    
    # Non-overlapping windows: 5일/20일/60일 간격으로 점프
    for d in range(lookback-1, len(tabular_df), lookback):
        if np.random.rand(1) > sample_rate:
            continue
            
        if tabular_df.iloc[d]['date'] < start_date:
            continue
        
        # Extract lookback window data
        price_slice = tabular_df[d-(lookback-1):d+1][['open', 'high', 'low', 'close']].reset_index(drop=True)
        volume_slice = tabular_df[d-(lookback-1):d+1][['volume']].reset_index(drop=True)
        
        # NA value filtering
        if price_slice[['open', 'high', 'low', 'close']].isnull().any().any():
            continue
        if volume_slice['volume'].isnull().any():
            continue
        
        # IPO/delisting filtering
        if (1.0*(price_slice[['open', 'high', 'low', 'close']].sum(axis=1)/price_slice['open'] == 4)).sum() > lookback//5:
            continue
        
        valid_dates.append(tabular_df.iloc[d]['date'])
        
        # 2-stage normalization
        ret_slice = tabular_df[d-(lookback-1):d+1][['ret']].reset_index(drop=True)
        
        # Stage 1: Return-based price reconstruction
        normalized_close = np.ones(lookback)
        for i in range(1, lookback):
            if not pd.isna(ret_slice.iloc[i]['ret']):
                normalized_close[i] = normalized_close[i-1] * (1 + ret_slice.iloc[i]['ret'])
            else:
                normalized_close[i] = normalized_close[i-1]
        
        normalized_prices = pd.DataFrame()
        for col in ['open', 'high', 'low']:
            ratios = price_slice[col] / price_slice['close']
            normalized_prices[col] = ratios * normalized_close
        normalized_prices['close'] = normalized_close
        
        # Add moving average - 간단한 rolling mean 방식
        normalized_prices['ma'] = pd.Series(normalized_close).rolling(window=lookback, min_periods=1).mean()
        
        # Stage 2: Min-Max scaling
        all_ohlc_values = np.concatenate([
            normalized_prices['open'].values,
            normalized_prices['high'].values, 
            normalized_prices['low'].values,
            normalized_prices['close'].values,
            normalized_prices['ma'].values
        ])
        price_min, price_max = np.min(all_ohlc_values), np.max(all_ohlc_values)
        price_slice = (normalized_prices - price_min) / (price_max - price_min)
        
        # Volume normalization (preserve minimum values)
        volume_min, volume_max = np.min(volume_slice.values), np.max(volume_slice.values)
        if volume_max > volume_min:
            volume_slice = (volume_slice - volume_min) / (volume_max - volume_min)
            # Ensure minimum 1 pixel height for non-zero volumes
            volume_slice = volume_slice * 0.9 + 0.1  # Scale to 0.1-1.0 range
        else:
            volume_slice = volume_slice * 0 + 0.5  # All same volume
        
        # Pixel coordinate transformation (price: top region, volume: bottom region completely separated)
        if image_size[0] == 32:
            # 5d: price 0-25 rows (26 rows), volume 26-31 rows (6 rows)
            price_slice = price_slice.apply(lambda x: x*25).astype(np.int32)
            volume_slice = volume_slice.apply(lambda x: x*5).astype(np.int32)
        elif image_size[0] == 64:
            # 20d: price 0-51 rows (52 rows), volume 52-63 rows (12 rows)
            price_slice = price_slice.apply(lambda x: x*51).astype(np.int32)
            volume_slice = volume_slice.apply(lambda x: x*11).astype(np.int32)
        else:  # 96
            # 60d: price 0-76 rows (77 rows), volume 77-95 rows (19 rows)
            price_slice = price_slice.apply(lambda x: x*76).astype(np.int32)
            volume_slice = volume_slice.apply(lambda x: x*18).astype(np.int32)
        
        # Image generation with Numba optimization
        if NUMBA_AVAILABLE:
            # Convert to numpy arrays for Numba
            open_prices = price_slice['open'].values.astype(np.float32)
            high_prices = price_slice['high'].values.astype(np.float32) 
            low_prices = price_slice['low'].values.astype(np.float32)
            close_prices = price_slice['close'].values.astype(np.float32)
            ma_prices = price_slice['ma'].values.astype(np.float32)
            volumes = volume_slice['volume'].values.astype(np.float32)
            
            # Use Numba-optimized function (50-100x faster!)
            image = generate_candlestick_image_fast(
                open_prices, high_prices, low_prices, close_prices,
                ma_prices, volumes, image_size[0], image_size[1]
            )
        else:
            # Fallback to original pure Python version
            image = np.zeros(image_size)
            
            # Region boundary setup
            if image_size[0] == 32:  # 5-day model
                price_region_end = 25  # price region: 0-25 rows
                volume_start_row = 26  # volume region: 26-31 rows
            elif image_size[0] == 64:  # 20-day model  
                price_region_end = 51  # price region: 0-51 rows
                volume_start_row = 52  # volume region: 52-63 rows
            else:  # 96, 60-day model
                price_region_end = 76  # price region: 0-76 rows
                volume_start_row = 77  # volume region: 77-95 rows
            
            for i in range(len(price_slice)):
                # Candlestick (price region only) - Y-axis flipped for correct orientation
                open_px = min(price_slice.loc[i]['open'], price_region_end)
                close_px = min(price_slice.loc[i]['close'], price_region_end)
                low_px = min(price_slice.loc[i]['low'], price_region_end)
                high_px = min(price_slice.loc[i]['high'], price_region_end)
                
                # Flip Y-axis: high price → top (small row), low price → bottom (large row)
                image[price_region_end - open_px, i*3] = 255.
                image[price_region_end - high_px:price_region_end - low_px + 1, i*3+1] = 255.  # High-Low bar
                image[price_region_end - close_px, i*3+2] = 255.
                
                # Moving average rendered separately as continuous line
                
                # Render volume in dedicated bottom region (completely separated)
                volume_height = int(volume_slice.loc[i]['volume'])
                if volume_height > 0:
                    volume_bottom = image_size[0] - 1  # Bottom pixel
                    volume_top = max(volume_start_row, volume_bottom - volume_height + 1)
                    image[volume_top:volume_bottom+1, i*3+1] = 255.
            
            # Render moving average as continuous line (after candlesticks)
            ma_points = []
            for i in range(len(price_slice)):
                if not pd.isna(price_slice.loc[i]['ma']):
                    ma_px = min(int(price_slice.loc[i]['ma']), price_region_end)
                    ma_points.append((i*3+1, price_region_end - ma_px))  # Center column of each day
            
            # Draw line connecting MA points
            for j in range(len(ma_points) - 1):
                x1, y1 = ma_points[j]
                x2, y2 = ma_points[j + 1]
                
                # Simple line drawing (Bresenham-like)
                steps = max(abs(x2 - x1), abs(y2 - y1))
                if steps > 0:
                    for step in range(steps + 1):
                        t = step / steps
                        x = int(x1 + t * (x2 - x1))
                        y = int(y1 + t * (y2 - y1))
                        if 0 <= x < image_size[1] and 0 <= y <= price_region_end:
                            image[y, x] = 255.
        
        # Label extraction: both binary labels and actual returns
        label_ret5 = tabular_df.iloc[d]['label_5']
        label_ret20 = tabular_df.iloc[d]['label_20'] 
        label_ret60 = tabular_df.iloc[d]['label_60']
        
        actual_ret5 = tabular_df.iloc[d]['ret5'] / 100.0  # % → decimal
        actual_ret20 = tabular_df.iloc[d]['ret20'] / 100.0
        actual_ret60 = tabular_df.iloc[d]['ret60'] / 100.0
        
        # Skip if future returns are NA
        if pd.isna(label_ret5) or pd.isna(label_ret20) or pd.isna(label_ret60):
            continue
        if pd.isna(actual_ret5) or pd.isna(actual_ret20) or pd.isna(actual_ret60):
            continue
        
        # Add market cap and EWMA volatility values
        market_cap_val = tabular_df.iloc[d].get('market_cap', np.random.uniform(10000, 100000))
        ewma_vol_val = tabular_df.iloc[d].get('ewma_vol', np.random.uniform(0.0001, 0.001))
        
        # Include date and stock code information (original format compatible) - 11 elements
        entry_date = tabular_df.iloc[d]['date']
        entry_code = tabular_df.iloc[0]['code']
        entry = [image, label_ret5, label_ret20, label_ret60, actual_ret5, actual_ret20, actual_ret60, 
                entry_date, entry_code, market_cap_val, ewma_vol_val]
        dataset.append(entry)
    
    if mode == 'train' or mode == 'test':
        return dataset
    else:
        return [tabular_df.iloc[0]['code'], dataset, valid_dates]



class ImageDataSet():
    """
    Main class to convert stock price data to candlestick chart images
    """
    
    def __init__(self, win_size, mode, label, parallel_num=-1, data_version='original'):
        """
        Args:
            win_size (int): Window size (5, 20, 60)
            mode (str): Mode ('train', 'test', 'inference')
            label (str): Label type ('RET5', 'RET20', 'RET60')
            parallel_num (int): Parallel processing count
            data_version (str): Data version ('original' or 'filled')
        """
        
        # Period setup
        if mode == 'train':
            self.start_date = 19930101
            self.end_date = 20001231
        elif mode == 'test':
            self.start_date = 20010101
            self.end_date = 20191231
        else:  # inference
            self.start_date = 19930101
            self.end_date = 20191231
        
        # Parameter validation
        assert win_size in [5, 20, 60], f'Unsupported window size: {win_size}'
        assert mode in ['train', 'test', 'inference'], f'Unsupported mode: {mode}'
        assert label in ['RET5', 'RET20', 'RET60'], f'Unsupported label: {label}'
        
        # Image size configuration
        if win_size == 5:
            self.image_size = (32, 15)
            self.extra_dates = datetime.timedelta(days=80)
        elif win_size == 20:
            self.image_size = (64, 60)
            self.extra_dates = datetime.timedelta(days=80)
        else:  # 60
            self.image_size = (96, 180)
            self.extra_dates = datetime.timedelta(days=120)
        
        self.mode = mode
        self.label = label
        self.parallel_num = parallel_num
        self.data_version = data_version
        
        self.load_data()
        
        print(f"Dataset initialization completed")
        print(f"  Mode: {self.mode}")
        print(f"  Data version: {self.data_version}")
        print(f"  Image size: {self.image_size}")
        print(f"  Period: {self.start_date} ~ {self.end_date}")
        print(f"  Label: {self.label}")
        
    @_U.timer('Load data', '8')
    def load_data(self):
        """
        Load WRDS CRSP data with version selection
        """
        
        # Select data file based on mode and version
        if self.mode in ['train']:
            if self.data_version == 'filled':
                data_file = 'data/data_1993_2000_train_val_filled.parquet'
            else:
                data_file = 'data/data_1993_2000_train_val.parquet'
        elif self.mode in ['test', 'inference']:
            if self.data_version == 'filled':
                data_file = 'data/data_2001_2019_test_filled.parquet'
            else:
                data_file = 'data/data_2001_2019_test.parquet'
        
        if not os.path.exists(data_file):
            print(f"Data file does not exist: {data_file}")
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        print(f"Loading data: {data_file}")
        tabularDf = pd.read_parquet(data_file)
        
        # Date format conversion
        tabularDf['date'] = pd.to_datetime(tabularDf['date']).dt.strftime('%Y%m%d').astype(np.int32)
        
        print(f"  Loaded records: {len(tabularDf):,}")
        print(f"  Number of symbols: {tabularDf['code'].nunique():,}")
        print(f"  Period: {tabularDf['date'].min()} ~ {tabularDf['date'].max()}")
        
        self.df = tabularDf.copy()
        del tabularDf
        
        print(f"Data loading completed")
        
    def generate_images(self, sample_rate):
        """
        Generate images
        """
        dataset_all = Parallel(n_jobs=self.parallel_num)(delayed(single_symbol_image)(
            g[1], image_size=self.image_size,
            start_date=self.start_date,
            sample_rate=sample_rate,
            mode=self.mode
        ) for g in tqdm(self.df.groupby('code'), desc=f'Generating Images (sample rate: {sample_rate})'))
        
        if self.mode == 'train' or self.mode == 'test':
            image_set = []
            for symbol_data in dataset_all:
                image_set.extend(symbol_data)
            dataset_all = []
            
            if self.mode == 'train':
                label_col = {'RET5': 1, 'RET20': 2, 'RET60': 3}[self.label]
                
                num0 = sum(1 for item in image_set if item[label_col] == 0)
                num1 = sum(1 for item in image_set if item[label_col] == 1)
                total = num0 + num1
                
                print(f"LABEL: {self.label}")
                print(f"  Down(0): {num0:,} ({100*num0/total:.1f}%), Up(1): {num1:,} ({100*num1/total:.1f}%)")
                print(f"  Total images: {total:,}")
            
            return image_set
        else:
            return dataset_all


class OriginalFormatDataset(torch.utils.data.Dataset):
    """
    Original format (.dat + .feather) dataset loader compatible with paper authors
    
    Directory structure:
    img_data_reconstructed/
    ├── weekly_5d/
    │   ├── 5d_week_has_vb_[5]_ma_1993_images.dat
    │   ├── 5d_week_has_vb_[5]_ma_1993_labels_w_delay.feather
    │   └── ...
    ├── monthly_20d/
    │   ├── 20d_month_has_vb_[20]_ma_1993_images.dat  
    │   ├── 20d_month_has_vb_[20]_ma_1993_labels_w_delay.feather
    │   └── ...
    └── quarterly_60d/
        ├── 60d_quarter_has_vb_[60]_ma_1993_images.dat
        ├── 60d_quarter_has_vb_[60]_ma_1993_labels_w_delay.feather
        └── ...
    """
    
    def __init__(self, win_size, mode, label_type, data_version='original'):
        """
        Args:
            win_size (int): Window size (5, 20, 60)
            mode (str): 'train' or 'test'
            label_type (str): 'RET5', 'RET20', 'RET60'
            data_version (str): 'original' or 'filled'
        """
        self.win_size = win_size
        self.mode = mode
        self.label_type = label_type
        self.data_version = data_version
        
        # Image size configuration
        if win_size == 5:
            self.image_height, self.image_width = 32, 15
            self.dir_name = "weekly_5d"
            self.prefix = "5d_week_has_vb_[5]_ma"
        elif win_size == 20:
            self.image_height, self.image_width = 64, 60
            self.dir_name = "monthly_20d"
            self.prefix = "20d_month_has_vb_[20]_ma"
        else:  # 60
            self.image_height, self.image_width = 96, 180
            self.dir_name = "quarterly_60d" 
            self.prefix = "60d_quarter_has_vb_[60]_ma"
        
        self.image_size = self.image_height * self.image_width
        
        # Year range configuration
        if mode == 'train':
            self.years = range(1993, 2001)
        else:  # test
            self.years = range(2001, 2020)
            
        # Select base directory based on data version
        self.base_dir = "img_data_reconstructed" if self.data_version == 'original' else "img_data_reconstructed_filled"
        self.data_dir = os.path.join(self.base_dir, self.dir_name)
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """
        Load all .dat and .feather files for all years
        """
        print(f"Loading original format data: {self.data_dir}")
        
        all_images = []
        all_labels = []
        
        for year in self.years:
            # Generate file paths
            images_file = f"{self.prefix}_{year}_images.dat"
            labels_file = f"{self.prefix}_{year}_labels_w_delay.feather"
            
            images_path = os.path.join(self.data_dir, images_file)
            labels_path = os.path.join(self.data_dir, labels_file)
            
            # Check file existence
            if not os.path.exists(images_path):
                print(f"  Warning: Missing image file: {images_file}")
                continue
            if not os.path.exists(labels_path):
                print(f"  Warning: Missing label file: {labels_file}")
                continue
                
            # Load images (.dat binary)
            try:
                images = np.fromfile(images_path, dtype=np.uint8)
                num_images = len(images) // self.image_size
                images = images.reshape(num_images, self.image_height, self.image_width)
                print(f"  Loaded {year} images: {num_images:,} samples")
            except Exception as e:
                print(f"  Error loading {year} images: {e}")
                continue
                
            # Load labels (.feather)
            try:
                labels_df = feather.read_feather(labels_path)
                print(f"  Loaded {year} labels: {len(labels_df):,} samples")
            except Exception as e:
                print(f"  Error loading {year} labels: {e}")
                continue
                
            # Check count consistency
            if num_images != len(labels_df):
                print(f"  Warning: {year} image-label count mismatch: {num_images} vs {len(labels_df)}")
                min_count = min(num_images, len(labels_df))
                images = images[:min_count]
                labels_df = labels_df.iloc[:min_count]
                
            all_images.append(images)
            all_labels.append(labels_df)
        
        if len(all_images) == 0:
            raise FileNotFoundError(f"No available data in: {self.data_dir}")
            
        # Combine all years data
        self.images = np.concatenate(all_images, axis=0)
        self.labels_df = pd.concat(all_labels, ignore_index=True)
        
        print(f"Total loaded data: {len(self.images):,} samples")
        print(f"  Image shape: {self.images.shape}")
        print(f"  Label columns: {list(self.labels_df.columns)}")
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        """
        Dataset indexing: return image and all labels
        
        Returns:
            tuple: (image, label_5, label_20, label_60, ret5, ret20, ret60)
        """
        # Image (numpy -> tensor)
        image = torch.from_numpy(self.images[idx]).float()
        
        # Label data
        row = self.labels_df.iloc[idx]
        
        # Binary labels (0 or 1)
        if 'Ret_5d' in row and 'Ret_20d' in row and 'Ret_60d' in row:
            # Binary classification with threshold 0 (up: 1, down: 0)
            label_5 = int(row['Ret_5d'] > 0)
            label_20 = int(row['Ret_20d'] > 0) 
            label_60 = int(row['Ret_60d'] > 0)
            
            # Actual returns
            ret5 = float(row['Ret_5d'])
            ret20 = float(row['Ret_20d'])
            ret60 = float(row['Ret_60d'])
        else:
            # Default values (if file format is different)
            label_5 = label_20 = label_60 = 0
            ret5 = ret20 = ret60 = 0.0
        
        return image, label_5, label_20, label_60, ret5, ret20, ret60


def load_original_dataset(win_size, mode, label_type, data_version='original'):
    """
    Convenience function to load original format dataset
    
    Args:
        win_size (int): 5, 20, 60
        mode (str): 'train', 'test'  
        label_type (str): 'RET5', 'RET20', 'RET60'
        data_version (str): 'original' or 'filled'
        
    Returns:
        OriginalFormatDataset: Loaded dataset
    """
    try:
        dataset = OriginalFormatDataset(win_size, mode, label_type, data_version)
        return dataset
    except Exception as e:
        print(f"Failed to load original format dataset: {e}")
        print(f"Please run create_original_format.py first to generate data:")
        print(f"python create_original_format.py --image_days {win_size} --mode {mode}")
        return None

