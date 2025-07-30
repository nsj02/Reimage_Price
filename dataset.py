#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset.py - 주가 데이터를 캔들차트 이미지로 변환하는 모듈
"""

from __init__ import *
import utils as _U
reload(_U)
import h5py


def single_symbol_image(tabular_df, image_size, start_date, sample_rate, mode):
    """
    개별 종목의 주가 데이터를 캔들차트 이미지로 변환
    
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
        
        # 룩백 윈도우 데이터 추출
        price_slice = tabular_df[d-(lookback-1):d+1][['open', 'high', 'low', 'close']].reset_index(drop=True)
        volume_slice = tabular_df[d-(lookback-1):d+1][['volume']].reset_index(drop=True)
        
        # NA 값 필터링
        if price_slice[['open', 'high', 'low', 'close']].isnull().any().any():
            continue
        if volume_slice['volume'].isnull().any():
            continue
        
        # IPO/상장폐지 필터링
        if (1.0*(price_slice[['open', 'high', 'low', 'close']].sum(axis=1)/price_slice['open'] == 4)).sum() > lookback//5:
            continue
        
        valid_dates.append(tabular_df.iloc[d]['date'])
        
        # 2단계 정규화
        ret_slice = tabular_df[d-(lookback-1):d+1][['ret']].reset_index(drop=True)
        
        # 1단계: 수익률 기반 가격 재구성
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
        
        # 이동평균선 추가
        normalized_prices['ma'] = pd.Series(normalized_close).rolling(window=lookback, min_periods=1).mean()
        
        # 2단계: Min-Max 스케일링
        all_ohlc_values = np.concatenate([
            normalized_prices['open'].values,
            normalized_prices['high'].values, 
            normalized_prices['low'].values,
            normalized_prices['close'].values,
            normalized_prices['ma'].values
        ])
        price_min, price_max = np.min(all_ohlc_values), np.max(all_ohlc_values)
        price_slice = (normalized_prices - price_min) / (price_max - price_min)
        
        # 거래량 정규화
        volume_slice = (volume_slice - np.min(volume_slice.values))/(np.max(volume_slice.values) - np.min(volume_slice.values))
        
        # 픽셀 좌표 변환 (가격: 상단 영역, 거래량: 하단 영역 완전 분리)
        if image_size[0] == 32:
            # 5d: 가격 0-25행 (26행), 거래량 26-31행 (6행)
            price_slice = price_slice.apply(lambda x: x*25).astype(np.int32)
            volume_slice = volume_slice.apply(lambda x: x*5).astype(np.int32)
        elif image_size[0] == 64:
            # 20d: 가격 0-51행 (52행), 거래량 52-63행 (12행)  
            price_slice = price_slice.apply(lambda x: x*51).astype(np.int32)
            volume_slice = volume_slice.apply(lambda x: x*11).astype(np.int32)
        else:  # 96
            # 60d: 가격 0-76행 (77행), 거래량 77-95행 (19행)
            price_slice = price_slice.apply(lambda x: x*76).astype(np.int32)
            volume_slice = volume_slice.apply(lambda x: x*18).astype(np.int32)
        
        # 이미지 생성
        image = np.zeros(image_size)
        
        # 영역 경계 설정
        if image_size[0] == 32:  # 5일 모델
            price_region_end = 25  # 가격 영역: 0-25행
            volume_start_row = 26  # 거래량 영역: 26-31행
        elif image_size[0] == 64:  # 20일 모델  
            price_region_end = 51  # 가격 영역: 0-51행
            volume_start_row = 52  # 거래량 영역: 52-63행
        else:  # 96, 60일 모델
            price_region_end = 76  # 가격 영역: 0-76행
            volume_start_row = 77  # 거래량 영역: 77-95행
        
        for i in range(len(price_slice)):
            # 캔들스틱 (가격 영역에만)
            open_px = min(price_slice.loc[i]['open'], price_region_end)
            close_px = min(price_slice.loc[i]['close'], price_region_end)
            low_px = min(price_slice.loc[i]['low'], price_region_end)
            high_px = min(price_slice.loc[i]['high'], price_region_end)
            
            image[open_px, i*3] = 255.
            image[low_px:high_px+1, i*3+1] = 255.  # High-Low bar 가격 영역에만
            image[close_px, i*3+2] = 255.
            
            # 이동평균선 (가격 영역에만)
            if not pd.isna(price_slice.loc[i]['ma']):
                ma_px = min(int(price_slice.loc[i]['ma']), price_region_end)
                image[ma_px, i*3:i*3+3] = 255.
            
            # 거래량을 하단 전용 영역에 렌더링 (완전 분리)
            volume_height = int(volume_slice.loc[i]['volume'])
            if volume_height > 0:
                volume_bottom = image_size[0] - 1  # 맨 아래 픽셀
                volume_top = max(volume_start_row, volume_bottom - volume_height + 1)
                image[volume_top:volume_bottom+1, i*3+1] = 255.
        
        # 라벨 추출: 이진 라벨과 실제 수익률 모두
        label_ret5 = tabular_df.iloc[d]['label_5']
        label_ret20 = tabular_df.iloc[d]['label_20'] 
        label_ret60 = tabular_df.iloc[d]['label_60']
        
        actual_ret5 = tabular_df.iloc[d]['ret5'] / 100.0  # % → 소수
        actual_ret20 = tabular_df.iloc[d]['ret20'] / 100.0
        actual_ret60 = tabular_df.iloc[d]['ret60'] / 100.0
        
        # 미래 수익률이 NA인 경우 건너뛰기
        if pd.isna(label_ret5) or pd.isna(label_ret20) or pd.isna(label_ret60):
            continue
        if pd.isna(actual_ret5) or pd.isna(actual_ret20) or pd.isna(actual_ret60):
            continue
        
        # 시가총액과 EWMA volatility 값 추가
        market_cap_val = tabular_df.iloc[d].get('market_cap', np.random.uniform(10000, 100000))
        ewma_vol_val = tabular_df.iloc[d].get('ewma_vol', np.random.uniform(0.0001, 0.001))
        
        # 날짜와 종목코드 정보 포함 (original format 호환) - 11개 요소
        entry_date = tabular_df.iloc[d]['date']
        entry_code = tabular_df.iloc[0]['code']
        entry = [image, label_ret5, label_ret20, label_ret60, actual_ret5, actual_ret20, actual_ret60, 
                entry_date, entry_code, market_cap_val, ewma_vol_val]
        dataset.append(entry)
    
    if mode == 'train' or mode == 'test':
        return dataset
    else:
        return [tabular_df.iloc[0]['code'], dataset, valid_dates]


class PrecomputedImageDataset(torch.utils.data.Dataset):
    """
    HDF5에 저장된 이미지를 로드하는 메모리 효율적인 Dataset
    """
    
    def __init__(self, image_dir, label_type):
        """
        Args:
            image_dir (str): 이미지가 저장된 디렉토리
            label_type (str): 'RET5', 'RET20', 'RET60'
        """
        self.image_dir = image_dir
        self.label_type = label_type
        
        # 메타데이터 로드
        metadata_file = os.path.join(image_dir, 'metadata.csv')
        if not os.path.exists(metadata_file):
            raise FileNotFoundError(f"메타데이터 파일이 없습니다: {metadata_file}")
        
        # HDF5 파일 경로
        self.hdf5_path = os.path.join(image_dir, 'images.h5')
        if not os.path.exists(self.hdf5_path):
            raise FileNotFoundError(f"HDF5 파일이 없습니다: {self.hdf5_path}")
        
        self.metadata = pd.read_csv(metadata_file)
        print(f"로드된 이미지 수: {len(self.metadata):,}개")
        
        # HDF5 파일 lazy loading을 위한 변수
        self._hdf5_file = None
        
    def __del__(self):
        """소멸자: HDF5 파일 안전하게 닫기"""
        if hasattr(self, '_hdf5_file') and self._hdf5_file is not None:
            try:
                self._hdf5_file.close()
            except:
                pass
        
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # HDF5 파일에서 이미지 로드 (lazy loading)
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r')
        
        image = self._hdf5_file['images'][idx].astype(np.float32)
        
        # 라벨 선택
        label_5 = int(row['label_5'])
        label_20 = int(row['label_20']) 
        label_60 = int(row['label_60'])
        
        actual_ret5 = float(row['ret5'])
        actual_ret20 = float(row['ret20'])
        actual_ret60 = float(row['ret60'])
        
        return image, label_5, label_20, label_60, actual_ret5, actual_ret20, actual_ret60


class ImageDataSet():
    """
    주가 데이터를 캔들차트 이미지로 변환하는 메인 클래스
    """
    
    def __init__(self, win_size, mode, label, parallel_num=-1):
        """
        Args:
            win_size (int): 윈도우 크기 (5, 20, 60)
            mode (str): 모드 ('train', 'test', 'inference')
            label (str): 라벨 타입 ('RET5', 'RET20', 'RET60')
            parallel_num (int): 병렬 처리 수
        """
        
        # 기간 설정
        if mode == 'train':
            self.start_date = 19930101
            self.end_date = 20001231
        elif mode == 'test':
            self.start_date = 20010101
            self.end_date = 20191231
        else:  # inference
            self.start_date = 19930101
            self.end_date = 20191231
        
        # 파라미터 검증
        assert win_size in [5, 20, 60], f'지원하지 않는 윈도우 크기: {win_size}'
        assert mode in ['train', 'test', 'inference'], f'지원하지 않는 모드: {mode}'
        assert label in ['RET5', 'RET20', 'RET60'], f'지원하지 않는 라벨: {label}'
        
        # 이미지 크기 설정
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
        
        self.load_data()
        
        print(f"데이터셋 초기화 완료")
        print(f"  모드: {self.mode}")
        print(f"  이미지 크기: {self.image_size}")
        print(f"  기간: {self.start_date} ~ {self.end_date}")
        print(f"  라벨: {self.label}")
        
    @_U.timer('데이터 로드', '8')
    def load_data(self):
        """
        WRDS CRSP 데이터 로드
        """
        
        if self.mode in ['train']:
            data_file = 'data/data_1993_2000_train_val.parquet'
        elif self.mode in ['test', 'inference']:
            data_file = 'data/data_2001_2019_test.parquet'
        
        if not os.path.exists(data_file):
            print(f"데이터 파일이 존재하지 않습니다: {data_file}")
            raise FileNotFoundError(f"Data file not found: {data_file}")
        
        print(f"데이터 로드 중: {data_file}")
        tabularDf = pd.read_parquet(data_file)
        
        # 날짜 형식 변환
        tabularDf['date'] = pd.to_datetime(tabularDf['date']).dt.strftime('%Y%m%d').astype(np.int32)
        
        print(f"  로드된 레코드: {len(tabularDf):,}개")
        print(f"  종목 수: {tabularDf['code'].nunique():,}개")
        print(f"  기간: {tabularDf['date'].min()} ~ {tabularDf['date'].max()}")
        
        self.df = tabularDf.copy()
        del tabularDf
        
        print(f"데이터 로드 완료")
        
    def generate_images(self, sample_rate):
        """
        이미지 생성
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
                print(f"  하락(0): {num0:,}개 ({100*num0/total:.1f}%), 상승(1): {num1:,}개 ({100*num1/total:.1f}%)")
                print(f"  총 이미지: {total:,}개")
            
            return image_set
        else:
            return dataset_all
    
    def save_images_to_disk(self, output_dir, sample_rate=1.0):
        """
        스트리밍 방식으로 이미지를 HDF5에 저장 (메모리 효율적)
        
        Args:
            output_dir (str): 이미지를 저장할 디렉토리
            sample_rate (float): 샘플링 비율
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"스트리밍 이미지 생성 및 저장 시작: {output_dir}")
        print(f"  모드: {self.mode}")
        print(f"  이미지 크기: {self.image_size}")
        print(f"  샘플링 비율: {sample_rate}")
        
        hdf5_path = os.path.join(output_dir, 'images.h5')
        metadata = []
        
        # 1단계: 총 이미지 개수 추정
        print("이미지 개수 추정 중...")
        total_images = 0
        for code, group_df in tqdm(self.df.groupby('code'), desc="Counting images"):
            lookback = self.image_size[1] // 3
            valid_count = 0
            for d in range(lookback-1, len(group_df), lookback):
                if group_df.iloc[d]['date'] >= self.start_date:
                    valid_count += 1
            total_images += int(valid_count * sample_rate)
        
        print(f"예상 이미지 개수: {total_images:,}개")
        
        # 2단계: HDF5 파일 초기화 및 스트리밍 저장
        with h5py.File(hdf5_path, 'w') as f:
            # 데이터셋 미리 할당
            images_dataset = f.create_dataset(
                'images', 
                shape=(total_images, *self.image_size),
                dtype=np.uint8,
                compression='gzip',
                compression_opts=6,  # 9는 너무 느림, 6이 적당
                chunks=True
            )
            
            image_idx = 0
            
            # 병렬 처리 없이 순차적으로 처리 (메모리 안전)
            for code, group_df in tqdm(self.df.groupby('code'), desc="Generating & saving images"):
                symbol_images = single_symbol_image(
                    group_df, 
                    image_size=self.image_size,
                    start_date=self.start_date,
                    sample_rate=sample_rate,
                    mode=self.mode
                )
                
                # 즉시 HDF5에 저장
                for entry in symbol_images:
                    if len(entry) == 11 and image_idx < total_images:  # 11개 요소 (날짜, 코드, 시가총액, EWMA 포함)
                        # 이미지를 HDF5에 직접 저장
                        images_dataset[image_idx] = entry[0].astype(np.uint8)
                        
                        # 메타데이터
                        if self.mode in ['train', 'test']:
                            metadata.append({
                                'image_id': image_idx,
                                'label_5': entry[1],
                                'label_20': entry[2],
                                'label_60': entry[3],
                                'ret5': entry[4],
                                'ret20': entry[5],
                                'ret60': entry[6]
                            })
                        else:  # inference
                            metadata.append({
                                'image_id': image_idx,
                                'code': code,
                                'label_5': entry[1],
                                'label_20': entry[2],
                                'label_60': entry[3],
                                'ret5': entry[4],
                                'ret20': entry[5],
                                'ret60': entry[6]
                            })
                        
                        image_idx += 1
                
                # 메모리 정리
                del symbol_images
                import gc
                gc.collect()
            
            # 실제 저장된 이미지 개수에 맞게 크기 조정
            if image_idx < total_images:
                images_dataset.resize((image_idx, *self.image_size))
        
        # 메타데이터 저장
        metadata_df = pd.DataFrame(metadata)
        metadata_path = os.path.join(output_dir, 'metadata.csv')
        metadata_df.to_csv(metadata_path, index=False)
        
        print(f"\n이미지 저장 완료:")
        print(f"  저장된 이미지: {len(metadata):,}개")
        print(f"  저장 경로: {output_dir}")
        
        # 라벨 분포 출력 (훈련 모드일 때만)
        if self.mode == 'train':
            label_col = {'RET5': 'label_5', 'RET20': 'label_20', 'RET60': 'label_60'}[self.label]
            num0 = (metadata_df[label_col] == 0).sum()
            num1 = (metadata_df[label_col] == 1).sum()
            total = num0 + num1
            
            print(f"\n라벨 분포 ({self.label}):")
            print(f"  하락(0): {num0:,}개 ({100*num0/total:.1f}%)")
            print(f"  상승(1): {num1:,}개 ({100*num1/total:.1f}%)")
        
        return len(metadata)  # 저장된 이미지 개수 반환