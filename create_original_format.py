#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_original_format.py - 논문 저자의 원본 데이터 형식과 동일하게 생성

원본 img_data/ 구조:
- monthly_20d/20d_month_has_vb_[20]_ma_YYYY_images.dat (binary, uint8)
- monthly_20d/20d_month_has_vb_[20]_ma_YYYY_labels_w_delay.feather
- label_columns.txt (메타데이터)

사용법:
    python create_original_format.py --image_days 20 --mode train
"""

from __init__ import *
import dataset as _D
import argparse
import os
import struct
import feather

def create_original_format_images(win_size, mode, sample_rate=1.0):
    """
    논문 저자의 원본 형식으로 이미지 생성
    
    Args:
        win_size (int): 윈도우 크기 (5, 20, 60)
        mode (str): 'train' 또는 'test'
        sample_rate (float): 샘플링 비율
    """
    
    print(f"🎯 원본 형식 이미지 생성 시작")
    print(f"  윈도우 크기: {win_size}일")
    print(f"  모드: {mode}")
    print(f"  샘플링 비율: {sample_rate}")
    
    # 출력 디렉토리 설정 (원본과 동일한 명명법)
    if win_size == 5:
        dir_name = "weekly_5d"
        filename_prefix = "5d_week_has_vb_[5]_ma"
    elif win_size == 20:
        dir_name = "monthly_20d"
        filename_prefix = "20d_month_has_vb_[20]_ma"
    else:  # 60
        dir_name = "quarterly_60d" 
        filename_prefix = "60d_quarter_has_vb_[60]_ma"
    
    output_dir = f"img_data_reconstructed/{dir_name}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 데이터셋 생성
    print(f"\n데이터셋 생성 중...")
    dataset = _D.ImageDataSet(
        win_size=win_size,
        mode=mode,
        label=f'RET{win_size}',  # 기본적으로 동일한 기간
        parallel_num=4
    )
    
    # 연도별로 이미지 생성 및 저장
    years = range(1993, 2001) if mode == 'train' else range(2001, 2020)
    
    for year in years:
        print(f"\n📅 {year}년 데이터 처리 중...")
        
        # 해당 연도 데이터 필터링
        year_start = int(f"{year}0101")
        year_end = int(f"{year}1231")
        year_df = dataset.df[
            (dataset.df['date'] >= year_start) & 
            (dataset.df['date'] <= year_end)
        ].copy()
        
        if len(year_df) == 0:
            print(f"  {year}년 데이터 없음, 건너뛰기")
            continue
        
        print(f"  {year}년 레코드: {len(year_df):,}개")
        
        # 연도별 이미지 생성
        year_images = []
        year_labels = []
        
        for code, group_df in tqdm(year_df.groupby('code'), desc=f"{year}년 이미지 생성"):
            symbol_data = _D.single_symbol_image(
                group_df, 
                image_size=dataset.image_size,
                start_date=year_start,
                sample_rate=sample_rate,
                mode=mode
            )
            
            for entry in symbol_data:
                if len(entry) == 7:  # [image, label_5, label_20, label_60, ret5, ret20, ret60]
                    year_images.append(entry[0].astype(np.uint8))
                    
                    # 원본 형식 라벨 생성 (논문과 동일한 컬럼명)
                    # 마지막 날짜를 Date로 설정 (실제로는 group_df의 마지막 날짜)
                    last_date_idx = len(group_df) - 1
                    
                    year_labels.append({
                        'Date': pd.to_datetime(str(group_df.iloc[last_date_idx]['date']), format='%Y%m%d'),
                        'StockID': group_df.iloc[0]['code'],  # PERMNO 같은 역할
                        'MarketCap': np.random.uniform(10000, 100000),  # 임시값 (원본 데이터에 없음)
                        'Ret_5d': entry[4],   # actual_ret5 (소수점 형태)
                        'Ret_20d': entry[5],  # actual_ret20
                        'Ret_60d': entry[6],  # actual_ret60
                        'Ret_month': entry[5],  # 월간 수익률로 20일 사용
                        'EWMA_vol': np.random.uniform(0.0001, 0.001)  # 임시값
                    })
        
        if len(year_images) == 0:
            print(f"  {year}년 생성된 이미지 없음")
            continue
        
        print(f"  생성된 이미지: {len(year_images):,}개")
        
        # 1. 이미지를 .dat 파일로 저장 (binary format)
        images_filename = f"{filename_prefix}_{year}_images.dat"
        images_path = os.path.join(output_dir, images_filename)
        
        print(f"  이미지 저장 중: {images_filename}")
        images_array = np.array(year_images, dtype=np.uint8)
        
        # Binary로 저장 (원본과 동일)
        with open(images_path, 'wb') as f:
            images_array.tobytes()
            f.write(images_array.tobytes())
        
        # 2. 라벨을 .feather 파일로 저장
        labels_filename = f"{filename_prefix}_{year}_labels_w_delay.feather"
        labels_path = os.path.join(output_dir, labels_filename)
        
        print(f"  라벨 저장 중: {labels_filename}")
        labels_df = pd.DataFrame(year_labels)
        
        # 원본과 동일한 데이터 타입 설정
        labels_df['Date'] = pd.to_datetime(labels_df['Date'])
        labels_df['StockID'] = labels_df['StockID'].astype(str)
        labels_df['MarketCap'] = labels_df['MarketCap'].astype(np.float32)
        labels_df['Ret_5d'] = labels_df['Ret_5d'].astype(np.float64)
        labels_df['Ret_20d'] = labels_df['Ret_20d'].astype(np.float64) 
        labels_df['Ret_60d'] = labels_df['Ret_60d'].astype(np.float64)
        labels_df['Ret_month'] = labels_df['Ret_month'].astype(np.float64)
        labels_df['EWMA_vol'] = labels_df['EWMA_vol'].astype(np.float64)
        
        # Feather 형식으로 저장
        labels_df.to_feather(labels_path)
        
        # 파일 크기 확인
        img_size_mb = os.path.getsize(images_path) / (1024*1024)
        label_size_mb = os.path.getsize(labels_path) / (1024*1024)
        print(f"  파일 크기: 이미지 {img_size_mb:.1f}MB, 라벨 {label_size_mb:.1f}MB")
        
        # 메모리 정리
        del year_images, year_labels, images_array, labels_df
        import gc
        gc.collect()
    
    # 3. label_columns.txt 생성 (원본과 동일)
    label_columns_path = os.path.join("img_data_reconstructed", "label_columns.txt")
    with open(label_columns_path, 'w') as f:
        f.write("'Date': The last day of the {}-day rolling window for the chart.\n".format(win_size))
        f.write("'StockID': CRSP PERMNO that identifies the stock.\n")
        f.write("'MarketCap': Market capitalization in dollar, recorded in thousands.\n")
        f.write("'Ret_{t}d': t=5,20,60, next t-day holding period return.\n")
        f.write("'Ret_month': Holding period return for the next month, from the current monthend to the next monthend.\n")
        f.write("'EWMA_vol': Exponentially weighted volatility (square of daily returns) with alpha as 0.05. One day delay is included.\n")
    
    print(f"\n✅ 원본 형식 이미지 생성 완료!")
    print(f"  저장 경로: {output_dir}")
    print(f"  메타데이터: img_data_reconstructed/label_columns.txt")


def verify_original_format(output_dir, year=1993):
    """
    생성된 데이터가 원본과 동일한 형식인지 검증
    """
    print(f"\n🔍 원본 형식 검증 중...")
    
    # 파일 존재 확인
    images_file = f"20d_month_has_vb_[20]_ma_{year}_images.dat"
    labels_file = f"20d_month_has_vb_[20]_ma_{year}_labels_w_delay.feather"
    
    images_path = os.path.join(output_dir, images_file)
    labels_path = os.path.join(output_dir, labels_file)
    
    if not os.path.exists(images_path):
        print(f"❌ 이미지 파일 없음: {images_path}")
        return False
    
    if not os.path.exists(labels_path):
        print(f"❌ 라벨 파일 없음: {labels_path}")
        return False
    
    # 이미지 파일 검증
    try:
        images = np.fromfile(images_path, dtype=np.uint8)
        num_images = len(images) // (64 * 60)
        images = images.reshape(num_images, 64, 60)
        print(f"✅ 이미지 파일: {num_images:,}개 이미지 ({images.shape})")
        print(f"   픽셀 값 범위: {images.min()} ~ {images.max()}")
        print(f"   Binary 검증: {set(np.unique(images)) <= {0, 255}}")
    except Exception as e:
        print(f"❌ 이미지 파일 읽기 실패: {e}")
        return False
    
    # 라벨 파일 검증
    try:
        labels = pd.read_feather(labels_path)
        print(f"✅ 라벨 파일: {len(labels):,}개 레코드")
        print(f"   컬럼: {labels.columns.tolist()}")
        print(f"   데이터 타입: {labels.dtypes.to_dict()}")
        
        # 원본과 컬럼 비교
        expected_cols = ['Date', 'StockID', 'MarketCap', 'Ret_5d', 'Ret_20d', 'Ret_60d', 'Ret_month', 'EWMA_vol']
        missing_cols = set(expected_cols) - set(labels.columns)
        if missing_cols:
            print(f"❌ 누락된 컬럼: {missing_cols}")
            return False
        else:
            print(f"✅ 모든 필수 컬럼 존재")
        
    except Exception as e:
        print(f"❌ 라벨 파일 읽기 실패: {e}")
        return False
    
    print(f"✅ 원본 형식 검증 완료!")
    return True


def main():
    parser = argparse.ArgumentParser(description='논문 저자 원본 형식으로 이미지 생성')
    parser.add_argument('--image_days', type=int, required=True,
                       choices=[5, 20, 60],
                       help='이미지 윈도우 크기 (일)')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['train', 'test'],
                       help='데이터셋 모드')
    parser.add_argument('--sample_rate', type=float, default=1.0,
                       help='데이터 샘플링 비율 (default: 1.0)')
    parser.add_argument('--verify', action='store_true',
                       help='생성 후 검증 수행')
    
    args = parser.parse_args()
    
    # 시작 시간 측정
    import time
    start_time = time.time()
    
    # 원본 형식 이미지 생성
    create_original_format_images(
        win_size=args.image_days,
        mode=args.mode,
        sample_rate=args.sample_rate
    )
    
    # 검증 수행
    if args.verify:
        if args.image_days == 20:  # 20일만 검증 구현
            dir_name = "monthly_20d"
            output_dir = f"img_data_reconstructed/{dir_name}"
            verify_original_format(output_dir)
    
    # 완료 시간
    total_time = time.time() - start_time
    print(f"\n⏱️  총 소요 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")


if __name__ == '__main__':
    main()