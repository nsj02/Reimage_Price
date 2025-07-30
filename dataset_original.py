#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dataset_original.py - 논문 저자 원본 형식 (.dat + .feather) 호환 Dataset

이 파일은 create_original_format.py로 생성된 원본 형식 파일들을 
PyTorch Dataset으로 읽어들이는 기능을 제공합니다.

파일 형식:
- Images: .dat (binary, uint8)
- Labels: .feather (pandas format)
"""

from __init__ import *
try:
    import pyarrow.feather as feather
except ImportError:
    # pandas 내장 feather 사용
    feather = type('feather', (), {
        'write_feather': pd.DataFrame.to_feather,
        'read_feather': pd.read_feather
    })()
import struct

class OriginalFormatDataset(torch.utils.data.Dataset):
    """
    논문 저자 원본 형식 (.dat + .feather) 데이터셋 로더
    
    디렉토리 구조:
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
    
    def __init__(self, win_size, mode, label_type):
        """
        Args:
            win_size (int): 윈도우 크기 (5, 20, 60)
            mode (str): 'train' 또는 'test'
            label_type (str): 'RET5', 'RET20', 'RET60'
        """
        self.win_size = win_size
        self.mode = mode
        self.label_type = label_type
        
        # 이미지 크기 설정
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
        
        # 연도 범위 설정
        if mode == 'train':
            self.years = range(1993, 2001)
        else:  # test
            self.years = range(2001, 2020)
            
        self.base_dir = "img_data_reconstructed"
        self.data_dir = os.path.join(self.base_dir, self.dir_name)
        
        # 데이터 로드
        self.load_data()
        
    def load_data(self):
        """
        모든 연도의 .dat 및 .feather 파일을 로드
        """
        print(f"원본 형식 데이터 로드 중: {self.data_dir}")
        
        all_images = []
        all_labels = []
        
        for year in self.years:
            # 파일 경로 생성
            images_file = f"{self.prefix}_{year}_images.dat"
            labels_file = f"{self.prefix}_{year}_labels_w_delay.feather"
            
            images_path = os.path.join(self.data_dir, images_file)
            labels_path = os.path.join(self.data_dir, labels_file)
            
            # 파일 존재 확인
            if not os.path.exists(images_path):
                print(f"  ⚠️ 이미지 파일 없음: {images_file}")
                continue
            if not os.path.exists(labels_path):
                print(f"  ⚠️ 라벨 파일 없음: {labels_file}")
                continue
                
            # 이미지 로드 (.dat binary)
            try:
                images = np.fromfile(images_path, dtype=np.uint8)
                num_images = len(images) // self.image_size
                images = images.reshape(num_images, self.image_height, self.image_width)
                print(f"  ✅ {year}년 이미지: {num_images:,}개")
            except Exception as e:
                print(f"  ❌ {year}년 이미지 로드 실패: {e}")
                continue
                
            # 라벨 로드 (.feather)
            try:
                labels_df = feather.read_feather(labels_path)
                print(f"  ✅ {year}년 라벨: {len(labels_df):,}개")
            except Exception as e:
                print(f"  ❌ {year}년 라벨 로드 실패: {e}")
                continue
                
            # 개수 일치 확인
            if num_images != len(labels_df):
                print(f"  ⚠️ {year}년 이미지-라벨 개수 불일치: {num_images} vs {len(labels_df)}")
                min_count = min(num_images, len(labels_df))
                images = images[:min_count]
                labels_df = labels_df.iloc[:min_count]
                
            all_images.append(images)
            all_labels.append(labels_df)
        
        if len(all_images) == 0:
            raise FileNotFoundError(f"사용 가능한 데이터가 없습니다: {self.data_dir}")
            
        # 모든 연도 데이터 결합
        self.images = np.concatenate(all_images, axis=0)
        self.labels_df = pd.concat(all_labels, ignore_index=True)
        
        print(f"총 로드된 데이터: {len(self.images):,}개")
        print(f"  이미지 형태: {self.images.shape}")
        print(f"  라벨 컬럼: {list(self.labels_df.columns)}")
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        """
        Dataset 인덱싱: 이미지와 모든 라벨 반환
        
        Returns:
            tuple: (image, label_5, label_20, label_60, ret5, ret20, ret60)
        """
        # 이미지 (numpy -> tensor)
        image = torch.from_numpy(self.images[idx]).float()
        
        # 라벨 데이터
        row = self.labels_df.iloc[idx]
        
        # 이진 라벨 (0 또는 1)
        if 'Ret_5d' in row and 'Ret_20d' in row and 'Ret_60d' in row:
            # 임계값 0으로 이진 분류 (상승: 1, 하락: 0)
            label_5 = int(row['Ret_5d'] > 0)
            label_20 = int(row['Ret_20d'] > 0) 
            label_60 = int(row['Ret_60d'] > 0)
            
            # 실제 수익률
            ret5 = float(row['Ret_5d'])
            ret20 = float(row['Ret_20d'])
            ret60 = float(row['Ret_60d'])
        else:
            # 기본값 (파일 형식이 다른 경우)
            label_5 = label_20 = label_60 = 0
            ret5 = ret20 = ret60 = 0.0
        
        return image, label_5, label_20, label_60, ret5, ret20, ret60


def load_original_dataset(win_size, mode, label_type):
    """
    원본 형식 데이터셋 로드 편의 함수
    
    Args:
        win_size (int): 5, 20, 60
        mode (str): 'train', 'test'  
        label_type (str): 'RET5', 'RET20', 'RET60'
        
    Returns:
        OriginalFormatDataset: 로드된 데이터셋
    """
    try:
        dataset = OriginalFormatDataset(win_size, mode, label_type)
        return dataset
    except Exception as e:
        print(f"❌ 원본 형식 데이터셋 로드 실패: {e}")
        print(f"create_original_format.py를 먼저 실행하여 데이터를 생성하세요:")
        print(f"python create_original_format.py --image_days {win_size} --mode {mode}")
        return None


if __name__ == '__main__':
    # 테스트 코드
    print("🧪 원본 형식 데이터셋 테스트")
    
    for win_size in [5, 20, 60]:
        for mode in ['train', 'test']:
            print(f"\n=== {win_size}일 {mode} 데이터셋 ===")
            try:
                dataset = load_original_dataset(win_size, mode, f'RET{win_size}')
                if dataset:
                    print(f"✅ 로드 성공: {len(dataset):,}개 샘플")
                    
                    # 첫 번째 샘플 확인
                    sample = dataset[0]
                    image, l5, l20, l60, r5, r20, r60 = sample
                    print(f"  이미지 크기: {image.shape}")
                    print(f"  라벨: {l5}, {l20}, {l60}")
                    print(f"  수익률: {r5:.4f}, {r20:.4f}, {r60:.4f}")
                else:
                    print("❌ 로드 실패")
            except Exception as e:
                print(f"❌ 오류: {e}")