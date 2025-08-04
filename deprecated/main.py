#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Re-Imaging Price Trends - Main Training Script
논문 구현: 고정 기간 학습 (1993-2000 train, 2001-2019 test)

사용법:
    python main.py --model CNN5d --image_days 5 --pred_days 5
    python main.py --model CNN20d --image_days 20 --pred_days 20
    python main.py --model CNN60d --image_days 60 --pred_days 60
"""

from __init__ import *
import model as _M
import train as _T
import dataset as _D
import dataset_original as _D_ORIG
import argparse
import os

def main():
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description='CNN 주가 예측 모델 학습')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['CNN5d', 'CNN20d', 'CNN60d'],
                       help='모델 타입')
    parser.add_argument('--image_days', type=int, required=True,
                       choices=[5, 20, 60],
                       help='이미지 윈도우 크기 (일)')
    parser.add_argument('--pred_days', type=int, required=True,
                       choices=[5, 20, 60], 
                       help='예측 기간 (일)')
    parser.add_argument('--sample_rate', type=float, default=1.0,
                       help='데이터 샘플링 비율 (default: 1.0)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='배치 크기 (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='최대 에포크 (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='학습률 (default: 1e-5)')
    parser.add_argument('--use_original_format', action='store_true',
                       help='원본 형식 (.dat + .feather) 사용')
    parser.add_argument('--ensemble_runs', type=int, default=1,
                       help='앙상블 실행 횟수 (논문: 5회, default: 1회)')
    
    args = parser.parse_args()
    
    print(f"모델 학습 시작: {args.model}")
    print(f"  이미지 윈도우: {args.image_days}일")
    print(f"  예측 기간: {args.pred_days}일") 
    print(f"  배치 크기: {args.batch_size}")
    print(f"  학습률: {args.lr}")
    
    # 디바이스 설정
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  디바이스: {device}")
    
    # 출력 디렉토리 생성
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # 앙상블 학습 루프
    for run_idx in range(args.ensemble_runs):
        print(f"\n{'='*60}")
        print(f"앙상블 실행 {run_idx+1}/{args.ensemble_runs}")
        print(f"{'='*60}")
        
        # 파일명 설정 (앙상블용)
        if args.ensemble_runs > 1:
            model_name = f"{args.model}_I{args.image_days}R{args.pred_days}_run{run_idx+1}"
        else:
            model_name = f"{args.model}_I{args.image_days}R{args.pred_days}"
        
        model_file = f"models/{model_name}.tar"
        log_file = f"logs/{model_name}.csv"
        
        # 이미 학습된 모델 확인
        if os.path.exists(model_file):
            print(f"이미 학습된 모델 존재: {model_file}")
            print("다음 앙상블 실행으로 건너뜁니다.")
            continue
        
        # 데이터셋 로드 (앙상블 실행마다 다시 로드하여 랜덤성 확보)
        if args.use_original_format:
            print(f"\n원본 형식 데이터셋 로드 중...")
            full_dataset = _D_ORIG.load_original_dataset(
                win_size=args.image_days,
                mode='train',
                label_type=f'RET{args.pred_days}'
            )
            if full_dataset is None:
                print(f"다음 명령어로 원본 형식 이미지를 먼저 생성하세요:")
                print(f"python create_original_format.py --image_days {args.image_days} --mode train")
                return
        else:
            # 사전 생성된 이미지 디렉토리 확인
            image_dir = f"images/train_I{args.image_days}R{args.pred_days}"
            metadata_file = os.path.join(image_dir, 'metadata.csv')
            
            if not os.path.exists(metadata_file):
                print(f"\n❌ 사전 생성된 이미지가 없습니다: {image_dir}")
                print(f"다음 명령어로 이미지를 먼저 생성하세요:")
                print(f"python create_images_optimized.py --image_days {args.image_days} --mode train --pred_days {args.pred_days}")
                return
            
            # 사전 생성된 이미지 데이터셋 로드
            print(f"\n사전 생성된 이미지 로드 중: {image_dir}")
            full_dataset = _D.PrecomputedImageDataset(
                image_dir=image_dir,
                label_type=f'RET{args.pred_days}'
            )
        
        # 훈련/검증 분할 (논문: 70:30 랜덤 분할) - 앙상블마다 다른 분할로 랜덤성 확보
        train_size = int(len(full_dataset) * 0.7)
        valid_size = len(full_dataset) - train_size
        
        train_dataset, valid_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, valid_size]
        )
        
        # DataLoader 생성
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=args.batch_size, shuffle=True
        )
        
        print(f"훈련 데이터: {len(train_dataset):,}개")
        print(f"검증 데이터: {len(valid_dataset):,}개")
        
        # 모델 초기화
        print(f"\n{args.model} 모델 초기화...")
        if args.model == 'CNN5d':
            model = _M.CNN5d()
        elif args.model == 'CNN20d':
            model = _M.CNN20d() 
        elif args.model == 'CNN60d':
            model = _M.CNN60d()
        
        model.to(device)
        
        # 파라미터 수 출력
        total_params = sum(p.numel() for p in model.parameters())
        print(f"모델 파라미터: {total_params:,}개")
        
        # 손실함수 및 옵티마이저 (Sigmoid + BCELoss)
        criterion = nn.BCELoss().to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
        
        # 모델 학습
        print(f"\n모델 학습 시작 (최대 {args.epochs}에포크)...")
        train_loss_set, valid_loss_set, train_acc_set, valid_acc_set = _T.train_n_epochs(
            n_epochs=args.epochs,
            model=model,
            label_type=f'RET{args.pred_days}',
            train_loader=train_loader,
            valid_loader=valid_loader,
            criterion=criterion,
            optimizer=optimizer,
            savefile=model_file,
            early_stop_epoch=2  # 논문: 2에포크
        )
        
        # 결과 로그 저장
        print(f"\n학습 로그 저장: {log_file}")
        log = pd.DataFrame({
            'train_loss': train_loss_set,
            'train_acc': train_acc_set,
            'valid_loss': valid_loss_set,
            'valid_acc': valid_acc_set
        })
        log.to_csv(log_file, index=False)
        
        print(f"\n학습 완료!")
        print(f"  모델: {model_file}")
        print(f"  로그: {log_file}")
        print(f"  최종 검증 정확도: {valid_acc_set[-1]:.4f}")
    
    print(f"\n🎉 모든 앙상블 실행 완료!")

if __name__ == '__main__':
    main()