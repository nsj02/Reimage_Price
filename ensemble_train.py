#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ensemble_train.py - 논문 방식 5모델 앙상블 학습

논문에서 언급한 대로 동일한 모델을 5번 독립적으로 훈련하여
확률적 최적화의 변동성을 줄이는 앙상블 방법 구현

사용법:
    python ensemble_train.py --model CNN5d --image_days 5 --pred_days 5 --ensemble_runs 5
"""

import subprocess
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description='CNN 앙상블 모델 학습')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['CNN5d', 'CNN20d', 'CNN60d'],
                       help='모델 타입')
    parser.add_argument('--image_days', type=int, required=True,
                       choices=[5, 20, 60],
                       help='이미지 윈도우 크기 (일)')
    parser.add_argument('--pred_days', type=int, required=True,
                       choices=[5, 20, 60], 
                       help='예측 기간 (일)')
    parser.add_argument('--ensemble_runs', type=int, default=5,
                       help='앙상블 실행 횟수 (논문: 5회, default: 5회)')
    parser.add_argument('--use_original_format', action='store_true',
                       help='원본 형식 (.dat + .feather) 사용')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='배치 크기 (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='최대 에포크 (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='학습률 (default: 1e-5)')
    
    args = parser.parse_args()
    
    print(f"🔥 논문 방식 앙상블 학습 시작")
    print(f"   모델: {args.model}")
    print(f"   앙상블 실행: {args.ensemble_runs}회")
    print(f"   이미지 윈도우: {args.image_days}일")
    print(f"   예측 기간: {args.pred_days}일")
    
    # 각 앙상블 실행
    successful_runs = 0
    for run_idx in range(1, args.ensemble_runs + 1):
        print(f"\n{'='*70}")
        print(f"🧠 앙상블 실행 {run_idx}/{args.ensemble_runs}")
        print(f"{'='*70}")
        
        # 모델 파일명 (앙상블용)
        model_name = f"{args.model}_I{args.image_days}R{args.pred_days}_run{run_idx}"
        model_file = f"models/{model_name}.tar"
        
        # 이미 훈련된 모델 확인
        if os.path.exists(model_file):
            print(f"✅ 이미 훈련된 모델: {model_file}")
            successful_runs += 1
            continue
        
        # main.py 실행 명령어 구성
        cmd = [
            'python', 'main.py',
            '--model', args.model,
            '--image_days', str(args.image_days),
            '--pred_days', str(args.pred_days),
            '--batch_size', str(args.batch_size),
            '--epochs', str(args.epochs),
            '--lr', str(args.lr)
        ]
        
        if args.use_original_format:
            cmd.append('--use_original_format')
        
        try:
            # 독립적인 학습 실행 (별도의 랜덤 시드로)
            print(f"실행 명령어: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=False)
            
            # 생성된 모델 파일을 앙상블용으로 이름 변경
            original_model = f"models/{args.model}_I{args.image_days}R{args.pred_days}.tar"
            if os.path.exists(original_model):
                os.rename(original_model, model_file)
                print(f"✅ 모델 저장: {model_file}")
                successful_runs += 1
            else:
                print(f"❌ 모델 파일을 찾을 수 없습니다: {original_model}")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ 앙상블 실행 {run_idx} 실패: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"🎯 앙상블 학습 완료")
    print(f"   성공한 실행: {successful_runs}/{args.ensemble_runs}")
    print(f"   저장된 모델들:")
    
    # 생성된 모델 파일들 확인
    for run_idx in range(1, args.ensemble_runs + 1):
        model_name = f"{args.model}_I{args.image_days}R{args.pred_days}_run{run_idx}"
        model_file = f"models/{model_name}.tar"
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file) / (1024**2)
            print(f"     ✅ {model_name}.tar ({file_size:.1f}MB)")
    
    if successful_runs >= 1:
        print(f"\n🚀 앙상블 모델 준비 완료!")
        print(f"이제 ensemble_test.py로 앙상블 예측을 수행하세요:")
        print(f"python ensemble_test.py --model {args.model} --image_days {args.image_days} --pred_days {args.pred_days}" + 
              (" --use_original_format" if args.use_original_format else ""))
    else:
        print(f"❌ 앙상블 학습 실패")

if __name__ == '__main__':
    main()