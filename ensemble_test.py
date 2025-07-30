#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ensemble_test.py - 논문 방식 5모델 앙상블 예측

논문에서 언급한 대로 5개의 독립적으로 훈련된 모델의 예측을 평균하여
최종 예측치로 사용하는 앙상블 백테스팅

사용법:
    python ensemble_test.py --model CNN5d --image_days 5 --pred_days 5
"""

from __init__ import *
import model as _M
import dataset as _D
import dataset_original as _D_ORIG
import argparse
import numpy as np
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_ensemble_models(model_class, model_base_name, num_models=5):
    """
    앙상블 모델들을 로드하여 리스트로 반환
    """
    models = []
    loaded_count = 0
    
    for run_idx in range(1, num_models + 1):
        model_file = f"models/{model_base_name}_run{run_idx}.tar"
        
        if os.path.exists(model_file):
            try:
                model = model_class()
                state_dict = torch.load(model_file, map_location=device)
                model.load_state_dict(state_dict['model_state_dict'])
                model.eval()
                model.to(device)
                models.append(model)
                loaded_count += 1
                print(f"✅ 모델 {run_idx} 로드: {model_file}")
            except Exception as e:
                print(f"❌ 모델 {run_idx} 로드 실패: {e}")
        else:
            print(f"⚠️ 모델 파일 없음: {model_file}")
    
    print(f"총 {loaded_count}개 모델 로드 완료")
    return models

def ensemble_predict(models, test_loader):
    """
    앙상블 모델들로 예측 수행 및 평균화
    """
    predictions = []
    num_models = len(models)
    
    if num_models == 0:
        raise ValueError("로드된 모델이 없습니다!")
    
    print(f"앙상블 예측 수행 중 ({num_models}개 모델)...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Ensemble Predicting")):
            images, label_5, label_20, label_60, ret5, ret20, ret60 = batch_data
            
            # GPU로 이미지 전송 (모델에서 unsqueeze 처리)
            images = images.float().to(device)  # [batch, H, W]
            
            # 모든 모델의 예측을 수집 (BCE 방식)
            batch_predictions = []
            for model in models:
                output = model(images)  # 이미 Sigmoid 적용됨
                up_probs = output.squeeze().cpu().numpy()  # 상승 확률
                batch_predictions.append(up_probs)
            
            # 앙상블 평균 (논문 방식)
            ensemble_up_probs = np.mean(batch_predictions, axis=0)
            
            # 배치 결과 저장
            batch_start_idx = batch_idx * test_loader.batch_size
            for i in range(len(ensemble_up_probs)):
                predictions.append({
                    'image_id': batch_start_idx + i,
                    'up_prob': ensemble_up_probs[i],
                    'actual_label_5': label_5[i].item(),
                    'actual_label_20': label_20[i].item(),
                    'actual_label_60': label_60[i].item(),
                    'actual_ret5': ret5[i].item(),
                    'actual_ret20': ret20[i].item(),
                    'actual_ret60': ret60[i].item()
                })
    
    return predictions

def decile_portfolio_backtest_ensemble(model_class, label_type, model_base_name, image_days, pred_days, use_original_format=False, num_models=5):
    """
    앙상블 모델을 사용한 decile 포트폴리오 백테스팅
    """
    
    print(f"🔥 앙상블 백테스팅 시작")
    print(f"   모델: {model_base_name}")
    print(f"   앙상블 모델 수: {num_models}개")
    print(f"   라벨 타입: {label_type}")
    
    # 앙상블 모델들 로드
    models = load_ensemble_models(model_class, model_base_name, num_models)
    
    if len(models) == 0:
        print("❌ 로드된 모델이 없습니다. ensemble_train.py를 먼저 실행하세요.")
        return None
    
    # 테스트 데이터셋 로드
    if use_original_format:
        print(f"원본 형식 테스트 데이터셋 로드 중...")
        test_dataset = _D_ORIG.load_original_dataset(
            win_size=image_days,
            mode='test',
            label_type=label_type
        )
        if test_dataset is None:
            print(f"원본 형식 테스트 데이터가 없습니다.")
            return None
    else:
        # 최적화된 형식
        image_dir = f"images/test_I{image_days}R{pred_days}"
        if not os.path.exists(os.path.join(image_dir, 'metadata.csv')):
            print(f"테스트 이미지가 없습니다: {image_dir}")
            return None
        test_dataset = _D.PrecomputedImageDataset(image_dir, label_type)
    
    # 테스트 데이터로더
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False  # 배치 크기 크게 (예측만)
    )
    
    print(f"테스트 데이터: {len(test_dataset):,}개")
    
    # 앙상블 예측 수행
    predictions = ensemble_predict(models, test_loader)
    
    print(f"예측 완료: {len(predictions):,}개")
    
    # DataFrame으로 변환하여 포트폴리오 백테스팅 수행
    df_predictions = []
    for pred in predictions:
        # 라벨 타입에 따라 해당하는 실제 라벨과 수익률 선택
        if label_type == 'RET5':
            actual_label = pred['actual_label_5']
            actual_return = pred['actual_ret5']
        elif label_type == 'RET20':
            actual_label = pred['actual_label_20']
            actual_return = pred['actual_ret20']
        else:  # RET60
            actual_label = pred['actual_label_60']
            actual_return = pred['actual_ret60']
        
        # 날짜/종목 정보 추가 (원본 데이터셋에서 가져오기)
        image_id = pred['image_id']
        if hasattr(test_dataset, 'labels_df') and image_id < len(test_dataset.labels_df):
            date = test_dataset.labels_df.iloc[image_id]['Date']
            stock_id = test_dataset.labels_df.iloc[image_id]['StockID']
        else:
            date = image_id  # 임시로 image_id 사용
            stock_id = f'stock_{image_id}'
        
        df_predictions.append({
            'image_id': image_id,
            'date': date,
            'stock_id': stock_id,
            'up_prob': pred['up_prob'],
            'actual_label': actual_label,
            'actual_return': actual_return
        })
    
    df = pd.DataFrame(df_predictions)
    print(f"앙상블 예측 DataFrame: {len(df):,}개")
    
    # test.py의 calculate_decile_performance 함수 재사용
    from test import calculate_decile_performance
    portfolio_performance = calculate_decile_performance(df, pred_days)
    
    return portfolio_performance

def main():
    parser = argparse.ArgumentParser(description='CNN 앙상블 모델 평가')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['CNN5d', 'CNN20d', 'CNN60d'],
                       help='모델 타입')
    parser.add_argument('--image_days', type=int, required=True,
                       choices=[5, 20, 60],
                       help='이미지 윈도우 크기 (일)')
    parser.add_argument('--pred_days', type=int, required=True,
                       choices=[5, 20, 60], 
                       help='예측 기간 (일)')
    parser.add_argument('--num_models', type=int, default=5,
                       help='앙상블 모델 수 (default: 5개)')
    parser.add_argument('--use_original_format', action='store_true',
                       help='원본 형식 (.dat + .feather) 사용')
    
    args = parser.parse_args()
    
    # 모델 클래스 매핑
    model_classes = {
        'CNN5d': _M.CNN5d,
        'CNN20d': _M.CNN20d, 
        'CNN60d': _M.CNN60d
    }
    
    model_class = model_classes[args.model]
    model_base_name = f"{args.model}_I{args.image_days}R{args.pred_days}"
    label_type = f'RET{args.pred_days}'
    
    # 앙상블 백테스팅 수행
    results = decile_portfolio_backtest_ensemble(
        model_class=model_class,
        label_type=label_type,
        model_base_name=model_base_name,
        image_days=args.image_days,
        pred_days=args.pred_days,
        use_original_format=args.use_original_format,
        num_models=args.num_models
    )
    
    if results:
        print(f"\n{'='*60}")
        print(f"🎉 앙상블 Decile 포트폴리오 성과 ({model_base_name})")
        print(f"{'='*60}")
        print(f"Long-Short Sharpe Ratio:  {results['ls_sharpe_ratio']:.2f}")
        print(f"Long-Short 연간 수익률:   {results['ls_annual_return']:.4f} ({results['ls_annual_return']*100:.2f}%)")
        print(f"Long-Short 연간 변동성:   {results['ls_annual_vol']:.4f} ({results['ls_annual_vol']*100:.2f}%)")
        print(f"")
        print(f"Long (Decile 1) 성과:")
        print(f"  연간 수익률:            {results['long_annual_return']:.4f} ({results['long_annual_return']*100:.2f}%)")
        print(f"  Sharpe Ratio:          {results['long_sharpe_ratio']:.2f}")
        print(f"")
        print(f"Short (Decile 10) 성과:")
        print(f"  연간 수익률:            {results['short_annual_return']:.4f} ({results['short_annual_return']*100:.2f}%)")
        print(f"  Sharpe Ratio:          {results['short_sharpe_ratio']:.2f}")
        print(f"")
        print(f"월간 Turnover:           {results['monthly_turnover']:.1f}%")
        print(f"총 리밸런싱 기간:        {results['total_periods']:,}개")
        print(f"앙상블 모델 수:          {args.num_models}개")
        print(f"{'='*60}")
        
        # Decile별 상세 성과
        print(f"\nDecile별 성과:")
        print(f"{'Decile':<8}{'연간수익률':<12}{'Sharpe':<8}")
        print(f"{'-'*28}")
        for decile_stat in results['decile_performance']:
            print(f"{decile_stat['decile']:<8}{decile_stat['annual_return']*100:>8.2f}%{decile_stat['sharpe_ratio']:>8.2f}")
        
        # 결과 저장
        os.makedirs('results', exist_ok=True)
        result_file = f"results/{model_base_name}_ensemble_performance.json"
        
        import json
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n결과 저장: {result_file}")
    else:
        print("❌ 앙상블 백테스팅 실패")

if __name__ == '__main__':
    main()