#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test.py - 논문 방식 Decile 포트폴리오 성능 평가

논문에서 사용한 방식대로 구현:
1. 상승 확률로 종목을 10개 decile로 분류
2. Decile 10 (Long) vs Decile 1 (Short) 전략
3. 연간화된 Sharpe ratio, 수익률, 월간 turnover 계산

사용법:
    python test.py --model CNN5d --image_days 5 --pred_days 5
"""
from __init__ import *
import model as _M
import dataset as _D
import argparse

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def decile_portfolio_backtest(model, label_type, model_file, image_days, pred_days):
    """
    논문 방식 decile 포트폴리오 백테스팅
    """
    
    print(f"모델 로드: {model_file}")
    if not os.path.exists(model_file):
        print(f"모델 파일이 존재하지 않습니다: {model_file}")
        return None
    
    state_dict = torch.load(model_file, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()
    
    # 사전 생성된 테스트 이미지 확인
    test_image_dir = f"images/test_I{image_days}R{pred_days}"
    metadata_file = os.path.join(test_image_dir, 'metadata.csv')
    
    if not os.path.exists(metadata_file):
        print(f"❌ 사전 생성된 테스트 이미지가 없습니다: {test_image_dir}")
        print(f"다음 명령어로 테스트 이미지를 먼저 생성하세요:")
        print(f"python create_images.py --image_days {image_days} --mode test --pred_days {pred_days}")
        return None
    
    # 사전 생성된 테스트 이미지 로드
    print(f"사전 생성된 테스트 이미지 로드 중: {test_image_dir}")
    test_dataset = _D.PrecomputedImageDataset(
        image_dir=test_image_dir,
        label_type=label_type
    )
    
    print(f"로드된 테스트 이미지: {len(test_dataset):,}개")
    
    # DataLoader로 배치 단위 예측
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=128, shuffle=False
    )
    
    predictions = []
    
    print(f"모델 예측 수행 중...")
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Predicting")):
            images, label_5, label_20, label_60, ret5, ret20, ret60 = batch_data
            
            # GPU로 이미지 전송
            images = images.unsqueeze(1).float().to(device)  # [batch, 1, H, W]
            
            # 모델 예측
            output = torch.softmax(model(images), dim=1)
            up_probs = output[:, 1].cpu().numpy()  # 상승 확률
            
            # 라벨 선택
            if label_type == 'RET5':
                actual_labels = label_5.numpy()
                actual_returns = ret5.numpy()
            elif label_type == 'RET20':
                actual_labels = label_20.numpy()  
                actual_returns = ret20.numpy()
            else:  # RET60
                actual_labels = label_60.numpy()
                actual_returns = ret60.numpy()
            
            # 배치 결과 저장
            for i in range(len(up_probs)):
                predictions.append({
                    'image_id': batch_start_idx + i,
                    'up_prob': up_probs[i],
                    'actual_label': actual_labels[i],
                    'actual_return': actual_returns[i]
                })
    
    if len(predictions) == 0:
        print("예측 결과가 없습니다.")
        return None
        
    df = pd.DataFrame(predictions)
    print(f"총 예측 수: {len(df):,}개")
    print(f"기간: {df['date'].min()} ~ {df['date'].max()}")
    
    # Decile 포트폴리오 백테스팅
    portfolio_performance = calculate_decile_performance(df, pred_days)
    
    return portfolio_performance

def calculate_decile_performance(df, pred_days):
    """
    논문 방식 decile 포트폴리오 성과 계산
    
    Args:
        df: 예측 결과 DataFrame
        pred_days: 예측 기간 (리밸런싱 주기)
    """
    
    # 날짜별 decile 포트폴리오 구성
    daily_returns = []
    portfolio_weights = {}  # turnover 계산용
    
    dates = sorted(df['date'].unique())
    
    for i, date in enumerate(dates):
        day_data = df[df['date'] == date].copy()
        
        if len(day_data) < 100:  # 최소 종목 수
            continue
            
        # 상승 확률로 decile 분류
        day_data = day_data.sort_values('up_prob', ascending=True)
        n_stocks = len(day_data)
        decile_size = n_stocks // 10
        
        decile_returns = []
        current_weights = {}
        
        for decile in range(1, 11):
            start_idx = (decile - 1) * decile_size
            if decile == 10:  # 마지막 decile은 남은 모든 종목
                end_idx = n_stocks
            else:
                end_idx = decile * decile_size
            
            decile_stocks = day_data.iloc[start_idx:end_idx]
            decile_return = decile_stocks['actual_return'].mean()
            decile_returns.append(decile_return)
            
            # 포트폴리오 가중치 저장 (동일가중)
            for _, stock in decile_stocks.iterrows():
                weight = 1.0 / len(decile_stocks)
                if decile == 1:  # Short
                    current_weights[stock['code']] = -weight
                elif decile == 10:  # Long  
                    current_weights[stock['code']] = weight
                else:
                    current_weights[stock['code']] = 0
        
        # Long-Short 수익률 (Decile 10 - Decile 1)
        long_return = decile_returns[9]   # Decile 10
        short_return = decile_returns[0]  # Decile 1
        ls_return = long_return - short_return
        
        daily_returns.append({
            'date': date,
            'decile_returns': decile_returns,
            'long_return': long_return,
            'short_return': short_return,
            'ls_return': ls_return
        })
        
        # 가중치 저장
        portfolio_weights[date] = current_weights
    
    if len(daily_returns) == 0:
        return None
        
    returns_df = pd.DataFrame(daily_returns)
    
    # 논문 방식 성과 지표 계산
    results = {}
    
    # 기본 통계
    mean_ls_return = returns_df['ls_return'].mean()
    std_ls_return = returns_df['ls_return'].std()
    
    # 연간화 (논문에서 사용하는 방식)
    if pred_days == 5:  # 주간
        annual_factor = 52
    elif pred_days == 20:  # 월간
        annual_factor = 12  
    elif pred_days == 60:  # 분기
        annual_factor = 4
    else:
        annual_factor = 252 / pred_days
    
    annual_return = mean_ls_return * annual_factor
    annual_vol = std_ls_return * np.sqrt(annual_factor)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Decile별 성과
    decile_stats = []
    for i in range(10):
        decile_returns = [day['decile_returns'][i] for day in daily_returns]
        decile_mean = np.mean(decile_returns)
        decile_std = np.std(decile_returns)
        decile_sharpe = (decile_mean * annual_factor) / (decile_std * np.sqrt(annual_factor)) if decile_std > 0 else 0
        
        decile_stats.append({
            'decile': i + 1,
            'mean_return': decile_mean,
            'annual_return': decile_mean * annual_factor,
            'annual_vol': decile_std * np.sqrt(annual_factor),
            'sharpe_ratio': decile_sharpe
        })
    
    # Turnover 계산 (논문 공식)
    turnover = calculate_monthly_turnover(portfolio_weights, returns_df, pred_days)
    
    results = {
        'ls_sharpe_ratio': sharpe_ratio,
        'ls_annual_return': annual_return,
        'ls_annual_vol': annual_vol,
        'long_annual_return': decile_stats[9]['annual_return'],
        'long_sharpe_ratio': decile_stats[9]['sharpe_ratio'],
        'short_annual_return': decile_stats[0]['annual_return'],
        'short_sharpe_ratio': decile_stats[0]['sharpe_ratio'],
        'monthly_turnover': turnover,
        'total_periods': len(returns_df),
        'decile_performance': decile_stats
    }
    
    return results

def calculate_monthly_turnover(portfolio_weights, returns_df, pred_days):
    """
    논문 공식에 따른 월간 turnover 계산:
    Turnover = (1/M) * Σ|w_{i,t+1} - w_{i,t} * (1+r_{i,t+1})| / 2
    """
    
    if len(portfolio_weights) < 2:
        return 0
        
    dates = sorted(portfolio_weights.keys())
    turnovers = []
    
    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]
        
        prev_weights = portfolio_weights[prev_date]
        curr_weights = portfolio_weights[curr_date]
        
        # 공통 종목들 찾기
        all_codes = set(prev_weights.keys()) | set(curr_weights.keys())
        
        turnover_sum = 0
        for code in all_codes:
            w_prev = prev_weights.get(code, 0)
            w_curr = curr_weights.get(code, 0)
            
            # 수익률은 실제로는 필요하지만 여기서는 간단히 0으로 가정
            # 실제 구현에서는 해당 종목의 실제 수익률을 사용해야 함
            return_rate = 0  # 임시
            
            drift_weight = w_prev * (1 + return_rate)
            turnover_sum += abs(w_curr - drift_weight)
        
        turnovers.append(turnover_sum / 2)
    
    # 월간으로 스케일링
    if pred_days == 5:  # 주간 -> 월간 (4주)
        monthly_turnover = np.mean(turnovers) * 4
    elif pred_days == 20:  # 이미 월간
        monthly_turnover = np.mean(turnovers)
    elif pred_days == 60:  # 분기 -> 월간 (1/3)
        monthly_turnover = np.mean(turnovers) / 3
    else:
        monthly_turnover = np.mean(turnovers)
    
    return monthly_turnover

def main():
    parser = argparse.ArgumentParser(description='CNN Decile 포트폴리오 성과 평가')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['CNN5d', 'CNN20d', 'CNN60d'],
                       help='모델 타입')
    parser.add_argument('--image_days', type=int, required=True,
                       choices=[5, 20, 60],
                       help='이미지 윈도우 크기')
    parser.add_argument('--pred_days', type=int, required=True,
                       choices=[5, 20, 60], 
                       help='예측 기간')
    
    args = parser.parse_args()
    
    print(f"Decile 포트폴리오 백테스팅: {args.model}")
    print(f"  이미지 윈도우: {args.image_days}일")
    print(f"  예측 기간: {args.pred_days}일")
    print(f"  디바이스: {device}")
    
    # 모델 초기화
    model_name = f"{args.model}_I{args.image_days}R{args.pred_days}"
    model_file = f"models/{model_name}.tar"
    
    if args.model == 'CNN5d':
        model = _M.CNN5d()
    elif args.model == 'CNN20d':
        model = _M.CNN20d()
    elif args.model == 'CNN60d':
        model = _M.CNN60d()
    
    model.to(device)
    
    # 백테스팅 실행
    results = decile_portfolio_backtest(
        model=model,
        label_type=f'RET{args.pred_days}',
        model_file=model_file,
        image_days=args.image_days,
        pred_days=args.pred_days
    )
    
    if results:
        print(f"\n{'='*60}")
        print(f"Decile 포트폴리오 성과 ({model_name})")
        print(f"{'='*60}")
        print(f"Long-Short Sharpe Ratio:  {results['ls_sharpe_ratio']:.2f}")
        print(f"Long-Short 연간 수익률:   {results['ls_annual_return']:.4f} ({results['ls_annual_return']*100:.2f}%)")
        print(f"Long-Short 연간 변동성:   {results['ls_annual_vol']:.4f} ({results['ls_annual_vol']*100:.2f}%)")
        print(f"")
        print(f"Long (Decile 10) 성과:")
        print(f"  연간 수익률:            {results['long_annual_return']:.4f} ({results['long_annual_return']*100:.2f}%)")
        print(f"  Sharpe Ratio:          {results['long_sharpe_ratio']:.2f}")
        print(f"")
        print(f"Short (Decile 1) 성과:")
        print(f"  연간 수익률:            {results['short_annual_return']:.4f} ({results['short_annual_return']*100:.2f}%)")
        print(f"  Sharpe Ratio:          {results['short_sharpe_ratio']:.2f}")
        print(f"")
        print(f"월간 Turnover:           {results['monthly_turnover']:.1f}%")
        print(f"총 리밸런싱 기간:        {results['total_periods']:,}개")
        print(f"{'='*60}")
        
        # Decile별 상세 성과
        print(f"\nDecile별 성과:")
        print(f"{'Decile':<8}{'연간수익률':<12}{'Sharpe':<8}")
        print(f"{'-'*28}")
        for decile_stat in results['decile_performance']:
            print(f"{decile_stat['decile']:<8}{decile_stat['annual_return']*100:>8.2f}%{decile_stat['sharpe_ratio']:>8.2f}")
        
        # 결과 저장
        os.makedirs('results', exist_ok=True)
        result_file = f"results/{model_name}_decile_performance.json"
        
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n결과 저장: {result_file}")
        
    else:
        print("백테스팅 실패")

if __name__ == '__main__':
    main()