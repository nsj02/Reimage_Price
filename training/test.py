#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test.py - CNN model evaluation with optional ensemble support

Evaluate single model or ensemble models with decile portfolio backtesting

Usage:
    # Single model evaluation
    python test.py --model CNN5d --image_days 5 --pred_days 5
    
    # Ensemble evaluation (paper method: average 5 independent predictions)
    python test.py --model CNN5d --image_days 5 --pred_days 5 --ensemble
"""

from __init__ import *
import model as _M
import dataset as _D
import argparse
import numpy as np
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_single_model(model_class, model_name):
    """
    Load a single trained model
    """
    model_file = f"models/{model_name}.tar"
    
    if not os.path.exists(model_file):
        print(f"Model file not found: {model_file}")
        return None
    
    try:
        model = model_class()
        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()
        model.to(device)
        print(f"Model loaded: {model_file}")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None


def load_ensemble_models(model_class, model_base_name, num_models=5):
    """
    Load ensemble models and return as list
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
                print(f"Ensemble model {run_idx} loaded: {model_file}")
            except Exception as e:
                print(f"Failed to load ensemble model {run_idx}: {e}")
        else:
            print(f"Ensemble model file not found: {model_file}")
    
    print(f"Total {loaded_count} ensemble models loaded")
    return models

def single_model_predict(model, test_loader):
    """
    Perform prediction with a single model
    """
    predictions = []
    
    print(f"Single model prediction...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Single Model Predicting")):
            images, label_5, label_20, label_60, ret5, ret20, ret60 = batch_data
            
            # Send images to device
            images = images.float().to(device)  # [batch, H, W]
            
            # Model prediction (BCE method)
            output = model(images)  # Already has Sigmoid applied
            up_probs = output.squeeze().cpu().numpy()  # Up probability
            
            # Store batch results
            batch_start_idx = batch_idx * test_loader.batch_size
            for i in range(len(up_probs)):
                predictions.append({
                    'image_id': batch_start_idx + i,
                    'up_prob': up_probs[i],
                    'actual_label_5': label_5[i].item(),
                    'actual_label_20': label_20[i].item(),
                    'actual_label_60': label_60[i].item(),
                    'actual_ret5': ret5[i].item(),
                    'actual_ret20': ret20[i].item(),
                    'actual_ret60': ret60[i].item()
                })
    
    return predictions


def ensemble_predict(models, test_loader):
    """
    Perform ensemble prediction by averaging multiple models
    """
    predictions = []
    num_models = len(models)
    
    if num_models == 0:
        raise ValueError("No models loaded!")
    
    print(f"Ensemble prediction ({num_models} models)...")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(test_loader, desc="Ensemble Predicting")):
            images, label_5, label_20, label_60, ret5, ret20, ret60 = batch_data
            
            # Send images to device
            images = images.float().to(device)  # [batch, H, W]
            
            # Collect predictions from all models (BCE method)
            batch_predictions = []
            for model in models:
                output = model(images)  # Already has Sigmoid applied
                up_probs = output.squeeze().cpu().numpy()  # Up probability
                batch_predictions.append(up_probs)
            
            # Ensemble average (paper method)
            ensemble_up_probs = np.mean(batch_predictions, axis=0)
            
            # Individual model predictions for debugging
            individual_probs = np.array(batch_predictions)  # [num_models, batch_size]
            
            # Store batch results
            batch_start_idx = batch_idx * test_loader.batch_size
            for i in range(len(ensemble_up_probs)):
                predictions.append({
                    'image_id': batch_start_idx + i,
                    'up_prob': ensemble_up_probs[i],
                    'individual_probs': individual_probs[:, i].tolist(),  # All model predictions
                    'prob_std': np.std(individual_probs[:, i]),  # Model agreement
                    'actual_label_5': label_5[i].item(),
                    'actual_label_20': label_20[i].item(),
                    'actual_label_60': label_60[i].item(),
                    'actual_ret5': ret5[i].item(),
                    'actual_ret20': ret20[i].item(),
                    'actual_ret60': ret60[i].item()
                })
    
    return predictions


def analyze_predictions(predictions, label_type, ensemble=False):
    """
    Analyze prediction quality and model behavior
    """
    print(f"\n{'='*60}")
    print(f"PREDICTION ANALYSIS")
    print(f"{'='*60}")
    
    # Extract relevant data
    up_probs = [pred['up_prob'] for pred in predictions]
    
    if label_type == 'RET5':
        actual_labels = [pred['actual_label_5'] for pred in predictions]
        actual_returns = [pred['actual_ret5'] for pred in predictions]
    elif label_type == 'RET20':
        actual_labels = [pred['actual_label_20'] for pred in predictions]
        actual_returns = [pred['actual_ret20'] for pred in predictions]
    else:  # RET60
        actual_labels = [pred['actual_label_60'] for pred in predictions]
        actual_returns = [pred['actual_ret60'] for pred in predictions]
    
    up_probs = np.array(up_probs)
    actual_labels = np.array(actual_labels)
    actual_returns = np.array(actual_returns)
    
    # Basic statistics
    print(f"Total predictions: {len(up_probs):,}")
    print(f"Up probability range: [{up_probs.min():.3f}, {up_probs.max():.3f}]")
    print(f"Up probability mean: {up_probs.mean():.3f}")
    print(f"Up probability std: {up_probs.std():.3f}")
    print(f"Actual up ratio: {actual_labels.mean():.3f}")
    
    # Classification accuracy
    predicted_labels = (up_probs > 0.5).astype(int)
    accuracy = (predicted_labels == actual_labels).mean()
    print(f"Classification accuracy: {accuracy:.3f}")
    
    # Extreme predictions analysis
    extreme_high = (up_probs > 0.9).sum()
    extreme_low = (up_probs < 0.1).sum()
    moderate = ((up_probs >= 0.4) & (up_probs <= 0.6)).sum()
    
    print(f"Extreme predictions:")
    print(f"  Very confident up (>0.9): {extreme_high} ({extreme_high/len(up_probs)*100:.1f}%)")
    print(f"  Very confident down (<0.1): {extreme_low} ({extreme_low/len(up_probs)*100:.1f}%)")
    print(f"  Uncertain (0.4-0.6): {moderate} ({moderate/len(up_probs)*100:.1f}%)")
    
    # Decile analysis
    print(f"\nProbability Decile Analysis:")
    print(f"{'Decile':<8}{'Count':<8}{'Avg_Prob':<10}{'Up_Ratio':<10}{'Avg_Return':<12}")
    print(f"{'-'*52}")
    
    # Sort by probability and create deciles
    sorted_indices = np.argsort(up_probs)
    decile_size = len(sorted_indices) // 10
    
    for decile in range(10):
        start_idx = decile * decile_size
        if decile == 9:  # Last decile gets remainder
            end_idx = len(sorted_indices)
        else:
            end_idx = (decile + 1) * decile_size
        
        decile_indices = sorted_indices[start_idx:end_idx]
        decile_probs = up_probs[decile_indices]
        decile_labels = actual_labels[decile_indices]
        decile_returns = actual_returns[decile_indices]
        
        avg_prob = decile_probs.mean()
        up_ratio = decile_labels.mean()
        avg_return = decile_returns.mean()
        
        print(f"{decile+1:<8}{len(decile_indices):<8}{avg_prob:<10.3f}{up_ratio:<10.3f}{avg_return*100:<12.2f}")
    
    # Ensemble-specific analysis
    if ensemble and 'individual_probs' in predictions[0]:
        print(f"\nEnsemble Model Agreement Analysis:")
        prob_stds = [pred['prob_std'] for pred in predictions]
        prob_stds = np.array(prob_stds)
        
        print(f"Model agreement (1-std): {1-prob_stds.mean():.3f}")
        print(f"High disagreement (std>0.2): {(prob_stds > 0.2).sum()} ({(prob_stds > 0.2).mean()*100:.1f}%)")
        
        # Show some examples of high/low agreement
        high_agreement_idx = np.argmin(prob_stds)
        low_agreement_idx = np.argmax(prob_stds)
        
        print(f"\nExample predictions:")
        print(f"High agreement (std={prob_stds[high_agreement_idx]:.3f}):")
        print(f"  Individual: {predictions[high_agreement_idx]['individual_probs']}")
        print(f"  Ensemble: {predictions[high_agreement_idx]['up_prob']:.3f}")
        print(f"  Actual: {actual_labels[high_agreement_idx]} (return: {actual_returns[high_agreement_idx]*100:.2f}%)")
        
        print(f"Low agreement (std={prob_stds[low_agreement_idx]:.3f}):")
        print(f"  Individual: {predictions[low_agreement_idx]['individual_probs']}")
        print(f"  Ensemble: {predictions[low_agreement_idx]['up_prob']:.3f}")
        print(f"  Actual: {actual_labels[low_agreement_idx]} (return: {actual_returns[low_agreement_idx]*100:.2f}%)")
    
    print(f"{'='*60}")


def decile_portfolio_backtest(model_class, label_type, model_name, image_days, pred_days, use_original_format=False, ensemble=False, num_models=5, data_version='reconstructed'):
    """
    Decile portfolio backtesting with single model or ensemble
    """
    
    if ensemble:
        print(f"Ensemble backtesting started")
        print(f"   Model: {model_name}")
        print(f"   Ensemble models: {num_models}")
        print(f"   Label type: {label_type}")
        
        # Load ensemble models
        models = load_ensemble_models(model_class, model_name, num_models)
        
        if len(models) == 0:
            print("No ensemble models loaded. Run train.py with --ensemble first.")
            return None
    else:
        print(f"Single model backtesting started")
        print(f"   Model: {model_name}")
        print(f"   Label type: {label_type}")
        
        # Load single model
        model = load_single_model(model_class, model_name)
        
        if model is None:
            print("No model loaded. Run train.py first.")
            return None
    
    # Load test dataset
    if use_original_format:
        print(f"Loading original format test dataset...")
        import dataset
        test_dataset = dataset.OriginalFormatDataset(
            win_size=image_days,
            mode='test',
            label_type=label_type,
            data_version=data_version
        )
        if test_dataset is None:
            print(f"Original format test data not available.")
            return None
    else:
        print("HDF5 format is no longer supported. Please use --use_original_format.")
        return None
    
    # Test data loader
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False  # Large batch size for prediction only
    )
    
    print(f"Test data: {len(test_dataset):,} samples")
    
    # Perform prediction
    if ensemble:
        predictions = ensemble_predict(models, test_loader)
    else:
        predictions = single_model_predict(model, test_loader)
    
    print(f"Prediction completed: {len(predictions):,} samples")
    
    # Analyze prediction quality
    analyze_predictions(predictions, label_type, ensemble)
    
    # Convert to DataFrame for portfolio backtesting
    df_predictions = []
    for pred in predictions:
        # Select actual label and return based on label type
        if label_type == 'RET5':
            actual_label = pred['actual_label_5']
            actual_return = pred['actual_ret5']
        elif label_type == 'RET20':
            actual_label = pred['actual_label_20']
            actual_return = pred['actual_ret20']
        else:  # RET60
            actual_label = pred['actual_label_60']
            actual_return = pred['actual_ret60']
        
        # Add date/stock information (from original dataset)
        image_id = pred['image_id']
        if hasattr(test_dataset, 'labels_df') and image_id < len(test_dataset.labels_df):
            date = test_dataset.labels_df.iloc[image_id]['Date']
            stock_id = test_dataset.labels_df.iloc[image_id]['StockID']
        else:
            date = image_id  # Use image_id as temporary date
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
    print(f"Prediction DataFrame: {len(df):,} samples")
    
    # Calculate decile performance
    portfolio_performance = calculate_decile_performance(df, pred_days)
    
    return portfolio_performance


def calculate_decile_performance(df, pred_days):
    """
    Paper-style decile portfolio performance calculation
    
    Args:
        df: Prediction results DataFrame
        pred_days: Prediction period (rebalancing frequency)
    """
    
    # Build decile portfolios by date
    daily_returns = []
    portfolio_weights = {}  # For turnover calculation
    
    dates = sorted(df['date'].unique())
    
    for i, date in enumerate(dates):
        day_data = df[df['date'] == date].copy()
        
        if len(day_data) < 100:  # Minimum number of stocks
            continue
            
        # Classify into deciles by up probability
        day_data = day_data.sort_values('up_prob', ascending=True)
        n_stocks = len(day_data)
        decile_size = n_stocks // 10
        
        decile_returns = []
        current_weights = {}
        
        for decile in range(1, 11):
            start_idx = (decile - 1) * decile_size
            if decile == 10:  # Last decile takes all remaining stocks
                end_idx = n_stocks
            else:
                end_idx = decile * decile_size
            
            decile_stocks = day_data.iloc[start_idx:end_idx]
            decile_return = decile_stocks['actual_return'].mean()
            decile_returns.append(decile_return)
            
            # Debug: Show decile info (first date only)
            if i == 0:  # Only for first date
                prob_range = f"{decile_stocks['up_prob'].min():.4f}~{decile_stocks['up_prob'].max():.4f}"
                print(f"Decile {decile}: Prob range {prob_range}, Avg return {decile_return:.4f} ({decile_return*100:.2f}%)")
            
            # Save portfolio weights (equal weight)
            for _, stock in decile_stocks.iterrows():
                weight = 1.0 / len(decile_stocks)
                if decile == 10:  # Long (high up probability)
                    current_weights[stock['stock_id']] = weight
                elif decile == 1:  # Short (low up probability)
                    current_weights[stock['stock_id']] = -weight
                else:
                    current_weights[stock['stock_id']] = 0
        
        # Long-Short returns (Decile 10 Long - Decile 1 Short)
        long_return = decile_returns[9]     # Decile 10 Long position (high up probability)  
        short_actual_return = decile_returns[0]  # Decile 1 actual return (low up probability)
        short_return = short_actual_return       # Short position return (we'll subtract this)
        ls_return = long_return - short_return   # Long - Short
        
        daily_returns.append({
            'date': date,
            'decile_returns': decile_returns,
            'long_return': long_return,
            'short_return': short_return,
            'ls_return': ls_return
        })
        
        # Save weights
        portfolio_weights[date] = current_weights
    
    if len(daily_returns) == 0:
        return None
    
    returns_df = pd.DataFrame(daily_returns)
    
    # Calculate paper-style performance metrics
    results = {}
    
    # Basic statistics
    mean_ls_return = returns_df['ls_return'].mean()
    std_ls_return = returns_df['ls_return'].std()
    
    # Annualization (paper method)
    if pred_days == 5:  # Weekly
        annual_factor = 52
    elif pred_days == 20:  # Monthly
        annual_factor = 12  
    elif pred_days == 60:  # Quarterly
        annual_factor = 4
    else:
        annual_factor = 252 / pred_days
    
    annual_return = mean_ls_return * annual_factor
    annual_vol = std_ls_return * np.sqrt(annual_factor)
    sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Decile-wise performance
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
    
    # Turnover calculation (paper formula)
    turnover = calculate_monthly_turnover(portfolio_weights, daily_returns, pred_days)
    
    results = {
        'ls_sharpe_ratio': sharpe_ratio,
        'ls_annual_return': annual_return,
        'ls_annual_vol': annual_vol,
        'long_annual_return': decile_stats[9]['annual_return'],   # Decile 10 = Long (high prob)
        'long_sharpe_ratio': decile_stats[9]['sharpe_ratio'],
        'short_annual_return': decile_stats[0]['annual_return'], # Decile 1 = Short (low prob)  
        'short_sharpe_ratio': decile_stats[0]['sharpe_ratio'],
        'monthly_turnover': turnover,
        'total_periods': len(returns_df),
        'decile_performance': decile_stats
    }
    
    return results


def calculate_monthly_turnover(portfolio_weights, daily_returns, pred_days):
    """
    Calculate monthly turnover according to paper formula:
    Turnover = (1/M) * Σ|w_{i,t+1} - w_{i,t} * (1+r_{i,t+1})| / 2
    """
    
    if len(portfolio_weights) < 2:
        return 0
        
    dates = sorted(portfolio_weights.keys())
    turnovers = []
    
    # Create dictionary for indexing daily_returns by date
    returns_by_date = {}
    for day_return in daily_returns:
        date = day_return['date']
        decile_returns = day_return['decile_returns']
        returns_by_date[date] = decile_returns
    
    for i in range(1, len(dates)):
        prev_date = dates[i-1]
        curr_date = dates[i]
        
        prev_weights = portfolio_weights[prev_date]
        curr_weights = portfolio_weights[curr_date]
        
        # Previous date returns (returns until next rebalancing)
        prev_returns = returns_by_date.get(prev_date, [0] * 10)
        
        # Find common stocks
        all_codes = set(prev_weights.keys()) | set(curr_weights.keys())
        
        turnover_sum = 0
        for code in all_codes:
            w_prev = prev_weights.get(code, 0)
            w_curr = curr_weights.get(code, 0)
            
            # Use actual returns (approximate with decile average returns)
            if w_prev > 0:  # Long position (Decile 10)
                return_rate = prev_returns[9] if len(prev_returns) > 9 else 0
            elif w_prev < 0:  # Short position (Decile 1)
                return_rate = prev_returns[0] if len(prev_returns) > 0 else 0
            else:
                return_rate = 0
            
            drift_weight = w_prev * (1 + return_rate)
            turnover_sum += abs(w_curr - drift_weight)
        
        turnovers.append(turnover_sum / 2)
    
    # Return monthly turnover (not annualized)
    monthly_turnover = np.mean(turnovers)
    
    return monthly_turnover


def main():
    parser = argparse.ArgumentParser(description='CNN model evaluation with optional ensemble support')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['CNN5d', 'CNN20d', 'CNN60d'],
                       help='Model type')
    parser.add_argument('--image_days', type=int, required=True,
                       choices=[5, 20, 60],
                       help='Image window size (days)')
    parser.add_argument('--pred_days', type=int, required=True,
                       choices=[5, 20, 60], 
                       help='Prediction period (days)')
    parser.add_argument('--ensemble', action='store_true',
                       help='Use ensemble evaluation (average multiple models)')
    parser.add_argument('--num_models', type=int, default=5,
                       help='Number of ensemble models (default: 5)')
    parser.add_argument('--use_original_format', action='store_true',
                       help='Use original format (.dat + .feather)')
    parser.add_argument('--data_version', type=str, default='reconstructed',
                       choices=['original_author', 'reconstructed', 'filled'],
                       help='Data version: original_author (img_data), reconstructed (img_data_reconstructed), filled (img_data_reconstructed_filled)')
    
    args = parser.parse_args()
    
    # Model class mapping
    model_classes = {
        'CNN5d': _M.CNN5d,
        'CNN20d': _M.CNN20d, 
        'CNN60d': _M.CNN60d
    }
    
    model_class = model_classes[args.model]
    label_type = f'RET{args.pred_days}'
    
    # Add data version to model name
    # Model naming with data version distinction
    if args.data_version == 'original_author':
        version_suffix = "_original_author"
    elif args.data_version == 'filled':
        version_suffix = "_filled"
    else:  # 'reconstructed'
        version_suffix = "_reconstructed"
    
    base_model_name = f"{args.model}_I{args.image_days}R{args.pred_days}{version_suffix}"
    
    if args.ensemble:
        model_name = base_model_name
        result_file = f"results/{base_model_name}_ensemble_performance.json"
        print_prefix = "Ensemble"
    else:
        model_name = base_model_name
        result_file = f"results/{base_model_name}_single_performance.json"
        print_prefix = "Single Model"
    
    # Perform backtesting
    results = decile_portfolio_backtest(
        model_class=model_class,
        label_type=label_type,
        model_name=model_name,
        image_days=args.image_days,
        pred_days=args.pred_days,
        use_original_format=args.use_original_format,
        ensemble=args.ensemble,
        num_models=args.num_models,
        data_version=args.data_version
    )
    
    if results:
        print(f"\n{'='*60}")
        print(f"{print_prefix} Decile Portfolio Performance ({model_name})")
        print(f"{'='*60}")
        print(f"Long-Short Sharpe Ratio:  {results['ls_sharpe_ratio']:.2f}")
        print(f"Long-Short Annual Return: {results['ls_annual_return']:.4f} ({results['ls_annual_return']*100:.2f}%)")
        print(f"Long-Short Annual Vol:    {results['ls_annual_vol']:.4f} ({results['ls_annual_vol']*100:.2f}%)")
        print(f"")
        print(f"Long (Decile 1) Performance:")
        print(f"  Annual Return:          {results['long_annual_return']:.4f} ({results['long_annual_return']*100:.2f}%)")
        print(f"  Sharpe Ratio:           {results['long_sharpe_ratio']:.2f}")
        print(f"")
        print(f"Short (Decile 10) Performance:")
        print(f"  Annual Return:          {results['short_annual_return']:.4f} ({results['short_annual_return']*100:.2f}%)")
        print(f"  Sharpe Ratio:           {results['short_sharpe_ratio']:.2f}")
        print(f"")
        print(f"Monthly Turnover:         {results['monthly_turnover']:.1f}%")
        print(f"Total Rebalancing Periods: {results['total_periods']:,}")
        if args.ensemble:
            print(f"Ensemble Models:          {args.num_models}")
        print(f"{'='*60}")
        
        # Detailed decile performance
        print(f"\nDecile Performance:")
        print(f"{'Decile':<8}{'Annual Return':<15}{'Sharpe':<8}")
        print(f"{'-'*31}")
        for decile_stat in results['decile_performance']:
            print(f"{decile_stat['decile']:<8}{decile_stat['annual_return']*100:>11.2f}%{decile_stat['sharpe_ratio']:>8.2f}")
        
        # Save results
        os.makedirs('results', exist_ok=True)
        
        import json
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved: {result_file}")
    else:
        print(f"{print_prefix} backtesting failed")

if __name__ == '__main__':
    main()