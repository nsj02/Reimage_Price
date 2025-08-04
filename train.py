#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py - CNN model training script

Trains CNN models for stock price prediction using candlestick chart images.
Supports both single model training and ensemble training.

Usage:
    # Single model training
    python train.py --model CNN5d --image_days 5 --pred_days 5
    
    # Ensemble training (5 models)
    python train.py --model CNN5d --image_days 5 --pred_days 5 --ensemble --ensemble_runs 5
"""

from __init__ import *
import model as _M
import dataset as _D
import argparse
import os
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_single_model(args):
    """Train a single CNN model"""
    
    print(f"Training {args.model} model")
    print(f"  Image window: {args.image_days} days")
    print(f"  Prediction period: {args.pred_days} days")
    print(f"  Data version: {args.data_version}")
    print(f"  Device: {device}")
    
    # Model filename with data version
    version_suffix = "_filled" if args.data_version == 'filled' else ""
    model_name = f"{args.model}_I{args.image_days}R{args.pred_days}{version_suffix}"
    model_file = f"models/{model_name}.tar"
    
    # Check if model already exists
    if os.path.exists(model_file) and not args.force_retrain:
        print(f"Model already exists: {model_file}")
        print("Use --force_retrain to retrain")
        return
    
    # Load dataset
    if args.use_original_format:
        print(f"Loading original format dataset...")
        train_dataset = _D.load_original_dataset(args.image_days, 'train', f'RET{args.pred_days}', args.data_version)
        if train_dataset is None:
            print(f"Please generate original format images first:")
            print(f"python datageneration.py --image_days {args.image_days} --mode train --data_version {args.data_version}")
            return
    else:
        print(f"Loading optimized format dataset...")
        # For now, use original format as default
        train_dataset = _D.load_original_dataset(args.image_days, 'train', f'RET{args.pred_days}', args.data_version)
        if train_dataset is None:
            print(f"Please generate images first:")
            print(f"python datageneration.py --image_days {args.image_days} --mode train --data_version {args.data_version}")
            return
    
    # Train/validation split
    train_size = int(0.7 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, valid_size]
    )
    
    # Data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )
    
    print(f"Training data: {len(train_dataset):,}")
    print(f"Validation data: {len(valid_dataset):,}")
    
    # Initialize model
    print(f"Initializing {args.model} model...")
    model = getattr(_M, args.model)().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Training setup
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training loop
    print(f"Starting training (max {args.epochs} epochs)...")
    
    best_valid_acc = 0.0
    patience_counter = 0
    train_losses, valid_losses = [], []
    train_accs, valid_accs = [], []
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, batch_data in enumerate(train_loader):
            if len(batch_data) == 7:  # [image, label_5, label_20, label_60, ret5, ret20, ret60]
                images, label_5, label_20, label_60, ret5, ret20, ret60 = batch_data
                
                # Select appropriate label based on prediction days
                if args.pred_days == 5:
                    labels = label_5
                elif args.pred_days == 20:
                    labels = label_20
                else:  # 60
                    labels = label_60
            else:
                print(f"Unexpected batch format: {len(batch_data)} elements")
                continue
            
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_acc = train_correct / train_total
        train_losses.append(train_loss / len(train_loader))
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for batch_data in valid_loader:
                if len(batch_data) == 7:
                    images, label_5, label_20, label_60, ret5, ret20, ret60 = batch_data
                    
                    if args.pred_days == 5:
                        labels = label_5
                    elif args.pred_days == 20:
                        labels = label_20
                    else:
                        labels = label_60
                else:
                    continue
                
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                valid_loss += loss.item()
                predicted = (outputs > 0.5).float()
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
        
        valid_acc = valid_correct / valid_total
        valid_losses.append(valid_loss / len(valid_loader))
        valid_accs.append(valid_acc)
        
        print(f"Epoch {epoch+1:3d}: Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_acc:.4f}, "
              f"Valid Loss: {valid_losses[-1]:.4f}, Valid Acc: {valid_acc:.4f}")
        
        # Early stopping
        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            patience_counter = 0
            
            # Save best model
            os.makedirs('models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'valid_acc': valid_acc,
                'train_losses': train_losses,
                'valid_losses': valid_losses,
                'train_accs': train_accs,
                'valid_accs': valid_accs,
                'args': vars(args)
            }, model_file)
        else:
            patience_counter += 1
            if patience_counter >= 2:  # Early stopping after 2 epochs
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"Training completed!")
    print(f"  Model: {model_file}")
    print(f"  Best validation accuracy: {best_valid_acc:.4f}")
    

def train_ensemble(args):
    """Train ensemble of models"""
    
    print(f"Training ensemble of {args.ensemble_runs} {args.model} models")
    
    successful_runs = 0
    for run_idx in range(1, args.ensemble_runs + 1):
        print(f"\n{'='*70}")
        print(f"Ensemble run {run_idx}/{args.ensemble_runs}")
        print(f"{'='*70}")
        
        # Modify args for this run
        run_args = argparse.Namespace(**vars(args))
        
        # Set random seed for each run
        torch.manual_seed(42 + run_idx)
        np.random.seed(42 + run_idx)
        
        # Create ensemble-specific model name with data version
        version_suffix = "_filled" if args.data_version == 'filled' else ""
        model_name = f"{args.model}_I{args.image_days}R{args.pred_days}{version_suffix}_run{run_idx}"
        ensemble_model_file = f"models/{model_name}.tar"
        
        if os.path.exists(ensemble_model_file) and not args.force_retrain:
            print(f"Ensemble model already exists: {ensemble_model_file}")
            successful_runs += 1
            continue
        
        # Train single model for this ensemble run
        try:
            # Temporarily change force_retrain for ensemble training
            run_args.force_retrain = True
            train_single_model(run_args)
            
            # Rename the generated model file
            original_model = f"models/{args.model}_I{args.image_days}R{args.pred_days}{version_suffix}.tar"
            if os.path.exists(original_model):
                os.rename(original_model, ensemble_model_file)
                print(f"Ensemble model saved: {ensemble_model_file}")
                successful_runs += 1
            else:
                print(f"Failed to find model file: {original_model}")
                
        except Exception as e:
            print(f"Ensemble run {run_idx} failed: {e}")
            continue
    
    print(f"\nEnsemble training completed")
    print(f"  Successful runs: {successful_runs}/{args.ensemble_runs}")
    
    if successful_runs >= 1:
        print(f"Ensemble models ready!")
        print(f"Now evaluate with: python test.py --model {args.model} --image_days {args.image_days} --pred_days {args.pred_days} --ensemble")


def main():
    parser = argparse.ArgumentParser(description='CNN model training')
    parser.add_argument('--model', type=str, required=True,
                       choices=['CNN5d', 'CNN20d', 'CNN60d'],
                       help='Model type')
    parser.add_argument('--image_days', type=int, required=True,
                       choices=[5, 20, 60],
                       help='Image window size (days)')
    parser.add_argument('--pred_days', type=int, required=True,
                       choices=[5, 20, 60],
                       help='Prediction period (days)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate (default: 1e-5)')
    parser.add_argument('--use_original_format', action='store_true',
                       help='Use original format (.dat + .feather)')
    parser.add_argument('--data_version', type=str, default='original',
                       choices=['original', 'filled'],
                       help='Data version: original (with missing values) or filled (missing values filled)')
    parser.add_argument('--ensemble', action='store_true',
                       help='Train ensemble of models')
    parser.add_argument('--ensemble_runs', type=int, default=5,
                       help='Number of ensemble runs (default: 5)')
    parser.add_argument('--force_retrain', action='store_true',
                       help='Force retrain even if model exists')
    
    args = parser.parse_args()
    
    if args.ensemble:
        train_ensemble(args)
    else:
        train_single_model(args)


if __name__ == '__main__':
    main()