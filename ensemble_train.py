#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ensemble_train.py - Paper-style 5-model ensemble training

Implements ensemble method by training the same model 5 times independently
to reduce variability of stochastic optimization as mentioned in the paper

사용법:
    python ensemble_train.py --model CNN5d --image_days 5 --pred_days 5 --ensemble_runs 5
"""

import subprocess
import os
import argparse

def main():

    parser = argparse.ArgumentParser(description='CNN ensemble model training')
    parser.add_argument('--model', type=str, required=True, 
                       choices=['CNN5d', 'CNN20d', 'CNN60d'],
                       help='Model type')
    parser.add_argument('--image_days', type=int, required=True,
                       choices=[5, 20, 60],
                       help='Image window size (days)')
    parser.add_argument('--pred_days', type=int, required=True,
                       choices=[5, 20, 60], 
                       help='Prediction period (days)')
    parser.add_argument('--ensemble_runs', type=int, default=5,
                       help='Number of ensemble runs (paper: 5 runs, default: 5)')
    parser.add_argument('--use_original_format', action='store_true',
                       help='Use original format (.dat + .feather)')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Maximum epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-5,
                       help='Learning rate (default: 1e-5)')
    
    args = parser.parse_args()
    
    print(f"Paper-style ensemble training started")
    print(f"   Model: {args.model}")
    print(f"   Ensemble runs: {args.ensemble_runs} times")
    print(f"   Image window: {args.image_days} days")
    print(f"   Prediction period: {args.pred_days} days")
    
    # Run each ensemble iteration
    successful_runs = 0
    for run_idx in range(1, args.ensemble_runs + 1):
        print(f"\n{'='*70}")
        print(f"Ensemble run {run_idx}/{args.ensemble_runs}")
        print(f"{'='*70}")
        
        # Model filename (for ensemble)
        model_name = f"{args.model}_I{args.image_days}R{args.pred_days}_run{run_idx}"
        model_file = f"models/{model_name}.tar"
        
        # Check for already trained model
        if os.path.exists(model_file):
            print(f"Already trained model: {model_file}")
            successful_runs += 1
            continue
        
        # Construct main.py execution command
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
            # Execute independent training (with separate random seed)
            print(f"Execution command: {' '.join(cmd)}")
            result = subprocess.run(cmd, check=True, capture_output=False)
            
            # Rename generated model file for ensemble
            original_model = f"models/{args.model}_I{args.image_days}R{args.pred_days}.tar"
            if os.path.exists(original_model):
                os.rename(original_model, model_file)
                print(f"Model saved: {model_file}")
                successful_runs += 1
            else:
                print(f"Model file not found: {original_model}")
                
        except subprocess.CalledProcessError as e:
            print(f"Ensemble run {run_idx} failed: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"Ensemble training completed")
    print(f"   Successful runs: {successful_runs}/{args.ensemble_runs}")
    print(f"   Saved models:")
    
    # Check generated model files
    for run_idx in range(1, args.ensemble_runs + 1):
        model_name = f"{args.model}_I{args.image_days}R{args.pred_days}_run{run_idx}"
        model_file = f"models/{model_name}.tar"
        if os.path.exists(model_file):
            file_size = os.path.getsize(model_file) / (1024**2)
            print(f"     {model_name}.tar ({file_size:.1f}MB)")
    
    if successful_runs >= 1:
        print(f"\nEnsemble models ready!")
        print(f"Now run ensemble prediction with ensemble_test.py:")
        print(f"python ensemble_test.py --model {args.model} --image_days {args.image_days} --pred_days {args.pred_days}" + 
              (" --use_original_format" if args.use_original_format else ""))
    else:
        print(f"Ensemble training failed")

if __name__ == '__main__':
    main()