#!/usr/bin/env python
# run.py

import os
import argparse
import subprocess

def main():
    """
    Script to run the stock prediction model with commonly used configurations.
    """
    parser = argparse.ArgumentParser(description='Run Stock Prediction Model')
    
    # High-level configuration groups
    parser.add_argument('--mode', type=str, default='debug',
                        choices=['debug', 'train_test', 'full_train', 'cross_validate'],
                        help='Mode to run the model in')
    
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'inductive'],
                        help='Type of model to use')
                        
    parser.add_argument('--forecast_horizon', type=int, default=1,
                        help='Number of days to forecast ahead')
                        
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Name of the experiment')
                        
    args = parser.parse_args()
    
    # Set default experiment name if not provided
    if args.experiment_name is None:
        args.experiment_name = f"{args.model_type}_{args.forecast_horizon}day"
        if args.mode == 'debug':
            args.experiment_name = f"debug_{args.experiment_name}"
    
    # Prepare command based on mode
    if args.mode == 'debug':
        # Debug mode - small dataset for quick testing
        cmd = [
            "python", "train.py",
            "--debug",
            "--debug_days", "3",
            "--debug_stocks", "10",
            "--batch_size", "16",
            "--num_epochs", "5",
            "--window_size", "10",
            "--hidden_dim", "32",
            "--time_dim", "32",
            "--num_transformer_layers", "2",
            "--num_attention_heads", "4",
            "--experiment_name", args.experiment_name
        ]
        
        if args.model_type == 'inductive':
            cmd.append("--inductive")
            
        if args.forecast_horizon > 1:
            cmd.extend(["--forecast_horizon", str(args.forecast_horizon)])
            
    elif args.mode == 'train_test':
        # Train on part 1, test on part 2
        cmd = [
            "python", "train.py",
            "--train_part1_test_part2",
            "--batch_size", "64",
            "--num_epochs", "50",
            "--window_size", "20",
            "--hidden_dim", "64",
            "--time_dim", "64",
            "--experiment_name", args.experiment_name
        ]
        
        if args.model_type == 'inductive':
            cmd.append("--inductive")
            
        if args.forecast_horizon > 1:
            cmd.extend(["--forecast_horizon", str(args.forecast_horizon)])
            
    elif args.mode == 'full_train':
        # Train on merged dataset (both parts)
        cmd = [
            "python", "train.py",
            "--use_merged",
            "--batch_size", "128",
            "--num_epochs", "100",
            "--window_size", "20",
            "--hidden_dim", "128",
            "--time_dim", "128",
            "--experiment_name", args.experiment_name
        ]
        
        if args.model_type == 'inductive':
            cmd.append("--inductive")
            
        if args.forecast_horizon > 1:
            cmd.extend(["--forecast_horizon", str(args.forecast_horizon)])
            
    elif args.mode == 'cross_validate':
        # Cross-validation mode
        # This runs the basic model with 3-fold cross-validation
        # We'll implement this by calling the train.py script multiple times
        base_cmd = [
            "python", "train.py",
            "--batch_size", "64",
            "--num_epochs", "30",
            "--window_size", "20",
            "--hidden_dim", "64",
            "--time_dim", "64"
        ]
        
        if args.model_type == 'inductive':
            base_cmd.append("--inductive")
            
        if args.forecast_horizon > 1:
            base_cmd.extend(["--forecast_horizon", str(args.forecast_horizon)])
        
        # Run 3-fold cross-validation
        for fold in range(3):
            fold_cmd = base_cmd + [
                "--experiment_name", f"{args.experiment_name}_fold{fold}",
                "--seed", str(42 + fold)  # Different seed for each fold
            ]
            print(f"Running fold {fold+1}/3...")
            subprocess.run(fold_cmd)
        
        print("Cross-validation completed!")
        return
    
    # Print the command being run
    print("Running command:", " ".join(cmd))
    
    # Execute the command
    subprocess.run(cmd)

if __name__ == "__main__":
    main()