# train.py

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
import argparse
from tqdm import tqdm
import json
import time
import inspect
from datetime import datetime

# Import our modules
from data_load import load_data, get_data
from feature_encoder import Preprocessor, CNNFeatureEncoder
from temporal_encoder import HybridTimeEncoder
from anomaly_filter import RobustStockAnomalyFilter
from attention_mechanism import HierarchicalTemporalAttention, CoAttentionModule
from stock_model import StockPredictionModel, InductiveStockPredictor, SimplifiedStockModel
from config import DATA_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class StockDataset(Dataset):
    """
    Dataset for stock prediction with sliding windows.
    
    Parameters:
        x_data : np.ndarray
            Alpha signals and other features
        y_data : np.ndarray
            Target values (next day returns)
        stock_indices : np.ndarray
            Stock indices
        day_indices : np.ndarray
            Day indices
        window_size : int
            Size of the sliding window
        forecast_horizon : int, default=1
            Number of days to forecast ahead
        stride : int, default=1
            Stride for the sliding window
    """
    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        stock_indices: np.ndarray,
        day_indices: np.ndarray,
        window_size: int,
        forecast_horizon: int = 1,
        stride: int = 1,
        preprocessor: Optional[Any] = None,
    ):
        self.x_data = x_data
        self.y_data = y_data
        self.stock_indices = stock_indices
        self.day_indices = day_indices
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self.preprocessor = preprocessor
        
        # Create column names for alpha signals and raw variables
        # Assuming the first part of x_data contains alpha signals and the rest are raw variables
        if hasattr(preprocessor, 'raw_columns') and preprocessor.raw_columns is not None:
            num_raw = len(preprocessor.raw_columns)
            num_alphas = x_data.shape[1] - num_raw
            self.feature_names = [f'alpha_{i+1}' for i in range(num_alphas)] + preprocessor.raw_columns
        else:
            # If no raw columns provided, assume all are alpha signals
            self.feature_names = [f'alpha_{i+1}' for i in range(x_data.shape[1])]
        
        # Create sliding windows
        self.windows = self._create_windows()
        
        # Log the number of windows created
        print(f"Created {len(self.windows)} sliding windows with window_size={window_size}, "
              f"forecast_horizon={forecast_horizon}, stride={stride}")
    
    def _create_windows(self) -> List[Tuple[int, int]]:
        """
        Create sliding windows based on stock indices and day indices.
        
        Returns:
            List[Tuple[int, int]]
                List of tuples (start_idx, end_idx) for each window
        """
        windows = []
        
        # Group data by stock index
        unique_stocks = np.unique(self.stock_indices)
        
        for stock_idx in unique_stocks:
            # Get all data points for this stock
            stock_mask = self.stock_indices == stock_idx
            stock_days = self.day_indices[stock_mask]
            stock_indices = np.where(stock_mask)[0]
            
            # Sort by day index
            sorted_indices = np.argsort(stock_days)
            stock_days = stock_days[sorted_indices]
            stock_indices = stock_indices[sorted_indices]
            
            # Create windows
            for i in range(0, len(stock_days) - self.window_size - self.forecast_horizon + 1, self.stride):
                # Get start and end indices
                start_idx = stock_indices[i]
                end_idx = stock_indices[i + self.window_size - 1]
                
                # Only include if the days are consecutive
                if np.all(np.diff(stock_days[i:i+self.window_size]) == 1):
                    windows.append((start_idx, end_idx))
        
        return windows
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get window indices
        start_idx, end_idx = self.windows[idx]
        
        # Get the window data
        window_indices = np.arange(start_idx, end_idx + 1)
        
        # Feature data
        x = self.x_data[window_indices]
        
        # Preprocess features if a preprocessor is provided
        if self.preprocessor is not None:
            try:
                # Convert to DataFrame with column names before preprocessing
                x_df = pd.DataFrame(x, columns=self.feature_names)
                x = self.preprocessor.transform(x_df)
            except Exception as e:
                # If preprocessing fails, just use the raw data
                print(f"Preprocessing failed: {str(e)}")
        
        # Convert to tensor
        x_tensor = torch.tensor(x, dtype=torch.float32)
        
        # Time indices
        time_indices = self.day_indices[window_indices]
        time_tensor = torch.tensor(time_indices, dtype=torch.long)
        
        # Stock indices
        stock_tensor = torch.tensor(self.stock_indices[window_indices], dtype=torch.long)
        
        # Target values (next day returns for the forecast horizon)
        target_indices = np.arange(end_idx + 1, end_idx + 1 + self.forecast_horizon)
        
        # Handle edge cases: if target_indices exceed data length
        valid_target_indices = target_indices[target_indices < len(self.y_data)]
        
        if len(valid_target_indices) < self.forecast_horizon:
            # Pad with zeros or last known value for missing targets
            targets = np.zeros(self.forecast_horizon)
            targets[:len(valid_target_indices)] = self.y_data[valid_target_indices]
        else:
            targets = self.y_data[valid_target_indices]
        
        y_tensor = torch.tensor(targets, dtype=torch.float32)
        
        return {
            'x': x_tensor,
            'time_indices': time_tensor,
            'stock_indices': stock_tensor,
            'y': y_tensor
        }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int,
    scheduler: Optional[Any] = None,
    early_stopping_patience: int = 10,
    checkpoint_dir: str = './checkpoints',
    experiment_name: str = 'stock_prediction',
):
    """
    Train the stock prediction model.
    
    Parameters:
        model : nn.Module
            Model to train
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        optimizer : optim.Optimizer
            Optimizer
        criterion : nn.Module
            Loss function
        device : torch.device
            Device to run training on
        num_epochs : int
            Number of epochs
        scheduler : Any, optional
            Learning rate scheduler
        early_stopping_patience : int
            Number of epochs to wait for improvement before stopping
        checkpoint_dir : str
            Directory to save model checkpoints
        experiment_name : str
            Name of the experiment for saving checkpoints
    
    Returns:
        Dict[str, List[float]]
            Dictionary containing training history
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Initialize history
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_mse': [],
        'val_mse': [],
    }
    
    # Training loop
    for epoch in range(num_epochs):
        # Train mode
        model.train()
        train_loss = 0.0
        train_mse = 0.0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch in train_pbar:
            # Get batch data
            x = batch['x'].to(device)
            time_indices = batch['time_indices'].to(device)
            stock_indices = batch['stock_indices'].to(device)
            targets = batch['y'].to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(model, InductiveStockPredictor):
                # For the inductive model with multi-step prediction
                outputs = model(x, time_indices, stock_indices)
                predictions = outputs['predictions']
            else:
                # For single-step prediction
                outputs = model(x, time_indices, stock_indices)
                predictions = outputs[:, -1, :]  # Use last time step for prediction
            
            # Calculate loss
            loss = criterion(predictions, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * x.size(0)
            train_mse += nn.MSELoss()(predictions, targets).item() * x.size(0)
            
            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average loss for the epoch
        train_loss /= len(train_loader.dataset)
        train_mse /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_mse = 0.0
        
        # Progress bar for validation
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
        
        with torch.no_grad():
            for batch in val_pbar:
                # Get batch data
                x = batch['x'].to(device)
                time_indices = batch['time_indices'].to(device)
                stock_indices = batch['stock_indices'].to(device)
                targets = batch['y'].to(device)
                
                # Forward pass
                if isinstance(model, InductiveStockPredictor):
                    # For the inductive model with multi-step prediction
                    outputs = model(x, time_indices, stock_indices)
                    predictions = outputs['predictions']
                else:
                    # For single-step prediction
                    outputs = model(x, time_indices, stock_indices)
                    predictions = outputs[:, -1, :]  # Use last time step for prediction
                
                # Calculate loss
                loss = criterion(predictions, targets)
                
                # Track statistics
                val_loss += loss.item() * x.size(0)
                val_mse += nn.MSELoss()(predictions, targets).item() * x.size(0)
                
                # Update progress bar
                val_pbar.set_postfix({'loss': loss.item()})
        
        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)
        val_mse /= len(val_loader.dataset)
        
        # Step the scheduler if provided
        if scheduler is not None:
            scheduler.step(val_loss)
        
        # Print epoch statistics
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Train Loss: {train_loss:.6f}, Train MSE: {train_mse:.6f}, "
                   f"Val Loss: {val_loss:.6f}, Val MSE: {val_mse:.6f}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_mse'].append(train_mse)
        history['val_mse'].append(val_mse)
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, checkpoint_path)
            
            logger.info(f"Saved best model to {checkpoint_path}")
        else:
            patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{experiment_name}_epoch{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
            }, checkpoint_path)
    
    return history


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    forecast_horizon: int = 1,
):
    """
    Evaluate the stock prediction model on the test set.
    
    Parameters:
        model : nn.Module
            Model to evaluate
        test_loader : DataLoader
            Test data loader
        criterion : nn.Module
            Loss function
        device : torch.device
            Device to run evaluation on
        forecast_horizon : int
            Number of steps to forecast
            
    Returns:
        Dict[str, float]
            Dictionary containing evaluation metrics
    """
    # Set model to evaluation mode
    model.eval()
    
    # Initialize variables
    total_loss = 0.0
    total_mse = 0.0
    
    # Storage for predictions and targets
    all_predictions = []
    all_targets = []
    
    # Progress bar
    test_pbar = tqdm(test_loader, desc="Evaluating")
    
    with torch.no_grad():
        for batch in test_pbar:
            # Get batch data
            x = batch['x'].to(device)
            time_indices = batch['time_indices'].to(device)
            stock_indices = batch['stock_indices'].to(device)
            targets = batch['y'].to(device)
            
            # Forward pass
            if isinstance(model, InductiveStockPredictor):
                # For the inductive model with multi-step prediction
                outputs = model(x, time_indices, stock_indices)
                predictions = outputs['predictions']
            else:
                # For single-step prediction (use only the last step)
                outputs = model(x, time_indices, stock_indices)
                predictions = outputs[:, -1, :]
            
            # Calculate loss
            loss = criterion(predictions, targets)
            
            # Track statistics
            total_loss += loss.item() * x.size(0)
            total_mse += nn.MSELoss()(predictions, targets).item() * x.size(0)
            
            # Store predictions and targets
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
            # Update progress bar
            test_pbar.set_postfix({'loss': loss.item()})
    
    # Calculate average metrics
    avg_loss = total_loss / len(test_loader.dataset)
    avg_mse = total_mse / len(test_loader.dataset)
    
    # Combine predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Calculate additional metrics
    metrics = {
        'test_loss': avg_loss,
        'test_mse': avg_mse,
        'test_rmse': np.sqrt(avg_mse),
    }
    
    # Calculate metrics per forecast step if multi-step
    if forecast_horizon > 1:
        for step in range(forecast_horizon):
            step_mse = np.mean((all_predictions[:, step] - all_targets[:, step]) ** 2)
            step_rmse = np.sqrt(step_mse)
            
            metrics[f'test_mse_step{step+1}'] = step_mse
            metrics[f'test_rmse_step{step+1}'] = step_rmse
    
    return metrics, all_predictions, all_targets


def plot_training_history(history: Dict[str, List[float]], save_path: Optional[str] = None):
    """
    Plot training history.
    
    Parameters:
        history : Dict[str, List[float]]
            Dictionary containing training history
        save_path : str, optional
            Path to save the plot
    """
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot MSE
    plt.subplot(1, 2, 2)
    plt.plot(history['train_mse'], label='Train MSE')
    plt.plot(history['val_mse'], label='Validation MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.title('Training and Validation MSE')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    forecast_horizon: int = 1,
    num_samples: int = 10,
    save_path: Optional[str] = None
):
    """
    Plot predictions against targets.
    
    Parameters:
        predictions : np.ndarray
            Predicted values
        targets : np.ndarray
            Target values
        forecast_horizon : int
            Number of steps to forecast
        num_samples : int
            Number of samples to plot
        save_path : str, optional
            Path to save the plot
    """
    plt.figure(figsize=(15, 10))
    
    # Select random samples
    indices = np.random.choice(len(predictions), min(num_samples, len(predictions)), replace=False)
    
    if forecast_horizon == 1:
        # Single-step forecast
        plt.scatter(targets[indices], predictions[indices], alpha=0.6)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'k--')
        plt.xlabel('Actual Returns')
        plt.ylabel('Predicted Returns')
        plt.title('Predicted vs Actual Stock Returns')
    else:
        # Multi-step forecast
        for i, idx in enumerate(indices):
            plt.subplot(num_samples, 1, i + 1)
            
            # Plot actual vs predicted
            plt.plot(range(forecast_horizon), targets[idx], 'b-', label='Actual')
            plt.plot(range(forecast_horizon), predictions[idx], 'r--', label='Predicted')
            
            plt.ylabel('Return')
            if i == 0:
                plt.title('Multi-step Forecast: Predicted vs Actual Returns')
            if i == len(indices) - 1:
                plt.xlabel('Forecast Step')
            
            plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()


def create_debug_dataset(data_dict, num_days=3, num_stocks=10, ensure_consecutive=True):
    """
    Create a smaller debug dataset by subsampling a few days and stocks.
    
    Parameters:
        data_dict : dict
            Original data dictionary
        num_days : int
            Number of days to include
        num_stocks : int
            Number of stocks to include
        ensure_consecutive : bool
            If True, ensures selected days are consecutive for windowing
            
    Returns:
        dict
            Subsampled data dictionary
    """
    # Extract data components
    x_data = data_dict['x_data']
    y_data = data_dict['y_data']
    si = data_dict['si']
    di = data_dict['di']
    raw_data = data_dict['raw_data']
    list_of_data = data_dict['list_of_data']
    
    # Get unique days and stocks
    unique_days = np.unique(di)
    unique_stocks = np.unique(si)
    
    # Sample stocks
    if len(unique_stocks) > num_stocks:
        sample_stocks = np.sort(np.random.choice(unique_stocks, num_stocks, replace=False))
    else:
        sample_stocks = unique_stocks
    
    # Sample days - ensuring they're consecutive if requested
    if ensure_consecutive and len(unique_days) > num_days:
        # Choose a random starting day that allows for consecutive days
        max_start_idx = len(unique_days) - num_days
        start_idx = np.random.randint(0, max_start_idx)
        sample_days = unique_days[start_idx:start_idx + num_days]
    else:
        # Just randomly sample days without ensuring consecutive
        if len(unique_days) > num_days:
            sample_days = np.sort(np.random.choice(unique_days, num_days, replace=False))
        else:
            sample_days = unique_days
    
    # Log the selected days and stocks
    logger.info(f"Debug dataset using days: {sample_days} and stocks: {sample_stocks}")
    
    # Create mask for selected days and stocks
    day_mask = np.isin(di, sample_days)
    stock_mask = np.isin(si, sample_stocks)
    combined_mask = day_mask & stock_mask
    
    # Apply mask to all data components
    debug_dict = {
        'x_data': x_data[combined_mask],
        'y_data': y_data[combined_mask],
        'si': si[combined_mask],
        'di': di[combined_mask],
        'raw_data': raw_data[combined_mask],
        'list_of_data': list_of_data
    }
    
    # Verify we have some data
    logger.info(f"Debug dataset contains {len(debug_dict['x_data'])} data points")
    
    return debug_dict


from dataset_fix import custom_collate_fn

def main():
    """Main function to train and evaluate the stock prediction model."""
    parser = argparse.ArgumentParser(description='Stock Prediction Model Training')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default=DATA_DIR,
                        help='Directory containing the data files')
    parser.add_argument('--use_merged', action='store_true',
                        help='Use merged dataset (both parts)')
    parser.add_argument('--window_size', type=int, default=20,
                        help='Size of the sliding window')
    parser.add_argument('--forecast_horizon', type=int, default=1,
                        help='Number of days to forecast ahead')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for the sliding window')
    parser.add_argument('--train_part1_test_part2', action='store_true',
                        help='Train on part 1 and test on part 2')
    parser.add_argument('--debug', action='store_true',
                        help='Use smaller debug dataset')
    parser.add_argument('--debug_days', type=int, default=3,
                        help='Number of days to include in debug dataset')
    parser.add_argument('--debug_stocks', type=int, default=10,
                        help='Number of stocks to include in debug dataset')
    parser.add_argument('--create_merged', action='store_true',
                        help='Create merged dataset before training')
    parser.add_argument('--analyze_data', action='store_true',
                        help='Run data analysis before training')
    parser.add_argument('--debug_shapes', action='store_true',
                        help='Print tensor shapes for debugging')
    
    # Model arguments
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Dimension of the hidden representation')
    parser.add_argument('--time_dim', type=int, default=64,
                        help='Dimension of the time embedding')
    parser.add_argument('--num_transformer_layers', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--num_attention_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--temporal_bin_size', type=int, default=5,
                        help='Size of temporal bins for hierarchical attention')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    parser.add_argument('--inductive', action='store_true',
                        help='Use inductive model for multi-step prediction')
    parser.add_argument('--skip_time_encoding', action='store_true',
                        help='Skip time encoding for debugging')
    parser.add_argument('--skip_anomaly_filter', action='store_true',
                        help='Skip anomaly filtering for debugging')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay for regularization')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--experiment_name', type=str, default='stock_prediction',
                        help='Name of the experiment')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation on test set')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create merged dataset if requested
    if args.create_merged:
        logger.info("Creating merged dataset...")
        file_path1 = os.path.join(args.data_dir, 'dict_of_data_Jan2025_part1.npy')
        file_path2 = os.path.join(args.data_dir, 'dict_of_data_Jan2025_part2.npy')
        output_path = os.path.join(args.data_dir, 'merged_dataset.npy')
        
        # Use merge_datasets function from data_load.py
        merged_dict = merge_datasets(file_path1, file_path2)
        save_merged_dataset(merged_dict, output_path)
        
        if args.analyze_data:
            logger.info("Saving data preview...")
            save_data_preview(n_samples=100)
    
    # Run data analysis if requested
    if args.analyze_data and not args.create_merged:
        logger.info("Analyzing stock characteristics...")
        file_path1 = os.path.join(args.data_dir, 'dict_of_data_Jan2025_part1.npy')
        file_path2 = os.path.join(args.data_dir, 'dict_of_data_Jan2025_part2.npy')
        
        # Analyze part 1
        logger.info("Analyzing part 1 dataset...")
        data_dict1 = load_data(file_path1)
        stock_info_df1 = analyze_stock_characteristics(data_dict1)
        stock_info_df1.to_csv('stock_universe_summary_part1.csv')
        
        # Analyze part 2
        logger.info("Analyzing part 2 dataset...")
        data_dict2 = load_data(file_path2)
        stock_info_df2 = analyze_stock_characteristics(data_dict2)
        stock_info_df2.to_csv('stock_universe_summary_part2.csv')
        
        # Exit if only analysis was requested
        if not any([args.train_part1_test_part2, args.use_merged]):
            logger.info("Data analysis completed. Exiting.")
            return
    
    # Load data based on arguments
    if args.train_part1_test_part2:
        # Load part 1 for training
        train_path = os.path.join(args.data_dir, 'dict_of_data_Jan2025_part1.npy')
        logger.info(f"Loading part 1 dataset for training from {train_path}")
        train_data_dict = load_data(train_path)
        
        # Load part 2 for testing
        test_path = os.path.join(args.data_dir, 'dict_of_data_Jan2025_part2.npy')
        logger.info(f"Loading part 2 dataset for testing from {test_path}")
        test_data_dict = load_data(test_path)
        
        # Create debug datasets if requested
        if args.debug:
            logger.info(f"Creating debug datasets with {args.debug_days} days and {args.debug_stocks} stocks")
            train_data_dict = create_debug_dataset(
                train_data_dict, 
                num_days=max(args.window_size + args.forecast_horizon + 5, args.debug_days),  # Ensure enough days for windowing
                num_stocks=args.debug_stocks,
                ensure_consecutive=True
            )
            test_data_dict = create_debug_dataset(
                test_data_dict, 
                num_days=max(args.window_size + args.forecast_horizon + 5, args.debug_days),
                num_stocks=args.debug_stocks,
                ensure_consecutive=True
            )
            
        # Extract components from both datasets
        train_x, train_y, train_si, train_di, train_raw, list_of_data = get_data(train_data_dict)
        test_x, test_y, test_si, test_di, test_raw, _ = get_data(test_data_dict)
        
        # Print data summary
        logger.info(f"Training data: x_data shape: {train_x.shape}, y_data shape: {train_y.shape}")
        logger.info(f"Test data: x_data shape: {test_x.shape}, y_data shape: {test_y.shape}")
        
        # Verify alpha signals are aligned (have same meaning) between parts
        # This is an assumption - in a real project, we would need to verify this
        logger.info("Assuming alpha signals are aligned between parts 1 and 2")
        
        # Create a preprocessor using only the training data
        num_alphas = train_x.shape[1]
        alpha_cols = [f'alpha_{i+1}' for i in range(num_alphas)]
        df_train_alphas = pd.DataFrame(train_x, columns=alpha_cols)
        df_train_raw = pd.DataFrame(train_raw, columns=list_of_data)
        df_train_features = pd.concat([df_train_alphas, df_train_raw], axis=1)
        
        preprocessor = Preprocessor(
            imputation_strategy='mean',
            scaling_method='standard',
            alpha_prefix='alpha_',
            raw_columns=list_of_data
        )
        
        # Fit preprocessor on training data only
        preprocessor.fit(df_train_features)
        
        # Create datasets
        train_dataset = StockDataset(
            x_data=np.hstack([train_x, train_raw]),
            y_data=train_y,
            stock_indices=train_si,
            day_indices=train_di,
            window_size=args.window_size,
            forecast_horizon=args.forecast_horizon,
            stride=args.stride,
            preprocessor=preprocessor
        )
        
        test_dataset = StockDataset(
            x_data=np.hstack([test_x, test_raw]),
            y_data=test_y,
            stock_indices=test_si,
            day_indices=test_di,
            window_size=args.window_size,
            forecast_horizon=args.forecast_horizon,
            stride=args.stride,
            preprocessor=preprocessor  # Use the same preprocessor fitted on training data
        )
        
        # Split train dataset into train and validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
        
        # Compute input dimension
        input_dim = train_x.shape[1] + train_raw.shape[1]
        
    else:
        # Original approach: load one dataset and split it
        if args.use_merged:
            data_path = os.path.join(args.data_dir, 'merged_dataset.npy')
            logger.info(f"Loading merged dataset from {data_path}")
        else:
            data_path = os.path.join(args.data_dir, 'dict_of_data_Jan2025_part1.npy')
            logger.info(f"Loading part 1 dataset from {data_path}")
        
        data_dict = load_data(data_path)
        
        # Create debug dataset if requested
        if args.debug:
            logger.info(f"Creating debug dataset with {args.debug_days} days and {args.debug_stocks} stocks")
            data_dict = create_debug_dataset(
                data_dict, num_days=args.debug_days, num_stocks=args.debug_stocks)
        
        x_data, y_data, si, di, raw_data, list_of_data = get_data(data_dict)
        
        # Print data summary
        logger.info(f"x_data shape: {x_data.shape}")
        logger.info(f"y_data shape: {y_data.shape}")
        logger.info(f"stock indices shape: {si.shape}")
        logger.info(f"day indices shape: {di.shape}")
        logger.info(f"raw_data shape: {raw_data.shape}")
        logger.info(f"List of raw data: {list_of_data}")
        
        # Combine alpha signals with raw data
        input_dim = x_data.shape[1] + raw_data.shape[1]
        logger.info(f"Combined input dimension: {input_dim}")
        
        # Construct DataFrames for alpha signals and raw variables
        num_alphas = x_data.shape[1]
        alpha_cols = [f'alpha_{i+1}' for i in range(num_alphas)]
        df_alphas = pd.DataFrame(x_data, columns=alpha_cols)
        df_raw = pd.DataFrame(raw_data, columns=list_of_data)
        
        # Concatenate features horizontally
        df_features = pd.concat([df_alphas, df_raw], axis=1)
        
        # Create preprocessor
        preprocessor = Preprocessor(
            imputation_strategy='mean',
            scaling_method='standard',
            alpha_prefix='alpha_',
            raw_columns=list_of_data
        )
        
        # Fit preprocessor on all data
        preprocessor.fit(df_features)
        
        # Create dataset
        dataset = StockDataset(
            x_data=np.hstack([x_data, raw_data]),
            y_data=y_data,
            stock_indices=si,
            day_indices=di,
            window_size=args.window_size,
            forecast_horizon=args.forecast_horizon,
            stride=args.stride,
            preprocessor=preprocessor
        )
        
        # Split dataset
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        logger.info(f"Test dataset size: {len(test_dataset)}")
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=custom_collate_fn)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=custom_collate_fn)
    
    # Create model
    if args.inductive and args.forecast_horizon > 1:
        logger.info("Creating InductiveStockPredictor model")
        model = InductiveStockPredictor(
            input_dim=input_dim,
            time_dim=args.time_dim,
            hidden_dim=args.hidden_dim,
            forecast_steps=args.forecast_horizon,
            use_uncertainty=True,
            num_transformer_layers=args.num_transformer_layers,
            num_attention_heads=args.num_attention_heads,
            temporal_bin_size=args.temporal_bin_size,
            dropout=args.dropout,
            skip_time_encoding=args.skip_time_encoding,
            skip_anomaly_filter=args.skip_anomaly_filter
        )
    else:
        # For debugging with very small datasets, use simplified model
        if args.debug and len(train_dataset) < 20:
            logger.info("Creating SimplifiedStockModel (debug mode with small dataset)")
            model = SimplifiedStockModel(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=1
            )
        else:
            logger.info("Creating StockPredictionModel")
            model = StockPredictionModel(
                input_dim=input_dim,
                time_dim=args.time_dim,
                hidden_dim=args.hidden_dim,
                output_dim=1,  # Single value prediction
                num_transformer_layers=args.num_transformer_layers,
                num_attention_heads=args.num_attention_heads,
                temporal_bin_size=args.temporal_bin_size,
                dropout=args.dropout,
                skip_time_encoding=args.skip_time_encoding,
                skip_anomaly_filter=args.skip_anomaly_filter
            )
    
    # Apply monkey patching to fix device and shape issues
    if args.debug and hasattr(model, 'forward'):
        logger.info("Applying fixed forward method to StockPredictionModel")
        try:
            from stock_model_fix2 import fixed_forward
            # Monkey patch the forward method
            model.__class__.forward = fixed_forward
        except ImportError as e:
            logger.warning(f"Could not import fixed_forward: {str(e)}")
    
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # If only evaluating, load the best model
    if args.eval_only:
        checkpoint_path = os.path.join(args.checkpoint_dir, f"{args.experiment_name}_best.pth")
        if os.path.exists(checkpoint_path):
            logger.info(f"Loading model from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}, using untrained model")
    else:
        # Train model
        logger.info("Starting training...")
        start_time = time.time()
        
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_epochs=args.num_epochs,
            scheduler=scheduler,
            early_stopping_patience=args.early_stopping_patience,
            checkpoint_dir=args.checkpoint_dir,
            experiment_name=args.experiment_name
        )
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Plot training history
        logger.info("Plotting training history...")
        plot_training_history(
            history,
            save_path=os.path.join(args.checkpoint_dir, f"{args.experiment_name}_history.png")
        )
    
    # Evaluate model on test set
    logger.info("Evaluating model on test set...")
    metrics, all_predictions, all_targets = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        forecast_horizon=args.forecast_horizon
    )
    
    # Print test metrics
    for key, value in metrics.items():
        logger.info(f"{key}: {value:.6f}")
    
    # Plot predictions
    logger.info("Plotting predictions...")
    plot_predictions(
        predictions=all_predictions,
        targets=all_targets,
        forecast_horizon=args.forecast_horizon,
        num_samples=10,
        save_path=os.path.join(args.checkpoint_dir, f"{args.experiment_name}_predictions.png")
    )
    
    # Save test metrics
    with open(os.path.join(args.checkpoint_dir, f"{args.experiment_name}_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=4)
    
    logger.info("Experiment completed successfully!")


if __name__ == "__main__":
    main()