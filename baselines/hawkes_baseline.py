# baselines/hawkes_baseline.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import math
from tqdm import tqdm

from data_load import load_data, get_data
from config import DATA_DIR

#############################################
# Hawkes Process Baseline Model
#############################################
class HawkesProcessBaseline:
    """
    A simplified Hawkes process model for stock returns.
    
    This model identifies extreme returns (based on a threshold) as events,
    then estimates a base intensity as the inverse of the mean inter-event time.
    A fixed excitation parameter is used. The predicted intensity at a given time
    is computed as:
    
        intensity(t) = base_intensity * exp(-decay * t) + alpha
    
    where t is the time (e.g. the last timestamp).
    """
    def __init__(self, threshold=0.05, decay=1.0, alpha=0.5):
        self.threshold = threshold
        self.decay = decay
        self.alpha = alpha
        self.base_intensity = None
        self.event_times = None
        self.mean_inter_event = None

    def fit(self, returns, timestamps):
        """
        Fit the Hawkes process on the extreme events from a stock's return series.
        
        Args:
            returns (np.array): 1D array of returns.
            timestamps (np.array): 1D array of corresponding timestamps.
        """
        # Identify extreme events where the absolute return exceeds the threshold
        extreme_indices = np.where(np.abs(returns) >= self.threshold)[0]
        if len(extreme_indices) < 2:
            self.base_intensity = 0.0
            return self
        
        self.event_times = timestamps[extreme_indices]
        inter_event = np.diff(self.event_times)
        self.mean_inter_event = np.mean(inter_event)
        
        # Avoid division by zero
        if self.mean_inter_event > 0:
            self.base_intensity = 1.0 / self.mean_inter_event
        else:
            self.base_intensity = 0.0
            
        return self

    def predict(self, t):
        """
        Predict the intensity at time t.
        
        Args:
            t (float): Time at which to predict intensity (e.g., the last timestamp).
        
        Returns:
            float: Predicted intensity.
        """
        if self.base_intensity is None:
            raise ValueError("Model must be fit before making predictions")
            
        return self.base_intensity * math.exp(-self.decay * t) + self.alpha
    
    def predict_return(self, t, scaling_factor=1.0):
        """
        Convert intensity to a predicted return.
        
        This is a simplified approach to convert intensity to returns.
        A better approach would involve calibrating this relationship.
        
        Args:
            t (float): Time at which to predict.
            scaling_factor (float): Scaling factor to convert intensity to returns.
            
        Returns:
            float: Predicted return.
        """
        intensity = self.predict(t)
        # A simple heuristic to convert intensity to returns
        # Higher intensity → higher magnitude of returns
        return scaling_factor * intensity * (1 if np.random.random() > 0.5 else -1)

#############################################
# Utility Function: Group Targets
#############################################
def group_targets(si, di, y):
    """
    Group targets into a 2D array (num_stocks, num_days) using a pivot table.
    
    Args:
        si (array-like): Stock indices.
        di (array-like): Day indices.
        y (array-like): Return values.
    
    Returns:
        np.ndarray: 2D array with shape (num_stocks, num_days).
        np.ndarray: Array of unique days (timestamps).
    """
    df = pd.DataFrame({"stock_index": si, "day_index": di, "y": y})
    
    # Get unique sorted days
    unique_days = np.sort(df["day_index"].unique())
    
    # Create pivot table
    df_pivot = df.pivot_table(index="stock_index", columns="day_index", values="y", aggfunc="first")
    df_pivot = df_pivot.sort_index(axis=1)
    
    return df_pivot.values, unique_days

#############################################
# Data Preparation Function
#############################################
def prepare_data(file_path):
    """
    Load and prepare data for Hawkes process modeling.
    
    Args:
        file_path (str): Path to the data file.
        
    Returns:
        tuple: (grouped_returns, unique_days, num_stocks)
    """
    print(f"Loading data from {file_path}...")
    data_dict = load_data(file_path)
    _, y_data, si, di, _, _ = get_data(data_dict)
    
    # Group returns by stock
    y_grouped, unique_days = group_targets(si, di, y_data)
    num_stocks = y_grouped.shape[0]
    print(f"Grouped returns shape: {y_grouped.shape}")
    print(f"Number of unique days: {len(unique_days)}")
    
    return y_grouped, unique_days, num_stocks

#############################################
# Model Training Function
#############################################
def train_model(y_grouped, unique_days, threshold=0.05, decay=1.0, alpha=0.5):
    """
    Train Hawkes process models for each stock.
    
    Args:
        y_grouped (np.ndarray): Grouped returns data.
        unique_days (np.ndarray): Array of unique days (timestamps).
        threshold (float): Threshold for extreme events.
        decay (float): Decay parameter for the Hawkes process.
        alpha (float): Fixed excitation parameter.
        
    Returns:
        list: List of trained Hawkes models for each stock.
    """
    num_stocks = y_grouped.shape[0]
    models = []
    
    print("Training Hawkes process models...")
    for stock in tqdm(range(num_stocks), desc="Training models"):
        stock_returns = y_grouped[stock, :]
        
        # Filter out NaN values
        valid_mask = ~np.isnan(stock_returns)
        valid_returns = stock_returns[valid_mask]
        valid_timestamps = unique_days[valid_mask]
        
        # Skip stocks with insufficient data
        if len(valid_returns) < 2:
            models.append(None)
            continue
        
        # Fit Hawkes process model
        hawkes = HawkesProcessBaseline(threshold=threshold, decay=decay, alpha=alpha)
        hawkes.fit(valid_returns, valid_timestamps)
        models.append(hawkes)
    
    valid_models = sum(1 for model in models if model is not None)
    print(f"Successfully trained {valid_models} models out of {num_stocks} stocks")
    
    return models

#############################################
# Model Evaluation Function
#############################################
def evaluate_model(models, y_grouped, unique_days, test_y_grouped=None, test_unique_days=None):
    """
    Evaluate Hawkes process models.
    
    Args:
        models (list): List of trained Hawkes models.
        y_grouped (np.ndarray): Training grouped returns data.
        unique_days (np.ndarray): Training unique days.
        test_y_grouped (np.ndarray, optional): Test grouped returns data.
        test_unique_days (np.ndarray, optional): Test unique days.
        
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    # Decide whether to use training or test data for evaluation
    if test_y_grouped is not None and test_unique_days is not None:
        eval_y_grouped = test_y_grouped
        eval_unique_days = test_unique_days
        print("Evaluating on test data...")
    else:
        eval_y_grouped = y_grouped
        eval_unique_days = unique_days
        print("Evaluating on training data...")
    
    num_stocks = len(models)
    intensities = []
    predictions = []
    ground_truth = []
    scaling_factors = []
    
    for stock in tqdm(range(num_stocks), desc="Evaluating models"):
        # Skip if model is None
        if models[stock] is None:
            intensities.append(np.nan)
            predictions.append(np.nan)
            ground_truth.append(np.nan)
            scaling_factors.append(np.nan)
            continue
        
        # Get stock returns and filter out NaNs
        stock_returns = eval_y_grouped[stock, :]
        valid_mask = ~np.isnan(stock_returns)
        
        if np.sum(valid_mask) < 2:
            intensities.append(np.nan)
            predictions.append(np.nan)
            ground_truth.append(np.nan)
            scaling_factors.append(np.nan)
            continue
            
        valid_returns = stock_returns[valid_mask]
        valid_timestamps = eval_unique_days[valid_mask]
        
        # Get the last timestamp for prediction
        t_forecast = valid_timestamps[-1]
        
        # Predict intensity at the last timestamp
        intensity = models[stock].predict(t_forecast)
        intensities.append(intensity)
        
        # Use the last return as ground truth
        actual_return = valid_returns[-1]
        ground_truth.append(actual_return)
        
        # Calculate a stock-specific scaling factor based on return volatility
        # This helps adapt the intensity->return conversion to each stock's scale
        if len(valid_returns) > 1:
            return_std = np.std(valid_returns[:-1])  # Exclude the last point
            if return_std > 0:
                scaling_factor = return_std / np.mean(np.abs(valid_returns[:-1]))
            else:
                scaling_factor = 1.0
        else:
            scaling_factor = 1.0
            
        scaling_factors.append(scaling_factor)
        
        # Predict return using intensity and scaling factor
        pred_return = models[stock].predict_return(t_forecast, scaling_factor)
        predictions.append(pred_return)
    
    # Convert to arrays
    intensities = np.array(intensities)
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Calculate metrics only on valid predictions
    valid_mask = ~np.isnan(ground_truth) & ~np.isnan(predictions)
    
    if np.sum(valid_mask) > 0:
        # Intensity-based error
        intensity_mse = np.mean((intensities[valid_mask] - ground_truth[valid_mask])**2)
        
        # Return prediction error
        return_mse = np.mean((predictions[valid_mask] - ground_truth[valid_mask])**2)
        return_mae = np.mean(np.abs(predictions[valid_mask] - ground_truth[valid_mask]))
        
        # Directional accuracy
        pred_sign = np.sign(predictions[valid_mask])
        true_sign = np.sign(ground_truth[valid_mask])
        directional_accuracy = np.mean(pred_sign == true_sign)
        
        # Correlation between predicted and actual returns
        correlation = np.corrcoef(predictions[valid_mask], ground_truth[valid_mask])[0, 1]
        
        metrics = {
            'intensity_mse': intensity_mse,
            'return_mse': return_mse,
            'return_mae': return_mae,
            'directional_accuracy': directional_accuracy,
            'correlation': correlation,
            'num_valid_stocks': np.sum(valid_mask)
        }
    else:
        metrics = {
            'intensity_mse': float('nan'),
            'return_mse': float('nan'),
            'return_mae': float('nan'),
            'directional_accuracy': float('nan'),
            'correlation': float('nan'),
            'num_valid_stocks': 0
        }
    
    # Print sample predictions for a few stocks
    valid_indices = np.where(valid_mask)[0]
    if len(valid_indices) > 0:
        sample_indices = valid_indices[:min(10, len(valid_indices))]
        print("\nSample predictions:")
        for idx in sample_indices:
            print(f"Stock {idx}: Pred = {predictions[idx]:.6f}, True = {ground_truth[idx]:.6f}, "
                  f"Intensity = {intensities[idx]:.6f}, Scaling = {scaling_factors[idx]:.6f}")
    
    return metrics

#############################################
# Main Function
#############################################
def main():
    # Model hyperparameters
    threshold = 0.05
    decay = 1.0
    alpha = 0.5
    
    # --- Training Phase ---
    # Prepare training data
    train_file = f"{DATA_DIR}/dict_of_data_Jan2025_part1.npy"
    y_grouped_train, unique_days_train, num_stocks = prepare_data(train_file)
    
    # Train models
    hawkes_models = train_model(
        y_grouped=y_grouped_train,
        unique_days=unique_days_train,
        threshold=threshold,
        decay=decay,
        alpha=alpha
    )
    
    # --- Evaluation Phase ---
    # Prepare test data
    test_file = f"{DATA_DIR}/dict_of_data_Jan2025_part2.npy"
    y_grouped_test, unique_days_test, _ = prepare_data(test_file)
    
    # Evaluate on training data
    print("\n--- Evaluating on Training Data ---")
    train_metrics = evaluate_model(
        models=hawkes_models,
        y_grouped=y_grouped_train,
        unique_days=unique_days_train
    )
    
    # Print training metrics
    print("\nTraining Metrics:")
    for metric_name, metric_value in train_metrics.items():
        print(f"{metric_name}: {metric_value}")
    
    # Evaluate on test data
    print("\n--- Evaluating on Test Data ---")
    test_metrics = evaluate_model(
        models=hawkes_models,
        y_grouped=y_grouped_train,
        unique_days=unique_days_train,
        test_y_grouped=y_grouped_test,
        test_unique_days=unique_days_test
    )
    
    # Print test metrics
    print("\nTest Metrics:")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value}")
    
    # Compare performance
    print("\n--- Performance Comparison ---")
    for metric in ['return_mse', 'return_mae', 'directional_accuracy']:
        if metric in train_metrics and metric in test_metrics:
            train_val = train_metrics[metric]
            test_val = test_metrics[metric]
            print(f"{metric}: Train = {train_val:.6f}, Test = {test_val:.6f}")

if __name__ == "__main__":
    main()