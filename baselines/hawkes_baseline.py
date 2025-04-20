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
# Full Hawkes Process Model Implementation
#############################################
class HawkesProcess:
    """
    Full Hawkes process model for stock returns.
    
    The model treats extreme return events (absolute returns above a threshold)
    as points in time and computes the intensity at a given time t based on the standard Hawkes formula:
    
        λ(t) = μ + α * Σ_{t_i < t} exp[-β * (t - t_i)]
    
    where:
        - μ is the background intensity (if not provided, estimated as (number of events) / T)
        - α is the excitation parameter
        - β is the decay parameter
        
    The predicted return is obtained by converting the predicted intensity using a scaling factor and
    using the average sign of the extreme events.
    """
    def __init__(self, threshold=0.05, mu=None, alpha=0.5, beta=1.0):
        self.threshold = threshold
        self.mu = mu      # Background intensity. If None, it will be estimated.
        self.alpha = alpha
        self.beta = beta
        self.event_times = None  # Times of extreme events
        self.event_signs = None  # Signs of the extreme returns

    def fit(self, returns, timestamps):
        """
        Fit the Hawkes process on a stock's return series.
        
        Extreme events are identified where |return| >= threshold.
        If μ is not provided, it is estimated as:
            
            μ = (number of events) / (observation window)
        
        Args:
            returns (np.array): 1D array of returns.
            timestamps (np.array): 1D array of corresponding timestamps.
            
        Returns:
            self
        """
        # Identify extreme events
        extreme_indices = np.where(np.abs(returns) >= self.threshold)[0]
        if len(extreme_indices) == 0:
            # No events: set background intensity to zero.
            self.mu = 0.0
            self.event_times = np.array([])
            self.event_signs = np.array([])
            return self

        self.event_times = timestamps[extreme_indices]
        self.event_signs = np.sign(returns[extreme_indices])
        
        # If background intensity is not provided, estimate it using the total observation window.
        if self.mu is None:
            T_total = timestamps[-1] - timestamps[0]
            self.mu = len(self.event_times) / T_total if T_total > 0 else 0.0

        return self

    def predict_intensity(self, t):
        """
        Predict the intensity at time t using the Hawkes process formula:
        
            λ(t) = μ + α * Σ_{t_i < t} exp[-β * (t - t_i)]
        
        Args:
            t (float): Time at which to predict intensity.
        
        Returns:
            float: Predicted intensity.
        """
        if self.mu is None or self.event_times is None:
            raise ValueError("Model must be fit before predicting intensity.")
        
        # Sum contributions of events that occurred before time t.
        valid_events = self.event_times[self.event_times < t]
        contribution = 0.0
        if len(valid_events) > 0:
            contribution = self.alpha * np.sum(np.exp(-self.beta * (t - valid_events)))
        return self.mu + contribution

    def predict_return(self, t, scaling_factor=1.0):
        """
        Convert predicted intensity to a return prediction.
        
        Uses the average sign of the past extreme events to determine the direction 
        (if no events, default to positive).
        
        Args:
            t (float): Time at which to predict.
            scaling_factor (float): Scaling factor to convert intensity into return magnitude.
            
        Returns:
            float: Predicted return.
        """
        intensity = self.predict_intensity(t)
        if self.event_signs.size > 0:
            sign = np.sign(np.mean(self.event_signs))
            if sign == 0:
                sign = 1
        else:
            sign = 1
        return scaling_factor * intensity * sign

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
def train_model(y_grouped, unique_days, threshold=0.05, mu=None, alpha=0.5, beta=1.0):
    """
    Train Hawkes process models for each stock.
    
    Args:
        y_grouped (np.ndarray): Grouped returns data.
        unique_days (np.ndarray): Array of unique days (timestamps).
        threshold (float): Threshold for extreme events.
        mu (float or None): Background intensity (if None, will be estimated).
        alpha (float): Excitation parameter.
        beta (float): Decay parameter.
        
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
        
        # Fit full Hawkes process model
        hawkes = HawkesProcess(threshold=threshold, mu=mu, alpha=alpha, beta=beta)
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
    
    The evaluation computes both the intensity error and the return prediction error.
    The mean squared error (MSE) on the test set is reported as 'return_mse'.
    
    Args:
        models (list): List of trained Hawkes models.
        y_grouped (np.ndarray): Training grouped returns data.
        unique_days (np.ndarray): Training unique days.
        test_y_grouped (np.ndarray, optional): Test grouped returns data.
        test_unique_days (np.ndarray, optional): Test unique days.
        
    Returns:
        dict: Dictionary of evaluation metrics.
    """
    # Use test data if provided; otherwise use training data.
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
        # Skip if no model for the stock
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
        
        # Use the last valid timestamp for prediction
        t_forecast = valid_timestamps[-1]
        
        # Predict intensity at the last timestamp using the full Hawkes model
        intensity = models[stock].predict_intensity(t_forecast)
        intensities.append(intensity)
        
        # Use the last available return as the ground truth
        actual_return = valid_returns[-1]
        ground_truth.append(actual_return)
        
        # Compute a scaling factor based on return volatility (for calibration)
        if len(valid_returns) > 1:
            return_std = np.std(valid_returns[:-1])  # Exclude the last point
            if return_std > 0:
                scaling_factor = return_std / np.mean(np.abs(valid_returns[:-1]))
            else:
                scaling_factor = 1.0
        else:
            scaling_factor = 1.0
        scaling_factors.append(scaling_factor)
        
        # Predict return using the Hawkes model's intensity and scaling factor
        pred_return = models[stock].predict_return(t_forecast, scaling_factor)
        predictions.append(pred_return)
    
    # Convert lists to numpy arrays for metric calculations
    intensities = np.array(intensities)
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)
    
    # Calculate metrics on valid predictions
    valid_eval_mask = ~np.isnan(ground_truth) & ~np.isnan(predictions)
    
    if np.sum(valid_eval_mask) > 0:
        # Intensity-based MSE (comparing predicted intensity against actual return as a proxy)
        intensity_mse = np.mean((intensities[valid_eval_mask] - ground_truth[valid_eval_mask])**2)
        
        # Return prediction error
        return_mse = np.mean((predictions[valid_eval_mask] - ground_truth[valid_eval_mask])**2)
        return_mae = np.mean(np.abs(predictions[valid_eval_mask] - ground_truth[valid_eval_mask]))
        
        # Directional accuracy calculation
        pred_sign = np.sign(predictions[valid_eval_mask])
        true_sign = np.sign(ground_truth[valid_eval_mask])
        directional_accuracy = np.mean(pred_sign == true_sign)
        
        # Correlation between predicted and actual returns
        correlation = np.corrcoef(predictions[valid_eval_mask], ground_truth[valid_eval_mask])[0, 1]
        
        metrics = {
            'intensity_mse': intensity_mse,
            'return_mse': return_mse,
            'return_mae': return_mae,
            'directional_accuracy': directional_accuracy,
            'correlation': correlation,
            'num_valid_stocks': int(np.sum(valid_eval_mask))
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
    
    # Print sample predictions for several stocks
    valid_indices = np.where(valid_eval_mask)[0]
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
    # Model hyperparameters for Hawkes process
    threshold = 0.05
    # Leave mu as None so that it is estimated from the data.
    alpha = 0.5
    beta = 1.0
    
    # --- Training Phase ---
    # Prepare training data
    train_file = f"{DATA_DIR}/dict_of_data_Jan2025_part1.npy"
    y_grouped_train, unique_days_train, num_stocks = prepare_data(train_file)
    
    # Train Hawkes process models for each stock
    hawkes_models = train_model(
        y_grouped=y_grouped_train,
        unique_days=unique_days_train,
        threshold=threshold,
        mu=None,
        alpha=alpha,
        beta=beta
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
    
    print("\nTraining Metrics:")
    for metric_name, metric_value in train_metrics.items():
        print(f"{metric_name}: {metric_value}")
    
    # Evaluate on test data (this is where the test set MSE is reported)
    print("\n--- Evaluating on Test Data ---")
    test_metrics = evaluate_model(
        models=hawkes_models,
        y_grouped=y_grouped_train,
        unique_days=unique_days_train,
        test_y_grouped=y_grouped_test,
        test_unique_days=unique_days_test
    )
    
    print("\nTest Metrics:")
    for metric_name, metric_value in test_metrics.items():
        print(f"{metric_name}: {metric_value}")
    
    # Performance comparison between train and test
    print("\n--- Performance Comparison ---")
    for metric in ['return_mse', 'return_mae', 'directional_accuracy']:
        if metric in train_metrics and metric in test_metrics:
            train_val = train_metrics[metric]
            test_val = test_metrics[metric]
            print(f"{metric}: Train = {train_val:.6f}, Test = {test_val:.6f}")

if __name__ == "__main__":
    main()
