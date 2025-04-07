# baselines/gan_baseline.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

from data_load import load_data, get_data
from config import DATA_DIR
from feature_encoder import Preprocessor, group_by_stock

#############################################
# Custom Dataset: WindowsDataset
#############################################
class WindowsDataset(Dataset):
    """
    Computes sliding windows on the fly from grouped 3D data.
    
    Grouped data is assumed to have shape:
         (num_stocks, num_days, num_features)
    Grouped targets are assumed to have shape:
         (num_stocks, num_days)
         
    For each stock, a window of length `window_size` is extracted and the target is the
    return on the day immediately following the window.
    """
    def __init__(self, grouped_data, grouped_targets, window_size=10):
        self.grouped_data = grouped_data
        self.grouped_targets = grouped_targets
        self.window_size = window_size
        
        # Build a list of (stock, window_start) indices for all valid windows.
        self.indices = []
        num_stocks, num_days, _ = grouped_data.shape
        for stock in range(num_stocks):
            for start in range(num_days - window_size):
                # Check if the target is valid (not NaN)
                target_idx = start + window_size
                if target_idx < num_days and not np.isnan(grouped_targets[stock, target_idx]):
                    # Check if all values in the window are valid (not NaN)
                    window = grouped_data[stock, start:start + window_size, :]
                    if not np.isnan(window).any():
                        self.indices.append((stock, start))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        stock, start = self.indices[idx]
        window = self.grouped_data[stock, start:start + self.window_size, :]
        target = self.grouped_targets[stock, start + self.window_size]
        return torch.tensor(window, dtype=torch.float32), torch.tensor(target, dtype=torch.float32).unsqueeze(0)

#############################################
# GAN Baseline Model
#############################################
class TimeSeriesGANBaseline(nn.Module):
    """
    GAN Baseline for Time Series Prediction.
    
    Generator: Maps a latent noise vector to a forecast window.
    Discriminator: Flattens a window and outputs a probability of being real.
    """
    def __init__(self, latent_dim, window_size, feature_dim, hidden_dim=64):
        super(TimeSeriesGANBaseline, self).__init__()
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.feature_dim = feature_dim
        
        # Generator: noise -> window (window_size x feature_dim)
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.Linear(hidden_dim * 4, window_size * feature_dim),
            nn.Tanh()  # assuming features are scaled to [-1, 1]
        )
        
        # Discriminator: window -> probability
        self.discriminator = nn.Sequential(
            nn.Linear(window_size * feature_dim, hidden_dim * 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid()
        )
    
    def generate(self, noise):
        """Generate a window from noise"""
        gen_out = self.generator(noise)
        return gen_out.view(-1, self.window_size, self.feature_dim)
    
    def forward(self, x):
        """Discriminate real from fake windows"""
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return self.discriminator(x_flat)

#############################################
# Utility Function: Group Targets
#############################################
def group_targets(si, di, y):
    """
    Group targets into a 2D array (num_stocks, num_days) using a pivot table.
    """
    df = pd.DataFrame({"stock_index": si, "day_index": di, "y": y})
    df_pivot = df.pivot_table(index="stock_index", columns="day_index", values="y", aggfunc="first")
    df_pivot = df_pivot.sort_index(axis=1)
    return df_pivot.values

#############################################
# Data Preparation Function
#############################################
def prepare_data(file_path, window_size, batch_size, shuffle=True):
    """
    Load and prepare data for GAN training or testing.
    
    Args:
        file_path: Path to the data file
        window_size: Size of the sliding window
        batch_size: Batch size for the data loader
        shuffle: Whether to shuffle the data
        
    Returns:
        data_loader: DataLoader for the prepared dataset
        feature_dim: Number of features in the data
    """
    # Load data
    print(f"Loading data from {file_path}...")
    data_dict = load_data(file_path)
    x_data, y_data, si, di, raw_data, list_of_data = get_data(data_dict)
    
    # Use only alpha signals
    num_alphas = x_data.shape[1]
    alpha_cols = [f'alpha_{i+1}' for i in range(num_alphas)]
    df_data = pd.DataFrame(x_data, columns=alpha_cols)
    df_data["stock_index"] = si
    df_data["day_index"] = di
    
    # Preprocess features
    df_ids = df_data[["stock_index", "day_index"]]
    df_features = df_data.drop(columns=["stock_index", "day_index"])
    
    preprocessor = Preprocessor(
        imputation_strategy='mean',
        scaling_method='standard',
        alpha_prefix='alpha_',
        raw_columns=[]  # No raw columns used
    )
    preprocessed = preprocessor.fit_transform(df_features)
    
    # Create dataframe with preprocessed features
    df_preprocessed = pd.DataFrame(preprocessed, columns=df_features.columns)
    df_preprocessed["stock_index"] = df_ids["stock_index"].values
    df_preprocessed["day_index"] = df_ids["day_index"].values
    
    # Check for NaNs in preprocessed data
    nan_count = df_preprocessed.isna().sum().sum()
    print(f"NaN count in preprocessed data: {nan_count}")
    if nan_count > 0:
        raise ValueError("Preprocessed data contains NaNs!")
    
    # Group data by stock
    data_3d = group_by_stock(df_preprocessed)
    print(f"Grouped data shape: {data_3d.shape}")
    
    # Group targets
    y_grouped = group_targets(si, di, y_data)
    print(f"Grouped targets shape: {y_grouped.shape}")
    
    # Check for NaNs after grouping
    nan_percentage = np.isnan(data_3d).mean() * 100
    print(f"Percentage of NaNs in grouped data: {nan_percentage:.2f}%")
    
    target_nan_percentage = np.isnan(y_grouped).mean() * 100
    print(f"Percentage of NaNs in grouped targets: {target_nan_percentage:.2f}%")
    
    # Fill NaNs with zeros
    print("Filling remaining NaNs with zeros...")
    data_3d = np.nan_to_num(data_3d, nan=0.0)
    
    # Create dataset and loader
    dataset = WindowsDataset(data_3d, y_grouped, window_size=window_size)
    print(f"Dataset contains {len(dataset)} windows")
    
    # Check if we have enough valid windows
    if len(dataset) < 1000:
        print("WARNING: Very few valid windows found. Check your data and preprocessing.")
    
    if len(dataset) == 0:
        raise ValueError("No valid windows found in dataset. Check for NaN values.")
    
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    # Get feature dimension from first window
    sample_window, _ = dataset[0]
    feature_dim = sample_window.shape[-1]
    print(f"Feature dimension: {feature_dim}")
    
    return data_loader, feature_dim

#############################################
# Training Function
#############################################
def train_model(train_loader, latent_dim, window_size, feature_dim, device, 
                num_epochs=5, lr=1e-4, hidden_dim=64, save_path="gan_model.pt"):
    """
    Train a GAN model on time series data.
    
    Args:
        train_loader: DataLoader for training data
        latent_dim: Dimension of the latent space
        window_size: Size of the sliding window
        feature_dim: Number of features in the data
        device: Device to train on
        num_epochs: Number of epochs to train for
        lr: Learning rate
        hidden_dim: Hidden dimension of the model
        save_path: Path to save the model
        
    Returns:
        model: Trained GAN model
    """
    # Initialize model
    gan_model = TimeSeriesGANBaseline(
        latent_dim=latent_dim, 
        window_size=window_size, 
        feature_dim=feature_dim, 
        hidden_dim=hidden_dim
    )
    gan_model.to(device)
    
    # Optimizers
    optimizer_G = optim.Adam(gan_model.generator.parameters(), lr=lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(gan_model.discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    
    # Training loop
    print("Starting training...")
    best_loss_G = float('inf')
    
    for epoch in range(num_epochs):
        gan_model.train()
        total_loss_D = 0.0
        total_loss_G = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (windows, _) in enumerate(progress_bar):
            windows = windows.to(device)
            cur_batch = windows.size(0)

            # Skip batches with NaNs
            if torch.isnan(windows).any():
                nan_counts = torch.isnan(windows).sum().item()
                print(f"Batch {batch_idx} contains {nan_counts} NaN values. Skipping...")
                continue
    
            # Label smoothing for more stable training
            valid = torch.ones(cur_batch, 1, device=device) * 0.9
            fake = torch.zeros(cur_batch, 1, device=device) + 0.1
            
            # Train Discriminator
            optimizer_D.zero_grad()
            
            # Train on real windows
            pred_real = gan_model(windows)
            pred_real = torch.clamp(pred_real, 1e-7, 1.0 - 1e-7)
            loss_real = criterion(pred_real, valid)
            
            # Train on fake windows
            noise = torch.randn(cur_batch, latent_dim, device=device)
            fake_windows = gan_model.generate(noise)
            pred_fake = gan_model(fake_windows.detach())
            pred_fake = torch.clamp(pred_fake, 1e-7, 1.0 - 1e-7)
            loss_fake = criterion(pred_fake, fake)
            
            # Combined discriminator loss
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()
            
            # Train Generator
            optimizer_G.zero_grad()
            noise = torch.randn(cur_batch, latent_dim, device=device)
            gen_windows = gan_model.generate(noise)
            
            pred_gen = gan_model(gen_windows)
            pred_gen = torch.clamp(pred_gen, 1e-7, 1.0 - 1e-7)
            loss_G = criterion(pred_gen, valid)
            loss_G.backward()
            optimizer_G.step()
            
            # Update metrics
            total_loss_D += loss_D.item() * cur_batch
            total_loss_G += loss_G.item() * cur_batch
            
            # Update progress bar
            progress_bar.set_postfix({
                'D_loss': loss_D.item(), 
                'G_loss': loss_G.item()
            })
        
        # Calculate average losses
        avg_loss_D = total_loss_D / len(train_loader.dataset)
        avg_loss_G = total_loss_G / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}: D Loss = {avg_loss_D:.4f}, G Loss = {avg_loss_G:.4f}")
        
        # Save best model
        if avg_loss_G < best_loss_G:
            best_loss_G = avg_loss_G
            torch.save({
                'generator_state_dict': gan_model.generator.state_dict(),
                'discriminator_state_dict': gan_model.discriminator.state_dict(),
                'optimizer_G_state_dict': optimizer_G.state_dict(),
                'optimizer_D_state_dict': optimizer_D.state_dict(),
                'latent_dim': latent_dim,
                'window_size': window_size,
                'feature_dim': feature_dim,
                'hidden_dim': hidden_dim,
                'epoch': epoch,
                'loss_G': avg_loss_G,
                'loss_D': avg_loss_D
            }, save_path)
            print(f"Saved best model at epoch {epoch+1} with G loss: {avg_loss_G:.4f}")
    
    # Load best model
    checkpoint = torch.load(save_path)
    gan_model.generator.load_state_dict(checkpoint['generator_state_dict'])
    gan_model.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    
    return gan_model

#############################################
# Evaluation Function
#############################################
def evaluate_model(model, test_loader, latent_dim, device):
    """
    Evaluate a trained GAN model.
    
    Args:
        model: Trained GAN model
        test_loader: DataLoader for test data
        latent_dim: Dimension of the latent space
        device: Device to evaluate on
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    metrics = {}
    
    # Reconstruction error
    print("Evaluating reconstruction error...")
    recon_mse_list = []
    recon_mae_list = []
    feature_mse_list = []
    
    with torch.no_grad():
        for windows, targets in tqdm(test_loader, desc="Evaluating"):
            if torch.isnan(windows).any():
                continue
                
            windows = windows.to(device)
            targets = targets.to(device)
            cur_batch = windows.size(0)
            
            # Generate samples
            noise = torch.randn(cur_batch, latent_dim, device=device)
            gen_windows = model.generate(noise)
            
            # Compute MSE and MAE
            mse = nn.MSELoss()(gen_windows, windows).item()
            mae = nn.L1Loss()(gen_windows, windows).item()
            recon_mse_list.append(mse)
            recon_mae_list.append(mae)
            
            # Compute MSE per feature
            for i in range(gen_windows.shape[2]):
                feature_mse = nn.MSELoss()(gen_windows[:, :, i], windows[:, :, i]).item()
                feature_mse_list.append(feature_mse)
    
    # Calculate average metrics
    if recon_mse_list:
        metrics['reconstruction_mse'] = np.mean(recon_mse_list)
        metrics['reconstruction_mae'] = np.mean(recon_mae_list)
        metrics['avg_feature_mse'] = np.mean(feature_mse_list)
        print(f"Reconstruction MSE: {metrics['reconstruction_mse']:.6f}")
        print(f"Reconstruction MAE: {metrics['reconstruction_mae']:.6f}")
        print(f"Average Feature MSE: {metrics['avg_feature_mse']:.6f}")
    else:
        print("No valid windows for evaluation.")
    
    # Evaluate return prediction
    print("Evaluating return prediction capability...")
    total_mse = 0.0
    total_mae = 0.0
    total_samples = 0
    
    # Simple approach: direct comparison with target returns
    with torch.no_grad():
        for windows, targets in tqdm(test_loader, desc="Return prediction"):
            if torch.isnan(windows).any() or torch.isnan(targets).any():
                continue
                
            windows = windows.to(device)
            targets = targets.to(device)
            cur_batch = windows.size(0)
            
            # Generate samples from noise
            noise = torch.randn(cur_batch, latent_dim, device=device)
            gen_windows = model.generate(noise)
            
            # Create a simple regression model to predict returns
            # Flatten the generated windows
            flattened = gen_windows.view(cur_batch, -1)
            
            # Simple linear regression for return prediction
            try:
                X = flattened.cpu().numpy()
                y = targets.cpu().numpy().flatten()
                
                # Skip if there are any NaNs
                if np.isnan(X).any() or np.isnan(y).any():
                    continue
                
                # Add bias term
                X_with_bias = np.column_stack([X, np.ones(X.shape[0])])
                
                # Solve for coefficients
                coeffs = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
                
                # Make predictions
                y_pred = X_with_bias @ coeffs
                
                # Calculate error
                mse = np.mean((y - y_pred) ** 2)
                mae = np.mean(np.abs(y - y_pred))
                
                total_mse += mse * cur_batch
                total_mae += mae * cur_batch
                total_samples += cur_batch
            except np.linalg.LinAlgError:
                continue
    
    # Record return prediction metrics
    if total_samples > 0:
        metrics['return_prediction_mse'] = total_mse / total_samples
        metrics['return_prediction_mae'] = total_mae / total_samples
        print(f"Return Prediction MSE: {metrics['return_prediction_mse']:.6f}")
        print(f"Return Prediction MAE: {metrics['return_prediction_mae']:.6f}")
    else:
        print("Could not evaluate return prediction due to numerical issues.")
    
    return metrics

#############################################
# Main Function
#############################################
def main():
    # Configuration
    window_size = 10
    latent_dim = 64
    batch_size = 64
    num_epochs = 5
    lr = 1e-4
    hidden_dim = 64
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare training data
    print("\n--- Preparing Training Data ---")
    train_file = f"{DATA_DIR}/dict_of_data_Jan2025_part1.npy"
    train_loader, feature_dim = prepare_data(train_file, window_size, batch_size, shuffle=True)
    
    # Train model
    print("\n--- Training Model ---")
    gan_model = train_model(
        train_loader=train_loader,
        latent_dim=latent_dim,
        window_size=window_size,
        feature_dim=feature_dim,
        device=device,
        num_epochs=num_epochs,
        lr=lr,
        hidden_dim=hidden_dim,
        save_path="gan_model.pt"
    )
    
    # Prepare test data
    print("\n--- Preparing Test Data ---")
    test_file = f"{DATA_DIR}/dict_of_data_Jan2025_part2.npy"
    test_loader, _ = prepare_data(test_file, window_size, batch_size, shuffle=False)
    
    # Evaluate model
    print("\n--- Evaluating Model ---")
    metrics = evaluate_model(
        model=gan_model,
        test_loader=test_loader,
        latent_dim=latent_dim,
        device=device
    )
    
    # Print final results
    print("\n--- Final Results ---")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")

if __name__ == "__main__":
    main()