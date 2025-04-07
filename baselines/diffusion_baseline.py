# baselines/diffusion_baseline.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math

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
                target_idx = start + window_size
                if target_idx < num_days and not np.isnan(grouped_targets[stock, target_idx]):
                    # Check that the window contains no NaNs
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
# Diffusion Scheduler
#############################################
class DiffusionScheduler:
    """
    Implements a noise scheduler for the diffusion process.
    
    This uses a cosine schedule for beta (noise level) as described in
    the Improved DDPM paper, which provides better sample quality.
    """
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, schedule_type='cosine'):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        
        if schedule_type == 'linear':
            # Linear schedule from Ho et al.
            self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        elif schedule_type == 'cosine':
            # Cosine schedule from Improved DDPM
            steps = num_timesteps + 1
            x = torch.linspace(0, num_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            self.betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
        
        # Pre-compute diffusion parameters
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
    
    def q_sample(self, x_0, t, noise=None):
        """
        Forward diffusion process: sample q(x_t | x_0)
        
        Args:
            x_0: The original clean data
            t: The timestep
            noise: The noise to add (if None, it will be generated)
            
        Returns:
            The noisy sample x_t
        """
        if noise is None:
            noise = torch.randn_like(x_0)
            
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
    
    def q_posterior_mean(self, x_0, x_t, t):
        """
        Compute the mean of q(x_{t-1} | x_t, x_0)
        """
        posterior_mean_coef1_t = self._extract(self.posterior_mean_coef1, t, x_0.shape)
        posterior_mean_coef2_t = self._extract(self.posterior_mean_coef2, t, x_0.shape)
        
        mean = posterior_mean_coef1_t * x_0 + posterior_mean_coef2_t * x_t
        return mean
    
    def _extract(self, a, t, broadcast_shape):
        """
        Extract timestep-specific values and reshape for broadcasting
        """
        out = torch.gather(a, 0, t)
        while len(out.shape) < len(broadcast_shape):
            out = out.unsqueeze(-1)
        return out.expand(broadcast_shape)
    
    def p_mean_variance(self, model, x_t, t, model_kwargs=None):
        """
        Compute the mean and variance of p(x_{t-1} | x_t)
        """
        model_output = model(x_t, t)
        model_variance = torch.exp(self._extract(self.posterior_log_variance_clipped, t, x_t.shape))
        
        # Use the model to predict x_0
        model_mean = self.q_posterior_mean(model_output, x_t, t)
        
        return {
            'mean': model_mean,
            'variance': model_variance,
            'log_variance': self._extract(self.posterior_log_variance_clipped, t, x_t.shape),
            'pred_x_0': model_output,
        }
    
    def p_sample(self, model, x_t, t, model_kwargs=None):
        """
        Sample from p(x_{t-1} | x_t) - the reverse diffusion process step
        """
        out = self.p_mean_variance(model, x_t, t, model_kwargs)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        
        # No noise when t == 0
        sample = out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        return {"sample": sample, "pred_x_0": out["pred_x_0"]}
    
    def p_sample_loop(self, model, shape, device, noise=None, progress=True):
        """
        Generate samples by sampling from p(x_{t-1} | x_t) sequentially
        """
        batch_size = shape[0]
        
        # Start from pure noise
        if noise is None:
            img = torch.randn(shape, device=device)
        else:
            img = noise
            
        # Iteratively denoise
        iterator = tqdm(reversed(range(self.num_timesteps)), desc='Sampling', total=self.num_timesteps) if progress else reversed(range(self.num_timesteps))
        
        for t in iterator:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            out = self.p_sample(model, img, t_batch)
            img = out["sample"]
            
        return img
    
    def loss(self, model, x_0, noise=None, loss_type="l2"):
        """
        Compute the diffusion loss for training
        """
        batch_size = x_0.shape[0]
        device = x_0.device
        
        # Sample random timesteps
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        
        if noise is None:
            noise = torch.randn_like(x_0)
            
        # Get noisy samples
        x_t = self.q_sample(x_0, t, noise)
        
        # Predict the noise (or x_0 for some models)
        model_output = model(x_t, t)
        
        if loss_type == "l1":
            loss = F.l1_loss(model_output, x_0)
        elif loss_type == "l2":
            loss = F.mse_loss(model_output, x_0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
            
        return loss

#############################################
# Improved Diffusion Model Components
#############################################
class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timestep t"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

class Block(nn.Module):
    """Basic residual block with normalization"""
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim, dim_out)
        self.norm = nn.LayerNorm(dim_out)
        self.act = nn.SiLU()  # SiLU = Swish activation
        
    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        x = self.act(x)
        return x

#############################################
# Diffusion Model Baseline
#############################################
class DiffusionModelBaseline(nn.Module):
    """
    Improved diffusion model for time series prediction.
    
    This model predicts x_0 directly from noisy input x_t and timestep t.
    It uses a more complex architecture with time embedding and residual blocks.
    """
    def __init__(self, window_size, feature_dim, hidden_dim=128, time_dim=32):
        super(DiffusionModelBaseline, self).__init__()
        self.window_size = window_size
        self.feature_dim = feature_dim
        
        # Time embedding
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        # Input projection
        self.input_proj = nn.Linear(window_size * feature_dim, hidden_dim)
        
        # Main network
        self.net = nn.Sequential(
            Block(hidden_dim + time_dim, hidden_dim),
            nn.Dropout(0.1),
            Block(hidden_dim, hidden_dim),
            nn.Dropout(0.1),
            Block(hidden_dim, hidden_dim),
            nn.Dropout(0.1),
        )
        
        # Output projection
        self.out = nn.Linear(hidden_dim, window_size * feature_dim)
    
    def forward(self, x, t=None):
        """
        Forward pass of the diffusion model.
        
        Args:
            x: Input tensor of shape (batch, window_size, feature_dim)
            t: Timestep tensor of shape (batch,)
        
        Returns:
            Output tensor of shape (batch, window_size, feature_dim)
        """
        # Handle the case when t is not provided (for compatibility)
        if t is None:
            t = torch.zeros(x.size(0), device=x.device).long()
            
        batch_size = x.shape[0]
        
        # Flatten input
        x_flat = x.view(batch_size, -1)  # (batch, window_size * feature_dim)
        
        # Project input
        h = self.input_proj(x_flat)  # (batch, hidden_dim)
        
        # Time embedding
        time_emb = self.time_mlp(t)  # (batch, time_dim)
        
        # Combine input and time embedding
        h = torch.cat((h, time_emb), dim=1)  # (batch, hidden_dim + time_dim)
        
        # Process through network
        h = self.net(h)  # (batch, hidden_dim)
        
        # Project to output
        x_pred = self.out(h)  # (batch, window_size * feature_dim)
        
        # Reshape to match input dimensions
        x_pred = x_pred.view(batch_size, self.window_size, self.feature_dim)
        
        return x_pred

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
# Data Processing Functions
#############################################
def prepare_data(file_path, window_size, batch_size):
    """
    Load and prepare data for training or testing.
    
    Args:
        file_path: Path to the data file
        window_size: Window size for the sliding window
        batch_size: Batch size for the data loader
        
    Returns:
        data_loader: DataLoader for the prepared dataset
        feature_dim: Number of features in the data
    """
    # Load data
    data_dict = load_data(file_path)
    x_data, y_data, si, di, raw_data, list_of_data = get_data(data_dict)
    
    # Use only alpha signals for simplicity
    num_alphas = x_data.shape[1]
    alpha_cols = [f'alpha_{i+1}' for i in range(num_alphas)]
    df_data = pd.DataFrame(x_data, columns=alpha_cols)
    df_data["stock_index"] = si
    df_data["day_index"] = di
    
    # Preprocess: scale alpha features
    df_ids = df_data[["stock_index", "day_index"]]
    df_features = df_data.drop(columns=["stock_index", "day_index"])
    preprocessor = Preprocessor(
        imputation_strategy='mean',
        scaling_method='standard',
        alpha_prefix='alpha_',
        raw_columns=[]
    )
    preprocessed = preprocessor.fit_transform(df_features)
    df_preprocessed = pd.DataFrame(preprocessed, columns=df_features.columns)
    df_preprocessed["stock_index"] = df_ids["stock_index"].values
    df_preprocessed["day_index"] = df_ids["day_index"].values
    
    # Group data by stock
    data_3d = group_by_stock(df_preprocessed)
    print(f"Grouped data shape: {data_3d.shape}")
    
    # Group targets
    target_grouped = group_targets(si, di, y_data)
    print(f"Grouped targets shape: {target_grouped.shape}")
    
    # Fill NaNs with zeros
    data_3d = np.nan_to_num(data_3d, nan=0.0)
    
    # Create dataset and loader
    dataset = WindowsDataset(data_3d, target_grouped, window_size=window_size)
    print(f"Dataset contains {len(dataset)} windows")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Get feature dimension
    feature_dim = data_3d.shape[-1]
    
    return data_loader, feature_dim

#############################################
# Training Function
#############################################
def train_model(train_loader, model, scheduler, device, num_epochs, lr=1e-4, model_save_path="best_diffusion_model.pt"):
    """
    Train the diffusion model.
    
    Args:
        train_loader: DataLoader for training data
        model: Diffusion model
        scheduler: Diffusion scheduler
        device: Device to train on
        num_epochs: Number of epochs to train for
        lr: Learning rate
        model_save_path: Path to save the best model
        
    Returns:
        model: Trained model
    """
    # Initialize optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("Starting training...")
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as t:
            for windows, _ in t:
                windows = windows.to(device)
                optimizer.zero_grad()
                
                # Calculate diffusion loss
                loss = scheduler.loss(model, windows, loss_type="l2")
                
                # Update model
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                # Track loss
                batch_loss = loss.item()
                epoch_loss += batch_loss * windows.shape[0]
                t.set_postfix(loss=batch_loss)
        
        # Calculate average loss for epoch
        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs} - Average Loss: {avg_epoch_loss:.6f}")
        
        # Update learning rate
        lr_scheduler.step(avg_epoch_loss)
        
        # Save best model
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model with loss: {best_loss:.6f}")
    
    # Load best model
    model.load_state_dict(torch.load(model_save_path))
    return model

#############################################
# Evaluation Function
#############################################
def evaluate_model(test_loader, model, scheduler, device, num_samples=5):
    """
    Evaluate the diffusion model.
    
    Args:
        test_loader: DataLoader for test data
        model: Diffusion model
        scheduler: Diffusion scheduler
        device: Device to evaluate on
        num_samples: Number of samples to visualize
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    metrics = {}
    
    # Generate sample predictions
    print("Generating sample predictions...")
    with torch.no_grad():
        # Sample windows from the dataset
        test_windows = []
        test_targets = []
        for i, (window, target) in enumerate(test_loader):
            if i >= num_samples:
                break
            test_windows.append(window[0])  # Take first window from batch
            test_targets.append(target[0])  # Take first target from batch
        
        # Stack windows into a batch
        test_batch = torch.stack(test_windows).to(device)
        test_targets = torch.stack(test_targets).to(device)
        
        # Add noise
        diffusion_steps = scheduler.num_timesteps
        t = torch.full((num_samples,), diffusion_steps-1, device=device, dtype=torch.long)
        noised_windows = scheduler.q_sample(test_batch, t)
        
        # Reconstruct through diffusion sampling
        sampled_windows = scheduler.p_sample_loop(
            model, 
            shape=test_batch.shape, 
            device=device,
            noise=noised_windows,
            progress=True
        )
        
        # Calculate reconstruction MSE
        recon_mse = F.mse_loss(sampled_windows, test_batch).item()
        print(f"Reconstruction MSE on {num_samples} samples: {recon_mse:.6f}")
        metrics['reconstruction_mse'] = recon_mse
        
        # Evaluate on larger test set
        print("Evaluating on full test set...")
        total_mse = 0.0
        total_samples = 0
        
        for windows, targets in tqdm(test_loader, desc="Testing"):
            windows = windows.to(device)
            targets = targets.to(device)
            batch_size = windows.shape[0]
            
            # Generate reconstructions from the model
            reconstructed = model(windows)
            
            # Compute MSE between reconstructions and original windows
            mse = F.mse_loss(reconstructed, windows).item()
            total_mse += mse * batch_size
            total_samples += batch_size
            
        avg_test_mse = total_mse / total_samples
        print(f"Average test MSE: {avg_test_mse:.6f}")
        metrics['avg_test_mse'] = avg_test_mse
        
        # Evaluate return prediction
        print("Evaluating return prediction...")
        total_pred_mse = 0.0
        total_mae = 0.0
        count = 0
        
        for windows, targets in tqdm(test_loader, desc="Return prediction"):
            windows = windows.to(device)
            targets = targets.to(device)
            batch_size = windows.shape[0]
            
            # Get representations from model (without timesteps for direct prediction)
            window_repr = model(windows)
            
            # Use a basic approach: calculate correlation between representation and target
            # Flatten the window representations
            flattened = window_repr.view(batch_size, -1)
            
            # Create a simple linear model on CPU
            X = flattened.cpu().numpy()
            y = targets.cpu().numpy().flatten()
            
            # Skip problematic batches
            if np.isnan(X).any() or np.isnan(y).any():
                continue
                
            # Simple linear regression using numpy
            # Add a bias term
            X_with_bias = np.column_stack([X, np.ones(X.shape[0])])
            
            try:
                # Solve for coefficients
                coeffs = np.linalg.lstsq(X_with_bias, y, rcond=None)[0]
                
                # Make predictions
                y_pred = X_with_bias @ coeffs
                
                # Calculate MSE and MAE
                pred_mse = np.mean((y - y_pred) ** 2)
                pred_mae = np.mean(np.abs(y - y_pred))
                
                total_pred_mse += pred_mse
                total_mae += pred_mae
                count += 1
            except np.linalg.LinAlgError:
                # Skip if there's a linear algebra error
                continue
        
        if count > 0:
            avg_pred_mse = total_pred_mse / count
            avg_mae = total_mae / count
            print(f"Average return prediction MSE: {avg_pred_mse:.6f}")
            print(f"Average return prediction MAE: {avg_mae:.6f}")
            metrics['return_pred_mse'] = avg_pred_mse
            metrics['return_pred_mae'] = avg_mae
        else:
            print("Could not evaluate return predictions due to numerical issues.")
            
    return metrics

#############################################
# Main Function
#############################################
def main():
    # Configuration
    window_size = 10
    batch_size = 64
    num_epochs = 5
    lr = 1e-4
    hidden_dim = 128
    time_dim = 32
    diffusion_steps = 100  # Reduced for faster training
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cpu':
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare training data
    print("\n--- Preparing Training Data ---")
    train_file = f"{DATA_DIR}/dict_of_data_Jan2025_part1.npy"
    train_loader, feature_dim = prepare_data(train_file, window_size, batch_size)
    
    # Initialize model and scheduler
    model = DiffusionModelBaseline(
        window_size=window_size,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        time_dim=time_dim
    )
    model.to(device)
    
    scheduler = DiffusionScheduler(
        num_timesteps=diffusion_steps,
        beta_start=1e-4,
        beta_end=0.02,
        schedule_type='cosine'
    )
    
    # Train model
    model = train_model(
        train_loader=train_loader,
        model=model,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        lr=lr,
        model_save_path="best_diffusion_model.pt"
    )
    
    # Prepare test data
    print("\n--- Preparing Test Data ---")
    test_file = f"{DATA_DIR}/dict_of_data_Jan2025_part2.npy"
    test_loader, _ = prepare_data(test_file, window_size, batch_size)
    
    # Evaluate model
    print("\n--- Evaluating Model ---")
    metrics = evaluate_model(
        test_loader=test_loader,
        model=model,
        scheduler=scheduler,
        device=device,
        num_samples=5
    )
    
    # Print final results
    print("\n--- Final Results ---")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")

if __name__ == "__main__":
    main()