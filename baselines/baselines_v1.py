# baselines_v1.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import data loading functions and DATA_DIR from your project
from data_load import load_data, get_data
from config import DATA_DIR

# Import the Preprocessor, group_by_stock, and get_feature_encoder from feature_encoder.py
from feature_encoder import Preprocessor, group_by_stock, get_feature_encoder

#############################################
# Custom Dataset: WindowsDataset
#############################################

class WindowsDataset(Dataset):
    """
    Computes sliding windows on the fly from grouped 3D data.
    
    The grouped data is assumed to have shape:
         (num_stocks, num_days, num_features)
    The grouped targets are assumed to have shape:
         (num_stocks, num_days)
         
    For each stock, a window of length `window_size` is extracted, and the target is the
    return on the day immediately following the window.
    """
    def __init__(self, grouped_data, grouped_targets, window_size=10):
        self.grouped_data = grouped_data
        self.grouped_targets = grouped_targets
        self.window_size = window_size
        
        # Build a mapping of (stock_idx, window_start) for all valid windows.
        self.indices = []
        num_stocks, num_days, _ = grouped_data.shape
        for stock in range(num_stocks):
            for start in range(num_days - window_size):
                # Only consider windows where the target (next day) is not NaN.
                if not np.isnan(grouped_targets[stock, start + window_size]):
                    self.indices.append((stock, start))
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        stock, start = self.indices[idx]
        window = self.grouped_data[stock, start:start + self.window_size, :]
        target = self.grouped_targets[stock, start + self.window_size]
        return torch.tensor(window, dtype=torch.float32), torch.tensor(target, dtype=torch.float32).unsqueeze(0)

#############################################
# Baseline 1: GAN for Time Series
#############################################

class TimeSeriesGANBaseline(nn.Module):
    """
    GAN Baseline for Time Series Prediction.
    
    Generator: Maps a noise vector to a forecast window.
    Discriminator: Tries to distinguish real from generated forecast windows.
    
    This is a simplified skeleton; in practice, you would condition the generator on past data.
    """
    def __init__(self, latent_dim, window_size, feature_dim, hidden_dim=64):
        super(TimeSeriesGANBaseline, self).__init__()
        self.latent_dim = latent_dim
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.generator = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, window_size * feature_dim),
            nn.Tanh()  # Assuming features scaled to [-1,1]
        )
        self.discriminator = nn.Sequential(
            nn.Linear(window_size * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    
    def generate(self, noise):
        gen_out = self.generator(noise)
        return gen_out.view(-1, self.window_size, self.feature_dim)
    
    def forward(self, x):
        # x is a real forecast window; flatten it and score.
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        return self.discriminator(x_flat)

#############################################
# Baseline 2: Diffusion Model
#############################################

class DiffusionModelBaseline(nn.Module):
    """
    Diffusion Model Baseline for Time Series Prediction.
    
    This simplified network attempts to iteratively refine a noisy input window
    toward the true next-day return window.
    
    In a full diffusion model, you would implement multiple reverse steps and a noise schedule.
    """
    def __init__(self, window_size, feature_dim, hidden_dim=64):
        super(DiffusionModelBaseline, self).__init__()
        self.window_size = window_size
        self.feature_dim = feature_dim
        self.net = nn.Sequential(
            nn.Linear(window_size * feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, window_size * feature_dim)
        )
    
    def forward(self, x):
        # x shape: (batch, window_size, feature_dim)
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)
        refined_flat = self.net(x_flat)
        refined = refined_flat.view(batch_size, self.window_size, self.feature_dim)
        return refined

#############################################
# Baseline 3: Hawkes Process Baseline
#############################################

class HawkesProcessBaseline:
    """
    Hawkes Process Baseline for modeling self-exciting events in stock returns.
    
    This implementation fits a very simplified Hawkes process using extreme events.
    """
    def __init__(self, threshold=0.05, decay=1.0):
        self.threshold = threshold
        self.decay = decay
        self.base_intensity = None
        self.alpha = None

    def fit(self, returns, timestamps):
        """
        Fit a Hawkes process on extreme events extracted from returns.
        
        Parameters:
            returns: np.array of shape (N,), the return time series for one stock.
            timestamps: np.array of shape (N,), corresponding time stamps.
        """
        # Identify extreme events (absolute return exceeds threshold)
        event_indices = np.where(np.abs(returns) >= self.threshold)[0]
        if len(event_indices) < 2:
            self.base_intensity = 0.0
            self.alpha = 0.0
            return
        event_times = timestamps[event_indices]
        inter_event = np.diff(event_times)
        self.base_intensity = 1.0 / np.mean(inter_event) if np.mean(inter_event) > 0 else 0.0
        self.alpha = 0.5  # Dummy fixed value for excitation parameter

    def predict(self, t):
        """
        Predict intensity at time t.
        """
        return self.base_intensity * np.exp(-self.decay * t) + self.alpha

#############################################
# Main Function: Baseline Prediction Pipeline
#############################################

def main():
    window_size = 10  # Length of input window (days)
    
    # --- Load and Process Training Data ---
    train_file = f"{DATA_DIR}/dict_of_data_Jan2025_part1.npy"
    train_dict = load_data(train_file)
    x_train, y_train, si_train, di_train, raw_train, list_of_data = get_data(train_dict)
    
    # Build DataFrames for 200 alphas and 11 raw variables
    num_alphas = x_train.shape[1]
    alpha_cols = [f'alpha_{i+1}' for i in range(num_alphas)]
    df_alphas = pd.DataFrame(x_train, columns=alpha_cols)
    df_raw = pd.DataFrame(raw_train, columns=list_of_data)
    df_train = pd.concat([df_alphas, df_raw], axis=1)
    df_train["stock_index"] = si_train
    df_train["day_index"] = di_train
    
    # Preprocess features (scale alphas, leave raw variables untouched)
    df_ids = df_train[["stock_index", "day_index"]]
    df_features = df_train.drop(columns=["stock_index", "day_index"])
    preprocessor = Preprocessor(
        imputation_strategy='mean',
        scaling_method='standard',
        alpha_prefix='alpha_',
        raw_columns=list_of_data
    )
    preprocessed = preprocessor.fit_transform(df_features)
    df_preprocessed = pd.DataFrame(preprocessed, columns=df_features.columns)
    df_preprocessed["stock_index"] = df_ids["stock_index"].values
    df_preprocessed["day_index"] = df_ids["day_index"].values
    
    # Group training data by stock into a 3D array: (num_stocks, num_days, num_features)
    train_3d = group_by_stock(df_preprocessed)
    print("Grouped training data shape:", train_3d.shape)
    
    # Group training targets similarly (pivot table for y_data)
    def group_targets(si, di, y):
        df_t = pd.DataFrame({"stock_index": si, "day_index": di, "y": y})
        df_pivot = df_t.pivot_table(index="stock_index", columns="day_index", values="y", aggfunc="first")
        df_pivot = df_pivot.sort_index(axis=1)
        return df_pivot.values
    y_train_grouped = group_targets(si_train, di_train, y_train)
    print("Grouped y_train shape:", y_train_grouped.shape)
    
    # Create WindowsDataset for training (computing windows on the fly)
    train_dataset = WindowsDataset(train_3d, y_train_grouped, window_size=window_size)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # --- Similarly Process Test Data ---
    test_file = f"{DATA_DIR}/dict_of_data_Jan2025_part2.npy"
    test_dict = load_data(test_file)
    x_test, y_test, si_test, di_test, raw_test, _ = get_data(test_dict)
    df_alphas_test = pd.DataFrame(x_test, columns=alpha_cols)
    df_raw_test = pd.DataFrame(raw_test, columns=list_of_data)
    df_test = pd.concat([df_alphas_test, df_raw_test], axis=1)
    df_test["stock_index"] = si_test
    df_test["day_index"] = di_test
    df_test_ids = df_test[["stock_index", "day_index"]]
    df_test_features = df_test.drop(columns=["stock_index", "day_index"])
    preprocessed_test = preprocessor.transform(df_test_features)
    df_test_preprocessed = pd.DataFrame(preprocessed_test, columns=df_test_features.columns)
    df_test_preprocessed["stock_index"] = df_test_ids["stock_index"].values
    df_test_preprocessed["day_index"] = df_test_ids["day_index"].values
    test_3d = group_by_stock(df_test_preprocessed)
    print("Grouped test data shape:", test_3d.shape)
    
    y_test_grouped = group_targets(si_test, di_test, y_test)
    print("Grouped y_test shape:", y_test_grouped.shape)
    
    test_dataset = WindowsDataset(test_3d, y_test_grouped, window_size=window_size)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #####################################################
    # Baseline 1: GAN for Time Series
    #####################################################
    print("\nTraining GAN for Time Series Baseline")
    # For GAN, we use the windows as "real" forecast windows.
    # Let latent_dim be 64; the generator maps noise to a forecast window.
    latent_dim = 64
    feature_dim = train_dataset[0][0].shape[-1]
    gan_window_size = window_size  # forecasting the same length window
    gan_model = TimeSeriesGANBaseline(latent_dim, gan_window_size, feature_dim, hidden_dim=64).to(device)
    
    # Optimizers for generator and discriminator
    optimizer_G = optim.Adam(gan_model.generator.parameters(), lr=1e-3)
    optimizer_D = optim.Adam(gan_model.discriminator.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    num_epochs = 5  # For demonstration; adjust as needed
    
    for epoch in range(num_epochs):
        gan_model.train()
        total_loss_G = 0.0
        total_loss_D = 0.0
        for real_windows, _ in train_loader:
            real_windows = real_windows.to(device)
            batch_size = real_windows.size(0)
            # Labels for real (1) and fake (0)
            valid = torch.ones(batch_size, 1, device=device)
            fake = torch.zeros(batch_size, 1, device=device)
            
            # --- Train Discriminator ---
            optimizer_D.zero_grad()
            # Real loss
            pred_real = gan_model.discriminator(real_windows.view(batch_size, -1))
            loss_real = criterion(pred_real, valid)
            # Fake loss: generate fake windows
            noise = torch.randn(batch_size, latent_dim, device=device)
            fake_windows = gan_model.generate(noise)
            pred_fake = gan_model.discriminator(fake_windows.view(batch_size, -1))
            loss_fake = criterion(pred_fake, fake)
            loss_D = (loss_real + loss_fake) / 2
            loss_D.backward()
            optimizer_D.step()
            
            # --- Train Generator ---
            optimizer_G.zero_grad()
            noise = torch.randn(batch_size, latent_dim, device=device)
            gen_windows = gan_model.generate(noise)
            # Generator wants discriminator to label fakes as real
            pred_gen = gan_model.discriminator(gen_windows.view(batch_size, -1))
            loss_G = criterion(pred_gen, valid)
            loss_G.backward()
            optimizer_G.step()
            
            total_loss_D += loss_D.item() * batch_size
            total_loss_G += loss_G.item() * batch_size
        
        print(f"GAN Epoch {epoch+1}/{num_epochs} - D Loss: {total_loss_D/len(train_loader.dataset):.4f}, G Loss: {total_loss_G/len(train_loader.dataset):.4f}")
    
    # For evaluation, we can generate forecast windows from noise and compare MSE against test windows.
    gan_model.eval()
    with torch.no_grad():
        all_gan_preds = []
        all_targets = []
        for windows, targets in test_loader:
            batch_size = windows.size(0)
            noise = torch.randn(batch_size, latent_dim, device=device)
            gen_windows = gan_model.generate(noise)
            # Here, we compare the generated forecast window to the true next-day window.
            # For simplicity, we compute MSE on the entire window.
            mse = nn.MSELoss()(gen_windows, windows)  # using the window as a proxy target
            all_gan_preds.append(mse.item())
            all_targets.append(targets.mean().item())
        gan_test_mse = np.mean(all_gan_preds)
    print("GAN Baseline Test MSE (proxy on window):", gan_test_mse)
    
    #####################################################
    # Baseline 2: Diffusion Model
    #####################################################
    print("\nTraining Diffusion Model Baseline")
    diffusion_model = DiffusionModelBaseline(window_size, feature_dim, hidden_dim=64).to(device)
    optimizer_diff = optim.Adam(diffusion_model.parameters(), lr=1e-3)
    num_epochs_diff = 5
    for epoch in range(num_epochs_diff):
        diffusion_model.train()
        total_loss = 0.0
        for windows, _ in train_loader:
            windows = windows.to(device)
            optimizer_diff.zero_grad()
            # Add noise to windows
            noise = torch.randn_like(windows) * 0.1
            noisy_input = windows + noise
            refined = diffusion_model(noisy_input)
            loss = nn.MSELoss()(refined, windows)
            loss.backward()
            optimizer_diff.step()
            total_loss += loss.item() * windows.size(0)
        print(f"Diffusion Epoch {epoch+1}/{num_epochs_diff}, Loss: {total_loss/len(train_loader.dataset):.4f}")
    
    diffusion_model.eval()
    with torch.no_grad():
        total_loss = 0.0
        for windows, _ in test_loader:
            windows = windows.to(device)
            noise = torch.randn_like(windows) * 0.1
            refined = diffusion_model(noise + windows)
            loss = nn.MSELoss()(refined, windows)
            total_loss += loss.item() * windows.size(0)
        diffusion_test_mse = total_loss / len(test_loader.dataset)
    print("Diffusion Baseline Test MSE:", diffusion_test_mse)
    
    #####################################################
    # Baseline 3: Hawkes Process Baseline
    #####################################################
    print("\nFitting Hawkes Process Baseline")
    # For the Hawkes process, we process each stock separately.
    # Here, we define an extreme return threshold.
    threshold = 0.05
    hawkes_intensities = []
    num_stocks, num_days, _ = train_3d.shape
    # We also need a time vector. Assume day indices are sequential.
    time_vector = np.arange(num_days)
    for stock in range(num_stocks):
        # Get the time series of returns for this stock from grouped targets.
        stock_returns = y_train_grouped[stock]
        # Identify extreme events where abs(return) exceeds threshold.
        extreme_indices = np.where(np.abs(stock_returns) >= threshold)[0]
        if len(extreme_indices) > 0:
            # Fit Hawkes on these event times.
            hawkes = HawkesProcessBaseline(threshold=threshold, decay=1.0)
            hawkes.fit(stock_returns, time_vector)
            # Predict intensity at the last time point (as a proxy forecast).
            intensity = hawkes.predict(t= time_vector[-1])
        else:
            intensity = 0.0
        hawkes_intensities.append(intensity)
    hawkes_intensities = np.array(hawkes_intensities)
    # For evaluation, compare to the mean return per stock (as a dummy proxy).
    stock_mean_returns = np.nanmean(y_train_grouped, axis=1)
    hawkes_mse = np.mean((hawkes_intensities - stock_mean_returns)**2)
    print("Hawkes Process Baseline MSE (proxy):", hawkes_mse)

if __name__ == "__main__":
    main()