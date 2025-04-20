import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_load import load_data, get_data
from config import DATA_DIR
from src.feature_encoder import Preprocessor, group_by_stock

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
# Baseline LSTM Model
#############################################
class LSTMBaseline(nn.Module):
    """
    A baseline LSTM for time series prediction.
    
    This model uses an LSTM network to process a sliding window of input features
    and outputs a prediction (e.g., the next-day return). The architecture consists of:
      - An LSTM layer (or stack) operating on the input sequence.
      - A fully connected layer that maps the hidden state of the final timestep to a scalar output.
    
    Parameters:
        feature_dim : int
            Number of features per time step.
        hidden_dim : int, default=64
            Dimensionality of the LSTM hidden state.
        num_layers : int, default=1
            Number of LSTM layers.
        dropout : float, default=0.0
            Dropout applied between LSTM layers.
    """
    def __init__(self, feature_dim, hidden_dim=64, num_layers=1, dropout=0.0):
        super(LSTMBaseline, self).__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x shape: (batch, window_size, feature_dim)
        lstm_out, _ = self.lstm(x)
        # Use output from the last timestep
        out_last = lstm_out[:, -1, :]  # shape: (batch, hidden_dim)
        pred = self.fc(out_last)       # shape: (batch, 1)
        return pred

#############################################
# Data Preparation Function
#############################################
def prepare_data(file_path, window_size, batch_size, shuffle=True):
    """
    Load and prepare data for LSTM training or testing.
    
    Args:
        file_path (str): Path to the data file.
        window_size (int): Size of the sliding window.
        batch_size (int): Batch size for the DataLoader.
        shuffle (bool): Whether to shuffle the data.
        
    Returns:
        DataLoader for the prepared dataset.
        feature_dim: Number of features per time step.
    """
    # Load data
    print(f"Loading data from {file_path}...")
    data_dict = load_data(file_path)
    x_data, y_data, si, di, raw_data, list_of_data = get_data(data_dict)
    
    # Use only the alpha signals as features
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
        raw_columns=[]  # Only use alphas
    )
    preprocessed = preprocessor.fit_transform(df_features)
    
    df_preprocessed = pd.DataFrame(preprocessed, columns=df_features.columns)
    df_preprocessed["stock_index"] = df_ids["stock_index"].values
    df_preprocessed["day_index"] = df_ids["day_index"].values
    
    # Group data by stock to create a 3D array: (num_stocks, num_days, num_features)
    data_3d = group_by_stock(df_preprocessed)
    
    # Group target returns
    target_grouped = group_targets(si, di, y_data)
    
    # Fill any remaining NaNs with zeros
    data_3d = np.nan_to_num(data_3d, nan=0.0)
    
    # Create the dataset and loader
    dataset = WindowsDataset(data_3d, target_grouped, window_size=window_size)
    print(f"Dataset contains {len(dataset)} windows")
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    # Get feature dimension (number of columns per time step)
    feature_dim = data_3d.shape[-1]
    return data_loader, feature_dim

#############################################
# Training Function
#############################################
def train_model(train_loader, model, device, num_epochs=5, lr=1e-3, model_save_path="lstm_model.pt"):
    """
    Train the LSTM model on the training dataset.
    
    Args:
        train_loader (DataLoader): DataLoader for training data.
        model (nn.Module): The LSTM model.
        device (torch.device): The device to run training on.
        num_epochs (int): Number of training epochs.
        lr (float): Learning rate.
        model_save_path (str): File path to save the best model.
        
    Returns:
        The trained model.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for windows, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            windows = windows.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            pred = model(windows)
            loss = criterion(pred, targets)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * windows.size(0)
        avg_loss = epoch_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.6f}")
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved best model at epoch {epoch+1} with Loss = {best_loss:.6f}")
    model.load_state_dict(torch.load(model_save_path))
    return model

#############################################
# Evaluation Function
#############################################
def evaluate_model(test_loader, model, device):
    """
    Evaluate the LSTM model on the test dataset by computing the MSE.
    
    Args:
        test_loader (DataLoader): DataLoader for test data.
        model (nn.Module): The trained LSTM model.
        device (torch.device): The device to run evaluation on.
        
    Returns:
        The average test MSE.
    """
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for windows, targets in test_loader:
            windows = windows.to(device)
            targets = targets.to(device)
            pred = model(windows)
            loss = criterion(pred, targets)
            total_loss += loss.item() * windows.size(0)
            total_samples += windows.size(0)
    avg_loss = total_loss / total_samples
    print(f"Test MSE: {avg_loss:.6f}")
    return avg_loss

#############################################
# Plotting Function
#############################################
def plot_lstm_predictions(model, test_loader, device, num_samples=10):
    """
    Plot a comparison of true versus LSTM-predicted returns from sample windows.
    
    This function collects predictions on a batch from the test loader and plots
    the first num_samples predictions alongside the corresponding true returns.
    
    Args:
        model (nn.Module): The trained LSTM model.
        test_loader (DataLoader): DataLoader for test data.
        device (torch.device): The computation device.
        num_samples (int): Number of samples to plot.
    """
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for windows, targets in test_loader:
            windows = windows.to(device)
            targets = targets.to(device)
            pred = model(windows)
            preds.append(pred.cpu().numpy())
            trues.append(targets.cpu().numpy())
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
    
    plt.figure(figsize=(10,6))
    plt.plot(trues[:num_samples], 'bo-', label="True Returns")
    plt.plot(preds[:num_samples], 'rx--', label="Predicted Returns")
    plt.xlabel("Sample Index")
    plt.ylabel("Return")
    plt.title("LSTM Predictions vs. True Returns")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

#############################################
# Main Function
#############################################
def main():
    # Configuration
    window_size = 10
    batch_size = 64
    num_epochs = 10
    lr = 1e-3
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Prepare training and test data
    train_file = f"{DATA_DIR}/dict_of_data_Jan2025_part1.npy"
    test_file = f"{DATA_DIR}/dict_of_data_Jan2025_part2.npy"
    
    print("\n--- Preparing Training Data ---")
    train_loader, feature_dim = prepare_data(train_file, window_size, batch_size, shuffle=True)
    
    print("\n--- Preparing Test Data ---")
    test_loader, _ = prepare_data(test_file, window_size, batch_size, shuffle=False)
    
    # Initialize the LSTM model
    model = LSTMBaseline(feature_dim, hidden_dim=64, num_layers=2, dropout=0.0)
    model.to(device)
    
    print("\n--- Training LSTM Model ---")
    model = train_model(train_loader, model, device, num_epochs=num_epochs, lr=lr, model_save_path="lstm_model.pt")
    
    print("\n--- Evaluating LSTM Model on Test Data ---")
    evaluate_model(test_loader, model, device)
    
    print("\n--- Plotting LSTM Predictions vs. True Returns ---")
    plot_lstm_predictions(model, test_loader, device, num_samples=10)

if __name__ == "__main__":
    main()
