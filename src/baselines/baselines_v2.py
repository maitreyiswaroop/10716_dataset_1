# baselines.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import data loading functions and config
from data_load import load_data, get_data
from config import DATA_DIR

# Import the Preprocessor, group_by_stock, and get_feature_encoder from feature_encoder.py
from src.feature_encoder import Preprocessor, group_by_stock, get_feature_encoder

#############################################
# Custom Dataset for Sliding Windows
#############################################
class WindowsDataset(Dataset):
    """
    Custom dataset that computes sliding windows on the fly from grouped data.
    
    The grouped data is assumed to have shape:
         (num_stocks, num_days, num_features)
    The grouped targets are assumed to have shape:
         (num_stocks, num_days)
         
    For each stock, this dataset yields windows of length `window_size` and the target
    is the return on the day immediately following the window.
    
    The dataset builds an index mapping of (stock_idx, start_idx) for all windows.
    """
    def __init__(self, grouped_data, grouped_targets, window_size=10):
        self.grouped_data = grouped_data
        self.grouped_targets = grouped_targets
        self.window_size = window_size
        
        # Build an index mapping: for each stock, count windows = (num_days - window_size)
        self.indices = []
        self.cum_counts = []
        total = 0
        num_stocks = grouped_data.shape[0]
        num_days = grouped_data.shape[1]
        for stock in range(num_stocks):
            # Use the full length (assume each stock has data for num_days)
            count = max(0, num_days - window_size)
            self.indices.append(count)
            total += count
            self.cum_counts.append(total)
        self.total_windows = total

    def __len__(self):
        return self.total_windows

    def __getitem__(self, idx):
        # Find which stock group this window belongs to using cumulative counts.
        stock_idx = np.searchsorted(self.cum_counts, idx, side='right')
        if stock_idx == 0:
            window_start = idx
        else:
            window_start = idx - self.cum_counts[stock_idx - 1]
        # Extract window and target:
        window = self.grouped_data[stock_idx, window_start: window_start + self.window_size, :]
        target = self.grouped_targets[stock_idx, window_start + self.window_size]
        # Convert to torch tensors
        window_tensor = torch.tensor(window, dtype=torch.float32)
        target_tensor = torch.tensor(target, dtype=torch.float32).unsqueeze(0)
        return window_tensor, target_tensor

#############################################
# Baseline Models
#############################################

class NaiveBaseline(nn.Module):
    """
    Naive Baseline: Predicts a constant value (mean return) for every sample.
    """
    def __init__(self, constant):
        super(NaiveBaseline, self).__init__()
        self.constant = constant

    def forward(self, x):
        batch_size = x.shape[0]
        return self.constant * torch.ones((batch_size, 1), device=x.device)

class LSTMBaseline(nn.Module):
    """
    LSTM Baseline: Processes an input window with an LSTM to predict next-day return.
    """
    def __init__(self, input_dim, hidden_dim=64, num_layers=1):
        super(LSTMBaseline, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        # x shape: (batch, window_size, input_dim)
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)

class TransformerBaseline(nn.Module):
    """
    Transformer Baseline: Uses a Transformer Encoder to process the window and predict next-day return.
    """
    def __init__(self, input_dim, nhead=4, num_layers=2, dim_feedforward=128):
        super(TransformerBaseline, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        # x shape: (batch, window_size, input_dim)
        # Transformer expects (window_size, batch, input_dim)
        x = x.permute(1, 0, 2)
        encoded = self.transformer(x)
        last_encoded = encoded[-1]  # (batch, input_dim)
        return self.fc(last_encoded)

class CNNMLPBaseline(nn.Module):
    """
    CNN+MLP Baseline (Hybrid): Uses the CNN-based feature encoder to extract a latent vector
    from the window, then an MLP to predict next-day return.
    """
    def __init__(self, input_dim, hidden_channels=64, latent_dim=64, kernel_size=3, num_layers=2, mlp_hidden=32):
        super(CNNMLPBaseline, self).__init__()
        self.encoder = get_feature_encoder(
            input_dim=input_dim,
            hidden_channels=hidden_channels,
            output_dim=latent_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            activation=nn.ReLU
        )
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)
        )
    
    def forward(self, x):
        # x shape: (batch, window_size, input_dim)
        latent = self.encoder(x)
        return self.mlp(latent)

#############################################
# Training and Evaluation Helpers
#############################################

def train_model(model, dataloader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_x.size(0)
        avg_loss = total_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0.0
    all_preds = []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            total_loss += loss.item() * batch_x.size(0)
            all_preds.append(preds)
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss, torch.cat(all_preds, dim=0)


# device setup 
def _setup_device():
    """Configure device(s) for training with priority: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"Found {n_gpus} CUDA GPU(s)")
        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"GPU {i}: {props.name}, {props.total_memory/1e9:.2f}GB memory")
        
        device = torch.device('cuda:0')  # Primary GPU
        torch.backends.cudnn.benchmark = True
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using MPS (Apple Silicon) for acceleration")
    else:
        device = torch.device('cpu')
        print("No GPU acceleration available, using CPU")
    
    return device

#############################################
# Main Function: Baseline Prediction Pipeline
#############################################

def main():
    window_size = 10  # Input window length (days)
    
    # Load training and test datasets from Dataset 1
    train_file = f"{DATA_DIR}/dict_of_data_Jan2025_part1.npy"
    test_file = f"{DATA_DIR}/dict_of_data_Jan2025_part2.npy"
    
    # --- Process Training Data ---
    train_dict = load_data(train_file)
    x_train, y_train, si_train, di_train, raw_train, list_of_data = get_data(train_dict)
    
    # Build DataFrame for 200 alpha signals and 11 raw variables
    num_alphas = x_train.shape[1]
    alpha_cols = [f'alpha_{i+1}' for i in range(num_alphas)]
    df_alphas_train = pd.DataFrame(x_train, columns=alpha_cols)
    df_raw_train = pd.DataFrame(raw_train, columns=list_of_data)
    df_train = pd.concat([df_alphas_train, df_raw_train], axis=1)
    df_train["stock_index"] = si_train
    df_train["day_index"] = di_train
    
    # Preprocess features (scale alphas, leave raw variables untouched)
    df_train_ids = df_train[["stock_index", "day_index"]]
    df_train_features = df_train.drop(columns=["stock_index", "day_index"])
    preprocessor = Preprocessor(
        imputation_strategy='mean',
        scaling_method='standard',
        alpha_prefix='alpha_',
        raw_columns=list_of_data
    )
    preprocessed_train = preprocessor.fit_transform(df_train_features)
    df_train_preprocessed = pd.DataFrame(preprocessed_train, columns=df_train_features.columns)
    df_train_preprocessed["stock_index"] = df_train_ids["stock_index"].values
    df_train_preprocessed["day_index"] = df_train_ids["day_index"].values
    
    # Group training data by stock into 3D array: (num_stocks, num_days, num_features)
    train_3d = group_by_stock(df_train_preprocessed)
    print("Grouped training data shape:", train_3d.shape)
    
    # Group training targets similarly
    def group_targets(si, di, y):
        df_t = pd.DataFrame({"stock_index": si, "day_index": di, "y": y})
        df_pivot = df_t.pivot_table(index="stock_index", columns="day_index", values="y", aggfunc="first")
        df_pivot = df_pivot.sort_index(axis=1)
        return df_pivot.values  # shape: (num_stocks, num_days)
    
    y_train_grouped = group_targets(si_train, di_train, y_train)
    print("Grouped y_train shape:", y_train_grouped.shape)
    
    # Create WindowsDataset for training windows (computes windows on the fly)
    train_dataset = WindowsDataset(train_3d, y_train_grouped, window_size=window_size)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    
    # --- Process Test Data ---
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
    
    #############################################
    # Baseline Model Training and Evaluation
    #############################################
    device = _setup_device()
    
    # 1. LSTM Baseline
    print("\nTraining LSTM Baseline")
    lstm_model = LSTMBaseline(input_dim=train_dataset[0][0].shape[-1], hidden_dim=64, num_layers=1).to(device)
    optimizer = optim.Adam(lstm_model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train_model(lstm_model, train_loader, criterion, optimizer, num_epochs=10)
    lstm_model.eval()
    lstm_loss, _ = evaluate_model(lstm_model, test_loader, criterion)
    print("LSTM Baseline Test MSE:", lstm_loss)
    
    # 2. Transformer Baseline
    print("\nTraining Transformer Baseline")
    transformer_model = TransformerBaseline(input_dim=train_dataset[0][0].shape[-1], nhead=4, num_layers=2, dim_feedforward=128).to(device)
    optimizer = optim.Adam(transformer_model.parameters(), lr=1e-3)
    train_model(transformer_model, train_loader, criterion, optimizer, num_epochs=10)
    transformer_model.eval()
    transformer_loss, _ = evaluate_model(transformer_model, test_loader, criterion)
    print("Transformer Baseline Test MSE:", transformer_loss)
    
    # 3. CNN+MLP Baseline (Hybrid)
    print("\nTraining CNN+MLP Baseline")
    cnnmlp_model = CNNMLPBaseline(input_dim=train_dataset[0][0].shape[-1], hidden_channels=64, latent_dim=64,
                                  kernel_size=3, num_layers=2, mlp_hidden=32).to(device)
    optimizer = optim.Adam(cnnmlp_model.parameters(), lr=1e-3)
    train_model(cnnmlp_model, train_loader, criterion, optimizer, num_epochs=10)
    cnnmlp_model.eval()
    cnnmlp_loss, _ = evaluate_model(cnnmlp_model, test_loader, criterion)
    print("CNN+MLP Baseline Test MSE:", cnnmlp_loss)
    
    # 4. Naive Baseline: predict the mean training return.
    mean_return = np.mean(y_train)  # Using original training target values
    print("\nNaive Baseline (predict mean return):")
    # Evaluate on test set by computing MSE against the constant prediction.
    all_targets = []
    for _, target in test_loader:
        all_targets.append(target)
    all_targets = torch.cat(all_targets, dim=0).to(device)
    naive_preds = mean_return * torch.ones_like(all_targets)
    naive_mse = criterion(naive_preds, all_targets).item()
    print("Naive Baseline Test MSE:", naive_mse)

if __name__ == "__main__":
    main()