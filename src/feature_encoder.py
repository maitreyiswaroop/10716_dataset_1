# feature_encoder.py

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Import data loading functions and DATA_DIR from your project
from data_load import load_data, get_data
from config import DATA_DIR

#############################################
# Data Preprocessor
#############################################

class Preprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessor for financial time series features.
    
    This transformer imputes missing values and scales a specified set of columns
    (e.g., the 200 alpha signals) while leaving a set of raw columns (e.g., 11 raw variables)
    untouched.
    
    Note: The DataFrame passed to this preprocessor should NOT include identifier columns 
    (e.g., stock_index and day_index); those are handled separately.
    
    Parameters:
        imputation_strategy : str, default='mean'
            Strategy for imputing missing values.
        scaling_method : str, default='standard'
            'standard' for StandardScaler or 'minmax' for MinMaxScaler.
        columns_to_scale : list of str, optional
            Explicit list of columns to scale. 
            If None and alpha_prefix is provided, all columns starting with the given alpha_prefix are scaled.
        columns_to_leave : list of str, optional
            Columns that should remain untouched.
        alpha_prefix : str, optional
            Prefix used to auto-select alpha columns (e.g., "alpha_").
        raw_columns : list of str, optional
            Names of raw columns to leave untouched.
    """
    def __init__(self, imputation_strategy='mean', scaling_method='standard',
                 columns_to_scale=None, columns_to_leave=None,
                 alpha_prefix=None, raw_columns=None, **kwargs):
        self.imputation_strategy = imputation_strategy
        self.scaling_method = scaling_method
        self.columns_to_scale = columns_to_scale
        self.columns_to_leave = columns_to_leave
        self.alpha_prefix = alpha_prefix
        self.raw_columns = raw_columns
        
        self.imputer = SimpleImputer(strategy=self.imputation_strategy)
        if self.scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif self.scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError("Unsupported scaling method. Choose 'standard' or 'minmax'.")
        
        # Initialize column indices to None
        self.alpha_indices = None
        self.raw_indices = None
    
    def fit(self, X, y=None):
        # Convert to DataFrame if it's a numpy array
        if isinstance(X, np.ndarray):
            if self.columns_to_scale is None and self.alpha_prefix is not None:
                # If X is a numpy array and we need to generate column names
                num_features = X.shape[1]
                if self.raw_columns is not None:
                    num_alphas = num_features - len(self.raw_columns)
                else:
                    num_alphas = num_features  # Assume all are alphas if no raw_columns

                # Generate column names
                column_names = [f"alpha_{i+1}" for i in range(num_alphas)]
                if self.raw_columns is not None:
                    column_names.extend(self.raw_columns)
                    
                X = pd.DataFrame(X, columns=column_names)
            else:
                # If there are explicit columns_to_scale or no alpha_prefix, just use numeric indices
                self.alpha_indices = list(range(X.shape[1] - (len(self.raw_columns) if self.raw_columns is not None else 0)))
                self.raw_indices = list(range(X.shape[1] - len(self.raw_columns), X.shape[1])) if self.raw_columns is not None else []
                
                # Fit imputer and scaler on alpha columns
                X_alpha = X[:, self.alpha_indices]
                self.imputer.fit(X_alpha)
                X_imputed = self.imputer.transform(X_alpha)
                self.scaler.fit(X_imputed)
                return self
        
        # If X is a DataFrame (original case)
        if isinstance(X, pd.DataFrame):
            # Select columns to scale automatically if not explicitly provided
            if self.columns_to_scale is None and self.alpha_prefix is not None:
                self.columns_to_scale = [col for col in X.columns if col.startswith(self.alpha_prefix)]
            elif self.columns_to_scale is None:
                self.columns_to_scale = X.select_dtypes(include=[np.number]).columns.tolist()
                if self.raw_columns is not None:
                    self.columns_to_scale = [col for col in self.columns_to_scale if col not in self.raw_columns]
            
            # Determine columns to leave untouched
            if self.columns_to_leave is None and self.raw_columns is not None:
                self.columns_to_leave = self.raw_columns
            
            # Store column names for transform method
            self.column_names = X.columns.tolist()
            
            X_scale = X[self.columns_to_scale]
            self.imputer.fit(X_scale)
            X_imputed = self.imputer.transform(X_scale)
            self.scaler.fit(X_imputed)
        
        return self
    
    def transform(self, X):
        # Handle numpy arrays
        if isinstance(X, np.ndarray):
            if self.alpha_indices is not None and self.raw_indices is not None:
                # Use the numeric indices if we calculated them in fit
                X_alpha = X[:, self.alpha_indices]
                X_imputed = self.imputer.transform(X_alpha)
                X_scaled = self.scaler.transform(X_imputed)
                
                if len(self.raw_indices) > 0:
                    X_raw = X[:, self.raw_indices]
                    X_final = np.hstack([X_scaled, X_raw])
                else:
                    X_final = X_scaled
                    
                return X_final
            
            # If indices not set, try to create a DataFrame with generated column names
            if self.columns_to_scale is not None:
                num_features = X.shape[1]
                if self.raw_columns is not None:
                    num_alphas = num_features - len(self.raw_columns)
                else:
                    num_alphas = num_features
                
                # Generate column names
                column_names = [f"alpha_{i+1}" for i in range(num_alphas)]
                if self.raw_columns is not None:
                    column_names.extend(self.raw_columns)
                
                # Convert to DataFrame for easier column selection
                X = pd.DataFrame(X, columns=column_names)
                # Fall through to DataFrame processing below
            else:
                # If we can't handle it as a numpy array, just return the original data
                # This is a fallback and should be avoided
                return X
        
        # Handle DataFrames
        if isinstance(X, pd.DataFrame):
            X_scale = X[self.columns_to_scale]
            X_imputed = self.imputer.transform(X_scale)
            X_scaled = self.scaler.transform(X_imputed)
            
            if self.columns_to_leave is not None:
                X_leave = X[self.columns_to_leave].values
                X_final = np.hstack([X_scaled, X_leave])
            else:
                X_final = X_scaled
                
            return X_final
        
        # If none of the above conditions are met
        raise ValueError("Input must be either a numpy array or a pandas DataFrame")
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)
    
#############################################
# CNN-based Feature Encoder
#############################################

class CNNFeatureEncoder(nn.Module):
    """
    A CNN-based feature encoder for time series data.
    
    Given a window of past observations (shape: [batch, L, input_dim]),
    this module applies a stack of 1D convolution layers (with non-linear activation)
    followed by global average pooling and a final linear projection to produce
    a stock-relevant latent representation.
    
    Parameters:
        input_dim : int
            Number of features per time step (e.g., 211 if you have 200 alphas + 11 raw variables).
        hidden_channels : int, default=64
            Number of channels for the convolution filters.
        output_dim : int, default=64
            Dimension of the final latent representation.
        kernel_size : int, default=3
            Size of the convolution kernel.
        num_layers : int, default=1
            Number of convolutional layers to stack.
        activation : nn.Module, default=nn.ReLU
            Activation function.
    """
    def __init__(self,
                 input_dim: int,
                 hidden_channels: int = 64,
                 output_dim: int = 64,
                 kernel_size: int = 3,
                 num_layers: int = 1,
                 activation=nn.ReLU):
        super().__init__()
        self.activation = activation()
        
        layers = []
        in_channels = input_dim
        for _ in range(num_layers):
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2  # maintain sequence length
            )
            layers.append(conv)
            layers.append(self.activation)
            in_channels = hidden_channels
        self.conv_stack = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_channels, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, L, input_dim)
        x = x.permute(0, 2, 1)  # => (batch, input_dim, L)
        x = self.conv_stack(x)  # => (batch, hidden_channels, L)
        x = self.global_pool(x) # => (batch, hidden_channels, 1)
        x = x.squeeze(-1)       # => (batch, hidden_channels)
        x = self.fc(x)          # => (batch, output_dim)
        return x

def get_feature_encoder(input_dim, hidden_channels=64, output_dim=64,
                        kernel_size=3, num_layers=1, activation=nn.ReLU):
    return CNNFeatureEncoder(
        input_dim=input_dim,
        hidden_channels=hidden_channels,
        output_dim=output_dim,
        kernel_size=kernel_size,
        num_layers=num_layers,
        activation=activation
    )

#############################################
# Data Grouping: Create 3D Array (Stocks x Days x Features)
#############################################

def group_by_stock(df):
    """
    Group the DataFrame by stock_index and pivot by day_index so that the
    final array has shape: (num_stocks, num_days, num_features).
    
    Missing entries (if a stock is missing data for a day) will be NaN.
    """
    # Pivot the table: index is stock_index, columns are day_index, values are features.
    # We use a pivot_table which can handle multiple feature columns.
    df_pivot = df.pivot_table(index="stock_index", columns="day_index", aggfunc="first")
    # The resulting DataFrame has a MultiIndex for columns: (feature_name, day_index)
    # Sort by day_index:
    df_pivot = df_pivot.sort_index(axis=1)
    # Rearrange to a 3D array:
    # The first level of the column MultiIndex are the feature names.
    feature_names = df_pivot.columns.levels[0]
    # Create a list of arrays, one for each feature.
    stock_array_list = []
    for feature in feature_names:
        # Extract the sub-dataframe for this feature; its shape is (num_stocks, num_days)
        feature_df = df_pivot[feature]
        # Convert to numpy array (shape: num_stocks x num_days)
        stock_array_list.append(feature_df.values)
    # Stack along the last axis to get shape: (num_stocks, num_days, num_features)
    data_3d = np.stack(stock_array_list, axis=-1)
    return data_3d

#############################################
# Main Function: Processing Given Dataset 1
#############################################

def main():
    # Load the training dataset (part 1) using data_load.py
    train_file = f'{DATA_DIR}/dict_of_data_Jan2025_part1.npy'
    data_dict = load_data(train_file)
    x_data, y_data, si, di, raw_data, list_of_data = get_data(data_dict)
    
    # Construct DataFrames for the 200 alpha signals and 11 raw variables
    num_alphas = x_data.shape[1]  # Expected to be 200
    alpha_cols = [f'alpha_{i+1}' for i in range(num_alphas)]
    df_alphas = pd.DataFrame(x_data, columns=alpha_cols)
    df_raw = pd.DataFrame(raw_data, columns=list_of_data)
    
    # Concatenate features horizontally
    df_features = pd.concat([df_alphas, df_raw], axis=1)
    
    # Add identifier columns for grouping
    df_features["stock_index"] = si
    df_features["day_index"] = di
    print("Combined DataFrame shape (raw):", df_features.shape)
    
    # --- Preprocessing ---
    # We want to preprocess only the feature columns (alphas + raw variables),
    # so we drop the identifier columns temporarily.
    df_ids = df_features[["stock_index", "day_index"]]
    df_data = df_features.drop(columns=["stock_index", "day_index"])
    
    preprocessor = Preprocessor(
        imputation_strategy='mean',
        scaling_method='standard',
        alpha_prefix='alpha_',
        raw_columns=list_of_data  # Leave the 11 raw variables unscaled
    )
    preprocessed_data = preprocessor.fit_transform(df_data)
    print("Preprocessed Data shape:", preprocessed_data.shape)  # Expect (num_samples, 211)
    
    # Put back the identifiers into a DataFrame
    df_preprocessed = pd.DataFrame(preprocessed_data, columns=df_data.columns)
    df_preprocessed["stock_index"] = df_ids["stock_index"].values
    df_preprocessed["day_index"] = df_ids["day_index"].values
    
    # --- Group Data by Stock ---
    # Create a 3D array with shape: (num_stocks, num_days, num_features)
    data_3d = group_by_stock(df_preprocessed)
    print("Grouped 3D data shape (stocks x days x features):", data_3d.shape)
    # For Dataset 1, expected shape is roughly (2329, ~751, 211)
    
    # --- Feature Encoder ---
    # We now assume the temporal dimension is the number of days.
    # You might want to restrict to a fixed window length in practice,
    # but here we encode the entire time series per stock.
    # For demonstration, we apply the CNN feature encoder on each stock’s time series.
    # We treat each stock as a separate "batch" element.
    num_stocks, num_days, num_features = data_3d.shape
    # Convert the 3D array to a torch tensor.
    # If desired, you can later slice or pad the time dimension.
    x_tensor = torch.tensor(data_3d, dtype=torch.float32)  # shape: (num_stocks, num_days, num_features)
    
    # Instantiate the feature encoder.
    # Here, input_dim = num_features. The encoder will process the entire time series per stock.
    encoder = get_feature_encoder(
        input_dim=num_features,
        hidden_channels=64,
        output_dim=64,
        kernel_size=3,
        num_layers=2,
        activation=nn.ReLU
    )
    
    # Forward pass through the feature encoder.
    # Note: Our CNN encoder expects input shape (batch, L, input_dim).
    latent_representations = encoder(x_tensor)
    print("Latent representations shape:", latent_representations.shape)
    # Expected shape: (num_stocks, 64)

if __name__ == "__main__":
    main()