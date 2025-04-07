# config.py

import os
from datetime import datetime

# Data directory 
DATA_DIR = os.path.join(os.getcwd(), 'data')

# Create data directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Output directory for model checkpoints and results
OUTPUT_DIR = os.path.join(os.getcwd(), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Logs directory
LOGS_DIR = os.path.join(os.getcwd(), 'logs')
os.makedirs(LOGS_DIR, exist_ok=True)

# Default experiment name (timestamp-based)
DEFAULT_EXPERIMENT_NAME = f"stock_pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

# Model configuration
MODEL_CONFIG = {
    # Feature encoder
    'cnn_kernel_size': 3,
    'cnn_num_layers': 2,
    
    # Time encoder
    'time_encoder_type': 'hybrid',  # 'kernel', 'multiscale', 'calendar', 'hybrid'
    'time_num_scales': 4,
    
    # Anomaly filter
    'filter_types': ['lowpass', 'bandpass', 'highpass'],
    'use_autoencoder': False,
    'output_uncertainty': True,
    
    # Transformer
    'num_transformer_layers': 3,
    'num_attention_heads': 8,
    'dim_feedforward': 256,
    
    # Hierarchical attention
    'temporal_bin_size': 5,  # Trading week
    'use_cross_bin_attention': True,
    
    # Output
    'output_dim': 1,  # Next day return
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 64,
    'num_epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'early_stopping_patience': 10,
    'window_size': 20,  # 20 trading days (about a month)
    'forecast_horizon': 1,  # Default to next day prediction
    'stride': 1,
    'train_ratio': 0.7,
    'val_ratio': 0.15,
    'test_ratio': 0.15,
    'seed': 42,
}

# Device configuration
DEVICE_CONFIG = {
    'use_cuda': True,
    'gpu_id': 0,
}