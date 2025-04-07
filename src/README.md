# Robust, Interpretable and Uncertainty-aware Stock Forecasting

A comprehensive deep learning framework for stock return prediction with innovative components for time embedding, anomaly filtering, and attention mechanisms.

## Overview

This repository implements a novel stock prediction framework that addresses key challenges in financial time series forecasting:

1. **Uncertainty Modeling**: Using Fourier-based filtering to handle anomalies and outliers
2. **Interpretability**: Implementing hierarchical and co-attention mechanisms for feature importance
3. **Time Embedding**: Employing advanced time encodings based on Bochner's theorem

The architecture is designed for inductive stock prediction, meaning it can generalize to stocks not seen during training and forecast multiple time steps into the future.

## Repository Structure

```
├── config.py                  # Configuration settings
├── data_load.py               # Data loading utilities
├── feature_encoder.py         # Feature encoding modules
├── temporal_encoder.py        # Time embedding modules
├── anomaly_filter.py          # Fourier-based outlier detection
├── attention_mechanism.py     # Hierarchical and co-attention modules
├── stock_model.py             # Main model implementations
├── train.py                   # Training and evaluation pipeline
├── dataset_fix.py             # Fixes for handling variable-sized batches
├── run.py                     # Convenience script for various modes
├── README.md                  # This file
└── DESCRIPTION.md             # Detailed project description
```

## Model Variants

The framework includes several model variants to address different tasks and scenarios:

### 1. StockPredictionModel

The primary model with all components:
- Feature encoding with CNNs
- Time embedding with kernelized functions
- Anomaly filtering with Fourier analysis
- Transformer-based sequence modeling
- Hierarchical temporal attention
- Co-attention mechanisms

### 2. InductiveStockPredictor

Extended version for multi-step prediction:
- Builds on the base StockPredictionModel
- Adds multi-step prediction capabilities
- Incorporates uncertainty in forecasts
- Designed for inductive learning (generalizing to unseen stocks)

### 3. SimplifiedStockModel

Streamlined version for debugging and fast experimentation:
- Simpler architecture with fewer components
- Linear layers instead of complex modules
- Basic transformers without custom attention
- Faster training for initial testing

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/stock-prediction.git
cd stock-prediction

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Usage

### Quick Start

The easiest way to run the model is using the `run.py` script:

```bash
# Debug mode (small dataset, simplified model)
python run.py --mode debug

# Train on part 1, test on part 2
python run.py --mode train_test --model_type standard

# Multi-step prediction with inductive model
python run.py --mode train_test --model_type inductive --forecast_horizon 5

# Full training on merged dataset
python run.py --mode full_train
```

### Available Run Modes

1. **Debug Mode**
   ```bash
   python run.py --mode debug
   ```
   - Uses a tiny subset of data (10 days, 10 stocks)
   - Simplified model with fewer layers
   - Prints tensor shapes for debugging
   - Quick iterations for testing code changes

2. **Train-Test Mode**
   ```bash
   python run.py --mode train_test
   ```
   - Trains on part 1 dataset and evaluates on part 2
   - Full model with all components
   - Standard hyperparameters for real-world performance

3. **Full Training Mode**
   ```bash
   python run.py --mode full_train
   ```
   - First merges both parts of the dataset
   - Trains on the combined data
   - Uses larger model capacity
   - Best for final model training

4. **Cross-Validation Mode**
   ```bash
   python run.py --mode cross_validate
   ```
   - Runs 3-fold cross-validation
   - Useful for hyperparameter tuning
   - Reports average performance across folds

5. **Data Analysis Mode**
   ```bash
   python run.py --mode data_analysis
   ```
   - Only performs data analysis without training
   - Generates statistics and visualizations
   - Helpful for understanding the dataset

### Advanced Options

For more control, use `train.py` directly with custom arguments:

```bash
# Customize model architecture
python train.py --train_part1_test_part2 --hidden_dim 128 --time_dim 128 \
  --num_transformer_layers 4 --num_attention_heads 8 --window_size 15

# Skip complex components for faster debugging
python train.py --debug --skip_time_encoding --skip_anomaly_filter

# Customize training parameters
python train.py --batch_size 32 --num_epochs 100 --learning_rate 0.0005 \
  --weight_decay 1e-6 --early_stopping_patience 15
```

### Key Configuration Arguments

#### Data Options
- `--window_size`: Size of the sliding window (default: 20)
- `--forecast_horizon`: Number of days to forecast ahead (default: 1)
- `--stride`: Stride for the sliding window (default: 1)
- `--create_merged`: Create merged dataset before training

#### Model Architecture
- `--hidden_dim`: Dimension of hidden representations (default: 64)
- `--time_dim`: Dimension of time embeddings (default: 64)
- `--num_transformer_layers`: Number of transformer layers (default: 3)
- `--num_attention_heads`: Number of attention heads (default: 8)
- `--temporal_bin_size`: Size of temporal bins for hierarchical attention (default: 5)

#### Training Parameters
- `--batch_size`: Batch size for training (default: 64)
- `--num_epochs`: Number of training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 0.001)
- `--weight_decay`: Weight decay for regularization (default: 1e-5)
- `--early_stopping_patience`: Patience for early stopping (default: 10)

#### Debugging Options
- `--debug_shapes`: Print tensor shapes during execution
- `--skip_time_encoding`: Skip complex time encoding
- `--skip_anomaly_filter`: Skip anomaly filtering
