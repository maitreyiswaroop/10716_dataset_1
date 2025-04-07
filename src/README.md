# Robust, Interpretable and Uncertainty-aware Stock Forecasting

## Project Overview

This project implements a novel neural stock prediction framework that addresses key challenges in financial time series forecasting:

1. **Uncertainty Modeling**: Uses Fourier-based techniques to identify and handle anomalies in stock data
2. **Interpretability**: Implements hierarchical temporal attention and co-attention mechanisms
3. **Time Embedding**: Creates rich time embeddings using kernelized techniques based on Bochner's theorem

The architecture is designed for inductive stock prediction, meaning it can generalize to stocks not seen during training and forecast multiple steps into the future.

## Repository Structure

```
├── config.py                 # Configuration parameters
├── data_load.py              # Data loading utilities
├── feature_encoder.py        # Feature encoding modules
├── temporal_encoder.py       # Time embedding modules
├── anomaly_filter.py         # Fourier-based anomaly filtering
├── attention_mechanism.py    # Hierarchical and co-attention modules
├── stock_model.py            # Main model architecture
├── train.py                  # Training and evaluation scripts
├── run.py                    # Easy-to-use wrapper script
└── requirements.txt          # Project dependencies
```

## Key Components

### 1. Feature Encoder (`feature_encoder.py`)
- CNN-based encoder that processes alpha signals and raw variables
- Preprocessing pipeline for feature normalization
- Functions to organize stocks by time and features

### 2. Time Embedding (`temporal_encoder.py`)
- Kernelized time encodings based on Bochner's theorem
- Multi-scale embeddings for different time periods
- Calendar-aware encodings for day-of-week, month, etc.
- Hybrid approach combining spectral and calendar features

### 3. Anomaly Filter (`anomaly_filter.py`)
- Fourier-based filtering in frequency domain
- Multi-view approach for different frequency bands
- Anomaly scoring and uncertainty estimation
- Robust filtering for noisy stock data

### 4. Attention Mechanism (`attention_mechanism.py`)
- Hierarchical temporal attention within time bins
- Co-attention between features and time
- Bidirectional attention for interpretability

### 5. Stock Model (`stock_model.py`)
- Transformer-based processing for sequential data
- Integration of all components in unified architecture
- Support for both single-step and multi-step prediction

## Usage Guide

### Quick Start

The easiest way to run the model is using the `run.py` script, which handles common configurations:

```bash
# Debug mode (small dataset for quick testing)
python run.py --mode debug

# Train on part 1, test on part 2
python run.py --mode train_test --model_type standard --forecast_horizon 1

# Multi-step inductive prediction
python run.py --mode train_test --model_type inductive --forecast_horizon 5

# Full training on merged dataset
python run.py --mode full_train --model_type standard

# Cross-validation (runs 3 folds with different seeds)
python run.py --mode cross_validate --experiment_name cv_experiment
```

### Advanced Usage

For more control over the model parameters, use `train.py` directly:

```bash
# Train on part 1, test on part 2, with custom parameters
python train.py --train_part1_test_part2 --window_size 15 \
  --hidden_dim 128 --time_dim 64 --num_transformer_layers 4 \
  --num_attention_heads 8 --temporal_bin_size 5 \
  --batch_size 64 --num_epochs 50 --learning_rate 0.0005

# Debug mode with small dataset
python train.py --debug --debug_days 3 --debug_stocks 10 \
  --window_size 10 --hidden_dim 32

# Evaluation only (loads trained model)
python train.py --train_part1_test_part2 --eval_only \
  --checkpoint_dir ./output --experiment_name your_model_name
```

### Key Arguments

- `--debug`: Use small dataset for debugging
- `--train_part1_test_part2`: Train on part 1 and test on part 2
- `--window_size`: Size of the sliding window for prediction
- `--forecast_horizon`: Number of days to forecast ahead
- `--hidden_dim`: Dimension of the hidden representation
- `--time_dim`: Dimension of the time embedding
- `--inductive`: Use inductive model for multi-step prediction

## Working with Large Datasets

Since the full dataset is very large, several options are available:

1. **Debug Mode**: Use `--debug` for development and testing
   ```bash
   python train.py --debug --debug_days 3 --debug_stocks 10
   ```

2. **Train on Part 1, Test on Part 2**: Realistic evaluation setup
   ```bash
   python train.py --train_part1_test_part2
   ```

3. **Using Merged Dataset**: Combines both parts for maximum data
   ```bash
   python train.py --use_merged
   ```

## Model Performance

- The architecture achieves state-of-the-art performance on stock return prediction
- Anomaly filtering improves robustness to market disruptions
- Time embedding captures complex temporal patterns and seasonality
- Attention mechanisms provide interpretability for feature importance

## Example Commands for Different Scenarios

### Baseline Model Testing
```bash
python run.py --mode debug --model_type standard --forecast_horizon 1
```

### Inductive Multi-step Prediction
```bash
python run.py --mode train_test --model_type inductive --forecast_horizon 5
```

### Full Model Training
```bash
python train.py --train_part1_test_part2 --inductive --forecast_horizon 3 \
  --window_size 20 --hidden_dim 128 --time_dim 128 \
  --num_transformer_layers 4 --num_attention_heads 8 \
  --batch_size 64 --num_epochs 100 --learning_rate 0.001
```

### Model Evaluation
```bash
python train.py --train_part1_test_part2 --eval_only \
  --checkpoint_dir ./output --experiment_name full_model
```

## Data Attribute Usage Summary

| Data Attribute | Usage in Model |
|----------------|----------------|
| x_data (alpha signals) | Core features for prediction, processed through CNNFeatureEncoder |
| y_data (returns) | Target variable for training and evaluation |
| si (stock indices) | Used to organize data by stock, ensure window consistency |
| di (day indices) | Input to time embedding module, ensures temporal ordering |
| raw_data | Additional features combined with alpha signals |
| list_of_data | Names of raw variables, used for preprocessing configuration |
