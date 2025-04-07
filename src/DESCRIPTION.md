# Robust, Interpretable and Uncertainty-aware Stock Forecasting: Technical Details

## 1. Introduction

Stock forecasting presents significant challenges due to the noisy, temporally-dependent nature of financial data. This project addresses three key challenges:

1. **Uncertainty Modeling**: How to handle anomalies and disruptions from outliers in financial time series
2. **Interpretability**: How to make predictions explainable and understand which features drive forecasts
3. **Time Embedding**: How to capture seasonality, long-range trends, and other temporal patterns

We propose a novel architecture for inductive stock prediction, capable of forecasting multiple future time steps while addressing these challenges.

## 2. Dataset Description

The framework is designed for a specific stock dataset with the following properties:

- **Scope**: 3 years of stock data (2021-01-04 to 2023-12-29) for ~1,200 stocks
- **Alpha Signals**: 400 alpha signals (200 per file) as input features
- **Target Variable**: Next-day stock returns
- **Size**: 1.12 million data points split across two files
- **Additional Features**: 11 raw variables (price data, volume, market indicators, sector info, etc.)

### Key Dataset Components

```python
data_dict keys: 'x_data', 'y_data', 'si', 'di', 'raw_data', 'list_of_data'
```

- `x_data`: 200 alpha signals per file. Shape: (1,123,742, 200)
- `y_data`: Next day returns. Shape: (1,123,742,)
- `si`: Stock indices. Shape: (1,123,742,)
- `di`: Day indices. Shape: (1,123,742,)
- `raw_data`: 11 raw variables. Shape: (1,123,742, 11)
- `list_of_data`: Names of raw variables ['close', 'open', 'low', 'high', 'volume', 'trading_days_til_next_ann', 'trading_days_since_last_ann', 'close_VIX', 'ret1_SPX', 'sector', 'industry']

## 3. Complete Pipeline Architecture

Our framework implements a comprehensive pipeline for stock prediction:
3
### 3.1 Data Processing

#### Data Loading and Organization
- Load dataset from `.npy` files
- Organize data by stock and day indices
- Handle missing values and inconsistent time series

#### Feature Preprocessing
- Implemented in `feature_encoder.py`
- Normalizes alpha signals and raw features
- Handles categorical variables (sector, industry)
- Creates sliding windows for time series forecasting

#### Dataset Creation
- Creates sliding windows of fixed size
- Ensures consecutive days within windows
- Manages target variables for prediction
- Handles variable-sized batches with custom collation

### 3.2 Core Components

#### 1. Feature Encoding Module

The feature encoder transforms raw input features into meaningful latent representations:

```python
class CNNFeatureEncoder(nn.Module):
    """CNN-based feature encoder for time series data."""
    def __init__(self, input_dim, hidden_channels=64, output_dim=64, 
                 kernel_size=3, num_layers=1, activation=nn.ReLU):
        # Implementation details...
```

- **Input**: Alpha signals and raw variables (211 features)
- **Architecture**: Stacked 1D convolution layers with non-linear activation
- **Global Pooling**: Aggregates temporal patterns across the window
- **Output**: Dense representation of stock features

#### 2. Time Embedding Module

One of our key innovations is a sophisticated time embedding approach based on Bochner's theorem:

```python
class KernelizedTimeEncoder(nn.Module):
    """Time encoder based on Bochner's theorem."""
    def __init__(self, embed_dim, num_frequencies=None, learnable=True,
                 time_scale=1.0, base=10000.0, trainable_time_shift=False):
        # Implementation details...
```

We implement four types of time encodings:

1. **Kernelized Time Encoder**: Uses spectral approaches inspired by Bochner's theorem
2. **Multi-Scale Time Encoder**: Captures patterns at different time scales (daily, weekly, monthly)
3. **Calendar Time Encoder**: Extracts date-specific features (day of week, month, etc.)
4. **Hybrid Time Encoder**: Combines all these approaches for comprehensive time representation

These encodings give the model an explicit way to represent time-of-year, time-of-week, seasonal cycles, and other temporal patterns, which is critical for financial time series.

#### 3. Anomaly Filtering Module

To handle outliers and anomalies in stock data, we implement Fourier-based filters:

```python
class FourierFilter(nn.Module):
    """Fourier-based filtering for time series data."""
    def __init__(self, input_dim, filter_type='learnable', cutoff_low=0.1,
                 cutoff_high=0.4, filter_init='gaussian', smoothing=0.1,
                 return_frequency=False):
        # Implementation details...
```

The anomaly filtering includes:

1. **Fourier Filter**: Applies frequency domain filtering to separate signal from noise
2. **Multi-view Fourier Filter**: Creates multiple filtered "views" of the data
3. **Anomaly Score Calculator**: Quantifies the degree of anomaly for each data point
4. **Robust Stock Anomaly Filter**: Combines detection with uncertainty estimation

This component contributes to theoretical robustness by explicitly handling heavy-tailed noise, improving the model's convergence and error bounds.

#### 4. Transformer Encoder Module

For sequence processing, we use a transformer architecture:

```python
class StockTransformer(nn.Module):
    """Transformer encoder for stock prediction."""
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048,
                 dropout=0.1, use_positional_encoding=True):
        # Implementation details...
```

The transformer encoder:
- Processes the sequence of stock data points
- Captures both short and long-range dependencies
- Uses multi-head attention to focus on different relationship patterns
- Integrates with our custom time embeddings instead of standard positional encodings

#### 5. Hierarchical Temporal Attention Module

We introduce a novel attention mechanism that segments the timeline into meaningful bins:

```python
class HierarchicalTemporalAttention(nn.Module):
    """Hierarchical attention operating within time bins."""
    def __init__(self, embed_dim, num_heads=8, bin_size=5, dropout=0.1,
                 use_cross_bin_attention=True):
        # Implementation details...
```

The hierarchical attention:
1. Segments data into temporal bins (e.g., weeks of 5 trading days)
2. Applies within-bin attention to capture local patterns
3. Applies cross-bin attention to capture relationships between time periods
4. Creates a multi-level understanding of temporal relationships

This structured approach adds an effective prior: that temporal relationships are partly hierarchical (local interactions first, then global).

#### 6. Co-Attention Module

To model interactions between features and time, we implement a co-attention mechanism:

```python
class CoAttentionModule(nn.Module):
    """Co-attention between stock features and time."""
    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        # Implementation details...
```

This module:
- Implements bidirectional attention between features and time
- Shows which features are important at which time points
- Creates enhanced representations of both features and time
- Provides interpretability by visualizing attention weights

### 3.3 Stock Prediction Models

The framework includes three primary model variants:

#### 1. StockPredictionModel

The complete stock prediction model implementing all proposed components:

```python
class StockPredictionModel(nn.Module):
    """Complete stock prediction model with all components."""
    def __init__(self, input_dim, time_dim, hidden_dim, output_dim=1,
                 num_transformer_layers=3, num_attention_heads=8,
                 temporal_bin_size=5, dropout=0.1,
                 skip_time_encoding=False, skip_anomaly_filter=False):
        # Implementation details...
```

This model:
- Integrates all six core components described above
- Processes input features through the complete pipeline
- Can be configured to skip complex components for debugging
- Outputs stock return predictions with optional attention weights
- Enables interpretability through attention visualization

#### 2. InductiveStockPredictor

Extended model for multi-step prediction with inductive capabilities:

```python
class InductiveStockPredictor(nn.Module):
    """Inductive model for multi-step stock prediction."""
    def __init__(self, input_dim, time_dim, hidden_dim, forecast_steps=1,
                 use_uncertainty=True, skip_time_encoding=False,
                 skip_anomaly_filter=False, **kwargs):
        # Implementation details...
```

This model:
- Uses the base StockPredictionModel as a feature extractor
- Adds multi-step prediction heads for forecasting several days ahead
- Incorporates uncertainty estimates in the predictions
- Is designed for inductive learning (generalizing to unseen stocks)
- Supports varying forecast horizons

#### 3. SimplifiedStockModel

Streamlined model for debugging and quick experimentation:

```python
class SimplifiedStockModel(nn.Module):
    """Simplified model for debugging purposes."""
    def __init__(self, input_dim, hidden_dim, output_dim=1):
        # Implementation details...
```

This model:
- Uses simple linear layers instead of complex components
- Includes basic transformers without custom attention
- Provides fast training for initial testing
- Acts as a fallback when debugging complex issues

### 3.4 Training and Evaluation

The training pipeline is implemented in `train.py` and includes:

#### Training Loop
- Mini-batch gradient descent with Adam optimizer
- Early stopping based on validation loss
- Learning rate scheduling
- Model checkpointing
- Comprehensive logging

#### Custom Dataset Handling
- `StockDataset` class for sliding window creation
- Ensures consecutive days within windows
- Custom collation for handling variable-sized batches
- Data preprocessing and normalization

#### Evaluation Metrics
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Per-step metrics for multi-step prediction
- Visualization of predictions vs. actuals

## 5. Running Modes

The framework supports several operational modes:

### 5.1 Debug Mode

For development and testing:
- Uses a small subset of data (10 days, 10 stocks)
- Simplified model with fewer layers
- Prints tensor shapes for debugging
- Quick iterations for testing code changes

### 5.2 Train-Test Mode

For real-world performance:
- Trains on part 1 dataset and evaluates on part 2
- Full model with all components
- Standard hyperparameters
- Comprehensive evaluation

### 5.3 Full Training Mode

For maximum model capacity:
- Merges both parts of the dataset
- Trains on the combined data
- Uses larger model capacity
- Suitable for final model training

### 5.4 Cross-Validation Mode

For hyperparameter tuning:
- Runs 3-fold cross-validation
- Reports average performance across folds
- Identifies optimal hyperparameters
- Validates model stability

### 5.5 Data Analysis Mode

For dataset understanding:
- Performs data analysis without training
- Generates statistics and visualizations
- Helps understand the dataset characteristics
- Informs modeling decisions

## 6. Technical Challenges and Solutions

### 6.1 Variable Sequence Lengths

**Challenge**: Different stocks have different numbers of consecutive days.

**Solution**: Implemented custom batch collation that truncates sequences to the minimum length in each batch, ensuring compatibility while preserving as much data as possible.

## 7. Theoretical Foundation

### 7.1 Bochner's Theorem and Time Embedding

Our time embedding approach is inspired by Bochner's theorem, which states that any continuous positive-definite kernel can be represented as the Fourier transform of a non-negative measure. By leveraging this theorem, we have a theoretical guarantee that with enough frequency components, our embeddings can approximate arbitrary stationary kernel features, which implies the model can represent a wide class of temporal relationships (periodic or smooth trends).

### 7.2 Fourier Filtering for Robustness

The Fourier filtering approach contributes to theoretical robustness by explicitly handling heavy-tailed noise. By decomposing signals into frequency components and filtering out noise, the model's convergence and error bounds are improved under assumptions of bounded outlier frequency, leading to better generalization.

### 7.3 Hierarchical Attention for Temporal Structure

The hierarchical temporal attention mechanism adds a structured prior: that temporal relationships are partly hierarchical (local interactions first, then global). The theoretical intuition is akin to multi-scale analysis or segmented modeling – it improves learning by first explaining short-term variations (within-bin) before trying to connect far-apart points.

