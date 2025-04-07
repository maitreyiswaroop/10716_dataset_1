# anomaly_filter.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional, Union
import math

class FourierFilter(nn.Module):
    """
    Fourier-based filtering module for time series data.
    
    This module applies Fourier Transform to the input signal, filters certain frequency
    components based on learnable/predefined parameters, and transforms back to create
    a cleaned/filtered signal. This helps in removing outliers and noise while preserving
    the overall trend.
    
    Parameters:
        input_dim : int
            Dimension of the input features.
        filter_type : str, default='learnable'
            Type of filter to apply. Options: 'learnable', 'lowpass', 'highpass', 'bandpass'.
        cutoff_low : float, default=0.1
            Low cutoff frequency (for lowpass and bandpass filters).
        cutoff_high : float, default=0.4
            High cutoff frequency (for highpass and bandpass filters).
        filter_init : str, default='gaussian'
            Initial filter shape when using 'learnable'. Options: 'gaussian', 'uniform', 'lowpass', 'highpass'.
        smoothing : float, default=0.1
            Smoothing factor for filter edges.
        return_frequency : bool, default=False
            If True, also returns the frequency domain representation.
    """
    def __init__(
        self,
        input_dim: int,
        filter_type: str = 'learnable',
        cutoff_low: float = 0.1,
        cutoff_high: float = 0.4,
        filter_init: str = 'gaussian',
        smoothing: float = 0.1,
        return_frequency: bool = False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.filter_type = filter_type
        self.cutoff_low = cutoff_low
        self.cutoff_high = cutoff_high
        self.smoothing = smoothing
        self.return_frequency = return_frequency
        
        # For learnable filters, create parameters
        if filter_type == 'learnable':
            # Create learnable filter coefficients
            # We use filter_init to determine the initial shape
            if filter_init == 'gaussian':
                # Initialize with a Gaussian-shaped filter (bell curve)
                init_filter = torch.exp(-torch.arange(input_dim/2 + 1).float() ** 2 / (2 * (input_dim/4) ** 2))
            elif filter_init == 'uniform':
                # Initialize with a uniform filter (all frequencies equally)
                init_filter = torch.ones(input_dim//2 + 1)
            elif filter_init == 'lowpass':
                # Initialize with a lowpass filter
                init_filter = torch.ones(input_dim//2 + 1)
                cutoff_idx = int(cutoff_low * (input_dim//2))
                init_filter[cutoff_idx:] = 0.0
                # Add smoothing at the edges
                edge_width = int(smoothing * cutoff_idx)
                if edge_width > 0:
                    edge = torch.linspace(1.0, 0.0, edge_width)
                    init_filter[cutoff_idx-edge_width:cutoff_idx] = edge
            elif filter_init == 'highpass':
                # Initialize with a highpass filter
                init_filter = torch.zeros(input_dim//2 + 1)
                cutoff_idx = int(cutoff_high * (input_dim//2))
                init_filter[cutoff_idx:] = 1.0
                # Add smoothing at the edges
                edge_width = int(smoothing * (input_dim//2 - cutoff_idx))
                if edge_width > 0:
                    edge = torch.linspace(0.0, 1.0, edge_width)
                    init_filter[cutoff_idx:cutoff_idx+edge_width] = edge
            else:
                raise ValueError(f"Unknown filter initialization type: {filter_init}")
                
            # Create the parameter and initialize it
            self.filter_coef = nn.Parameter(init_filter, requires_grad=True)
        else:
            # For fixed filters, we'll create them during forward pass
            self.register_buffer('filter_coef', None)
    
    def _get_filter(self, seq_len: int) -> torch.Tensor:
        """
        Get the filter coefficients based on the filter type.
        
        Parameters:
            seq_len : int
                Length of the input sequence.
                
        Returns:
            torch.Tensor
                Filter coefficients in frequency domain.
        """
        # For learnable filters, use the parameters
        if self.filter_type == 'learnable':
            return self.filter_coef
        
        # Number of frequency components
        num_freqs = seq_len // 2 + 1
        
        # Create fixed filters based on type
        if self.filter_type == 'lowpass':
            # Low-pass filter: keep low frequencies, attenuate high frequencies
            filter_coef = torch.ones(num_freqs, device=self.device)
            cutoff_idx = int(self.cutoff_low * num_freqs)
            
            # Apply smoothing near the cutoff
            smooth_width = int(self.smoothing * cutoff_idx)
            if smooth_width > 0:
                filter_coef[cutoff_idx:] = 0.0
                # Add smooth transition at the cutoff
                smooth_trans = torch.linspace(1.0, 0.0, smooth_width, device=self.device)
                filter_coef[cutoff_idx-smooth_width:cutoff_idx] = smooth_trans
            else:
                filter_coef[cutoff_idx:] = 0.0
                
        elif self.filter_type == 'highpass':
            # High-pass filter: attenuate low frequencies, keep high frequencies
            filter_coef = torch.zeros(num_freqs, device=self.device)
            cutoff_idx = int(self.cutoff_high * num_freqs)
            
            # Apply smoothing near the cutoff
            smooth_width = int(self.smoothing * (num_freqs - cutoff_idx))
            if smooth_width > 0:
                filter_coef[cutoff_idx:] = 1.0
                # Add smooth transition at the cutoff
                smooth_trans = torch.linspace(0.0, 1.0, smooth_width, device=self.device)
                filter_coef[cutoff_idx:cutoff_idx+smooth_width] = smooth_trans
            else:
                filter_coef[cutoff_idx:] = 1.0
                
        elif self.filter_type == 'bandpass':
            # Band-pass filter: attenuate very low and very high frequencies
            filter_coef = torch.zeros(num_freqs, device=self.device)
            low_idx = int(self.cutoff_low * num_freqs)
            high_idx = int(self.cutoff_high * num_freqs)
            
            # Core passband has value 1.0
            filter_coef[low_idx:high_idx] = 1.0
            
            # Apply smoothing at edges
            smooth_low_width = int(self.smoothing * low_idx)
            smooth_high_width = int(self.smoothing * (num_freqs - high_idx))
            
            if smooth_low_width > 0:
                smooth_trans = torch.linspace(0.0, 1.0, smooth_low_width, device=self.device)
                filter_coef[low_idx-smooth_low_width:low_idx] = smooth_trans
                
            if smooth_high_width > 0:
                smooth_trans = torch.linspace(1.0, 0.0, smooth_high_width, device=self.device)
                filter_coef[high_idx:high_idx+smooth_high_width] = smooth_trans
                
        else:
            raise ValueError(f"Unknown filter type: {self.filter_type}")
            
        return filter_coef
    
    @property
    def device(self):
        """Get the device of this module."""
        return next(self.parameters()).device if list(self.parameters()) else torch.device('cpu')
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply Fourier filtering to the input signal.
        
        Parameters:
            x : torch.Tensor
                Input tensor of shape (batch_size, seq_len, input_dim)
                
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
                Filtered signal, and optionally the frequency domain representation.
        """
        # Save original shape
        batch_size, seq_len, input_dim = x.shape
        
        # Reshape to process each feature independently
        x_reshaped = x.view(-1, seq_len)  # (batch_size * input_dim, seq_len)
        
        # Apply FFT to convert to frequency domain
        x_fft = torch.fft.rfft(x_reshaped, dim=-1)  # (batch_size * input_dim, seq_len//2 + 1)
        
        # Get appropriate filter
        if self.filter_type == 'learnable':
            # Ensure the learnable filter matches the sequence length
            if self.filter_coef.size(0) != seq_len // 2 + 1:
                # Interpolate if sizes don't match (though ideally they should)
                filter_coef = F.interpolate(
                    self.filter_coef.unsqueeze(0).unsqueeze(0),
                    size=seq_len // 2 + 1,
                    mode='linear'
                ).squeeze(0).squeeze(0)
            else:
                filter_coef = self.filter_coef
        else:
            # Generate fixed filter based on sequence length
            filter_coef = self._get_filter(seq_len)
        
        # Apply filter in frequency domain
        x_fft_filtered = x_fft * filter_coef.unsqueeze(0)  # Broadcasting
        
        # Convert back to time domain
        x_filtered = torch.fft.irfft(x_fft_filtered, n=seq_len, dim=-1)
        
        # Reshape back to original shape
        x_filtered = x_filtered.view(batch_size, input_dim, seq_len).transpose(1, 2)
        
        if self.return_frequency:
            # Also return the frequency domain for analysis/visualization
            x_freq = x_fft.view(batch_size, input_dim, -1)
            return x_filtered, x_freq
        else:
            return x_filtered


class MultiviewFourierFilter(nn.Module):
    """
    Multi-view Fourier Filter that applies multiple frequency filters to create
    different "views" of the same time series data.
    
    This module creates multiple filtered versions of the input signal, each
    focusing on different frequency bands. This provides the model with multiple
    perspectives of the data and can help in separating trends, seasonality, and noise.
    
    Parameters:
        input_dim : int
            Dimension of the input features.
        num_filters : int, default=3
            Number of different filters to apply.
        filter_types : List[str], optional
            Types of filters to apply. If None, uses ['lowpass', 'bandpass', 'highpass'].
        cutoffs_low : List[float], optional
            Low cutoff frequencies for each filter.
        cutoffs_high : List[float], optional
            High cutoff frequencies for each filter.
        learnable : bool, default=True
            If True, filter parameters are learned during training.
        combine_strategy : str, default='concat'
            How to combine the filtered signals. Options: 'concat', 'add', 'attention'.
    """
    def __init__(
        self,
        input_dim: int,
        num_filters: int = 3,
        filter_types: Optional[List[str]] = None,
        cutoffs_low: Optional[List[float]] = None,
        cutoffs_high: Optional[List[float]] = None,
        learnable: bool = True,
        combine_strategy: str = 'concat',
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_filters = num_filters
        self.combine_strategy = combine_strategy
        
        # Default filter setup
        if filter_types is None:
            if num_filters == 3:
                # Common setup: low-frequency trends, mid-frequency patterns, high-frequency noise
                filter_types = ['lowpass', 'bandpass', 'highpass']
            elif num_filters == 4:
                # Extended setup with an additional learnable filter
                filter_types = ['lowpass', 'bandpass', 'highpass', 'learnable']
            else:
                # Create a mix of filter types
                filter_types = ['lowpass', 'bandpass', 'highpass'] + ['learnable'] * (num_filters - 3)
                # Ensure we don't exceed num_filters
                filter_types = filter_types[:num_filters]
        
        assert len(filter_types) == num_filters, "Number of filter types must match num_filters"
        
        # Default cutoff frequencies
        if cutoffs_low is None:
            # Create evenly spaced cutoffs across the frequency range
            cutoffs_low = [0.05 + 0.2 * i for i in range(num_filters)]
            # Ensure we don't exceed 1.0
            cutoffs_low = [min(c, 0.9) for c in cutoffs_low]
            
        if cutoffs_high is None:
            # Create evenly spaced cutoffs, offset from cutoffs_low
            cutoffs_high = [0.2 + 0.2 * i for i in range(num_filters)]
            # Ensure we don't exceed 1.0
            cutoffs_high = [min(c, 0.95) for c in cutoffs_high]
        
        assert len(cutoffs_low) == num_filters and len(cutoffs_high) == num_filters, \
            "Number of cutoff values must match num_filters"
        
        # Create a list of filters
        self.filters = nn.ModuleList([
            FourierFilter(
                input_dim=input_dim,
                filter_type=filter_types[i],
                cutoff_low=cutoffs_low[i],
                cutoff_high=cutoffs_high[i],
                filter_init='lowpass' if filter_types[i] == 'learnable' else 'gaussian',
                return_frequency=False,
            ) for i in range(num_filters)
        ])
        
        # If using attention to combine filters, create attention mechanism
        if combine_strategy == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, num_filters),
                nn.Softmax(dim=-1)
            )
            
        # If using concatenation, create projection to original dimension
        if combine_strategy == 'concat':
            self.projection = nn.Linear(input_dim * num_filters, input_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multiple Fourier filters to create different views of the input signal.
        
        Parameters:
            x : torch.Tensor
                Input tensor of shape (batch_size, seq_len, input_dim)
                
        Returns:
            torch.Tensor
                Filtered signal with shape (batch_size, seq_len, output_dim)
                where output_dim depends on the combine_strategy.
        """
        # Apply each filter to get multiple views
        filtered_views = [filter_module(x) for filter_module in self.filters]
        
        # Combine the filtered views based on the strategy
        if self.combine_strategy == 'concat':
            # Concatenate along the feature dimension
            combined = torch.cat(filtered_views, dim=-1)
            # Project back to original dimension
            return self.projection(combined)
            
        elif self.combine_strategy == 'add':
            # Simply add all filtered views (average if you want)
            return sum(filtered_views) / len(filtered_views)
            
        elif self.combine_strategy == 'attention':
            # Use attention to weight the importance of each view
            # Calculate attention scores
            attention_scores = self.attention(x)  # (batch_size, seq_len, num_filters)
            
            # Apply attention scores to each view
            weighted_views = []
            for i, view in enumerate(filtered_views):
                # Extract the i-th attention score and expand for broadcasting
                score = attention_scores[..., i:i+1]
                weighted_view = view * score
                weighted_views.append(weighted_view)
                
            # Sum the weighted views
            return sum(weighted_views)
            
        else:
            raise ValueError(f"Unknown combine strategy: {self.combine_strategy}")


class AnomalyScoreCalculator(nn.Module):
    """
    Calculates anomaly scores for time series data using frequency domain analysis.
    
    This module compares the original signal to its filtered version to identify
    potential anomalies. Points with high deviation from the filtered signal are
    likely to be anomalies or outliers.
    
    Parameters:
        input_dim : int
            Dimension of the input features.
        filter_type : str, default='lowpass'
            Type of filter to use for generating the baseline signal.
        cutoff : float, default=0.2
            Cutoff frequency for the filter.
        score_type : str, default='mse'
            Method to calculate the anomaly score. Options: 'mse', 'mae', 'z_score'.
        smoothing_window : int, default=5
            Window size for smoothing the anomaly scores.
        threshold_factor : float, default=2.0
            Factor to multiply the mean score to determine the anomaly threshold.
    """
    def __init__(
        self,
        input_dim: int,
        filter_type: str = 'lowpass',
        cutoff: float = 0.2,
        score_type: str = 'mse',
        smoothing_window: int = 5,
        threshold_factor: float = 2.0,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.score_type = score_type
        self.smoothing_window = smoothing_window
        self.threshold_factor = threshold_factor
        
        # Create filter for generating baseline signal
        self.filter = FourierFilter(
            input_dim=input_dim,
            filter_type=filter_type,
            cutoff_low=cutoff,
            cutoff_high=cutoff*2,  # Only used for bandpass/highpass
            return_frequency=False,
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculate anomaly scores for the input time series.
        
        Parameters:
            x : torch.Tensor
                Input tensor of shape (batch_size, seq_len, input_dim)
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                - Anomaly scores of shape (batch_size, seq_len, 1)
                - Anomaly mask (boolean) of shape (batch_size, seq_len, 1)
                - Filtered signal of shape (batch_size, seq_len, input_dim)
        """
        # Generate filtered (baseline) signal
        x_filtered = self.filter(x)
        
        # Calculate deviation from the filtered signal
        if self.score_type == 'mse':
            # Mean squared error between original and filtered
            scores = torch.mean((x - x_filtered) ** 2, dim=-1, keepdim=True)
        elif self.score_type == 'mae':
            # Mean absolute error
            scores = torch.mean(torch.abs(x - x_filtered), dim=-1, keepdim=True)
        elif self.score_type == 'z_score':
            # Z-score of the deviations
            diff = x - x_filtered
            mean = torch.mean(diff, dim=-1, keepdim=True)
            std = torch.std(diff, dim=-1, keepdim=True) + 1e-8  # Add small epsilon to avoid division by zero
            scores = torch.abs((diff - mean) / std)
            scores = torch.mean(scores, dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown score type: {self.score_type}")
        
        # Apply smoothing to the scores if requested
        if self.smoothing_window > 1:
            # Use average pooling for smoothing
            # Reshape for 1D conv
            scores_reshaped = scores.transpose(1, 2)  # (batch_size, 1, seq_len)
            # Apply 1D average pooling
            padding = (self.smoothing_window - 1) // 2
            scores_smoothed = F.avg_pool1d(
                scores_reshaped, 
                kernel_size=self.smoothing_window, 
                stride=1, 
                padding=padding
            )
            # Reshape back
            scores = scores_smoothed.transpose(1, 2)  # (batch_size, seq_len, 1)
        
        # Calculate anomaly threshold
        mean_score = torch.mean(scores, dim=1, keepdim=True)
        std_score = torch.std(scores, dim=1, keepdim=True)
        threshold = mean_score + self.threshold_factor * std_score
        
        # Create anomaly mask
        anomaly_mask = (scores > threshold).float()
        
        return scores, anomaly_mask, x_filtered


class RobustStockAnomalyFilter(nn.Module):
    """
    Robust anomaly filtering module specifically designed for stock data.
    
    This module combines multiple techniques (Fourier filtering, statistical outlier detection,
    and potentially autoencoders) to identify and handle anomalies in stock time series data.
    It can output both the filtered data and uncertainty estimates for downstream tasks.
    
    Parameters:
        input_dim : int
            Dimension of the input features.
        hidden_dim : int, default=64
            Dimension of the hidden representation for the autoencoder (if used).
        filter_types : List[str], default=['lowpass', 'bandpass']
            Types of filters to apply.
        use_autoencoder : bool, default=False
            Whether to use an autoencoder for anomaly detection.
        output_uncertainty : bool, default=True
            Whether to output uncertainty estimates.
        combine_strategy : str, default='attention'
            How to combine multiple filtering results.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        filter_types: List[str] = ['lowpass', 'bandpass'],
        use_autoencoder: bool = False,
        output_uncertainty: bool = True,
        combine_strategy: str = 'attention',
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_uncertainty = output_uncertainty
        self.use_autoencoder = use_autoencoder
        
        # Create a multi-view Fourier filter
        self.fourier_filter = MultiviewFourierFilter(
            input_dim=input_dim,
            num_filters=len(filter_types),
            filter_types=filter_types,
            combine_strategy=combine_strategy,
        )
        
        # Create anomaly score calculator
        self.anomaly_detector = AnomalyScoreCalculator(
            input_dim=input_dim,
            filter_type='lowpass',
            score_type='z_score',
        )
        
        # Optionally create autoencoder for additional anomaly detection
        self.autoencoder = None
        if use_autoencoder:
            self.autoencoder = nn.Sequential(
                # Encoder
                nn.Linear(input_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                # Decoder
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, input_dim),
            )
        
        # Create a mechanism to combine all anomaly detection results
        if output_uncertainty:
            self.uncertainty_projector = nn.Sequential(
                nn.Linear(input_dim + 1, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply robust anomaly filtering to the input stock data.
        
        Parameters:
            x : torch.Tensor
                Input tensor of shape (batch_size, seq_len, input_dim)
                
        Returns:
            torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
                - Filtered signal of shape (batch_size, seq_len, input_dim)
                - If output_uncertainty=True, also returns uncertainty estimates
                  of shape (batch_size, seq_len, 1)
        """
        # Apply Fourier filtering
        x_filtered = self.fourier_filter(x)
        
        # Calculate anomaly scores
        anomaly_scores, anomaly_mask, baseline_filtered = self.anomaly_detector(x)
        
        # If using autoencoder, incorporate that information
        if self.use_autoencoder and self.autoencoder is not None:
            # Get autoencoder reconstruction
            ae_output = self.autoencoder(x)
            
            # Calculate reconstruction error
            ae_error = torch.mean((x - ae_output) ** 2, dim=-1, keepdim=True)
            
            # Normalize error to [0, 1] range
            ae_error_normalized = ae_error / (torch.max(ae_error) + 1e-8)
            
            # Combine with anomaly scores
            anomaly_scores = (anomaly_scores + ae_error_normalized) / 2
            
            # Recalculate anomaly mask
            threshold = torch.mean(anomaly_scores, dim=1, keepdim=True) + \
                        2.0 * torch.std(anomaly_scores, dim=1, keepdim=True)
            anomaly_mask = (anomaly_scores > threshold).float()
        
        # Create the final filtered output
        # For anomalies, use the filtered value; otherwise use the original
        x_robust = x * (1 - anomaly_mask) + x_filtered * anomaly_mask
        
        if self.output_uncertainty:
            # Generate uncertainty estimates
            # Concatenate filtered data with anomaly scores
            features_with_scores = torch.cat([x_robust, anomaly_scores], dim=-1)
            uncertainty = self.uncertainty_projector(features_with_scores)
            return x_robust, uncertainty
        else:
            return x_robust