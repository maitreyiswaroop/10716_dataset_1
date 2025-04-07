# temporal_encoder.py

import numpy as np
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union

class KernelizedTimeEncoder(nn.Module):
    """
    Kernelized Time Encoder based on Bochner's theorem.
    
    This module creates time embeddings using a spectral approach inspired by Bochner's theorem,
    which states that any continuous positive-definite kernel can be represented as the
    Fourier transform of a non-negative measure.
    
    Parameters:
        embed_dim : int
            Dimension of the output embedding (must be even).
        num_frequencies : int, default=None
            Number of frequency components. If None, uses embed_dim//2.
        learnable : bool, default=True
            If True, frequency parameters are learned during training.
        time_scale : float, default=1.0
            Scale factor applied to input time values.
        base : float, default=10000.0
            Base for frequency scaling.
        trainable_time_shift : bool, default=False
            If True, adds a trainable time shift parameter.
    """
    def __init__(
        self,
        embed_dim: int,
        num_frequencies: Optional[int] = None,
        learnable: bool = True,
        time_scale: float = 1.0,
        base: float = 10000.0,
        trainable_time_shift: bool = False,
    ):
        super().__init__()
        assert embed_dim % 2 == 0, "Embedding dimension must be even"
        
        self.embed_dim = embed_dim
        self.time_scale = time_scale
        self.num_frequencies = num_frequencies or (embed_dim // 2)
        
        # Create frequency bands similar to transformer positional encoding
        # but make them learnable if specified
        if learnable:
            # Initialize with transformer-like frequency distribution
            inv_freq = 1.0 / (base ** (torch.arange(0, self.num_frequencies).float() / self.num_frequencies))
            self.frequencies = nn.Parameter(inv_freq, requires_grad=True)
        else:
            inv_freq = 1.0 / (base ** (torch.arange(0, self.num_frequencies).float() / self.num_frequencies))
            self.register_buffer("frequencies", inv_freq)
        
        self.trainable_time_shift = trainable_time_shift
        if trainable_time_shift:
            self.time_shift = nn.Parameter(torch.zeros(1), requires_grad=True)
        
        # Create a projection layer to map from frequency space to embedding space if needed
        self.projection = None
        if 2 * self.num_frequencies != embed_dim:
            self.projection = nn.Linear(2 * self.num_frequencies, embed_dim)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode time values into high-dimensional embeddings.
        
        Parameters:
            t : torch.Tensor
                Time values of shape (...) - can be any shape
                
        Returns:
            torch.Tensor
                Time embeddings of shape (..., embed_dim)
        """
        # Get device from input tensor
        device = t.device
        
        # Scale time and apply shift if applicable
        if self.trainable_time_shift:
            t = t * self.time_scale + self.time_shift
        else:
            t = t * self.time_scale
            
        # Reshape time for broadcasting with frequencies
        time_shape = list(t.shape) + [1]  # Add dimension for frequencies
        t_reshaped = t.view(*time_shape)
        
        # Ensure frequencies is on the same device as t
        frequencies = self.frequencies.to(device)
        
        # Compute phase: time * frequency for each frequency component
        # Shape: (..., num_frequencies)
        phase = t_reshaped * frequencies
        
        # Apply sine and cosine to get the embeddings
        # Shape: (..., num_frequencies)
        sin_embeddings = torch.sin(phase)
        cos_embeddings = torch.cos(phase)
        
        # Concatenate sin and cos embeddings
        # Shape: (..., 2*num_frequencies)
        embeddings = torch.cat([sin_embeddings, cos_embeddings], dim=-1)
        
        # Apply projection if needed
        if self.projection is not None:
            embeddings = self.projection(embeddings)
            
        return embeddings

class MultiScaleTimeEncoder(nn.Module):
    """
    Multi-Scale Time Encoder that captures patterns at different time scales.
    
    This encoder combines multiple kernelized time encoders, each focusing on
    a different time scale (e.g., days, weeks, months, years). This helps in
    capturing both short-term and long-term temporal patterns.
    
    Parameters:
        embed_dim : int
            Dimension of the output embedding.
        num_scales : int, default=4
            Number of time scales to encode.
        scale_factors : list of float, optional
            Scale factors for each time encoder. If None, uses [1, 1/7, 1/30, 1/365].
        learnable : bool, default=True
            If True, frequency parameters are learned during training.
    """
    def __init__(
        self,
        embed_dim: int,
        num_scales: int = 4,
        scale_factors: Optional[list] = None,
        learnable: bool = True,
    ):
        super().__init__()
        
        # Ensure embed_dim is divisible by num_scales
        assert embed_dim % num_scales == 0, "Embedding dimension must be divisible by num_scales"
        
        self.embed_dim = embed_dim
        self.num_scales = num_scales
        
        # Default scale factors for day, week, month, year
        if scale_factors is None:
            # For financial data, use scales relevant to trading:
            # Daily, weekly, monthly, quarterly
            scale_factors = [1.0, 1.0/5, 1.0/21, 1.0/63]  # Trading days
        
        assert len(scale_factors) == num_scales, "Must provide scale factor for each time scale"
        
        # Create a time encoder for each scale
        self.time_encoders = nn.ModuleList([
            KernelizedTimeEncoder(
                embed_dim=embed_dim // num_scales,
                learnable=learnable,
                time_scale=scale_factors[i],
                trainable_time_shift=(i > 0),  # Allow time shifts for non-daily scales
            ) for i in range(num_scales)
        ])
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode time values at multiple scales.
        
        Parameters:
            t : torch.Tensor
                Time values of shape (...)
                
        Returns:
            torch.Tensor
                Multi-scale time embeddings of shape (..., embed_dim)
        """
        # Get embeddings at each scale
        embeddings = [encoder(t) for encoder in self.time_encoders]
        
        # Concatenate all scale embeddings
        return torch.cat(embeddings, dim=-1)


class CalendarTimeEncoder(nn.Module):
    """
    Calendar-aware Time Encoder that extracts date-specific features.
    
    This encoder extracts calendar features like day of week, day of month, 
    month of year, etc., and embeds them into a fixed-dimensional space.
    This helps the model learn patterns specific to calendar cycles.
    
    Parameters:
        embed_dim : int
            Dimension of the output embedding.
        include_time_of_day : bool, default=False
            If True, includes time of day features (assuming input has intraday timestamps).
        base_date : str, default='2021-01-01'
            Base date for converting integer day indices to calendar dates.
    """
    def __init__(
        self,
        embed_dim: int,
        include_time_of_day: bool = False,
        base_date: str = '2021-01-01',
    ):
        super().__init__()
        import datetime
        
        self.embed_dim = embed_dim
        self.include_time_of_day = include_time_of_day
        
        # Parse base date (used to convert day indices to actual dates)
        self.base_date = datetime.datetime.strptime(base_date, '%Y-%m-%d')
        
        # Define the feature dimensions
        # Calendar features: [day_of_week, day_of_month, day_of_year, month, quarter, is_month_start, is_month_end, is_quarter_start, is_quarter_end, is_year_start, is_year_end]
        self.num_features = 11
        
        if include_time_of_day:
            # Add time features: [hour, minute, is_market_open]
            self.num_features += 3
        
        # Define embedding layers for categorical features
        self.day_of_week_embed = nn.Embedding(7, embed_dim // self.num_features)
        self.month_embed = nn.Embedding(12, embed_dim // self.num_features)
        self.quarter_embed = nn.Embedding(4, embed_dim // self.num_features)
        
        # Linear projections for numerical features
        self.day_of_month_proj = nn.Linear(1, embed_dim // self.num_features)
        self.day_of_year_proj = nn.Linear(1, embed_dim // self.num_features)
        
        # Binary features projection
        self.binary_proj = nn.Linear(6, embed_dim // self.num_features * 6)
        
        if include_time_of_day:
            self.hour_embed = nn.Embedding(24, embed_dim // self.num_features)
            self.minute_proj = nn.Linear(1, embed_dim // self.num_features)
            self.market_status_embed = nn.Embedding(2, embed_dim // self.num_features)
    
    def _day_index_to_date(self, day_index):
        """Convert integer day index to a datetime object."""
        import datetime
        return self.base_date + datetime.timedelta(days=int(day_index))
    
    def _extract_calendar_features(self, day_indices):
        """Extract calendar features from day indices."""
        import datetime
        import pandas as pd
        
        # Convert day indices to dates
        dates = [self._day_index_to_date(idx.item()) for idx in day_indices.view(-1)]
        dates = pd.DatetimeIndex(dates)
        
        # Extract calendar features
        day_of_week = torch.tensor([d.weekday() for d in dates], device=day_indices.device)
        day_of_month = torch.tensor([d.day for d in dates], device=day_indices.device, dtype=torch.float32).unsqueeze(-1)
        day_of_year = torch.tensor([d.timetuple().tm_yday for d in dates], device=day_indices.device, dtype=torch.float32).unsqueeze(-1)
        month = torch.tensor([d.month - 1 for d in dates], device=day_indices.device)  # 0-indexed
        quarter = torch.tensor([(d.month - 1) // 3 for d in dates], device=day_indices.device)
        
        # Binary features
        is_month_start = torch.tensor([d.is_month_start for d in dates], device=day_indices.device, dtype=torch.float32)
        is_month_end = torch.tensor([d.is_month_end for d in dates], device=day_indices.device, dtype=torch.float32)
        is_quarter_start = torch.tensor([d.is_quarter_start for d in dates], device=day_indices.device, dtype=torch.float32)
        is_quarter_end = torch.tensor([d.is_quarter_end for d in dates], device=day_indices.device, dtype=torch.float32)
        is_year_start = torch.tensor([d.is_year_start for d in dates], device=day_indices.device, dtype=torch.float32)
        is_year_end = torch.tensor([d.is_year_end for d in dates], device=day_indices.device, dtype=torch.float32)
        
        # Combine binary features
        binary_features = torch.stack([
            is_month_start, is_month_end, is_quarter_start, 
            is_quarter_end, is_year_start, is_year_end
        ], dim=-1)
        
        # Reshape tensors to match input shape
        original_shape = list(day_indices.shape)
        new_shape = original_shape + [-1]
        
        day_of_week = day_of_week.view(*original_shape)
        day_of_month = day_of_month.view(*original_shape, 1)
        day_of_year = day_of_year.view(*original_shape, 1)
        month = month.view(*original_shape)
        quarter = quarter.view(*original_shape)
        binary_features = binary_features.view(*original_shape, 6)
        
        return day_of_week, day_of_month, day_of_year, month, quarter, binary_features
    
    def forward(self, day_indices: torch.Tensor) -> torch.Tensor:
        """
        Encode day indices into calendar-aware embeddings.
        
        Parameters:
            day_indices : torch.Tensor
                Integer day indices of shape (...)
                
        Returns:
            torch.Tensor
                Calendar embeddings of shape (..., embed_dim)
        """
        # Extract calendar features
        day_of_week, day_of_month, day_of_year, month, quarter, binary_features = self._extract_calendar_features(day_indices)
        
        # Embed categorical features
        dow_embed = self.day_of_week_embed(day_of_week)
        month_embed = self.month_embed(month)
        quarter_embed = self.quarter_embed(quarter)
        
        # Project numerical features
        dom_embed = self.day_of_month_proj(day_of_month)
        doy_embed = self.day_of_year_proj(day_of_year)
        
        # Project binary features
        binary_embed = self.binary_proj(binary_features)
        binary_embed = binary_embed.view(*binary_features.shape[:-1], 6, -1)
        binary_embed = binary_embed.sum(dim=-2)  # Combine the 6 binary features
        
        # Concatenate all embeddings
        embeddings = [dow_embed, dom_embed, doy_embed, month_embed, quarter_embed, binary_embed]
        
        if self.include_time_of_day:
            # For stock data, we typically don't have time of day
            # but this would be implemented here if needed
            pass
        
        # Combine all embeddings
        return torch.cat(embeddings, dim=-1)


class HybridTimeEncoder(nn.Module):
    """
    Hybrid Time Encoder that combines spectral and calendar-aware encodings.
    
    This encoder combines the strengths of both the Kernelized Time Encoder
    (capturing continuous time patterns) and the Calendar Time Encoder
    (capturing discrete calendar patterns).
    
    Parameters:
        embed_dim : int
            Dimension of the final embedding.
        spectral_dim : int, default=None
            Dimension of the spectral embedding. If None, uses embed_dim // 2.
        num_scales : int, default=4
            Number of time scales for the spectral encoder.
        learnable : bool, default=True
            If True, frequency parameters are learned during training.
        include_calendar : bool, default=True
            If True, includes calendar embeddings.
        base_date : str, default='2021-01-01'
            Base date for calendar encoding.
        final_projection : bool, default=True
            If True, adds a final linear projection layer.
    """
    def __init__(
        self,
        embed_dim: int,
        spectral_dim: Optional[int] = None,
        num_scales: int = 4,
        learnable: bool = True,
        include_calendar: bool = True,
        base_date: str = '2021-01-01',
        final_projection: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.include_calendar = include_calendar
        
        # Determine dimensions
        if spectral_dim is None:
            spectral_dim = embed_dim // 2 if include_calendar else embed_dim
        
        calendar_dim = embed_dim - spectral_dim if include_calendar else 0
        
        # Create spectral encoder (Kernelized, Multi-scale)
        self.spectral_encoder = MultiScaleTimeEncoder(
            embed_dim=spectral_dim,
            num_scales=num_scales,
            learnable=learnable,
        )
        
        # Create calendar encoder if needed
        self.calendar_encoder = None
        if include_calendar:
            self.calendar_encoder = CalendarTimeEncoder(
                embed_dim=calendar_dim,
                base_date=base_date,
            )
            
        # Add final projection
        self.final_projection = None
        if final_projection:
            # Ensure input dimension matches the concatenated embeddings
            input_dim = spectral_dim + calendar_dim
            self.final_projection = nn.Sequential(
                nn.Linear(input_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.ReLU()
            )
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode time values using both spectral and calendar features.
        
        Parameters:
            t : torch.Tensor
                Time values (day indices) of shape (...)
                
        Returns:
            torch.Tensor
                Hybrid time embeddings of shape (..., embed_dim)
        """
        # Get device of input
        device = t.device
        
        # Get spectral embeddings - move encoders to correct device
        self.spectral_encoder = self.spectral_encoder.to(device)
        spectral_embed = self.spectral_encoder(t)
        
        # Get calendar embeddings if needed
        if self.include_calendar and self.calendar_encoder is not None:
            self.calendar_encoder = self.calendar_encoder.to(device)
            calendar_embed = self.calendar_encoder(t)
            # Concatenate the embeddings
            embeddings = torch.cat([spectral_embed, calendar_embed], dim=-1)
        else:
            embeddings = spectral_embed
            
        # Apply final projection if available
        if self.final_projection is not None:
            # Debugging: print shapes
            print(f"Embeddings shape before projection: {embeddings.shape}")
            # Move projection to correct device
            self.final_projection = self.final_projection.to(device)
            embeddings = self.final_projection(embeddings)
            
        return embeddings
    
def get_time_encoder(
    embed_dim: int,
    encoder_type: str = 'hybrid',
    **kwargs
) -> nn.Module:
    """
    Factory function to create a time encoder based on the specified type.
    
    Parameters:
        embed_dim : int
            Dimension of the output embedding.
        encoder_type : str, default='hybrid'
            Type of time encoder to create. Options: 'kernel', 'multiscale', 'calendar', 'hybrid'.
        **kwargs : 
            Additional arguments to pass to the selected encoder.
            
    Returns:
        nn.Module
            The specified time encoder instance.
    """
    if encoder_type == 'kernel':
        return KernelizedTimeEncoder(embed_dim=embed_dim, **kwargs)
    elif encoder_type == 'multiscale':
        return MultiScaleTimeEncoder(embed_dim=embed_dim, **kwargs)
    elif encoder_type == 'calendar':
        return CalendarTimeEncoder(embed_dim=embed_dim, **kwargs)
    elif encoder_type == 'hybrid':
        return HybridTimeEncoder(embed_dim=embed_dim, **kwargs)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")