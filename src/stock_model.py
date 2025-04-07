# stock_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Union, Any

from feature_encoder import CNNFeatureEncoder
from temporal_encoder import HybridTimeEncoder
from anomaly_filter import RobustStockAnomalyFilter
from attention_mechanism import HierarchicalTemporalAttention, CoAttentionModule

class TransformerEncoderLayer(nn.Module):
    """
    A standard Transformer encoder layer.
    
    Parameters:
        d_model : int
            Dimension of the model
        nhead : int
            Number of attention heads
        dim_feedforward : int
            Dimension of the feedforward network
        dropout : float
            Dropout probability
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.ReLU()
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the Transformer encoder layer.
        
        Parameters:
            src : torch.Tensor
                Source sequence, shape (batch_size, seq_len, d_model)
            src_mask : torch.Tensor, optional
                Mask for the src sequence
            src_key_padding_mask : torch.Tensor, optional
                Key padding mask for src
                
        Returns:
            torch.Tensor
                Output of shape (batch_size, seq_len, d_model)
        """
        # Multi-head self-attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask,
                                  key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feed-forward network
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class StockTransformer(nn.Module):
    """
    Transformer encoder model for stock prediction.
    
    Parameters:
        d_model : int
            Dimension of the model
        nhead : int
            Number of attention heads
        num_layers : int
            Number of transformer layers
        dim_feedforward : int
            Dimension of the feedforward network
        dropout : float
            Dropout probability
        positional_encoding : bool
            Whether to use positional encoding
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.use_positional_encoding = use_positional_encoding
        
        # Create transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Optional positional encoding
        self.pos_encoder = None
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(d_model, dropout)
    
    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the transformer.
        
        Parameters:
            src : torch.Tensor
                Source sequence, shape (batch_size, seq_len, d_model)
            src_mask : torch.Tensor, optional
                Mask for the src sequence
            src_key_padding_mask : torch.Tensor, optional
                Key padding mask for src
                
        Returns:
            torch.Tensor
                Output of shape (batch_size, seq_len, d_model)
        """
        output = src
        
        # Apply positional encoding if requested
        if self.use_positional_encoding and self.pos_encoder is not None:
            output = self.pos_encoder(output)
        
        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            output = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
        
        return output


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for transformer models.
    
    Parameters:
        d_model : int
            Dimension of the model
        dropout : float
            Dropout probability
        max_len : int
            Maximum sequence length
    """
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
        max_len: int = 5000,
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (persistent state)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Parameters:
            x : torch.Tensor
                Input tensor of shape (batch_size, seq_len, d_model)
                
        Returns:
            torch.Tensor
                Output with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class StockPredictionModel(nn.Module):
    """
    Complete stock prediction model implementing the proposed approach.
    
    This model combines:
    1. Feature encoding
    2. Time embedding
    3. Anomaly filtering
    4. Transformer-based processing
    5. Hierarchical temporal attention
    6. Co-attention mechanism
    
    Parameters:
        input_dim : int
            Dimension of the input features
        time_dim : int
            Dimension of the time embedding
        hidden_dim : int
            Dimension of the hidden representation
        output_dim : int, default=1
            Dimension of the output (prediction)
        num_transformer_layers : int, default=3
            Number of transformer encoder layers
        num_attention_heads : int, default=8
            Number of attention heads in transformer
        temporal_bin_size : int, default=5
            Size of temporal bins for hierarchical attention
        dropout : float, default=0.1
            Dropout probability
    """
    def __init__(
        self,
        input_dim: int,
        time_dim: int,
        hidden_dim: int,
        output_dim: int = 1,
        num_transformer_layers: int = 3,
        num_attention_heads: int = 8,
        temporal_bin_size: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 1. Feature Encoder
        self.feature_encoder = CNNFeatureEncoder(
            input_dim=input_dim,
            hidden_channels=hidden_dim,
            output_dim=hidden_dim,
            kernel_size=3,
            num_layers=2,
            activation=nn.ReLU
        )
        
        # 2. Time Embedding
        self.time_encoder = HybridTimeEncoder(
            embed_dim=time_dim,
            spectral_dim=time_dim // 2,
            num_scales=4,
            learnable=True,
            include_calendar=True,
            final_projection=True
        )
        
        # 3. Anomaly Filter
        self.anomaly_filter = RobustStockAnomalyFilter(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            filter_types=['lowpass', 'bandpass', 'highpass'],
            use_autoencoder=False,
            output_uncertainty=True,
            combine_strategy='attention'
        )
        
        # 4. Transformer Encoder
        self.transformer = StockTransformer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            num_layers=num_transformer_layers,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            use_positional_encoding=False  # We use our custom time encoding
        )
        
        # 5. Hierarchical Temporal Attention
        self.hierarchical_attention = HierarchicalTemporalAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            bin_size=temporal_bin_size,
            dropout=dropout,
            use_cross_bin_attention=True
        )
        
        # 6. Co-Attention Module
        self.co_attention = CoAttentionModule(
            hidden_dim=hidden_dim,
            num_heads=num_attention_heads // 2,
            dropout=dropout
        )
        
        # Final prediction layers
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        time_indices: torch.Tensor,
        stock_indices: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Any]]]:
        """
        Forward pass through the stock prediction model.
        
        Parameters:
            x : torch.Tensor
                Input features of shape (batch_size, seq_len, input_dim)
            time_indices : torch.Tensor
                Time indices of shape (batch_size, seq_len)
            stock_indices : torch.Tensor, optional
                Stock indices of shape (batch_size, seq_len)
            return_attention : bool, default=False
                Whether to return attention weights for visualization
                
        Returns:
            torch.Tensor or Tuple[torch.Tensor, Dict[str, Any]]
                - Predicted stock returns of shape (batch_size, seq_len, output_dim)
                - If return_attention=True, also returns a dictionary of attention weights and intermediate outputs
        """
        batch_size, seq_len, _ = x.shape
        
        # Save intermediate outputs for return if needed
        intermediate_outputs = {}
        attention_weights = {}
        
        # 1. Apply anomaly filtering
        x_filtered, uncertainty = self.anomaly_filter(x)
        intermediate_outputs['filtered_input'] = x_filtered
        intermediate_outputs['uncertainty'] = uncertainty
        
        # 2. Create time embeddings
        time_embeddings = self.time_encoder(time_indices)
        intermediate_outputs['time_embeddings'] = time_embeddings
        
        # 3. Apply feature encoding (through CNN)
        # We need to process each sequence element separately
        feature_embeddings = []
        for t in range(seq_len):
            # Extract features at this time step for all batches
            x_t = x_filtered[:, t, :]  # (batch_size, input_dim)
            # Add a sequence dimension for the CNN (expects batch, seq, dim)
            x_t = x_t.unsqueeze(1)  # (batch_size, 1, input_dim)
            # Apply feature encoder
            feat_t = self.feature_encoder(x_t)  # (batch_size, hidden_dim)
            feature_embeddings.append(feat_t)
        
        # Stack the embeddings along sequence dimension
        feature_embeddings = torch.stack(feature_embeddings, dim=1)  # (batch_size, seq_len, hidden_dim)
        intermediate_outputs['feature_embeddings'] = feature_embeddings
        
        # 4. Apply transformer encoder
        # Combine feature embeddings with time information
        combined_embeddings = feature_embeddings + time_embeddings
        transformer_output = self.transformer(combined_embeddings)
        intermediate_outputs['transformer_output'] = transformer_output
        
        # 5. Apply hierarchical temporal attention
        if return_attention:
            hier_attn_output, hier_attn_weights = self.hierarchical_attention(
                transformer_output, return_attention=True)
            attention_weights['hierarchical'] = hier_attn_weights
        else:
            hier_attn_output = self.hierarchical_attention(transformer_output)
        
        intermediate_outputs['hierarchical_attention_output'] = hier_attn_output
        
        # 6. Apply co-attention between features and time
        # Reshape feature_embeddings to (batch_size, input_dim, hidden_dim) for co-attention
        feature_repr = feature_embeddings.transpose(1, 2)  # (batch_size, hidden_dim, seq_len)
        feature_repr = feature_repr.transpose(1, 2)  # (batch_size, seq_len, hidden_dim)
        
        # Use hierarchical attention output as time representation
        time_repr = hier_attn_output
        
        if return_attention:
            enhanced_features, enhanced_time, co_attn_weights = self.co_attention(
                feature_repr, time_repr, return_attention=True)
            attention_weights['co_attention'] = co_attn_weights
        else:
            enhanced_features, enhanced_time = self.co_attention(feature_repr, time_repr)
        
        intermediate_outputs['enhanced_features'] = enhanced_features
        intermediate_outputs['enhanced_time'] = enhanced_time
        
        # 7. Combine enhanced representations for final prediction
        # Concatenate enhanced time and feature representations
        combined_repr = torch.cat([enhanced_time, enhanced_features], dim=-1)
        
        # Apply prediction head
        predictions = self.prediction_head(combined_repr)
        
        if return_attention:
            return predictions, {'attention': attention_weights, 'intermediates': intermediate_outputs}
        else:
            return predictions


class InductiveStockPredictor(nn.Module):
    """
    Inductive stock prediction model capable of forecasting multiple future steps.
    
    This model extends the base StockPredictionModel to support multi-step prediction
    in an inductive manner, meaning it can generalize to stocks not seen during training.
    
    Parameters:
        input_dim : int
            Dimension of the input features
        time_dim : int
            Dimension of the time embedding
        hidden_dim : int
            Dimension of the hidden representation
        forecast_steps : int, default=1
            Number of future steps to predict
        use_uncertainty : bool, default=True
            Whether to incorporate uncertainty estimates into predictions
    """
    def __init__(
        self,
        input_dim: int,
        time_dim: int,
        hidden_dim: int,
        forecast_steps: int = 1,
        use_uncertainty: bool = True,
        **kwargs
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.time_dim = time_dim
        self.hidden_dim = hidden_dim
        self.forecast_steps = forecast_steps
        self.use_uncertainty = use_uncertainty
        
        # Base stock prediction model
        self.base_model = StockPredictionModel(
            input_dim=input_dim,
            time_dim=time_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,  # Output latent representation instead of direct prediction
            **kwargs
        )
        
        # Multi-step prediction head
        self.multi_step_predictor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)  # Predict one value per step
            ) for _ in range(forecast_steps)
        ])
        
        # Uncertainty weighting for predictions
        if use_uncertainty:
            self.uncertainty_weight = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.Sigmoid()
            )
    
    def forward(
        self,
        x: torch.Tensor,
        time_indices: torch.Tensor,
        stock_indices: Optional[torch.Tensor] = None,
        future_time_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-step stock prediction.
        
        Parameters:
            x : torch.Tensor
                Input features of shape (batch_size, seq_len, input_dim)
            time_indices : torch.Tensor
                Time indices of shape (batch_size, seq_len)
            stock_indices : torch.Tensor, optional
                Stock indices of shape (batch_size, seq_len)
            future_time_indices : torch.Tensor, optional
                Future time indices for prediction. If None, assumes consecutive days.
                
        Returns:
            Dict[str, torch.Tensor]
                Dictionary containing:
                - 'predictions': Tensor of shape (batch_size, forecast_steps)
                - 'uncertainty': Tensor of shape (batch_size, forecast_steps) if use_uncertainty=True
        """
        batch_size, seq_len, _ = x.shape
        
        # Get base model outputs and intermediates
        base_output, intermediates = self.base_model(
            x, time_indices, stock_indices, return_attention=True)
        
        # Extract the latent representation for the last time step
        final_repr = base_output[:, -1, :]  # (batch_size, hidden_dim)
        
        # Get uncertainty estimates if available
        uncertainty = None
        if self.use_uncertainty and 'uncertainty' in intermediates['intermediates']:
            uncertainty = intermediates['intermediates']['uncertainty'][:, -1, :]  # (batch_size, 1)
            
            # Apply uncertainty weighting
            uncertainty_weights = self.uncertainty_weight(uncertainty)
            final_repr = final_repr * uncertainty_weights
        
        # Generate predictions for each future step
        predictions = []
        for step in range(self.forecast_steps):
            step_prediction = self.multi_step_predictor[step](final_repr)
            predictions.append(step_prediction)
        
        # Stack predictions along a new dimension
        stacked_predictions = torch.cat(predictions, dim=1)  # (batch_size, forecast_steps)
        
        # Create return dictionary
        result = {'predictions': stacked_predictions}
        
        if uncertainty is not None:
            # Expand uncertainty to match predictions
            expanded_uncertainty = uncertainty.expand(-1, self.forecast_steps)
            result['uncertainty'] = expanded_uncertainty
        
        return result