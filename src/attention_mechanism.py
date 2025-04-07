# attention_mechanism.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Union

class HierarchicalTemporalAttention(nn.Module):
    """
    Hierarchical Temporal Attention module that operates within time bins.
    
    This module implements the novel attention mechanism described in the proposal,
    segmenting the timeline into meaningful bins (e.g., weeks, months) and applying
    attention within each bin first, then across bins. This captures relationships
    between points that are close in time or belong to the same context.
    
    Parameters:
        embed_dim : int
            Dimension of the input embeddings.
        num_heads : int, default=8
            Number of attention heads.
        bin_size : int, default=5
            Size of each temporal bin (e.g., 5 for week, 21 for month in trading days).
        dropout : float, default=0.1
            Dropout rate.
        use_cross_bin_attention : bool, default=True
            Whether to use attention across bins after within-bin attention.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        bin_size: int = 5,  # Default to week (5 trading days)
        dropout: float = 0.1,
        use_cross_bin_attention: bool = True,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bin_size = bin_size
        self.use_cross_bin_attention = use_cross_bin_attention
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        self.head_dim = embed_dim // num_heads
        
        # Within-bin attention
        self.within_bin_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-bin attention (if used)
        if use_cross_bin_attention:
            self.cross_bin_attention = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        # Output projection and normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.layer_norm3 = nn.LayerNorm(embed_dim) if use_cross_bin_attention else None
        
        # Feed-forward networks
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Apply hierarchical temporal attention to the input sequence.
        
        Parameters:
            x : torch.Tensor
                Input tensor of shape (batch_size, seq_len, embed_dim)
            mask : torch.Tensor, optional
                Mask to apply to attention weights.
            return_attention : bool, default=False
                Whether to return attention weights for visualization/analysis.
                
        Returns:
            torch.Tensor or Tuple[torch.Tensor, Dict[str, torch.Tensor]]
                - Output tensor of shape (batch_size, seq_len, embed_dim)
                - If return_attention=True, also returns a dictionary of attention weights
        """
        batch_size, seq_len, embed_dim = x.shape
        
        # Pad sequence if needed to be divisible by bin_size
        pad_len = 0
        if seq_len % self.bin_size != 0:
            pad_len = self.bin_size - (seq_len % self.bin_size)
            padding = torch.zeros(batch_size, pad_len, embed_dim, device=x.device)
            x_padded = torch.cat([x, padding], dim=1)
            # If mask provided, pad it too
            if mask is not None:
                mask_padding = torch.zeros(batch_size, pad_len, device=mask.device, dtype=mask.dtype)
                mask = torch.cat([mask, mask_padding], dim=1)
        else:
            x_padded = x
        
        # Calculate number of bins
        padded_seq_len = seq_len + pad_len
        num_bins = padded_seq_len // self.bin_size
        
        # Reshape to separate bins: (batch_size, num_bins, bin_size, embed_dim)
        x_binned = x_padded.view(batch_size, num_bins, self.bin_size, embed_dim)
        
        # Process mask if provided
        binned_mask = None
        if mask is not None:
            binned_mask = mask.view(batch_size, num_bins, self.bin_size)
        
        # Apply within-bin attention for each bin
        within_attn_outputs = []
        within_attn_weights = []
        
        for bin_idx in range(num_bins):
            # Extract this bin's data
            bin_data = x_binned[:, bin_idx, :, :]  # (batch_size, bin_size, embed_dim)
            
            # Extract this bin's mask if available
            bin_mask = None
            if binned_mask is not None:
                bin_mask = binned_mask[:, bin_idx, :]  # (batch_size, bin_size)
                
                # Create attention mask from sequence mask
                attn_mask = torch.zeros(batch_size, bin_data.size(1), bin_data.size(1), 
                                        device=bin_data.device, dtype=torch.bool)
                for b in range(batch_size):
                    valid_len = torch.sum(bin_mask[b])
                    if valid_len < bin_data.size(1):
                        # Mask out padding tokens
                        attn_mask[b, :, valid_len:] = True
                        attn_mask[b, valid_len:, :] = True
            else:
                attn_mask = None
            
            # Apply attention within this bin
            bin_output, bin_attn_weights = self.within_bin_attention(
                query=bin_data,
                key=bin_data,
                value=bin_data,
                attn_mask=attn_mask if attn_mask is not None else None,
                need_weights=return_attention
            )
            
            # Apply residual connection and layer normalization
            bin_output = self.layer_norm1(bin_data + bin_output)
            
            # Apply feed-forward network
            ffn_output = self.ffn(bin_output)
            bin_output = self.layer_norm2(bin_output + ffn_output)
            
            within_attn_outputs.append(bin_output)
            if return_attention:
                within_attn_weights.append(bin_attn_weights)
        
        # Stack outputs from all bins
        within_bin_output = torch.stack(within_attn_outputs, dim=1)  # (batch_size, num_bins, bin_size, embed_dim)
        
        # If using cross-bin attention
        cross_bin_attn_weights = None
        if self.use_cross_bin_attention:
            # Reshape to (batch_size, num_bins * bin_size, embed_dim) for cross-bin attention
            reshaped_output = within_bin_output.view(batch_size, num_bins, self.bin_size, embed_dim)
            
            # Create bin-level representations (e.g., by mean-pooling within each bin)
            bin_reps = torch.mean(reshaped_output, dim=2)  # (batch_size, num_bins, embed_dim)
            
            # Apply cross-bin attention
            cross_bin_output, cross_bin_attn_weights = self.cross_bin_attention(
                query=bin_reps,
                key=bin_reps,
                value=bin_reps,
                need_weights=return_attention
            )
            
            # Apply residual connection and layer normalization
            cross_bin_output = self.layer_norm3(bin_reps + cross_bin_output)  # (batch_size, num_bins, embed_dim)
            
            # Expand cross-bin attention results to influence all timepoints within each bin
            expanded_cross_bin = cross_bin_output.unsqueeze(2).expand(-1, -1, self.bin_size, -1)
            
            # Combine with within-bin output (e.g., by addition)
            combined_output = within_bin_output + expanded_cross_bin
        else:
            combined_output = within_bin_output
        
        # Reshape back to (batch_size, seq_len, embed_dim)
        output = combined_output.reshape(batch_size, padded_seq_len, embed_dim)
        
        # Remove padding if added
        if pad_len > 0:
            output = output[:, :seq_len, :]
        
        # Return output and attention weights if requested
        if return_attention:
            attention_weights = {
                'within_bin': torch.stack(within_attn_weights, dim=1) if within_attn_weights else None,
                'cross_bin': cross_bin_attn_weights
            }
            return output, attention_weights
        else:
            return output


class CoAttentionModule(nn.Module):
    """
    Co-Attention Module that models interactions between stock features and time.
    
    This module allows the model to focus on different features at different time points,
    capturing the varying importance of different signals over time. It implements a
    bidirectional attention mechanism between time and features.
    
    Parameters:
        hidden_dim : int
            Dimension of the hidden representation.
        num_heads : int, default=4
            Number of attention heads.
        dropout : float, default=0.1
            Dropout rate.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Feature transformation layers
        self.feature_linear = nn.Linear(hidden_dim, hidden_dim)
        self.time_linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Attention mechanisms
        self.feature_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.time_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projections
        self.feature_output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        self.time_output = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        feature_repr: torch.Tensor,
        time_repr: torch.Tensor,
        return_attention: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], 
               Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Apply co-attention between feature and time representations.
        
        Parameters:
            feature_repr : torch.Tensor
                Feature representation of shape (batch_size, num_features, hidden_dim)
            time_repr : torch.Tensor
                Time representation of shape (batch_size, seq_len, hidden_dim)
            return_attention : bool, default=False
                Whether to return attention weights.
                
        Returns:
            Tuple[torch.Tensor, torch.Tensor] or Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]
                - Enhanced feature representation of shape (batch_size, num_features, hidden_dim)
                - Enhanced time representation of shape (batch_size, seq_len, hidden_dim)
                - If return_attention=True, also returns a dictionary of attention weights
        """
        batch_size, num_features, _ = feature_repr.shape
        _, seq_len, _ = time_repr.shape
        
        # Transform feature and time representations
        feature_q = self.feature_linear(feature_repr)
        time_q = self.time_linear(time_repr)
        
        # Feature-to-time attention: How features attend to time points
        f2t_output, f2t_attn = self.feature_attn(
            query=feature_q,
            key=time_q,
            value=time_repr,
            need_weights=return_attention
        )
        
        # Time-to-feature attention: How time points attend to features
        t2f_output, t2f_attn = self.time_attn(
            query=time_q,
            key=feature_q,
            value=feature_repr,
            need_weights=return_attention
        )
        
        # Combine original and attention-enhanced representations
        enhanced_feature = self.feature_output(torch.cat([feature_repr, f2t_output], dim=-1))
        enhanced_time = self.time_output(torch.cat([time_repr, t2f_output], dim=-1))
        
        if return_attention:
            attention_weights = {
                'feature_to_time': f2t_attn,
                'time_to_feature': t2f_attn
            }
            return enhanced_feature, enhanced_time, attention_weights
        else:
            return enhanced_feature, enhanced_time