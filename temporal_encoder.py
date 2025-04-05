import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TemporalEncoding(nn.Module):
    def __init__(self, d_out, num_frequencies=16, method='mlp_res'):
        """
        Temporal Encoding layer using sinusoidal embeddings based on Bochner's theorem.

        Args:
            d_out (int): Dimensionality of output time embeddings.
            num_frequencies (int): Number of frequencies to use in the encoding.
            method (str): Method for generating frequencies (currently only 'mlp_res' is supported).
        """
        super(TemporalEncoding, self).__init__()
        self.d_out = d_out
        self.num_frequencies = num_frequencies
        self.method = method

        # Define MLP layers for the 'mlp_res' method (as used in curvature encoding)
        if method == 'mlp_res':
            self.mlp1 = nn.Linear(num_frequencies, num_frequencies)
            self.mlp2 = nn.Linear(num_frequencies, num_frequencies)
            self.mlp3 = nn.Linear(num_frequencies, num_frequencies)
        else:
            raise ValueError(f"Method '{self.method}' not implemented in TemporalEncoding.")

        # Linear layer to project the concatenated sine and cosine features to dimension d_out
        self.projection = nn.Linear(2 * num_frequencies, d_out)

    def generate_frequencies(self, device):
        """
        Generates frequencies using an MLP-based inverse CDF transformation.

        Returns:
            Tensor of shape (1, num_frequencies) with generated frequencies.
        """
        if self.method == 'mlp_res':
            # Sample frequencies from a uniform distribution
            sampled_freq = torch.rand(1, self.num_frequencies, device=device)
            # Apply transformation: sampled_freq = 1 / (10 ** sampled_freq)
            sampled_freq = 1 / (10 ** sampled_freq)
            # Pass through MLP layers with residual connections
            sampled_freq1 = F.relu(self.mlp1(sampled_freq))
            sampled_freq2 = self.mlp2(sampled_freq1)
            sampled_freq = self.mlp3(sampled_freq2 + sampled_freq)
        else:
            raise ValueError(f"Method '{self.method}' not implemented in TemporalEncoding.")
        return sampled_freq  # Shape: (1, num_frequencies)

    def forward(self, t):
        """
        Args:
            t: Time values as a tensor of shape (N,) or (N, 1), where N is the number of time stamps.
        Returns:
            temporal_embeddings: Tensor of shape (N, d_out)
        """
        if t is None:
            raise ValueError("Time input t must be provided.")
        
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=torch.float32)

        # Ensure t has shape (N, 1)
        if t.dim() == 1:
            t = t.unsqueeze(1)

        N = t.shape[0]
        device = t.device

        # Generate frequencies and expand them to match the batch size
        sampled_freq = self.generate_frequencies(device=device)  # Shape: (1, num_frequencies)
        sampled_freq = sampled_freq.expand(N, self.num_frequencies)  # Shape: (N, num_frequencies)

        # Compute sine and cosine components of the time embedding
        sin_features = torch.sin(t * sampled_freq)
        cos_features = torch.cos(t * sampled_freq)
        features = torch.cat([sin_features, cos_features], dim=-1)  # Shape: (N, 2 * num_frequencies)

        # Scale embeddings (using a similar scaling factor as in curvature encoding)
        features = features * np.sqrt(1 / self.d_out)

        # Project features to the desired output dimension
        temporal_embeddings = self.projection(features)  # Shape: (N, d_out)

        return temporal_embeddings

if __name__ == "__main__":
    # Testing the temporal encoder for our stock prediction framework.
    
    # Create an instance of the TemporalEncoding module
    encoder = TemporalEncoding(d_out=64, num_frequencies=16, method='mlp_res')
    
    # Generate a tensor of time stamps (for example, days represented as floats)
    time_stamps = torch.arange(0, 11, dtype=torch.float32)  # Shape: (11,)
    
    # Compute time embeddings
    embeddings = encoder(time_stamps)
    
    # Display the output shape and an example embedding
    print("Time embeddings shape:", embeddings.shape)  # Expected shape: (11, 64)
    print("Embedding for time 0:", embeddings[0])