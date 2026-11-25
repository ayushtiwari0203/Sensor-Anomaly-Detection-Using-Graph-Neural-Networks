"""
Deep Graph Neural Network Models for Industrial Anomaly Detection
Author: Ayush Kumar Tiwari (Reg No: 12319158)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvolution(nn.Module):
    """
    Kipf & Welling Spectral Graph Convolution Layer:
    Z = D^-0.5 * A * D^-0.5 * X * W + b
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, NumNodes) or (NumNodes, InFeatures)
        support = torch.matmul(x, self.weight)
        output = torch.matmul(norm_adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output

class GraphAutoencoder(nn.Module):
    """
    Adaptive Decoupled Graph Autoencoder for Non-Euclidean Sensor Embeddings.
    Encodes spatial dependencies via Graph Convolutions and reconstructs
    expected normal operational telemetry to detect anomalies via MSE residual.
    """
    def __init__(self, num_sensors: int, hidden_dim: int = 32, latent_dim: int = 16):
        super().__init__()
        self.num_sensors = num_sensors
        
        # Spatial Graph Encoder
        self.encoder_gc1 = nn.Linear(num_sensors, hidden_dim)
        self.encoder_gc2 = nn.Linear(hidden_dim, latent_dim)
        
        # Reconstructive Decoder
        self.decoder_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.decoder_fc2 = nn.Linear(hidden_dim, num_sensors)

    def encode(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        # Spatial message passing: norm_adj @ x
        h1 = F.relu(self.encoder_gc1(torch.matmul(x, norm_adj)))
        z = F.relu(self.encoder_gc2(h1))
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h2 = F.relu(self.decoder_fc1(z))
        x_recon = self.decoder_fc2(h2)
        return x_recon

    def forward(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, norm_adj)
        x_recon = self.decode(z)
        return x_recon

    def compute_anomaly_scores(self, x: torch.Tensor, norm_adj: torch.Tensor) -> torch.Tensor:
        """Calculates per-sample Mean Squared Error reconstruction degradation."""
        self.eval()
        with torch.no_grad():
            recon = self.forward(x, norm_adj)
            # MSE per timestamp across all sensor channels
            scores = torch.mean((recon - x) ** 2, dim=1)
        return scores
