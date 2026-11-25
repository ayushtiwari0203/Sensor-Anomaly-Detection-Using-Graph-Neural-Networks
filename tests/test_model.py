import torch
from src.models import GraphAutoencoder

def test_model_forward_and_shapes():
    num_sensors = 10
    batch_size = 32
    hidden_dim = 16
    latent_dim = 8
    
    model = GraphAutoencoder(num_sensors=num_sensors, hidden_dim=hidden_dim, latent_dim=latent_dim)
    
    x = torch.randn(batch_size, num_sensors)
    norm_adj = torch.eye(num_sensors)
    
    # Forward pass
    recon = model(x, norm_adj)
    assert recon.shape == (batch_size, num_sensors)
    
    # Latent encoding shape
    z = model.encode(x, norm_adj)
    assert z.shape == (batch_size, latent_dim)

def test_compute_anomaly_scores():
    num_sensors = 6
    batch_size = 20
    model = GraphAutoencoder(num_sensors=num_sensors)
    x = torch.randn(batch_size, num_sensors)
    norm_adj = torch.eye(num_sensors)
    
    scores = model.compute_anomaly_scores(x, norm_adj)
    assert scores.shape == (batch_size,)
    assert torch.all(scores >= 0)
