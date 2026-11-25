import numpy as np
import torch
from src.graph_builder import compute_pearson_correlation_graph

def test_pearson_correlation_graph_dimensions():
    # 100 timestamps, 8 sensors
    data = np.random.normal(0, 1, (100, 8))
    adj, norm_adj, edge_index = compute_pearson_correlation_graph(data, threshold=0.3)
    
    assert adj.shape == (8, 8)
    assert norm_adj.shape == (8, 8)
    assert edge_index.shape[0] == 2
    # Diagonal must be 1.0 (self-loops)
    assert np.all(np.diag(adj) == 1.0)

def test_norm_adj_symmetric():
    data = np.random.normal(0, 1, (80, 6))
    _, norm_adj, _ = compute_pearson_correlation_graph(data, threshold=0.4)
    # Check symmetric normalization
    diff = norm_adj - norm_adj.T
    assert torch.max(torch.abs(diff)).item() < 1e-5
