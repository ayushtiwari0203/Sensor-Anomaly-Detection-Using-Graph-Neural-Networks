"""
Graph Construction & Topology Modeling for Multi-Sensor Industrial IoT
Author: Ayush Kumar Tiwari (Reg No: 12319158)
Course: INT423 Advanced Machine Learning / SIH Theme (CA2)
"""
import numpy as np
import torch

def compute_pearson_correlation_graph(sensor_data: np.ndarray, threshold: float = 0.5):
    """
    Computes pairwise Pearson correlation across sensor channels
    and constructs an adjacency matrix with self-loops.
    
    Args:
        sensor_data: (T, N) numpy array where T is timestamps, N is sensors.
        threshold: Correlation coefficient threshold for edge creation.
        
    Returns:
        adj: (N, N) binary adjacency matrix with self-loops.
        norm_adj: (N, N) symmetrically normalized adjacency matrix (D^-0.5 * A * D^-0.5).
        edge_index: (2, E) PyG-compatible edge indices tensor.
    """
    # Calculate Pearson correlation matrix
    corr = np.corrcoef(sensor_data, rowvar=False)
    # Handle NaNs if constant channels exist
    corr = np.nan_to_num(corr, nan=0.0)
    
    num_nodes = corr.shape[0]
    # Edge exists if correlation > threshold (or absolute correlation > threshold)
    adj = (np.abs(corr) > threshold).astype(np.float32)
    # Ensure self-loops
    np.fill_diagonal(adj, 1.0)
    
    # Compute normalized Laplacian for GCN: D^-0.5 * A * D^-0.5
    deg = np.sum(adj, axis=1)
    deg_inv_sqrt = np.zeros_like(deg)
    np.power(deg, -0.5, where=deg > 0, out=deg_inv_sqrt)
    d_mat = np.diag(deg_inv_sqrt)
    norm_adj = d_mat @ adj @ d_mat
    
    # Generate PyG COO sparse edge_index
    rows, cols = np.where(adj > 0)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    
    return adj, torch.tensor(norm_adj, dtype=torch.float32), edge_index
