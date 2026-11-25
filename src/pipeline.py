"""
End-to-End Pipeline for Graph Neural Network Anomaly Detection
Author: Ayush Kumar Tiwari (Reg No: 12319158)
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any

from .graph_builder import compute_pearson_correlation_graph
from .models import GraphAutoencoder
from .metrics import calculate_anomaly_metrics

def generate_synthetic_sensor_benchmark(n_samples: int = 1200, n_sensors: int = 12, anomaly_ratio: float = 0.08) -> pd.DataFrame:
    """
    Generates realistic synthetic multi-channel industrial pump telemetry:
    - Normal operation: coupled sinusoids + white noise representing rotational harmony.
    - Anomalies: simulated bearing friction, thermal runaways, pressure shocks.
    """
    np.random.seed(42)
    t = np.linspace(0, 50, n_samples)
    data_dict = {}
    
    # Base rotational harmonics
    base_signal = np.sin(t)
    for i in range(n_sensors):
        phase = (i * np.pi) / n_sensors
        noise = np.random.normal(0, 0.1, n_samples)
        # Correlated channels
        data_dict[f"sensor_{i:02d}"] = np.sin(t + phase) * (1.0 + 0.1 * i) + noise

    df = pd.DataFrame(data_dict)
    labels = np.zeros(n_samples, dtype=int)
    
    # Inject multi-channel anomalous events
    n_anomalies = int(n_samples * anomaly_ratio)
    anomaly_indices = np.random.choice(n_samples, size=n_anomalies, replace=False)
    
    for idx in anomaly_indices:
        labels[idx] = 1
        # Spike in vibration and temperature sensors
        df.iloc[idx, 0] += np.random.uniform(2.5, 4.0)
        df.iloc[idx, 1] += np.random.uniform(-3.0, -1.5)
        df.iloc[idx, 2] += np.random.uniform(1.8, 3.2)

    df["label"] = labels
    return df

def run_anomaly_detection_pipeline(
    data_path: str = None,
    correlation_threshold: float = 0.45,
    hidden_dim: int = 32,
    latent_dim: int = 16,
    epochs: int = 25,
    lr: float = 0.005,
    percentile_threshold: float = 95.0
) -> Dict[str, Any]:
    """Executes the full GNN data loading, graph building, training and evaluation pipeline."""
    # 1. Load or Generate Dataset
    if data_path and os.path.exists(data_path):
        df = pd.read_csv(data_path)
        sensor_cols = [c for c in df.columns if "sensor" in c.lower()]
        if "machine_status" in df.columns:
            labels = df["machine_status"].apply(lambda x: 1 if str(x).upper() in ["BROKEN", "RECOVERING"] else 0).values
        elif "label" in df.columns:
            labels = df["label"].values
        else:
            labels = np.zeros(len(df))
    else:
        df = generate_synthetic_sensor_benchmark()
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        labels = df["label"].values

    # 2. Normalization
    scaler = MinMaxScaler()
    normalized_values = scaler.fit_transform(df[sensor_cols].fillna(0).values)
    
    # 3. Construct Graph Topology
    adj, norm_adj, edge_index = compute_pearson_correlation_graph(normalized_values, threshold=correlation_threshold)
    
    X_tensor = torch.tensor(normalized_values, dtype=torch.float32)
    num_sensors = len(sensor_cols)
    
    # 4. Instantiate GNN Model
    model = GraphAutoencoder(num_sensors=num_sensors, hidden_dim=hidden_dim, latent_dim=latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    # 5. Training Loop
    model.train()
    training_losses = []
    for ep in range(epochs):
        optimizer.zero_grad()
        recon = model(X_tensor, norm_adj)
        loss = loss_fn(recon, X_tensor)
        loss.backward()
        optimizer.step()
        training_losses.append(round(float(loss.item()), 6))
        
    # 6. Anomaly Scoring & Thresholding
    scores = model.compute_anomaly_scores(X_tensor, norm_adj).numpy()
    threshold = np.percentile(scores, percentile_threshold)
    preds = (scores > threshold).astype(int)
    
    # 7. Evaluation Metrics
    metrics = calculate_anomaly_metrics(labels, preds)
    
    return {
        "num_sensors": num_sensors,
        "sample_count": len(df),
        "graph_edges": edge_index.shape[1],
        "final_train_loss": training_losses[-1],
        "anomaly_threshold": round(float(threshold), 6),
        "detected_anomalies": int(np.sum(preds)),
        "ground_truth_anomalies": int(np.sum(labels)),
        "metrics": metrics
    }
