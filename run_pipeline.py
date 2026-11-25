"""
Sensor Anomaly Detection Runner
Author: Ayush Kumar Tiwari (Reg No: 12319158)
"""
import sys
from pathlib import Path
import json

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.pipeline import run_anomaly_detection_pipeline

if __name__ == "__main__":
    print("=================================================================")
    print("  Sensor Anomaly Detection using Graph Neural Networks (GNN)")
    print("  Author: Ayush Kumar Tiwari (12319158) | INT423 CA2 / SIH Theme")
    print("=================================================================\n")
    
    print("[1/3] Loading sensor telemetry & building Pearson correlation graph...")
    result = run_anomaly_detection_pipeline(
        correlation_threshold=0.45,
        hidden_dim=32,
        latent_dim=16,
        epochs=30,
        lr=0.005,
        percentile_threshold=95.0
    )
    
    print("\n[2/3] Model Architecture & Graph Properties:")
    print(f"  • Monitored Sensor Channels: {result['num_sensors']}")
    print(f"  • Telemetry Timestamps:     {result['sample_count']}")
    print(f"  • Active Correlation Edges: {result['graph_edges']}")
    print(f"  • Final Reconstruction Loss: {result['final_train_loss']:.6f}")
    
    print("\n[3/3] Industrial Anomaly Diagnostic Metrics:")
    metrics = result["metrics"]
    print(f"  • Accuracy:        {metrics['accuracy'] * 100:.2f}%")
    print(f"  • Precision:       {metrics['precision'] * 100:.2f}%")
    print(f"  • Recall:          {metrics['recall'] * 100:.2f}%")
    print(f"  • F1-Score:        {metrics['f1_score'] * 100:.2f}%")
    print(f"  • Anomaly Cutoff:  {result['anomaly_threshold']:.6f}")
    print(f"  • Detected Faults: {result['detected_anomalies']} / {result['ground_truth_anomalies']} ground-truth")
    print("\nConfusion Matrix [ [TN, FP], [FN, TP] ]:")
    print(f"  {metrics['confusion_matrix']}")
    print("\n=================================================================")
    print("  Diagnostic Run Completed Successfully!")
    print("=================================================================")
