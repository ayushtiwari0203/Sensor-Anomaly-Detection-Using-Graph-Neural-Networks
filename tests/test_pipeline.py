from src.pipeline import generate_synthetic_sensor_benchmark, run_anomaly_detection_pipeline
from src.metrics import calculate_anomaly_metrics
import numpy as np

def test_synthetic_data_generation():
    df = generate_synthetic_sensor_benchmark(n_samples=500, n_sensors=8, anomaly_ratio=0.1)
    assert len(df) == 500
    assert "label" in df.columns
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    assert len(sensor_cols) == 8
    # Verify presence of anomalies
    assert df["label"].sum() == 50

def test_end_to_end_pipeline():
    result = run_anomaly_detection_pipeline(
        data_path=None,  # Use synthetic benchmark
        correlation_threshold=0.4,
        epochs=10,
        percentile_threshold=95.0
    )
    assert result["num_sensors"] == 12
    assert result["sample_count"] == 1200
    assert result["final_train_loss"] < 0.2
    assert "metrics" in result
    assert result["metrics"]["accuracy"] > 0.8
    assert result["detected_anomalies"] > 0

def test_metrics_calculation():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1])
    m = calculate_anomaly_metrics(y_true, y_pred)
    assert m["accuracy"] > 0.8
    assert m["f1_score"] > 0.7
