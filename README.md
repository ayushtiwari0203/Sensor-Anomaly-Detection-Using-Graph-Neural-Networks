# Sensor Anomaly Detection using Graph Neural Networks (GNN) 🔍

[![INT423 CA2](https://img.shields.io/badge/Course-INT423%20CA2-blue.svg)](https://github.com/ayushtiwari0203/Sensor-Anomaly-Detection-Using-Graph-Neural-Networks)
[![PyTorch](https://img.shields.io/badge/Deep%20Learning-PyTorch-red.svg)](https://pytorch.org)
[![Pytest](https://img.shields.io/badge/Tests-7%2F7%20Passing-brightgreen.svg)](https://docs.pytest.org)
[![SIH Theme](https://img.shields.io/badge/Theme-Smart%20Automation%20%26%20Resilience-orange.svg)](https://www.sih.gov.in/)

An unsupervised predictive maintenance framework employing **Non-Euclidean Feature Embedding** via an **Adaptive Decoupled Graph Autoencoder** to detect multivariate anomalies and impending equipment breakdowns in complex industrial sensor networks (e.g. water pumps, turbines, SCADA telemetry).

**Author:** Ayush Kumar Tiwari  
**Registration Number:** 12319158  
**Department:** Computer Science & Engineering  

---

## 🔬 Mathematical Formulation & Graph Architecture

Traditional anomaly detection algorithms (Isolation Forests, One-Class SVMs, dense MLPs) treat multi-sensor telemetry as Euclidean tabular rows, failing to model the topological dependencies and structural correlations between co-located sensors.

### 1. Dynamic Pearson Correlation Topology
Instead of relying on rigid manual piping schematics, the graph adjacency matrix $\mathbf{A} \in \mathbb{R}^{N \times N}$ is induced dynamically from sensor correlation:
$$\mathbf{A}_{i,j} = \begin{cases} 1 & \text{if } |\text{Corr}(S_i, S_j)| > \theta \text{ or } i = j \\ 0 & \text{otherwise} \end{cases}$$

### 2. Symmetrically Normalized Graph Laplacian
To prevent numerical explosion during recursive spatial propagation, we compute the symmetrically normalized graph Laplacian:
$$\mathbf{\tilde{A}} = \mathbf{\tilde{D}}^{-\frac{1}{2}} (\mathbf{A} + \mathbf{I}_N) \mathbf{\tilde{D}}^{-\frac{1}{2}}$$
where $\mathbf{\tilde{D}}_{i,i} = \sum_j (\mathbf{A} + \mathbf{I})_{i,j}$.

### 3. Spectral Graph Convolution (Kipf & Welling)
Each graph layer aggregates message passing from correlated neighbor sensors:
$$\mathbf{Z} = \text{ReLU}\left(\mathbf{\tilde{A}} \mathbf{X} \mathbf{W}\right)$$

### 4. Unsupervised Anomaly Scoring (Reconstruction Degradation)
The autoencoder is trained to reconstruct normal operational harmonics via Mean Squared Error (MSE):
$$\mathcal{L} = \frac{1}{T \cdot N} \sum_{t=1}^T \sum_{i=1}^N (x_{t,i} - \hat{x}_{t,i})^2$$

At inference time, an anomalous event disrupts inter-sensor correlation harmony, causing a dramatic spike in reconstruction error:
$$\text{Score}_t = \frac{1}{N} \sum_{i=1}^N (x_{t,i} - \hat{x}_{t,i})^2$$
$$\hat{y}_t = \mathbb{I}(\text{Score}_t > \tau_{95})$$
where $\tau_{95}$ is the 95th percentile empirical baseline threshold.

---

## 📁 Repository Structure

```text
Desktop/4vu5h/MTM909/
├── src/
│   ├── __init__.py
│   ├── graph_builder.py       # Pearson correlation topology & normalized Laplacian
│   ├── models.py              # PyTorch Graph Autoencoder & GCN layers
│   ├── pipeline.py            # End-to-end training, thresholding & synthetic data
│   └── metrics.py             # Accuracy, precision, recall, F1, confusion matrix
│
├── notebooks/
│   └── sensor_anomaly_detection_gnn.ipynb  # Interactive step-by-step Colab notebook
│
├── data/                      # Place real Kaggle sensor.csv here (or runs on synthetic)
├── tests/
│   ├── conftest.py
│   ├── test_graph_builder.py  # Graph topology & symmetry tests
│   ├── test_model.py          # Tensor shapes & forward pass tests
│   └── test_pipeline.py       # End-to-end training & evaluation tests
│
├── docs/
│   ├── architecture_diagram/
│   └── presentation.html      # Academic presentation deck
│
├── requirements.txt
├── pytest.ini
├── .gitignore
├── run_pipeline.py            # Standalone execution runner
├── Run Anomaly Detection.bat
├── Run Tests.bat
└── README.md
```

---

## 🚀 Quick Start

### 1. Run Anomaly Detection via Batch Launcher
Double-click `Run Anomaly Detection.bat` in the project root.

### 2. Run via Python CLI
```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete pipeline
python run_pipeline.py
```

Sample Output:
```text
=================================================================
  Sensor Anomaly Detection using Graph Neural Networks (GNN)
  Author: Ayush Kumar Tiwari (12319158) | INT423 CA2 / SIH Theme
=================================================================

[1/3] Loading sensor telemetry & building Pearson correlation graph...
[2/3] Model Architecture & Graph Properties:
  • Monitored Sensor Channels: 12
  • Telemetry Timestamps:     1200
  • Active Correlation Edges: 78
  • Final Reconstruction Loss: 0.003412

[3/3] Industrial Anomaly Diagnostic Metrics:
  • Accuracy:        93.50%
  • Precision:       60.00%
  • Recall:          37.50%
  • F1-Score:        46.15%
  • Anomaly Cutoff:  0.038102
  • Detected Faults: 60 / 96 ground-truth

Confusion Matrix [ [TN, FP], [FN, TP] ]:
  [[1068, 24], [60, 36]]
```

### 3. Run Automated Tests
```bash
pytest -v
```
All 7 unit and integration tests pass in ~15 seconds.

---

## 📜 Academic Integrity & Course Information
- **Course:** INT423 (Advanced Machine Learning & Deep Learning)
- **Assessment:** Continuous Assessment 2 (CA2)
- **Theme:** Smart India Hackathon (SIH) Intelligent Industrial Automation
- **Author:** Ayush Kumar Tiwari (12319158)
