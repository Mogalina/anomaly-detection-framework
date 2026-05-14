# Federated Anomaly Detection Framework

A privacy-preserving, federated anomaly detection framework for distributed microservice architectures. 

> **Full Thesis Manuscript**: The complete, final written thesis manuscript (PDF and LaTeX source code) is available in the [`./docs/thesis`](./docs/thesis) directory.

---

## Definition
This framework is an end-to-end Site Reliability Engineering (SRE) tool designed to monitor distributed microservices. It utilizes local deep learning at the edge to detect anomalies in real-time, collaborates globally using Federated Learning to preserve data privacy, and automatically traces the origin of cascading failures using Causal Graph theory.

## Abstract
Modern distributed systems, composed of microservices and heterogeneous components, are increasingly complex and prone to performance anomalies that can compromise reliability and service quality. This framework integrates real-time edge detection via an LSTM-Autoencoder, federated learning for collaborative model training without centralizing raw telemetry, causal graph-based root cause analysis, and adaptive Q-learning thresholding. By combining localized anomaly detection at the edge with global model aggregation via the FedAvg algorithm, the framework enhances detection accuracy while preserving data privacy through Differential Privacy (DP-SGD).

## Motivation
Traditional centralized monitoring systems struggle with two major issues in modern cloud environments:
1. **Bandwidth & Latency**: Sending raw telemetry from thousands of edge nodes to a central server creates immense network bottlenecks.
2. **Data Privacy**: Centralizing sensitive system metrics exposes infrastructure to severe security and privacy risks.

This framework solves these by pushing the intelligence to the edge. Nodes train models locally and only share encrypted model weights (gradients), severely reducing bandwidth overhead and maintaining strict data privacy.

## State of the Art
The framework builds upon and integrates several cutting-edge research domains:
- **Time-Series Forecasting**: Long Short-Term Memory Autoencoders (LSTM-AE).
- **Privacy & Security**: Federated Averaging (FedAvg) combined with Differential Privacy (DP-SGD) gradient clipping and noise injection.
- **AIOps (AI for IT Operations)**: Dynamic thresholding using Reinforcement Learning (Q-learning) to replace static alerting rules.
- **Tracing**: Causal Graph PageRank algorithms for Root Cause Analysis (RCA).

## Results and Discoveries
- **Detection Fidelity**: The centralized LSTM-AE baseline achieved strong detection performance on multi-dimensional telemetry with an AUC-ROC of $0.932 \pm 0.017$.
- **Privacy-Utility Tradeoff**: Incorporating Differential Privacy (DP) exposed a sharp, critical phase transition. While the model tolerates moderate clipping, noise multipliers above $\sigma=0.001$ caused total performance collapse (F1 dropped to 0.000), proving that extreme privacy guarantees destroy the structural embeddings required for anomaly reconstruction.
- **Bandwidth Reduction**: Federated compression mechanisms (like Zstd over TorchSave) reduced network payloads by over 80% compared to raw JSON transmissions.
- **Automated RCA**: The PageRank causal analyzer successfully localized simulated failure injections on 11-node microservice topologies, providing highly interpretable trace paths.

## Technology Stack
* **Deep Learning & FL**: `PyTorch`, `NumPy`, `Scikit-Learn`
* **Graph Algorithms**: `NetworkX`
* **Telemetry & Simulation**: `Prometheus`, `Grafana`, `psutil`
* **Deployment & Infrastructure**: `Docker`, `Docker Compose`, `Kubernetes`, `Terraform` (AWS)
* **Language**: `Python 3.9+`

## Datasets Used
1. **Server Machine Dataset (SMD)**: A comprehensive 5-week dataset from a large internet company containing 38-dimensional system telemetry (CPU, Memory, Network). Used to train and benchmark the LSTM-AE detection fidelity.
2. **Train-Ticket**: An open-source benchmark microservice application. Its topology and dependency maps were used to test and validate the Causal Graph Root Cause Analysis engine.
3. **Alibaba Cluster Traces**: Used as a reference for cloud-scale workload distributions and scalability testing.

## Conclusion
The framework proves that it is viable to decentralize anomaly detection in massive microservice architectures. While Federated Learning successfully eliminates the need for raw telemetry centralization, engineers must strictly tune Differential Privacy hyperparameters to prevent the catastrophic degradation of detection utility. 

---

### Running the Live Demonstration
If you want to see the framework in action, we have provided a fully self-contained Dockerized demo containing a live Prometheus exporter and a massive 4-dashboard Grafana SRE suite. 

Read the [Demo Guide](./demo_guide.md) or launch it directly:
```bash
cd demo
docker-compose up -d
source ../venv/bin/activate
python3 prometheus_exporter.py
```
*(Grafana will be available at `http://localhost:3000`)*
