# Testing & Benchmarking Plan for Anomaly Detection Thesis

Based on the structure of your thesis (`chapters/11_evaluation/main.tex`) and the components implemented in your framework, here is a comprehensive testing and benchmarking plan. This guide details what to test, how to test it, and the datasets you can use to produce rigorous academic results.

## 1. Recommended Trusted Academic Datasets

For testing an anomaly detection framework in microservices, you need datasets that contain multimodal data (metrics, logs, traces) and clear *ground truth labels* for injected faults. 

**Recommended Datasets:**
1. **Train-Ticket Benchmark Dataset**: Widely used in AIOps research. It simulates a distributed microservice ticket booking system. It contains structured logs, distributed traces (like Jaeger), and metrics, along with precisely labeled anomaly injections (CPU exhaustion, network delays, memory leaks).
2. **Alibaba Cluster Trace Data (2021/2022)**: Massive production traces from thousands of microservices. It's excellent for demonstrating the *scalability* of your federated learning approach and testing performance on real-world heterogeneity.
3. **AnoMod (2026)**: A very recent, robust multimodal dataset built on the TrainTicket and SocialNetwork benchmarks containing five modalities with explicit injected anomalies. If you can get access to this, it is highly impactful for newly published literature. 
4. **Server Machine Dataset (SMD)**: Useful if you want to focus heavily on the edge-node multivariate metric prediction aspect (LSTM-AE).

*Recommendation:* Use **Train-Ticket** for your primary evaluation, as it perfectly mimics the microservice architecture and provides clean ground truth for RCA. Use a subset of **Alibaba Cluster Trace** to explicitly test and prove your framework's scalability (Section 11.6).

---

## 2. Components to Test and Benchmark

Below are the explicit components to test, mapped directly to your Evaluation Chapter (Chapter 11) sections.

### A. Anomaly Detection Performance (Sec 11.2)
You must benchmark the core localized detection mechanism (Federated LSTM-AE).
- **What to test**: The model's ability to accurately classify time windows as anomalous or normal.
- **Metrics**: Precision, Recall, F1-score, Area Under the ROC Curve (AUC-ROC), and Area Under the Precision-Recall Curve (AUC-PR).
- **Baselines to compare against**:
  - Static Thresholding (e.g., $3\sigma$ rule)
  - Unfederated, Centralized LSTM-AE (moving all data to one server)
  - Your proposed framework (Fed-LSTM-AE + DP)

### B. Impact of Differential Privacy (Sec 11.2.3)
Since you utilize DP in the Federated framework, you must benchmark the privacy-accuracy trade-off. 
- **What to test**: How much detection accuracy drops as privacy guarantees become stricter.
- **Metrics**: F1-score tracked against varying privacy budgets ($\epsilon$).
- **Experiment**: Run federated training for 50 rounds with no DP, $\epsilon=10$, $\epsilon=1$, and $\epsilon=0.1$, and plot the resulting detection accuracy lines.

### C. Federated Learning Overhead (Sec 11.3)
You must prove that doing FL over edge nodes doesn't bottleneck network traffic compared to sending raw metrics to a central server.
- **What to test**: Model payload transmission size, compression efficiency, and latency.
- **Metrics**: 
  - Payload Size (KB) using different serialization (JSON vs Protobuf) and compression algorithms (LZ4 vs Zstd).
  - Round-Trip Time (RTT) per FL training round.
- **Experiment**: Measure the CPU overhead vs. bandwidth saved by using adaptive compression over mocked WAN vs. LAN environments.

### D. Root Cause Analysis Accuracy (Sec 11.4)
RCA is arguably the hardest part of AIOps. You need to prove your causal graph method correctly highlights the broken microservice.
- **What to test**: When an anomaly is detected, does the RCA algorithm point to the correct faulty container/service?
- **Metrics**: 
  - **Top-1 Accuracy**: % of times the actual root cause was ranked #1.
  - **Top-3 Accuracy**: % of times the actual root cause was in the top 3.
  - **Mean Reciprocal Rank (MRR)**.
- **Experiment**: Inject faults (e.g., using Chaos Mesh) onto specific pods, then check the RCA rankings. Compare your causal graph approach against a random baseline or a simple PageRank.

### E. Adaptive Thresholding / Q-Learning (Sec 11.5)
You must benchmark how the reinforcement learning reduces "alert fatigue".
- **What to test**: The reduction of false positives during "normal" system spikes (e.g., regular daily traffic peaks) that normally trigger static thresholds.
- **Metrics**: False Positive Rate (FPR), Alert Count per Hour, and Q-Learning Reward/Convergence over time.
- **Experiment**: Plot the threshold boundary over time alongside the actual metric values. Show how the Q-learning boundary adapts and "learns" periodic spikes compared to an EWMA threshold.

### F. System Scalability (Sec 11.6)
Prove the limits of the coordinator server.
- **What to test**: System resource consumption as edge nodes scale up. 
- **Metrics**: Coordinator CPU/Memory usage, and time to aggregate weights. 
- **Experiment**: Simulate 5, 20, 50, and 100 edge nodes participating in FL. Plot the Aggregation Time vs. Number of Nodes.

---

## 3. Recommended Workflow for Execution

1. **Setup the Testbed**: Deploy the framework on a local cluster (Minikube/Kind) or a small cloud cluster. Utilize a microservice demo application like the *TrainTicket* app in the same cluster.
2. **Simulate Traffic**: Use a load generator (like Locust or JMeter) to generate normal synthetic traffic.
3. **Inject Faults**: Use a chaos engineering tool like **Chaos Mesh** or **LitmusChaos** to inject controlled network delays, CPU hogs, and pod failures into specific nodes. 
4. **Collect Logs**: Output the alerts and RCA rankings from your framework, and compare them programmatically against the historical injection timestamps from Chaos Mesh.
