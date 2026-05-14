# Federated Anomaly Detection: Demo Guide

This guide explains how to start, interact with, and interpret the live Grafana & Prometheus demonstration built for your bachelor's thesis defense. 

## 1. Overview
The demo provides a highly authentic, real-time visualization of your entire framework exactly as it would run in a production SRE (Site Reliability Engineering) environment. 

It consists of three components:
1. **The Exporter (`demo/prometheus_exporter.py`)**: A Python engine that actively loads your pre-trained `lstm_best.pt` model, streams the SMD test dataset through it, and exposes real-time telemetry, Federated Learning states, Differential Privacy bounds, and simulated RCA traces.
2. **Prometheus**: A time-series database running in Docker that scrapes the Python exporter every second.
3. **Grafana**: The industry-standard visualization layer running in Docker, pre-provisioned with 4 massive dashboards.

---

## 2. How to Start the Demo

To launch the demo, open a terminal in the root of your project and run these exact commands:

```bash
# 1. Start the Grafana and Prometheus containers
cd demo
docker-compose up -d

# 2. Activate your python virtual environment
source ../venv/bin/activate

# 3. Start the Live Exporter
python3 prometheus_exporter.py
```

*Note: The exporter will print `Prometheus Exporter running on http://localhost:8000/metrics`. Leave this terminal running in the background during your defense.*

To stop the demo when you are done:
```bash
# Press Ctrl+C in the exporter terminal to kill the python script, then:
docker-compose down
```

---

## 3. How to Use and Read Grafana

Once the exporter is running, open your web browser and go to:  
**👉 `http://localhost:3000`**

*(If asked for a login, the default is username: `admin`, password: `admin`)*.

To find your dashboards, click on the **Dashboards** icon (four squares) in the left-hand navigation menu, then click **Browse**. You will see 4 distinct dashboards. 

### Dashboard 1: Thesis Command Center
**What it is:** The executive overview of your entire system.
**How to read it:** 
- The top row shows your KPIs: Current system status (Normal vs Anomaly), active FL round, and your Differential Privacy guarantee ($\varepsilon$). 
- The central graph tracks the live LSTM-AE reconstruction error against your dynamically computed 95th-percentile threshold.
- The bottom heatmap gives you a massive, at-a-glance view of latency across all microservices.

### Dashboard 2: Federated Learning & Privacy
**What it is:** A deep dive into your DP-FedAvg architecture.
**How to read it:**
- Watch the **Global Loss Convergence** graph simulate the model training over rounds.
- Look at the **Communication Payload Size** bar chart to prove to the examiners how much bandwidth is saved by compressing TorchSave binaries vs JSON. 
- Look at the top stats to point out your specific DP bounds ($C$ and $\sigma$).

### Dashboard 3: Root Cause Analysis Engine (Deep Dive)
**What it is:** The incident response monitor.
**How to read it:**
- Wait for an anomaly to trigger in the Command Center, then quickly switch to this dashboard.
- The **Active Incident Trace Paths** table will populate with the exact Incident ID, the Root Cause Service, the Fault Type (e.g., Latency Spike), and the entire downstream dependency trace (e.g., `ts-order-service -> ts-payment-service -> ts-security-service`).
- The **Probability Gauges** will instantly isolate the highest-scoring culprit service based on your framework's RCA engine.

### Dashboard 4: Microservice Deep Drilldown
**What it is:** Independent node-level telemetry (RED/USE metrics).
**How to read it:**
- This dashboard proves your framework can monitor deep, complex topologies. It tracks CPU, Memory, Latency, Network RX/TX, and Connections.
- **Crucial Interaction:** At the very top-left of this dashboard, there is a dropdown menu labeled `service`. Click it to select specific microservices (e.g., `ts-route-service`). The entire dashboard will instantly filter to show *only* that service's metrics. You can select one, or multiple to compare them side-by-side!

---

## 4. Tips for your Defense Presentation

1. **Start on the Command Center:** Keep the screen on Dashboard 1 while you introduce the architecture. It looks highly active and impressive.
2. **Wait for an Anomaly:** When the "System Status" flashes RED, switch to the **Root Cause Analysis** dashboard and show how the system immediately traced the fault down the dependency graph.
3. **Show the Drilldown:** Go to Dashboard 4, use the `$service` dropdown, and isolate the exact service that caused the anomaly to prove that its CPU or Latency spiked exactly when the LSTM-AE triggered.
