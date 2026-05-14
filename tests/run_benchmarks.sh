#!/bin/bash

cleanup() {
    echo "Cleaning up..."
    kubectl delete -f manifests/chaos-experiments/network-delay.yaml 2>/dev/null || true
    if [ -n "$LOCUST_PID" ]; then
        kill $LOCUST_PID 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting Thesis Benchmarking Pipeline..."

# 1. Ensure the service is exposed robustly for load testing
echo "Configuring NodePort for heavy traffic..."
kubectl patch svc ticket-service -p '{"spec": {"type": "NodePort"}}'
sleep 2 # Give Kubernetes a moment to apply the patch

# 2. Get the actual direct IP and port from Minikube
TARGET_URL=$(minikube service ticket-service --url | head -n 1)
echo "Routing traffic directly to Minikube at: $TARGET_URL"

# 3. Start Locust Traffic Generation in background
echo "Starting background traffic generation..."
mkdir -p logs
locust -f utils/traffic_simulator.py --headless -u 50 -r 10 --host $TARGET_URL > logs/locust.log 2>&1 &
LOCUST_PID=$!

# 4. Wait for normal baseline data collection
echo "Collecting baseline metrics for 2 minutes..."
sleep 120

# 5. Inject Chaos (Network Delay)
echo "Injecting Network Delay Chaos..."
kubectl apply -f manifests/chaos-experiments/network-delay.yaml

# 6. Run Benchmarks
echo "Running Python Benchmark Suite..."
export PYTHONPATH=$(pwd)/..:$(pwd)/../src
export CUDA_VISIBLE_DEVICES=''
python3 benchmarks/test_anomaly_detection.py
python3 benchmarks/test_dp_impact.py
python3 benchmarks/test_fl_overhead.py
python3 benchmarks/test_rca_accuracy.py
python3 benchmarks/test_adaptive_threshold.py
python3 benchmarks/test_scalability.py

echo "Benchmarking complete. All data saved to tests/logs/"