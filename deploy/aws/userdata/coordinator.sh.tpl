#!/bin/bash
set -euo pipefail

exec > /var/log/adf-bootstrap.log 2>&1
echo "Anomaly Detection Framework: Coordinator bootstrap started at $(date -u)"

# ─── Install Docker ───
dnf update -y -q
dnf install -y docker aws-cli
systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user

# ─── Authenticate to ECR ───
aws ecr get-login-password --region ${aws_region} \
  | docker login --username AWS --password-stdin ${ecr_registry}

# ─── Pull coordinator image (with retry) ───
MAX_RETRIES=5
for i in $(seq 1 $MAX_RETRIES); do
  echo "Pull attempt $i/$MAX_RETRIES..."
  if docker pull ${ecr_coordinator_url}:latest; then
    echo "Pull succeeded"
    break
  fi
  if [ $i -eq $MAX_RETRIES ]; then
    echo "ERROR: Failed to pull image after $MAX_RETRIES attempts"
    exit 1
  fi
  echo "Pull failed, retrying in 30s..."
  sleep 30
  # Re-authenticate in case token expired
  aws ecr get-login-password --region ${aws_region} \
    | docker login --username AWS --password-stdin ${ecr_registry}
done

# ─── Run Redis and Coordinator containers ───
docker network create adf-net || true

docker run -d \
  --name redis_cache \
  --network adf-net \
  --restart unless-stopped \
  redis:alpine

docker run -d \
  --name adf-coordinator \
  --network adf-net \
  --restart unless-stopped \
  -p 50051:50051 \
  -p 8080:8080 \
  -p 9090:9090 \
  -e COORDINATOR_HOST=0.0.0.0 \
  -e COORDINATOR_PORT=50051 \
  -e PYTHONPATH=/app/src \
  -v /opt/adf/checkpoints:/app/checkpoints \
  -v /opt/adf/logs:/app/logs \
  ${ecr_coordinator_url}:latest

echo "Anomaly Detection Framework: Coordinator bootstrap completed at $(date -u)"
