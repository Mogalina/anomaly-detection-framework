#!/bin/bash
set -euo pipefail

# ──────────────────────────────────────────────────────────────
# ADF Edge Node — EC2 User-Data Bootstrap
# ──────────────────────────────────────────────────────────────

exec > /var/log/adf-bootstrap.log 2>&1
echo "=== ADF Edge Node bootstrap started at $(date -u) ==="

# ── Install Docker ───────────────────────────────────────────
dnf update -y -q
dnf install -y docker aws-cli
systemctl enable docker
systemctl start docker
usermod -aG docker ec2-user

# ── Authenticate to ECR (cross-region pull from primary) ─────
aws ecr get-login-password --region ${ecr_region} \
  | docker login --username AWS --password-stdin ${ecr_registry}

# ── Pull edge image (with retry) ─────────────────────────────
MAX_RETRIES=5
for i in $(seq 1 $MAX_RETRIES); do
  echo "Pull attempt $i/$MAX_RETRIES..."
  if docker pull ${ecr_edge_url}:latest; then
    echo "Pull succeeded"
    break
  fi
  if [ $i -eq $MAX_RETRIES ]; then
    echo "ERROR: Failed to pull image after $MAX_RETRIES attempts"
    exit 1
  fi
  echo "Pull failed, retrying in 30s..."
  sleep 30
  aws ecr get-login-password --region ${ecr_region} \
    | docker login --username AWS --password-stdin ${ecr_registry}
done

# ── Run edge container ──────────────────────────────────────
docker run -d \
  --name adf-edge \
  --restart unless-stopped \
  -p 9090:9090 \
  -e CLIENT_ID=${client_id} \
  -e COORDINATOR_HOST=${coordinator_host} \
  -e COORDINATOR_PORT=50051 \
  -e NODE_PROFILE=${node_profile} \
  -e PYTHONPATH=/app/src \
  -v /opt/adf/logs:/app/logs \
  ${ecr_edge_url}:latest

echo "=== ADF Edge Node (${client_id}, ${node_profile}) bootstrap completed at $(date -u) ==="
# Force update 1
