#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────
# ADF — End-to-End AWS Deployment Script
# ──────────────────────────────────────────────────────────────
# Usage: ./deploy.sh [--destroy]
#
# This script:
#   1. Validates prerequisites (AWS CLI, Terraform, Docker)
#   2. Builds Docker images for linux/amd64
#   3. Creates ECR repositories via Terraform
#   4. Pushes images to ECR
#   5. Deploys all infrastructure (10 VMs across 4 regions)
#   6. Waits for coordinator health check
#   7. Prints connection summary
# ──────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log()   { echo -e "${GREEN}[ADF]${NC} $*"; }
warn()  { echo -e "${YELLOW}[ADF]${NC} $*"; }
error() { echo -e "${RED}[ADF]${NC} $*" >&2; }
info()  { echo -e "${CYAN}[ADF]${NC} $*"; }

# ── Destroy mode ─────────────────────────────────────────────

if [[ "${1:-}" == "--destroy" ]]; then
    log "Destroying all ADF infrastructure..."
    cd "$SCRIPT_DIR"
    terraform destroy -auto-approve
    log "All resources destroyed."
    exit 0
fi

# ── Prerequisites Check ─────────────────────────────────────

log "Checking prerequisites..."

for cmd in aws terraform docker; do
    if ! command -v "$cmd" &>/dev/null; then
        error "$cmd is not installed. Please install it first."
        exit 1
    fi
done

# Verify AWS credentials
if ! aws sts get-caller-identity &>/dev/null; then
    error "AWS credentials not configured. Run 'aws configure' first."
    exit 1
fi

AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION="${AWS_REGION:-eu-central-1}"
ECR_REGISTRY="${AWS_ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com"
PREFIX="adf"

log "AWS Account: $AWS_ACCOUNT"
log "Primary Region: $AWS_REGION"
log "ECR Registry: $ECR_REGISTRY"

# ── Step 1: Initialize Terraform ─────────────────────────────

log "Step 1/6: Initializing Terraform..."
cd "$SCRIPT_DIR"
terraform init -input=false

# ── Step 2: Create ECR Repositories ──────────────────────────

log "Step 2/6: Creating ECR repositories..."
terraform apply -auto-approve \
    -target=aws_ecr_repository.images \
    -input=false

# ── Step 3: Build Docker Images ──────────────────────────────

log "Step 3/6: Building Docker images (linux/amd64)..."

docker build \
    --platform linux/amd64 \
    -t "${PREFIX}-coordinator:latest" \
    -f "${PROJECT_ROOT}/deploy/docker/Dockerfile.coordinator" \
    "$PROJECT_ROOT"

docker build \
    --platform linux/amd64 \
    -t "${PREFIX}-edge:latest" \
    -f "${PROJECT_ROOT}/deploy/docker/Dockerfile.edge" \
    "$PROJECT_ROOT"

docker build \
    --platform linux/amd64 \
    -t "${PREFIX}-edge-lightweight:latest" \
    -f "${PROJECT_ROOT}/deploy/docker/Dockerfile.edge-lightweight" \
    "$PROJECT_ROOT"

log "Docker images built successfully"

# ── Step 4: Push to ECR ──────────────────────────────────────

log "Step 4/6: Pushing images to ECR..."

aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "$ECR_REGISTRY"

for image in coordinator edge edge-lightweight; do
    docker tag "${PREFIX}-${image}:latest" \
        "${ECR_REGISTRY}/${PREFIX}/${PREFIX}-${image}:latest"
    docker push "${ECR_REGISTRY}/${PREFIX}/${PREFIX}-${image}:latest"
    log "  Pushed ${PREFIX}-${image}"
done

# ── Step 5: Deploy Infrastructure ────────────────────────────

log "Step 5/6: Deploying infrastructure across 4 regions..."
terraform apply -auto-approve -input=false

COORDINATOR_IP=$(terraform output -raw coordinator_public_ip)
log "Coordinator deployed at: $COORDINATOR_IP"

# ── Step 6: Wait for Health Check ────────────────────────────

log "Step 6/6: Waiting for coordinator to become healthy..."

MAX_WAIT=120
WAITED=0
HEALTH_URL="http://${COORDINATOR_IP}:8080/health"

while [ $WAITED -lt $MAX_WAIT ]; do
    if curl -sf --connect-timeout 3 "$HEALTH_URL" &>/dev/null; then
        log "Coordinator is healthy!"
        break
    fi
    sleep 10
    WAITED=$((WAITED + 10))
    info "  Waiting... (${WAITED}s / ${MAX_WAIT}s)"
done

if [ $WAITED -ge $MAX_WAIT ]; then
    warn "Coordinator did not respond within ${MAX_WAIT}s."
    warn "It may still be bootstrapping. Check with: make logs"
fi

# ── Summary ──────────────────────────────────────────────────

echo ""
echo "========================================================"
echo "  ADF Multi-Region Deployment Complete"
echo "========================================================"
echo ""
info "Coordinator:"
info "  IP:      $COORDINATOR_IP"
info "  API:     http://${COORDINATOR_IP}:8080"
info "  Health:  http://${COORDINATOR_IP}:8080/health"
info "  gRPC:    ${COORDINATOR_IP}:50051"
echo ""
info "SSH Access:"
info "  ssh -i deploy/aws/adf-key.pem ec2-user@${COORDINATOR_IP}"
echo ""
info "Edge Nodes:"
terraform output -json deployment_summary 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
for region, nodes in data.get('edge_nodes', {}).items():
    std_ips = nodes.get('standard', [])
    lwt_ips = nodes.get('lightweight', [])
    if std_ips or lwt_ips:
        print(f'  {region}:')
        for ip in std_ips:
            print(f'    [standard]    {ip}')
        for ip in lwt_ips:
            print(f'    [lightweight] {ip}')
" 2>/dev/null || echo "  (run 'terraform output deployment_summary' to see all IPs)"
echo ""
info "Commands:"
info "  make status     — check all instances"
info "  make logs       — tail coordinator logs"
info "  make destroy    — tear down everything"
echo ""
echo "========================================================"
