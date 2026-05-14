# AWS Multi-Region Deployment Guide

This document outlines the step-by-step procedures for deploying the anomaly detection framework across four distinct AWS regions utilizing ten virtual machines.

## Prerequisites

The following utilities are required on the host system:

```bash
aws --version
terraform --version
docker --version
```

### Installation of Dependencies:

```bash
brew install awscli
brew install hashicorp/tap/terraform
brew install --cask docker
```

## Step 1: AWS Credential Configuration

```bash
aws configure
```

The configuration tool will prompt for four parameters:

| Parameter | Retrieval Location |
|--------|-----------------|
| **Access Key ID** | AWS Console → IAM → Users → Security Credentials → Create Access Key |
| **Secret Access Key** | Provided upon access key generation |
| **Default region** | Input `eu-central-1` |
| **Output format** | Input `json` |

Verify the configuration:
```bash
aws sts get-caller-identity
```
The output must display the active account ID. A failure indicates invalid credentials.

### IAM Permission Requirements

The executing IAM user must be assigned the following managed policies:

- `AmazonEC2FullAccess`
- `AmazonEC2ContainerRegistryFullAccess`
- `IAMFullAccess` (required for EC2 instance profile creation)

> **Note**: For initial validation, `AdministratorAccess` may be utilized, followed by subsequent restriction according to the principle of least privilege.

## Step 2: Docker Daemon Verification

```bash
docker info
```

If a connection error is returned, ensure the Docker daemon is active on the host machine.

## Step 3: Deployment Execution

```bash
cd /Users/mogalina/Desktop/anomaly-detection-framework/deploy/aws
make all
```

This command initiates the automated deployment pipeline:
1. **`make build`** — Compiles the three required Docker images (coordinator, edge, edge-lightweight) for the `linux/amd64` architecture.
2. **`make push`** — Initializes ECR repositories and uploads the compiled images.
3. **`make deploy`** — Executes Terraform configurations to provision the virtual machines across the specified regions.
4. **`make status`** — Outputs the final network configuration and coordinator health status.

> **Note**: The initial build phase may require significant time due to dependency resolution (e.g., PyTorch). Subsequent executions will utilize cached layers.

### Alternative: Procedural Deployment Script

For granular execution and logging:

```bash
cd /Users/mogalina/Desktop/anomaly-detection-framework/deploy/aws
chmod +x deploy.sh
./deploy.sh
```

This script mirrors the `make` pipeline but includes health-check polling and an explicit connection summary upon completion.

## Step 4: Deployment Verification

```bash
# Check all instance IPs
make status

# Check coordinator health
curl http://<coordinator-ip>:8080/health

# Tail coordinator logs (watch edge nodes register)
make logs

# SSH into the coordinator
make ssh-coordinator

# SSH into a specific edge node
make ssh-edge REGION=eu-west-1 N=1
make ssh-edge REGION=us-east-1 N=2 PROFILE=lightweight
```

### Observation Parameters

Within the execution logs, verify the following:
- Edge nodes successfully registering from all four distinct regions.
- The commencement of federated learning rounds (triggered upon the connection of $\geq 3$ clients).
- The dynamic selection of compression algorithms corresponding to regional Round-Trip Times (RTT).

## Step 5: Teardown Procedure

> **Caution**: Failure to destroy the infrastructure post-evaluation will result in ongoing AWS charges.

```bash
make destroy
```

Then verify in the AWS Console (check EC2 in all 4 regions):
- `eu-central-1` (Frankfurt)
- `eu-west-1` (Ireland)
- `us-east-1` (N. Virginia)
- `ap-southeast-1` (Singapore)

## Cost

| Resource | Quantity | Hourly Cost | 8-Hour Run |
|----------|----------|-------------|------------|
| t3.micro instances | 10 | $0.0104/hr each | ~$0.83 |
| EBS gp3 volumes | 10 | ~$0.01/hr total | ~$0.08 |
| ECR storage | 3 images | negligible | ~$0.00 |
| Data transfer | varies | ~$0.09/GB | ~$0.50 |
| **Total** | | | **~$1.50–$2.50** |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `terraform: command not found` | `brew install hashicorp/tap/terraform` |
| `Cannot connect to Docker daemon` | Open Docker Desktop and wait for it to start |
| `No valid credential sources found` | Run `aws configure` with valid keys |
| `UnauthorizedOperation` on EC2 | Add `AmazonEC2FullAccess` to your IAM user |
| Edge nodes not registering | Wait 2–3 minutes for bootstrap; check `make logs` |
| Coordinator health check timeout | SSH in with `make ssh-coordinator`, then `cat /var/log/adf-bootstrap.log` |
| `make destroy` missed something | Check EC2 in all 4 regions manually in the AWS Console |
