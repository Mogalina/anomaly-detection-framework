# How to Run the AWS Multi-Region Deployment

Step-by-step guide to deploy the anomaly detection framework across 4 AWS regions with 10 VMs.

## Prerequisites to Install

You need 4 tools on your Mac. Check what you already have:

```bash
aws --version       # AWS CLI
terraform --version # Terraform
docker --version    # Docker
```

### Install anything missing:

```bash
# 1. AWS CLI (if missing)
brew install awscli

# 2. Terraform (if missing)
brew install hashicorp/tap/terraform

# 3. Docker Desktop (if missing)
brew install --cask docker
# Then launch Docker Desktop from Applications and wait for it to start
```

## Step 1: Configure AWS Credentials

> [!IMPORTANT]
> You need an AWS account. If you don't have one, create one at https://aws.amazon.com — the free tier covers `t3.micro` for 750 hrs/month.

```bash
aws configure
```

It will ask for 4 values:

| Prompt | Where to find it |
|--------|-----------------|
| **Access Key ID** | AWS Console → IAM → Users → your user → Security Credentials → Create Access Key |
| **Secret Access Key** | Shown once when you create the access key (save it!) |
| **Default region** | Enter `eu-central-1` |
| **Output format** | Enter `json` |

Verify it works:
```bash
aws sts get-caller-identity
```
You should see your account ID. If you get an error, your credentials are wrong.

### IAM Permissions Needed

Your IAM user needs these permissions (attach these managed policies in the AWS Console → IAM → your user → Permissions):

- `AmazonEC2FullAccess`
- `AmazonEC2ContainerRegistryFullAccess`
- `IAMFullAccess` (for creating EC2 instance profiles)

> [!TIP]
> For a quick test, you can temporarily use `AdministratorAccess` and restrict later.

## Step 2: Ensure Docker is Running

```bash
docker info
```

If you see an error like "Cannot connect to the Docker daemon", open **Docker Desktop** from Applications and wait for the whale icon to appear in the menu bar.

## Step 3: Deploy Everything (One Command)

```bash
cd /Users/mogalina/Desktop/anomaly-detection-framework/deploy/aws
make all
```

This runs the full pipeline:
1. **`make build`** — Builds 3 Docker images (coordinator, edge, edge-lightweight) for `linux/amd64`
2. **`make push`** — Creates ECR repos and pushes images
3. **`make deploy`** — Runs `terraform init` + `terraform apply` → provisions 10 VMs across 4 regions
4. **`make status`** — Shows all instance IPs and coordinator health

> [!NOTE]
> The first `make build` takes 5–10 minutes (downloading PyTorch, etc.). Subsequent builds are cached and fast.

### Alternative: Step-by-Step Script

If you prefer more visibility into each step:

```bash
cd /Users/mogalina/Desktop/anomaly-detection-framework/deploy/aws
chmod +x deploy.sh
./deploy.sh
```

This does the same thing but with colored output, health-check polling, and a connection summary at the end.

## Step 4: Verify the Deployment

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

### What to Look For

In `make logs`, you should see:
- Edge nodes registering from all 4 regions
- Federated learning rounds starting (once ≥3 clients connect)
- Compression algorithm selection varying by RTT

## Step 5: Tear Everything Down

> [!CAUTION]
> Always destroy when done to avoid charges!

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

> [!TIP]
> `t3.micro` is **free-tier eligible** (750 hrs/month for 12 months after account creation). If your account is within the free tier window, the compute cost is $0.

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
