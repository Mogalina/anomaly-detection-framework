# ──────────────────────────────────────────────────────────────
# ADF — Terraform Root Module (Primary Region)
# ──────────────────────────────────────────────────────────────
# Provisions:
#   • ECR repositories for all Docker images
#   • SSH key pair (auto-generated)
#   • Security group (gRPC, HTTP, Prometheus, SSH)
#   • Coordinator EC2 instance
#   • Primary-region edge nodes (standard + lightweight)
# ──────────────────────────────────────────────────────────────

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    tls = {
      source  = "hashicorp/tls"
      version = "~> 4.0"
    }
  }
}

# ── Primary region provider ──────────────────────────────────

provider "aws" {
  region = var.primary_region

  default_tags {
    tags = {
      Project     = var.project_prefix
      ManagedBy   = "terraform"
      Environment = "thesis-evaluation"
    }
  }
}

# ── Data sources ─────────────────────────────────────────────

data "aws_caller_identity" "current" {}

data "aws_vpc" "default" {
  default = true
}

data "aws_subnets" "default" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.default.id]
  }
  filter {
    name   = "default-for-az"
    values = ["true"]
  }
}

data "aws_ami" "amazon_linux" {
  most_recent = true
  owners      = [var.ami_owner]

  filter {
    name   = "name"
    values = [var.ami_name_filter]
  }

  filter {
    name   = "architecture"
    values = ["x86_64"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

# ── ECR Repositories ────────────────────────────────────────

resource "aws_ecr_repository" "images" {
  for_each = toset(var.ecr_repo_names)

  name                 = "${var.project_prefix}/${each.value}"
  image_tag_mutability = "MUTABLE"
  force_delete         = true

  image_scanning_configuration {
    scan_on_push = false
  }

  tags = {
    Name = "${var.project_prefix}-${each.value}"
  }
}

# ── SSH Key Pair (auto-generated) ────────────────────────────

resource "tls_private_key" "ssh" {
  algorithm = "RSA"
  rsa_bits  = 4096
}

resource "aws_key_pair" "deployer" {
  key_name   = "${var.project_prefix}-deployer"
  public_key = tls_private_key.ssh.public_key_openssh

  tags = {
    Name = "${var.project_prefix}-deployer"
  }
}

resource "local_file" "ssh_private_key" {
  content         = tls_private_key.ssh.private_key_pem
  filename        = "${path.module}/adf-key.pem"
  file_permission = "0400"
}

# ── Security Group ───────────────────────────────────────────

resource "aws_security_group" "adf" {
  name_prefix = "${var.project_prefix}-"
  description = "ADF framework: gRPC, HTTP API, Prometheus, SSH"
  vpc_id      = data.aws_vpc.default.id

  # SSH
  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  # gRPC (FL coordinator)
  ingress {
    description = "gRPC FL"
    from_port   = 50051
    to_port     = 50051
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # HTTP API
  ingress {
    description = "HTTP API"
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Prometheus metrics
  ingress {
    description = "Prometheus metrics"
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # All egress
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_prefix}-sg"
  }
}

# ── IAM Role for EC2 (ECR pull access) ──────────────────────

resource "aws_iam_role" "ec2_ecr" {
  name = "${var.project_prefix}-ec2-ecr-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name = "${var.project_prefix}-ec2-ecr-role"
  }
}

resource "aws_iam_role_policy_attachment" "ecr_read" {
  role       = aws_iam_role.ec2_ecr.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_instance_profile" "ec2_ecr" {
  name = "${var.project_prefix}-ec2-ecr-profile"
  role = aws_iam_role.ec2_ecr.name
}

# ── Locals ───────────────────────────────────────────────────

locals {
  ecr_registry      = "${data.aws_caller_identity.current.account_id}.dkr.ecr.${var.primary_region}.amazonaws.com"
  ecr_coordinator   = "${local.ecr_registry}/${var.project_prefix}/adf-coordinator"
  ecr_edge          = "${local.ecr_registry}/${var.project_prefix}/adf-edge"
  ecr_edge_lw       = "${local.ecr_registry}/${var.project_prefix}/adf-edge-lightweight"
}

# ── Coordinator Instance ────────────────────────────────────

resource "aws_instance" "coordinator" {
  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = var.coordinator_instance_type
  key_name               = aws_key_pair.deployer.key_name
  vpc_security_group_ids = [aws_security_group.adf.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_ecr.name

  user_data = templatefile("${path.module}/userdata/coordinator.sh.tpl", {
    aws_region           = var.primary_region
    ecr_registry         = local.ecr_registry
    ecr_coordinator_url  = local.ecr_coordinator
  })

  root_block_device {
    volume_size = 20
    volume_type = "gp3"
  }

  tags = {
    Name = "${var.project_prefix}-coordinator"
    Role = "coordinator"
  }
}

# ── Primary Region Edge Nodes (Standard) ────────────────────

resource "aws_instance" "edge_standard" {
  count = var.primary_edge_count_standard

  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = var.edge_instance_type
  key_name               = aws_key_pair.deployer.key_name
  vpc_security_group_ids = [aws_security_group.adf.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_ecr.name

  user_data = templatefile("${path.module}/userdata/edge.sh.tpl", {
    ecr_region       = var.primary_region
    ecr_registry     = local.ecr_registry
    ecr_edge_url     = local.ecr_edge
    client_id        = "${var.primary_region}-std-${count.index + 1}"
    coordinator_host = aws_instance.coordinator.public_ip
    node_profile     = "standard"
  })

  root_block_device {
    volume_size = 15
    volume_type = "gp3"
  }

  tags = {
    Name        = "${var.project_prefix}-edge-${var.primary_region}-std-${count.index + 1}"
    Role        = "edge"
    NodeProfile = "standard"
    Region      = var.primary_region
  }

  depends_on = [aws_instance.coordinator]
}

# ── Primary Region Edge Nodes (Lightweight) ─────────────────

resource "aws_instance" "edge_lightweight" {
  count = var.primary_edge_count_lightweight

  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = var.edge_instance_type
  key_name               = aws_key_pair.deployer.key_name
  vpc_security_group_ids = [aws_security_group.adf.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_ecr.name

  user_data = templatefile("${path.module}/userdata/edge.sh.tpl", {
    ecr_region       = var.primary_region
    ecr_registry     = local.ecr_registry
    ecr_edge_url     = local.ecr_edge_lw
    client_id        = "${var.primary_region}-lwt-${count.index + 1}"
    coordinator_host = aws_instance.coordinator.public_ip
    node_profile     = "lightweight"
  })

  root_block_device {
    volume_size = 10
    volume_type = "gp3"
  }

  tags = {
    Name        = "${var.project_prefix}-edge-${var.primary_region}-lwt-${count.index + 1}"
    Role        = "edge"
    NodeProfile = "lightweight"
    Region      = var.primary_region
  }

  depends_on = [aws_instance.coordinator]
}
