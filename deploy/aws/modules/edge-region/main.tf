terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
}

# ─── AMI lookup ───

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

# ─── Default VPC ───

data "aws_vpc" "default" {
  default = true
}

# ─── Security Group ───

resource "aws_security_group" "edge" {
  name_prefix = "${var.project_prefix}-edge-"
  description = "ADF edge nodes: SSH + Prometheus metrics"
  vpc_id      = data.aws_vpc.default.id

  ingress {
    description = "SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.allowed_ssh_cidr]
  }

  ingress {
    description = "Prometheus metrics"
    from_port   = 9090
    to_port     = 9090
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_prefix}-edge-sg-${var.region}"
  }
}

# ─── IAM Role ───

resource "aws_iam_role" "ec2_ecr" {
  name = "${var.project_prefix}-ec2-ecr-${var.region}"

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
    Name = "${var.project_prefix}-ec2-ecr-${var.region}"
  }
}

resource "aws_iam_role_policy_attachment" "ecr_read" {
  role       = aws_iam_role.ec2_ecr.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly"
}

resource "aws_iam_instance_profile" "ec2_ecr" {
  name = "${var.project_prefix}-ec2-ecr-${var.region}"
  role = aws_iam_role.ec2_ecr.name
}

# ─── SSH Key Pair (import public key from primary region) ───

resource "aws_key_pair" "deployer" {
  key_name   = "${var.project_prefix}-deployer"
  public_key = var.ssh_public_key

  tags = {
    Name = "${var.project_prefix}-deployer-${var.region}"
  }
}

# ─── ECR registry URL (from primary region) ───

locals {
  ecr_registry = split("/", var.ecr_edge_url)[0]
}

# ─── Standard Edge Nodes ───

resource "aws_instance" "edge_standard" {
  count = var.edge_count_standard

  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.deployer.key_name
  vpc_security_group_ids = [aws_security_group.edge.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_ecr.name

  user_data = templatefile("${path.root}/userdata/edge.sh.tpl", {
    ecr_region       = var.primary_region
    ecr_registry     = local.ecr_registry
    ecr_edge_url     = var.ecr_edge_url
    client_id        = "${var.region}-std-${count.index + 1}"
    coordinator_host = var.coordinator_public_ip
    node_profile     = "standard"
  })

  root_block_device {
    volume_size = 15
    volume_type = "gp3"
  }

  tags = {
    Name        = "${var.project_prefix}-edge-${var.region}-std-${count.index + 1}"
    Role        = "edge"
    NodeProfile = "standard"
    Region      = var.region
  }
}

# ─── Lightweight Edge Nodes ───

resource "aws_instance" "edge_lightweight" {
  count = var.edge_count_lightweight

  ami                    = data.aws_ami.amazon_linux.id
  instance_type          = var.instance_type
  key_name               = aws_key_pair.deployer.key_name
  vpc_security_group_ids = [aws_security_group.edge.id]
  iam_instance_profile   = aws_iam_instance_profile.ec2_ecr.name

  user_data = templatefile("${path.root}/userdata/edge.sh.tpl", {
    ecr_region       = var.primary_region
    ecr_registry     = local.ecr_registry
    ecr_edge_url     = var.ecr_edge_lightweight_url
    client_id        = "${var.region}-lwt-${count.index + 1}"
    coordinator_host = var.coordinator_public_ip
    node_profile     = "lightweight"
  })

  root_block_device {
    volume_size = 10
    volume_type = "gp3"
  }

  tags = {
    Name        = "${var.project_prefix}-edge-${var.region}-lwt-${count.index + 1}"
    Role        = "edge"
    NodeProfile = "lightweight"
    Region      = var.region
  }
}
