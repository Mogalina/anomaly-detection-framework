# ──────────────────────────────────────────────────────────────
# ADF — Secondary Region Providers & Edge Modules
# ──────────────────────────────────────────────────────────────
# Deploys edge nodes in eu-west-1, us-east-1, ap-southeast-1
# using the reusable edge-region module.
# ──────────────────────────────────────────────────────────────

# ── Export public key for secondary regions ──────────────────

resource "local_file" "ssh_public_key" {
  content  = tls_private_key.ssh.public_key_openssh
  filename = "${path.module}/adf-key.pub"
}

# ── Secondary Region Providers ───────────────────────────────

provider "aws" {
  alias  = "eu_west_1"
  region = "eu-west-1"

  default_tags {
    tags = {
      Project     = var.project_prefix
      ManagedBy   = "terraform"
      Environment = "thesis-evaluation"
    }
  }
}

provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"

  default_tags {
    tags = {
      Project     = var.project_prefix
      ManagedBy   = "terraform"
      Environment = "thesis-evaluation"
    }
  }
}

provider "aws" {
  alias  = "ap_southeast_1"
  region = "ap-southeast-1"

  default_tags {
    tags = {
      Project     = var.project_prefix
      ManagedBy   = "terraform"
      Environment = "thesis-evaluation"
    }
  }
}

# ── eu-west-1 (Ireland) — 1 standard + 1 lightweight ────────

module "edge_eu_west_1" {
  source = "./modules/edge-region"

  providers = {
    aws = aws.eu_west_1
  }

  project_prefix           = var.project_prefix
  region                   = "eu-west-1"
  edge_count_standard      = lookup(var.secondary_regions, "eu-west-1", { edge_count_standard = 0, edge_count_lightweight = 0 }).edge_count_standard
  edge_count_lightweight   = lookup(var.secondary_regions, "eu-west-1", { edge_count_standard = 0, edge_count_lightweight = 0 }).edge_count_lightweight
  instance_type            = var.edge_instance_type
  coordinator_public_ip    = aws_instance.coordinator.public_ip
  ecr_coordinator_url      = local.ecr_coordinator
  ecr_edge_url             = local.ecr_edge
  ecr_edge_lightweight_url = local.ecr_edge_lw
  key_name                 = aws_key_pair.deployer.key_name
  ssh_public_key           = tls_private_key.ssh.public_key_openssh
  allowed_ssh_cidr         = var.allowed_ssh_cidr
  ami_name_filter          = var.ami_name_filter
  ami_owner                = var.ami_owner
  primary_region           = var.primary_region

  depends_on = [aws_instance.coordinator, local_file.ssh_public_key]
}

# ── us-east-1 (N. Virginia) — 2 standard + 1 lightweight ───

module "edge_us_east_1" {
  source = "./modules/edge-region"

  providers = {
    aws = aws.us_east_1
  }

  project_prefix           = var.project_prefix
  region                   = "us-east-1"
  edge_count_standard      = lookup(var.secondary_regions, "us-east-1", { edge_count_standard = 0, edge_count_lightweight = 0 }).edge_count_standard
  edge_count_lightweight   = lookup(var.secondary_regions, "us-east-1", { edge_count_standard = 0, edge_count_lightweight = 0 }).edge_count_lightweight
  instance_type            = var.edge_instance_type
  coordinator_public_ip    = aws_instance.coordinator.public_ip
  ecr_coordinator_url      = local.ecr_coordinator
  ecr_edge_url             = local.ecr_edge
  ecr_edge_lightweight_url = local.ecr_edge_lw
  key_name                 = aws_key_pair.deployer.key_name
  ssh_public_key           = tls_private_key.ssh.public_key_openssh
  allowed_ssh_cidr         = var.allowed_ssh_cidr
  ami_name_filter          = var.ami_name_filter
  ami_owner                = var.ami_owner
  primary_region           = var.primary_region

  depends_on = [aws_instance.coordinator, local_file.ssh_public_key]
}

# ── ap-southeast-1 (Singapore) — 0 standard + 1 lightweight ─

module "edge_ap_southeast_1" {
  source = "./modules/edge-region"

  providers = {
    aws = aws.ap_southeast_1
  }

  project_prefix           = var.project_prefix
  region                   = "ap-southeast-1"
  edge_count_standard      = lookup(var.secondary_regions, "ap-southeast-1", { edge_count_standard = 0, edge_count_lightweight = 0 }).edge_count_standard
  edge_count_lightweight   = lookup(var.secondary_regions, "ap-southeast-1", { edge_count_standard = 0, edge_count_lightweight = 0 }).edge_count_lightweight
  instance_type            = var.edge_instance_type
  coordinator_public_ip    = aws_instance.coordinator.public_ip
  ecr_coordinator_url      = local.ecr_coordinator
  ecr_edge_url             = local.ecr_edge
  ecr_edge_lightweight_url = local.ecr_edge_lw
  key_name                 = aws_key_pair.deployer.key_name
  ssh_public_key           = tls_private_key.ssh.public_key_openssh
  allowed_ssh_cidr         = var.allowed_ssh_cidr
  ami_name_filter          = var.ami_name_filter
  ami_owner                = var.ami_owner
  primary_region           = var.primary_region

  depends_on = [aws_instance.coordinator, local_file.ssh_public_key]
}
