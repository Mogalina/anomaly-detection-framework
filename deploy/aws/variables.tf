# ──────────────────────────────────────────────────────────────
# ADF — Terraform Variables
# ──────────────────────────────────────────────────────────────

variable "project_prefix" {
  description = "Prefix applied to all resource names"
  type        = string
  default     = "adf"
}

# ── Regions ──────────────────────────────────────────────────

variable "primary_region" {
  description = "AWS region for the coordinator and primary edge nodes"
  type        = string
  default     = "eu-central-1"
}

variable "secondary_regions" {
  description = "Map of secondary regions to deploy edge nodes in"
  type = map(object({
    edge_count_standard    = number
    edge_count_lightweight = number
  }))
  default = {
    "eu-west-1" = {
      edge_count_standard    = 1
      edge_count_lightweight = 1
    }
    "us-east-1" = {
      edge_count_standard    = 2
      edge_count_lightweight = 1
    }
    "ap-southeast-1" = {
      edge_count_standard    = 0
      edge_count_lightweight = 1
    }
  }
}

# ── Primary region edge counts ───────────────────────────────

variable "primary_edge_count_standard" {
  description = "Number of standard-profile edge nodes in the primary region"
  type        = number
  default     = 1
}

variable "primary_edge_count_lightweight" {
  description = "Number of lightweight-profile edge nodes in the primary region"
  type        = number
  default     = 1
}

# ── Instance types ───────────────────────────────────────────

variable "coordinator_instance_type" {
  description = "EC2 instance type for the coordinator"
  type        = string
  default     = "t3.small"
}

variable "edge_instance_type" {
  description = "EC2 instance type for edge nodes"
  type        = string
  default     = "t3.micro"
}

# ── Networking & SSH ─────────────────────────────────────────

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH into instances (restrict for production)"
  type        = string
  default     = "0.0.0.0/0"
}

# ── ECR ──────────────────────────────────────────────────────

variable "ecr_repo_names" {
  description = "ECR repository names for the Docker images"
  type        = list(string)
  default     = ["adf-coordinator", "adf-edge", "adf-edge-lightweight"]
}

# ── AMI ──────────────────────────────────────────────────────

variable "ami_name_filter" {
  description = "AMI name filter for Amazon Linux 2023"
  type        = string
  default     = "al2023-ami-2023.*-x86_64"
}

variable "ami_owner" {
  description = "AMI owner (amazon)"
  type        = string
  default     = "amazon"
}
