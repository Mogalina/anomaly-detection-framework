variable "project_prefix" {
  description = "Prefix applied to all resource names"
  type        = string
}

variable "region" {
  description = "AWS region for this module (informational, provider is set externally)"
  type        = string
}

variable "edge_count_standard" {
  description = "Number of standard-profile edge nodes"
  type        = number
  default     = 0
}

variable "edge_count_lightweight" {
  description = "Number of lightweight-profile edge nodes"
  type        = number
  default     = 0
}

variable "instance_type" {
  description = "EC2 instance type for edge nodes"
  type        = string
  default     = "t3.micro"
}

variable "coordinator_public_ip" {
  description = "Public IP of the coordinator for edge→coordinator gRPC connection"
  type        = string
}

variable "ecr_coordinator_url" {
  description = "Full ECR URL for coordinator image (used to derive account/region for ECR login)"
  type        = string
}

variable "ecr_edge_url" {
  description = "Full ECR URL for the standard edge Docker image"
  type        = string
}

variable "ecr_edge_lightweight_url" {
  description = "Full ECR URL for the lightweight edge Docker image"
  type        = string
}

variable "key_name" {
  description = "Name of the SSH key pair to use"
  type        = string
}

variable "allowed_ssh_cidr" {
  description = "CIDR block allowed to SSH into instances"
  type        = string
  default     = "0.0.0.0/0"
}

variable "ami_name_filter" {
  description = "AMI name filter"
  type        = string
  default     = "al2023-ami-2023.*-x86_64"
}

variable "ami_owner" {
  description = "AMI owner"
  type        = string
  default     = "amazon"
}

variable "primary_region" {
  description = "Primary region (where ECR lives) for cross-region ECR pull"
  type        = string
}

variable "ssh_public_key" {
  description = "SSH public key content (OpenSSH format) for the deployer key pair"
  type        = string
}

variable "edge_userdata_template" {
  description = "Rendered user-data template for edge instances"
  type        = string
  default     = ""
}
