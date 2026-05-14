# ─── Coordinator ───

output "coordinator_public_ip" {
  description = "Public IP of the coordinator instance"
  value       = aws_instance.coordinator.public_ip
}

output "coordinator_public_dns" {
  description = "Public DNS of the coordinator instance"
  value       = aws_instance.coordinator.public_dns
}

output "coordinator_health_url" {
  description = "Coordinator health check URL"
  value       = "http://${aws_instance.coordinator.public_ip}:8080/health"
}

# ─── ECR ───

output "ecr_coordinator_url" {
  description = "ECR URL for the coordinator image"
  value       = local.ecr_coordinator
}

output "ecr_edge_url" {
  description = "ECR URL for the standard edge image"
  value       = local.ecr_edge
}

output "ecr_edge_lightweight_url" {
  description = "ECR URL for the lightweight edge image"
  value       = local.ecr_edge_lw
}

output "ecr_registry" {
  description = "ECR registry base URL"
  value       = local.ecr_registry
}

# ─── Primary Region Edge Nodes ───

output "primary_edge_standard_ips" {
  description = "Public IPs of standard edge nodes in the primary region"
  value       = aws_instance.edge_standard[*].public_ip
}

output "primary_edge_lightweight_ips" {
  description = "Public IPs of lightweight edge nodes in the primary region"
  value       = aws_instance.edge_lightweight[*].public_ip
}

# ─── Secondary Region Edge Nodes ───

output "eu_west_1_standard_ips" {
  description = "Standard edge node IPs in eu-west-1"
  value       = module.edge_eu_west_1.edge_standard_public_ips
}

output "eu_west_1_lightweight_ips" {
  description = "Lightweight edge node IPs in eu-west-1"
  value       = module.edge_eu_west_1.edge_lightweight_public_ips
}

output "us_east_1_standard_ips" {
  description = "Standard edge node IPs in us-east-1"
  value       = module.edge_us_east_1.edge_standard_public_ips
}

output "us_east_1_lightweight_ips" {
  description = "Lightweight edge node IPs in us-east-1"
  value       = module.edge_us_east_1.edge_lightweight_public_ips
}

output "ap_southeast_1_standard_ips" {
  description = "Standard edge node IPs in ap-southeast-1"
  value       = module.edge_ap_southeast_1.edge_standard_public_ips
}

output "ap_southeast_1_lightweight_ips" {
  description = "Lightweight edge node IPs in ap-southeast-1"
  value       = module.edge_ap_southeast_1.edge_lightweight_public_ips
}

# ─── SSH ───

output "ssh_private_key_path" {
  description = "Path to the generated SSH private key"
  value       = local_file.ssh_private_key.filename
}

output "ssh_coordinator_command" {
  description = "SSH command to connect to the coordinator"
  value       = "ssh -i ${local_file.ssh_private_key.filename} ec2-user@${aws_instance.coordinator.public_ip}"
}

# ─── Summary ───

output "deployment_summary" {
  description = "Summary of the deployment"
  value = {
    coordinator = {
      ip  = aws_instance.coordinator.public_ip
      dns = aws_instance.coordinator.public_dns
    }
    edge_nodes = {
      "eu-central-1 (primary)" = {
        standard    = aws_instance.edge_standard[*].public_ip
        lightweight = aws_instance.edge_lightweight[*].public_ip
      }
      "eu-west-1" = {
        standard    = module.edge_eu_west_1.edge_standard_public_ips
        lightweight = module.edge_eu_west_1.edge_lightweight_public_ips
      }
      "us-east-1" = {
        standard    = module.edge_us_east_1.edge_standard_public_ips
        lightweight = module.edge_us_east_1.edge_lightweight_public_ips
      }
      "ap-southeast-1" = {
        standard    = module.edge_ap_southeast_1.edge_standard_public_ips
        lightweight = module.edge_ap_southeast_1.edge_lightweight_public_ips
      }
    }
    total_instances = (
      1 +
      var.primary_edge_count_standard + var.primary_edge_count_lightweight +
      length(module.edge_eu_west_1.edge_standard_public_ips) + length(module.edge_eu_west_1.edge_lightweight_public_ips) +
      length(module.edge_us_east_1.edge_standard_public_ips) + length(module.edge_us_east_1.edge_lightweight_public_ips) +
      length(module.edge_ap_southeast_1.edge_standard_public_ips) + length(module.edge_ap_southeast_1.edge_lightweight_public_ips)
    )
  }
}
