output "edge_standard_public_ips" {
  description = "Public IPs of standard-profile edge nodes in this region"
  value       = aws_instance.edge_standard[*].public_ip
}

output "edge_lightweight_public_ips" {
  description = "Public IPs of lightweight-profile edge nodes in this region"
  value       = aws_instance.edge_lightweight[*].public_ip
}

output "edge_standard_ids" {
  description = "Instance IDs of standard-profile edge nodes"
  value       = aws_instance.edge_standard[*].id
}

output "edge_lightweight_ids" {
  description = "Instance IDs of lightweight-profile edge nodes"
  value       = aws_instance.edge_lightweight[*].id
}
