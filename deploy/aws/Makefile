.PHONY: all build push deploy destroy status ssh-coordinator ssh-edge logs plan validate init clean help

SHELL := /bin/bash
PROJECT_ROOT := $(shell cd ../.. && pwd)
AWS_REGION := $(shell terraform output -raw ecr_registry 2>/dev/null | cut -d. -f4 || echo "eu-central-1")
AWS_ACCOUNT := $(shell aws sts get-caller-identity --query Account --output text 2>/dev/null)
ECR_REGISTRY := $(AWS_ACCOUNT).dkr.ecr.$(AWS_REGION).amazonaws.com
PREFIX := adf

# ─── Full Pipeline ───

all: build push deploy status
	@echo ""
	@echo "===================================="
	@echo "  ADF deployment complete!"
	@echo "===================================="

# ─── Docker Build ───

build:
	@echo "==> Building Docker images (linux/amd64)..."
	docker build \
		--platform linux/amd64 \
		-t $(PREFIX)-coordinator:latest \
		-f $(PROJECT_ROOT)/deploy/docker/Dockerfile.coordinator \
		$(PROJECT_ROOT)

	docker build \
		--platform linux/amd64 \
		-t $(PREFIX)-edge:latest \
		-f $(PROJECT_ROOT)/deploy/docker/Dockerfile.edge \
		$(PROJECT_ROOT)

	docker build \
		--platform linux/amd64 \
		-t $(PREFIX)-edge-lightweight:latest \
		-f $(PROJECT_ROOT)/deploy/docker/Dockerfile.edge-lightweight \
		$(PROJECT_ROOT)

	@echo "==> Docker images built successfully"

# ─── ECR Push ───

push:
	@echo "==> Authenticating to ECR in $(AWS_REGION)..."
	aws ecr get-login-password --region $(AWS_REGION) \
		| docker login --username AWS --password-stdin $(ECR_REGISTRY)

	@echo "==> Tagging and pushing images..."
	docker tag $(PREFIX)-coordinator:latest \
		$(ECR_REGISTRY)/$(PREFIX)/$(PREFIX)-coordinator:latest
	docker push $(ECR_REGISTRY)/$(PREFIX)/$(PREFIX)-coordinator:latest

	docker tag $(PREFIX)-edge:latest \
		$(ECR_REGISTRY)/$(PREFIX)/$(PREFIX)-edge:latest
	docker push $(ECR_REGISTRY)/$(PREFIX)/$(PREFIX)-edge:latest

	docker tag $(PREFIX)-edge-lightweight:latest \
		$(ECR_REGISTRY)/$(PREFIX)/$(PREFIX)-edge-lightweight:latest
	docker push $(ECR_REGISTRY)/$(PREFIX)/$(PREFIX)-edge-lightweight:latest

	@echo "==> Images pushed to ECR"

# ─── Terraform ───

init:
	terraform init

validate: init
	terraform validate

plan: init
	terraform plan

deploy: init
	@echo "==> Deploying infrastructure across 4 regions..."
	terraform apply -auto-approve
	@echo ""
	@echo "==> Waiting 60s for instances to bootstrap..."
	sleep 60
	@echo "==> Deployment complete. Run 'make status' to verify."

destroy:
	@echo "==> Destroying ALL AWS infrastructure..."
	terraform destroy -auto-approve
	@echo "==> All resources destroyed."

# ─── Status & Monitoring ───

status:
	@echo ""
	@echo "========== ADF Deployment Status =========="
	@terraform output -json deployment_summary 2>/dev/null | python3 -m json.tool || echo "Run 'make deploy' first"
	@echo ""
	@echo "--- Health Check ---"
	@COORD_IP=$$(terraform output -raw coordinator_public_ip 2>/dev/null); \
	if [ -n "$$COORD_IP" ]; then \
		echo "Coordinator: http://$$COORD_IP:8080/health"; \
		curl -s --connect-timeout 5 "http://$$COORD_IP:8080/health" 2>/dev/null || echo "(not responding yet)"; \
	fi
	@echo ""
	@echo "==========================================="

ssh-coordinator:
	@KEY=$$(terraform output -raw ssh_private_key_path 2>/dev/null); \
	IP=$$(terraform output -raw coordinator_public_ip 2>/dev/null); \
	echo "Connecting to coordinator at $$IP..."; \
	ssh -i "$$KEY" -o StrictHostKeyChecking=no ec2-user@$$IP

ssh-edge:
	@REGION=$${REGION:?"Usage: make ssh-edge REGION=eu-west-1 N=1 [PROFILE=standard]"}; \
	N=$${N:?"Usage: make ssh-edge REGION=eu-west-1 N=1 [PROFILE=standard]"}; \
	PROFILE=$${PROFILE:-standard}; \
	KEY=$$(terraform output -raw ssh_private_key_path 2>/dev/null); \
	REGION_SLUG=$$(echo "$$REGION" | tr '-' '_'); \
	if [ "$$PROFILE" = "lightweight" ]; then \
		OUTPUT_KEY="$${REGION_SLUG}_lightweight_ips"; \
	else \
		OUTPUT_KEY="$${REGION_SLUG}_standard_ips"; \
	fi; \
	IP=$$(terraform output -json "$$OUTPUT_KEY" 2>/dev/null | python3 -c "import json,sys; ips=json.load(sys.stdin); print(ips[$$N-1])" 2>/dev/null); \
	if [ -z "$$IP" ]; then \
		echo "ERROR: No $$PROFILE edge node #$$N found in $$REGION"; \
		echo "Available outputs:"; \
		terraform output 2>/dev/null | grep -E '_ips'; \
		exit 1; \
	fi; \
	echo "Connecting to $$PROFILE edge node #$$N in $$REGION ($$IP)..."; \
	ssh -i "$$KEY" -o StrictHostKeyChecking=no ec2-user@$$IP

logs:
	@KEY=$$(terraform output -raw ssh_private_key_path 2>/dev/null); \
	IP=$$(terraform output -raw coordinator_public_ip 2>/dev/null); \
	ssh -i "$$KEY" -o StrictHostKeyChecking=no ec2-user@$$IP \
		"sudo docker logs --tail 100 -f adf-coordinator"

# ─── Cleanup ───

clean:
	rm -f adf-key.pem adf-key.pub
	rm -rf .terraform .terraform.lock.hcl terraform.tfstate*

# ─── Help ───

help:
	@echo "ADF AWS Deployment Targets:"
	@echo "  make all              Full pipeline: build → push → deploy"
	@echo "  make build            Build Docker images for linux/amd64"
	@echo "  make push             Push images to ECR"
	@echo "  make deploy           Terraform init + apply"
	@echo "  make destroy          Terraform destroy (cleanup all)"
	@echo "  make status           Show instance IPs and health"
	@echo "  make ssh-coordinator  SSH into coordinator"
	@echo "  make ssh-edge REGION=eu-west-1 N=1  SSH into specific edge node"
	@echo "                        PROFILE=standard|lightweight (default: standard)"
	@echo "  make logs             Tail coordinator container logs"
	@echo "  make plan             Terraform dry-run"
	@echo "  make validate         Terraform validate"
	@echo "  make clean            Remove local state files"
