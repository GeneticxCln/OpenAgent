# OpenAgent Deployment Guide

## Overview

This guide covers deployment strategies, configurations, and operational considerations for running OpenAgent in production environments.

## Table of Contents

- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Configuration Management](#configuration-management)
- [Deployment Strategies](#deployment-strategies)
- [Security Considerations](#security-considerations)
- [Monitoring and Observability](#monitoring-and-observability)
- [Scaling and Performance](#scaling-and-performance)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Single Instance Deployment

```bash
# 1. Install OpenAgent
pip install openagent[all]

# 2. Create configuration
cp .env.example .env
# Edit .env with your settings

# 3. Start the service
uvicorn openagent.server.app:app --host 0.0.0.0 --port 8000

# 4. Verify deployment
curl http://localhost:8000/health
```

### Docker Deployment

```bash
# Using Docker Compose
docker-compose up -d

# Using Docker directly
docker run -d \
  --name openagent \
  -p 8000:8000 \
  -e OPENAGENT_MODEL=tiny-llama \
  -e OPENAGENT_DEVICE=cpu \
  -v $(pwd)/config:/app/config \
  openagent:latest
```

## Environment Setup

### System Requirements

#### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 10GB (for models and cache)
- **OS**: Linux, macOS, Windows

#### Recommended Requirements
- **CPU**: 4+ cores
- **RAM**: 16GB+ (for larger models)
- **Storage**: 50GB+ SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)

### Operating System Setup

#### Ubuntu/Debian
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.9+
sudo apt install python3.11 python3.11-venv python3.11-dev

# Install system dependencies
sudo apt install git curl build-essential

# Install Docker (optional)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
```

#### CentOS/RHEL
```bash
# Install Python 3.9+
sudo dnf install python3.11 python3.11-pip python3.11-devel

# Install development tools
sudo dnf groupinstall "Development Tools"

# Install Docker (optional)
sudo dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo dnf install docker-ce docker-ce-cli containerd.io
```

#### macOS
```bash
# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python@3.11

# Install Docker Desktop (optional)
# Download from https://docker.com/products/docker-desktop
```

## Configuration Management

### Environment Variables

#### Core Configuration
```bash
# Application Settings
OPENAGENT_MODEL=codellama-7b          # Default model to use
OPENAGENT_DEVICE=auto                 # cuda/cpu/auto
OPENAGENT_PORT=8000                   # Server port
OPENAGENT_HOST=0.0.0.0               # Server host

# Model Settings
OPENAGENT_LOAD_IN_4BIT=true          # Enable 4-bit quantization
OPENAGENT_MAX_MEMORY=8G              # Maximum memory usage
OPENAGENT_MODEL_CACHE_DIR=/app/models # Model cache directory

# Security Settings
OPENAGENT_EXEC_STRICT=false          # Strict execution mode
OPENAGENT_POLICY_FILE=/app/config/policy.yaml
OPENAGENT_BLOCK_RISKY=true           # Block risky operations by default

# API Settings
OPENAGENT_API_CORS_ORIGINS="*"       # CORS origins
OPENAGENT_API_RATE_LIMIT=100         # Requests per minute
OPENAGENT_JWT_SECRET_KEY="your-secret-key"

# Database Settings
OPENAGENT_DATABASE_URL="postgresql://user:pass@localhost/openagent"
OPENAGENT_REDIS_URL="redis://localhost:6379/0"

# Observability
OPENAGENT_LOG_LEVEL=INFO             # DEBUG/INFO/WARNING/ERROR
OPENAGENT_METRICS_ENABLED=true       # Enable Prometheus metrics
OPENAGENT_TRACES_ENABLED=false       # Enable request tracing
```

#### Model-Specific Configuration
```bash
# HuggingFace Settings
HUGGINGFACE_TOKEN="hf_your_token_here"
HUGGINGFACE_CACHE_DIR="/app/cache/huggingface"

# Ollama Settings
OLLAMA_HOST="http://localhost:11434"
OLLAMA_MODEL="llama3.2:7b"

# OpenAI Settings (if using)
OPENAI_API_KEY="sk-your-key-here"
OPENAI_BASE_URL="https://api.openai.com/v1"
```

### Configuration File Structure

#### Main Configuration (`config/app.yaml`)
```yaml
# Application Configuration
app:
  name: "OpenAgent"
  version: "0.1.3"
  debug: false
  
# Server Configuration
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  
# Model Configuration
models:
  default: "codellama-7b"
  providers:
    huggingface:
      cache_dir: "/app/models"
      load_in_4bit: true
      torch_dtype: "auto"
    ollama:
      host: "http://localhost:11434"
      timeout: 60
      
# Security Configuration
security:
  policy_file: "/app/config/policy.yaml"
  exec_strict: false
  block_risky: true
  audit_logs: true
  
# Database Configuration
database:
  url: "${OPENAGENT_DATABASE_URL}"
  pool_size: 20
  max_overflow: 0
  
# Cache Configuration  
cache:
  redis_url: "${OPENAGENT_REDIS_URL}"
  ttl: 3600
  
# Logging Configuration
logging:
  level: "INFO"
  format: "json"
  handlers:
    - "console"
    - "file"
  file_path: "/app/logs/openagent.log"
  max_size: "100MB"
  backup_count: 5
```

#### Security Policy (`config/policy.yaml`)
```yaml
# Command Policies
command_policies:
  - pattern: "rm -rf .*"
    action: "DENY"
    reason: "Dangerous recursive delete"
    
  - pattern: "sudo .*"
    action: "REQUIRE_APPROVAL"
    reason: "Privileged operation"
    
  - pattern: "git .*"
    action: "ALLOW"
    reason: "Version control operations"

# File Access Policies
file_policies:
  safe_paths:
    - "/home/user/projects"
    - "/tmp"
    - "/app/workspace"
  
  restricted_paths:
    - "/etc"
    - "/root"
    - "/usr/bin"
    - "/sys"
    - "/proc"

# Network Policies
network_policies:
  allowed_domains:
    - "github.com"
    - "huggingface.co"
    - "pypi.org"
  
  blocked_ips:
    - "169.254.0.0/16"  # Link-local
    - "10.0.0.0/8"      # Private networks (if needed)

# Resource Limits
resource_limits:
  max_memory_mb: 8192
  max_cpu_percent: 80
  max_execution_time: 300
  max_file_size_mb: 100
```

## Deployment Strategies

### 1. Single Instance Deployment

#### Systemd Service (Linux)
```ini
# /etc/systemd/system/openagent.service
[Unit]
Description=OpenAgent AI Assistant
After=network.target

[Service]
Type=forking
User=openagent
Group=openagent
WorkingDirectory=/opt/openagent
Environment=PATH=/opt/openagent/venv/bin
ExecStart=/opt/openagent/venv/bin/uvicorn openagent.server.app:app --host 0.0.0.0 --port 8000 --workers 4
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable openagent
sudo systemctl start openagent
sudo systemctl status openagent
```

#### Docker Container
```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create application user
RUN useradd --create-home --shell /bin/bash openagent

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Set ownership
RUN chown -R openagent:openagent /app

# Switch to application user
USER openagent

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "openagent.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 2. High Availability Deployment

#### Docker Compose Setup
```yaml
version: '3.8'

services:
  openagent:
    image: openagent:latest
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 8G
        reservations:
          cpus: '1.0'
          memory: 4G
    environment:
      - OPENAGENT_DATABASE_URL=postgresql://openagent:password@postgres:5432/openagent
      - OPENAGENT_REDIS_URL=redis://redis:6379/0
    networks:
      - openagent-net
    depends_on:
      - postgres
      - redis

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl/certs
    networks:
      - openagent-net
    depends_on:
      - openagent

  postgres:
    image: postgres:15
    environment:
      - POSTGRES_DB=openagent
      - POSTGRES_USER=openagent
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - openagent-net

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    networks:
      - openagent-net

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    networks:
      - openagent-net

volumes:
  postgres_data:
  redis_data:

networks:
  openagent-net:
    driver: bridge
```

#### Kubernetes Deployment
```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: openagent
  labels:
    app: openagent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: openagent
  template:
    metadata:
      labels:
        app: openagent
    spec:
      containers:
      - name: openagent
        image: openagent:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAGENT_DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: openagent-secrets
              key: database-url
        - name: OPENAGENT_REDIS_URL
          valueFrom:
            secretKeyRef:
              name: openagent-secrets
              key: redis-url
        resources:
          requests:
            memory: "4Gi"
            cpu: "1"
          limits:
            memory: "8Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: openagent-service
spec:
  selector:
    app: openagent
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

### 3. Cloud Deployment

#### AWS ECS with Fargate
```json
{
  "family": "openagent-task",
  "taskRoleArn": "arn:aws:iam::account:role/openagent-task-role",
  "executionRoleArn": "arn:aws:iam::account:role/openagent-execution-role",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "8192",
  "containerDefinitions": [
    {
      "name": "openagent",
      "image": "your-account.dkr.ecr.region.amazonaws.com/openagent:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "OPENAGENT_MODEL",
          "value": "codellama-7b"
        }
      ],
      "secrets": [
        {
          "name": "OPENAGENT_DATABASE_URL",
          "valueFrom": "arn:aws:secretsmanager:region:account:secret:openagent/database"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/openagent",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

## Security Considerations

### Network Security
- Use HTTPS/TLS for all communications
- Implement proper firewall rules
- Use VPNs for internal communications
- Regular security updates

### Authentication & Authorization
```python
# JWT Configuration
JWT_SECRET_KEY = "your-secure-secret-key"
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_TIME = 24 * 60 * 60  # 24 hours

# Role-based access control
RBAC_ROLES = {
    "admin": ["*"],
    "user": ["chat", "tools", "system_info"],
    "readonly": ["chat", "system_info"]
}
```

### Data Protection
- Encrypt sensitive data at rest
- Use secrets management systems
- Implement proper backup strategies
- Regular security audits

## Monitoring and Observability

### Prometheus Metrics
```yaml
# prometheus.yml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'openagent'
    static_configs:
      - targets: ['openagent:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### Grafana Dashboard
Key metrics to monitor:
- Request rate and latency
- Model inference time
- Memory and CPU usage
- Error rates
- Active connections

### Log Aggregation
```yaml
# filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /app/logs/*.log
  json.keys_under_root: true
  json.add_error_key: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
```

## Scaling and Performance

### Horizontal Scaling
- Use load balancers (Nginx, HAProxy, AWS ALB)
- Implement session stickiness if needed
- Share model cache across instances
- Use distributed caching (Redis)

### Vertical Scaling
- Monitor resource usage
- Adjust memory limits for models
- Optimize model loading strategies
- Use GPU acceleration when available

### Performance Tuning
```python
# Model optimization
OPTIMIZATION_SETTINGS = {
    "load_in_4bit": True,
    "torch_compile": True,
    "attention_dropout": 0.1,
    "max_position_embeddings": 4096,
    "use_cache": True
}
```

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
docker stats openagent

# Optimize model loading
export OPENAGENT_LOAD_IN_4BIT=true
export OPENAGENT_MAX_MEMORY=6G
```

#### Slow Response Times
```bash
# Check GPU availability
nvidia-smi

# Enable GPU acceleration
export OPENAGENT_DEVICE=cuda

# Increase worker processes
export OPENAGENT_WORKERS=4
```

#### Connection Issues
```bash
# Check service status
systemctl status openagent

# Check logs
journalctl -u openagent -f

# Test connectivity
curl -v http://localhost:8000/health
```

### Health Checks
```python
# Custom health check endpoint
@app.get("/health")
async def health_check():
    """Comprehensive health check."""
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": __version__,
        "checks": {
            "database": await check_database(),
            "cache": await check_redis(),
            "model": await check_model_availability(),
            "memory": check_memory_usage(),
            "disk": check_disk_space()
        }
    }
    
    # Determine overall health
    all_healthy = all(check["status"] == "healthy" for check in health_status["checks"].values())
    
    if not all_healthy:
        health_status["status"] = "unhealthy"
        raise HTTPException(status_code=503, detail=health_status)
    
    return health_status
```

### Backup and Recovery
```bash
# Database backup
pg_dump openagent > backup-$(date +%Y%m%d).sql

# Model cache backup
tar -czf models-backup.tar.gz /app/models/

# Configuration backup
cp -r /app/config/ /backup/config-$(date +%Y%m%d)/

# Restore procedure
psql openagent < backup-20240101.sql
tar -xzf models-backup.tar.gz -C /app/
```

This deployment guide provides comprehensive coverage for deploying OpenAgent in various environments while maintaining security, performance, and reliability standards.
