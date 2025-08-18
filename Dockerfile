# Multi-stage build for OpenAgent
FROM python:3.11-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash openagent

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt requirements-dev.txt ./

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Development stage
FROM base as development

# Install development dependencies
RUN pip install -r requirements-dev.txt

# Copy source code
COPY . .

# Set ownership
RUN chown -R openagent:openagent /app

USER openagent

# Expose ports
EXPOSE 8000

# Default command for development
CMD ["python", "-m", "openagent.server.app"]

# Production stage
FROM base as production

# Copy only production code
COPY --chown=openagent:openagent openagent/ ./openagent/
COPY --chown=openagent:openagent setup.py pyproject.toml README.md ./

# Install the package
RUN pip install -e .

# Switch to non-root user
USER openagent

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["python", "-m", "openagent.server.app", "--host", "0.0.0.0", "--port", "8000"]

# Lightweight stage for minimal deployments
FROM python:3.11-alpine as lightweight

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install minimal system dependencies
RUN apk add --no-cache \
    curl \
    git

# Create user
RUN adduser -D -s /bin/sh openagent

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY --chown=openagent:openagent openagent/ ./openagent/
COPY --chown=openagent:openagent setup.py README.md ./

# Install package
RUN pip install -e .

USER openagent

EXPOSE 8000

CMD ["python", "-m", "openagent.server.app", "--host", "0.0.0.0"]
