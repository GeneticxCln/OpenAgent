#!/bin/bash

# OpenAgent Development Environment Setup Script
# Usage: ./scripts/dev.sh [command] [options]

set -e

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCKER_COMPOSE_FILE="$PROJECT_ROOT/docker-compose.yml"
PYTHON_VERSION="3.11"
VENV_DIR="$PROJECT_ROOT/venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
OpenAgent Development Environment Manager

Usage: $0 [COMMAND] [OPTIONS]

Commands:
  setup           Set up development environment
  start           Start development server
  stop            Stop all services
  restart         Restart services
  test            Run test suite
  lint            Run linting and formatting
  build           Build Docker images
  clean           Clean up development environment
  logs            Show service logs
  shell           Open shell in container
  db-reset        Reset database
  backup          Create backup of data
  restore         Restore from backup

Options:
  --production    Use production configuration
  --monitoring    Include monitoring stack
  --logging       Include logging stack
  --help          Show this help message

Examples:
  $0 setup                    # Set up development environment
  $0 start --monitoring       # Start with monitoring
  $0 test --unit              # Run unit tests only
  $0 logs openagent-api       # Show API server logs
  $0 shell openagent-api      # Open shell in API container

EOF
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    local deps=("docker" "docker-compose" "python3" "git")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing dependencies: ${missing_deps[*]}"
        log_info "Please install the missing dependencies and try again."
        exit 1
    fi
    
    log_success "All dependencies are installed"
}

# Set up development environment
setup_environment() {
    log_info "Setting up development environment..."
    
    # Check dependencies
    check_dependencies
    
    # Create directories
    mkdir -p "$PROJECT_ROOT/data"
    mkdir -p "$PROJECT_ROOT/logs"
    mkdir -p "$PROJECT_ROOT/models"
    mkdir -p "$PROJECT_ROOT/backups"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv "$VENV_DIR"
    fi
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip and install dependencies
    log_info "Installing Python dependencies..."
    pip install --upgrade pip
    
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements.txt"
    fi
    
    if [ -f "$PROJECT_ROOT/requirements-dev.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements-dev.txt"
    fi
    
    # Install package in development mode
    pip install -e "$PROJECT_ROOT"
    
    # Create .env file if it doesn't exist
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        log_info "Creating .env file template (you must edit secrets before use)..."
        cat > "$PROJECT_ROOT/.env" << 'EOF'
# OpenAgent Environment Configuration

# Authentication
# Set AUTH_ENABLED=true and provide a secure JWT_SECRET_KEY in production
AUTH_ENABLED=false
# JWT_SECRET_KEY should be set to a secure random string in production
# example: JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_urlsafe(32))")
JWT_SECRET_KEY=
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Rate Limiting
RATE_LIMIT_ENABLED=false

# Database
# For local development, SQLite is recommended; to use Postgres, set DATABASE_URL
DATABASE_URL=sqlite:///./data/openagent.db
POSTGRES_DB=openagent
POSTGRES_USER=openagent
POSTGRES_PASSWORD=

# Monitoring
GRAFANA_USER=admin
GRAFANA_PASSWORD=

# Development
DEVELOPMENT=true
LOG_LEVEL=DEBUG
EOF
        log_success "Created .env template at .env - edit and set secrets before starting services"
    fi
    
    # Pre-commit hooks
    if command -v pre-commit &> /dev/null; then
        log_info "Installing pre-commit hooks..."
        pre-commit install
    fi
    
    log_success "Development environment set up successfully"
    log_info "To activate the virtual environment, run: source venv/bin/activate"
    log_info "To start the development server, run: $0 start"
}

# Start development services
start_services() {
    local profiles=("dev")
    
    # Parse additional profiles
    while [[ $# -gt 0 ]]; do
        case $1 in
            --production)
                profiles=("production")
                shift
                ;;
            --monitoring)
                profiles+=("monitoring")
                shift
                ;;
            --logging)
                profiles+=("logging")
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    log_info "Starting OpenAgent services with profiles: ${profiles[*]}"
    
    # Build profile arguments
    profile_args=""
    for profile in "${profiles[@]}"; do
        profile_args="$profile_args --profile $profile"
    done
    
    # Start services
    docker-compose -f "$DOCKER_COMPOSE_FILE" $profile_args up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 5
    
    # Check health
    check_service_health
    
    log_success "Services started successfully"
    show_service_urls
}

# Stop services
stop_services() {
    log_info "Stopping OpenAgent services..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" down
    log_success "Services stopped"
}

# Restart services
restart_services() {
    log_info "Restarting OpenAgent services..."
    stop_services
    start_services "$@"
}

# Check service health
check_service_health() {
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f http://localhost:8000/health > /dev/null 2>&1; then
            log_success "API server is healthy"
            return 0
        fi
        
        log_info "Waiting for API server... (attempt $attempt/$max_attempts)"
        sleep 2
        ((attempt++))
    done
    
    log_warning "API server health check failed"
    return 1
}

# Show service URLs
show_service_urls() {
    log_info "Service URLs:"
    echo "  🌐 API Server: http://localhost:8000"
    echo "  📚 API Docs: http://localhost:8000/docs"
    echo "  🔍 Health Check: http://localhost:8000/health"
    
    # Check if monitoring services are running
    if docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "grafana"; then
        echo "  📊 Grafana: http://localhost:3000 (admin/admin123)"
        echo "  📈 Prometheus: http://localhost:9090"
    fi
    
    # Check if logging services are running
    if docker-compose -f "$DOCKER_COMPOSE_FILE" ps | grep -q "kibana"; then
        echo "  🔍 Kibana: http://localhost:5601"
    fi
}

# Run tests
run_tests() {
    log_info "Running tests..."
    
    # Activate virtual environment if it exists
    if [ -d "$VENV_DIR" ]; then
        source "$VENV_DIR/bin/activate"
    fi
    
    # Run tests based on arguments
    case "${1:-all}" in
        unit)
            pytest tests/unit/ -v
            ;;
        integration)
            pytest tests/integration/ -v
            ;;
        api)
            # Start services for API testing
            start_services
            sleep 10
            pytest tests/api/ -v
            stop_services
            ;;
        all|*)
            pytest tests/ -v --cov=openagent --cov-report=term-missing --cov-report=html
            ;;
    esac
    
    log_success "Tests completed"
}

# Run linting and formatting
run_lint() {
    log_info "Running linting and formatting..."
    
    # Activate virtual environment if it exists
    if [ -d "$VENV_DIR" ]; then
        source "$VENV_DIR/bin/activate"
    fi
    
    # Format with black
    log_info "Formatting with black..."
    black openagent/ tests/
    
    # Sort imports with isort
    log_info "Sorting imports with isort..."
    isort openagent/ tests/
    
    # Lint with flake8
    log_info "Linting with flake8..."
    flake8 openagent/ tests/
    
    # Type checking with mypy
    log_info "Type checking with mypy..."
    mypy openagent/ --ignore-missing-imports
    
    log_success "Linting completed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    docker-compose -f "$DOCKER_COMPOSE_FILE" build
    
    log_success "Docker images built successfully"
}

# Show logs
show_logs() {
    local service="${1:-}"
    
    if [ -n "$service" ]; then
        log_info "Showing logs for $service..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f "$service"
    else
        log_info "Showing logs for all services..."
        docker-compose -f "$DOCKER_COMPOSE_FILE" logs -f
    fi
}

# Open shell in container
open_shell() {
    local service="${1:-openagent-api}"
    
    log_info "Opening shell in $service..."
    docker-compose -f "$DOCKER_COMPOSE_FILE" exec "$service" /bin/bash
}

# Clean up development environment
cleanup() {
    log_info "Cleaning up development environment..."
    
    # Stop and remove containers
    docker-compose -f "$DOCKER_COMPOSE_FILE" down -v
    
    # Remove images
    docker-compose -f "$DOCKER_COMPOSE_FILE" down --rmi all
    
    # Clean up Python cache
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true
    
    # Remove virtual environment
    if [ -d "$VENV_DIR" ]; then
        log_warning "Removing virtual environment..."
        rm -rf "$VENV_DIR"
    fi
    
    log_success "Cleanup completed"
}

# Main command dispatcher
main() {
    local command="${1:-help}"
    shift 2>/dev/null || true
    
    case "$command" in
        setup)
            setup_environment "$@"
            ;;
        start)
            start_services "$@"
            ;;
        stop)
            stop_services "$@"
            ;;
        restart)
            restart_services "$@"
            ;;
        test)
            run_tests "$@"
            ;;
        lint)
            run_lint "$@"
            ;;
        build)
            build_images "$@"
            ;;
        logs)
            show_logs "$@"
            ;;
        shell)
            open_shell "$@"
            ;;
        clean)
            cleanup "$@"
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            log_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
