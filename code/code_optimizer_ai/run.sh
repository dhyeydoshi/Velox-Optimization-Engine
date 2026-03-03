#!/bin/bash

# Velox Optimization Engine Run Script
# Main entry point for the application

set -e

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
PYTHON_PATH="${SCRIPT_DIR}:${PYTHON_PATH}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
DEBUG="${DEBUG:-false}"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Check if running in development mode
if [ "$DEBUG" = "true" ]; then
    log_info "Running in DEBUG mode"
    export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"
else
    log_info "Running in production mode"
fi

# Create necessary directories
mkdir -p /app/data /app/models /app/logs

# Set up environment
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Wait for dependencies if in production
if [ "$DEBUG" != "true" ]; then
    log_info "Waiting for dependencies..."
    
    # Wait for database
    if [ -n "$DATABASE_URL" ]; then
        wait-for-it $(echo $DATABASE_URL | sed 's/.*@\([^:]*\):.*/\1/') -t 60 || {
            log_warn "Database not ready, continuing anyway..."
        }
    fi
    
    # Wait for Redis
    if [ -n "$REDIS_URL" ]; then
        wait-for-it $(echo $REDIS_URL | sed 's/.*@\([^:]*\):.*/\1/') -t 30 || {
            log_warn "Redis not ready, continuing anyway..."
        }
    fi
fi

# Function to wait for a service to be ready
wait-for-it() {
    local hostport=$1
    local timeout=${2:-30}
    local service=$(echo $hostport | cut -d: -f1)
    local port=$(echo $hostport | cut -d: -f2)
    
    log_info "Waiting for $service:$port to be ready..."
    
    while ! nc -z $service $port 2>/dev/null; do
        sleep 1
        timeout=$((timeout - 1))
        if [ $timeout -eq 0 ]; then
            log_warn "Timeout waiting for $service:$port"
            return 1
        fi
    done
    
    log_info "$service:$port is ready"
    return 0
}

# Run database migrations if needed
if [ "$DEBUG" != "true" ]; then
    log_info "Running database migrations..."
    python -c "from code.code_optimizer_ai.database.migrate import create_database_schema; import asyncio; asyncio.run(create_database_schema())" || log_warn "Migration failed, continuing anyway..."
fi

# Start the application
log_info "Starting Velox Optimization Engine..."

if [ "$DEBUG" = "true" ]; then
    # Development mode - run with auto-reload
    exec uvicorn api_main:app --host 0.0.0.0 --port 8000 --reload --log-level ${LOG_LEVEL,,}
else
    # Production mode - run with proper configuration
    exec uvicorn api_main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --workers 4 \
        --log-level ${LOG_LEVEL,,} \
        --access-log \
        --proxy-headers \
        --forwarded-allow-ips='*'
fi
