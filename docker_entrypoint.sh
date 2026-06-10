#!/bin/bash
# Container entrypoint that downloads models from GCS before starting the API server.
#
# Optional environment variables:
#   BARRIKADA_GCS_BUCKET: GCS bucket name (must be publicly readable)
#
# If valid local models already exist under /app/core/models, the container uses
# them as-is and skips the GCS download.

set -euo pipefail

MODELS_DIR="/app/core/models"
LOG_PREFIX="[BARRIKADA INIT]"
DEFAULT_GCS_BUCKET="barrikade-bundles"

log_info() {
    echo "$LOG_PREFIX INFO: $1"
}

log_warn() {
    echo "$LOG_PREFIX WARN: $1" >&2
}

log_error() {
    echo "$LOG_PREFIX ERROR: $1" >&2
}

has_local_models() {
    if [ ! -d "$MODELS_DIR" ]; then
        return 1
    fi

    if find "$MODELS_DIR" -type f ! -path "*/archives/*" | grep -q .; then
        return 0
    fi

    return 1
}

# Download models from GCS
download_models_from_gcs() {
    local bucket="${BARRIKADA_GCS_BUCKET:-$DEFAULT_GCS_BUCKET}"

    log_info "Downloading models from GCS (bucket: $bucket)..."

    rm -rf "$MODELS_DIR"
    mkdir -p "$MODELS_DIR"

    cd /app
    if python -m scripts.bundling.gcs_download \
        --bucket "$bucket" \
        --no-archive-old \
        --validate \
        2>&1; then
        log_info "Models downloaded successfully from GCS"
        return 0
    else
        log_error "Failed to download models from GCS"
        return 1
    fi
}

# Validate downloaded models
validate_models() {
    log_info "Validating local models..."

    if cd /app && python -m scripts.bundling.validate_models --verbose; then
        log_info "Models validation successful"
        return 0
    fi

    log_error "Model validation failed"
    return 1
}

# Main logic
main() {
    log_info "Starting Barrikada container initialization..."
    log_info "Models directory: $MODELS_DIR"

    if has_local_models; then
        log_info "Existing local models detected, validating before startup..."
        if validate_models; then
            log_info "Using existing local models"
        else
            log_warn "Existing local models are invalid, re-downloading from GCS"
            if ! download_models_from_gcs; then
                log_error "Failed to initialize models from GCS"
                exit 1
            fi
            if ! validate_models; then
                exit 1
            fi
        fi
    else
        log_info "No local models found, downloading from GCS..."
        if ! download_models_from_gcs; then
            log_error "Failed to initialize models from GCS"
            exit 1
        fi
        if ! validate_models; then
            exit 1
        fi
    fi

    log_info "Initialization complete, starting API server..."
    
    # Execute the main command
    exec "$@"
}

main "$@"
