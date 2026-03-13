#!/bin/bash
# Shared utilities for training scripts
# Source this file: source "$(dirname "$0")/utils.sh"

# Find GPU with most free memory
find_best_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Error: nvidia-smi not found." >&2
        exit 1
    fi

    local best_gpu=0
    local max_free=0

    while IFS=',' read -r gpu_id name used total; do
        gpu_id=$(echo "$gpu_id" | tr -d ' ')
        used=$(echo "$used" | tr -d ' ')
        total=$(echo "$total" | tr -d ' ')

        if [[ "$gpu_id" =~ ^[0-9]+$ ]] && [[ "$used" =~ ^[0-9]+$ ]] && [[ "$total" =~ ^[0-9]+$ ]]; then
            free=$((total - used))
            if [ $free -gt $max_free ]; then
                max_free=$free
                best_gpu=$gpu_id
            fi
        fi
    done < <(nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)

    if [ $max_free -eq 0 ]; then
        echo "Warning: Could not parse GPU info, defaulting to GPU 0" >&2
        best_gpu=0
    fi

    echo $best_gpu
}

# Get GPU info string for display
get_gpu_info() {
    local gpu_id=$1
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free \
        --format=csv,noheader,nounits | sed -n "$((gpu_id + 1))p" 2>/dev/null \
        || echo "$gpu_id, Unknown GPU, -, -, -"
}

# Process wandb parameter: true/false -> enable/disable wandb logging
process_wandb() {
    local wandb=${1:-false}
    if [[ "$wandb" == "true" || "$wandb" == "True" || "$wandb" == "yes" ]]; then
        echo "False"
    else
        echo "True"
    fi
}

# Print a colored config line: key=value
print_config() {
    local key=$1
    local value=$2
    echo -e "\033[1;33m${key}:\033[0m \033[1;32m${value}\033[0m"
}

# Print section header
print_header() {
    local title=$1
    echo -e "\033[1;36m=== ${title} ===\033[0m"
}

# Print warning
print_warn() {
    local msg=$1
    echo -e "\033[1;33mWARNING:\033[0m ${msg}"
}

# Print launch message
print_launch() {
    local msg=$1
    echo -e "\033[1;32m${msg}\033[0m"
}

# Validate file exists
validate_file() {
    local path=$1
    local label=$2
    if [ ! -f "$path" ]; then
        echo -e "\033[1;31mError:\033[0m ${label} does not exist: ${path}"
        exit 1
    fi
}
