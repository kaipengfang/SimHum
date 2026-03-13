#!/usr/bin/env bash
# setup_env.sh — Create and setup conda environment
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

ENV_NAME="H_data_col"
PYTHON_VERSION="3.11"

echo "[setup_env] Project directory: ${PROJECT_DIR}"

# Check if conda is available
if ! command -v conda &>/dev/null; then
    echo "[setup_env] Error: conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[setup_env] Environment '${ENV_NAME}' already exists."
    read -r -p "[setup_env] Recreate it? [y/N] " answer
    case "${answer}" in
        [yY]|[yY][eE][sS])
            echo "[setup_env] Removing existing environment..."
            conda env remove -n "${ENV_NAME}" -y
            ;;
        *)
            echo "[setup_env] Skipping creation, installing dependencies only..."
            eval "$(conda shell.bash hook 2>/dev/null)"
            conda activate "${ENV_NAME}"
            pip install --only-binary=:all: av
            pip install --only-binary=av -r "${PROJECT_DIR}/requirements.txt"
            echo "[setup_env] Done! Use 'conda activate ${ENV_NAME}' to activate."
            exit 0
            ;;
    esac
fi

echo "[setup_env] Creating conda environment '${ENV_NAME}' with Python ${PYTHON_VERSION}..."
conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate "${ENV_NAME}"

echo "[setup_env] Step 1/2: Installing av (pre-built binary)..."
pip install --only-binary=:all: av

echo "[setup_env] Step 2/2: Installing remaining dependencies..."
pip install --only-binary=av -r "${PROJECT_DIR}/requirements.txt"

echo ""
echo "========================================"
echo "[setup_env] Done!"
echo "  Environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"
echo "  Activate:    conda activate ${ENV_NAME}"
echo "========================================"
