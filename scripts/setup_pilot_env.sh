#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "[1/5] creating conda env calibrate (python=3.11)"
conda create -n calibrate python=3.11 -y

echo "[2/5] upgrading pip"
conda run -n calibrate python -m pip install --upgrade pip

echo "[3/5] installing torch (cu128 / Blackwell sm_120)"
conda run -n calibrate pip install torch --index-url https://download.pytorch.org/whl/cu128

echo "[4/5] verifying GPU"
conda run -n calibrate python -c "import torch; print('torch', torch.__version__); print('cuda_available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'); print('capability', torch.cuda.get_device_capability(0) if torch.cuda.is_available() else 'NONE')"

echo "[5/5] installing pilot deps"
conda run -n calibrate pip install -r $ROOT/requirements-pilot.txt
conda run -n calibrate python -c "import esm, transformers, Bio, sklearn, pandas, scipy, matplotlib, seaborn, statsmodels, h5py, datasets; print('IMPORTS_OK')"

echo "ENV_SETUP_DONE"
