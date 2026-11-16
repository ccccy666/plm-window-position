#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME=$ROOT/data/cache/models/hf
source "$(conda info --base)/etc/profile.d/conda.sh"
echo "downloading SaProt 650M weights (HF, resume-able)..."
conda run --no-capture-output -n calibrate python -u -c "
from huggingface_hub import snapshot_download
p = snapshot_download('westlake-repl/SaProt_650M_AF2')
print('SAPROT_DOWNLOADED', p)
import os
print('files:', sorted(os.listdir(p))[:20])
"
echo "SAPROT_DL_DONE"
