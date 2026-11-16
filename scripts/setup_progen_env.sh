#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
export HF_HOME=$ROOT/data/cache/models/hf

echo "[1/4] remove partial progen env (if any)"
conda env remove -n progen -y 2>/dev/null || true
echo "[2/4] clone calibrate -> progen (local copy, reuses GPU torch, no big download)"
conda create -n progen --clone calibrate -y
echo "[3/4] downgrade transformers to 4.44 (small download)"
conda run -n progen pip install "transformers==4.44.2" "tokenizers<0.20"
echo "[4/4] verify ProGen2 load on GPU (causal RoPE)"
conda run --no-capture-output -n progen python -u - <<'PY'
import torch, traceback
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo="hugohrban/progen2-base"
    tok=AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    m=AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True).eval()
    n=sum(p.numel() for p in m.parameters())
    print(f"PROGEN2_OK params={n/1e6:.1f}M transformers_ok cuda={torch.cuda.is_available()}")
except Exception as e:
    print("PROGEN2_FAILED:", repr(e)); traceback.print_exc()
PY
echo "PROGEN_ENV_DONE"
