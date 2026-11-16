#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TORCH_HOME=$ROOT/data/cache/models/torch
export HF_HOME=$ROOT/data/cache/models/hf
mkdir -p "$TORCH_HOME" "$HF_HOME"
source "$(conda info --base)/etc/profile.d/conda.sh"

echo "=== [1] ESM-2 650M (fair-esm) ==="
conda run --no-capture-output -n calibrate python -u - <<'PY'
import os, torch, esm, numpy as np
print("TORCH_HOME=", os.environ.get("TORCH_HOME"))
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model = model.eval().half().cuda()
bc = alphabet.get_batch_converter()
_,_,toks = bc([("p","MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ")])
with torch.no_grad():
    out = model(toks.cuda(), repr_layers=[33], need_head_weights=False)
rep = out["representations"][33]
print("ESM2_OK repr shape:", tuple(rep.shape), "finite:", bool(np.isfinite(rep.float().sum().item())))
print("mem after ESM2 (GiB free/total):", [round(v/1024**3,2) for v in torch.cuda.mem_get_info()])
PY
echo "ESM2_EXIT=$?"

echo "=== [2] ProGen2-764M (base) via transformers ==="
conda run --no-capture-output -n calibrate python -u - <<'PY'
import os, traceback
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    repo = "hugohrban/progen2-base"
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(repo, trust_remote_code=True)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"PROGEN2_OK params={nparams/1e6:.1f}M repo={repo}")
except Exception as e:
    print("PROGEN2_FAILED:", repr(e))
    traceback.print_exc()
PY
echo "PROGEN2_EXIT=$?"
echo "DOWNLOAD_DONE"
