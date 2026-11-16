#!/usr/bin/env bash
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(conda info --base)/etc/profile.d/conda.sh"
export HF_HOME=$ROOT/data/cache/models/hf

echo "[1/4] remove partial esmc env"
conda env remove -n esmc -y 2>/dev/null || true
echo "[2/4] clone calibrate -> esmc (reuse GPU torch)"
conda create -n esmc --clone calibrate -y
echo "[3/4] swap fair-esm -> EvolutionaryScale esm SDK (namespace 'esm')"
conda run -n esmc pip uninstall -y fair-esm
conda run -n esmc pip install esm
echo "[4/4] verify ESM C 300M load + masked logits"
conda run --no-capture-output -n esmc python -u - <<'PY'
import torch, traceback
try:
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig
    m = ESMC.from_pretrained("esmc_300m").to("cuda").eval()
    p = ESMProtein(sequence="MKTAYIAKQRQISFVKSHF")
    t = m.encode(p)
    lc = m.logits(t, LogitsConfig(sequence=True))
    print("ESMC_OK logits shape", tuple(lc.logits.sequence.shape))
except Exception as e:
    print("ESMC_FAILED:", repr(e)); traceback.print_exc()
PY
echo "ESMC_ENV_DONE"
