"""D4-SS: does crop position degrade a residue-level task (secondary structure)?

Pipeline:
  1) Linear probe (LogReg, Q3) on frozen ESM-2 full-context embeddings (train proteins).
  2) Reference Q3 accuracy on test proteins (full context).
  3) Position-stratified eval: place each non-terminal target residue at rel-pos
     {0.1..0.9} of a W=256 window; predict SS; accuracy by window position.
     D4 = center accuracy > edge accuracy (position gap), within-residue paired bootstrap.
  4) Fix: symmetric multi-crop (avg probe class-probs over placements) vs edge.

Usage: python scripts/phase1/d4_ss_run.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from calibrate.models import load_esm2
from calibrate.seed import set_global_seed
from calibrate.stats import paired_bootstrap_ci

DATA = (_ROOT / "data/processed/d4_ss")
OUT = (_ROOT / "experiments/phase1_diagnosis/d4_ss")
FIG = (_ROOT / "results/figures/d4_ss_position.png")
Q3 = "HEC"
CLS = {c: i for i, c in enumerate(Q3)}
W = 256
REL_POS = [0.1, 0.3, 0.5, 0.7, 0.9]
MARGIN = round(0.9 * W)
N_TRAIN_PROT = 200
MAX_TRAIN_RES = 40000
TARGETS_PER_PROT = 6


def load(split):
    return [json.loads(l) for l in (DATA / f"{split}.jsonl").read_text().splitlines()]


def main() -> None:
    set_global_seed(42)
    rng = np.random.default_rng(42)
    OUT.mkdir(parents=True, exist_ok=True)
    model = load_esm2()
    train, test = load("train"), load("test")

    # 1) probe on full-context embeddings
    X, y = [], []
    for p in tqdm(train[:N_TRAIN_PROT], desc="embed train"):
        emb = model.embed([p["seq"]])[0]
        X.append(emb); y.extend(CLS[c] for c in p["labels"])
    X = np.concatenate(X); y = np.asarray(y)
    if len(y) > MAX_TRAIN_RES:
        idx = rng.choice(len(y), MAX_TRAIN_RES, replace=False); X, y = X[idx], y[idx]
    probe = LogisticRegression(max_iter=2000, C=1.0).fit(X, y)
    order = list(probe.classes_)
    print(f"probe fit on {len(y)} residues")

    # 2) reference Q3 on test (full context)
    cor = tot = 0
    for p in test:
        emb = model.embed([p["seq"]])[0]
        pred = probe.predict(emb)
        truth = np.array([CLS[c] for c in p["labels"]])
        cor += int((pred == truth).sum()); tot += len(truth)
    ref_q3 = cor / tot
    print(f"reference full-context Q3 accuracy (test) = {ref_q3:.3f}")

    # 3) position-stratified eval
    rows = []
    for p in tqdm(test, desc="position eval"):
        seq, lab = p["seq"], p["labels"]
        L = len(seq)
        if L < 2 * MARGIN + 1:
            continue
        cand = list(range(MARGIN, L - MARGIN))
        picks = rng.choice(cand, min(TARGETS_PER_PROT, len(cand)), replace=False)
        for t in picks:
            true_c = CLS[lab[t]]
            probas = {}
            for r in REL_POS:
                s = t - round(r * W)
                emb = model.embed([seq[s : s + W]])[0]
                pr = probe.predict_proba(emb[t - s][None])[0]
                probas[r] = pr
            rows.append(dict(acc_target=true_c, **{f"p_{r}": probas[r] for r in REL_POS}))

    # 4) analyze: per-target correctness by placement + multicrop fix
    def correct(prob_vec, true_c):
        return int(order[int(np.argmax(prob_vec))] == true_c)

    center = np.array([correct(row["p_0.5"], row["acc_target"]) for row in rows])
    edge = np.array([(correct(row["p_0.1"], row["acc_target"]) +
                      correct(row["p_0.9"], row["acc_target"])) / 2 for row in rows])
    mc = []
    for row in rows:
        avg = np.mean([row[f"p_{r}"] for r in REL_POS], axis=0)
        mc.append(correct(avg, row["acc_target"]))
    mc = np.array(mc)

    res = paired_bootstrap_ci(center.astype(float), edge, seed=42)
    accs = {r: np.mean([correct(row[f"p_{r}"], row["acc_target"]) for row in rows]) for r in REL_POS}
    print(f"\n=== D4-SS position eval (n={len(rows)} target residues, W={W}) ===")
    print("  Q3 accuracy by rel_pos:", {r: round(float(a), 3) for r, a in accs.items()})
    print(f"  center(0.5)={center.mean():.3f}  edge(0.1,0.9)={edge.mean():.3f}  "
          f"gap=+{res['diff']*100:.2f}pp CI=[{res['ci_low']*100:.2f},{res['ci_high']*100:.2f}] p={res['p_value']:.3f}")
    fix = paired_bootstrap_ci(mc.astype(float), edge, seed=42)
    print(f"  multicrop={mc.mean():.3f}  multicrop-edge=+{fix['diff']*100:.2f}pp (p={fix['p_value']:.3f})")

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    ax.plot(REL_POS, [accs[r] for r in REL_POS], marker="o")
    ax.set(title="D4-SS: secondary-structure Q3 accuracy vs window position",
           xlabel="target residue rel_pos in window", ylabel="Q3 accuracy")
    fig.tight_layout(); FIG.parent.mkdir(parents=True, exist_ok=True); fig.savefig(FIG, dpi=130)
    print(f"figure -> {FIG}")

    # persist a summary (for T1.D scoping consolidation; uniform with VEP/contacts)
    summary = dict(
        task="secondary_structure_Q3", n=len(rows), W=W, rel_pos=REL_POS,
        reference_full_q3=float(ref_q3),
        accuracy_by_rel={float(r): float(accs[r]) for r in REL_POS},
        center=float(center.mean()), edge=float(edge.mean()),
        gap_pp=float(res["diff"] * 100),
        gap_ci_pp=[float(res["ci_low"] * 100), float(res["ci_high"] * 100)],
        gap_p=float(res["p_value"]),
        multicrop=float(mc.mean()), multicrop_minus_edge_pp=float(fix["diff"] * 100),
        multicrop_p=float(fix["p_value"]),
    )
    (OUT / "ss_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary -> {OUT/'ss_summary.json'}")


if __name__ == "__main__":
    main()
