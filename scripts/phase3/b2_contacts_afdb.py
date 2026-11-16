"""Camera-ready group B2 (no GPU): are the model's confident contacts real contacts in the AlphaFold structure?

PRE-REGISTERED (fixed 2026-11-16 before downloading any structure; do not change after seeing results)
-------------------------------------------------------------------------------------------------------
Pairs    : the 1178 confident long-range contacts of the partner-out probe
           (experiments/phase3/b1_contacts_partner_out/pairs.parquet; ESM-2 full-context probability > 0.5,
           separation 30-200; 99 proteins).
Structure: the AlphaFold DB model (latest version, via the AFDB API) of each protein, used only if its
           sequence equals the sequence we scored; otherwise the protein is excluded.
Label    : contact iff C-beta distance (C-alpha for Gly) < 8 A and both residues have pLDDT >= 70;
           non-contact iff the distance is >= 8 A with both pLDDT >= 70; otherwise unverifiable (excluded).
Report   : precision = contact / (contact + non-contact); the partner-out result (center vs partner flush at the
           window edge, paired bootstrap) and the tiling split rates, on verified contacts vs all pairs.
Decision : the paper may describe the partner-out loss as a loss on structure-verified contacts iff
           precision >= 0.80 AND the center - partner_edge gap on verified contacts is > 0 at p < 0.01.
Not covered: the 444-pair position sweep of Section 6.1 (stored without residue identities).

Usage    : python scripts/phase3/b2_contacts_afdb.py [--fetch]   (calibrate env; --fetch downloads missing models,
           one request at a time)
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = _ROOT
sys.path.insert(0, str(ROOT / "src"))
from calibrate import structure as st  # noqa: E402
from calibrate.stats import paired_bootstrap_ci  # noqa: E402

PAIRS = ROOT / "experiments/phase3/b1_contacts_partner_out/pairs.parquet"
SEQS = ROOT / "data/processed/d4_ss/test.jsonl"
AFDB = ROOT / "data/raw/afdb"
OUT = ROOT / "experiments/phase3/b2_contacts_afdb"
API = "https://alphafold.ebi.ac.uk/api/prediction/{}"
PRECISION_BAR, P_BAR = 0.80, 0.01
GRID = [(w, name, frac) for w in (128, 256, 384)
        for name, frac in (("non-overlap", 1.0), ("50% overlap", 0.5), ("75% overlap", 0.25))]


def get(url, timeout=60, tries=4):
    import requests
    for k in range(tries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001  (network: retry with backoff, then give up)
            if k == tries - 1:
                raise
            print(f"    retry {k + 1} ({type(e).__name__})", flush=True)
            time.sleep(5 * (k + 1))


def fetch(seqs):
    AFDB.mkdir(parents=True, exist_ok=True)
    status = {}
    for n, (acc, seq) in enumerate(sorted(seqs.items()), 1):
        pdb, meta = AFDB / f"AF-{acc}.pdb", AFDB / f"AF-{acc}.json"
        if pdb.exists() and meta.exists():
            status[acc] = "cached"
            continue
        try:
            entries = get(API.format(acc)).json()
            e = next((x for x in entries if x.get("modelEntityId", "").endswith("-F1")), entries[0])
            meta.write_text(json.dumps({k: e.get(k) for k in ("modelEntityId", "latestVersion", "sequence",
                                                                "pdbUrl", "globalMetricValue")}))
            if e.get("sequence") != seq:
                status[acc] = "sequence mismatch"
            else:
                pdb.write_text(get(e["pdbUrl"]).text)
                status[acc] = "downloaded"
        except Exception as ex:  # noqa: BLE001
            status[acc] = f"failed: {type(ex).__name__}"
        print(f"  [{n}/{len(seqs)}] {acc}: {status[acc]}", flush=True)
        time.sleep(0.5)
    return status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    pairs = pd.read_parquet(PAIRS)
    seqs = {}
    for line in SEQS.read_text().splitlines():
        r = json.loads(line)
        seqs[r["acc"]] = r["seq"]
    seqs = {a: s for a, s in seqs.items() if a in set(pairs.prot)}
    print(f"{len(pairs)} pairs in {len(seqs)} proteins", flush=True)
    if args.fetch:
        status = fetch(seqs)
        print("fetch:", pd.Series(status).value_counts().to_dict(), flush=True)

    labels = pd.Series("no structure", index=pairs.index, dtype=object)
    for acc, g in pairs.groupby("prot"):
        pdb = AFDB / f"AF-{acc}.pdb"
        if not pdb.exists():
            continue
        s = st.parse_pdb_cb(pdb.read_text())
        if s["seq"] != seqs[acc]:
            labels[g.index] = "sequence mismatch"
            continue
        labels[g.index] = st.label_pairs(s["coords"], s["plddt"], g.i.to_numpy(), g.j.to_numpy())
    pairs["label"] = labels
    counts = pairs.label.value_counts().to_dict()
    n_c, n_nc = counts.get("contact", 0), counts.get("non-contact", 0)
    precision = n_c / (n_c + n_nc) if (n_c + n_nc) else float("nan")

    def partner_out(d):
        r = paired_bootstrap_ci(d.center.values, d.partner_edge.values, seed=42)
        return dict(n=len(d), full=float(d.prob_full.mean()), center=float(d.center.mean()),
                    partner_edge=float(d.partner_edge.mean()), gap_pp=r["diff"] * 100,
                    ci=[r["ci_low"] * 100, r["ci_high"] * 100], p=r["p_value"],
                    frac_degrading=float((d.partner_edge < d.center).mean()))

    def tiling(d):
        return {f"W{w} {name}": float(st.tiling_split(d.i.to_numpy(), d.j.to_numpy(), w,
                                                      max(1, int(round(w * frac)))).mean())
                for w, name, frac in GRID}

    ver = pairs[pairs.label == "contact"]
    po_all, po_ver = partner_out(pairs), partner_out(ver)
    decision = ("structure-verified wording justified"
                if precision >= PRECISION_BAR and po_ver["gap_pp"] > 0 and po_ver["p"] < P_BAR
                else "keep the self-consistency wording")
    summary = dict(prereg=dict(precision_bar=PRECISION_BAR, p_bar=P_BAR, cutoff_A=8.0, min_plddt=70.0),
                   label_counts=counts, precision=precision,
                   n_proteins_with_structure=int(pairs[pairs.label != "no structure"].prot.nunique()),
                   partner_out=dict(all=po_all, verified=po_ver),
                   tiling=dict(all=tiling(pairs), verified=tiling(ver)), decision=decision)
    OUT.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(OUT / "pairs_labeled.parquet")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"labels: {counts}  precision={precision:.3f}")
    for k, v in (("all", po_all), ("verified", po_ver)):
        print(f"partner-out {k:8s} n={v['n']:4d} full {v['full']:.3f} center {v['center']:.3f} edge "
              f"{v['partner_edge']:.3f} gap {v['gap_pp']:+.1f}pp CI[{v['ci'][0]:+.1f},{v['ci'][1]:+.1f}] "
              f"p={v['p']:.4f} degrading {v['frac_degrading'] * 100:.0f}%")
    for k in ("W256 non-overlap", "W256 50% overlap", "W128 non-overlap", "W128 75% overlap"):
        print(f"tiling {k:18s} all {summary['tiling']['all'][k] * 100:5.1f}%  "
              f"verified {summary['tiling']['verified'][k] * 100:5.1f}%")
    print(f"decision: {decision}  -> {OUT}")


if __name__ == "__main__":
    main()
