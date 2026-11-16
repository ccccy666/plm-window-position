"""Reviewer-3 RC, step 1 (no GPU): fetch curated domain boundaries and build target sets.

Parses accessions from the fidelity-probe pool, fetches UniProt curated `Domain` features per
accession (cached to disk to survive flaky networks), and builds two target sets for the
domain-aware windowing ablation (W=256, MARGIN=230, same geometry as the taxonomy probe):

  containable : target = midpoint of a curated domain D with |D| <= W-20 (so a width-W window can
                fully contain D), target non-terminal. Used to compare position-only edge/center vs a
                domain-aware window that retains the residue's whole domain.
  large_domain: target = midpoint of a curated domain D large enough (>= 2*W) that EVERY window
                placement (edge and center) stays entirely inside D. Used to test whether the
                center-edge effect persists with NO domain boundary crossed (concern 1).

Writes rebuttal/reviewer3/targets.json (+ domains_cache.json).
Usage: python rebuttal/reviewer3/r3_fetch_domains.py
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = _ROOT
SP = ROOT / "data/processed/p1_0/swissprot_sample.fasta"
CACHE = ROOT / "rebuttal/reviewer3/domains_cache.json"
OUT = ROOT / "rebuttal/reviewer3/targets.json"

W = 256
MARGIN = round(0.9 * W)          # 230, same as t1a_fidelity
AA = set("ACDEFGHIKLMNPQRSTVWY")
MAX_FETCH = 0                    # 0 = build from the committed domains_cache.json (no network);
                                 # raise this on a fresh machine to fetch uncached accessions


def parse_pool():
    seqs, acc, cur, a = [], [], [], None
    for l in SP.read_text().splitlines():
        if l.startswith(">"):
            if cur:
                seqs.append("".join(cur)); acc.append(a); cur = []
            m = re.match(r">\w+\|([^|]+)\|", l); a = m.group(1) if m else None
        elif l.strip():
            cur.append(l.strip())
    if cur:
        seqs.append("".join(cur)); acc.append(a)
    return [(x, s) for x, s in zip(acc, seqs) if x and set(s) <= AA and len(s) >= 461]


def fetch_domains(acc, cache):
    if acc in cache:
        return cache[acc]
    url = f"https://rest.uniprot.org/uniprotkb/{acc}.json?fields=ft_domain"
    try:
        d = json.load(urllib.request.urlopen(url, timeout=12))
        doms = []
        for f in d.get("features", []):
            if f.get("type") == "Domain":
                s = f["location"]["start"]["value"]; e = f["location"]["end"]["value"]
                if isinstance(s, int) and isinstance(e, int):
                    doms.append([s - 1, e - 1])   # 0-based inclusive
        cache[acc] = doms
    except Exception as ex:
        cache[acc] = None
        print(f"  fetch fail {acc}: {str(ex)[:40]}")
    return cache[acc]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    pool = parse_pool()
    print(f"{len(pool)} long candidate proteins")

    containable, large_domain = [], []
    n_fetched = 0
    for acc, seq in pool:
        if acc not in cache and n_fetched >= MAX_FETCH:
            continue
        if acc not in cache:
            fetch_domains(acc, cache); n_fetched += 1
            if n_fetched % 25 == 0:
                CACHE.write_text(json.dumps(cache)); time.sleep(0.2)
        doms = cache.get(acc)
        if not doms:
            continue
        L = len(seq)
        for (d0, d1) in doms:
            if d0 < 0 or d1 >= L or d1 <= d0:
                continue
            t = (d0 + d1) // 2
            if not (MARGIN <= t <= L - MARGIN):
                continue
            size = d1 - d0 + 1
            if size <= W - 20:
                containable.append(dict(acc=acc, seq=seq, t=t, d0=d0, d1=d1, dsize=size))
                break                      # one target per protein
        # large-domain (all windows intra-domain): need t >= d0+MARGIN and t <= d1-(W-round(0.1*W))
        for (d0, d1) in doms:
            if d1 >= L or d1 - d0 + 1 < 2 * W:
                continue
            t = (d0 + d1) // 2
            need_lo = d0 + MARGIN
            need_hi = d1 - (W - round(0.1 * W))       # ensure rel=0.1 window's right end <= d1
            if need_lo <= t <= need_hi and MARGIN <= t <= L - MARGIN:
                large_domain.append(dict(acc=acc, seq=seq, t=t, d0=d0, d1=d1, dsize=d1 - d0 + 1))
                break

    CACHE.write_text(json.dumps(cache))
    # de-dup proteins across sets not needed; each set has <=1 target per protein
    OUT.write_text(json.dumps(dict(W=W, MARGIN=MARGIN,
                                   containable=containable, large_domain=large_domain), indent=1))
    print(f"cached {sum(1 for v in cache.values() if v is not None)} proteins with domains "
          f"({sum(1 for v in cache.values() if v)} non-empty)")
    print(f"target sets: containable (|D|<= {W-20}) = {len(containable)}; "
          f"large_domain (|D|>= {2*W}, all-windows-intra-domain) = {len(large_domain)}")
    if containable:
        ds = [c["dsize"] for c in containable]
        print(f"  containable domain sizes: min {min(ds)} median {sorted(ds)[len(ds)//2]} max {max(ds)}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
