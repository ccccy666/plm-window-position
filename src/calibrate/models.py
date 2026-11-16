"""ESM-2 inference API for the pilot probes (P0.0.2).

Position convention: callers pass **0-based residue indices** into the raw
sequence. The wrapper handles the +1 BOS offset and strips BOS/EOS internally.

Memory note (CLAUDE.md): `attention_into` uses fair-esm `need_head_weights`,
which materializes per-layer L x L attention. Fine for the pilot's short/medium
L; for the full P1 grid at L>=2000 this must be replaced with per-layer hooks
that capture only the target columns. Tracked in execution_plan P1.3.
"""
from __future__ import annotations
from pathlib import Path as _Path  # repository root, for portable paths
_ROOT = _Path(__file__).resolve().parents[2]

import os
from typing import Sequence

import numpy as np
import torch


class ESM2Wrapper:
    def __init__(self, model, alphabet, device: str):
        self.model = model
        self.alphabet = alphabet
        self.device = device
        self._bc = alphabet.get_batch_converter()
        self.num_layers = int(model.num_layers)
        self.num_heads = int(model.layers[0].self_attn.num_heads)
        self.embed_dim = int(model.embed_dim)

    # --- helpers ---
    def _tokens(self, sequences: Sequence[str]) -> torch.Tensor:
        data = [(str(i), s) for i, s in enumerate(sequences)]
        _, _, toks = self._bc(data)
        return toks.to(self.device)

    @staticmethod
    def _check_positions(seq: str, positions: Sequence[int]) -> None:
        for p in positions:
            if p < 0 or p >= len(seq):
                raise ValueError(f"position {p} out of range for sequence of length {len(seq)}")

    # --- API ---
    @torch.no_grad()
    def embed(self, sequences: Sequence[str], layer: int = 33) -> list[np.ndarray]:
        """Per-residue embeddings; returns one (L_i, embed_dim) array per sequence."""
        toks = self._tokens(sequences)
        out = self.model(toks, repr_layers=[layer], need_head_weights=False)
        rep = out["representations"][layer]
        result = []
        for i, s in enumerate(sequences):
            result.append(rep[i, 1 : 1 + len(s)].float().cpu().numpy())
        return result

    @torch.no_grad()
    def recovery_prob(self, sequence: str, positions: Sequence[int]) -> np.ndarray:
        """Masked-LM recovery probability of the true residue at each position.

        All `positions` are masked simultaneously (motif-style), then the softmax
        probability of each true residue is read off.
        """
        self._check_positions(sequence, positions)
        toks = self._tokens([sequence]).clone()
        for p in positions:
            toks[0, p + 1] = self.alphabet.mask_idx
        out = self.model(toks, need_head_weights=False)
        logits = out["logits"][0]  # (L+2, vocab)
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        probs = []
        for p in positions:
            true_idx = self.alphabet.get_idx(sequence[p])
            probs.append(float(logprobs[p + 1, true_idx].exp().item()))
        return np.asarray(probs, dtype=float)

    @torch.no_grad()
    def attention_into(
        self,
        sequence: str,
        target_positions: Sequence[int],
        layers: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Attention mass each target residue *receives* (summed over query tokens).

        Returns (n_layers, n_heads, n_targets). `layers` are 1-indexed (1..num_layers);
        default = all layers.
        """
        self._check_positions(sequence, target_positions)
        if layers is None:
            layers = list(range(1, self.num_layers + 1))
        toks = self._tokens([sequence])
        out = self.model(toks, need_head_weights=True)
        att = out["attentions"][0]  # (n_layers, n_heads, T, T): [..., q, k]
        target_toks = [p + 1 for p in target_positions]
        # mass into target k = sum over queries q of att[..., q, k]
        mass = att[:, :, :, target_toks].sum(dim=2)  # (n_layers, n_heads, n_targets)
        layer_idx = [ell - 1 for ell in layers]
        mass = mass[layer_idx]
        return mass.float().cpu().numpy()

    @torch.no_grad()
    def position_logprobs(self, sequence: str) -> np.ndarray:
        """Per-position log-softmax over the vocab from ONE forward on the full
        (unmasked) sequence — the 'wt-marginals' distribution. Returns (L, vocab).
        """
        toks = self._tokens([sequence])
        out = self.model(toks, need_head_weights=False)
        logits = out["logits"][0]  # (L+2, vocab)
        lp = torch.log_softmax(logits.float(), dim=-1)
        return lp[1 : 1 + len(sequence)].cpu().numpy()

    def aa_token_index(self, aa: str) -> int:
        return self.alphabet.get_idx(aa)

    @torch.no_grad()
    def contacts(self, sequence: str) -> np.ndarray:
        """ESM-2 predicted contact map (L, L) in [0,1] (BOS/EOS already stripped by fair-esm)."""
        toks = self._tokens([sequence])
        c = self.model.predict_contacts(toks)[0]
        return c.float().cpu().numpy()

    @torch.no_grad()
    def masked_distribution(self, sequence: str, positions: Sequence[int]) -> np.ndarray:
        """Full softmax distribution over the vocab at each masked position.

        Masks all `positions` simultaneously (as in recovery_prob); returns
        (len(positions), vocab). Used for symmetric multi-crop ensembling (D5).
        """
        self._check_positions(sequence, positions)
        toks = self._tokens([sequence]).clone()
        for p in positions:
            toks[0, p + 1] = self.alphabet.mask_idx
        out = self.model(toks, need_head_weights=False)
        probs = torch.softmax(out["logits"][0].float(), dim=-1)
        return np.stack([probs[p + 1].cpu().numpy() for p in positions])


def load_esm2(name: str = "esm2_t33_650M_UR50D", device: str | None = None) -> ESM2Wrapper:
    """Load an ESM-2 checkpoint via fair-esm (weights cached under TORCH_HOME)."""
    os.environ.setdefault(
        "TORCH_HOME",
        str(_ROOT / "data/cache/models/torch"),
    )
    import esm  # fair-esm

    model, alphabet = getattr(esm.pretrained, name)()
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.eval()
    if device == "cuda":
        model = model.half()
    model = model.to(device)
    return ESM2Wrapper(model, alphabet, device)
