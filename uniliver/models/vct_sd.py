"""VCT Sequential Denoising (VCT-SD).

Progressively refines the three target-specific feature maps by propagating
structural cues along the anatomical dependency DAG (Sec. 2.3):

    G = (V, E),  V = {V, C, T},  E = { V->C, C->T, V->T }

Each edge is a cross-attention block where the downstream target *queries*
structural cues from the upstream target. Refinement unfolds over S steps, each
conditioned on a sinusoidal step embedding e_s (Eq. 4):

    F_j^{(s+1)} = F_j^{(s)} + sum_{i: (i->j) in E} A_{i->j}( F_j^{(s)}, F_i^{(s)}, e_s )

Targets are updated in topological order (V, then C, then T) so that each target
consumes already-refined predecessors within the same step.

NOTE
----
Reference skeleton for the camera-ready stage. The DAG wiring, topological
schedule and step loop are final; the cross-attention internals are pseudocode
and raise ``NotImplementedError``. Full code to follow when the paper is online.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

# Canonical target ordering = topological order of the anatomical DAG.
VESSEL, COUINAUD, TUMOR = "V", "C", "T"
TOPO_ORDER: List[str] = [VESSEL, COUINAUD, TUMOR]
EDGES: List[Tuple[str, str]] = [
    (VESSEL, COUINAUD),  # vessel topology defines Couinaud partitioning
    (COUINAUD, TUMOR),   # segmental identity localizes tumors
    (VESSEL, TUMOR),     # vascular proximity localizes tumors
]


def sinusoidal_step_embedding(step: int, dim: int) -> torch.Tensor:
    """e_s: sinusoidal embedding of the current refinement step.

    Pseudocode
    ----------
        half = dim // 2
        freqs = exp(-log(10000) * arange(half) / half)
        args  = step * freqs
        return concat([sin(args), cos(args)])      # (dim,)
    """
    # --- PSEUDOCODE: standard sinusoidal timestep embedding ---
    raise NotImplementedError(
        "sinusoidal_step_embedding: released with full code."
    )


class DirectedCrossAttention(nn.Module):
    r"""A single edge A_{i->j}: target j (query) attends to target i (key/value).

    The step embedding e_s conditions the block so early steps apply coarse
    corrections and later steps fine-grained ones.
    """

    def __init__(self, channels: int, num_heads: int = 4, step_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.to_q = nn.Conv3d(channels, channels, kernel_size=1)
        self.to_k = nn.Conv3d(channels, channels, kernel_size=1)
        self.to_v = nn.Conv3d(channels, channels, kernel_size=1)
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)

        # Inject the step embedding as a per-channel (scale, shift).
        self.step_film = nn.Linear(step_dim, 2 * channels)

    def forward(
        self,
        feat_down: torch.Tensor,   # F_j^{(s)}  query target
        feat_up: torch.Tensor,     # F_i^{(s)}  key/value target
        step_embed: torch.Tensor,  # e_s
    ) -> torch.Tensor:
        """Return the residual update for target j from edge i->j.

        Pseudocode
        ----------
            q = to_q(feat_down); k = to_k(feat_up); v = to_v(feat_up)
            # flatten spatial dims, split heads
            attn = softmax(q @ k^T / sqrt(head_dim))
            ctx  = attn @ v
            ctx  = reshape_back(ctx)
            scale, shift = step_film(step_embed).chunk(2)
            ctx  = (1 + scale) * ctx + shift          # condition on step
            return proj(ctx)
        """
        # --- PSEUDOCODE: 3-D cross-attention with step conditioning ---
        raise NotImplementedError(
            "DirectedCrossAttention.forward: released with full code."
        )


class VCTSequentialDenoising(nn.Module):
    """VCT-SD module: S-step DAG-ordered cross-target refinement (Eq. 4).

    Parameters
    ----------
    channels : int       per-target feature channel dim.
    num_steps : int      number of refinement steps S.
    num_heads : int      attention heads per edge.
    step_dim : int       sinusoidal step-embedding dim.
    """

    def __init__(
        self,
        channels: int,
        num_steps: int = 3,
        num_heads: int = 4,
        step_dim: int = 64,
    ):
        super().__init__()
        self.num_steps = num_steps
        self.step_dim = step_dim

        # One cross-attention block per directed edge of the DAG.
        self.edges = nn.ModuleDict(
            {
                f"{src}->{dst}": DirectedCrossAttention(channels, num_heads, step_dim)
                for (src, dst) in EDGES
            }
        )
        # Per-target output projection applied after the S steps.
        self.out_proj = nn.ModuleDict(
            {t: nn.Conv3d(channels, channels, kernel_size=1) for t in TOPO_ORDER}
        )

    def _incoming(self, dst: str) -> List[str]:
        """Upstream targets that condition ``dst`` (i.e., i s.t. i->dst in E)."""
        return [src for (src, d) in EDGES if d == dst]

    def forward(self, feats: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Refine ``feats`` = {"V":F_V, "C":F_C, "T":F_T} over S steps.

        Targets are updated in topological order within each step, so a target
        already sees its predecessors' step-s update when it is refined.
        """
        feats = dict(feats)  # shallow copy; we overwrite per target per step
        for s in range(self.num_steps):
            e_s = sinusoidal_step_embedding(s, self.step_dim).to(
                next(iter(feats.values())).device
            )
            for dst in TOPO_ORDER:  # topological schedule: V -> C -> T
                update = torch.zeros_like(feats[dst])
                for src in self._incoming(dst):
                    update = update + self.edges[f"{src}->{dst}"](
                        feats[dst], feats[src], e_s
                    )
                feats[dst] = feats[dst] + update  # Eq. (4) residual update

        # Project the S-step-refined features for the segmentation heads.
        return {t: self.out_proj[t](feats[t]) for t in TOPO_ORDER}
