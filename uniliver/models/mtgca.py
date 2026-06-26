"""Multi-Target Gradient-Conditioned Adapter (MTGCA).

This module resolves multi-target optimization conflicts by conditioning the
shared feature modulation on inter-target gradient coherence (Sec. 2.1 of the
paper). At every training step it maintains a per-target gradient bank, builds a
target affinity matrix from the projected gradients, and uses the resulting
conflict signal to drive a feature-wise linear modulation (FiLM) of the shared
encoder features.

NOTE
----
This is a *reference skeleton* released for the camera-ready stage. The module
interfaces, tensor shapes and the high-level forward flow are final, but the
core gradient-bank / orthogonal-decomposition logic is given as pseudocode and
raises ``NotImplementedError`` at the placeholder sites. The full implementation
will be released once the paper is officially online.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GradientBank(nn.Module):
    r"""Per-target gradient bank :math:`\mathcal{B} = \{g_t\}_{t=1}^{n}`.

    Stores the random-projected gradient of each target w.r.t. the shared
    parameters (Eq. 1):

        g_t = P^T  \nabla_theta L_t  \in  R^d

    and exposes the target affinity matrix A (Eq. just below Eq. 1):

        A[i, j] = < g_i / ||g_i||, g_j / ||g_j|| >     (cosine similarity)

    The conflict vector for target t is the t-th row  c_t = A[t, :].
    """

    def __init__(self, num_targets: int, param_dim: int, proj_dim: int = 256):
        super().__init__()
        self.num_targets = num_targets
        self.proj_dim = proj_dim

        # Fixed random projection P in R^{p x d}. Registered as a buffer so it is
        # serialized with the model but never receives gradients.
        # NOTE: param_dim (p) can be very large; the released code uses a
        # memory-efficient blockwise / sparse projection instead of a dense buffer.
        self.register_buffer("projection", torch.empty(0))  # placeholder

        # Rolling store of projected gradients, shape (num_targets, proj_dim).
        self.register_buffer("bank", torch.zeros(num_targets, proj_dim))

    @torch.no_grad()
    def update(self, target_idx: int, grad_flat: torch.Tensor) -> None:
        """Project ``grad_flat`` and write it into slot ``target_idx``.

        Pseudocode
        ----------
            g = P^T @ grad_flat            # (proj_dim,)
            bank[target_idx] = g           # overwrite slot for this target
        """
        # --- PSEUDOCODE: random-projection update of the gradient bank ---
        # g = self.projection.t() @ grad_flat
        # self.bank[target_idx] = g
        raise NotImplementedError(
            "GradientBank.update: gradient projection released with full code."
        )

    @torch.no_grad()
    def affinity(self) -> torch.Tensor:
        """Return the target affinity matrix A in R^{n x n} (cosine similarity).

        Pseudocode
        ----------
            G_hat = normalize(bank, dim=-1)     # row-wise L2 normalize
            A = G_hat @ G_hat.t()               # (n, n)
            return A
        """
        # --- PSEUDOCODE: affinity = normalized-gradient Gram matrix ---
        # g_hat = torch.nn.functional.normalize(self.bank, dim=-1)
        # return g_hat @ g_hat.t()
        raise NotImplementedError(
            "GradientBank.affinity: affinity construction released with full code."
        )

    def conflict_vector(self, target_idx: int) -> torch.Tensor:
        """Conflict vector c_t = A[t, :] for a single target."""
        return self.affinity()[target_idx]


class _FiLMHead(nn.Module):
    """Maps an embedding to a (scale, shift) pair for FiLM modulation.

    Used twice inside MTGCA:
      * phi(e_t)  -> (gamma_sem, beta_sem)   semantic target identity
      * psi(c_t)  -> (gamma_grad, beta_grad) gradient-derived relational signal
    """

    def __init__(self, in_dim: int, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, channels),
            nn.SiLU(),
            nn.Linear(channels, 2 * channels),
        )
        self.channels = channels

    def forward(self, emb: torch.Tensor):
        gamma, beta = self.net(emb).chunk(2, dim=-1)
        return gamma, beta


class MTGCA(nn.Module):
    r"""Multi-Target Gradient-Conditioned Adapter.

    Applies the gradient-conditioned FiLM of Eq. (2):

        F' = (1 + gamma_t(e_t, c_t)) ⊙ F + beta_t(e_t, c_t)

    with the decomposition

        gamma_t = gamma_sem_t + gamma_grad_t ,   [gamma_sem, beta_sem] = phi(e_t)
        beta_t  = beta_sem_t  + beta_grad_t  ,   [gamma_grad, beta_grad] = psi(c_t)

    Parameters
    ----------
    channels : int
        Channel dimension C of the feature tensor F (B, C, D, H, W).
    num_targets : int
        Number of segmentation targets n (here 3: vessel / Couinaud / tumor).
    embed_dim : int
        Dimension d_e of the per-target semantic embedding e_t.
    param_dim : int
        Size p of the shared-parameter gradient that feeds the gradient bank.
    proj_dim : int
        Subspace dimension d of the random projection P.
    """

    def __init__(
        self,
        channels: int,
        num_targets: int = 3,
        embed_dim: int = 64,
        param_dim: int = 1 << 20,
        proj_dim: int = 256,
    ):
        super().__init__()
        self.num_targets = num_targets

        # Learnable semantic embedding e_t for each target.
        self.target_embedding = nn.Embedding(num_targets, embed_dim)

        # phi : e_t -> (gamma_sem, beta_sem)
        self.phi = _FiLMHead(embed_dim, channels)
        # psi : c_t -> (gamma_grad, beta_grad);  c_t lives in R^{num_targets}
        self.psi = _FiLMHead(num_targets, channels)

        self.gradient_bank = GradientBank(num_targets, param_dim, proj_dim)

    # --- training-time hook -------------------------------------------------
    def observe_gradient(self, target_idx: int, grad_flat: torch.Tensor) -> None:
        """Push the latest per-target shared-parameter gradient into the bank.

        Called by the training loop after each target's backward pass, *before*
        the optimizer step, so the affinity used at step t reflects step t-1's
        gradients. See ``scripts/train.py`` for the exact ordering.
        """
        self.gradient_bank.update(target_idx, grad_flat)

    # --- forward ------------------------------------------------------------
    def forward(self, feat: torch.Tensor, target_idx: int) -> torch.Tensor:
        """Modulate ``feat`` for ``target_idx`` (Eq. 2).

        Parameters
        ----------
        feat : (B, C, D, H, W) feature tensor F.
        target_idx : int target id t in [0, num_targets).

        Returns
        -------
        (B, C, D, H, W) modulated tensor F'.
        """
        device = feat.device
        e_t = self.target_embedding(
            torch.tensor(target_idx, device=device)
        )  # (embed_dim,)
        c_t = self.gradient_bank.conflict_vector(target_idx)  # (num_targets,)

        gamma_sem, beta_sem = self.phi(e_t)
        gamma_grad, beta_grad = self.psi(c_t)
        gamma = gamma_sem + gamma_grad  # (C,)
        beta = beta_sem + beta_grad     # (C,)

        # Broadcast (C,) -> (1, C, 1, 1, 1) for the 5-D feature tensor.
        gamma = gamma.view(1, -1, 1, 1, 1)
        beta = beta.view(1, -1, 1, 1, 1)
        return (1.0 + gamma) * feat + beta
