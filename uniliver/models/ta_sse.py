"""Target-Aware Spectral-Spatial Encoding (TA-SSE).

Jointly models spectral characteristics and positional information at multiple
network scales, modulated by a per-target embedding (Sec. 2.2, Eq. 3):

    F_hat = F + alpha * G_omega(F) + beta * psi(F, e_t, gamma(p))

where
  * G_omega(F) = IFFT( FFT(F) ⊙ sum_k r_k W_c^{(k)} )   spectral filtering with
    learnable complex filter banks W_c^{(k)} and input-adaptive routing r,
  * psi(F, e_t, gamma(p))                                positional term whose
    contribution is dynamically weighted by the target embedding e_t,
  * gamma(p)                                             Fourier feature mapping
    of normalized 3D coordinates p = (x, y, z) in [-1, 1]^3.

NOTE
----
Reference skeleton for the camera-ready stage. Shapes and module wiring are
final; the spectral-routing and Fourier-positional internals are pseudocode and
raise ``NotImplementedError`` at the placeholder sites. Full code to follow when
the paper is online.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def fourier_feature_mapping(coords: torch.Tensor, num_bands: int) -> torch.Tensor:
    """gamma(p): map 3D coords in [-1, 1]^3 to sinusoidal features.

    Pseudocode
    ----------
        freqs = 2^[0..num_bands-1] * pi
        feats = concat([sin(coords * f), cos(coords * f) for f in freqs])
        return feats        # (..., 3 * 2 * num_bands)
    """
    # --- PSEUDOCODE: sinusoidal positional encoding of (x, y, z) ---
    raise NotImplementedError(
        "fourier_feature_mapping: released with full code."
    )


class SpectralFilter(nn.Module):
    r"""Input-adaptive spectral filtering G_omega(F) via 3-D FFT.

        G_omega(F) = F^{-1}( F(F) ⊙ sum_{k=1}^{K} r_k W_c^{(k)} )

    A small router maps a global descriptor of F to mixing weights r over K
    learnable complex-valued filter banks W_c^{(k)}.
    """

    def __init__(self, channels: int, num_banks: int = 4):
        super().__init__()
        self.channels = channels
        self.num_banks = num_banks

        # K complex filter banks. Stored as (real, imag) so they remain ordinary
        # trainable parameters. Spatial spectrum size is set lazily / via config
        # in the released code; here we only declare the parameter holders.
        self.filter_real = nn.Parameter(torch.empty(0))  # placeholder
        self.filter_imag = nn.Parameter(torch.empty(0))  # placeholder

        # Routing network: global-pooled feature -> K mixing weights.
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, num_banks),
            nn.Softmax(dim=-1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """Apply routed spectral filtering. Returns a tensor shaped like ``feat``.

        Pseudocode
        ----------
            r   = router(feat)                       # (B, K)
            Wc  = sum_k r_k * (filter_real_k + i*filter_imag_k)
            Xf  = rfftn(feat, dims=(-3,-2,-1))       # 3-D FFT
            Yf  = Xf * Wc                            # complex elementwise
            out = irfftn(Yf, dims=(-3,-2,-1))        # back to spatial
            return out
        """
        # --- PSEUDOCODE: routed complex-spectral filtering ---
        raise NotImplementedError(
            "SpectralFilter.forward: FFT routing released with full code."
        )


class PositionalTerm(nn.Module):
    """Target-weighted positional term psi(F, e_t, gamma(p)).

    The positional contribution is gated by the target embedding so that, e.g.,
    vessels rely strongly on absolute position while tumors rely more on local
    appearance.
    """

    def __init__(self, channels: int, embed_dim: int, num_bands: int = 6):
        super().__init__()
        self.num_bands = num_bands
        pos_dim = 3 * 2 * num_bands
        self.pos_proj = nn.Conv3d(pos_dim, channels, kernel_size=1)
        # Scalar-ish gate produced from the target embedding e_t.
        self.target_gate = nn.Linear(embed_dim, channels)

    def forward(self, feat: torch.Tensor, target_embed: torch.Tensor) -> torch.Tensor:
        """Return the (gated) positional feature, shaped like ``feat``.

        Pseudocode
        ----------
            p     = normalized_grid(feat.shape)      # (x,y,z) in [-1,1]^3
            g_p   = fourier_feature_mapping(p, num_bands)
            pos   = pos_proj(g_p)                     # (B, C, D, H, W)
            gate  = sigmoid(target_gate(e_t))         # (C,)
            return gate[:, None, None, None] * pos
        """
        # --- PSEUDOCODE: build coord grid, encode, project, gate by target ---
        raise NotImplementedError(
            "PositionalTerm.forward: positional gating released with full code."
        )


class DilatedBottleneck(nn.Module):
    """Multi-scale dilated-conv block used at the encoder bottleneck.

    Matches Fig. 1(c): Conv -> DilatedConv(d=2) -> DilatedConv(d=4) ->
    DilatedConv(d=2) -> Conv, aggregating multi-scale spatial context.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.Conv3d(channels, channels, 3, padding=2, dilation=2),
            nn.Conv3d(channels, channels, 3, padding=4, dilation=4),
            nn.Conv3d(channels, channels, 3, padding=2, dilation=2),
            nn.Conv3d(channels, channels, 3, padding=1),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return feat + self.block(feat)


class TASSE(nn.Module):
    """Target-Aware Spectral-Spatial Encoding (Eq. 3).

    Parameters
    ----------
    channels : int      feature channel dim C.
    embed_dim : int     target embedding dim d_e.
    at_bottleneck : bool  if True, also apply the multi-scale dilated bottleneck.
    """

    def __init__(
        self,
        channels: int,
        embed_dim: int = 64,
        num_banks: int = 4,
        num_bands: int = 6,
        at_bottleneck: bool = False,
    ):
        super().__init__()
        self.spectral = SpectralFilter(channels, num_banks)
        self.positional = PositionalTerm(channels, embed_dim, num_bands)
        self.bottleneck = DilatedBottleneck(channels) if at_bottleneck else None

        # Learnable scalar mixing weights alpha, beta in Eq. (3).
        self.alpha = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.ones(1))

    def forward(self, feat: torch.Tensor, target_embed: torch.Tensor) -> torch.Tensor:
        """Eq. (3): F_hat = F + alpha * G_omega(F) + beta * psi(F, e_t, gamma(p))."""
        out = feat + self.alpha * self.spectral(feat) \
            + self.beta * self.positional(feat, target_embed)
        if self.bottleneck is not None:
            out = self.bottleneck(out)
        return out
