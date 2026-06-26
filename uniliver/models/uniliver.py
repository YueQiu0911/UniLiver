"""UniLiver: Gradient-Conditioned Unified Model for Multi-target Hepatic Segmentation.

Assembles the three proposed modules into a DynUNet backbone (Fig. 1a):

    Input -> Encoder --(TA-SSE @ skips/bottleneck)--> Decoder
          \\-- MTGCA modulates shared features per target
          -> {Vessel, Couinaud, Tumor} heads -> VCT-SD refinement -> outputs

The backbone is initialized from pre-trained VesselFM weights (paper Sec. 3.1).

NOTE
----
Reference skeleton for the camera-ready stage. The end-to-end wiring, the public
``forward`` contract and the output dict are final. The backbone hook-up and the
per-target feature routing contain pseudocode placeholders
(``NotImplementedError``) where the released code plugs in DynUNet internals.
Full code to follow when the paper is online.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .mtgca import MTGCA
from .ta_sse import TASSE
from .vct_sd import VCTSequentialDenoising, VESSEL, COUINAUD, TUMOR, TOPO_ORDER


class SegmentationHead(nn.Module):
    """1x1x1 conv head mapping refined features to per-target logits."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.head = nn.Conv3d(in_channels, num_classes, kernel_size=1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.head(feat)


class DynUNetBackbone(nn.Module):
    """Thin wrapper around the DynUNet backbone (MONAI), with TA-SSE hooks.

    Responsibilities:
      * run encoder/decoder and expose the shared decoder feature map;
      * apply TA-SSE at the skip connections and bottleneck.

    The actual DynUNet is constructed from config in the released code; here we
    only declare the interface and where TA-SSE plugs in.
    """

    def __init__(self, in_channels: int, feat_channels: int, embed_dim: int):
        super().__init__()
        self.feat_channels = feat_channels

        # Placeholder for the MONAI DynUNet (initialized from VesselFM weights).
        self.backbone = None  # set up in build()/load_pretrained() of full code

        # TA-SSE applied at skip connections (spectral) and bottleneck (spectral
        # + multi-scale dilated). One instance shown per location for clarity.
        self.tasse_skip = TASSE(feat_channels, embed_dim, at_bottleneck=False)
        self.tasse_bottleneck = TASSE(feat_channels, embed_dim, at_bottleneck=True)

    def forward(self, x: torch.Tensor, target_embed: torch.Tensor) -> torch.Tensor:
        """Return the shared decoder feature map (B, feat_channels, D, H, W).

        Pseudocode
        ----------
            skips, bottleneck = encoder(x)
            bottleneck = tasse_bottleneck(bottleneck, target_embed)
            skips = [tasse_skip(s, target_embed) for s in skips]
            feat = decoder(bottleneck, skips)
            return feat
        """
        # --- PSEUDOCODE: DynUNet enc/dec with TA-SSE injection ---
        raise NotImplementedError(
            "DynUNetBackbone.forward: DynUNet hook-up released with full code."
        )


class UniLiver(nn.Module):
    """Unified multi-target hepatic segmentation model.

    Parameters
    ----------
    in_channels : int            input modality channels (CT = 1).
    feat_channels : int          shared decoder feature channels.
    num_classes : dict           per-target #classes, keys "V"/"C"/"T".
    embed_dim : int              target-embedding dim (shared by MTGCA/TA-SSE).
    vct_steps : int              VCT-SD refinement steps S.
    """

    TARGETS = TOPO_ORDER  # ["V", "C", "T"]

    def __init__(
        self,
        in_channels: int = 1,
        feat_channels: int = 32,
        num_classes: Dict[str, int] | None = None,
        embed_dim: int = 64,
        vct_steps: int = 3,
    ):
        super().__init__()
        if num_classes is None:
            # Couinaud has 8 functional segments; vessel/tumor are binary.
            num_classes = {VESSEL: 1, COUINAUD: 8, TUMOR: 1}
        self.num_classes = num_classes

        self.backbone = DynUNetBackbone(in_channels, feat_channels, embed_dim)
        self.mtgca = MTGCA(feat_channels, num_targets=len(self.TARGETS),
                           embed_dim=embed_dim)
        self.vct_sd = VCTSequentialDenoising(feat_channels, num_steps=vct_steps)

        self.heads = nn.ModuleDict(
            {t: SegmentationHead(feat_channels, num_classes[t]) for t in self.TARGETS}
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the full pipeline.

        Parameters
        ----------
        x : (B, in_channels, D, H, W) input volume.

        Returns
        -------
        dict with logits for each target:
            {"V": (B, 1, D,H,W), "C": (B, 8, D,H,W), "T": (B, 1, D,H,W)}
        """
        # 1) Per-target shared features: TA-SSE-enhanced backbone + MTGCA FiLM.
        target_feats: Dict[str, torch.Tensor] = {}
        for t_idx, t in enumerate(self.TARGETS):
            embed = self.mtgca.target_embedding(
                torch.tensor(t_idx, device=x.device)
            )
            shared = self.backbone(x, embed)            # (B, C, D, H, W)
            target_feats[t] = self.mtgca(shared, t_idx)  # Eq. (2) modulation

        # 2) VCT-SD: DAG-ordered cross-target refinement (Eq. 4).
        refined = self.vct_sd(target_feats)

        # 3) Per-target segmentation heads.
        return {t: self.heads[t](refined[t]) for t in self.TARGETS}
