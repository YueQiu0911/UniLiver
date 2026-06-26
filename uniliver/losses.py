"""Loss functions for UniLiver.

Per the paper (Sec. 3.1), each target uses a combined Dice + Cross-Entropy loss,
with target weights:

    lambda_vessel = 1.0,  lambda_tumor = 3.0,  lambda_couinaud = 1.0
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .models import VESSEL, COUINAUD, TUMOR

# Target loss weights from the paper.
TARGET_WEIGHTS: Dict[str, float] = {VESSEL: 1.0, COUINAUD: 1.0, TUMOR: 3.0}


class DiceCELoss(nn.Module):
    """Combined Dice + Cross-Entropy loss for one target.

    Binary targets (vessel/tumor) use sigmoid+Dice with BCE; the multi-class
    Couinaud target uses softmax-Dice with CE. The released code defers to
    MONAI's ``DiceCELoss``; the signature here is the public contract.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return scalar loss.

        Pseudocode
        ----------
            dice = 1 - soft_dice(activation(logits), one_hot(target))
            ce   = cross_entropy(logits, target)
            return dice + ce
        """
        # --- PSEUDOCODE: Dice + CE (see MONAI DiceCELoss) ---
        raise NotImplementedError("DiceCELoss.forward: released with full code.")


class MultiTargetLoss(nn.Module):
    """Weighted sum of per-target Dice+CE losses; also returns the per-target
    components so the training loop can backprop them separately for MTGCA's
    gradient bank.
    """

    def __init__(self, num_classes: Dict[str, int]):
        super().__init__()
        self.losses = nn.ModuleDict(
            {t: DiceCELoss(num_classes[t]) for t in num_classes}
        )

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Return {target: weighted_loss} for every target present in ``targets``."""
        out: Dict[str, torch.Tensor] = {}
        for t, gt in targets.items():
            out[t] = TARGET_WEIGHTS[t] * self.losses[t](preds[t], gt)
        return out
