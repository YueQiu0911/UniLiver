"""Evaluation metrics used in the paper (Tab. 1).

  * Couinaud: Dice, ACC, ASD
  * Vessel:   Dice, VOE, ASSD, HD95
  * Tumor:    Global-Dice (G-Dice), Per-Case-Dice (C-Dice), VOE

All metrics operate on hard label maps (after argmax/threshold). Surface metrics
(ASD/ASSD/HD95) are computed in millimetres using the voxel spacing.
"""

from __future__ import annotations

import torch


def dice(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> float:
    """Soft/hard Dice overlap. Pseudocode: 2|P∩G| / (|P|+|G|)."""
    raise NotImplementedError("dice: released with full code.")


def voe(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Volumetric Overlap Error = 1 - |P∩G|/|P∪G|."""
    raise NotImplementedError("voe: released with full code.")


def accuracy(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Voxel classification accuracy (used for Couinaud)."""
    raise NotImplementedError("accuracy: released with full code.")


def asd(pred: torch.Tensor, gt: torch.Tensor, spacing) -> float:
    """Average Surface Distance (mm)."""
    raise NotImplementedError("asd: released with full code.")


def assd(pred: torch.Tensor, gt: torch.Tensor, spacing) -> float:
    """Average Symmetric Surface Distance (mm)."""
    raise NotImplementedError("assd: released with full code.")


def hd95(pred: torch.Tensor, gt: torch.Tensor, spacing) -> float:
    """95th-percentile Hausdorff Distance (mm)."""
    raise NotImplementedError("hd95: released with full code.")


def global_dice(preds, gts) -> float:
    """G-Dice: Dice over the pooled voxels of all cases (tumor)."""
    raise NotImplementedError("global_dice: released with full code.")


def per_case_dice(preds, gts) -> float:
    """C-Dice: mean of per-case Dice scores (tumor)."""
    raise NotImplementedError("per_case_dice: released with full code.")
