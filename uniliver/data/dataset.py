"""Composite LiTS + MSD8 hepatic dataset.

Provides volumes with up to three label maps (vessel / Couinaud / tumor). Both
datasets are split 8:2 train/test (paper Sec. 3.1); a "joint" mode concatenates
the two training splits and evaluates per dataset.

NOTE
----
Reference skeleton: the I/O contract and the returned sample schema are final;
the LiTS/MSD8 readers and preprocessing transforms are pseudocode placeholders.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset

from ..models import VESSEL, COUINAUD, TUMOR


class HepaticDataset(Dataset):
    """Returns dict samples: {"image": (1,D,H,W), "V"/"C"/"T": (D,H,W) labels}.

    Not every case has all three labels; missing targets are simply absent from
    the sample dict, and the loss/metrics skip them.
    """

    def __init__(
        self,
        root: str,
        datasets: List[str],   # e.g. ["LiTS", "MSD8"] for the joint setting
        split: str = "train",  # "train" | "test"
        patch_size=(96, 96, 96),
    ):
        super().__init__()
        self.root = root
        self.datasets = datasets
        self.split = split
        self.patch_size = patch_size

        # Pseudocode: enumerate case ids for the requested datasets/split.
        #   self.cases = build_index(root, datasets, split, ratio=0.8)
        self.cases: List[str] = []  # placeholder; populated in full code

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load one preprocessed case.

        Pseudocode
        ----------
            img  = load_ct(case);  img = normalize_hu(img)
            labels = {t: load_label(case, t) for t in available_targets(case)}
            img, labels = random_crop_or_resample(img, labels, patch_size)
            return {"image": img, **labels}
        """
        # --- PSEUDOCODE: LiTS/MSD8 reader + preprocessing ---
        raise NotImplementedError("HepaticDataset.__getitem__: released with full code.")


# Convenience target list for callers.
ALL_TARGETS = [VESSEL, COUINAUD, TUMOR]
