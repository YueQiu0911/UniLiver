"""Inference / evaluation entry point for UniLiver.

Runs sliding-window inference over a test set and reports the per-target metrics
of Tab. 1. Supports the paper's "Single" (train==test dataset) and "Multi"
(joint-trained) checkpoints.

NOTE
----
Reference skeleton. The CLI and the metric-reporting structure are final; the
sliding-window predictor and metric aggregation are pseudocode placeholders.

Example
-------
    python scripts/inference.py --config configs/uniliver_lits.yaml \\
        --checkpoint checkpoints/uniliver_multi.pth --dataset LiTS
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from uniliver.models import UniLiver, TOPO_ORDER, VESSEL, COUINAUD, TUMOR
from uniliver.data import HepaticDataset
from uniliver.utils import metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate UniLiver.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--data-root", type=str, default="data/")
    p.add_argument("--dataset", type=str, default="LiTS")
    p.add_argument("--roi", nargs=3, type=int, default=[96, 96, 96])
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


@torch.no_grad()
def sliding_window_predict(model, image, roi):
    """Tile the volume, run the model per tile, stitch logits back.

    Pseudocode
    ----------
        for each ROI window with overlap:
            logits = model(window)
            accumulate logits into the full-volume canvas (Gaussian-weighted)
        return {t: argmax/threshold(canvas[t]) for t in targets}
    """
    # --- PSEUDOCODE: MONAI sliding_window_inference per target ---
    raise NotImplementedError("sliding_window_predict: released with full code.")


def evaluate(model, loader, roi, device) -> dict:
    """Return a {target: {metric: value}} report mirroring Tab. 1."""
    model.eval()
    # Accumulators per target; final reduction does the per-metric aggregation.
    report = {VESSEL: {}, COUINAUD: {}, TUMOR: {}}
    for batch in loader:
        image = batch["image"].to(device)
        preds = sliding_window_predict(model, image, roi)
        # --- PSEUDOCODE: accumulate dice/voe/asd/assd/hd95/g-dice/c-dice ---
        # for t in present_targets(batch): update report[t] with metrics.*()
    return report


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = UniLiver().to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    test_set = HepaticDataset(args.data_root, [args.dataset], split="test")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    report = evaluate(model, test_loader, tuple(args.roi), device)
    for target, scores in report.items():
        print(f"== {target} ==", scores)


if __name__ == "__main__":
    main()
