"""Training entry point for UniLiver.

Highlights the MTGCA training-time ordering: each target is backward-ed
separately so its per-target shared-parameter gradient can be pushed into the
gradient bank *before* the optimizer step, making the affinity used at step t
reflect step (t-1)'s gradients.

NOTE
----
Reference skeleton. The argument parsing, model/optimizer construction and the
high-level loop structure are final; the data plumbing and the gradient-bank
hook sites are pseudocode placeholders. Run config lives in ``configs/``.

Example
-------
    python scripts/train.py --config configs/uniliver_lits.yaml
"""

from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from uniliver.models import UniLiver, TOPO_ORDER
from uniliver.losses import MultiTargetLoss
from uniliver.data import HepaticDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train UniLiver.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--data-root", type=str, default="data/")
    p.add_argument("--datasets", nargs="+", default=["LiTS"])
    p.add_argument("--epochs", type=int, default=150)        # paper Sec. 3.1
    p.add_argument("--batch-size", type=int, default=2)      # paper Sec. 3.1
    p.add_argument("--lr", type=float, default=2e-5)         # paper Sec. 3.1
    p.add_argument("--lr-decay", type=float, default=0.2)    # x0.2 every 5 epochs
    p.add_argument("--lr-step", type=int, default=5)
    p.add_argument("--pretrained", type=str, default="vesselfm.pth")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def flat_shared_grad(model: UniLiver) -> torch.Tensor:
    """Flatten the gradient of the *shared* backbone params into a 1-D vector.

    Pseudocode
    ----------
        return cat([p.grad.flatten() for p in model.backbone.parameters()
                    if p.grad is not None])
    """
    # --- PSEUDOCODE: collect shared-encoder gradient for the bank ---
    raise NotImplementedError("flat_shared_grad: released with full code.")


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    running = 0.0
    for batch in loader:
        image = batch["image"].to(device)
        targets = {t: batch[t].to(device) for t in TOPO_ORDER if t in batch}

        preds = model(image)
        per_target_loss = criterion(preds, targets)  # {target: scalar}

        optimizer.zero_grad()
        # --- MTGCA gradient-bank ordering ---------------------------------
        # Backprop each target separately, snapshot its shared-param gradient
        # into the bank, then accumulate. Pseudocode:
        #
        #   for t in present_targets:
        #       per_target_loss[t].backward(retain_graph=True)
        #       model.mtgca.observe_gradient(idx(t), flat_shared_grad(model))
        #   total = sum(per_target_loss.values())
        #   total.backward()  # or reuse accumulated grads
        #   optimizer.step()
        # ------------------------------------------------------------------
        total = sum(per_target_loss.values())
        total.backward()
        optimizer.step()
        running += float(total.detach())
    return running / max(len(loader), 1)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = UniLiver().to(device)
    # model.backbone.load_pretrained(args.pretrained)  # VesselFM init (full code)

    train_set = HepaticDataset(args.data_root, args.datasets, split="train")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    criterion = MultiTargetLoss(model.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_step, gamma=args.lr_decay
    )

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()
        print(f"[epoch {epoch:03d}] loss={loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
        # torch.save(model.state_dict(), f"checkpoints/uniliver_e{epoch:03d}.pth")


if __name__ == "__main__":
    main()
