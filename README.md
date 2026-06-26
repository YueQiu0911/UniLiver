# UniLiver: Gradient-Conditioned Unified Model for Multi-target Hepatic Segmentation

Official repository for **UniLiver**, a unified model that jointly segments
hepatic **vessels**, **Couinaud segments**, and **tumors** by exploiting their
anatomical dependencies.

> **Status — code release in progress.**
> This repository currently provides a **reference code skeleton** released for
> the camera-ready stage. The module interfaces, tensor shapes, and the
> end-to-end forward/training flow are final and faithful to the paper, while
> the core algorithmic internals are given as **pseudocode** and marked with
> `NotImplementedError` at the placeholder sites. The **full implementation will
> be released once the paper is officially online.** Watch / star the repo for
> the update.

## Method overview

UniLiver integrates three modules into a [DynUNet](https://docs.monai.io)
backbone (initialized from pre-trained [VesselFM](https://github.com/) weights):

| Module | What it does | Code |
| --- | --- | --- |
| **MTGCA** — Multi-Target Gradient-Conditioned Adapter | Resolves multi-target optimization conflicts by conditioning shared feature modulation (FiLM) on inter-target gradient coherence via a per-target **gradient bank** + affinity matrix. | [`uniliver/models/mtgca.py`](uniliver/models/mtgca.py) |
| **TA-SSE** — Target-Aware Spectral-Spatial Encoding | Jointly models spectral (learnable complex FFT filter banks + input-adaptive routing) and positional (target-weighted Fourier features) cues per target. | [`uniliver/models/ta_sse.py`](uniliver/models/ta_sse.py) |
| **VCT-SD** — VCT Sequential Denoising | Progressively refines features along the anatomical DAG (Vessel → Couinaud → Tumor) with directed cross-attention over S steps. | [`uniliver/models/vct_sd.py`](uniliver/models/vct_sd.py) |

See the paper for the formal definitions (Eqs. 1–4).

## Repository layout

```
uniliver/
  models/
    uniliver.py     # full model assembly (Fig. 1a)
    mtgca.py        # Multi-Target Gradient-Conditioned Adapter   (Sec. 2.1)
    ta_sse.py       # Target-Aware Spectral-Spatial Encoding      (Sec. 2.2)
    vct_sd.py       # VCT Sequential Denoising                    (Sec. 2.3)
  data/
    dataset.py      # composite LiTS + MSD8 dataset
  utils/
    metrics.py      # Dice / VOE / ACC / ASD / ASSD / HD95 / G-Dice / C-Dice
  losses.py         # Dice + CE multi-target loss (lambda_tumor = 3.0)
configs/
  uniliver_lits.yaml
scripts/
  train.py          # training (with the MTGCA gradient-bank ordering)
  inference.py      # sliding-window evaluation -> Tab. 1 metrics
```

## Installation

```bash
pip install -r requirements.txt
```

## Data

UniLiver is trained and evaluated on the **LiTS** and **MSD8** liver datasets,
each split 8:2 (train/test). For the joint ("Multi") setting, the two training
splits are combined and evaluated per dataset. Place the preprocessed volumes
under `data/` (see [`uniliver/data/dataset.py`](uniliver/data/dataset.py) for the
expected layout).

## Usage

> The commands below show the intended interface. They run end-to-end **once the
> full code is released**; in the current skeleton the placeholder sites raise
> `NotImplementedError`.

Train:

```bash
python scripts/train.py --config configs/uniliver_lits.yaml
```

Evaluate:

```bash
python scripts/inference.py \
    --config configs/uniliver_lits.yaml \
    --checkpoint checkpoints/uniliver_multi.pth \
    --dataset LiTS
```

## Results

UniLiver achieves state-of-the-art performance across all three targets on LiTS
and MSD8 (see Tab. 1 of the paper), e.g. Couinaud Dice **91.15** (LiTS), vessel
Dice **69.15** / HD95 **10.61 mm** (LiTS), tumor G-Dice **83.86** (LiTS).

## Paper

**UniLiver: Gradient-Conditioned Unified Model for Multi-target Hepatic Segmentation**

*(Full citation will be added once the paper is officially online.)*

## License

Released under the [MIT License](LICENSE).
