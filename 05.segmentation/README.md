# Semantic Segmentation — Retinal Blood Vessel Detection

## Overview

This experiment implements **semantic segmentation** to identify blood vessels in retinal fundus images. Accurate vessel segmentation is critical for diagnosing eye diseases such as diabetic retinopathy and glaucoma. The model performs pixel-level binary classification: each pixel is labeled as either **vessel** or **background**.

## Dataset

- **Source:** ICPR Retinal Blood Vessel Dataset
- **Train:** 268 images + masks
- **Test:** 112 images + masks
- **Classes:** 2 — background, blood vessel
- **Input resolution:** 512×512 pixels (trained on 256×256 random patches)

## Approach

1. **Architecture** — U-Net with an ImageNet-pretrained `tf_efficientnet_b3.in1k` encoder (via `timm`, `features_only=True`) and a symmetric decoder using double-conv blocks (Conv → BN → GELU) with skip connections.
2. **Augmentation** — Albumentations pipeline including flips, 90° rotations, affine, elastic transform, grid distortion, brightness/contrast, CLAHE, Gaussian noise, and ImageNet normalization.
3. **Loss function** — `DiceFocalLoss` from MONAI to handle the severe vessel/background class imbalance.
4. **Training** — Patch size: 256×256, Batch size: 8, Epochs: 100, Learning rate: 3e-4, Weight decay: 1e-6, confidence threshold: 0.4.
5. **Patch-based training** — 256×256 random crops with **vessel oversampling** (~70% of each batch contains crops with >1% vessel pixels) to boost vessel-pixel signal.
6. **Encoder freezing** — Backbone is frozen for the first few epochs so the decoder warms up against pretrained features, then unfrozen for fine-tuning.
7. **Evaluation** — Dice coefficient, mean IoU (manual foreground+background averaging), pixel accuracy, recall, and F1 score.

## Results

Best validation metrics from training (epoch 34, threshold 0.4):

| Metric    | Value  |
|-----------|--------|
| Dice      | 0.6950 |
| mIoU      | 0.7495 |
| IoU (fg)  | 0.5637 |
| IoU (bg)  | 0.9352 |
| Accuracy  | 0.9402 |
| Recall    | 0.8302 |
| F1        | 0.7210 |
| Eval loss | 0.5026 |

After **threshold sweeping with 4-way flip TTA**, the optimal sigmoid cutoff was **0.575**, lifting performance to:

| Metric   | Value  |
|----------|--------|
| Dice     | 0.7435 |
| mIoU     | 0.7692 |
| IoU (fg) | 0.5918 |
| IoU (bg) | 0.9467 |

TTA + threshold tuning added roughly **+5 points of Dice** and **+2 points of mIoU** with no extra training.

## Key Techniques

- U-Net decoder over a pretrained EfficientNet-B3 encoder (`timm`)
- Patch-based training with vessel-aware oversampling via custom `collate_fn`
- Encoder freeze/unfreeze schedule through a custom `TrainerCallback`
- Cosine LR schedule with warmup, plus early stopping
- MONAI `DiceFocalLoss` for class imbalance
- 4-way flip **Test-Time Augmentation (TTA)** (identity, hflip, vflip, hflip+vflip)
- **Threshold sweeping** on the validation set to pick the optimal sigmoid cutoff post-training
- Best model checkpointing by Dice score
