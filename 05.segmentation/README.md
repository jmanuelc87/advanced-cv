# Semantic Segmentation — Retinal Blood Vessel Detection

## Overview

This experiment implements **semantic segmentation** to identify blood vessels in retinal fundus images. Accurate vessel segmentation is critical for diagnosing eye diseases such as diabetic retinopathy and glaucoma. The model performs pixel-level binary classification: each pixel is labeled as either **vessel** or **background**.

## Dataset

- **Source:** ICPR Retinal Blood Vessel Dataset
- **Train:** 268 images + masks
- **Test:** 112 images + masks
- **Classes:** 2 — background, blood vessel
- **Input resolution:** 640×640 pixels

## Approach

1. **Architecture** — Encoder-decoder network with pretrained backbones from the TIMM library (CNN and Vision Transformer variants).
2. **Augmentation** — Albumentations pipeline with geometric and color transforms to improve robustness.
3. **Loss functions** — Combination of Dice loss and Focal loss from the MONAI medical imaging library.
4. **Training** — Batch size: 4, Epochs: 40, Learning rate: 1e-4, confidence threshold: 0.4.
5. **Evaluation** — Dice coefficient, Jaccard Index (IoU), and pixel accuracy.

## Key Techniques

- Encoder-decoder segmentation architecture
- Pretrained TIMM backbones (ResNet, ViT, and others)
- Albumentations-based data augmentation
- MONAI Dice + Focal loss for class imbalance handling
- Metric tracking during training and validation
- Best model checkpointing and wandb experiment tracking

## Files

- `segmentation_blood_vessels.ipynb` — Complete implementation: dataset preparation → model training → evaluation and visualization.
- `Project3_Semantic_Segmentation_Starter_Notebook.ipynb` — Starter template with guided structure for learning.
- `retinal_blood_vessel_icpr_seg/` — Dataset directory (train/test images and masks).
