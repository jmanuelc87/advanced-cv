# Transfer Learning — COVID-19 Detection from Chest X-Rays

## Overview

This experiment applies transfer learning to classify chest X-ray images into four categories: **COVID-19**, **Normal**, **Lung Opacity** (non-COVID infections), and **Viral Pneumonia**.

Transfer learning is particularly useful here because models like ResNet and EfficientNet are trained on millions of images (e.g., ImageNet) and learn general visual representations — edges, textures, shapes. Even though ImageNet contains no medical images, those learned parameters transfer remarkably well to a related but previously unseen task. By freezing early layers and fine-tuning only the later ones, the model adapts to chest X-ray classification with far less data and compute than training from scratch.

## Dataset

| Class | Images |
|---|---|
| COVID-19 | 3,616 |
| Normal | 10,192 |
| Lung Opacity (non-COVID) | 6,012 |
| Viral Pneumonia | 1,345 |
| **Total** | **~21,000** |

- **Input resolution:** 224×224 pixels

## Approach

1. **Backbone selection** — Two pretrained architectures are compared: ResNet34 and EfficientNet B4.
2. **Fine-tuning strategy** — Layers are selectively frozen so early feature extractors are preserved while later layers adapt to medical imaging.
3. **Training** — Batch size: 16, Epochs: 5, Learning rate: 1e-3 with OneCycleLR scheduler.
4. **Evaluation** — Accuracy, recall, precision, F1 score, and confusion matrix.

## Results

| Model | Accuracy |
|---|---|
| ResNet34 | ~92.9% |
| EfficientNet B4 | ~92.1% |

## Key Techniques

- Transfer learning with partial layer freezing
- OneCycleLR learning rate scheduling
- Data augmentation: affine transforms, random rotations and scaling
- Custom `DetectorNet` wrapper for flexible backbone swapping

## Files

- `transfer_learning_COVID19.ipynb` — Main notebook with full pipeline.
- `transfer_learning_COVID19.py` — Script version for batch execution.
- `common.py` — Shared training utilities.
- `dev_config.py` / `local_config.py` — Training configuration files.
