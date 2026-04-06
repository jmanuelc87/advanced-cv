# Optical Character Recognition — Curved Text with TrOCR

## Overview

This experiment implements OCR using a transformer-based architecture (**TrOCR**) to recognize curved and non-standard text from scene images. Curved text is significantly harder than horizontal text, requiring models that understand flexible spatial layouts.

## Dataset

- **Source:** SCUT (Scene-text Understanding) curved text dataset
- **Train:** ~22 images with labels
- **Test:** ~1,665 images for evaluation
- **Text characteristics:** Curved, tilted, or irregularly placed text in natural scenes

## Approach

1. **Model** — `microsoft/trocr-base-printed`, a Vision Encoder-Decoder architecture combining a ViT-based image encoder with a transformer text decoder.
2. **Preprocessing** — Images are resized and normalized; labels are tokenized and padded to a maximum length of 25 characters.
3. **Training** — Batch size: 12, Epochs: 5, Learning rate: 1e-5, FP16 precision for efficiency.
4. **Evaluation** — Character Error Rate (CER) measures how closely predicted text matches the ground truth.

## Results

- **Mean CER on test set:** ~0.403

## Key Techniques

- Transformer-based sequence-to-sequence OCR
- Hugging Face `Trainer` API for distributed training
- Weights & Biases (wandb) for experiment tracking
- Dataset caching and preprocessing pipelines
- Early stopping and best-model checkpointing

## Files

- `ocr_curved_text.ipynb` — End-to-end pipeline: dataset loading → model fine-tuning → CER evaluation.
