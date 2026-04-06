# Object Detection — Traffic Light State Recognition with YOLOv11

## Overview

This experiment trains an object detection model to locate and classify traffic lights in urban environments. Images are high-resolution with varying lighting conditions across day and night. The objects to detect are **small** — mean size ~50×50 px, ranging from 20 to 300 px wide and 20 to 150 px tall — and appear in both vertical and horizontal traffic light formats.

The task uses the **YOLOv11 Large** model via the Ultralytics library, chosen for its strong backbone, bi-directional Feature Pyramid Network (BiFPN) neck, and excellent trade-off between mAP and inference FPS.

## Dataset

- **Source:** Small Traffic Light v1 (YOLO format)
- **Train:** 855 images | **Validation:** 367 images
- **Classes:** 5 — `green`, `off`, `red`, `wait_on`, `yellow`
- **Input resolution:** 640×640 pixels

## Model Architecture — Key Blocks

### 1. Conv-Batch-SiLU
Stacked CBS layers extract hierarchical features — from low-level edges in early layers to high-level object representations in deeper layers. Progressive downsampling halves spatial dimensions while doubling channel depth, reducing computation while retaining critical spatial structure.

### 2. C3k2
Behavior controlled by a `c3` boolean flag:

- **`c3=False`** — Three phases: *compression* (squeeze channels to distill critical information), *processing* (refine patterns via convolutions/activations), and *expansion* (restore channel depth for downstream tasks).
- **`c3=True`** — Multi-stage processing through nested bottleneck blocks, followed by concatenation and a final convolution that balances dimensionality and feature richness.

### 3. SPPF (Spatial Pyramid Pooling — Fast)
Applies max-pooling with kernel size 5 at multiple scales, then concatenates the outputs into a rich multi-scale feature map. Enhances detection of objects of varying sizes while preserving spatial relationships.

### 4. C2PSA
Splits features into two parallel paths — one for convolutions, one for attention (PSABlock). The attention path captures long-range spatial dependencies and subtle feature interactions. Outputs are fused to produce expressive feature maps that help detect complex patterns.

## Results

| Class | mAP@50 |
|---|---|
| Green | 0.895 |
| Red | 0.938 |
| Wait_on | 0.953 |
| Yellow | 0.916 |
| **Overall mAP@50:95** | **0.489** |

## Demo

[![Traffic Light Detection](https://img.youtube.com/vi/IZ-DNqnv66E/0.jpg)](https://www.youtube.com/watch?v=IZ-DNqnv66E)

## Files

- `object_detection_traffic_light.ipynb` — Full pipeline: dataset exploration → training → validation → video inference.
- `runs.zip` — Training artifacts and model checkpoints.
