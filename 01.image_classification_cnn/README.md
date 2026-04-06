# Image Classification with CNNs — German Traffic Sign Recognition

## Overview

This experiment tackles multi-class image classification using Convolutional Neural Networks (CNNs) on the [German Traffic Sign Recognition Benchmark (GTSRB)](https://benchmark.ini.rub.de/gtsrb_news.html). The goal is to build a model capable of accurately identifying 43 different types of traffic signs under real-world conditions such as varying lighting, weather, and viewpoints.

The baseline architecture follows the well-known **Conv-Batch-ReLU (CBR) + MaxPooling** pattern. The CBR module is a specialized layer that works with images by creating feature maps that retain specific features such as edges, textures, shapes, and objects. MaxPooling reduces the spatial dimensions, preserving the activations of the CBR modules while reducing the parameter count — effectively creating a compact tensor suitable as input to a Feed-Forward Network (FFN) for classification.

## Spatial Transformer Layer

This approach is extended with a **Spatial Transformer Layer**, which learns to apply geometric corrections to the input before it reaches the classifier. To understand it, consider the affine transformation:

$$
\begin{bmatrix}
x' \\
y'
\end{bmatrix} =
\begin{bmatrix}
a & b \\
c & d
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix}
+
\begin{bmatrix}
t_x \\
t_y
\end{bmatrix}
$$

where:

- $a$, $b$, $c$, and $d$ encode linear operations such as rotation, scaling, and shear.
- $t_x$ and $t_y$ represent translation.
- $(x, y)$ is the original point and $(x', y')$ is the transformed point.

The Spatial Transformer Layer predicts these affine parameters using a CBR block and an FFN, warping the image before feeding it to the classification CNN. This helps minimize the overall cost function and allows the network to handle rotated, skewed, or perspective-distorted inputs. It can also be used to downsample or upsample a feature map by setting output dimensions independently from the input.

## Dataset

- **Source:** GTSRB (German Traffic Sign Recognition Benchmark)
- **Classes:** 43 traffic sign categories
- **Size:** 50,000+ images
- **Input resolution:** 64×64 pixels (3 channels RGB)

## Training Configuration

| Parameter | Value |
|---|---|
| Batch size | 32 |
| Epochs | 30 |
| Learning rate | 3e-3 |
| Weight init | Xavier / He |

## Evaluation Metrics

Accuracy, precision, recall, and F1 score — reported per class and globally.

## Files

- `image_classification_cnn_GTSRB.ipynb` — End-to-end pipeline: data exploration → model training → evaluation.
