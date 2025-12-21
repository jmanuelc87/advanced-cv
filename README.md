# Advanced Computer Vision Topics


### 1. Image Classification using CNNs

The folder [Image Classification](01.image_classification_cnn) contains the notebook for classifying images using the GTSRB dataset the baseline model uses the layers:
 - Convolution
 - Batch Normalization
 - ReLU
 - Max Pooling

Above the baseline is utilized the spatial transformer layer used for increasing the accuracy of the network.


### 2. Transfer Learning

In this second module [link](02.transfer_learning) I used transfer learning for detecting COVID-19 using the covid 19 radiography database, this technique is particularly useful because given a baseline model like ResNet or EfficientNet trained on millions of images I'm able to transfer the task and reuse the layers to a different but related task previously unknown to this models.

ResNet or EfficientNet are trained on datasets like imagenet but in this dataset aren't any medical images which is purposeful to transfer this learned parameters and fine-tune the model for the new task giving better results regardless training from scratch.
