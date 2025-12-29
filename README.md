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

### 3. Object Detection

In the object detection task [link](03.object_detection/) I used the ultralytics library for detecting Traffic Lights using the yolo11 model. I choose this model because the robust backbone components formed from Conv-Batch-SiLU and C3k2 convolution layers the former is excellent for extracting features, hierarchical learning and progressive downsampling from the images the latter is new block that behaves different regarding the boolean parameter c3. The C3k2 is used for compressing or squeezeing the features, then it refines the patterns using a set of convolutions and activations and finally expands the channels back to ensure the network retains capacity for complex pattern modeling. At the end of the backbone is used two blocks SPPF (Spatial Pyramid Pooling Fast) which introduces multiscale agregation applying max pooling operations to capture spatial information at multiple scales then concatenates the outputs to create a feature map to enhance the network ability to detect objects of varying sizes and the C2PSA block which introduced an attention mechanism to focus on the most relevant spatial features improving the model's ability to model long range dependencies.
