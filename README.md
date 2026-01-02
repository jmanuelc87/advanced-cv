# Advanced Computer Vision Topics

### 1. Image Classification using CNNs

In this [module](01.image_classification_cnn) we start by creating an image classifier using the well known Conv-Batch-Relu and MaxPooling modules this will set a baseline for image classification using the GTSRB dataset. The CBR module is a specilized layer that works with images by creating features maps that retain specifc features such as edges, textures, shapes and objects in contrast the MaxPooling layer reduces the spatial dimensions preverving the activations of the CBR modules while reducing the spatial dimensions allowing to reduce the parameter of the model and effectively creating a tensor that can be used as input to a FFN (Feed Forward Network) for classifying the image.

This approach is effective to classify images and will work as an standard way to extract features from images and creating backbones for other models and tasks while this approach is effective it can be pushed a little bit using a Spatial Tranformation Layer, before going into the details of this layer let's explain what is an affine transformation.

The affine transformation can be expressed in the form os a matrix multiplication followed by a vector addition and we can use it to express rotations (linear transformation), translations (vector additions), and scale operations (linear transformations), in essence an affine transformation represents a relation between two images preservesing the proportions on lines, while it does not necessarily preverves angles or lengths and mathematically can be represented as:

$$
\begin{bmatrix}
x' \\
y'
\end{bmatrix}
=
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
 - $(x,y)$ is the original point and $(x',y')$ is the transformed point

The Spatial Transformation Layer effectively tries to predict the affine transformation values by using a CBR block and FFN transforming the image before feeding it again the the classification CNN. The Spatial Transformation Layer helps minimizing the overall cost function of the network during training and It is also possible ot use spatial transformer to downsample or oversample a feature map, as one can define the output dimensions to be different to the input dimensions.

### 2. Transfer Learning

In this second [module](02.transfer_learning) I used transfer learning for detecting COVID-19 using the covid 19 radiography database, this technique is particularly useful because given a baseline model like ResNet or EfficientNet trained on millions of images I'm able to transfer the task and reuse the layers to a different but related task previously unknown to this models.

ResNet or EfficientNet are trained on datasets like imagenet but in this dataset aren't any medical images which is purposeful to transfer this learned parameters and fine-tune the model for the new task giving better results regardless training from scratch.


### 3. Object Detection

In this [module](03.object_detection/) the task was to create an object detector for classifying traffic lights considering and urban environment the images were in high resolution and variance of light conditions during day and night the sizes of the objects to be detected fall into the small category with a mean of 50px width and height, this is because the lights are not only in vertical format but also horizontal, the varying sizes goes from 20px to 300px width and 20px to 150px height. The number classes are 6 and fall into the following labels `green`, `off`, `red`, `wait_on` and `yellow`.
For this task I used the ultralytics library in specific the yolo11 model because of its strong backbone modules, and the neck architecture featuring a bi-directional feature pyramid network, this model also gives an excelent tradeoff between mAP and FPS.

The featuring relevant blocks that brings with this model are:

  1. Conv-Batch-SiLU: this combination are execelent for extracting features from the image by stacking multiple layers allows to learn patterns such as edges, textures, shapes and objects. Hierarchical learning for capturing increasingly abstract and complex features, from low-level edges in early layers to high-level representation in deeper layers. Progressive downsampling reducing the spatial dimensions of the input image by half while increating the channel depth this process ensures efficient input reduction minimizing computational costs while retaining critical information and preserves spatial relationship maintaining the structure and arrangement of data ensuring meaninfull features are retained.
  2. C3k2: behaves different reagarding the c3 boolean parameter.
     2.1 When set to **False** the block goes through several phases which are compression, processing, and expansion.
        - The compression phase squeezes the features reducing the input channels distilling the most critical information while discarding less important features
        - The processing phase transforms the input using convolutions and activations to refine patterns and emphasizes core patterns while preserving computational resources
        - The expansion pahse expands the channels to ensure the network retains capacity for complex pattern modeling and combines critical features from compression phase with the structural richness needed for downstream tasks
     2.2 When set to **True** the block undergoes through a multi-stage processing
        - Refining the input through a set of nested bottleneck blocks sequentially transforming the compressed features to emphsize core patterns while discarding redundancies
        - The output of the bottlenecks are recombined and expanded through concatentation and a final convolution ensuring a balance of dimensionality and feature richness.
  3. SPPF (Spatial Pyramid Pooling - Fast): Aggregates features from multiple scales by applying a max-pooling operation with kernel size = 5, then concatenates the outputs to create a rich, multi-scale feature map enhancing the model's abilty to detect objects of varying sizes. Preserves spatial relationships while reducing resolution ensuring compact and meaningful feature representation.
  4. C2PSA: splits input features into two paths one for convolutional operation and the other for attention transformations based on the PSABlock module allowing to capture both local and global feature dependecies. The attention mechanism focus on the most relevant spatial features improving the network's ability to model long-range dependencies and subtle feature interactions and performing a fusion of features that combines the outputs of convolutional and attention pathways to produce richer and more expressive feature maps enhancing the model to detect complex patterns.
