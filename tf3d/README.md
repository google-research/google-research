# TensorFlow 3D
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)

Creating accurate machine learning models capable of localizing and identifying
multiple objects in a 3D scene and assigning semantic labels to the scene components is a core challenge in computer vision with applications in robotics and autonomous driving. The TensorFlow 3D codebase is an open source framework built on top of TensorFlow 2 and Keras that makes it easy to construct, train and deploy 3D Object Detection, 3D Semantic Segmentation and 3D Instance Segmentation models. At Google we’ve certainly found this codebase to be useful for our computer vision needs, and we hope that you will as well.

<p align="center"><img src="doc/img/tf3d.png" width=676 height=254></p>

Contributions to the codebase are welcome and we would love to hear back from
you if you find this codebase useful. Finally if you use the TensorFlow 3D for a research publication, please consider citing:

* [DOPS: Learning to Detect 3D Objects and Predict their 3D Shapes](https://arxiv.org/abs/2004.01170), 
<em>Mahyar Najibi, Guangda Lai, Abhijit Kundu, Zhichao Lu, Vivek Rathod, Tom Funkhouser, Caroline Pantofaru, David Ross, Larry Davis, Alireza Fathi, CVPR 2020</em>

* [An LSTM Approach to Temporal 3D Object Detection in LiDAR Point Clouds](https://arxiv.org/abs/2007.12392), <em>Rui Huang, Wanyue Zhang, Abhijit Kundu, Caroline Pantofaru, David A Ross, Thomas Funkhouser, Alireza Fathi, ECCV 2020</em>

# Release Notes

This release includes:

* GPU/CPU op for 3d submanifold sparse convolution.
* A configurable 3d sparse voxel unet network that is used as the feature extractor in our models.
* Training and evaluation code for 3D Semantic Segmentation, 3D Object Detection and 3D Instance Segmentation.
* Data and configuration for training and evaluation on [Waymo Open Dataset](https://waymo.com/open/), [ScanNet Dataset](http://www.scan-net.org/), and [Rio Dataset](https://waldjohannau.github.io/RIO/).

# Resources

* [Requirements, Installation and Usage](doc/usage.md)
* [Datasets](doc/tf3d_datasets.md)
* [TensorFlow 3D Model](doc/models.md)
* [Preparing and Compiling the Sparse Conv Op](ops/README.md)

# Maintainers

* [Rui Huang](https://sites.google.com/corp/view/ruihuang/home) ([@GitHub HRLTY](https://github.com/HRLTY))
* [Alireza Fathi](https://www.alirezafathi.org/) ([@GitHub afathi3](https://github.com/afathi3))

# Acknowledgement

We thank [Guangda Lai](https://www.linkedin.com/in/guangda-lai-31a5ab53/?originalSubdomain=cn) and [Abhijit Kundu](https://abhijitkundu.info/) for their contributions to this code. We also like to thank [Thomas Funkhouser](https://www.cs.princeton.edu/~funk/), [David Ross](http://www.cs.toronto.edu/~dross/) and [Caroline Pantofaru](https://www.linkedin.com/in/carolinepantofaru/) for very insightful and helpful discussions throughout this project. Finally we thank [Pei Sun](https://www.linkedin.com/in/pei-sun-4a817816/) for helping us with the Waymo Open dataset, [Johanna Wald](https://scholar.google.de/citations?user=dfjN3YAAAAAJ&hl=en) for her help with the Rio dataset and [Angela Dai](https://www.3dunderstanding.org/team.html) and [Matthias Niessner](https://www.niessnerlab.org/) for their help with the ScanNet dataset.

