# TensorFlow 3D
[![TensorFlow 2.2](https://img.shields.io/badge/TensorFlow-2.2-FF6F00?logo=tensorflow)](https://github.com/tensorflow/tensorflow/releases/tag/v2.2.0)
[![Python 3.6](https://img.shields.io/badge/Python-3.6-3776AB)](https://www.python.org/downloads/release/python-360/)

Creating accurate machine learning models capable of localizing and identifying
multiple objects in a 3D scene and assigning semantic labels to the scene components is a core challenge in computer vision with applications in robotics and autonomous driving. The TensorFlow 3D codebase is an open source framework built on top of TensorFlow 2 and Keras that makes it easy to construct, train and deploy 3D Object Detection, 3D Semantic Segmentation and 3D Instance Segmentation models. At Google we’ve certainly found this codebase to be useful for our computer vision needs, and we hope that you will as well.

<p align="center"><img src="doc/img/tf3d.png" width=676 height=254></p>

Contributions to the codebase are welcome and we would love to hear back from
you if you find this codebase useful. Finally if you use the TensorFlow 3D for a research publication, please consider citing:

* <a href='https://arxiv.org/abs/2004.01170' target='_blank'>DOPS: Learning to Detect 3D Objects and Predict their 3D Shapes</a>,
<em>Mahyar Najibi, Guangda Lai, Abhijit Kundu, Zhichao Lu, Vivek Rathod, Tom Funkhouser, Caroline Pantofaru, David Ross, Larry Davis, Alireza Fathi, CVPR 2020</em>

* <a href='https://arxiv.org/abs/2007.12392' target='_blank'>An LSTM Approach to Temporal 3D Object Detection in LiDAR Point Clouds</a>, <em>Rui Huang, Wanyue Zhang, Abhijit Kundu, Caroline Pantofaru, David A Ross, Thomas Funkhouser, Alireza Fathi, ECCV 2020</em>

# Release Notes

This release includes:

* GPU/CPU op for 3d submanifold sparse convolution.
* A configurable 3d sparse voxel unet network that is used as the feature extractor in our models.
* Training and evaluation code for 3D Semantic Segmentation, 3D Object Detection and 3D Instance Segmentation.
* Data and configuration for training and evaluation on <a href='https://waymo.com/open/' target='_blank'>Waymo Open Dataset</a>, <a href='http://www.scan-net.org/' target='_blank'>ScanNet Dataset</a>, and <a href='https://waldjohannau.github.io/RIO/' target='_blank'>Rio Dataset</a>.

# Resources

* <a href='doc/usage.md' target='_blank'>Requirements, Installation and Usage</a>
* <a href='doc/tf3d_datasets.md' target='_blank'>Datasets</a>
* <a href='doc/models.md' target='_blank'>TensorFlow 3D Model</a>
* <a href='ops/README.md' target='_blank'>Preparing and Compiling the Sparse Conv Op</a>

# Maintainers

* <a href='https://sites.google.com/corp/view/ruihuang/home' target='_blank'>Rui Huang</a> (<a href='https://github.com/HRLTY' target='_blank'>@GitHub HRLTY</a>)
* <a href='https://www.alirezafathi.org/' target='_blank'>Alireza Fathi</a> (<a href='https://github.com/afathi3' target='_blank'>@GitHub afathi3</a>)

# Acknowledgement

We thank <a href='https://www.linkedin.com/in/guangda-lai-31a5ab53/?originalSubdomain=cn' target='_blank'>Guangda Lai</a> and <a href='https://abhijitkundu.info/' target='_blank'>Abhijit Kundu</a> for their contributions to this code. We also like to thank <a href='https://www.cs.princeton.edu/~funk/' target='_blank'>Thomas Funkhouser</a>, <a href='http://www.cs.toronto.edu/~dross/' target='_blank'>David Ross</a> and <a href='https://www.linkedin.com/in/carolinepantofaru/' target='_blank'>Caroline Pantofaru</a> for very insightful and helpful discussions throughout this project. Finally we thank <a href='https://www.linkedin.com/in/pei-sun-4a817816/' target='_blank'>Pei Sun</a> for helping us with the Waymo Open dataset, <a href='https://scholar.google.de/citations?user=dfjN3YAAAAAJ&hl=en' target='_blank'>Johanna Wald</a> for her help with the Rio dataset and <a href='https://www.3dunderstanding.org/team.html' target='_blank'>Angela Dai</a> and <a href='https://www.niessnerlab.org/' target='_blank'>Matthias Niessner</a> for their help with the ScanNet dataset.

