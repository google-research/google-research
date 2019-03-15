# Multi-scale Homography Estimation using Deep Neural Network

__Warning: WORK IN PROGRESS. Currently the code is for demonstration purpose and does not run at the moment.__

This project extends the deep homography estimation method from DeTone et al. [1] with a multi-scale strategy. Given a pair of input images, a homography is first estimated at the lowest resolution and then is progressively refined at higher resolutions. The training can be conducted using a synthetic dataset derived from the MS-COCO benchmark or other image/video datasets by following the method from DeTone et al [1].

The code builds upon Tensorflow(https://www.tensorflow.org/).

[1] DeTone, Daniel, Tomasz Malisiewicz, and Andrew Rabinovich. "Deep image homography estimation." arXiv preprint arXiv:1606.03798 (2016).
https://arxiv.org/abs/1606.03798
