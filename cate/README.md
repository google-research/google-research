# Composable Augmentation Encoding for Video Representation Learning (CATE)

This directory contains a colab to run inference with a pretrained CATE model. The model is used
to extract self-supervised video representations, and we also provide demo code
for nearest neighbor classification on the UCF-101 dataset.

For demo purposes we use the CATE model pretrained on the Kinetics-400 dataset.
The checkpoints (exported with tfhub) for Something-Something v1 and v2 are also
available in the same Google Cloud Storage directories.

For more information, please refer to our [ICCV 2021 paper](https://arxiv.org/abs/2104.00616)
and [project page](https://sites.google.com/corp/brown.edu/cate-iccv2021/).

If you use the code and the released checkpoints, please consider citing:
```
@InProceedings{Sun_CATE_CVPR2021,
author = {Chen Sun and Arsha Nagrani and Yonglong Tian and Cordelia Schmid},
title = {Composable Augmentation Encoding for Video Representation Learning},
booktitle = {ICCV},
year = {2021},
}
```
