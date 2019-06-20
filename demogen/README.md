# DEep MOdel GENeralization dataset (DEMOGEN)


This codebase contains code necessary for using the generalization dataset used
in "Predicting the Generalization Gap in Deep Networks with Margin
Distributions" (ICLR 2019) https://arxiv.org/abs/1810.00113

## Overview
The DEMOGEN dataset consists of 756 trained deep models, along with their training and test performance on the CIFAR-10 and CIFAR-100 datasets. The models are variants of CNNs (with architectures that resemble Network-in-Network) and ResNet-32 with different regularization techniques and hyperparameter settings.

## Variations
The variations available in DEMOGEN are among the most common techniques used by practitioners. For example, we apply weight decay and dropout with different strengths; we use networks with and without batch normalization (and group normalization for ResNet) and data augmentation; we change the width or the number of hidden units in the hidden layers; we explore different initial learning rates (for ResNet). These variations induce a wide spectrum of generalization behaviors. For example, the models of CNNs trained on CIFAR-10 have the test accuracies ranging from 60% to 90.5%, and the generalization gaps ranging from 1% to 35%.

## USAGE
A typical use case can be found in `example.py`. Run this by `python -m demogen.example`.

Examples of computing the margin and total variation on the dataset can be found
in the docstring of `margin_utils.py` and `total_variation_util.py`.

## Dataset:
Downaload and unzip the dataset from: [Link](https://storage.googleapis.com/margin_dist_public_files/demogen_models.tar.gz)
Total size of the model is approximately 15.57GB.

If you find this dataset useful, please consider cite our paper using:

```
@inproceedings{
jiang2018predicting,
title={Predicting the Generalization Gap in Deep Networks with Margin Distributions},
author={Yiding Jiang and Dilip Krishnan and Hossein Mobahi and Samy Bengio},
booktitle={International Conference on Learning Representations},
year={2019},
url={https://openreview.net/forum?id=HJlQfnCqKX},
}
```

If you run into any problems using this dataset, please file an GitHub issue.

Disclaimer: This is not an official Google product.
