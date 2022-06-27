# Dataset Metalearning from Kernel Ridge-Regression

## Overview
This directory contains a publicly available colab notebook [KIP.ipynb](https://colab.research.google.com/github/google-research/google-research/blob/master/kip/KIP.ipynb) for the paper [Dataset Metalearning from Kernel Ridge-Regression](https://arxiv.org/abs/2011.00050)
by Timothy Nguyen, Zhourong Chen, and Jaehoon Lee, published in ICLR 2021. The colab implements the Kernel Inducing Points (KIP)
and Label Solve (LS) algorithms.

We also release the datasets constructed in [Dataset Distillation with
Infinitely Wide Convolutional Networks](https://arxiv.org/abs/2107.13034) by Timothy Nguyen, Roman Novak,
Lechao Xiao, Jaehoon Lee. We distill datasets using the KIP algorithm developed in the previous
paper together with large-scale distributed computational resources to handle convolutional networks. See [dataset.ipynb](https://colab.research.google.com/github/google-research/google-research/blob/master/kip/dataset.ipynb) for access to the datasets stored in GCS.

An example of finite network (ConvNet) training using one of the distilled data in GCS is also included in [Neural Network Training Notebook](https://colab.research.google.com/github/google-research/google-research/blob/master/kip/nn_training.ipynb).


## Citation

If you find this code or paper useful, please cite:

```
@inproceedings{
nguyen2021dataset,
title={Dataset Meta-Learning from Kernel-Ridge Regression},
author={Timothy Nguyen and Zhourong Chen and Jaehoon Lee},
booktitle={International Conference on Learning Representations},
year={2021},
url={https://openreview.net/forum?id=l-PrrQrK0QR}
}
```

```
@misc{nguyen2021dataset,
      title={Dataset Distillation with Infinitely Wide Convolutional Networks}, 
      author={Timothy Nguyen and Roman Novak and Lechao Xiao and Jaehoon Lee},
      year={2021},
      eprint={2107.13034},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Contact

Please send pull requests and issues to Timothy Nguyen
([@timothycnguyen](https://github.com/timothyn617)) or Jaehoon Lee
([@jaehlee](https://github.com/jaehlee))


## Disclaimer

This is not an officially supported Google product.
