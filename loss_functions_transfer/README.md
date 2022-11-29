## Why Do Better Loss Functions Lead to Less Transferable Features?

This repository contains code to load models from
[Why Do Better Loss Functions Lead to Less Transferable Features?](https://arxiv.org/abs/2010.16402)
as well as implementations of the loss functions.

The model checkpoints and [SavedModels](https://www.tensorflow.org/api_docs/python/tf/saved_model)  can be found at [this link](https://console.cloud.google.com/storage/browser/gresearch/loss_functions_transfer). `load_model.py` contains code to construct model graphs and load checkpoints in graph mode (i.e., TensorFlow 1 compatibility mode).

Alternatively, it is also possible to load and run the SavedModels directly in TensorFlow 2 without the need to use any code from this repository besides the preprocessing code. The SavedModel returns the activations from each block in the ResNet, the penultimate (average pooling) layer (`final_avg_pool`), the final layer (`final_layer`), and the normalized/scaled model outputs (`outputs`). Note that the model outputs contain 1001 classes, one more than are actually present in ImageNet. This is because the copy of ImageNet we used to train the model added an additional background class as the first class. This class contains no images and should be dropped when evaluating.

See [this Colab](https://colab.research.google.com/github/google-research/google-research/blob/master/loss_functions_transfer/load_models.ipynb) for example code for how to use the SavedModels and restore the checkpoints.

Please cite as:
```
@inproceedings{
kornblith2021why,
title={Why Do Better Loss Functions Lead to Less Transferable Features?},
author={Simon Kornblith and Ting Chen and Honglak Lee and Mohammad Norouzi},
booktitle={Advances in Neural Information Processing Systems},
editor={A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
year={2021},
url={https://openreview.net/forum?id=8twKpG5s8Qh}
}
```

### Disclaimer

This is not an official Google product.
