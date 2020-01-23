# Compositional Freebase Questions (CFQ) Tools

This repository contains code for training and evaluating ML architectures on
the Compositional Freebase Questions (CFQ) dataset.

The dataset can be downloaded from the following URL:

[Download the CFQ dataset](https://storage.cloud.google.com/cfq_dataset/cfq.tar.gz)

The dataset and details about its construction and use are described in this ICLR 2020 paper: [Measuring Compositional Generalization: A Comprehensive Method on Realistic Data](https://openreview.net/forum?id=SygcCnNKwr).

## Requirements

This library requires Python3 and the following Python3 libraries:

*   [absl-py](https://pypi.org/project/absl-py/)
*   [tensorflow](https://www.tensorflow.org/)
*   [tensor2tensor](https://github.com/tensorflow/tensor2tensor)

We recommend getting [pip3](https://pip.pypa.io/en/stable/) and then running the
following command, which will install all required libraries in one go:

```shell
sudo pip3 install absl-py tensorflow tensor2tensor
```

## Training and evaluating a model

First download the CFQ dataset (link above), and ensure the dataset and the
splits directory are in the same directory as this library (e.g. by unpacking
the file in the library directory). In order to train and evaluate a model,
run the following:

```shell
bash run_experiment.sh
```

This will run preprocessing on the dataset and train an LSTM model with
attention on the random split of the CFQ dataset, after which it will directly
be evaluated.

NOTE This may take quite long and consume a lot of memory. It is tested on a
machine with 6-core/12-hyperthread CPUs at 3.7Ghz, and 64Gb RAM, which took
about 20 hours. Also note that this will consume roughly 35Gb of RAM during
preprocessing. The run-time can be sped up significantly by running Tensorflow
with GPU support.

The expected accuracy using the default setting of the script is 97.4 +/- 0.3.
For the other expected accuracies of the other splits, please see the paper.

In order to run a different model or try a different split, simply modify the
parameters in `run_experiment.sh`. See that file for additional details.
