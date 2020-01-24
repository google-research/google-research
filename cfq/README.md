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

In order to run a different model or try a different split, simply modify the
parameters in `run_experiment.sh`. See that file for additional details.

For the expected accuracies of the other splits and architectures, please see
the paper (Table 4). In the paper we report the averages and confidence
intervals based on 5 runs. For the MCD splits, these numbers vary between MCD1,
MCD2, and MCD3, and the numbers reported in Table 4 are the averages over the 3
splits. Accuracies vary between 5% and 37% over splits and architectures:

|      | LSTM+attention | Transformer | Universal Transformer |
|-------|--------------|--------------|--------------|
| MCD1  | 28.9 +/- 1.8 | 34.9 +/- 1.1 | 37.4 +/- 2.2 |
| MCD2  |  5.0 +/- 0.8 |  8.2 +/- 0.3 |  8.1 +/- 1.6 |
| MCD3  | 10.8 +/- 0.6 | 10.6 +/- 1.1 | 11.3 +/- 0.3 |
