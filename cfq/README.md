# Complex Freebase Questions (CFQ) Tools

This repository contains code for training and evaluating ML architectures on
the Complex Freebase Questions (CFQ) dataset.

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

In order to download the dataset, train and evaluate a model, run the following:

```shell
bash run_experiment.sh
```

This will download the dataset, run preprocessing on the dataset, and train an
LSTM model with attention on the random split of the CFQ dataset, after which it
will directly be evaluated.

NOTE This may take quite long and consume a lot of memory. It is tested on a
machine with 6-core/12-hyperthread CPUs at 3.7Ghz, and 64Gb RAM, which took
about 20 hours. Also note that this will consume roughly 20Gb of RAM during
preprocessing. The run-time can be sped up significantly by running Tensorflow
with GPU support.

The expected accuracy using the default setting of the script is 97.4 +/- 0.3.
For the other expected accuracies of the other splits, please see the paper.

In order to run a different model or try a different split, simply modify the
parameters in `run_experiment.sh`. See that file for additional details.
