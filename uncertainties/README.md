# Uncertainties for classification in Deep Neural Networks

__Warning: work in progress__

This project focuses on computing uncertainties in Deep Neural Networks for
multi-class classification tasks.

Authors: Nicolas Brosse, Carlos Riquelme.

The code is implemented in [Tensorflow](https://www.tensorflow.org/) and the
required packages are in `requirements.txt`.

## Data

### Datasets

Four datasets are necessary to execute the code: MNIST, NOTMNIST, CIFAR10,
CIFAR100. The data should be organized in the following way:

```
baseroute/data/mnist/mnist.npz
baseroute/data/notmnist/labels_test_notmnist.npy
baseroute/data/notmnist/pictures_test_notmnist.npy
baseroute/data/cifar10_data/cifar10_batches_py
baseroute/data/cifar100_data/cifar-100-python
```

where `baseroute` is a base route directory that has to be appropriately
modified in the code.

For NOTMNIST, several links exist to download and preprocess the data:
[1](https://leemeng.tw/simple-image-recognition-using-notmnist-dataset.html)
[2](https://github.com/davidflanagan/notMNIST-to-MNIST)
[3](http://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html)
[4](https://www.ritchieng.com/machine-learning/deep-learning/tensorflow/notmnist/).
For simplicity, the test dataset will be available.

For MNIST, use for example the
[keras API](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist)
to download a numpy file `mnist.npz` that contains the training and test
datasets.

For CIFAR10/100, please refer to this
[link](https://www.cs.toronto.edu/~kriz/cifar.html) to download the python
versions of CIFAR10 and CIFAR100.

### Features and weights

The models `models.7z` will be available shortly. 
It contains the features of the last layer associated with
the different datasets MNIST, NOTMNIST, CIFAR10/100 and the weights of the
pretrained neural network. See this [paragraph](training) for more information
about the features and training the networks.

## Algorithms and metrics

We propose 4 different algorithms acting on the last layer of a neural network
(the rest of the network is frozen) to compute uncertainties. They are located
in `uncertainties/sources/models` and use the pre-computed features.

*   `bootstrap.py`, a bootstrap algorithm.
*   `dropout.py`, a Monte Carlo dropout algorithm.
*   `simple.py`, a MCMC algorithm using Stochastic Gradient Descent (SGD) or
    SGLD (Stochastic Gradient Langevin Dynamics).
*   `precond.py`, a preconditioned version of the `simple.py` algorithm to try
    to accelerate the convergence. Note: may be unstable numerically. Work in
    progress.

Execution: using the script `scripts/train_local.sh`, work in progress.

## Training the neural networks and computing the features <training>

For CIFAR10/100, we trained the neural network using the
[tensorflow tutorial](https://www.tensorflow.org/tutorials/images/deep_cnn), see
also the associated
[github page](https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10).
To compute the features of the last but one layer, execute

```
python -m features_cifar.py
```

in the folder `uncertainties/sources/cifar`.

For MNIST/NOTMNIST, we implemented a simple fully connected feedforward network
with two hidden layers of size 512 and 20 respectively. To train the network and
compute the features of the last but one layer, execute

```
python -m train_features_mnist.py
```

in the folder `uncertainties/sources/mnist`.
