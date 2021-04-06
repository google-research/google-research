# dnn-predict-accuracy

This is the source code accompanying the paper "Predicting Neural Network
Accuracy from Weights", living at https://github.com/google-research/google-research/tree/master/dnn_predict_accuracy
If you find it useful, please cite:

```
  @misc{unterthiner2020predicting,
      title={Predicting Neural Network Accuracy from Weights},
      author={Thomas Unterthiner and Daniel Keysers and Sylvain Gelly and Olivier Bousquet and Ilya Tolstikhin},
      year={2020},
      eprint={2002.11448},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
  }
```

## Small DNN Zoo dataset

We release the dataset used in this study under the name "Small CNN Zoo dataset". It consists of instances of a specific Convolutional Neural Net (CNN) architecture that takes 1-channel input images as input, with 3 conv layers of 16 hidden units each, followed by global average pooling and 10-output nodes dense layer, using a total of 4970 parameters.

We trained 30k instances of this network on each of the four mentioned datasets, each one with different hyperparameters. We trained for a total of 86 epochs, and collected intermediate checkpoints at epoch 0 (initialization), 1, 2, 3, 20, 40, 60, 80 and 86 (end of training). For each of these 9 checkpoints, we take all the parameters of the network and flatten it to obtain a vector of length 4970. We then concatenate these into a large matrix (one per dataset), which we provide here in Numpy Matrix Format. Since a small number of training processes crashed/got stuck, we don’t have the full (30k * 9) rows in these matrices. The exact sizes are:

* [Cifar10](https://storage.cloud.google.com/gresearch/smallcnnzoo-dataset/cifar10.tar.xz): 270000 samples
* [Fashion-MNIST](https://storage.cloud.google.com/gresearch/smallcnnzoo-dataset/fashion_mnist.tar.xz): 270000 samples
* [MNIST](https://storage.cloud.google.com/gresearch/smallcnnzoo-dataset/mnist.tar.xz): 269973 samples
* [SVHN](https://storage.cloud.google.com/gresearch/smallcnnzoo-dataset/svhn_cropped.tar.xz): 269892 samples


For each of these matrices, we also have a “metrics.csv.gz” file that contains some annotation. Concretely, it contains the following columns:

```
config.activation: The activation function of the network
config.b_init: The initialization used for the bias weights
config.dataset: the dataset this was trained on
config.dnn_architecture: the network architecture (always “cnn”)
config.dropout: the dropout rate
config.epochs:  the total number of epochs (always 86)
config.epochs_between_checkpoints: the number of epochs between checkpoints (always 20)
config.init_std: the standard deviation for initialization if it isn’t determined by the init scheme (config.w_init)
config.l2reg: the amount of L2 weight decay
config.learning_rate: the learning rate
config.num_layers: the number of layers (always 3)
config.num_units: the number of hidden units per layer (always 16)
config.optimizer: the optimizer used
config.random_seed: the random seed used (always 42)
config.train_fraction: the fraction of the total number of training samples used (0.1, 0.25, 0.5 or 1.0)
config.w_init: the Initialization scheme used for the weights
modeldir: the internal directory used for training the network
step: the training step at which the checkpoint was taken
test_accuracy: the accuracy of this checkpoint on the test-set
test_loss: the loss of this checkpoint on the test-set
train_accuracy: the accuracy of this checkpoint on the training set
train_loss: the loss of this checkpoint on the training step
```

An additional file "layout.csv" maps the individual columns of the matrices to the corresponding layers in the neural network.


## License

This repository is licensed under the Apache License, Version 2.0. See LICENSE for details. The Small CNN Zoo dataset itself is licensed under a [CC-BY 4.0 license](https://creativecommons.org/licenses/by/4.0/).
Note: This is not an official Google product.
