# Predicting Generalization in Deep Learning:

## About the competition

Welcome! At NeurIPS 2020, we run a competition aimed at furthering our understanding of generalization in deep learning. To do so, competitors are asked to submit a python function whose input is a trained neural network and its training data and output is a complexity measure or generalization predictor that quantifies how well the trained model generalizes on the test data. You can find general information of the competition at https://sites.google.com/view/pgdl2020/.

This repository opensource the dataset (trained models and associated statistics) used for the competition.

The competition data was split into several "tasks" (types of models and image dataset used), given to the competitors in three phases:

 * Phase 0: Public data was given to the competitors: they were able to download Task1 and Task2 to measure the performance of their metrics locally, before uploading submissions to our servers.
 * Phase 1: First online leaderboard, accessible at the beginning of the competion (also called _public leaderboard_). This leaderboard is composed of Task4 and Task5 and was used to compute the scores displayed on the leaderboard for the first phase of the competition. There was no Task3 released in this competition, but we keep the original numbering of the tasks to avoid any confusion.
 * Phase 2: Private leaderboard, only accessible in the last phase of the competition, where competitors can upload their very best metrics. Winners are determined only on their score on this leaderboard (to prevent overfitting of the public leaderboard, as usual). This phase is composed of Tasks6, Task7, Task8 and Task8.

## In this repository

### What is available?

We opensource everything necessary to compute the scores for new predictors:

 * The checkpoints for each model: one for the trained model, and one for the initial weights.
 * The dataset on which the model was trained (some data augmentation have been applied on top of this dataset for some tasks).
 * A JSON file describing the architecture of the model.
 * A model_configs.json describing the hyper-parameters used to train the models. This is useful to compute the metrics. Please note that the parameters that are the same for all models in the task are generally omitted from this file, as they have no effect on the metric.

All these are included in the zip files that can be downloaded with the links below.

### And what isn't?

We do not opensource the following:

 * The codebase used to train the models.

## Where's the data?

 * Public dataset (8.8 GB): [download link](http://storage.googleapis.com/gresearch/pgdl/public_data.zip)
 * Public leaderboard dataset (2.5 GB): [download link](http://storage.googleapis.com/gresearch/pgdl/phase_one_data.zip)
 * Private leaderboard dataset (9.0 GB): [download link](http://storage.googleapis.com/gresearch/pgdl/phase_two_data.zip)

## How to use the tasks.

The code for computing the metrics can be downloaded on the competition's [Codalab page](https://competitions.codalab.org/competitions/25301).

## Description of the tasks

### Public tasks

#### Task 1:

 * Model: VGG-like models, with 2 or 6 convolutional layers [conv-relu-conv-relu-maxpool] x 1 or x 3.  One or two dense layers of 128 units on top of the model. When dropout is used, it is added after each dense layer.
 * Dataset: Cifar10 (10 classes, 3 channels).
 * Training: Trained for at most 1200 epochs, learning rate is multiplied by 0.2 after 300, 600 and 900 epochs. Cross entropy and SGD with momentum 0.9. Initial learning rate of 0.001
 * Hparams: Number of filters of the last convolutional layer in [256, 512]. Dropout probability in [0, 0.5]. Number of convolutional blocks in [1, 3]. Number of dense layers (excluding the output layer) in [1, 2]. Weight decay in [0.0, 0.001]. Batch size in [8, 32, 512].

#### Task 2

 * Model: Network in Network. When dropout is used, it is added at the end of each block.
 * Dataset: SVHN (10 classes, 3 channels)
 * Training: Trained for at most 1200 epochs, learning rate is multiplied by 0.2 after 300, 600 and 900 epochs. Cross entropy and SGD with momentum 0.9. Initial learning rate of 0.01.
 * Hparams: Number of convolutional layers in [6, 9, 12], dropout probability in [0.0, 0.25, 0.5], weight decay in [0.0, 0.001], batch size in [32, 512, 1024].


### Public leaderboard


#### Task 4

 * Model: Fully convolutional with no downsampling. Global average pooling at the end of the model. Batch normalization (pre-relu) on top of each convolutional layer.
 * Dataset: CINIC10 (random subset of 40% of the original dataset).
 * Training: Trained for at most 3000 epochs, learning rate is multiplied by 0.2 after 1800, 2200 and 2400 epochs. Initial learning rate of 0.001.
 * Hparams: Number of parameters in [1M, 2.5M], Depth [4 or 6 conv layers], Reversed [True of False]. If False, deeper layers have more filters. If True, this is reversed and the layers closer to the input have more filters. Weight decay in [0, 0.0005]. Learning rate in [0.01, 0.001]. Batch size in [32, 256].

#### Task 5

Identical to Task 4 but without batch normalization.


### Private leaderboard

#### Task 6

 * Model: Network in networks.
 * Dataset: Oxford Flowers (102 classes, 3 channels), downsized to 32 pixels.
 * Training: Trained for 10000 epochs maximum,  learning rate is multiplied by 0.2 after 4000, 6000, and 8000 epochs. Initial learning rate: 0.01. During training, we randomly apply data augmentation (the cifar10 policy from AutoAugment) to half the examples.
 * Hparams: Weight decay in [0.0, 0.001], batch size in [512, 1024], number of filters in convolutional layer in [256, 512], number of convolutional layers in [6, 9, 12], dropout probability in [0.0, 0.25], learning rate in [0.1, 0.01].

#### Task 7

 * Model: Network in networks, with dense layer added on top on the global average pooling layer. Example: For a NiN with 256-256 dense, the global average pooling layer will have an output size of 256, and another dense layer of 256 units is added on top of it before the output layer.
 * Dataset: Oxford pets (37 classes, downsized to 32 pixels, 4 channels). During training, we randomly apply data augmentation (the cifar10 policy from AutoAugment) to half the examples.
 * Training: Trained for at most 5000 epochs, learning rate is multiplied by 0.2 after 2000, 3000 and 4000 epochs. Initial learning rate of 0.1.
 * Hparams: Depth in [6, 9], dropout probability in [0.0, 0.25], weight decay in [0.0, 0.001], batch size in [512, 1024], dense architecture in [128-128-128, 256-256, 512]


#### Task 8

 * Model: VGG-like models (same as in task1) with one hidden layer of 128 units.
 * Dataset: Fashion MNIST (28x28 pixels, one channel).
 * Training: Trained for at most 1800 epochs, learning rate is multiplied by 0.2 after 300, 600 and 900 epochs. Initial learning rate of 0.001. Weight decay of 0.0001 applied to all models.
 * Hparams: Number of filters of the last convolutional layer in [256, 512]. Dropout probability in [0, 0.5]. Number of convolutional blocks in [1, 3]. Learning rate in [0.001, 0.01]. Batch size in [32, 512].


#### Task 9

 * Model: Network in Network.
 * Dataset: Cifar10, with the standard data augmentation (random horizontal flips and random crops after padding by 4 pixels.
 * Training: Trained for at most 1200 epochs, learning rate is multiplied by 0.2 after 300, 600 and 900 epochs. Cross entropy and SGD with momentum 0.9. Initial learning rate of 0.01.
 * Hparams: Number of filters in the convolutional layers in [256, 512], Number of convolutional layers in [9, 12], dropout probability in [0.0, 0.25], weight decay in [0.0, 0.001], batch size in [32, 512].


## License

The trained checkpoints as well as other metadata are released under [Apache 2.0 license](https://tldrlegal.com/license/apache-license-2.0-(apache-2.0)).

To aid reproducibility, we've also included the datasets that are used to train the models
in the zip files; these datasets retain their original licenses and terms.


## Reference

If you would like to use this dataset in your work, please cite our competition proposal:

[NeurIPS 2020 Competition: Predicting Generalization in Deep Learning](https://arxiv.org/abs/2012.07976)
