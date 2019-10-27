# Codebase for "Data Valuation using Reinforcement Learning"

Authors: Jinsung Yoon, Sercan O. Arik, Tomas Pfister

Paper: Jinsung Yoon, Sercan O Arik, Tomas Pfister, "Data Valuation using
Reinforcement Learning", arXiv preprint arXiv:1909.11671 (2019).
https://arxiv.org/abs/1909.11671

This directory contains implementations of Data Valuation using Reinforcement
Learning (DVRL) for the following four applications.

-   Data valuation
-   Corrupted sample discovery and robust learning (on tabular data)
-   Corrupted sample discovery and robust learning with transfer learning
    (on image data)
-   Domain adaptation

To run the pipeline for training and evaluation on data valuation application,
simply run python3 -m main_data_valuation.py or take a look at the following
jupyter-notebook file (main_data_valuation.ipynb).

To run the pipeline for training and evaluation on corrupted sample discovery
and robust learning (on tabular data), simply run python3 -m
main_corrupted_sample_discovery.py or take a look at the following
jupyter-notebook file (main_corrupted_sample_discovery.ipynb).

To run the pipeline for training and evaluation on corrupted sample discovery
and robust learning with transfer learning (on image data) application,
simply run python3 -m main_dvrl_image_transfer_learning.py or take a look at
the following jupyter-notebook file (main_dvrl_image_transfer_learning.ipynb).

To run the pipeline for training and evaluation on domain adaptation
application, simply run python3 -m main_domain_adaptation.py or take a look at
the following jupyter-notebook file (main_domain_adaptation.ipynb).

Note that any model architecture can be used as the predictor model, either
randomly initialized or pre-trained with the training data. The condition for
predictor model is to have fit and predict functions as its subfunctions.

To reduce complexity further, instead of raw features, the encoded features
(e.g. last ResNet layer for images) can also be used as the input to DVRL.

## Stages of the data valuation experiment:

-   Adult Income dataset (you can replace with any other dataset)
-   Train DVRL and estimate values of training samples
-   Sorted training samples based on the estimated data values
-   Show the top 5 high/low valued samples
-   Evaluate the prediction performance after removing high/low valued samples

### Command inputs:

-   data_name: Name of dataset ('adult')
-   normalization: Data normalization method ('minmax' or 'standard')
-   train_no: Number of training samples (1000)
-   valid_no: Number of valiation samples (400)
-   hidden_dim: Hidden state dimensions (100)
-   comb_dim: Hidden state dimensions after combining with prediction diff (10)
-   iterations: Number of RL iterations (2000)
-   layer_number: Number of layers (5)
-   batch_size: Number of mini-batch samples for RL (2000)
-   inner_iterations: Number of iterations for predictor (100)
-   batch_size_predictor: Number of mini-batch samples for predictor (256)
-   learning_rate: Learning rate for RL (0.01)
-   checkpoint_file_name: File name for saving and loading the trained model
    (./tmp/model.ckpt)

### Example command

```shell
$ python3 main_data_valuation.py --data_name adult --train_no 1000 \
--valid_no 400 --hidden_dim 100 --comb_dim 10 --iterations 2000 \
--layer_number 5 --batch_size 2000 --inner_iterations 100
--batch_size_predictor 256  --learning_rate 0.01 --n_exp 5 \
--checkpoint_file_name ./tmp/model.ckpt
```

### Outputs

-   Sorted training samples according to the estimated data values
-   Prediction performances after removing high/low valued samples
-   Top 5 high/low valued samples

## Stages of the corrupted sample discovery and robust learning (on tabular) experiment:

-   Adult Income dataset with a portion of samples' labels corrupted
    (You can replace with any other dataset)
-   Train DVRL and estimate values of training samples
-   Evaluate the robust learning performance
-   Evaluate the prediction performance after removing high/low valued samples
-   Evaluate the corrupted sample discovery rate

### Command inputs:

-   data_name: Name of dataset ('adult')
-   normalization: Data normalization method ('minmax' or 'standard')
-   train_no: Number of training samples (1000)
-   valid_no: Number of validation samples (400)
-   noise_rate: Ratio of label noise (0.2)
-   hidden_dim: Hidden state dimensions (100)
-   comb_dim: Hidden state dimensions after combining with prediction diff (10)
-   iterations: Number of RL iterations (2000)
-   layer_number: Number of layers (5)
-   batch_size: Number of mini-batch samples for RL (2000)
-   learning_rate: Learning rate for RL (0.01)
-   checkpoint_file_name: File name for saving and loading the trained model
    (./tmp/model.ckpt)

### Example command

```shell
$ python3 main_corrupted_sample_discovery.py --data_name adult --train_no 1000 \
--valid_no 400 --noise_rate 0.2 --hidden_dim 100 --comb_dim 10 \
--iterations 2000 --layer_number 5 --batch_size 2000 \
--learning_rate 0.01 --checkpoint_file_name ./tmp/model.ckpt
```

### Outputs

-   Robust learning performance
-   Prediction performances after removing high/low valued samples
-   Corrupted sample discovery rate

## Stages of corrupted sample discovery and robust learning with transfer learning (on image data) experiment:

-   CIFAR10 or CIFAR100 dataset with a portion of samples' labels corrupted
    (You can replace with any other dataset)
-   Use encoder model to encode the image datasets
-   Train DVRL and estimate values of training samples
-   Evaluate the robust learning performance
-   Evaluate the prediction performance after removing high/low valued samples
-   Evaluate the corrupted sample discovery rate

### Command inputs:

-   data_name: Name of dataset ('cifar10')
-   train_no: Number of training samples (4000)
-   valid_no: Number of validation samples (1000)
-   test_no: Number of testing samples (2000)
-   noise_rate: Ratio of label noise (0.2)
-   hidden_dim: Hidden state dimensions (100)
-   comb_dim: Hidden state dimensions after combining with prediction diff (10)
-   iterations: Number of RL iterations (2000)
-   layer_number: Number of layers (5)
-   batch_size: Number of mini-batch samples for RL (2000)
-   inner_iterations: Number of iterations for predictor networks (100)
-   batch_size_predictor: Number of mini-batch samples for predictor networks (256)
-   learning_rate: Learning rate for RL (0.01)
-   checkpoint_file_name: File name for saving and loading the trained model
    (./tmp/model.ckpt)

### Example command

```shell
$ python3 main_dvrl_image_transfer_learning.py --data_name cifar10 \
--train_no 4000 --valid_no 1000 --test_no 2000 --noise_rate 0.2 \
--hidden_dim 100 --comb_dim 10 \
--iterations 2000 --layer_number 5 --batch_size 2000 \
--inner_iterations 100 --batch_size_predictor 256 \
--learning_rate 0.01 --checkpoint_file_name ./tmp/model.ckpt
```

### Outputs

-   Robust learning performance
-   Prediction performances after removing high/low valued samples
-   Corrupted sample discovery rate (if noise_rate > 0)

## Stages of the domain adaptation experiment:

-   Rossmann dataset (you can replace with any other dataset)
-   Select the experiment setting and target store type
-   Train DVRL and estimate values of training samples
-   Evaluate the dvrl performance

### Command inputs:

-   normalization: Data normalization method ('minmax' or 'standard')
-   train_no: Number of training samples (667027)
-   valid_no: Number of validation samples (8443)
-   hidden_dim: Hidden state dimensions (100)
-   comb_dim: Hidden state dimensions after combining with prediction diff (10)
-   iterations: Number of RL iterations (1000)
-   layer_number: Number of layers (5)
-   batch_size: Number of mini-batch samples for RL (50000)
-   learning_rate: Learning rate for RL (0.001)
-   checkpoint_file_name: File name for saving and loading the trained model
    (./tmp/model.ckpt)

### Example command

```shell
$ python3 main_domain_adaptation.py --train_no 667027 \
--valid_no 8443 --hidden_dim 100 --comb_dim 10 --iterations 1000 \
--layer_number 5 --batch_size 50000 \
--learning_rate 0.001 --checkpoint_file_name ./tmp/model.ckpt
```

### Outputs

-   DVRL performance
