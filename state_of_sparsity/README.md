# The State of Sparsity in Deep Neural Networks

This directory contains the code accompanying the paper ["The State of Sparsity in Deep Neural Networks"](https://arxiv.org/abs/1902.09574). All authors contributed to this code.

The `layers` subdirectory contains implementations of variational dropout and l0 regularization in TensorFlow. The `sparse_transformer` and `sparse_rn50` subdirectories contain code for the Transformer and ResNet-50 experiments from the aforementioned paper. The `results` subdirectory contains CSV files of the results of all hyperparameter configurations that we explored for each model, sparsity technique, and sparsity level.

## Build Docker Image

To build a Docker image with all required dependencies, run `sudo docker build -t <image_name> .`. The base setup installs TensorFlow with GPU support and is based off Nvidia's CUDA-9.0 image with all the required libraries to run TensorFlow. To launch the container, run `sudo docker run --runtime=nvidia -v ~/:/mount/ -it <image_name>:latest`. This command additionaly makes your home directory accessbile at `/mount` inside the container.

To run with GPU support, swap `tensorflow` for `tensorflow-gpu` in `requirements.txt`.

## Sparse Transformer

Once inside the container, this repo contains all of the code and data needed to decode the WMT English-German 2014 test set and calculate the BLEU score for each of the checkpoints we provided.

Small scripts to decode from Transformer checkpoints trained with each technique are provided in `sparse_transformer/decode/`. For random pruning checkpoints, use the `decode_mp.sh` script. For variational dropout, you'll need to pass in the same log alpha threshold that was used to achieve the BLEU score in checkpoint directory, which is provided as the last number in the checkpoint directory name.

The results of decoding from the model checkpoint will be saved in the `sparse_transformer/decode/` directory with a name like `newstest2014.end.sparse_transformer...`. To calculate the BLEU score for these decodes, run `sh get_ende_bleu.sh <decode_output>`. This script relies on the mosesdecoder project (https://github.com/moses-smt/mosesdecoder), and assumes this is installed at `/mount/mosesdecoder` inside the container. The output of the script should match the BLEU score reported in the checkpoint directory.

## Sparse ResNet-50

Scripts to evaluate ResNet-50 checkpoints on the ImageNet test set are provided in `sparse_rn50/evaluate/`. For random pruning checkpoints, use the `decode_mp.sh` script. You'll similarly need to pass in the log alpha threshold to evaluate va¯riaitonal dropout checkpoints, which was 0.5 for all our models. This repository does not include the ImageNet dataset, so you'll also need to point these scripts at a local version of the ImageNet test set stored as TFRecords. The output of the script should match the top-1 accuracy reported in the checkpoint directory.

## Calculate Weight Sparsity

To calculate the weight sparsity for a checkpoint, use the `checkpoint_sparsity.py` script and pass the checkpoint file, sparsity technique, and model ("transformer" or "rn50"). For variational dropout, also pass the same log alpha threshold.

## Trained Checkpoints

The top performing checkpoints for each model and sparsity technique can be downloaded with the following links. See https://github.com/google-research/google-research/issues/392 about how to inspect the checkpoints.

|    Model    |      Technique      | Sparsity |  BLEU |                                                                       Link                                                                      |
|-----------|-------------------|--------|-----|-----------------------------------------------------------------------------------------------------------------------------------------------|
| Transformer |  Magnitude Pruning  |    50%   | 26.33 |         [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/magnitude_pruning_0.5_100000_400000_10000_0.1_0.1.tar)        |
| Transformer |  Magnitude Pruning  |    60%   | 25.94 |         [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/magnitude_pruning_0.6_100000_400000_1000_0.1_0.1.tar)         |
| Transformer |  Magnitude Pruning  |    70%   | 25.21 |         [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/magnitude_pruning_0.7_100000_300000_1000_0.1_0.1.tar)         |
| Transformer |  Magnitude Pruning  |    80%   | 24.65 |         [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/magnitude_pruning_0.8_200000_400000_10000_0.0_0.1.tar)        |
| Transformer |  Magnitude Pruning  |    90%   | 23.26 |           [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/magnitude_pruning_0.9_0_400000_10000_0.0_0.1.tar)           |
| Transformer |  Magnitude Pruning  |    95%   | 20.75 |         [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/magnitude_pruning_0.95_100000_300000_1000_0.0_0.1.tar)        |
| Transformer |  Magnitude Pruning  |    98%   | 16.37 |           [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/magnitude_pruning_0.98_0_300000_1000_0.0_0.0.tar)           |
| Transformer | Variational Dropout |    50%   | 26.26 | [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/variational_dropout_0.5_cubic_400000_100000_2.22E-08_0.1_0.1_2.5.tar) |
| Transformer | Variational Dropout |    60%   | 25.37 |   [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/variational_dropout_0.6_linear_0_500000_2.22E-08_0.0_0.1_2.5.tar)   |
| Transformer | Variational Dropout |    70%   | 25.08 |   [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/variational_dropout_0.7_linear_0_500000_2.22E-08_0.0_0.0_1.5.tar)   |
| Transformer | Variational Dropout |    80%   | 24.33 |    [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/variational_dropout_0.8_cubic_0_300000_2.22E-08_0.0_0.1_1.5.tar)   |
| Transformer | Variational Dropout |    90%   | 21.43 |    [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/variational_dropout_0.9_cubic_0_200000_8.89E-08_0.0_0.1_2.5.tar)   |
| Transformer | Variational Dropout |    95%   | 19.13 |   [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/variational_dropout_0.95_cubic_0_200000_2.22E-07_0.0_0.0_2.5.tar)   |
| Transformer | Variational Dropout |    98%   | 14.45 |     [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/variational_dropout_0.98_linear_0_1_2.22E-07_0.0_0.1_2.0.tar)     |
| Transformer |  L0 Regularization  |    50%   | 26.72 |   [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/l0_regularization_0.5_cubic_100000_200000_0.000000289_0.0_0.1.tar)  |
| Transformer |  L0 Regularization  |    60%   | 26.16 |   [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/l0_regularization_0.6_cubic_200000_100000_0.000000778_0.0_0.0.tar)  |
| Transformer |  L0 Regularization  |    70%   | 25.29 |     [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/l0_regularization_0.7_cubic_0_100000_0.000000289_0.0_0.0.tar)     |
| Transformer |  L0 Regularization  |    80%   | 24.15 |     [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/l0_regularization_0.8_linear_0_100000_0.000000556_0.0_0.0.tar)    |
| Transformer |  L0 Regularization  |    90%   | 20.05 |   [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/l0_regularization_0.9_cubic_200000_100000_0.000002222_0.1_0.1.tar)  |
| Transformer |  L0 Regularization  |    95%   | 19.78 |     [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/l0_regularization_0.95_cubic_0_400000_0.000002222_0.0_0.0.tar)    |
| Transformer |  L0 Regularization  |    98%   | 16.83 |       [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/l0_regularization_0.98_linear_0_1_0.000002222_0.0_0.1.tar)      |
| Transformer |    Random Pruning   |    50%   | 24.56 |           [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/random_pruning_0.5_200000_500000_1000_0.0_0.1.tar)          |
| Transformer |    Random Pruning   |    60%   | 24.45 |             [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/random_pruning_0.6_0_300000_1000_0.0_0.1.tar)             |
| Transformer |    Random Pruning   |    70%   | 24.01 |             [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/random_pruning_0.7_0_400000_1000_0.0_0.1.tar)             |
| Transformer |    Random Pruning   |    80%   | 23.15 |             [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/random_pruning_0.8_0_300000_10000_0.0_0.1.tar)            |
| Transformer |    Random Pruning   |    90%   | 20.67 |             [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/random_pruning_0.9_0_100000_1000_0.0_0.1.tar)             |
| Transformer |    Random Pruning   |    95%   | 17.42 |             [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/random_pruning_0.95_0_100000_1000_0.0_0.0.tar)            |
| Transformer |    Random Pruning   |    98%   | 10.94 |             [link](https://storage.googleapis.com/tsos/checkpoints/sparse_transformer/random_pruning_0.98_0_100000_1000_0.0_0.0.tar)            |

|   Model   |                 Technique                | Sparsity | Top-1 Accuracy |                                                                 Link                                                                |
|---------|----------------------------------------|--------|--------------|-----------------------------------------------------------------------------------------------------------------------------------|
| ResNet-50 |             Magnitude Pruning            |    50%   |      76.53     |         [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/standard_threshold_0.5_40000_100000_2000_0.1.tar)        |
| ResNet-50 |             Magnitude Pruning            |    70%   |      76.38     |         [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/standard_threshold_0.7_40000_100000_4000_0.1.tar)        |
| ResNet-50 |             Magnitude Pruning            |    80%   |      75.58     |         [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/standard_threshold_0.8_40000_100000_2000_0.1.tar)        |
| ResNet-50 |             Magnitude Pruning            |    90%   |      73.91     |         [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/standard_threshold_0.9_40000_76000_2000_0.1.tar)         |
| ResNet-50 |             Magnitude Pruning            |    95%   |      70.59     |         [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/standard_threshold_0.95_40000_68000_2000_0.0.tar)        |
| ResNet-50 |             Magnitude Pruning            |    98%   |      57.9      |        [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/standard_threshold_0.98_40000_100000_2000_0.1.tar)        |
| ResNet-50 | Magnitude Pruning (extended/non-uniform) |    80%   |      76.52     |         [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/extended_threshold_0.8_12500_40000_2000_0.1.tar)         |
| ResNet-50 | Magnitude Pruning (extended/non-uniform) |    90%   |      75.16     |         [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/extended_threshold_0.91_12500_36000_2000_0.1.tar)        |
| ResNet-50 | Magnitude Pruning (extended/non-uniform) |    95%   |      72.71     |         [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/extended_threshold_0.96_7500_36000_1000_0.0.tar)         |
| ResNet-50 | Magnitude Pruning (extended/non-uniform) |   96.5%  |      69.26     |         [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/extended_threshold_0.98_12500_36000_500_0.0.tar)         |
| ResNet-50 |              Random Pruning              |    50%   |      74.59     |     [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/standard_random_cumulative_0.5_20000_40000_4000_0.1.tar)     |
| ResNet-50 |              Random Pruning              |    70%   |      72.2      |      [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/standard_random_cumulative_0.7_8000_40000_8000_0.1.tar)     |
| ResNet-50 |              Random Pruning              |    80%   |      70.21     |       [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/standard_random_cumulative_0.8_0_40000_8000_0.0.tar)       |
| ResNet-50 |              Random Pruning              |    90%   |       65       |       [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/standard_random_cumulative_0.9_0_40000_2000_0.0.tar)       |
| ResNet-50 |              Random Pruning              |    95%   |      58.04     |       [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/standard_random_cumulative_0.95_0_40000_2000_0.0.tar)      |
| ResNet-50 |              Random Pruning              |    98%   |      43.99     |     [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/standard_random_cumulative_0.98_20000_40000_8000_0.0.tar)    |
| ResNet-50 |            Variational Dropout           |    50%   |      76.55     |   [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/variational_dropout_0.5_8000_68000_7.80538368534e-09_0.1.tar)  |
| ResNet-50 |            Variational Dropout           |    80%   |      75.28     |  [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/variational_dropout_0.8_40000_68000_3.90269184267e-08_0.0.tar)  |
| ResNet-50 |            Variational Dropout           |    90%   |      73.84     |  [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/variational_dropout_0.9_20000_68000_7.80538368534e-08_0.0.tar)  |
| ResNet-50 |            Variational Dropout           |    95%   |      71.91     | [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/variational_dropout_0.95_40000_100000_3.90269184267e-07_0.1.tar) |
| ResNet-50 |            Variational Dropout           |    98%   |      67.36     |  [link](https://storage.googleapis.com/tsos/checkpoints/sparse_rn50/variational_dropout_0.98_8000_76000_7.80538368534e-07_0.1.tar)  |
