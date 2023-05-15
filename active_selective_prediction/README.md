# Active Selective Prediction

This is the official repository for the paper --
[ASPEST: Bridging the Gap Between Active Learning and Selective Prediction](https://arxiv.org/abs/2304.03870).

## Requirements

*   It is tested under Debian 4.19.260-1 (2022-09-29) x86_64 GNU/Linux and
    Python 3.7.12 environment.
*   To install requirements: `pip install -r requirements.txt`.

## Datasets

We create six benchmark datasets: `mnist->svhn`, `cifar10->cinic10`, `fmow`,
`amazon_review`, `domainnet`, and `otto`. The following datasets are already
included in TensorFlow Datasets: `mnist`, `svhn`, `cifar10`, and `domainnet`.
The following datasets need to be downloaded manually:
* `cinic10`: can be downloaded from [cinic10](https://datashare.is.ed.ac.uk/handle/10283/3192).
Place the downloaded data in `$RAW_DATASET_DIR/cinic10`.
* `fmow`: can be downloaded from [wilds](https://wilds.stanford.edu/get_started/).
Place the downloaded data in `$RAW_DATASET_DIR/wilds_data/fmow_v1.1`.
* `amazon_review`: can be downloaded from [wilds](https://wilds.stanford.edu/get_started/).
Place the downloaded data in `$RAW_DATASET_DIR/wilds_data/amazon_v2.1`.
* `otto`: can be downloaded from
[otto](https://www.kaggle.com/competitions/otto-group-product-classification-challenge/data).
Place the downloaded data in `$RAW_DATASET_DIR/otto-group-product-classification`.

Run the following command to build Tensorflow datasets:

`./active_selective_prediction/build_datasets.sh $RAW_DATASET_DIR $DATA_DIR`.

where `$RAW_DATASET_DIR` is the directory to store the raw datasets and
`$DATA_DIR` is the directory to store the generated TensorFlow datasets.

Note that for the `amazon_review` dataset, we need to use a GPU to extract the
embeddings from RoBERTa. The generated Tensorflow datasets can then be easily
loaded via functions in `utils.data_util` (need to set the global variable
`DATA_DIR` in `active_selective_prediction/utils/data_util.py`. The default
value of `DATA_DIR` is `~/tensorflow_datasets/`).

To add a new dataset to the codebase, follow these instructions:
* If the new dataset is not included in the
[TensorFlow datasets](https://www.tensorflow.org/datasets/catalog/overview#all_datasets),
implement a `tfds.core.GeneratorBasedBuilder` class for this new dataset and
put the class file under the directory `./tfds_generators`.
* Add a builder function for the new dataset in `build_datasets.py`. Modify the
script `build_datasets.sh` and run it to build the new dataset.
* Add a function to load the new dataset as `tf.data.Dataset` in `utils/data_util.py`.
* If the new dataset requires a new model architecture, implement the model
architecture in `models/custom_model.py` and add a function to load
the new model in `utils/model_util.py`.
* Add source training code for the new dataset in `train.py`.
Add evaluation code for the new dataset in
`eval_model.py` and `eval_pipeline.py`.
* Modify the config files under the folder
`./configs/` to include the new dataset.

## Selective Prediction Methods

*   `SR`: Softmax Response. Implemented in `methods/sr.py`.
*   `DE`: Deep Ensembles. Implemented in `methods/de.py`.
*   `ASPEST`: the proposed method Active Selective Prediction using Ensembles
    and Self-Training (ASPEST). Implemented in `methods/aspest.py`.

## Sampling Methods

Currently, we support the following sampling methods for active learning:
* `UniformSampling`: uniform random sampling. Implemented in
`sampling_methods/uniform_sampling.py`.
* `ConfidenceSampling`: Sampling based
on model confidence. Implemented in `sampling_methods/confidence_sampling.py`.
* `EntropySampling`: Sampling based on model entropy. Implemented in
`sampling_methods/entropy_sampling.py`.
* `MarginSampling`: Sampling based on
model margin. Implemented in `sampling_methods/margin_sampling.py`.
* `KCenterGreedySampling`: K-center greedy sampling method. Implemented in
`sampling_methods/kcenter_greedy_sampling.py`.
* `CLUESampling`: CLUE sampling
method. Implemented in `sampling_methods/clue_sampling.py`.
* `BADGESampling`:
BADGE sampling method. Implemented in `sampling_methods/badge_sampling.py`.
* `AverageKLDivergenceSampling`: average KL divergence sampling method.
Implemented in `sampling_methods/average_kl_divergence_sampling.py`.
* `AverageMarginSampling`: sampling method for the proposed method ASPEST.
Implemented in `sampling_methods/average_margin_sampling.py`.

## Training

*   To train a standard model via supervised learning on the source training
    dataset, use the following command:

`python -m active_selective_prediction.train --gpu $gpu --dataset $dataset`

## Evaluation

*   To evaluate the accuracy of a source trained model on the source validation
    dataset and the target test dataset, use the following command:

`python -m active_selective_prediction.eval_model --gpu $gpu --source-dataset $source --model-path $path`

where `$path` should point to the directory that stores the source trained model
checkpoint (e.g., the default value of `$path` for the `color_mnist` dataset
should be `./checkpoints/standard_supervised/color_mnist`).

* To evaluate an active selective prediction method, use the following command:

`python -m active_selective_prediction.eval_pipeline --gpu $gpu --source-dataset $source --method $method --method-config-file $config`

where `$config` should point to the method config file in the
`active_selective_prediction/configs` directory. We provide the default config
files for the SR, DE and ASPEST methods.

## Running Example

We provide a script `./active_selective_prediction/run_mnist_to_svhn_exp.sh`
to run experiments on MNIST->SVHN. It will train a model on MNIST and then
evaluate its accuracy on the MNIST test set and the SVHN test set. After that,
it will evaluate the SR+Margin, DE+Margin and ASPEST methods.
