# Measuring the Effects of Data Parallelism on Neural Network Training

This directory contains the publicly available material for the paper:

[**Measuring the Effects of Data Parallelism on Neural Network Training**](
https://arxiv.org/abs/1811.03600).

Christopher J. Shallue\*, Jaehoon Lee\*, Joseph Antognini, Jascha Sohl-Dickstein,
Roy Frostig, and George E. Dahl (2018).

\* denotes equal contribution.

## Citation

If you find this code or data useful, please use the following citation:

```
@article{shallue2018measuring,
  author  = {Christopher J. Shallue and Jaehoon Lee and Joseph Antognini and Jascha Sohl-Dickstein and Roy Frostig and George E. Dahl},
  title   = {Measuring the Effects of Data Parallelism on Neural Network Training},
  journal = {Journal of Machine Learning Research},
  year    = {2019},
  volume  = {20},
  number  = {112},
  pages   = {1-49},
  url     = {http://jmlr.org/papers/v20/18-789.html}
}
```

## Contact

Please send pull requests and issues to Chris Shallue ([@cshallue](https://github.com/cshallue))

## Downloading the data

The data archive is available in the following file:
[batch_science_data.tar.bz2](https://storage.googleapis.com/batch_science_data/batch_science_data.tar.bz2)
(~802MB). The file is a `bzip2` compressed `tar` with checksum (`sha256sum`):

```
6460ab86a6ab0f22a02e1c9b982e1ca31220bf41f669304ce86f14d19053f435  batch_science_data.tar.bz2
```

Run the following command to extract the contents of the file. It will unpack
the data into a directory called `batch_science/`:

```
tar -xvf batch_science_data.tar.bz2
```

## Reproducing plots from the paper

The Python files in this folder contain code for loading and manipulating the
raw data.

[This Colaboratory notebook](https://colab.research.google.com/github/google-research/google-research/blob/master/batch_science/reproduce_paper_plots.ipynb)
reproduces all plots in the main section of the paper.

## Description of the data

We will use the following terminology when describing the data:

  * A *workload* is a specific choice of dataset, model, and optimizer.
  * A *study* is a hyperparameter search for a given workload and batch size.
  * A *trial* is a particular training run within a study for a particular
    choice of metaparameter values.

The extracted data are generally organized in a directory structure like this:

```
dataset/model/optimizer/batch_size/study.json
dataset/model/optimizer/batch_size/trial_id/metadata.json
dataset/model/optimizer/batch_size/trial_id/measurements.csv
```

Most workloads appear in the top-level directory, but a few special workloads
are grouped together under specific sub-directories:

  * ```mnist_subsets/```: Workloads trained on subsets of the MNIST dataset (see
    Section 4.5 of the paper).
  * ```imagenet_subsets/```: Workloads trained on subsets of the ImageNet
    dataset (see Section 4.5 of the paper).
  * ```solution_quality/```: Workloads trained on the MNIST and Fashion MNIST
    datasets that used very large training budgets in order to saturate
    performance at every batch size (see Section 4.8 of the paper).

As indicated above, each study is accompanied by a file `study.json`, which
looks like this:

```
{
  "batch_size": 256, 
  "dataset": "imagenet", 
  "early_stopping": false, 
  "model": "resnet_50", 
  "optimizer": "nesterov_momentum", 
  "parameter_configs": {
    "end_learning_rate_factor": {
      "max_value": 0.1, 
      "min_value": 0.0001, 
      "scale": "LOG_SCALE", 
      "type": "DOUBLE"
    }, 
    "label_smoothing": {
      "feasible_points": [
        0.0, 
        0.01, 
        0.1
      ], 
      "type": "DISCRETE"
    }, 
    "learning_rate": {
      "max_value": 10.0, 
      "min_value": 0.0001, 
      "scale": "LOG_SCALE", 
      "type": "DOUBLE"
    }, 
    "learning_rate_decay_steps": {
      "max_value": 600000, 
      "min_value": 300000, 
      "scale": "", 
      "type": "INTEGER"
    }, 
    "momentum": {
      "max_value": 0.9999, 
      "min_value": 0.9, 
      "scale": "REVERSE_LOG_SCALE", 
      "type": "DOUBLE"
    }
  }, 
  "train_steps": 600000
}
```

The fields in `study.json` have the following meanings:

  * `batch_size`: The batch size used in the study.
  * `dataset`: The dataset used in the study.
  * `early_stopping`: Whether an early stopping criterion was used to terminate
    bad trials early.
  * `model`: The model used in the study.
  * `optimizer`: The optimizer used in the study.
  * `parameter_configs`: The metaparameter search configuration for each
    metaparameter tuned in the study.
      * `feasible_points`: The discrete search space for this metaparameter
        (applies for type DISCRETE).
      * `max_value`: The maximum value of the search space for this
        metaparameter (applies for types DOUBLE and INTEGER).
      * `min_value`: The minimum value of the search space for this
        metaparameter (applies for types DOUBLE and INTEGER).
      * `scale`: Transformation on the search space (applies for type DOUBLE).
          * `LINEAR_SCALE`: Uniformly sample in linear space.
          * `LOG_SCALE`: Uniformly sample in log space.
          * `REVERSE_LOG_SCALE`: Uniformly sample (1 - value) in log space.
      * `type`: One of `DISCRETE`, `DOUBLE`, `INTEGER`
  * `train_steps`: The minimum number of training steps for a trial to be
    considered `COMPLETE`. Note that some trials may have trained for longer
    than `train_steps`. Note also that some trials have `train_steps = 0`, which
    indicates that those trials were trained with a time budget rather than a
    particular number of steps, in which case all trials that did not diverge
    are considered `COMPLETE`.

Each trial in each study is accompanied by files `metadata.json` and
`measurements.csv`.

The `metadata.json` file looks like this:

```
{
  "_internal_study_name": "resnet-20180601-smooth-bs256", 
  "_internal_trial_id": 2, 
  "parameters": {
    "end_learning_rate_factor": 0.0002861573844378761, 
    "label_smoothing": 0.01, 
    "learning_rate": 0.0124894465250831, 
    "learning_rate_decay_steps": 522526, 
    "momentum": 0.9788223543494348
  }, 
  "status": "COMPLETE", 
  "steps": 600000, 
  "trial_id": 2
}
```

The fields in `trial_id/metadata.json` have the following meanings:

  * `_internal_study_name`: Internal identifier, please ignore.
  * `_internal_trial_id`: Internal identifier, please ignore.
  * `parameters`: The values of each metaparameter in the metaparameter search.
  * `status`: One of:
    * `COMPLETE`: If the trial was completed.
    * `INCOMPLETE`: If the trial was not completed for some reason (these trials
      can usually be ignored).
    * `INFEASIBLE`: If training diverged at any point.
  * `steps`: The number of training steps taken.
  * `trial_id`: The trial id within the study.

The `measurements.csv` file contains data for each evaluation performed during
training each trial. It looks like this:

| step   |   train/cross_entropy_error |   train/classification_error |   val/cross_entropy_error |   val/classification_error |   test/cross_entropy_error |   test/classification_error |
|-------:|----------------------------:|-----------------------------:|--------------------------:|---------------------------:|---------------------------:|----------------------------:|
|      0 |                    6.90948  |                     0.999223 |                   6.90956 |                    0.99904 |                    6.90984 |                    0.999301 |
|   1000 |                    6.79853  |                     0.993921 |                   6.84734 |                    0.99416 |                    6.81384 |                    0.993566 |
|   2000 |                    6.08953  |                     0.956254 |                   6.18395 |                    0.95976 |                    6.11969 |                    0.957578 |
|   3000 |                    5.14154  |                     0.904496 |                   5.26828 |                    0.90812 |                    5.16471 |                    0.902428 |
|   4500 |                    4.79305  |                     0.867726 |                   4.93154 |                    0.87614 |                    4.82189 |                    0.86672  |
|    ... |                         ... |                          ... |                       ... |                        ... |                        ... |                        ...  |
| 597500 |                    0.592795 |                     0.120157 |                   1.13756 |                    0.24424 |                    1.00505 |                    0.213348 |
| 598500 |                    0.592241 |                     0.119539 |                   1.13629 |                    0.24404 |                    1.00443 |                    0.213688 |
| 600000 |                    0.592377 |                     0.119519 |                   1.13728 |                    0.24406 |                    1.00498 |                    0.213268 |

Note that different models have different metrics available, and that the time
between successive evaluations is not necessarily constant.

## Summary of all available data

|     | Dataset (Base Directory)               | Model                          | Optimizer         |   Batch Size |   Complete Trials |   Incomplete Trials |   Infeasible Trials |
|----:|:---------------------------------------|:-------------------------------|:------------------|-------------:|------------------:|--------------------:|--------------------:|
|   1 | cifar_10                               | resnet_8                       | nesterov_momentum |            2 |               165 |                   0 |                 108 |
|   2 |                                        |                                |                   |            4 |               167 |                   0 |                  85 |
|   3 |                                        |                                |                   |            8 |               166 |                   1 |                  95 |
|   4 |                                        |                                |                   |           16 |               167 |                   1 |                  86 |
|   5 |                                        |                                |                   |           32 |               168 |                   2 |                  73 |
|   6 |                                        |                                |                   |           64 |               167 |                   3 |                  54 |
|   7 |                                        |                                |                   |          128 |               167 |                   2 |                  48 |
|   8 |                                        |                                |                   |          256 |               171 |                   0 |                  38 |
|   9 |                                        |                                |                   |          512 |               166 |                   0 |                  37 |
|  10 |                                        |                                |                   |         1024 |               162 |                   0 |                  36 |
|  11 |                                        |                                |                   |         2048 |               162 |                   0 |                  32 |
|  12 |                                        |                                |                   |         4096 |               159 |                   0 |                  31 |
|  13 |                                        |                                |                   |         8192 |               162 |                   0 |                  41 |
|  14 | cifar_10                               | resnet_8                       | sgd               |            2 |               117 |                   0 |                  42 |
|  15 |                                        |                                |                   |            4 |               118 |                   1 |                  40 |
|  16 |                                        |                                |                   |            8 |               117 |                   1 |                  21 |
|  17 |                                        |                                |                   |           16 |               117 |                   1 |                  19 |
|  18 |                                        |                                |                   |           32 |               109 |                   0 |                  21 |
|  19 |                                        |                                |                   |           64 |               116 |                   0 |                  12 |
|  20 |                                        |                                |                   |          128 |               114 |                   1 |                  12 |
|  21 |                                        |                                |                   |          256 |               110 |                   0 |                  17 |
|  22 |                                        |                                |                   |          512 |               113 |                   0 |                  22 |
|  23 |                                        |                                |                   |         1024 |               112 |                   0 |                  20 |
|  24 |                                        |                                |                   |         2048 |               117 |                   0 |                  29 |
|  25 |                                        |                                |                   |         4096 |               114 |                   0 |                  31 |
|  26 |                                        |                                |                   |         8192 |               113 |                   0 |                  22 |
|  27 | common_crawl                           | transformer_base               | nesterov_momentum |           32 |                72 |                   2 |                 121 |
|  28 |                                        |                                |                   |           64 |                70 |                   5 |                  83 |
|  29 |                                        |                                |                   |          256 |                59 |                   1 |                  64 |
|  30 |                                        |                                |                   |         1024 |                58 |                   1 |                  65 |
|  31 |                                        |                                |                   |         4096 |                54 |                   1 |                  39 |
|  32 |                                        |                                |                   |        16384 |                53 |                   0 |                  43 |
|  33 | fashion_mnist                          | simple_cnn_base                | nesterov_momentum |            2 |               100 |                 372 |                 490 |
|  34 |                                        |                                |                   |            8 |               105 |                 383 |                 347 |
|  35 |                                        |                                |                   |           32 |               159 |                 339 |                 348 |
|  36 |                                        |                                |                   |          128 |               177 |                 141 |                 214 |
|  37 |                                        |                                |                   |          512 |               199 |                 288 |                 201 |
|  38 |                                        |                                |                   |         2048 |               137 |                 341 |                 181 |
|  39 |                                        |                                |                   |         8192 |               183 |                 309 |                 196 |
|  40 |                                        |                                |                   |        32768 |               121 |                 363 |                 164 |
|  41 |                                        |                                |                   |        55000 |               119 |                 370 |                 155 |
|  42 | imagenet                               | resnet_50                      | nesterov_momentum |           64 |               119 |                   6 |                  23 |
|  43 |                                        |                                |                   |          128 |               116 |                 105 |                  44 |
|  44 |                                        |                                |                   |          256 |               127 |                   8 |                  26 |
|  45 |                                        |                                |                   |          512 |               133 |                   3 |                  28 |
|  46 |                                        |                                |                   |         1024 |               123 |                   8 |                  16 |
|  47 |                                        |                                |                   |         2048 |               131 |                   1 |                   8 |
|  48 |                                        |                                |                   |         4096 |               108 |                   1 |                  10 |
|  49 |                                        |                                |                   |         8192 |               122 |                   2 |                  12 |
|  50 |                                        |                                |                   |        16384 |               113 |                   1 |                  11 |
|  51 |                                        |                                |                   |        32768 |               128 |                   3 |                  15 |
|  52 |                                        |                                |                   |        65536 |               100 |                   0 |                  14 |
|  53 | imagenet                               | vgg_11                         | nesterov_momentum |           32 |               128 |                  22 |                 213 |
|  54 |                                        |                                |                   |           64 |               121 |                  29 |                 185 |
|  55 |                                        |                                |                   |          256 |               120 |                   3 |                  87 |
|  56 |                                        |                                |                   |          512 |               104 |                   1 |                  74 |
|  57 |                                        |                                |                   |         1024 |               103 |                   1 |                  88 |
|  58 |                                        |                                |                   |         2048 |               104 |                   1 |                  77 |
|  59 |                                        |                                |                   |         4096 |               100 |                   4 |                  72 |
|  60 |                                        |                                |                   |         8192 |               102 |                   1 |                  81 |
|  61 |                                        |                                |                   |        16384 |               101 |                   1 |                  90 |
|  62 |                                        |                                |                   |        32768 |               106 |                  17 |                  81 |
|  63 |                                        |                                |                   |        65536 |               103 |                  27 |                1035 |
|  64 | imagenet_subsets/imagenet_half_classes | resnet_50                      | nesterov_momentum |           64 |               143 |                   7 |                  36 |
|  65 |                                        |                                |                   |          128 |               142 |                   8 |                  27 |
|  66 |                                        |                                |                   |          256 |               118 |                   2 |                  17 |
|  67 |                                        |                                |                   |          512 |               110 |                  10 |                  20 |
|  68 |                                        |                                |                   |         1024 |               119 |                   1 |                  18 |
|  69 |                                        |                                |                   |         2048 |               108 |                   1 |                  19 |
|  70 |                                        |                                |                   |         4096 |               108 |                   9 |                  20 |
|  71 |                                        |                                |                   |         8192 |               101 |                   2 |                   3 |
|  72 |                                        |                                |                   |        16384 |               103 |                   0 |                  10 |
|  73 |                                        |                                |                   |        32768 |               110 |                   4 |                  24 |
|  74 | imagenet_subsets/imagenet_half_images  | resnet_50                      | nesterov_momentum |           64 |               125 |                  84 |                  56 |
|  75 |                                        |                                |                   |          128 |               116 |                 107 |                  37 |
|  76 |                                        |                                |                   |          256 |               121 |                  38 |                  23 |
|  77 |                                        |                                |                   |          512 |               151 |                   7 |                  20 |
|  78 |                                        |                                |                   |         1024 |               126 |                  33 |                  21 |
|  79 |                                        |                                |                   |         2048 |               117 |                   7 |                  15 |
|  80 |                                        |                                |                   |         4096 |               111 |                   1 |                   8 |
|  81 |                                        |                                |                   |         8192 |               100 |                   2 |                   5 |
|  82 |                                        |                                |                   |        16384 |               101 |                   1 |                   7 |
|  83 |                                        |                                |                   |        32768 |               101 |                  11 |                  61 |
|  84 | lm1b                                   | lstm                           | nesterov_momentum |           16 |                65 |                   5 |                  67 |
|  85 |                                        |                                |                   |           64 |                61 |                   8 |                  52 |
|  86 |                                        |                                |                   |          256 |                65 |                   4 |                  42 |
|  87 |                                        |                                |                   |         1024 |                63 |                   7 |                  43 |
|  88 |                                        |                                |                   |         4096 |                54 |                   1 |                  25 |
|  89 |                                        |                                |                   |        16384 |                52 |                   0 |                  29 |
|  90 |                                        |                                |                   |        32768 |                50 |                   2 |                  26 |
|  91 | lm1b                                   | transformer_base               | nesterov_momentum |           16 |               148 |                   2 |                 350 |
|  92 |                                        |                                |                   |           32 |               100 |                  50 |                 253 |
|  93 |                                        |                                |                   |           64 |               147 |                   0 |                 208 |
|  94 |                                        |                                |                   |          128 |               143 |                   6 |                 234 |
|  95 |                                        |                                |                   |          256 |               118 |                   1 |                 158 |
|  96 |                                        |                                |                   |          512 |               115 |                   3 |                 125 |
|  97 |                                        |                                |                   |         1024 |               119 |                   1 |                 147 |
|  98 |                                        |                                |                   |         2048 |               114 |                   6 |                 128 |
|  99 |                                        |                                |                   |         4096 |               107 |                   2 |                 122 |
| 100 |                                        |                                |                   |         8192 |               108 |                   1 |                 125 |
| 101 |                                        |                                |                   |        16384 |               105 |                   4 |                 118 |
| 102 |                                        |                                |                   |        32768 |               104 |                   6 |                 145 |
| 103 | lm1b                                   | transformer_narrow_and_shallow | nesterov_momentum |           16 |               145 |                   2 |                 178 |
| 104 |                                        |                                |                   |           32 |               112 |                  37 |                 135 |
| 105 |                                        |                                |                   |           64 |               145 |                   2 |                 183 |
| 106 |                                        |                                |                   |          128 |               146 |                   4 |                 167 |
| 107 |                                        |                                |                   |          256 |               103 |                  47 |                 135 |
| 108 |                                        |                                |                   |          512 |               147 |                   0 |                 148 |
| 109 |                                        |                                |                   |         1024 |               149 |                   1 |                 135 |
| 110 |                                        |                                |                   |         2048 |               127 |                  22 |                 123 |
| 111 |                                        |                                |                   |         4096 |               115 |                   4 |                  91 |
| 112 |                                        |                                |                   |         8192 |               112 |                   7 |                 116 |
| 113 |                                        |                                |                   |        16384 |               113 |                   6 |                  85 |
| 114 |                                        |                                |                   |        32768 |               102 |                  18 |                  76 |
| 115 | lm1b                                   | transformer_shallow            | momentum          |           32 |               115 |                  23 |                 258 |
| 116 |                                        |                                |                   |          128 |               132 |                   3 |                 228 |
| 117 |                                        |                                |                   |          512 |               100 |                  10 |                 177 |
| 118 |                                        |                                |                   |         2048 |               101 |                   8 |                 127 |
| 119 |                                        |                                |                   |         8192 |               100 |                   3 |                 150 |
| 120 |                                        |                                |                   |        32768 |               109 |                   0 |                 173 |
| 121 | lm1b                                   | transformer_shallow            | nesterov_momentum |           16 |               119 |                   0 |                 237 |
| 122 |                                        |                                |                   |           32 |               115 |                   5 |                 194 |
| 123 |                                        |                                |                   |           64 |               117 |                   2 |                 203 |
| 124 |                                        |                                |                   |          128 |               118 |                   1 |                 200 |
| 125 |                                        |                                |                   |          256 |               109 |                   0 |                 166 |
| 126 |                                        |                                |                   |          512 |               100 |                   9 |                 140 |
| 127 |                                        |                                |                   |         1024 |               118 |                   2 |                 181 |
| 128 |                                        |                                |                   |         2048 |               110 |                   7 |                 115 |
| 129 |                                        |                                |                   |         4096 |               117 |                   2 |                 153 |
| 130 |                                        |                                |                   |         8192 |               108 |                   1 |                 120 |
| 131 |                                        |                                |                   |        16384 |               106 |                   2 |                 127 |
| 132 |                                        |                                |                   |        32768 |               107 |                   1 |                 120 |
| 133 | lm1b                                   | transformer_shallow            | sgd               |           32 |                58 |                  52 |                  38 |
| 134 |                                        |                                |                   |          128 |                65 |                  45 |                  56 |
| 135 |                                        |                                |                   |          512 |                66 |                  45 |                  48 |
| 136 |                                        |                                |                   |         2048 |                62 |                  43 |                  46 |
| 137 |                                        |                                |                   |         8192 |                55 |                  49 |                  38 |
| 138 | lm1b                                   | transformer_wide               | nesterov_momentum |           16 |               117 |                  83 |                 391 |
| 139 |                                        |                                |                   |           32 |               103 |                  96 |                 302 |
| 140 |                                        |                                |                   |           64 |               108 |                  91 |                 314 |
| 141 |                                        |                                |                   |          128 |               105 |                  26 |                 175 |
| 142 |                                        |                                |                   |          256 |               104 |                  24 |                 159 |
| 143 |                                        |                                |                   |          512 |               114 |                   1 |                 138 |
| 144 |                                        |                                |                   |         1024 |               123 |                   0 |                 153 |
| 145 |                                        |                                |                   |         2048 |               109 |                   1 |                 112 |
| 146 |                                        |                                |                   |         4096 |               108 |                   1 |                 123 |
| 147 |                                        |                                |                   |         8192 |               103 |                   1 |                  96 |
| 148 |                                        |                                |                   |        16384 |               101 |                   0 |                  77 |
| 149 |                                        |                                |                   |        32768 |               101 |                   0 |                  91 |
| 150 | mnist                                  | fc_1024                        | sgd               |            1 |               274 |                 226 |                   0 |
| 151 |                                        |                                |                   |            2 |               265 |                 235 |                   0 |
| 152 |                                        |                                |                   |            4 |               247 |                 253 |                   0 |
| 153 |                                        |                                |                   |            8 |               295 |                 205 |                   0 |
| 154 |                                        |                                |                   |           16 |               291 |                 209 |                   0 |
| 155 |                                        |                                |                   |           32 |               291 |                 209 |                   0 |
| 156 |                                        |                                |                   |           64 |               309 |                 191 |                   0 |
| 157 |                                        |                                |                   |          128 |               287 |                 213 |                   0 |
| 158 |                                        |                                |                   |          256 |               285 |                 215 |                   0 |
| 159 |                                        |                                |                   |          512 |               278 |                 222 |                   0 |
| 160 |                                        |                                |                   |         1024 |               289 |                 211 |                   0 |
| 161 |                                        |                                |                   |         2048 |               297 |                 203 |                   0 |
| 162 |                                        |                                |                   |         4096 |               304 |                 196 |                   0 |
| 163 |                                        |                                |                   |         8192 |               286 |                 214 |                   0 |
| 164 |                                        |                                |                   |        16384 |               274 |                 226 |                   0 |
| 165 |                                        |                                |                   |        32768 |               302 |                 198 |                   0 |
| 166 |                                        |                                |                   |        55000 |               281 |                 219 |                   0 |
| 167 | mnist                                  | fc_1024_1024                   | sgd               |            1 |               248 |                 212 |                  40 |
| 168 |                                        |                                |                   |            2 |               259 |                 241 |                   0 |
| 169 |                                        |                                |                   |            4 |               262 |                 238 |                   0 |
| 170 |                                        |                                |                   |            8 |               219 |                 149 |                 132 |
| 171 |                                        |                                |                   |           16 |               253 |                 163 |                  84 |
| 172 |                                        |                                |                   |           32 |               287 |                 170 |                  43 |
| 173 |                                        |                                |                   |           64 |               270 |                 196 |                  34 |
| 174 |                                        |                                |                   |          128 |               197 |                 122 |                 181 |
| 175 |                                        |                                |                   |          256 |               209 |                 133 |                 158 |
| 176 |                                        |                                |                   |          512 |               215 |                 136 |                 149 |
| 177 |                                        |                                |                   |         1024 |               210 |                 130 |                 160 |
| 178 |                                        |                                |                   |         2048 |               210 |                 124 |                 166 |
| 179 |                                        |                                |                   |         4096 |               266 |                 132 |                 102 |
| 180 |                                        |                                |                   |         8192 |               298 |                 176 |                  26 |
| 181 |                                        |                                |                   |        16384 |               276 |                 195 |                  29 |
| 182 |                                        |                                |                   |        32768 |               305 |                 160 |                  35 |
| 183 |                                        |                                |                   |        55000 |               312 |                 166 |                  22 |
| 184 | mnist                                  | fc_1024_1024_1024              | sgd               |            1 |               234 |                 193 |                  73 |
| 185 |                                        |                                |                   |            2 |               250 |                 226 |                  24 |
| 186 |                                        |                                |                   |            4 |               252 |                 227 |                  21 |
| 187 |                                        |                                |                   |            8 |               217 |                 136 |                 147 |
| 188 |                                        |                                |                   |           16 |               266 |                 161 |                  73 |
| 189 |                                        |                                |                   |           32 |               267 |                 169 |                  64 |
| 190 |                                        |                                |                   |           64 |               291 |                 178 |                  31 |
| 191 |                                        |                                |                   |          128 |               212 |                 118 |                 170 |
| 192 |                                        |                                |                   |          256 |               204 |                 134 |                 162 |
| 193 |                                        |                                |                   |          512 |               216 |                 130 |                 154 |
| 194 |                                        |                                |                   |         1024 |               198 |                 112 |                 190 |
| 195 |                                        |                                |                   |         2048 |               204 |                 123 |                 173 |
| 196 |                                        |                                |                   |         4096 |               220 |                 139 |                 141 |
| 197 |                                        |                                |                   |         8192 |               240 |                 124 |                 136 |
| 198 |                                        |                                |                   |        16384 |               228 |                 196 |                  76 |
| 199 |                                        |                                |                   |        32768 |               250 |                 173 |                  77 |
| 200 |                                        |                                |                   |        55000 |               242 |                 189 |                  69 |
| 201 | mnist                                  | fc_128_128_128                 | sgd               |            1 |               124 |                 325 |                  51 |
| 202 |                                        |                                |                   |            2 |               120 |                 379 |                   1 |
| 203 |                                        |                                |                   |            4 |               272 |                 226 |                   2 |
| 204 |                                        |                                |                   |            8 |               236 |                 170 |                  94 |
| 205 |                                        |                                |                   |           16 |               265 |                 204 |                  31 |
| 206 |                                        |                                |                   |           32 |               296 |                 199 |                   5 |
| 207 |                                        |                                |                   |           64 |               283 |                 217 |                   0 |
| 208 |                                        |                                |                   |          128 |               233 |                 182 |                  85 |
| 209 |                                        |                                |                   |          256 |               247 |                 189 |                  64 |
| 210 |                                        |                                |                   |          512 |               260 |                 195 |                  45 |
| 211 |                                        |                                |                   |         1024 |               270 |                 196 |                  34 |
| 212 |                                        |                                |                   |         2048 |               263 |                 209 |                  28 |
| 213 |                                        |                                |                   |         4096 |               267 |                 206 |                  27 |
| 214 |                                        |                                |                   |         8192 |               289 |                 206 |                   5 |
| 215 |                                        |                                |                   |        16384 |               291 |                 204 |                   5 |
| 216 |                                        |                                |                   |        32768 |               292 |                 207 |                   1 |
| 217 |                                        |                                |                   |        55000 |               294 |                 201 |                   5 |
| 218 | mnist                                  | fc_2048_2048_2048              | sgd               |            1 |               205 |                 208 |                  87 |
| 219 |                                        |                                |                   |            2 |               226 |                 248 |                  26 |
| 220 |                                        |                                |                   |            4 |               234 |                 247 |                  19 |
| 221 |                                        |                                |                   |            8 |               193 |                 141 |                 166 |
| 222 |                                        |                                |                   |           16 |               241 |                 151 |                 108 |
| 223 |                                        |                                |                   |           32 |               255 |                 175 |                  70 |
| 224 |                                        |                                |                   |           64 |               274 |                 183 |                  43 |
| 225 |                                        |                                |                   |          128 |               185 |                 102 |                 213 |
| 226 |                                        |                                |                   |          256 |               207 |                 116 |                 177 |
| 227 |                                        |                                |                   |          512 |               179 |                 102 |                 219 |
| 228 |                                        |                                |                   |         1024 |               176 |                 117 |                 207 |
| 229 |                                        |                                |                   |         2048 |               196 |                 132 |                 172 |
| 230 |                                        |                                |                   |         4096 |               197 |                 122 |                 181 |
| 231 |                                        |                                |                   |         8192 |               214 |                 115 |                 171 |
| 232 |                                        |                                |                   |        16384 |               233 |                 176 |                  91 |
| 233 |                                        |                                |                   |        32768 |               227 |                 180 |                  93 |
| 234 |                                        |                                |                   |        55000 |               234 |                 165 |                 101 |
| 235 | mnist                                  | fc_256_256_256                 | sgd               |            1 |               264 |                 175 |                  61 |
| 236 |                                        |                                |                   |            2 |               193 |                 220 |                   5 |
| 237 |                                        |                                |                   |            4 |               272 |                 221 |                   7 |
| 238 |                                        |                                |                   |            8 |               228 |                 156 |                 116 |
| 239 |                                        |                                |                   |           16 |               266 |                 163 |                  71 |
| 240 |                                        |                                |                   |           32 |               281 |                 190 |                  29 |
| 241 |                                        |                                |                   |           64 |               291 |                 207 |                   2 |
| 242 |                                        |                                |                   |          128 |               231 |                 128 |                 141 |
| 243 |                                        |                                |                   |          256 |               226 |                 162 |                 112 |
| 244 |                                        |                                |                   |          512 |               250 |                 152 |                  98 |
| 245 |                                        |                                |                   |         1024 |               255 |                 160 |                  85 |
| 246 |                                        |                                |                   |         2048 |               256 |                 151 |                  93 |
| 247 |                                        |                                |                   |         4096 |               248 |                 168 |                  84 |
| 248 |                                        |                                |                   |         8192 |               280 |                 176 |                  44 |
| 249 |                                        |                                |                   |        16384 |               293 |                 174 |                  33 |
| 250 |                                        |                                |                   |        32768 |               287 |                 191 |                  22 |
| 251 |                                        |                                |                   |        55000 |               291 |                 182 |                  27 |
| 252 | mnist                                  | fc_512_512_512                 | sgd               |            1 |               232 |                 199 |                  69 |
| 253 |                                        |                                |                   |            2 |               243 |                 233 |                  24 |
| 254 |                                        |                                |                   |            4 |               245 |                 242 |                  13 |
| 255 |                                        |                                |                   |            8 |               226 |                 157 |                 117 |
| 256 |                                        |                                |                   |           16 |               253 |                 160 |                  87 |
| 257 |                                        |                                |                   |           32 |               281 |                 170 |                  49 |
| 258 |                                        |                                |                   |           64 |               296 |                 188 |                  16 |
| 259 |                                        |                                |                   |          128 |               219 |                 115 |                 166 |
| 260 |                                        |                                |                   |          256 |               211 |                 141 |                 148 |
| 261 |                                        |                                |                   |          512 |               230 |                 147 |                 123 |
| 262 |                                        |                                |                   |         1024 |               222 |                 147 |                 131 |
| 263 |                                        |                                |                   |         2048 |               224 |                 152 |                 124 |
| 264 |                                        |                                |                   |         4096 |               248 |                 140 |                 112 |
| 265 |                                        |                                |                   |         8192 |               258 |                 154 |                  88 |
| 266 |                                        |                                |                   |        16384 |               262 |                 177 |                  61 |
| 267 |                                        |                                |                   |        32768 |               275 |                 173 |                  52 |
| 268 |                                        |                                |                   |        55000 |               261 |                 186 |                  53 |
| 269 | mnist                                  | fc_64_64_64                    | sgd               |            1 |               266 |                 213 |                  21 |
| 270 |                                        |                                |                   |            2 |               278 |                 222 |                   0 |
| 271 |                                        |                                |                   |            4 |               279 |                 221 |                   0 |
| 272 |                                        |                                |                   |            8 |               261 |                 196 |                  43 |
| 273 |                                        |                                |                   |           16 |               270 |                 221 |                   9 |
| 274 |                                        |                                |                   |           32 |               278 |                 222 |                   0 |
| 275 |                                        |                                |                   |           64 |               289 |                 211 |                   0 |
| 276 |                                        |                                |                   |          128 |               259 |                 215 |                  26 |
| 277 |                                        |                                |                   |          256 |               281 |                 210 |                   9 |
| 278 |                                        |                                |                   |          512 |               283 |                 217 |                   0 |
| 279 |                                        |                                |                   |         1024 |               259 |                 239 |                   2 |
| 280 |                                        |                                |                   |         2048 |               285 |                 213 |                   2 |
| 281 |                                        |                                |                   |         4096 |               282 |                 218 |                   0 |
| 282 |                                        |                                |                   |         8192 |               304 |                 196 |                   0 |
| 283 |                                        |                                |                   |        16384 |               292 |                 208 |                   0 |
| 284 |                                        |                                |                   |        32768 |               290 |                 210 |                   0 |
| 285 |                                        |                                |                   |        55000 |               291 |                 209 |                   0 |
| 286 | mnist                                  | simple_cnn_base                | momentum          |            1 |               229 |                 250 |                  21 |
| 287 |                                        |                                |                   |            2 |               242 |                 240 |                  18 |
| 288 |                                        |                                |                   |            4 |               240 |                 233 |                  27 |
| 289 |                                        |                                |                   |            8 |               197 |                 191 |                 112 |
| 290 |                                        |                                |                   |           16 |               232 |                 206 |                  62 |
| 291 |                                        |                                |                   |           32 |               232 |                 220 |                  48 |
| 292 |                                        |                                |                   |           64 |               249 |                 221 |                  30 |
| 293 |                                        |                                |                   |          128 |               192 |                 146 |                 162 |
| 294 |                                        |                                |                   |          256 |               194 |                 146 |                 160 |
| 295 |                                        |                                |                   |          512 |               213 |                 126 |                 161 |
| 296 |                                        |                                |                   |         1024 |               202 |                 144 |                 154 |
| 297 |                                        |                                |                   |         2048 |               208 |                 142 |                 150 |
| 298 |                                        |                                |                   |         4096 |               211 |                 159 |                 130 |
| 299 |                                        |                                |                   |         8192 |               207 |                 159 |                 134 |
| 300 |                                        |                                |                   |        16384 |               209 |                 163 |                 128 |
| 301 |                                        |                                |                   |        32768 |               204 |                 167 |                 129 |
| 302 |                                        |                                |                   |        55000 |               203 |                 160 |                 137 |
| 303 | mnist                                  | simple_cnn_base                | nesterov_momentum |            1 |               311 |                  30 |                 286 |
| 304 |                                        |                                |                   |            2 |               347 |                   0 |                 152 |
| 305 |                                        |                                |                   |            8 |               347 |                   0 |                  95 |
| 306 |                                        |                                |                   |           32 |               274 |                  73 |                  69 |
| 307 |                                        |                                |                   |          128 |               334 |                   9 |                 106 |
| 308 |                                        |                                |                   |          512 |               336 |                   0 |                  95 |
| 309 |                                        |                                |                   |         2048 |               344 |                   0 |                  96 |
| 310 |                                        |                                |                   |         8192 |               343 |                   0 |                  81 |
| 311 |                                        |                                |                   |        32768 |               342 |                   1 |                  75 |
| 312 |                                        |                                |                   |        55000 |               345 |                   0 |                  73 |
| 313 | mnist                                  | simple_cnn_base                | sgd               |            1 |               194 |                 274 |                  32 |
| 314 |                                        |                                |                   |            2 |               223 |                 252 |                  25 |
| 315 |                                        |                                |                   |            4 |               265 |                 232 |                   3 |
| 316 |                                        |                                |                   |            8 |               190 |                 213 |                  97 |
| 317 |                                        |                                |                   |           16 |               229 |                 212 |                  59 |
| 318 |                                        |                                |                   |           32 |               274 |                 212 |                  14 |
| 319 |                                        |                                |                   |           64 |               274 |                 224 |                   2 |
| 320 |                                        |                                |                   |          128 |               216 |                 167 |                 117 |
| 321 |                                        |                                |                   |          256 |               219 |                 167 |                 114 |
| 322 |                                        |                                |                   |          512 |               215 |                 165 |                 120 |
| 323 |                                        |                                |                   |         1024 |               215 |                 152 |                 133 |
| 324 |                                        |                                |                   |         2048 |               219 |                 151 |                 130 |
| 325 |                                        |                                |                   |         4096 |               214 |                 163 |                 123 |
| 326 |                                        |                                |                   |         8192 |               214 |                 160 |                 126 |
| 327 |                                        |                                |                   |        16384 |               228 |                 154 |                 118 |
| 328 |                                        |                                |                   |        32768 |               204 |                 165 |                 131 |
| 329 |                                        |                                |                   |        55000 |               235 |                 148 |                 117 |
| 330 | mnist                                  | simple_cnn_narrow              | sgd               |            1 |               228 |                 268 |                   4 |
| 331 |                                        |                                |                   |            2 |               224 |                 255 |                  21 |
| 332 |                                        |                                |                   |            4 |               236 |                 252 |                  12 |
| 333 |                                        |                                |                   |            8 |               206 |                 230 |                  64 |
| 334 |                                        |                                |                   |           16 |               249 |                 213 |                  38 |
| 335 |                                        |                                |                   |           32 |               258 |                 231 |                  11 |
| 336 |                                        |                                |                   |           64 |               286 |                 210 |                   4 |
| 337 |                                        |                                |                   |          128 |               249 |                 172 |                  79 |
| 338 |                                        |                                |                   |          256 |               222 |                 191 |                  87 |
| 339 |                                        |                                |                   |          512 |               240 |                 178 |                  82 |
| 340 |                                        |                                |                   |         1024 |               202 |                 195 |                 103 |
| 341 |                                        |                                |                   |         2048 |               233 |                 169 |                  98 |
| 342 |                                        |                                |                   |         4096 |               230 |                 174 |                  96 |
| 343 |                                        |                                |                   |         8192 |               240 |                 182 |                  78 |
| 344 |                                        |                                |                   |        16384 |               229 |                 172 |                  99 |
| 345 |                                        |                                |                   |        32768 |               222 |                 184 |                  94 |
| 346 |                                        |                                |                   |        55000 |               226 |                 190 |                  84 |
| 347 | mnist                                  | simple_cnn_wide                | sgd               |            1 |               190 |                 268 |                  42 |
| 348 |                                        |                                |                   |            2 |               205 |                 250 |                  45 |
| 349 |                                        |                                |                   |            4 |               254 |                 246 |                   0 |
| 350 |                                        |                                |                   |            8 |               212 |                 191 |                  97 |
| 351 |                                        |                                |                   |           16 |               238 |                 202 |                  60 |
| 352 |                                        |                                |                   |           32 |               278 |                 209 |                  13 |
| 353 |                                        |                                |                   |           64 |               283 |                 215 |                   2 |
| 354 |                                        |                                |                   |          128 |               202 |                 163 |                 135 |
| 355 |                                        |                                |                   |          256 |               202 |                 147 |                 151 |
| 356 |                                        |                                |                   |          512 |               220 |                 142 |                 138 |
| 357 |                                        |                                |                   |         1024 |               208 |                 139 |                 153 |
| 358 |                                        |                                |                   |         2048 |               145 |                 207 |                 148 |
| 359 |                                        |                                |                   |         4096 |               203 |                 161 |                 136 |
| 360 |                                        |                                |                   |         8192 |               223 |                 157 |                 120 |
| 361 |                                        |                                |                   |        16384 |               190 |                 172 |                 138 |
| 362 |                                        |                                |                   |        32768 |               165 |                 212 |                 123 |
| 363 | mnist_subsets/mnist_13750              | simple_cnn_base                | nesterov_momentum |            1 |               488 |                   0 |                 342 |
| 364 |                                        |                                |                   |            2 |               488 |                   0 |                 209 |
| 365 |                                        |                                |                   |            4 |               454 |                   0 |                 149 |
| 366 |                                        |                                |                   |            8 |               448 |                   0 |                 101 |
| 367 |                                        |                                |                   |           16 |               458 |                   0 |                 133 |
| 368 |                                        |                                |                   |           32 |               470 |                   0 |                 104 |
| 369 |                                        |                                |                   |           64 |               495 |                   0 |                 128 |
| 370 |                                        |                                |                   |          128 |               490 |                   0 |                 123 |
| 371 |                                        |                                |                   |          256 |               476 |                   0 |                 139 |
| 372 |                                        |                                |                   |          512 |               467 |                   0 |                 118 |
| 373 |                                        |                                |                   |         1024 |               478 |                   0 |                  71 |
| 374 |                                        |                                |                   |         2048 |               484 |                   0 |                  79 |
| 375 |                                        |                                |                   |         4096 |               480 |                   0 |                  78 |
| 376 |                                        |                                |                   |         8192 |               489 |                   0 |                  62 |
| 377 |                                        |                                |                   |        13750 |               483 |                   0 |                  59 |
| 378 | mnist_subsets/mnist_27500              | simple_cnn_base                | nesterov_momentum |            1 |               493 |                   0 |                 289 |
| 379 |                                        |                                |                   |            2 |               479 |                   0 |                 183 |
| 380 |                                        |                                |                   |            4 |               464 |                   0 |                 124 |
| 381 |                                        |                                |                   |            8 |               479 |                   0 |                  98 |
| 382 |                                        |                                |                   |           16 |               473 |                   0 |                 132 |
| 383 |                                        |                                |                   |           32 |               485 |                   0 |                 115 |
| 384 |                                        |                                |                   |           64 |               499 |                   0 |                 130 |
| 385 |                                        |                                |                   |          128 |               493 |                   0 |                 123 |
| 386 |                                        |                                |                   |          256 |               482 |                   0 |                 116 |
| 387 |                                        |                                |                   |          512 |               476 |                   0 |                 113 |
| 388 |                                        |                                |                   |         1024 |               487 |                   0 |                  69 |
| 389 |                                        |                                |                   |         2048 |               484 |                   0 |                  73 |
| 390 |                                        |                                |                   |         4096 |               485 |                   0 |                  64 |
| 391 |                                        |                                |                   |         8192 |               484 |                   0 |                  57 |
| 392 |                                        |                                |                   |        16384 |               491 |                   0 |                  68 |
| 393 |                                        |                                |                   |        27500 |               494 |                   0 |                 214 |
| 394 | mnist_subsets/mnist_55000              | simple_cnn_base                | nesterov_momentum |            1 |               482 |                   0 |                 274 |
| 395 |                                        |                                |                   |            2 |               483 |                   0 |                 207 |
| 396 |                                        |                                |                   |            4 |               480 |                   0 |                 133 |
| 397 |                                        |                                |                   |            8 |               489 |                   0 |                  81 |
| 398 |                                        |                                |                   |           16 |               476 |                   0 |                 155 |
| 399 |                                        |                                |                   |           32 |               483 |                   0 |                 112 |
| 400 |                                        |                                |                   |           64 |               492 |                   0 |                 117 |
| 401 |                                        |                                |                   |          128 |               491 |                   0 |                 114 |
| 402 |                                        |                                |                   |          256 |               497 |                   0 |                  91 |
| 403 |                                        |                                |                   |          512 |               491 |                   0 |                  78 |
| 404 |                                        |                                |                   |         1024 |               476 |                   0 |                  69 |
| 405 |                                        |                                |                   |         2048 |               492 |                   0 |                  48 |
| 406 |                                        |                                |                   |         4096 |               491 |                   0 |                  55 |
| 407 |                                        |                                |                   |         8192 |               495 |                   0 |                  51 |
| 408 |                                        |                                |                   |        16384 |               494 |                   0 |                  40 |
| 409 |                                        |                                |                   |        32768 |               497 |                   0 |                  59 |
| 410 |                                        |                                |                   |        55000 |               492 |                   0 |                  40 |
| 411 | mnist_subsets/mnist_6875               | simple_cnn_base                | nesterov_momentum |            1 |               490 |                   0 |                 297 |
| 412 |                                        |                                |                   |            2 |               473 |                   0 |                 221 |
| 413 |                                        |                                |                   |            4 |               453 |                   0 |                 123 |
| 414 |                                        |                                |                   |            8 |               455 |                   0 |                 100 |
| 415 |                                        |                                |                   |           16 |               444 |                   0 |                 125 |
| 416 |                                        |                                |                   |           32 |               455 |                   0 |                 109 |
| 417 |                                        |                                |                   |           64 |               472 |                   0 |                 132 |
| 418 |                                        |                                |                   |          128 |               493 |                   0 |                 153 |
| 419 |                                        |                                |                   |          256 |               485 |                   0 |                 148 |
| 420 |                                        |                                |                   |          512 |               468 |                   0 |                 143 |
| 421 |                                        |                                |                   |         1024 |               476 |                   0 |                 135 |
| 422 |                                        |                                |                   |         2048 |               487 |                   0 |                  89 |
| 423 |                                        |                                |                   |         4096 |               486 |                   0 |                  77 |
| 424 |                                        |                                |                   |         6875 |               490 |                   0 |                  84 |
| 425 | open_images_v4                         | resnet_50                      | nesterov_momentum |           64 |               132 |                  50 |                 140 |
| 426 |                                        |                                |                   |          128 |               103 |                  46 |                  80 |
| 427 |                                        |                                |                   |          256 |               176 |                  20 |                 107 |
| 428 |                                        |                                |                   |          512 |               105 |                  21 |                  58 |
| 429 |                                        |                                |                   |         1024 |               117 |                  33 |                  66 |
| 430 |                                        |                                |                   |         2048 |               107 |                   2 |                  31 |
| 431 |                                        |                                |                   |         4096 |               109 |                   0 |                  60 |
| 432 |                                        |                                |                   |         8192 |               100 |                   3 |                  30 |
| 433 |                                        |                                |                   |        16384 |               102 |                   0 |                  23 |
| 434 |                                        |                                |                   |        32768 |               102 |                   2 |                  18 |
| 435 | solution_quality/fashion_mnist         | simple_cnn_base                | nesterov_momentum |            1 |               413 |                   3 |                 360 |
| 436 |                                        |                                |                   |            2 |               403 |                  10 |                 229 |
| 437 |                                        |                                |                   |            8 |               408 |                   9 |                 142 |
| 438 |                                        |                                |                   |           32 |               405 |                   7 |                 237 |
| 439 |                                        |                                |                   |          128 |               346 |                  69 |                 250 |
| 440 |                                        |                                |                   |          512 |               406 |                   9 |                 175 |
| 441 |                                        |                                |                   |         2048 |               223 |                 193 |                 184 |
| 442 |                                        |                                |                   |         8192 |               340 |                  74 |                 173 |
| 443 |                                        |                                |                   |        32768 |               409 |                   1 |                 156 |
| 444 |                                        |                                |                   |        55000 |               403 |                   7 |                 148 |
| 445 | solution_quality/mnist                 | simple_cnn_base                | nesterov_momentum |            1 |               413 |                   6 |                 263 |
| 446 |                                        |                                |                   |            2 |               415 |                   3 |                 149 |
| 447 |                                        |                                |                   |            8 |               418 |                   0 |                 104 |
| 448 |                                        |                                |                   |           32 |               417 |                   0 |                 160 |
| 449 |                                        |                                |                   |          128 |               398 |                  20 |                 153 |
| 450 |                                        |                                |                   |          512 |               418 |                   1 |                 143 |
| 451 |                                        |                                |                   |         2048 |               226 |                 192 |                  92 |
| 452 |                                        |                                |                   |         8192 |               408 |                   5 |                  83 |
| 453 |                                        |                                |                   |        32768 |               209 |                 209 |                  98 |
| 454 |                                        |                                |                   |        55000 |               359 |                  55 |                  96 |
