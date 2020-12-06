### This codebase is no longer maintained, active development has moved to [Uncertainty Baselines](https://github.com/google/uncertainty-baselines).



# Can You Trust Your Model's Uncertainty?
## Evaluating Predictive Uncertainty Under Dataset Shift

This repository contains code used to run experiments in
[Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift](https://arxiv.org/abs/1906.02530),
which benchmarks a handful of methods for training models with robust uncertainty on varying datasets with off-distribution holdout test-sets.

The study compared the following methods and datasets:

### Methods
1. **Vanilla**: Maximum-likelihood DNN.
2. **Temperature Scaling**: Vanilla model calibrated on in-distribution as described in [Guo et al. 2017](https://arxiv.org/abs/1706.04599)

3. **Ensemble** of vanilla models as described in [Lakshminarayanan et al. 2017](https://arxiv.org/abs/1612.01474)
4. **Dropout**: MC Dropout as described by [Gal & Ghahramani](http://proceedings.mlr.press/v48/gal16.pdf)
5. **SVI**: Mean field BNN optimized by stochastic variational inference.
6. **LL-Dropout** and **LL-SVI**: Simplified variants of Dropout and SVI where the method is only applied to the last layer.

### Datasets
1. MNIST and rotated MNIST.
2. CIFAR-10 with corruptions by [Hendrycks & Dietterich 2019](https://github.com/hendrycks/robustness)
3. ImageNet 2012 with corruptions by [Hendrycks & Dietterich 2019](https://github.com/hendrycks/robustness)
4. Criteo's [ad-click prediction dataset](https://www.kaggle.com/c/criteo-display-ad-challenge) with synthetic random feature corruptions.
5. [20 Newsgroups text](http://qwone.com/~jason/20Newsgroups/) with out-of-distribution data from LM1B.

**Abstract:**
Modern machine learning methods including deep learning have achieved great success in predictive accuracy for supervised learning tasks, but may still fall short in giving useful estimates of their predictive _uncertainty_. Quantifying uncertainty is especially critical in real-world settings, which often involve input distributions that are shifted from the training distribution due to a variety of factors including sample bias and non-stationarity. In such settings, well calibrated uncertainty estimates convey information about when a model's output should (or should not) be trusted. Many probabilistic deep learning methods, including Bayesian-and non-Bayesian methods, have been proposed in the literature for quantifying predictive uncertainty, but to our knowledge there has not previously been a rigorous large-scale empirical comparison of these methods under dataset shift. We present a large-scale benchmark of existing state-of-the-art methods on classification problems and investigate the effect of dataset shift on accuracy and calibration. We find that traditional post-hoc calibration does indeed fall short, as do several other previous methods. However, some methods that marginalize over models give surprisingly strong results across a broad spectrum of tasks.


### Predictions
To facilitate follow-up work, we also make available all of our trained
models along with their predictions on each dataset at the GCS bucket:
[`uq-benchmark-2019`](https://console.cloud.google.com/storage/browser/uq-benchmark-2019).

The predictions files are structured as follows:
#### MNIST and CIFAR
`mnist_model_predictions.hdf5` and `cifar_model_predictions.hdf5` are HDF5 file with dataset group hierarchy:

`METHOD > DATASET > {labels, probs}`

where `METHOD` corresponds to the modeling method, and `DATASET` corresponds to an MNIST data split (or off-distribution variant) with posible corruptions.

Labels have shape `[5, N]` where `N` is the number of samples in the dataset, and 5 is the number of independently trained models for verifying reproducibility. Probabilities datasets have shape `[5, N, 10]` where 10 is the number of MNIST / CIFAR classes.

#### CRITEO
`criteo_model_predictions.hdf5` is organized identically to MNIST and CIFAR except the shape of the probabilities array is `[5, N]` since Criteo is not a multiclass problem.

#### IMAGENET
`imagenet_predictions.hdf5` is organized identically to the other datasets, except the probabilities array has shape `[N, 1000]` and the labels are a vector of length `N`. For the ensemble, dropout, and SVI methods, probabilties represent the means over ensemble members/samples. Probabilities less than `1e-6` have been rounded to 0 to increase gzip compressibilty.
