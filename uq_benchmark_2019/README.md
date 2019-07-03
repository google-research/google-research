# Can You Trust Your Model's Uncertainty?
## Evaluating Predictive Uncertainty Under Dataset Shift

This repository contains code used to run experiments in
[Can You Trust Your Model's Uncertainty? Evaluating Predictive Uncertainty Under Dataset Shift](https://arxiv.org/abs/1906.02530),
which benchmarks a handful of methods for training models with robust uncertainty on varying datasets with off-distribution holdout test-sets.

To facilitate follow-up work, we also make available all of our trained
models alongw ith their predictions on each dataset at the GCS bucket:
`gs://uq_benchmark_2019/`.

The study compared the following methods and datasets:

### Methods
1. **Vanilla**: Maximum-likelihood DNN.
2. **Temperature Scaling**: Vanilla model calibrated on in-distribution
3. **Ensemble** of vanilla models.
4. **Dropout**: MC Dropout as described by [Gal & Ghahramani](http://proceedings.mlr.press/v48/gal16.pdf)
5. **SVI**: Mean field BNN optimized by stochastic variational inference.
6. **LL-Dropout** and **LL-SVI**: Simplified variants of Dropout and SVI where the method is only applied to the last layer.

### Datasets
1. MNIST and rotated MNIST.
2. CIFAR-10 with corruptions by [Hendrycks et al. 2019](https://github.com/hendrycks/robustness)
3. ImageNet 2012 with corruptions by [Hendrycks et al. 2019](https://github.com/hendrycks/robustness)
4. Criteo's ad-click prediction dataset with synthetic random feature corruptions.
5. 20 Newsgroups text with out-of-distribution data from LM1B.

**Abstract:**
Modern machine learning methods including deep learning have achieved great success in predictive accuracy for supervised learning tasks, but may still fall short in giving useful estimates of their predictive _uncertainty_. Quantifying uncertainty is especially critical in real-world settings, which often involve input distributions that are shifted from the training distribution due to a variety of factors including sample bias and non-stationarity. In such settings, well calibrated uncertainty estimates convey information about when a model's output should (or should not) be trusted. Many probabilistic deep learning methods, including Bayesian-and non-Bayesian methods, have been proposed in the literature for quantifying predictive uncertainty, but to our knowledge there has not previously been a rigorous large-scale empirical comparison of these methods under dataset shift. We present a large-scale benchmark of existing state-of-the-art methods on classification problems and investigate the effect of dataset shift on accuracy and calibration. We find that traditional post-hoc calibration does indeed fall short, as do several other previous methods. However, some methods that marginalize over models give surprisingly strong results across a broad spectrum of tasks.

