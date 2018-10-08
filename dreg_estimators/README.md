# Doubly Reparameterized Gradient (DReG) Estimators
*Implemention of DReG estimators as described in [Doubly Reparameterized Gradient Estimators for Monte Carlo Objectives](https://drive.google.com/open?id=1XmNbCn4-AuzobofdHzFaAwb1nLTHX-s-) by George Tucker (gjt@google.com), Dieterich Lawson, Shixiang Gu, Chris J. Maddison.*

Deep latent variable models have become a popular model choice due to the scalable learning algorithms introduced by (Kingma & Welling, 2013; Rezende et al., 2014). These approaches maximize a variational lower bound on the intractable log likelihood of the observed data. Burda et al. (2015) introduced a multi-sample variational bound, IWAE, that is at least as tight as the standard variational lower bound and becomes increasingly tight as the number of samples increases.  Counterintuitively, the typical inference network gradient estimator for the IWAE bound performs poorly as the number of samples increases (Rainforth et al., 2018; Le et al., 2018). Roeder et al. (2017) propose an improved gradient estimator, however, are unable to show it is unbiased. We show that it is in fact biased and that the bias can be estimated efficiently with a second application of the reparameterization trick. The doubly reparameterized gradient (DReG) estimator does not suffer as the number of samples increases, resolving the previously raised issues.  The same idea can be used to improve many recently introduced training techniques for latent variable models. In particular, we show that this estimator reduces the variance of the IWAE gradient, the reweighted wake-sleep update (RWS) (Bornschein & Bengio, 2014), and the jackknife variational inference (JVI) gradient (Nowozin, 2018). Finally, we show that this computationally efficient, drop-in estimator translates to improved performance for all three objectives on several modeling tasks.

This repository implements DReG estimators for IWAE, RWS, and JVI for continuous latent variable generative models. For comparison, we also implemented the approach from [Roeder et al. 2017 NIPS](http://papers.nips.cc/paper/7268-sticking-the-landing-simple-lower-variance-gradient-estimators-for-variational-inference).

We hope that this code will be a useful starting point for future research in this area.

## Quick Start:

Requirements:
* TensorFlow (see tensorflow.org for how to install)
* MNIST dataset
* Omniglot dataset

First download the MNIST and Omniglot datasets. E.g., from:
* [MNIST train_xs.npy](https://drive.google.com/open?id=1BaEWtwo3SQ8m7_Xs9VpTEPX10zpdbklX)
* [MNIST valid_xs.npy](https://drive.google.com/open?id=1Z4ItIhpUMXF_NIx3k_14pCTMyeQV8v69)
* [MNIST test_xs.npy](https://drive.google.com/open?id=1OsyM_2tlZOoPGHYM7tQs8KTpAxSo5bYq)
* [Omniglot](https://drive.google.com/open?id=1ZgNzUjHskBbwZd4so0VxILkVFMCm0hbg)

Then edit utils.py to point to the datasets

```
MNIST_LOCATION = "your/path"
OMNIGLOT_LOCATION = "your/path"
```

Then, to run the train/eval loop,

```
# From the deepest google-research/ run:
python -m dreg_estimators.main_loop
```

The model components are in model.py and the training/eval loop is in
main_loop.py. The main_loop script can run in three modes:
* Standard train/eval with a specific estimator (set the --estimator flag)
* Bias checking mode, which computes t statistics of the gradient estimator versus the
  standard IWAE gradient estimator. This does not train or evaluate the model.
  (set --bais_check to run this mode)
* Just training the inference network from a fixed checkpoint. (set
  --initial_checkpoint_dir to run this mode)
Check main_loop.py to see a description of the options.

This is not an officially supported Google product. It is maintained by George Tucker (gjt@google.com, [@georgejtucker](https://twitter.com/georgejtucker), github user: gjtucker).
