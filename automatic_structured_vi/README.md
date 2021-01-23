# Automatic Structured Variational Inference

This repository contains code used to run experiments in
[Automatic Structured Variational Inference](https://arxiv.org/abs/2002.00643),
which proposes a fully automated method for constructing a structured surrogate
posterior for VI in a way that incorporates the graphical structure of the
prior distribution.

## How to Launch the Code
### Create a new virtual environment.
```
virtualenv -p python3
source ./bin/activate
```
### Download code and install dependencies.
```
svn export https://github.com/google-research/google-research/trunk/automatic_structured_vi
pip install -r automatic_structured_vi/requirements.txt
```
### Evaluate ASVI on the brownian motion model.
```
python -m automatic_structured_vi.run_vi --model_name=brownian_motion --posterior_type=asvi --num_steps=1000
```

This should generate a loss plot, a JSON of samples from the posterior, and a JSON continaing values including losses and the final ELBO.

## Experimental Pipeline
The experimental pipeline in this repository compares the following variational
posteriors on the following models from TensorFlow Probability's [Inference Gym](https://pypi.org/project/inference-gym/):

### Variational Posteriors
1. **ASVI**: Automatic structured variational inference
2. **Mean-field**: Mean-field ADVI
3. **Small IAF** Inverse autoregressive flows with eight hidden units
4. **Large IAF**: Inverse autoregressive flows with 512 hidden units
5. **MVN**: Multivariate normal posterior
6. **AR(1)**: Autoregressive model

### Inference Gym Models
1. **Brownian Motion**: 30-step Brownian motion without drift, as well as a
variant that includes global variables where the innovation and observation
noise scale parameters are unknown
2. **Lorenz Bridge**: 30-step Stochastic Lorenz dynamical system, as well as a
variant that includes global variables where the innovation and observation
noise scale parameters are unknown
3. **Eight Schools**: Standard Bayesian hierarchical model as described in
[Gelman et al. 2013](http://www.stat.columbia.edu/~gelman/book/)
3. **Radon**: Hierarchical linear regression model as described in
[Gelman and Hill, 2007](http://www.stat.columbia.edu/~gelman/arm/)

**Abstract:**
Stochastic variational inference offers an attractive option as a default method for differentiable probabilistic programming. However, the performance of the variational approach depends on the choice of an appropriate variational family. Here, we introduce automatic structured variational inference (ASVI), a fully automated method for constructing structured variational families, inspired by the closed-form update in conjugate Bayesian models. These pseudo-conjugate families incorporate the forward pass of the input probabilistic program and can therefore capture complex statistical dependencies. Pseudo-conjugate families have the same space and time complexity of the input probabilistic program and are therefore tractable for a very large family of models including both continuous and discrete variables.  We validate our automatic variational method on a wide range of both low- and high-dimensional inference problems. We find that ASVI provides a clear improvement in performance when compared with other popular approaches such as mean field family and inverse autoregressive flows. We provide a fully automatic open source implementation of ASVI in TensorFlow Probability.


