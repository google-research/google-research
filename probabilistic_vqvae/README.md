This directory contains code for vector quantized-variational autoencoder (VQ-VAE) experiments applied to MNIST.

## Overview

`mnist_experiments.py` compares the following models and training procedures (note that there are not yet necessarily tuned):

1. Baseline VQ-VAE.
2. Categorical variational distribution.
    1. Using loss from [Roy et al. (2018)](https://arxiv.org/abs/1805.11063).
    2. Adding entropy term to loss.
    3. Correctly scaling entropy and prior terms in loss.
    4. Not using straight-through on the prior.
    5. Adding IAFs.
3. Gumbel-Softmax variational distribution.
4. Gumbel-Softmax VQ-VAE with IAFs.
    1. Joint training.
    2. Two-stage training.
        1. Still training other parameters after introducing IAF.
        2. Not training other parameters after introducing IAF.

## Running Locally

To train a model locally:

```bash
# Run with standard hyperparameters:
# From google-research/
python -m probabilistic_vqvae.mnist_experiments
```


The following commands will train specific hyperparameter and model settings:

1\. Baseline VQ-VAE.

```bash
$ python -m probabilistic_vqvae.mnist_experiments \
  --bottleneck_type=deterministic
```

2\.1\. Categorical variational distribution with loss from Roy et al. (2018).

```bash
$ python -m probabilistic_vqvae.mnist_experiments \
  --bottleneck_type=categorical --sum_over_latents=False \
  --entropy_scale=0.0 --num_samples=10 --stop_gradient_for_prior=True
```

2\.2\. Loss from Roy et al. (2018) plus entropy term.

```bash
$ python -m probabilistic_vqvae.mnist_experiments \
  --bottleneck_type=categorical --sum_over_latents=False \
  --entropy_scale=1.0 --num_samples=10  --stop_gradient_for_prior=True \
  --beta=0.05
```

2\.3\. Same as above but correctly scaling loss.

```bash
$ python -m probabilistic_vqvae.mnist_experiments \
  --bottleneck_type=categorical --sum_over_latents=True \
  --entropy_scale=1.0 --num_samples=10  --stop_gradient_for_prior=True \
  --beta=0.05
```

2\.4\. No longer stopping gradient for prior.

```bash
$ python -m probabilistic_vqvae.mnist_experiments \
  --bottleneck_type=categorical --sum_over_latents=True \
  --entropy_scale=1.0 --num_samples=10 --stop_gradient_for_prior=False \
  --beta=0.05
```

2\.5\. Categorical bottleneck with IAFs.

```bash
$ python -m probabilistic_vqvae.mnist_experiments \
--bottleneck_type=categorical --sum_over_latents=False --entropy_scale=1.0 \
--num_samples=10 --beta=0.05 --num_iaf_flows=1 --stop_gradient_for_prior=True \
--stop_training_encoder_after_startup=True --iaf_startup_steps=5000
```

3\. Gumbel-Softmax variational distribution.

```bash
$ python -m probabilistic_vqvae.mnist_experiments \
  --bottleneck_type=gumbel_softmax --sum_over_latents=True \
  --entropy_scale=1.0 --num_samples=10 --beta=0.05
```

4\.1\. Gumbel-Softmax variational distribution with joint training and IAFs.

```bash
$ python -m probabilistic_vqvae.mnist_experiments \
  --bottleneck_type=gumbel_softmax --sum_over_latents=True \
  --entropy_scale=1.0 --num_samples=10 --beta=0.05 --num_iaf_flows=1 \
  --iaf_startup_steps=0
```

4\.2\.1\. Gumbel-Softmax variational distribution with two-stage training (still training encoder after introducing IAFs).

```bash
$ python -m probabilistic_vqvae.mnist_experiments \
  --bottleneck_type=gumbel_softmax --sum_over_latents=True \
  --entropy_scale=1.0 --num_samples=10 --beta=0.05 --num_iaf_flows=1 \
  --iaf_startup_steps=5000 --stop_training_encoder_after_startup=False
```

4\.2\.2\. Gumbel-Softmax variational distribution with two-stage training (stop
training encoder after startup steps).

```bash
$ python -m probabilistic_vqvae.mnist_experiments \
  --bottleneck_type=gumbel_softmax --sum_over_latents=True \
  --entropy_scale=1.0 --num_samples=10 --beta=0.05 --num_iaf_flows=1 \
  --iaf_startup_steps=5000 --stop_training_encoder_after_startup=True
```
