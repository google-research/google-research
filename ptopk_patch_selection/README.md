# Differentiable Patch Selection for Image Recognition 

-----------------------------------------------------------------
### UPDATE
A few bugs in our implementation were brought to our attention by Zongtao ([Github](https://github.com/wztdream)) and 
Tim Beyer ([Google Scholar](https://scholar.google.com/citations?user=mad4ZNsAAAAJ))

The bugs are
* We were using the transpose jacobian instead of the jacobian.
* We take a dot product when computing the expected gradient by multiplying the `perturbed_output` and `noise_gradient`, whereas an outer product needs to be computed.

A bug fixed pytorch implementation is available at https://github.com/benbergner/dps . We apologies for any inconvenience caused.

-----------------------------------------------------------------

This is a reference implementation for 
"Differentiable Patch Selection for Image Recognition"
by Jean-Baptiste Cordonnier, Aravindh Mahendran,
Alexey Dosovitskiy, Dirk Weissenborn, Jakob Uszkoreit, and Thomas Unterthiner.


Contact: [aravindhm@google.com](mailto:aravindhm@google.com)

When using this code, please cite the paper:

```
@article{cordonnier2021differentiable,
    title={Differentiable Patch Selection for Image Recognition},
    author={Cordonnier, Jean-Baptiste and Mahendran, Aravindh and Dosovitskiy, Alexey and Weissenborn, Dirk and Uszkoreit, Jakob and Unterthiner, Thomas}
    journal={CVPR},
    year={2021}
}
```

## Requirements

This code was developed using [JAX](https://github.com/jax-ml/jax) and
[FLAX](https://github.com/google/flax). It also requires the following
packages:

```
chex
clu
einops
numpy
ml_collections
optax
```

## Dataset

This release includes the data for the `Billiard Experiments` (see paper). To
reproduce these (or re-use the data for your own purposes) please download
the data from
http://storage.googleapis.com/gresearch/ptokp_patch_selection/billiard.tar.xz
and extract it to "libs/datasets/billiard".

## Running

To run the data, pick a config file and run it, for example:

```
python3 image_classification.py --config configs/billiard/topk.py --workdir /tmp/topk
```

## License

This repository is licensed under the Apache License, Version 2.0. See LICENSE for details.

## Disclaimer

This is not an official Google product.
