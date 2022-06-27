# Autoregressive Diffusion Models

### Emiel Hoogeboom, Alexey A. Gritsenko, Jasmijn Bastings, Ben Poole, Rianne van den Berg and Tim Salimans

## Summary

*   New generative model that models variables in any order, i.e. order
    agnostically.
*   ARDMs are trained with inspirations from diffusion, by sampling a step in
    the generative process, and optimizing that step.
*   ARDMs perform well on three data modalities: images, text and audio.
*   ARDMs adaptable parallel generation make (de)compression more practical,
    since only a modest amount of network calls is required.

Paper: [arXiv](https://arxiv.org/abs/2110.02037)

## Abstract

We introduce Autoregressive Diffusion Models (ARDMs), a model class encompassing
and generalizing order-agnostic autoregressive models (Uria et al., 2014) and
absorbing discrete diffusion (Austin et al., 2021), which we show are a special
cases of ARDMs under mild assumptions. ARDMs are simple to implement and easy to
train. Unlike standard ARMs, they do not require causal masking of model
representations, and can be trained using an efficient objective similar to
modern probabilistic diffusion models that scales favourably to
highly-dimensional data. At test time, ARDMs support parallel generation which
can be adapted to fit any given generation budget. We find that ARDMs require
significantly fewer steps than discrete diffusion models to attain the same test
performance. Finally, we apply ARDMs to lossless compression, and show that they
are uniquely suited to this task. Contrary to existing approaches based on
bits-back coding, ARDMs obtain compelling results not only on complete datasets,
but also on compressing single data points. Moreover, the model's adaptable
parallel generation affords these results using a modest number of network calls
for (de)compression, making a significant step towards practical compression
using neural networks.

## Source code

The code is divided into a `model/` folder which contains all important model
logic that is shared between data modalities. In addition, the data-specific
code is located in `experiments/`. Here the experiment can be launched for each
data modality by running main.py.

## Citation

```none
@article{hoogeboom2021ardm,
    title={Autoregressive Diffusion Models},
    author={Emiel Hoogeboom and Alexey A. Gritsenko and Jasmijn Bastings and Ben Poole and Rianne van den Berg and Tim Salimans},
    journal={arXiv preprint arXiv:2110.02037},
    year={2021}
}
```
