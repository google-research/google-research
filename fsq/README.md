# Finite Scalar Quantization: VQ-VAE Made Simple

### [[Paper on ArXiv](https://arxiv.org/abs/2309.15505)] [[Colab with FSQ code](https://colab.research.google.com/github/google-research/google-research/blob/master/fsq/fsq.ipynb)]



## Abstract

We propose to replace vector quantization (VQ) in the latent representation
of VQ-VAEs with a simple scheme termed finite scalar quantization (FSQ), where
we project the VAE representation down to a few dimensions (typically less than
10). Each dimension is quantized to a small set of fixed values, leading to an
(implicit) codebook given by the product of these sets. By appropriately
choosing the number of dimensions and values each dimension can take, we obtain
the same codebook size as in VQ. On top of such discrete representations, we
can train the same models that have been trained on VQ-VAE representations. For
example, autoregressive and masked transformer models for image generation,
multimodal generation, and dense prediction computer vision tasks. Concretely,
we employ FSQ with MaskGIT for image generation, and with UViM for depth
estimation, colorization, and panoptic segmentation. Despite the much simpler
design of FSQ, we obtain competitive performance in all these tasks. We
emphasize that FSQ does not suffer from codebook collapse and does not need the
complex machinery employed in VQ (commitment losses, codebook reseeding, code
splitting, entropy penalties, etc.) to learn expressive discrete
representations.


## Comparison

|                  | VQ | FSQ |
|------------------|----|-----|
| Quantization     | argmin_c \|\| z-c \|\| | round(f(z)) |
| Gradients        | Straight Through Estimation (STE) | STE |
| Auxiliary Losses | Commitment, codebook, entropy loss, ... | N/A |
| Tricks           | EMA on codebook, codebook splitting, projections, ...| N/A |
| Parameters       | Codebook | N/A |


## Code

- [Official Colab with FSQ JAX code](https://colab.research.google.com/github/google-research/google-research/blob/master/fsq/fsq.ipynb)
- [External PyTorch port](https://github.com/lucidrains/vector-quantize-pytorch#finite-scalar-quantization)


