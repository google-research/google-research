
Code to accompany the NeurIPS paper
"Structured Denoising Diffusion Models in Discrete State-Spaces".

```
@inproceedings{lorberbom2021generalized,
author    = {Jacob Austin and
             Daniel D. Johnson and
             Daniel Tarlow and
             Jonathan Ho and
             Rianne van den Berg},
title     = {Structured Denoising Diffusion Models in Discrete State-Spaces},
booktitle = {Advances in Neural Information Processing Systems},
year      = {2021}
}
```

The `images` subdirectory contains the code for D3PMs in image space, which can
be used to to reproduce our experiments from Section 6.
It contains code for evaluating bits-per-dimension metrics, but
the FID and IS metric evaluation code is not currently available.

We plan to add a new subdirectory `text` which will contain our code for D3PMs
in text space, corresponding to our experiments in Section 5.
