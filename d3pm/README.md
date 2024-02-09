
Code to accompany the NeurIPS paper
"Structured Denoising Diffusion Models in Discrete State-Spaces".

```
@inproceedings{austin2021structured,
author    = {Jacob Austin and
             Daniel D. Johnson and
             Jonathan Ho and
             Daniel Tarlow and
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

The `text` subdirectory contains our code for D3PMs in text space, corresponding
to our experiments in Section 5.

The `insertdelete` subdirectory contains code for D3PMs augmented with insertion
and deletion operations, as described in the follow-up paper "Beyond In-Place
Corruption: Insertion and Deletion In Denoising Probabilistic Models".

```
@inproceedings{johnson2021beyond,
author    = {Daniel D. Johnson and
             Jacob Austin and
             Rianne van den Berg and
             Daniel Tarlow},
title     = {Beyond In-Place Corruption: Insertion and Deletion In Denoising Probabilistic Models},
booktitle = {ICML Workshop on Invertible Neural Networks, Normalizing Flows, and Explicit Likelihood Models},
year      = {2021}
}
```