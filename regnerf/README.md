# RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs

This codebase contains code for the paper [RegNeRF](https://arxiv.org/abs/2112.00724).
Please refer to the [project website](https://m-niemeyer.github.io/regnerf) for more information.

The code base is built upon mip-NeRF which can be found here:
https://github.com/google/mipnerf.

## About RegNeRF

![Teaser Image](readme_imgs/teaser.png)

Neural Radiance Fields (NeRF) have emerged as a powerful representation for the
task of novel view synthesis due to their simplicity and state-of-the-art
performance. Though NeRF can produce photorealistic renderings of unseen
viewpoints when many input views are available, its performance drops
significantly when this number is reduced. We observe that the majority of
artifacts in sparse input scenarios are caused by errors in the estimated scene
geometry, and by divergent behavior at the start of training. We address this by
regularizing the geometry and appearance of patches rendered from unobserved
viewpoints, and annealing the ray sampling space during training. We
additionally use a normalizing flow model to regularize the color of unobserved
viewpoints. Our model outperforms not only other methods that optimize over a
single scene, but in many cases also conditional models that are extensively
pre-trained on large multi-view datasets.

**TL;DR:** We regularize unseen views during optimization to enable view
synthesis from sparse inputs with as few as 3 input images.

## Running the code
TODO.
