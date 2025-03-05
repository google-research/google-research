# JaxBARF

This is a [JAX](https://github.com/jax-ml/jax) implementation of [BARF: Bundle-Adjusting Neural Radiance Fields](https://arxiv.org/abs/2104.06405). A PyTorch implementation of BARF has been open-sourced by the authors [here](https://github.com/chenhsuanlin/bundle-adjusting-NeRF).

This code is built on [JaxNeRF](https://github.com/google-research/google-research/tree/master/jaxnerf) modified to match the BARF model (trainable camera poses, modified MLP, etc).

## Citation
If you use this software package, please also cite the original BARF paper:

```
@inproceedings{lin2021barf,
  title={BARF: Bundle-Adjusting Neural Radiance Fields},
  author={Lin, Chen-Hsuan and Ma, Wei-Chiu and Torralba, Antonio and Lucey, Simon},
  booktitle={IEEE International Conference on Computer Vision ({ICCV})},
  year={2021}
}
```

## Training on Blender scenes.
To train the model on Blender scenes you first need to download the data provided with the [NERF Repo](https://github.com/bmild/nerf).

### Installation.
Installation (environment setup) commands are contained in the following script:

```
./install.sh
```

### Training on a GPU.
The following script trains the model sequentially for each Blender scene:

```
./train.sh
```

The results from the training are reported below (juxtaposed with PyTorch-BARF
results).

### Rotation error (in degree)

| Scene   |   Chair   |   Drums   |   Ficus   |   Hotdog  |    Lego   | Materials |    Mic    |    Ship   |    Mean   |
|---------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| Pytorch-BARF | 0.043 |  0.043   |   0.085   |   0.248   |   0.082   |   0.844   |   0.071   |   0.075   |   0.193   |
| JaxBARF      | 0.130 |  0.053   |   0.094   |   0.262   |   0.086   |   0.405   |   0.079   |   0.108   |   0.152   |

### Translation error

| Scene   |   Chair   |   Drums   |   Ficus   |   Hotdog  |    Lego   | Materials |    Mic    |    Ship   |    Mean   |
|---------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| Pytorch-BARF | 0.428 |  0.225   |   0.474   |   1.308   |   0.291   |   2.692   |   0.301   |   0.326   |  0.756    |
| JaxBARF |      0.47  |  0.17    |   0.40    |   1.41    |   0.25    |   1.78    |   0.25    |   0.38    |  0.639    |


**Note**: Since the initial camera pose perturbations are generated randomly, the results are sensitive to choice of random seed, and we do not expect the results to match exactly between implementations. We see a large variance in results across random seeds for both the JAX-BARF and PyTorch-BARF libraries.

### Evaluation
The included evaluation file `src/eval.py` does not implement the test-time
optimization used in BARF so the view synthesis results will not be comparable
to the BARF synthesis examples.

### Implementation Notes
Many of the modules like camera.py are translated almost line-by-line from PyTorch which might not result in the best or most efficient JAX implementation.
