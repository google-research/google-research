# Dimensions of Motion: Monocular Prediction through Flow Subspaces
### Richard Strong Bowen, Richard Tucker, Ramin Zabih, Noah Snavely 3DV 2022


This release contains code for the model definition, basis formulation and losses from our "Dimensions of Motion" paper.

The library code in `libs` is organized as follows:

* `flow_basis.py` – functions to generate flow bases for camera rotation and translation from disparity
* `geometry.py` – flow warping for visualization
* `loss.py` – flow projection loss
* `nets.py` – the scene-prediction network
* `solvers.py` – the SVD solver for projecting flow into a subspace
* `utils.py` – some simple utility functions

The notebook `dimensions_of_motion_example.ipynb`
[[click to open in Google Colab](https://colab.research.google.com/github/google-research/google-research/blob/master/dimensions_of_motion/dimensions_of_motion_example.ipynb)]
shows how to apply our loss functions to a predicted disparity and object embedding, and how to visualize the resulting basis flow fields.


### Links

Our project page with links to the paper, video and examples is at **[dimensions-of-motion.github.io](https://dimensions-of-motion.github.io/)**.

### Citation details
```
@inproceedings{bowen2022dimensions,
  title     = {Dimensions of Motion: Monocular Prediction through Flow Subspaces},
  author    = {Richard Strong Bowen and Richard Tucker and Ramin Zabih and Noah Snavely},
  booktitle = {Proceedings of the International Conference on {3D} Vision (3DV)},
  year      = {2022}
}
```
