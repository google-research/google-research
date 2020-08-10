# Single-View View Synthesis with Multiplane Images
## Richard Tucker and Noah Snavely, CVPR 2020

This release contains code for defining a model to predict 32-layer Multiplane Images (MPIs) from individual input RGB images, for rendering novel views from the resulting MPIs, and for generating disparity maps. It uses Tensorflow 2.

The notebook `example.ipynb` [[click to open in Google Colab](https://colab.research.google.com/github/google-research/google-research/blob/master/single_view_mpi/example.ipynb)]
demonstrates how to:

  * define the network
  * load pre-trained weights for our "full" model
  * run inference on an example image
  * render new views and synthesized disparity
  * visualize the MPI in a 3D web-based viewer.

The library code in `libs` is organized as follows:

  * `utils.py` – various utility function
  * `geometry.py` – camera poses, projections, homographies
  * `mpi.py` – MPI-rendering
  * `nets.py` – the MPI-prediction network

The `run.sh` script runs some basic unit tests of these libraries.

### Links

Our project page with links to the paper, video and interactive examples is at **[single-view-mpi.github.io](https://single-view-mpi.github.io/)**.

### Citation details
```
@InProceedings{single_view_mpi,
  author = {Tucker, Richard and Snavely, Noah},
  title = {Single-view View Synthesis with Multiplane Images},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2020}
}
```
