# Pushing the Boundaries of View Extrapolation with Multiplane Images

[Project Page](https://people.eecs.berkeley.edu/~pratul/publication/mpi_extrapolation/)

Code for the paper [Pushing the Boundaries of View Extrapolation with Multiplane Images](https://arxiv.org/abs/1905.00413)
Pratul P. Srinivasan, Richard Tucker, Jonathan T. Barron, Ravi Ramamoorthi, Ren Ng, Noah Snavely, CVPR 2019.
If you use this code, please cite our paper:

```
@article{srinivasan19,
  author    = {Pratul P. Srinivasan and Richard Tucker and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng and Noah Snavely},
  title     = {Pushing the Boundaries of View Extrapolation with Multiplane Images},
  journal   = {CVPR},
  year      = {2019}}
}
```

Both this CVPR 2019 paper and the included code are based on the [Stereo Magnification](https://people.eecs.berkeley.edu/~tinghuiz/projects/mpi/) method for training deep networks to predict MPIs.

This release contains code for predicting an MPI from a narrow-baseline pair of RGB images, and rendering novel camera views from this predicted MPI.

## Running a pretrained model

`render_sway.py` contains an example script for running a pretrained model on an input batch to render a novel view camera path. Please download and unzip the [pretrained model](https://drive.google.com/file/d/1Mf3t2SAl7vhAK4LaAuiMLr7LRTLrA-oD/view?usp=sharing) and [training examples](https://drive.google.com/file/d/1xBpQzJwQJjx9fc1ild59IoQzIeevwAKx/view?usp=sharing) files, and then include the corresponding file/directory names as command line flags when running ``render_sway.py``.

Example usage (edit paths to match your directory structure): ``python -m mpi_extrapolation.render_sway --input_file="mpi_extrapolation/examples/0.npz" --output_dir="mpi_extrapolation/outputs/dir/0/" --model_dir="mpi_extrapolation/models/"``

## Training

Please refer to the ``build_train_graph()`` and ``train()`` functions in ``mpi.py`` for starter code to use for training your own model.

This model was trained using the [RealEstate10K](https://google.github.io/realestate10k/) dataset. This code release currently does not contain a Tensorflow dataset loader/iterator, but please refer to the code release for [Stereo Magnification](https://github.com/google/stereo-magnification) for an example.

If you would like to train using a perceptual loss based on VGG features, please download the ``imagenet-vgg-verydeep-19.mat`` [pretrained VGG model](http://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models), and pass the corresponding path to the ``build_train_graph()`` function when setting up your training op.

## Extra

Visualizing MPIs with interactive viewers is really fun! Check out the repositories for these two related papers for examples: [Local Light Field Fusion](https://people.eecs.berkeley.edu/~bmild/llff/), [DeepView](https://augmentedperception.github.io/deepview/).

If you are experiencing any issues or seeing unexpected outputs, it may be helpful to test your environment using ``predict_mpi_test.py``. First, download a [ground truth MPI](https://drive.google.com/file/d/13k3LRZIC_T-8Juyf7850-EvOteGtsk83/view?usp=sharing), put it in the same directory as the input batch examples, and then run the test.

Example usage (edit paths to match your directory structure): ``python -m mpi_extrapolation.predict_mpi_test --input_dir="mpi_extrapolation/examples/" --model_dir="mpi_extrapolation/models/"``

This conducts a test using one of the provided examples to make sure that the predicted MPI is the same as one computed at the time of this code release.

This code repository is shared with all of Google Research, so it's not very useful for reporting or tracking bugs. If you have any issues using this code, please do not open an issue, and instead just email pratul@berkeley.edu.
