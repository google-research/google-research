# There and back again: Cycle consistency across sets for isolating factors of isolation


This code accompanies the paper

**There and back again: Cycle consistency across sets for isolating factors of isolation** \
Kieran A. Murphy, Varun Jampani, Srikumar Ramalingam, Ameesh Makadia

[arXiv link available soon]

## Repository Contents
- `shapes3d/` Trains a model from scratch to isolate generative factors from
the [Shapes3D dataset](https://github.com/deepmind/3d-shapes)
- `mnist.ipynb` iPython notebook to isolate digit style (stroke and thickness)
after training on images grouped by digit class
- `pose_estimation/` Evaluates trained pose estimation models from
Section 4.3 of the paper in both the dictionary lookup and regression scenarios
- `run.sh` runs a minimal example of the Shapes3D experiment for testing

## Python environment
The code was run successfully with Python 3.6.12.  The necessary libraries may
be installed using pip with the following line:

`pip install -r requirements.txt`

## Isolating factors of variation from Shapes3D

A large variety of experiments may be run using the script `shapes3d/train.py`.  The [Shapes3D dataset](https://github.com/deepmind/3d-shapes) will be automatically downloaded by [tensorflow_datasets](https://www.tensorflow.org/datasets/catalog/shapes3d).  

The following example call

`python -m shapes3d.train --inactive_vars=03 --curate_both_stacks=False`

trains a network from scratch with the wall hue and scale generative factors inactive, with only one out of each pair of training sets curated.  The second set, for every training batch, will be sampled randomly across all images (the 'One random set' variant of the experiments in Figure 3c).

The command line flag `inactive_vars` takes a string of digits from 0-5, one for
each of the six generative factors of the Shapes3D dataset (wall hue, object
hue, floor hue, scale, shape, and orientation).  `01` curates stacks with
wall and object hue as inactive variables, for example.  Note that this curation
process uses `tf.data.Dataset.filter` to run through the Shapes3D dataset, which
requires searching through more of the dataset to find each training set when
there are more inactive factors of variation.

Other noteworthy flags:

- `save_pngs` outputs images during training of sample embeddings and the mutual
information, as in Figure 3b of the manuscript.  Only effective if
`num_latent_dims=2`.

- `similarity_type` sets which distance metric to use when computing the loss.
The best results seem to come from squared Euclidean distance (`l2sq`), but
several others are implemented and there's room to explore.

- `run_augmentation_experiment` is a boolean flag which will run the double augmentation comparison
in the bottom subplot of Figure 3c if set to `True`.

Training progress can be monitored with Tensorboard.

## Fast digit style isolation on MNIST

The [iPython notebook](mnist.ipynb) partitions the MNIST training set into 10 different tf.data.Datasets,
with the option to withhold one digit for test time (as in the paper).  Embedding visualizations are of the two PCA dimensions with the largest variance, also as in the paper.

## Pose estimation on Pascal3D+

#### Trained models
We supply trained models for the pose estimation results of the paper at the
following link:

https://storage.googleapis.com/gresearch/isolating-factors/pose_estimation_models.zip

The four directories in the zip file 

- `car_lookup/`
- `chair_lookup/`
- `car_regression/`
- `chair_regression/`

are keras models which may be loaded with `tf.keras.models.load_model()`.  The two lookup models are from the annotation-less part of the pose estimation results, where a lookup dictionary of synthetic images was used to convert the 64-dimensional embeddings to a rotation.  These models provided the results of Table 1.

The two regression models were trained with a spherical regression head ([Liao et al. 2019](https://ivi.fnwi.uva.nl/isis/publications/2019/LiaoCVPR2019/LiaoCVPR2019.pdf)) on top of an embedding space conditioned with the CCS loss. These models provided the results of Table 2.

#### Datasets

Both were evaluated on Pascal3D+; the dataset may be downloaded [here](ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip). 
We have included the image names and annotations used for the test set in the paper in the files

- `pose_estimation/car_test.txt`
- `pose_estimation/chair_test.txt`

Additionally, to evaluate the embedding models, renderings from the KeypointNet paper were used for the dictionary and may be downloaded (as tf records) through [the KeypointNet project page](https://keypointnet.github.io/).

#### Inference and evaluation

The eval script loads the images listed in the `txt` files and parses the rotation annotations.  The regression and lookup settings may be evaluated by providing filepaths to the model and data.

`python -m pose_estimation.run_pose_estimation --model_dir path/to/regression.model --images_dir path/to/images --object_class car --mode regression`

and

`python -m pose_estimation.run_pose_estimation --model_dir path/to/lookup/model --images_dir path/to/images --dict_dir path/to/dict/dir --object_class car --mode lookup`

will run the regression and lookup models, respectively, and print the median angular error.  Note the lookup results (Table 1 of the paper) report the mean of the median angular error over 50 random dictionaries from the ShapeNet images, so some variation is expected in the error values for the lookup scenario.
