# Learn your ABCs: Approximate Bijective Correspondence for isolating factors of variation


This code accompanies the paper

[**Learn your ABCs: Approximate Bijective Correspondence for isolating factors of variation**](https://arxiv.org/abs/2103.03240) \
Kieran A. Murphy, Varun Jampani, Srikumar Ramalingam, Ameesh Makadia

<img src="https://kieranamurphy.com/s/abc983.png" width=983/>

## Repository Contents
- `shapes3d/` Trains a model from scratch to isolate generative factors from
the [Shapes3D dataset](https://github.com/deepmind/3d-shapes), and measures the
information content of the representations using mutual information neural
estimation ([Belghazi et al., 2018](https://proceedings.mlr.press/v80/belghazi18a.html))
- `mnist.ipynb` iPython notebook to isolate digit style (stroke and thickness)
after training on images grouped by digit class
- `pose_estimation/` Evaluates trained pose estimation models from
Section 4.3 of the paper in both the dictionary lookup and regression scenarios

## Python environment
The code was run successfully with Python 3.6.12.  The necessary libraries may
be installed using pip with the following line:

`pip install -r requirements.txt`

## Isolating factors of variation from Shapes3D

The Shapes3D dataset allows complete control and knowledge of factors of variation.
We use it to pin down precisely the factor isolation which results from different set supervision settings.
A large variety of experiments may be run using the script `shapes3d/train.py`.  The [Shapes3D dataset](https://github.com/deepmind/3d-shapes) will be automatically downloaded by [tensorflow_datasets](https://www.tensorflow.org/datasets/catalog/shapes3d).

The following example call (from the parent directory)

`python -m isolating_factors.shapes3d.train --inactive_vars=03 --curate_both_stacks=False`

trains a network from scratch with the wall hue and scale generative factors inactive, with only one out of each pair of training sets curated.  The second set, for every training batch, will be sampled randomly across all images (the 'One random set' variant of the experiments in Figure 4).

The command line flag `inactive_vars` takes a string of digits from 0-5, one for
each of the six generative factors of the Shapes3D dataset (wall hue, object
hue, floor hue, scale, shape, and orientation).  `01` curates stacks with
wall and object hue as inactive variables, for example.  Note that this curation
process uses `tf.data.Dataset.filter` to run through the Shapes3D dataset, which
requires searching through more of the dataset to find each training set when
there are more inactive factors of variation.

<img src="https://kieranamurphy.com/s/pca_abc.png" width=669 height=246/>

The resulting embeddings tend to be relatively low dimensional, so visualization via PCA (as above) is informative.  We also include functionality to estimate the mutual information between the learned embeddings and each of the generative factors.

Other noteworthy flags:

- `save_pngs` outputs images during training like the two above of sample embeddings and the mutual information measurements.

- `similarity_type` sets which distance metric to use when computing the loss. The best results seem to come from squared Euclidean distance (`l2sq`), but several others are implemented and there's room to explore.

- `run_augmentation_experiment` is a boolean flag which will run a double augmentation comparison if set to `True`; this is another means to isolating factors of variation with different strengths.

Training progress can be monitored with Tensorboard.

## Fast digit style isolation on MNIST

<img src="https://kieranamurphy.com/s/abc_mnist.gif" width="450"/>

The [iPython notebook](mnist.ipynb) partitions the MNIST training set into 10 different tf.data.Datasets,
with the option to withhold one digit for test time (as in the paper).
Embedding visualizations are of the two PCA dimensions with the largest variance, also as in the paper.

## Pose estimation on Pascal3D+

#### Trained models
We supply trained models for the pose estimation results of the paper at the following link:

https://storage.googleapis.com/gresearch/isolating-factors/pose_estimation.zip

The four directories in the zip file

- `car_lookup/`
- `chair_lookup/`
- `car_regression/`
- `chair_regression/`

are keras models which may be loaded with `tf.keras.models.load_model()`.
The two lookup models are from the annotation-less part of the pose estimation results, where a lookup dictionary of synthetic images was used to convert the 64-dimensional embeddings to a rotation.
These models provided the results of Table 1.

The two regression models were trained with a spherical regression head ([Liao et al. 2019](https://ivi.fnwi.uva.nl/isis/publications/2019/LiaoCVPR2019/LiaoCVPR2019.pdf)) on top of an embedding space conditioned with the CCS loss. These models provided the results of Table 2.

#### Datasets

Both were evaluated on Pascal3D+; the dataset may be downloaded from `ftp://cs.stanford.edu/cs/cvgl/PASCAL3D+_release1.1.zip`.
We have included the image names and annotations used for the test set in the paper in the files

- `pose_estimation/car_test.txt`
- `pose_estimation/chair_test.txt`

Additionally, to evaluate the embedding models, renderings from the KeypointNet paper (unseen during training) were used for the dictionary and may be downloaded as tf records through the [KeypointNet project page](https://keypointnet.github.io/).

#### Inference and evaluation

The eval script loads the images listed in the `txt` files and parses the rotation annotations.
The regression and lookup settings may be evaluated by providing filepaths to the model and data.

`python -m isolating_factors.pose_estimation.run_pose_estimation --model_dir path/to/regression.model --images_dir path/to/images --object_class car --mode regression`

and

`python -m isolating_factors.pose_estimation.run_pose_estimation --model_dir path/to/lookup/model --images_dir path/to/images --dict_dir path/to/dict/dir --object_class car --mode lookup`

will run the regression and lookup models, respectively, and print the median angular error.  Note the lookup results (Table 1 of the paper) report the mean of the median angular error over 50 random dictionaries from the ShapeNet images, so some variation is expected in the error values for the lookup scenario.
