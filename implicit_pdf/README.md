# Implicit Representation of Probability Distributions on the Rotation Manifold

<img src="https://implicit-pdf.github.io/ipdf_files/cube.gif" width="800"/>

[Project Page](https://implicit-pdf.github.io/) |
[Paper](https://arxiv.org/abs/2106.05965)

This code accompanies the paper

**Implicit Representation of Probability Distributions on the Rotation Manifold**

Kieran Murphy*, Carlos Esteves*, Varun Jampani, Srikumar Ramalingam, Ameesh Makadia

ICML 2021 ([arxiv](https://arxiv.org/abs/2106.05965))


## Repository Contents

## Python environment
The code was run successfully with Python 3.6.12.  The necessary libraries may
be installed using pip with the following line:

`pip install -r implicit_pdf/requirements.txt`

## Training a pose estimator from scratch

The following example call

`python -m implicit_pdf.train --symsol_shapes cube --so3_sampling_mode random --number_fourier_components 4`

trains an IPDF pose estimator from scratch on the cube from the SYMSOL I dataset, with four positional encoding frequencies on the rotation query.

## Reproducing SYMSOL results 
To reproduce the SYMSOL1 results from the paper, use the following command line:

`python -m implicit_pdf.train --symsol_shapes symsol1 --so3_sampling_mode random --num_fourier_components 3 --head_network_specs [256,256,256,256] --number_training_iterations 100000 --batch_size 128 --test_batch_size 1 --number_eval_queries 2000000`

Note the resolution of the evaluation grid must be very high (more than 2 million points) to resolve the tightly-focused probability density well enough for correct evaluation.  For speed and memory considerations, you may want to evaluate with such a large grid only after training.

## Reproducing PASCAL3D+ results
The following hyperparameters were used for training on PASCAL3D+.

```python
args={
    'number_training_iterations': 150_000,
    'optimizer': 'Adam',
    'batch_size': 64,
    'number_train_queries': 2**12,
    'so3_sampling_mode': ‘grid’,
    'object_classes': ‘all’,
    'head_network_specs': [256] * 2,
    'dense_activation_head': 'relu',
    'eval_every': 10_000,
    'number_eval_queries': 2**14,
    'real_rate': 0.25,
    'learning_rate': 1e-5,
    }
```

We used an internal format for the PASCAL3D+ dataset. Synthetic data from [Su et al](https://arxiv.org/abs/1505.05641) was also used during training. You should be able to adapt the following to reproduce the results using the publicly available datasets. 

<details>
 <summary>Expand code block…</summary>

```python
def load_pascal():
  img_dims = [224, 224]
  mode = 'train'
  real_rate = 0.25
  # Object classes only supported if a single class (e.g. for eval)
  object_classes = 'all'

  all_object_classes = ['aeroplane', 'bicycle', 'boat', 'bottle',
                        'bus', 'car', 'chair', 'diningtable', 'motorbike',
                        'sofa', 'train', 'tvmonitor']

  def parse_example_real(example, context=1.166666):
    feature_description = {
        'image_buffer': tf.FixedLenFeature([], tf.string),
        'left': tf.FixedLenFeature([], tf.float32),
        'top': tf.FixedLenFeature([], tf.float32),
        'right': tf.FixedLenFeature([], tf.float32),
        'bottom': tf.FixedLenFeature([], tf.float32),
        'azimuth': tf.FixedLenFeature([], tf.float32),
        'elevation': tf.FixedLenFeature([], tf.float32),
        'theta': tf.FixedLenFeature([], tf.float32),
        'easy': tf.FixedLenFeature([], tf.int64),
        'class_name': tf.FixedLenFeature([], tf.string),
        'class_num': tf.FixedLenFeature([], tf.int64),
    }
    fd = tf.parse_single_example(
        serialized=example, features=feature_description)
    imenc = fd['image_buffer']
    im = tf.io.decode_jpeg(imenc, channels=3)

    im = tf.image.convert_image_dtype(im, tf.float32)

    easy = fd['easy']
    class_name = fd['class_name']
    class_num = tf.reshape(fd['class_num'], ())

    # crop (original data in matlab format -- assume start is 1).
    left = fd['left'] - 1.0
    top = fd['top'] - 1.0
    right = fd['right'] - 1.0
    bottom = fd['bottom'] - 1.0

    # bounding box can be invalid at two points, input or after clip.
    valid_box = left < right and top < bottom
    im_crop = im
    if valid_box:
      mid_left_right = (right + left) / 2.0
      mid_top_bottom = (bottom + top) / 2.0
      # add context
      left = mid_left_right - context * (mid_left_right - left)
      right = mid_left_right + context * (right - mid_left_right)
      top = mid_top_bottom - context * (mid_top_bottom - top)
      bottom = mid_top_bottom + context * (bottom - mid_top_bottom)
      # crop takes normalized coordinates.
      im_shape = tf.cast(tf.shape(im), tf.float32)
      y1 = tf.cast(top, tf.float32) / (im_shape[0] - 1.0)
      y2 = tf.cast(bottom, tf.float32) / (im_shape[0] - 1.0)
      x1 = tf.cast(left, tf.float32) / (im_shape[1] - 1.0)
      x2 = tf.cast(right, tf.float32) / (im_shape[1] - 1.0)
      y1 = tf.clip_by_value(y1, 0.0, 1.0)
      y2 = tf.clip_by_value(y2, 0.0, 1.0)
      x1 = tf.clip_by_value(x1, 0.0, 1.0)
      x2 = tf.clip_by_value(x2, 0.0, 1.0)
      valid_box = y1 < y2 and x1 < x2
      if valid_box:
        bbox = tf.reshape(tf.stack([y1, x1, y2, x2]), (1, 4))
        imb = tf.expand_dims(im, 0)
        im_crop = tf.image.crop_and_resize(
            image=imb, boxes=bbox, box_indices=[0], crop_size=img_dims)[0]

    # Inputs are in degrees, convert to rad.
    az = tf.reshape(fd['azimuth'], (1, 1)) * np.pi / 180.0
    el = tf.reshape(fd['elevation'], (1, 1)) * np.pi / 180.0
    th = tf.reshape(fd['theta'], (1, 1)) * np.pi / 180.0

    # R = R_z(th) * R_x(el−pi/2) * R_z(−az)
    R1 = tfg.rotation_matrix_3d.from_euler(
        tf.concat([tf.zeros_like(az), tf.zeros_like(az), -az], -1))
    R2 = tfg.rotation_matrix_3d.from_euler(
        tf.concat([el-np.pi/2.0, tf.zeros_like(el), th], -1))
    R = tf.matmul(R2, R1)
    R = tf.reshape(R, (3, 3))
    class_one_hot = tf.one_hot(class_num, len(all_object_classes))
    return im_crop, R, az, el, th, valid_box, tf.cast(easy,
                                                      tf.int32), class_one_hot

  def parse_example_synth(example):
    feature_description = {
        'image_buffer': tf.FixedLenFeature([], tf.string),
        'image_filename': tf.FixedLenFeature([], tf.string),
        'azimuth': tf.FixedLenFeature([], tf.float32),
        'elevation': tf.FixedLenFeature([], tf.float32),
        'theta': tf.FixedLenFeature([], tf.float32),
    }
    fd = tf.parse_single_example(
        serialized=example, features=feature_description)
    imenc = fd['image_buffer']
    im = tf.io.decode_png(imenc, channels=3)
    im = tf.image.convert_image_dtype(im, tf.float32)
    im = tf.image.resize(im, img_dims)

    # Inputs are in degrees, convert to rad.
    az = tf.reshape(fd['azimuth'], (1, 1)) * np.pi / 180.0
    el = tf.reshape(fd['elevation'], (1, 1)) * np.pi / 180.0
    th = tf.reshape(fd['theta'], (1, 1)) * np.pi / 180.0
    # Reversing theta for RenderForCNN data since that theta was set from filename
    # which has negative theta (see github.com/ShapeNet/RenderForCNN).
    th = -th

    # R = R_z(th) * R_x(el−pi/2) * R_z(−az)
    R1 = tfg.rotation_matrix_3d.from_euler(
        tf.concat([tf.zeros_like(az), tf.zeros_like(az), -az], -1))
    R2 = tfg.rotation_matrix_3d.from_euler(
        tf.concat([el-np.pi/2.0, tf.zeros_like(el), th], -1))
    R = tf.matmul(R2, R1)
    R = tf.reshape(R, (3, 3))

    # The merged shuffled synth [file] doesn't store the class label. The hack
    # we use is to pull it out of the image file path, which is available.
    categories = [b'aeroplane', b'bicycle', b'boat', b'bottle',
                  b'bus', b'car', b'chair', b'diningtable', b'motorbike',
                  b'sofa', b'train', b'tvmonitor']
    tokens = tf.strings.split(fd['image_filename'], '/')
    class_num = tf.compat.v1.py_func(
        categories.index, [tokens.values[0]], tf.int64, stateful=False)
    class_num = tf.reshape(tf.cast(class_num, tf.int32), ())
    class_one_hot = tf.one_hot(class_num, len(all_object_classes))
    # Create dummy entries for valid_box and easy so synth and real datasets
    # match.
    return im, R, az, el, th, True, 0, class_one_hot

  def is_easy(im, R, az, el, th, valid_box, easy, class_one_hot):
    def return_true():
      return True
    def return_false():
      return False
    return tf.cond(easy > 0, return_true, return_false)

  def is_valid_box(im, R, az, el, th, valid_box, easy, class_one_hot):
    return valid_box

  data_real = tf.data.Dataset(glob.Glob(
      'regular_expression_for_real_data_here'))
  data_real = data_real.map(lambda _, t: parse_example_real(t),
                            num_parallel_calls=4)
  data_real = data_real.filter(is_valid_box)

  if real_rate < 1. and mode == 'train':
    # We repeat the datasets here so that sample_from_datasets works as intended
    # Where if the number of elements in real vs synth is unbalanced, they stay
    # in proportion to the fraction set by real_rate
    # We don't want to repeat the validation set (where synth isn't used anyway)
    data_real = data_real.repeat()
    data_synth = tf.data.Dataset(glob.Glob(
        'regular_expression_for_synth_data_here'))
    data_synth = data_synth.map(lambda _, t: parse_example_synth(t),
                                num_parallel_calls=4)
    data_synth = data_synth.repeat()
    data_merged = tf.data.experimental.sample_from_datasets(
        (data_real, data_synth), (real_rate, 1.0 - real_rate))
  else:
    data_merged = data_real.filter(is_easy)

  # If a single object class, filter by it
  if len(object_classes) == 1 and object_classes != ['all']:

    class_ind = all_object_classes.index(object_classes[0])
    data_merged = data_merged.filter(
        lambda im, R, _a, _b, _c, _d, _e, class_one_hot: tf.reduce_all(
            class_one_hot == tf.one_hot(class_ind, len(all_object_classes))))

  # remove the extra info in the dataset
  data_merged = data_merged.map(
      lambda im, R, _a, _b, _c, _d, _e, class_one_hot: (im, R))
  return data_merged
```
</details>

## Generating an equivolumetric grid on SO(3)

Evaluation requires exact normalization of the predicted distributions.
To this end we use an equivolumetric grid covering SO(3).
Following [Yershova et al. (2010)](http://lavalle.pl/papers/YerLavMit08.pdf), we include the necessary code to produce grids of different sizes using the [HealPix](https://healpix.jpl.nasa.gov/) method as a starting point.
The [HealPy library](https://healpy.readthedocs.io/en/latest/) is required and is installed with the above `pip` call with `requirements.txt`.
The grids are automatically created as needed through the training script but may also be used in a standalone fashion with `generate_healpix_grid()` in `implicit_pdf/models.py`.

During training, a simpler mode of querying SO(3) -- sampling uniformly at random -- is also effective, even though normalization is no longer exact.
This can be set via `--so3_sampling_mode=random` in the above call, where the number of queries during training may be specified exactly.

## SYMSOL dataset
Accompanying this code release, the symmetric solid (SYMSOL) datasets introduced with the paper have been added to tensorflow_datasets.
At present, it can be accessed through [tfds-nightly](https://www.tensorflow.org/datasets/catalog/symmetric_solids).

There are 50,000 renderings each of eight shapes (five featureless from SYMSOL I and three marked shapes from SYMSOL II).
Each 224x224 RGB image is accompanied by the single ground truth rotation of the camera
during rendering, as well as the full set of equivalent ground truths under symmetry (discretized at 1 degree intervals for the cone and cylinder) for the five shapes of SYMSOL I.
![dataset](https://implicit-pdf.github.io/ipdf_files/symsol_dataset.gif)

### Citation

If you found this work or dataset useful, please cite

```
@inproceedings{implicitpdf2021,
  title = {Implicit Representation of Probability Distributions on the Rotation Manifold},
  author = {Murphy, Kieran and Esteves, Carlos and Jampani, Varun and Ramalingam, Srikumar and Makadia, Ameesh}
  booktitle = {International Conference on Machine Learning}
  year = {2021}
}
```
