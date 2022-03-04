# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for working with dsprites and 3dident datasets.
"""

import datetime
import functools
import os

from absl import flags

import numpy as np
import tensorflow.compat.v2 as tf


tf.compat.v1.enable_v2_behavior()


FLAGS = flags.FLAGS

DSPRITES_SHAPE_NAMES = ['square', 'ellipse', 'heart']

DSPRITES_LABEL_NAMES = [
    'label_scale', 'label_orientation', 'label_x_position', 'label_y_position'
]

DSPRITES_VALUE_NAMES = [
    'value_scale', 'value_orientation', 'value_x_position', 'value_y_position'
]

### general useful functions for pandas and tensorflow


def tf_train_eval_split(ds, num_examples, eval_split=0.0):
  """Splits tensorflow dataset into train/eval datasets.

  Since we can't sample from tf datasets, the eval set is just the first n
  examples so ensure the dataset is shuffled beforehand!

  Args:
    ds: Tensorflow dataset to split.
    num_examples: Number of examples in dataset (required because there's no
      easy way of counting the size of the dataset without iterating through).
    eval_split: Float in range (0, 1), fraction of dataset to split out as eval
      set.

  Returns:
    Tuple (train dataset, num train examples, eval dataset, num eval examples).

  """
  num_eval_examples = int(num_examples * eval_split)
  num_train_examples = num_examples - num_eval_examples
  ds_eval = ds.take(num_eval_examples)
  ds_train = ds.skip(num_eval_examples)
  return ds_train, num_train_examples, ds_eval, num_eval_examples


def pd_train_eval_split(df, eval_split=0.0, seed=None, reset_index=False):
  """Splits pandas dataframe into train/eval sets.

  Uses pandas DataFrame.sample to create the split.

  Args:
    df: Dataframe of examples to be split.
    eval_split: Float in range (0, 1), fraction of dataset to split out as eval
      set.
    seed: Optional, for reproducibility of eval split.
    reset_index: If False (default) then retains original df indexing in
      returned train/eval dataframes, if True then resets index of both.

  Returns:
    Tuple (train dataset, num train examples, eval dataset, num eval examples).

    Sizes of returned dataframes aren't necessary but included to match the
    corresponding tensorflow function.
  """
  eval_df = df.sample(frac=eval_split, random_state=seed, replace=False)
  eval_idx = eval_df.index
  train_df = df.drop(index=eval_idx)
  num_eval_examples, num_train_examples = len(eval_df), len(train_df)
  if reset_index:
    train_df.reset_index(drop=True, inplace=True)
    eval_df.reset_index(drop=True, inplace=True)
  return train_df, num_train_examples, eval_df, num_eval_examples


def make_backup(file_path, overwrite=False):
  """Makes a backup copy of a file in a 'backups' subfolder.

  Use for e.g. pandas dataframes of datasets where accidentally modifying/losing
  the dataframe later would be really really annoying.

  Args:
    file_path: Path (str) to where file is located.
    overwrite: If True, overwrites any existing backup. If False (default)
      creates new file name with current date/time rather than overwrite an
      existing backup.

  Returns:
    Path of backup file (str).
  """
  file_name = os.path.basename(file_path)
  backup_dir = os.path.join(os.path.dirname(file_path), 'backups')
  tf.io.gfile.makedirs(backup_dir)
  backup_path = os.path.join(backup_dir, file_name)
  try:
    tf.io.gfile.copy(file_path, backup_path, overwrite=overwrite)
  except tf.errors.OpError:
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    # insert current_time right before the file extension
    new_backup_path = backup_path.split('.')
    new_backup_path = '.'.join(new_backup_path[:-1] +
                               [current_time, new_backup_path[-1]])
    tf.io.gfile.copy(file_path, new_backup_path)
    backup_path = new_backup_path
  return backup_path


def df_to_ds(df, preprocess_fn):
  dataset = tf.data.Dataset.from_tensor_slices(dict(df))
  ds = dataset.map(preprocess_fn)
  return ds


def get_image(img_path, num_channels=None):
  # set expand_animations to false to ensure image has a shape attribute;
  # otherwise causes problems later with tf.image.resize.
  img = tf.image.decode_image(
      tf.io.read_file(img_path), expand_animations=False, channels=num_channels)
  return img


def get_image_by_id(df, idx, num_channels=None):
  return get_image(df.loc[idx, 'img_path'], num_channels=num_channels)


def latent_lookup_map(df, latents):
  """Finds indices of examples in dataframe whose latents match those specified.

  Args:
    df: The dataframe in which to search for latents.
    latents: Dict of latent values to match on. Ignores any keys that aren't
      present in dataframe columns.

  Returns:
    List of indices of all examples in dataframe which match the given latents.

  """
  keys = [i for i in latents.keys() if i in df.columns]

  # use np.isclose rather than equality test because we're often dealing with
  # floats.
  check_conditions = np.all([np.isclose(df[k], latents[k]) for k in keys],
                            axis=0)
  return df[check_conditions].index.values


### dsprites-specific functions


def preprocess_dsprites_images(x, img_size=None, num_channels=None):
  """Fetches dsprite image and preprocess.

  Args:
    x: Pandas Series, dict, etc containing 'img_path' key whose value points to
      the image in question.
    img_size: (optional) 2-tuple to resize image to. Use None (default) for no
      resizing.
    num_channels: (optional) How many channels the resulting image will have.

  Returns:
    Tensorflow tensor of image, with values scaled to range [0, 1].
  """
  img = get_image(x['img_path'], num_channels=num_channels)
  if img_size is not None:
    img = tf.image.resize(img, img_size)
  img = tf.cast(img, tf.float32)
  img = tf.clip_by_value(img, 0., 1.)
  return img


def preprocess_dsprites_latents(x):
  """Convertes dsprites latents into a standard format for our dataset.

  Args:
    x: Row from dsprites dataframe containing all latents for example.

  Returns:
    Tuple of tensors (labels, values). Both contain one-hot encoding of the
    example's shape.  Orientation value is scaled by 1/2pi.

  """
  shapes = [float(x[i]) for i in DSPRITES_SHAPE_NAMES]
  labels = shapes + [float(x[i]) for i in DSPRITES_LABEL_NAMES]
  values = shapes + [float(x[i]) for i in DSPRITES_VALUE_NAMES]
  values = values * tf.constant([1, 1, 1, 1, 1 / (2 * np.pi), 1, 1])
  return tf.convert_to_tensor(labels), tf.convert_to_tensor(values)


def preprocess_dsprites(x, img_size=None, num_channels=None):
  """All dsprites preprocessing functions in one function for use in map fns.

  Args:
    x: Row from pandas dataframe.
    img_size: 2-tuple of desired image size, or None for no resizing.
    num_channels: Desired number of channels in image, or None for the default
      determined by tf.image.decode_image.

  Returns:
    Dict with 'image' and 'values' keys, containing the input image and latent
    values.
  """
  image = preprocess_dsprites_images(x, img_size, num_channels)
  labels, values = preprocess_dsprites_latents(x)
  return {'image': image, 'labels': labels, 'values': values}


def get_preprocess_dsprites_fn(img_size=None, num_channels=None):
  return functools.partial(
      preprocess_dsprites,
      img_size=img_size,
      num_channels=num_channels)


##### 3dident-specific functions


THREEDIDENT_VALUE_NAMES = [
    'pos_x', 'pos_y', 'pos_z', 'rot_phi', 'rot_theta', 'rot_psi', 'spotlight',
    'hue_object', 'hue_spotlight', 'hue_background'
]


def preprocess_threedident_images(x, img_size=None, num_channels=3):
  """Fetches 3DIdent image and preprocess.

  Args:
    x: Pandas Series, dict, etc containing 'img_path' key whose value contains
      the path to the image in question.
    img_size: (optional) 2-tuple to resize image to. Use None (default) for no
      resizing.
    num_channels: How many channels the resulting image will have. (Default 3)

  Returns:
    Tensorflow tensor of image, values in range [0, 1].
  """
  img = get_image(
      x['img_path'], num_channels=num_channels)
  img = tf.cast(img, tf.float32)
  img = img / 255.0
  if img_size is not None:
    img = tf.image.resize(img, img_size)
  img = tf.clip_by_value(img, 0., 1.)
  return img


def preprocess_threedident_values(x):
  values = [float(x[i]) for i in THREEDIDENT_VALUE_NAMES]
  return tf.convert_to_tensor(values)


def preprocess_threedident(x, img_size, num_channels):
  """All 3DIdent preprocessing functions in one convenience function.

  Args:
    x: Row from pandas dataframe.
    img_size: 2-tuple of desired image size, or None for no resizing.
    num_channels: Desired number of channels in image.

  Returns:
    Dict with 'image' and 'values' keys, containing the input image and latent
    values.
  """
  image = preprocess_threedident_images(x, img_size, num_channels)
  values = preprocess_threedident_values(x)
  return {'image': image, 'values': values}


def get_preprocess_threedident_fn(img_size, num_channels):
  return functools.partial(
      preprocess_threedident,
      img_size=img_size,
      num_channels=num_channels)


### functions for contrastive example generation


def get_contrastive_example_idx(z, df, sample_fn, deterministic=False):
  """Finds a suitable contrastive example z_prime conditioned on z.

  Used for generating contrastive pairs from a dataframe of examples, where we
  modify the example at the level of the latent values and are constrained to
  return another example from the given dataframe.

  Args:
    z: The example to condition on.
    df: The dataframe of examples to sample the conditional z_prime from.
    sample_fn: The function used to generate z_prime given z.
    deterministic: If True, always returns first result; if False, returns a
      random one (default False).

  Returns:
    Index of new example z_prime in dataframe df.
  """
  zprime_latents = sample_fn(z, df)
  if not isinstance(zprime_latents, dict):
    raise TypeError(
        'Contrastive sample function should return a dict of latents.')
  if deterministic:
    idx = latent_lookup_map(df, zprime_latents)[0]
  else:
    idx = np.random.choice(latent_lookup_map(df, zprime_latents))
  return int(idx)


def dsprites_simple_noise_fn(z, df=None):
  """Applies random noise to dsprites example for generating contrastive pair.

  Given latents for a dsprites example z, generates the latent labels for
  z_prime | z by randomly adding +1/-1 to some subset of scale, orientation,
  x_pos, y_pos. Orientation is computed mod 40 (the max label value) to ensure
  it is treated as circular.

  Args:
    z: Pandas Series containing latents for a dsprite example. Must
      contain the label and shape latents, value latents are optional.
    df: Dataframe, not used here.

  Returns:
    zprime_latents: Dict containing shape and label latents for zprime.
      Guaranteed to be different to z, guaranteed to leave shape latent
      unchanged.

  """
  del df  # not used

  features = DSPRITES_LABEL_NAMES
  shapes = DSPRITES_SHAPE_NAMES
  max_values = np.array([6, 40, 32, 32])

  zprime = z[features].to_numpy()
  while np.array_equal(z[features], zprime):
    zprime += np.random.randint(-1, 2, size=4)
    zprime[1] = zprime[1] % max_values[1]
    zprime = np.minimum(np.maximum(zprime, np.zeros(4)), max_values - 1)

  zprime = np.concatenate((z[shapes], zprime)).astype('int32')
  all_features = shapes + features
  zprime_latents = {all_features[i]: zprime[i] for i in range(7)}
  return zprime_latents


def threedident_simple_noise_fn(z, df, tol=1.0, mult=1.1, deterministic=False):
  """Finds example that is a small perturbation away from given example z.

  Given an example z, finds a nearby example z_prime subject to the condition
  that z_prime exists in the dataset. This is achieved by considering a ball of
  radius tol around each latent and sampling a (non-identity) example from the
  intersection.

  Args:
    z: Dataframe row of the example to condition on.
    df: Dataframe of all available examples.
    tol: Starting radius of balls around each latent (Default 1.0).
    mult: Float > 1, multiplier to scale ball radius by, used when intersection
      of balls is empty.
    deterministic: If True, always returns first result; if False, returns a
      random one. (default False).

  Returns:
    Dict containing latent values of new example z_prime (guaranteed to be
    different from input example).

  """
  latents = THREEDIDENT_VALUE_NAMES
  # first three latents have twice the range of the others
  scaling = np.array([1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
  tol_array = scaling * tol
  while True:
    # for each latent, get a ball of examples around that value
    balls = [
        df[np.abs(df[latents[i]] - z[latents[i]]) < tol_array[i]]
        for i in range(len(latents))
    ]
    # drop the original z
    new_idx = set(balls[0].drop(index=[z.name]).index)
    for b in balls:
      new_idx = new_idx.intersection(b.index)
    if new_idx:
      if deterministic:
        result_id = list(new_idx)[0]
      else:
        result_id = np.random.choice(list(new_idx))
      result = df.loc[result_id]
      return {k: result[k] for k in latents}
    else:
      # slightly increase the size of the balls and try again
      tol_array *= mult
