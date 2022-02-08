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

"""Generate and load dsprites and 3dident datasets.

Creates dataframes of dsprites and 3dident datasets so that we can sample from
and search them, and handles the train/eval/test splitting and conversion to
tf.data.Dataset format in a reproducible way.

Also generates and loads datasets for contrastive learning experiments, where
we need to sample large datasets of similar pairs of images based on their
latents.
"""

import os

from absl import flags
from absl import logging

import numpy as np
import pandas as pd

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from graph_compression.contrastive_learning import data_utils

FLAGS = flags.FLAGS


### functions for loading existing datasets


def get_standard_dataset(name,
                         dataframe_path=None,
                         img_size=None,
                         num_channels=None,
                         eval_split=None,
                         seed=None,
                         reset_index=True):
  """Loads dsprites/3dident in pandas dataframe and tensorflow dataset formats.

  Note that 3dident already has a train/test split by default, so specifying
  eval_split will split off a third dataframe/dataset out of the train set.
  Similarly if a non-default original dataframe is used, with a 'split'
  column containing values 'train' or 'test' for each entry, this will produce
  the same effect.

  Args:
    name: 'dsprites' or 'threedident'.
    dataframe_path: Str, location of saved csv containing latent values and
      paths to saved images for each example. For dsprites this can be created
      using build_dsprites_dataframe.
    img_size: 2-tuple, optional, size to reshape images to during preprocessing.
    num_channels: Int, optional, specify number of channels in processed images.
    eval_split: Float in [0,1], optional, splits out this fraction of train
      dataset into a separate dataframe+dataset.
    seed: Int, optional, specify random seed for reproducibility of eval set.
    reset_index: Bool, default True, whether to reindex the train and eval
      dataframes after splitting.

  Returns:
    Dict of tuples (pd dataframe, tf dataset, number of examples) with keys
    'train', 'test' (if train/test split exists in original dataframe) and
    'eval' (if eval_split is not None).

  """
  if name == 'dsprites':
    preprocess_fn = data_utils.get_preprocess_dsprites_fn(
        img_size, num_channels)
    path = dataframe_path


  elif name == 'threedident':
    preprocess_fn = data_utils.get_preprocess_threedident_fn(
        img_size, num_channels)
    path = dataframe_path


  else:
    raise ValueError(
        f'Dataset name must be one of "dsprites" or "threedident", you provided {name}'
    )

  with tf.io.gfile.GFile(path, 'rb') as f:
    df = pd.read_csv(f)

  datasets = {}
  if 'split' in df.columns:
    test_df, df = df[df.split == 'test'].copy(), df[df.split == 'train'].copy()
    num_test_examples = len(test_df)
    test_ds = data_utils.df_to_ds(test_df, preprocess_fn)
    datasets.update({'test': (test_df, test_ds, num_test_examples)})

  num_examples = len(df)
  if eval_split is None:
    ds = data_utils.df_to_ds(df, preprocess_fn)
    datasets.update({'train': (df, ds, num_examples)})
  else:
    train_df, num_train_examples, eval_df, num_eval_examples = data_utils.pd_train_eval_split(
        df, eval_split, seed, reset_index)
    train_ds = data_utils.df_to_ds(train_df, preprocess_fn)
    eval_ds = data_utils.df_to_ds(eval_df, preprocess_fn)
    datasets.update({
        'train': (train_df, train_ds, num_train_examples),
        'eval': (eval_df, eval_ds, num_eval_examples)
    })
  return datasets


def get_contrastive_dataset(name,
                            dataframe_path=None,
                            img_size=None,
                            num_channels=None):
  """Loads existing contrastive dataset.

  Args:
    name: 'dsprites' or 'threedident'.
    dataframe_path: Str, optional; location of specific dataframe to load. If
      not specified, the default is used.
    img_size: 2-tuple, optional; specify image resizing during preprocessing
      step.
    num_channels: Int, optional; specify number of channels for processed
      images.

  Returns:
    Tuple of form (pandas dataframe, tf dataset, number of examples).

  """
  if name == 'dsprites':
    preprocess_fn = data_utils.get_preprocess_dsprites_fn(
        img_size, num_channels)
    path = dataframe_path

  elif name == 'threedident':
    preprocess_fn = data_utils.get_preprocess_threedident_fn(
        img_size, num_channels)
    path = dataframe_path


  else:
    raise ValueError(
        f'Dataset name must be one of "dsprites" or "threedident", you provided {name}'
    )

  with tf.io.gfile.GFile(path, 'rb') as f:
    # header=[0,1] handles the multi-index; remove this if build_dataset changes
    contrastive_df = pd.read_csv(f, header=[0, 1])

  z_ds = data_utils.df_to_ds(contrastive_df['z'], preprocess_fn)
  zprime_ds = data_utils.df_to_ds(contrastive_df['zprime'], preprocess_fn)

  contrastive_ds = tf.data.Dataset.zip({'z': z_ds, 'zprime': zprime_ds})

  return contrastive_df, contrastive_ds, len(contrastive_df)


### functions to recreate a dataset from scratch


def build_dsprites_dataframe(target_path):
  """Recreates the dsprites dataframe from base tfds version.

  Each image is converted to png and written to the 'images' subfolder of the
  specified target_path.

  The dataframe contains the latent values and labels of each example, a one-hot
  encoding of its shape, and the path to the corresponding image.

  Args:
    target_path: Str, path to where the dataframe and images should be saved.

  Returns:
    Location where dataframe was saved.
  """

  tfds_dataset, tfds_info = tfds.load(
      'dsprites', split='train', with_info=True, shuffle_files=False)
  num_examples = tfds_info.splits['train'].num_examples

  # list the features we care about
  feature_keys = list(tfds_info.features.keys())
  feature_keys.remove('image')
  feature_keys.remove('value_shape')
  feature_keys.remove('label_shape')
  shapes = ['square', 'ellipse', 'heart']

  # helper function to modify how the data is stored in the tf dataset before
  # we convert it to a pandas dataframe
  def pandas_setup(x):
    # encoding the image as a png byte string turns out to be a convenient way
    # of temporarily storing the images until we can write them to disk.
    img = tf.io.encode_png(x['image'])
    latents = {k: x[k] for k in feature_keys}
    latents.update(
        {k: int(x['label_shape'] == i) for i, k in enumerate(shapes)})
    latents['png'] = img
    return latents

  temp_ds = tfds_dataset.map(pandas_setup)
  dsprites_df = tfds.as_dataframe(temp_ds)
  dsprites_df = dsprites_df[shapes + feature_keys + ['png']]  # reorder columns

  # setup for saving the pngs to disk
  if os.path.basename(target_path).endswith('.csv'):
    dataset_dir = os.path.dirname(target_path)
    dataframe_location = target_path
  else:
    dataset_dir = target_path
    dataframe_location = os.path.join(target_path, 'dsprites_df.csv')

  images_path = os.path.join(dataset_dir, 'images')
  tf.io.gfile.makedirs(images_path)  # creates any missing parent directories

  padding = len(str(num_examples))
  temp_index = pd.Series(range(num_examples))

  def create_image_paths(x):
    path_to_file = os.path.join(images_path, str(x).zfill(padding) + '.png')
    return path_to_file

  # create a col in the dataframe for the image file path
  dsprites_df['img_path'] = temp_index.apply(create_image_paths)

  # iterate through the dataframe and save each image to specified folder
  for i, x in dsprites_df.iterrows():
    img = tf.io.decode_image(x['png'])
    with tf.io.gfile.GFile(x['img_path'], 'wb') as f:
      tf.keras.preprocessing.image.save_img(f, img.numpy(), file_format='PNG')
    if i % 100 == 0:
      logging.info('%s of %s images processed', i + 1, num_examples)

  dsprites_df.drop(columns=['png'], inplace=True)
  logging.info('finished processing images')

  logging.info('conversion complete, saving...')
  with tf.io.gfile.GFile(dataframe_location, 'wb') as f:
    dsprites_df.to_csv(f, index=False)

  # also make a copy so if you screw up the original df you don't have to run
  # the entire generation process again
  _ = data_utils.make_backup(dataframe_location)

  return dataframe_location


def build_contrastive_dataframe(df,
                                save_location,
                                num_samples,
                                sample_fn,
                                seed=None):
  """Builds a contrastive dataframe from scratch.

  Given a universe of examples and a rule for how to choose z_prime conditioned
  on z, generates a dataframe consisting of num_sample positive pairs for use in
  contrastive training.

  Args:
    df: The dataframe of available examples.
    save_location: Str, where to save the new contrastive dataframe.
    num_samples: int, how many pairs to generate.
    sample_fn: Function that specifies how to choose z_prime given z.
    seed: Int, optional; use if the dataframe construction needs to be
      reproducible.

  Returns:
    Location of new contrastive dataframe.
  """
  z_df = df.sample(n=num_samples, random_state=seed, replace=True)

  if seed is not None:
    np.random.seed(seed)

  tf.io.gfile.makedirs(os.path.split(save_location)[0])

  zprime_index = []
  counter = 0
  for _, z in z_df.iterrows():
    z_prime = data_utils.get_contrastive_example_idx(z, df, sample_fn)
    zprime_index.append(z_prime)
    # need a separate counter because iterrows keys off the index which is not
    # sequential here
    counter += 1
    if counter % 100 == 0:
      logging.info('%s of %s examples generated', counter + 1, num_samples)

  zprime_df = df.loc[zprime_index].reset_index(drop=True)
  z_df.reset_index(drop=True, inplace=True)

  contrastive_df = pd.concat([z_df, zprime_df], axis=1, keys=['z', 'zprime'])

  with tf.io.gfile.GFile(save_location, 'wb') as f:
    contrastive_df.to_csv(f, index=False)

  # also make a copy so if you screw up the original df you don't have to run
  # the entire generation process again
  _ = data_utils.make_backup(save_location)

  logging.info('dataframe saved at %s', save_location)

  return save_location
