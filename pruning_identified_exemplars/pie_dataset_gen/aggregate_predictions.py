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

# Lint as: python3
r"""This script aggregates saved csv files of imagenet predictions.

Given multiple sets of predictions for the same image, it computes the modal
label at a given level of sparsity.


"""

import os
import time
from absl import app
from absl import flags
from absl import logging
import pandas as pd
import tensorflow.compat.v1 as tf

FLAGS = flags.FLAGS

flags.DEFINE_string('mode', 'train', 'Data stored from training or from eval')
flags.DEFINE_string('output_path', '', 'Directory to save the csv data to.')
flags.DEFINE_string('subdirectory',
                    'imagenet-with-bbox-training-01023-of-01024',
                    'Shard being processed.')
flags.DEFINE_string('data_directory', '', 'Pathway to data directory.')
flags.DEFINE_float('sparsity_fraction', 0.1, 'Fraction pruned.')


def aggregate_image_measures(df, output_path, subdir, n_images, sparsity):
  """Computes modal label for each image across predictions for each checkpoint.

  Args:
    df: Pathway to the directory where dataset is stored.
    output_path: Pathway where aggregated prediction measures should be saved.
    subdir: String specifying the shard of images to process.
    n_images: Number of images in the shard.
    sparsity: Indicates sparsity records to compared to baseline.
  """

  # unique directory to store an output file for every level of sparsity
  dest_dir = os.path.join(output_path, 'imagenet', 'predictions_dataframe',
                          FLAGS.mode, subdir)

  if not tf.gfile.IsDirectory(dest_dir):
    tf.gfile.MkDir(dest_dir)

  count = 0
  data = pd.DataFrame([])
  for i in range(n_images + 1):
    # store number of observations in both sparse and non-sparse sets
    variant = df[((df['fraction_pruned'] == sparsity) &
                  (df['Unnamed: 0'] == i))]
    baseline = df[((df['fraction_pruned'] == 0.0) & (df['Unnamed: 0'] == i))]
    baseline_number_observ = baseline.shape[0]
    variant_number_observ = variant.shape[0]

    # modal stored integer labels
    predicted_mode_variant = variant['predictions'].mode().max()
    predicted_mode_base = baseline['predictions'].mode().max()

    data = data.append(
        pd.DataFrame(
            {
                'pruning_fraction': sparsity,
                'image_index': count,
                'true_class_label': df['true_class'][i].max(),
                'variant_modal_label': predicted_mode_variant,
                'variant_number_observ': variant_number_observ,
                'baseline_modal_label': predicted_mode_base,
                'baseline_number_observ': baseline_number_observ,
            },
            index=[0]),
        ignore_index=True)
    count += 1

  time_ = str(time.time())
  file_name = '{}_{}_.csv'.format(sparsity, time_)
  file_path = os.path.join(dest_dir, file_name)
  with tf.gfile.Open(file_path, 'w') as f:
    data.to_csv(f)


def read_all_eval_subdir(data_directory):
  """Aggregate image level measures across saved checkpoints.

  Args:
    data_directory: Pathway to the directory where dataset is stored.

  Returns:
    pandas dataframe with all individual predictions.
  """

  filenames = tf.gfile.Glob(data_directory + '/*.csv')

  max_n_images_lst = []
  df = []
  for filename in filenames:
    with tf.gfile.Open(filename) as f:
      df_ = pd.read_csv(f)
      max_n_images_lst.append(df_['Unnamed: 0'].max())
      df.append(df_)

  df_ = pd.concat(df, ignore_index=False)

  unique_image_values = set(max_n_images_lst)
  unique_list = (list(unique_image_values))
  logging.info(max_n_images_lst)
  assert len(unique_list) == 1

  max_value = max(max_n_images_lst)

  df_.reset_index()

  return df_, max_value


def main(argv):
  del argv  # Unused.

  data_directory = os.path.join(FLAGS.data_directory, FLAGS.mode,
                                FLAGS.subdirectory)

  df, max_value = read_all_eval_subdir(data_directory=data_directory)

  for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
    aggregate_image_measures(
        df=df,
        output_path=FLAGS.output_path,
        subdir=FLAGS.subdirectory,
        n_images=max_value,
        sparsity=p)


if __name__ == '__main__':
  app.run(main)
