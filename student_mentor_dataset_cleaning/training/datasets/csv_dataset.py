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

"""A class to load an image dataset from a CSV file."""

import os

import pandas as pd
from PIL import Image
import tensorflow as tf


class CsvDataset:
  """Reads a list of photos from a CSV file and gives random access to them."""

  def __init__(self,
               base_dir,
               csv,
               image_id_col = 0,
               label_col = 2):
    """Initializes the CsvDataset class.

    Images are expected to be found in `base_dir`/x/y/z/id.jpg, where x, y and z
    are the first three characters of the image is and id is the image id found
    in `csv`.

    Args:
      base_dir: Directory containing the dataset images.
      csv: Either the path of the CSV file containing the dataset or a Pandas
        dataframe containing the CSV data. The CSV must contain columns for
        image id and label, as specified in the following two argument.
      image_id_col: The 0-based index of the column containing the image id.
      label_col: The 0-based index of the column containing the image label.
    """

    self.base_dir = base_dir
    if isinstance(csv, str):
      self.csv_df = pd.read_csv(csv)
      self.csv_df = self.csv_df.sort_values(
          by=self.csv_df.columns[label_col]).head(1996)
    else:
      self.csv_df = csv
    self.image_id_col = image_id_col
    self.label_col = label_col

  def get_image(self, image_id):
    """Returns the image of the given `image_id` as a PIL.image."""
    if not isinstance(image_id, str):
      image_id = f'{image_id:016d}'
    image_path = os.path.join(self.base_dir, image_id[0], image_id[1],
                              image_id[2], image_id + '.jpg')
    return Image.open(image_path)

  def split(self, new_split_ratio):
    """Splits the dataset into two parts according to `new_split_ratio`.

    This function modifies this dataset to contain the first split and returns
    the second split.

    Args:
      new_split_ratio: The fraction of images to sample for the first split. The
        remaining images will be returned as the second split.

    Returns:
      The second split of the dataset.
    """

    first_split = self.csv_df.sample(frac=(1 - new_split_ratio))
    second_split = self.csv_df.drop(first_split.index)

    self.csv_df = first_split
    return CsvDataset(self.base_dir, second_split)

  def __len__(self):
    return self.csv_df.shape[0]

  def get_image_dataset(self, preprocess_fn=None, image_size = 321):
    """Creates a dataset of triplets of easy positives and hard negatives.

    Args:
      preprocess_fn: A preprocessing function to be run on each image.
      image_size: The size that loaded images will be resized to. Images will be
        square with dimensions `image_size` x `image_size`.

    Returns:
      The loaded dataset as a tf.data.Dataset.
    """

    label_mapping = {}
    labels = []
    label_list = list(self.csv_df.to_numpy()[:, self.label_col])
    for label in label_list:
      if label not in label_mapping:
        label_mapping[label] = len(label_mapping)
      labels.append(label_mapping[label])

    ds = tf.keras.preprocessing.image_dataset_from_directory(
        self.base_dir,
        labels=labels,
        label_mode='int',
        color_mode='rgb',
        batch_size=1,
        image_size=(image_size, image_size),
        shuffle=False,
        interpolation='bilinear').unbatch()
    if preprocess_fn is not None:
      ds = ds.map(lambda a, b: (preprocess_fn(a), b))

    return ds
