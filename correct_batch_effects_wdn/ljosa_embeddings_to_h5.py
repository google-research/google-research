# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

# Lint as: python2, python3
"""Convert Ljosa embeddings to h5 dataframe."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl import app
from absl import flags
import numpy as np
import pandas
from six.moves import range
from tensorflow.compat.v1 import gfile

from correct_batch_effects_wdn import io_utils

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "ljosa_data_directory", None,
    "Directory containing Ljosa metadata and embeddings as csv files.")

ORIGINAL_NUM_MEASUREMENTS = 476
NEW_NUM_MEASUREMENTS = 462


def make_map_to_moa(df_ground):
  """Generate map from compound to MOA.

  Args:
    df_ground (pandas df): contains three columns: compounds, dosage, and MOA.

  Returns:
    compound_to_moa (dict): Map from compound to MOA.
  """
  compound_to_moa = {}
  for row in np.asarray(df_ground):
    compound_to_moa[row[0]] = row[2]
  return compound_to_moa


def make_map_to_index(df_image, compound_to_moa):
  """Generate a map from table and image number to multi-index.

  Args:
    df_image (pandas dataframe): image dataframe mapping table and image number
      to other file information.
    compound_to_moa (dict): dictionary from each compound to its MOA.

  Returns:
    table_image_to_index (dict): map from table and image number to multi-index.
      The multi-index used here is:
        ["moa", "compound", "concentration", "batch", "plate", "well", "row",
         "column", "site"].
  """
  table_image_to_index = {}
  for df_row in np.asarray(df_image):
    ## site wasn't one of the argument, so read it off from file name...
    args = df_row[31].split("_")
    if len(args) == 5:
      site_str = args[3]
    elif len(args) == 6:
      site_str = args[4]
    else:
      site_str = args[1]
    ## Other arguments can be found as single entries
    site = int(site_str[1:])
    week_plate = df_row[10]
    well = df_row[11]
    ## Assumes each well has single letter (for row) followed by
    ## number (for column)
    row = well[0]
    column = int(well[1:])
    compound = df_row[37]
    concentration = df_row[38]

    ## This seems to be the convention in existing dataframes.
    if compound == "DMSO":
      treatment_group = "NEGATIVE_CONTROL"
    else:
      treatment_group = "EXPERIMENTAL"

    d = [
        treatment_group, compound_to_moa[compound], compound, concentration,
        df_row[0], week_plate, well, row, column, site
    ]
    table_image_to_index[df_row[0], df_row[1]] = d
  return table_image_to_index


def make_new_indices_df(df_object, table_image_to_index):
  """Generate pandas dataframe containing indices we would like.

  Args:
    df_object (pandas dataframe): First two indices are table and image number.
    table_image_to_index (dict): maps image and table number to desired indices.

  Returns:
    new_indices_df (pandas dataframe): dataframe containing our multi-indices.
  """
  new_indices = []
  for _, row in df_object.iterrows():
    new_indices.append(
        table_image_to_index[row["TableNumber"], row["ImageNumber"]])

  new_indices_df = pandas.DataFrame(new_indices)
  return new_indices_df


def main(argv):
  del argv  # Unused.
  loc = FLAGS.ljosa_data_directory
  image_txt_file = os.path.join(loc, "supplement_Image.txt")
  object_txt_file = os.path.join(loc, "supplement_Object.txt")
  ground_txt_file = os.path.join(loc, "supplement_GroundTruth.txt")

  image_text = gfile.Open(image_txt_file, "rb")
  object_text = gfile.Open(object_txt_file, "rb")
  ground_text = gfile.Open(ground_txt_file, "rb")

  ## Load image metadata
  df_image = pandas.read_csv(
      image_text, sep="\t", lineterminator="\n", header=None)

  ## Load embeddings... this takes a while! (1.7GB)
  df_object = pandas.read_csv(
      object_text, sep="\t", lineterminator="\n", header=None)

  ## Add appropriate column values
  df_object.columns = ["TableNumber", "ImageNumber", "ObjectNumber"
                      ] + [i for i in range(ORIGINAL_NUM_MEASUREMENTS)]

  ## indices not used for analysis.
  ## The dropped indices were generated using the table on page 15 of
  ## http://journals.sagepub.com/doi/suppl/10.1177/1087057113503553

  dropped_indices = [
      0, 1, 88, 91, 97, 100, 195, 196, 268, 271, 277, 280, 346, 347
  ]

  ## drop unused indices
  df_object = df_object.drop(columns=dropped_indices)

  ## re-name columns
  df_object.columns = ["TableNumber", "ImageNumber", "ObjectNumber"
                      ] + [i for i in range(NEW_NUM_MEASUREMENTS)]

  ## table with compounds, dosages, and MOAs
  df_ground = pandas.read_csv(
      ground_text, sep="\t", lineterminator="\n", header=None)

  ## generate dictionaries to use later
  compound_to_moa = make_map_to_moa(df_ground)
  table_image_to_index = make_map_to_index(df_image, compound_to_moa)

  ## dataframe with indices to append to embeddings
  new_indices_df = make_new_indices_df(df_object, table_image_to_index)

  ## get just the embeddings
  embeddings_df = df_object[df_object.columns[3:ORIGINAL_NUM_MEASUREMENTS + 3]]
  df_new_index = pandas.concat([new_indices_df, embeddings_df], axis=1)

  ## Creat new dataframe with appropriate index
  column_values = [
      "treatment_group", "moa", "compound", "concentration", "batch", "plate",
      "well", "row", "column", "site"
  ]
  df_new_index.columns = column_values + list(range(NEW_NUM_MEASUREMENTS))
  df_new_index = df_new_index.set_index(column_values)

  ## Write to file
  write_path = os.path.join(loc, "ljosa_embeddings_462.h5")
  io_utils.write_dataframe_to_hdf5(df_new_index, write_path)


if __name__ == "__main__":
  app.run(main)
