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

# Lint as: python3
"""Pre-processing step for Ljosa embeddings.

The goal is to replicate the work done in the Ljosa paper
(https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3884769/), within our framework so
we can try the WDN method on them (i.e. the Wasserstein Distance Network method
for batch correction).

There are two steps: The first is to normalize the components between the 1st
and 99th percentile of the controls. The second step is factor analysis with
50 dimesions.

The input is an H5 dataframe with 462 embedding dimensions. The output should
be two embeddings dataframes, after each step of the analysis. The first
dataframe has 456 components, since some dimensions are dropped (when the
normalization is not successful). The post-factor analysis embedding has
50 dimensions because this is the number of factors used.

The methods here could also be used as post-processing after e.g. the WDN
method.
"""

from absl import app
from absl import flags

import numpy as np
import pandas as pd

from correct_batch_effects_wdn import io_utils
from correct_batch_effects_wdn import metadata
from correct_batch_effects_wdn import transform

FLAGS = flags.FLAGS

flags.DEFINE_string("original_df", None,
                    "Original Ljosa embeddings, with unused variables taken "
                    "out.")

flags.DEFINE_string("post_normliazation_path", None,
                    "Where to save post-normalization dataframe.")

flags.DEFINE_string("post_fa_path", None,
                    "Where to save post-factor analysis dataframe.")


def normalize_ljosa(df):
  """Apply the normalization for each dimension of the embedding.

  Find the 1st and 99th percentile of the controls for each dimension, for each
  plate. Then normalize the remaining treatments so that they have matching
  1st and 99th percentiles.

  Args:
    df (pandas dataframe): Input embedding dataframe
  Returns:
    emb_df_normalized (pandas dataframe): Normalized dataframe.
  """
  emb_df_normalized = []
  for (_, _), df_batch_plate in df.groupby(level=[metadata.BATCH,
                                                  metadata.PLATE]):
    df_batch_plate_new = df_batch_plate.copy()

    ## Find 1st and 99th percentile of controls (for each plate)
    df_batch_plate_control = df_batch_plate.xs("DMSO", level=metadata.MOA)
    percentile_1st = np.percentile(df_batch_plate_control, q=1, axis=0)
    percentile_99th = np.percentile(df_batch_plate_control, q=99, axis=0)

    ## normalize remainingd treatments
    slope = 1 / (percentile_99th - percentile_1st)
    intercept = - slope * percentile_1st
    for i, col in enumerate(df_batch_plate.columns):
      df_batch_plate_new[col] = df_batch_plate[col] * slope[i] + intercept[i]
    emb_df_normalized.append(df_batch_plate_new)
  emb_df_normalized = pd.concat(emb_df_normalized)

  ## Drop columns with NaN entries.
  col_to_drop = np.where(emb_df_normalized.isnull().any())[0]
  emb_df_normalized.drop(col_to_drop, axis=1, inplace=True)
  return emb_df_normalized


def main(argv):
  del argv  # Unused.
  df = io_utils.read_dataframe_from_hdf5(FLAGS.original_df)

  ## Generate the percentile-normalized embeddings
  emb_df_normalized = normalize_ljosa(df)

  ## Generate the post-factor analysis embeddings
  np.random.seed(0)
  emb_df_fa = transform.factor_analysis(emb_df_normalized, 0.15, 50)

  io_utils.write_dataframe_to_hdf5(emb_df_fa, FLAGS.post_fa_path)
  io_utils.write_dataframe_to_hdf5(emb_df_normalized,
                                   FLAGS.post_normliazation_path)


if __name__ == "__main__":
  app.run(main)
