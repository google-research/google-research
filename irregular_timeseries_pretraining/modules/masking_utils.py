# coding=utf-8
# Copyright 2025 The Google Research Authors.
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

"""Helper functions for sampling masks (for input data augmentations)."""
import copy
import os
import numpy as np


def gen_random_mask(shape, mask_rate):
  rand_nums = np.random.random(shape)
  return (rand_nums > mask_rate).astype(int)


def gen_masked_inputs(
    full_sequences,
    input_mask,
    masking_values=(0, 0, 0),
    include_padding=False,
    inmask="all",
):
  """full_sequences: list of times, features, values (each one num_samples x max_len).

  Args:
    full_sequences:  Input data.
    input_mask:  Mask to apply to samples. Should be in the shape num_samples
    x max_len (1 for keep, 0 for mask)
    masking_values:  mask value to use for times, values, and features.
    include_padding:  Whether padding should also be masked.
    inmask:  Which elements of time series (times, values, features) to mask.
  Returns:
    data: Masked data.
  """
  data = copy.deepcopy(full_sequences)

  if include_padding:
    tokeep = input_mask
  else:
    # set mask=1 for all padding (where features==0) so that they don't get
    # replaced with the mask token
    tokeep = input_mask | (data[2] == 0).astype(int)

  # check which type (times, features, values) are to be masked and only
  # update those values
  if inmask == "all":
    types_to_mask = [0, 1, 2]
  else:
    types_to_mask = []
    for i, val in enumerate(["t", "v", "f"]):
      if val in inmask:
        types_to_mask.append(i)

  for row_i, elt in enumerate(tokeep):
    for type_i in types_to_mask:
      data[type_i][row_i, np.where(elt == 0)] = masking_values[type_i]

  return data


def gen_fullreconstruction_labels(y_true, include_padding=False, outpred="all"):
  """Generating labels for reconstruction task.

  Args:
    y_true: True [times, features, values] list for the given samples.
    include_padding:  whether or not we will penalize the model for incorrect
    predictions on padding.
    outpred: Which elements among times, features, and values we are trying to
    reconstruct (masks will be set to 0 if not reconstructing that element)
  Returns:
    final_data: List of [times,features,values] arrays - each of shape
    num_samples x max_len*2 where the first columns [:max_len] refer to the
    actual true unmasked values, and then the second half of columns [max_len:]
    indicate whether the location should be used in loss calculation (1=yes).
  """

  # Generate mask of the form: 1=keep,0=ignore
  # If ignoring padding, we mask out all the locations where there are trailing
  # 0s in the true variables sequence
  if include_padding:
    mask = np.ones(y_true[2].shape)
  else:
    mask = (y_true[2] > 0).astype(int)

  if outpred == "all":
    return [np.hstack([x, mask]) for i, x in enumerate(y_true)]
  else:
    final_data = []
    for i, val in enumerate(["t", "v", "f"]):
      if val in outpred:
        final_data.append(np.hstack([y_true[i], mask]))
    if len(final_data) == 1:
      return final_data[0]
    else:
      return final_data


def load_geometric_masks(geo_masks_dir, mask_rate):
  """Function to load pre-computed geometric masks.

  Args:
    geo_masks_dir: path to saved masks
    mask_rate: masking rate for geometric masks
  Returns:
    global_reg_mask: list of masks that were saved
    lm: lm parameter for geometric masking
  """

  if mask_rate in [0.3, 0.5]:
    lm = 3
  elif mask_rate in [0.8]:
    lm = 5

  saved_val_mask_path = os.path.join(
      geo_masks_dir, "mask_{}_{}_geometric.txt".format(mask_rate, lm)
  )

  global_reg_mask = np.loadtxt(saved_val_mask_path, dtype="int")
  print("loaded geometric masks from:", saved_val_mask_path)

  return global_reg_mask, lm

