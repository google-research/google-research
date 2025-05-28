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

"""Functions to generate augmentations of input data and load data for model training."""


from modules.masking_utils import gen_masked_inputs
from modules.masking_utils import gen_random_mask
import numpy as np
from scipy import sparse
from tensorflow import keras

#  AUGMENTATIONS


def aug_random_masking(
    maskrate,
    mask_len,
    mask_vals,
    include_demos=True,
    include_padding=False,
    inmask="all",
    return_mask=False,
):
  """Creates augmentation function to mask input data with MCAR missingness.

  Args:
    maskrate:  Rate of missingness to induce.
    mask_len:  Expected length of time series sequence
    mask_vals:  list of mask value for times, values, and features.
    include_demos:  Whether the augmentation function should return
      [demographics, times, values, features] or just [times, values, features].
    include_padding:  Whether padding should also be masked.
    inmask:  Which elements of time series (times, values, features) to mask.
    return_mask:  Whether to return binary mask in addition to the masked data.

  Returns:
    get_augmentation: A function which takes a batch of input data and
      returns a masked version. This function is called by the data generator
      for each new batch during training.
  """

  def get_augmentation(x):
    """Augmentation function to be used during training."""
    batch_size = len(x[0])
    mask = gen_random_mask((batch_size, mask_len), maskrate)
    masked_x = gen_masked_inputs(
        x[1:],
        mask,
        masking_values=mask_vals,
        include_padding=include_padding,
        inmask=inmask,
    )

    if include_demos:
      to_return = [x[0]] + list(masked_x)
    else:
      to_return = list(masked_x)

    if return_mask:
      return to_return, mask
    else:
      return to_return

  return get_augmentation


def aug_geometric_masking(
    v,
    global_mask,
    mask_vals,
    include_demos=True,
    include_padding=False,
    inmask="all",
    raw_times=True,
    time_mean=None,
    time_stddev=None,
    return_mask=False,
):
  """Creates augmentation function to mask input data with geometric missingness.

  Geometric masking was proposed by Zerveas et al for regularly sampled data:
  https://github.com/gzerveas/mvts_transformer/tree/master
  We adjust the method to first sample masks at regular time intervals, and
  then apply the time-based mask to our irregularly sampled data (masking all
  events in the masking window for a given feature). Because re-computing masks
  at each iteration is slow, we save a large set of pre-sampled masks and sample
  rows among the saved mask for each feature within each time series.

  Args:
    v: Number of features.
    global_mask: Precomputed array of geometric masks to sample from.
    mask_vals:  List of mask value for times, values, and features.
    include_demos:  Whether the augmentation function should return
      [demographics, times, values, features] or just [times, values, features].
    include_padding:  Whether padding should also be masked.
    inmask:  Which elements of time series (times, values, features) to mask.
    raw_times: Whether the times in the input time series data are raw values or
      have been normalized.
    time_mean: Mean time value (which may have been used for normalization).
    time_stddev: Standard deviation of time value in training data (which may
      have been used for normalization).
    return_mask:  Whether to return binary mask in addition to the masked data

  Returns:
    get_augmentation: a function which takes a batch of input data and
      returns a masked version. This function is called by the data generator
      for each new batch during training.
  """

  def map_reg_mask_to_triplets(full_sequences):
    """Randomly samples a geometric mask for each feature and then maps the mask to times of events."""
    final_mask = []

    # Sample V masks for each time series (one per feature)
    reg_mask_idxs = np.random.randint(
        0, len(global_mask), size=(len(full_sequences[0]), v)
    )

    # Map sampled masks to time series
    for i in range(len(full_sequences[0])):
      t, _, f = [x[i] for x in full_sequences]

      # if times are normalized, need to un-normalize before mapping
      if not raw_times:
        t = t * time_stddev + time_mean

      regular_mask = global_mask[reg_mask_idxs[i]].T
      tmp_mask = []
      for j, elt in enumerate(f):
        if elt == 0:
          tmp_mask.append(True)
        # catch any observations not in the [time window) range
        elif t[j] >= len(regular_mask):
          tmp_mask.append(False)
        else:
          tmp_mask.append(regular_mask[max(int(t[j]), 0), elt - 1])

      final_mask.append(tmp_mask)
    return np.array(final_mask).astype(int)

  def aug(x):
    """Augmentation function to be used during training."""
    mask = map_reg_mask_to_triplets(x[1:])
    masked_x = gen_masked_inputs(
        x[1:],
        mask,
        masking_values=mask_vals,
        include_padding=include_padding,
        inmask=inmask,
    )

    if include_demos:
      to_return = [x[0]] + list(masked_x)
    else:
      to_return = list(masked_x)

    if return_mask:
      return to_return, mask
    else:
      return to_return

  return aug


# Jitter continuous values with gaussian noise:
def aug_jitter(noise_stddev, include_demos=True, include_padding=False):
  """Creates augmentation function to add Gaussian noise to inputs.

  Args:
    noise_stddev:  Standard deviation for sampling Gaussian noise.
    include_demos:  Whether the augmentation function should return
      [demographics, times, values, features] or just [times, values, features].
    include_padding:  Whether padding should also be masked.

  Returns:
    get_augmentation: a function which takes a batch of input data and
      adds Gaussian noise to continuous elements of the time series.
  """

  def aug(x):
    """Augmentation function to be used during training."""
    jittered = [np.zeros(x[1].shape), np.zeros(x[2].shape), x[3]]
    for i in range(len(x[0])):
      t, v, f = [x[i] for x in x[1:]]
      if include_padding:
        t_noise = np.random.normal(loc=0, scale=noise_stddev, size=f.shape)
        v_noise = np.random.normal(loc=0, scale=noise_stddev, size=f.shape)
      else:
        t_noise = (f != 0) * np.random.normal(
            loc=0, scale=noise_stddev, size=f.shape
        )
        v_noise = (f != 0) * np.random.normal(
            loc=0, scale=noise_stddev, size=f.shape
        )

      jittered[0][i] = t + t_noise
      jittered[1][i] = v + v_noise
    if include_demos:
      return [x[0]] + jittered
    else:
      return jittered

  return aug


def final_aug_generator(augs_list):
  """Function to chain a list of augmentations.

  Args:
    augs_list:  List of augmentations to apply one after the other.

  Returns:
    aug: A single function which applies all augmentations in succession
      which can be used for data generator.
  """

  def aug(x):
    auged = x
    for fun in augs_list:
      auged = fun(auged)
    return auged

  return aug


#  DATA GENERATOR


class DataGenerator(keras.utils.Sequence):
  """Defines data generator - wrapper for data for training models.

  The DataGenerator generates batches of data and applies an augmentation.
  We also specify what training labels to provide (different between pretraining
  and finetuning).
  """

  def __init__(
      self,
      input_data,
      aug_function,
      forecast=False,
      forecast_out=None,
      reconstruct=False,
      reconstruct_out=None,
      batch_size=32,
      shuffle=True,
  ):
    # Initialization
    self.input_data = input_data
    self.num_samples = input_data[0].shape[0]
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.aug_function = aug_function

    if forecast:
      self.forecast_y = forecast_out
      self.forecast = True
    else:
      self.forecast = False

    if reconstruct:
      self.reconstruct_y = reconstruct_out
      self.reconstruct = True
    else:
      self.reconstruct = False

    self.on_epoch_end()

    print(len(self.input_data))
    print(self.num_samples)
    print(self.batch_size)

  def __len__(self):
    """Returns the number of batches per epoch."""
    return int(np.floor(self.num_samples / self.batch_size))

  def __getitem__(self, index):
    """Generate one batch of data."""

    # Generate indexes of the batch
    indexes = self.indexes[
        index * self.batch_size : (index + 1) * self.batch_size
    ]

    # Generate 2 views of batch input data
    raw_batch_x = [x[indexes] for x in self.input_data]
    x1 = self.aug_function(raw_batch_x)

    if not self.forecast and not self.reconstruct:
      outputs = None
    else:
      outputs = []
      if self.forecast:
        if isinstance(self.forecast_y, sparse.csr.csr_matrix):
          outputs.append(self.forecast_y[indexes].todense())
        else:
          outputs.append(self.forecast_y[indexes])
      if self.reconstruct:
        if isinstance(self.reconstruct_y, list):
          for o in self.reconstruct_y:
            outputs.append(o[indexes])
        else:
          outputs.append(self.reconstruct_y[indexes])
    return x1, outputs

  def on_epoch_end(self):
    self.indexes = np.arange(self.num_samples)
    if self.shuffle:
      np.random.shuffle(self.indexes)
