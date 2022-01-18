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
"""Loads dataset."""

import os

from absl import logging
import numpy as np

from jax_dft import scf


# pytype: disable=attribute-error


_TEST_DISTANCE_X100 = {
    'h2_plus': set([
        64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 248, 256,
        264, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464,
        480, 496, 512, 528, 544, 560, 576, 592, 608, 624, 640, 656, 672, 688,
        704, 720, 736, 752, 768, 784, 800, 816, 832, 848]),
    'h2': set([
        40, 56, 72, 88, 104, 120, 136, 152, 184, 200, 216, 232,
        248, 264, 280, 312, 328, 344, 360, 376, 392, 408, 424,
        456, 472, 488, 504, 520, 536, 568, 584, 600]),
    'h4': set([
        104, 120, 136, 152, 168, 200, 216, 232, 248, 280, 296,
        312, 344, 360, 376, 392, 408, 424, 440, 456, 472, 488,
        520, 536, 552, 568, 584, 600]),
    'h2_h2': set([
        16, 48, 80, 112, 144, 176, 208, 240, 272, 304, 336, 368,
        400, 432, 464, 496, 528, 560, 592, 624, 656, 688, 720, 752,
        784, 816, 848, 880, 912, 944, 976]),
}


class Dataset(object):
  """Loads dataset from path.

  """

  def __init__(self, path=None, data=None, num_grids=None, name=None):
    """Initializer.

    Args:
      path: String, the path to the data.
      data: Dict of numpy arrays containing
          * num_electrons, float scalar.
          * grids, float numpy array with shape (num_grids,).
          * locations, float numpy array with shape (num_samples, num_nuclei).
          * nuclear_charges, float numpy array with shape
                (num_samples, num_nuclei).
          * distances_x100, float numpy array with shape (num_samples,).
          * distances, float numpy array with shape (num_samples,).
          * total_energies, float numpy array with shape (num_samples,).
          * densities, float numpy array with shape (num_samples, num_grids).
          * external_potentials, float numpy array with shape
                (num_samples, num_grids).
      num_grids: Integer, specify the number of grids for the density and
          external potential. If None, the original grids size are used.
          Otherwise, the original grids are trimmed into num_grids grid points
          in the center.
      name: String, the name of the dataset.

    Raises:
      ValueError: if path and data are both None.
    """
    if path is None and data is None:
      raise ValueError('path and data cannot both be None.')
    self.name = name
    # Load from path if data is not given in the input argument.
    if data is None and path is not None:
      data = self._load_from_path(path)
    for name, array in data.items():
      setattr(self, name, array)
    self._set_num_grids(num_grids)
    self.total_num_samples = self.distances.shape[0]

  def _load_from_path(self, path):
    """Loads npy files from path."""
    file_open = open
    data = {}
    with file_open(os.path.join(path, 'num_electrons.npy'), 'rb') as f:
      # make sure num_electrons is scalar not np.array(scalar)
      data['num_electrons'] = int(np.load(f))
    with file_open(os.path.join(path, 'grids.npy'), 'rb') as f:
      data['grids'] = np.load(f)
    with file_open(os.path.join(path, 'locations.npy'), 'rb') as f:
      data['locations'] = np.load(f)
    with file_open(os.path.join(path, 'nuclear_charges.npy'), 'rb') as f:
      data['nuclear_charges'] = np.load(f)
    with file_open(os.path.join(path, 'distances_x100.npy'), 'rb') as f:
      data['distances_x100'] = np.load(f).astype(int)
    with file_open(os.path.join(path, 'distances.npy'), 'rb') as f:
      data['distances'] = np.load(f)
    with file_open(os.path.join(path, 'total_energies.npy'), 'rb') as f:
      data['total_energies'] = np.load(f)
    with file_open(os.path.join(path, 'densities.npy'), 'rb') as f:
      data['densities'] = np.load(f)
    with file_open(os.path.join(path, 'external_potentials.npy'), 'rb') as f:
      data['external_potentials'] = np.load(f)
    return data

  def _set_num_grids(self, num_grids):
    """Sets number of grids and trim arrays with grids dimension."""
    # grids is 1d array.
    original_num_grids = len(self.grids)
    if num_grids is None:
      self.num_grids = original_num_grids
      logging.info('This dataset has %d grids.', self.num_grids)
    else:
      if num_grids > original_num_grids:
        raise ValueError(
            'num_grids (%d) cannot be '
            'greater than the original number of grids (%d).'
            % (num_grids, original_num_grids))
      self.num_grids = num_grids
      diff = original_num_grids - num_grids
      if diff % 2:
        left_grids_removed = (diff - 1) // 2
        right_grids_removed = (diff + 1) // 2
      else:
        left_grids_removed = diff // 2
        right_grids_removed = diff // 2
      self.grids = self.grids[
          left_grids_removed: original_num_grids - right_grids_removed]
      self.densities = self.densities[
          :, left_grids_removed: original_num_grids - right_grids_removed]
      self.external_potentials = self.external_potentials[
          :, left_grids_removed: original_num_grids - right_grids_removed]
      logging.info(
          'The original number of grids (%d) are trimmed into %d grids.',
          original_num_grids, self.num_grids)

  def get_mask(self, selected_distance_x100=None):
    """Gets mask from distance_x100."""
    if selected_distance_x100 is None:
      mask = np.ones(self.total_num_samples, dtype=bool)
    else:
      selected_distance_x100 = set(selected_distance_x100)
      mask = np.array([
          distance in selected_distance_x100
          for distance in self.distances_x100])
      if len(selected_distance_x100) != np.sum(mask):
        raise ValueError(
            'selected_distance_x100 contains distance that is not in the '
            'dataset.')
    return mask

  def get_test_mask(self):
    """Gets mask for test set."""
    return self.get_mask(_TEST_DISTANCE_X100[self.name])

  def get_subdataset(self, selected_distance_x100=None, downsample_step=None):
    """Gets subdataset."""
    mask = self.get_mask(selected_distance_x100)
    if downsample_step is not None:
      sample_mask = np.zeros(self.total_num_samples, dtype=bool)
      sample_mask[::downsample_step] = True
      mask = np.logical_and(mask, sample_mask)
    return Dataset(
        data={
            'num_electrons': self.num_electrons,
            'grids': self.grids,
            'locations': self.locations[mask],
            'nuclear_charges': self.nuclear_charges[mask],
            'distances_x100': self.distances_x100[mask],
            'distances': self.distances[mask],
            'total_energies': self.total_energies[mask],
            'densities': self.densities[mask],
            'external_potentials': self.external_potentials[mask],
        })

  def get_molecules(self, selected_distance_x100=None):
    """Selects molecules from list of integers."""
    mask = self.get_mask(selected_distance_x100)
    num_samples = np.sum(mask)

    return scf.KohnShamState(
        density=self.densities[mask],
        total_energy=self.total_energies[mask],
        locations=self.locations[mask],
        nuclear_charges=self.nuclear_charges[mask],
        external_potential=self.external_potentials[mask],
        grids=np.tile(
            np.expand_dims(self.grids, axis=0), reps=(num_samples, 1)),
        num_electrons=np.repeat(self.num_electrons, repeats=num_samples),
        converged=np.repeat(True, repeats=num_samples),
        )
