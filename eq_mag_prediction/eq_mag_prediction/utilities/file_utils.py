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

"""Utility functions related to file handling."""

import os
import xarray as xr


def chain_files_iterator(dirname):
  """An iterator that iterates over all lines in all files in the directory."""
  files = os.listdir(dirname)
  for f in files:
    with open(os.path.join(dirname, f), 'rt') as buffer:
      for line in buffer:
        yield line.rstrip('\n')


def load_xr_dataset(file_path):
  """Helper function that loads a serialized xr.Dataset from a netcdf file.

  Args:
    file_path: Path (string) to the file.

  Returns:
    The loaded xarray.Dataset.
  """
  with open(file_path, 'rb') as f:
    return xr.open_dataset(f.read())


def save_xr_dataset(file_path, dset):
  """Helper function that serializes an xr.Dataset to a netcdf file.

  Args:
    file_path: Path to the file.
    dset: The dset to be serialized.
  """
  with open(file_path, 'wb') as f:
    f.write(dset.to_netcdf())
