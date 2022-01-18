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
"""IO related utilities."""

import pandas
from tensorflow.compat.v1 import gfile


def write_dataframe_to_hdf5(df, path, complib='zlib', complevel=5, key='data'):
  """Write a DataFrame to the given path as an HDF5 file.

  Args:
    df: pandas.DataFrame to save.
    path: string path to which to save the path.
    complib: optional string giving the compression library to use.
    complevel: optional integer giving the desired level of compression.
    key: optional string name for the DataFrame in the HDF5 file.
  """
  if not isinstance(df, pandas.DataFrame):
    raise TypeError('write_dataframe_to_hdf5 input must be a DataFrame.')
  with pandas.HDFStore(
      'in_memory',
      mode='w',
      complib=complib,
      complevel=complevel,
      driver='H5FD_CORE',
      driver_core_backing_store=0) as store:
    store[key] = df
    # pylint: disable=protected-access
    buf = store._handle.get_file_image()
    with gfile.GFile(path, 'wb') as f:
      f.write(buf)


def read_dataframe_from_hdf5(path, key='data'):
  """Read a DataFrame from the given HDF5 file.

  Args:
    path: string path where the DataFrame is saved.
    key: optional string name for the DataFrame in the HDF5 file.

  Returns:
    pandas.DataFrame loaded from the HDF5 file.
  """
  with gfile.GFile(path, 'rb') as f:
    with pandas.HDFStore(
        'in_memory',
        mode='r',
        driver='H5FD_CORE',
        driver_core_backing_store=0,
        driver_core_image=f.read()) as store:
      return store[key]
