# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""Downloads datasets via keras.datasets and saves them in npz format."""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
import keras
import numpy as np
import tqdm

FLAGS = flags.FLAGS

_SUPPORTED_DATASETS = [
    'mnist',
    'cifar',
    'cifar10',
    'cifar100',
    'fashion_mnist',
    'imdb',
    'boston_housing',
    'california_housing',
    'reuters',
]

flags.DEFINE_enum(
    'dataset',
    'mnist',
    _SUPPORTED_DATASETS,
    'Dataset to download (everything that can be downloaded from'
    ' keras.datasets).',
)

flags.DEFINE_string(
    'dataset_dir',
    None,
    'Directory to save the dataset. Defaults to'
    ' third_party/google_research/google_research/reversible_data_attribution/{dataset}.',
)

flags.DEFINE_string(
    'output_filename', None, 'Filename to save the Keras dataset.'
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  dataset_name = FLAGS.dataset
  keras_dataset_name = 'cifar10' if dataset_name == 'cifar' else dataset_name
  print(f'Downloading {dataset_name} dataset via keras.datasets...')

  dataset_module = getattr(keras.datasets, keras_dataset_name)
  (x_train, y_train), (x_test, y_test) = dataset_module.load_data()

  # Flatten labels from (N, 1) to (N,) to ensure consistent 1D label format
  # across datasets.
  if y_train.ndim > 1 and y_train.shape[-1] == 1:
    y_train = np.squeeze(y_train, axis=-1)
  if y_test.ndim > 1 and y_test.shape[-1] == 1:
    y_test = np.squeeze(y_test, axis=-1)

  if np.issubdtype(y_train.dtype, np.integer):
    y_train = y_train.astype(np.int64)
  if np.issubdtype(y_test.dtype, np.integer):
    y_test = y_test.astype(np.int64)

  print(f'{dataset_name} dataset downloaded.')
  print(
      f'x_train shape: {x_train.shape}, y_train shape: {y_train.shape},'
      f' x_test shape: {x_test.shape}, y_test shape: {y_test.shape}'
  )

  # Resolve output filename
  local_filename = FLAGS.output_filename
  if local_filename is None:
    if dataset_name in ('cifar', 'cifar10'):
      local_filename = 'cifar_local.npz'
    else:
      local_filename = f'{dataset_name}_local.npz'
  if not local_filename.endswith('.npz'):
    local_filename += '.npz'

  # Resolve output directory
  workspace_dir = os.environ.get('BUILD_WORKSPACE_DIRECTORY', '')
  working_dir = os.environ.get('BUILD_WORKING_DIRECTORY', '')

  if FLAGS.dataset_dir:
    dataset_dir = FLAGS.dataset_dir
  else:
    # Default to {folder_name} within the package directory.
    folder_name = (
        'cifar' if dataset_name in ('cifar', 'cifar10') else dataset_name
    )
    relative_pkg_dir = os.path.join(
        'third_party/google_research/google_research/reversible_data_attribution',
        folder_name,
    )
    if workspace_dir:
      dataset_dir = os.path.join(workspace_dir, relative_pkg_dir)
    elif working_dir:
      if working_dir.endswith(
          'third_party/google_research/google_research/reversible_data_attribution'
      ):
        dataset_dir = os.path.join(working_dir, folder_name)
      else:
        dataset_dir = os.path.join(working_dir, relative_pkg_dir)
    else:
      dataset_dir = relative_pkg_dir

  # If a relative directory was provided, anchor it to workspace/working dir
  if not os.path.isabs(dataset_dir):
    base_dir = workspace_dir or working_dir
    if base_dir:
      dataset_dir = os.path.join(base_dir, dataset_dir)

  output_path = os.path.join(dataset_dir, local_filename)

  # Ensure parent directory exists
  out_dir = os.path.dirname(output_path)
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)

  with tqdm.tqdm(total=1, desc=f'Saving dataset to {output_path}') as pbar:
    np.savez_compressed(
        output_path,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    pbar.update(1)

  print(f"Dataset successfully saved locally to '{output_path}'")


if __name__ == '__main__':
  app.run(main)
