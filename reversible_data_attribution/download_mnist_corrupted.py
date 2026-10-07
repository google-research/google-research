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

"""Downloads Corrupted MNIST dataset from TFDS and saves it in npz format compatible with counterfactual estimation."""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
import numpy as np
import tensorflow_datasets as tfds
import tqdm

FLAGS = flags.FLAGS

_CORRUPTIONS = [
    'identity',
    'shot_noise',
    'impulse_noise',
    'defocus_blur',
    'glass_blur',
    'motion_blur',
    'zoom_blur',
    'snow',
    'frost',
    'fog',
    'brightness',
    'contrast',
    'elastic_transform',
    'pixelate',
    'rotate',
    'jpeg_compression',
]

flags.DEFINE_string(
    'output_filename',
    'mnist_corrupted_local.npz',
    'Filename to save the corrupted MNIST dataset.',
)
flags.DEFINE_enum(
    'corruption',
    'shot_noise',
    _CORRUPTIONS,
    'Type of corruption to download from TFDS mnist_corrupted catalog.',
)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  corruption_type = FLAGS.corruption
  dataset_name = f'mnist_corrupted/{corruption_type}'

  with tqdm.tqdm(total=3, desc='Downloading Corrupted MNIST') as pbar:
    pbar.set_description(f'Loading {dataset_name} from TFDS')
    ds_train, ds_test = tfds.load(
        dataset_name,
        split=['train', 'test'],
        batch_size=-1,
        as_supervised=True,
    )
    pbar.update(1)

    pbar.set_description('Converting tensors to numpy arrays')
    x_train, y_train = tfds.as_numpy(ds_train)
    x_test, y_test = tfds.as_numpy(ds_test)

    # Squeeze channel dimension if shape is (N, 28, 28, 1) -> (N, 28, 28)
    if x_train.ndim == 4 and x_train.shape[-1] == 1:
      x_train = np.squeeze(x_train, axis=-1)
    if x_test.ndim == 4 and x_test.shape[-1] == 1:
      x_test = np.squeeze(x_test, axis=-1)

    y_train = np.squeeze(y_train).astype(np.int64)
    y_test = np.squeeze(y_test).astype(np.int64)
    pbar.update(1)

    # Resolve output path
    local_filename = FLAGS.output_filename
    working_dir = os.environ.get('BUILD_WORKING_DIRECTORY', '')
    if working_dir:
      output_path = os.path.join(working_dir, local_filename)
    else:
      output_path = local_filename

    pbar.set_description(f'Compressing and saving to {output_path}')
    np.savez_compressed(
        output_path,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )
    pbar.update(1)

  print(f'Corrupted MNIST ({corruption_type}) successfully saved.')
  print(f'x_train shape: {x_train.shape}, y_train shape: {y_train.shape}')
  print(f'x_test shape: {x_test.shape}, y_test shape: {y_test.shape}')
  print(f"Dataset saved locally to '{output_path}'")


if __name__ == '__main__':
  app.run(main)
