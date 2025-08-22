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

"""Downloads OGB datasets, adapted from the original OGB library."""

import os
import shutil
import tempfile

from absl import app
from absl import flags

import ogb
import ogb.utils.url
import pandas as pd
import requests

_DATASET_NAME = flags.DEFINE_string(
    'dataset_name', 'ogbn-arxiv', help='Name of the dataset to load.')
_DATASET_ROOT = flags.DEFINE_string(
    'dataset_root', 'datasets/', help='Root directory for all datasets.')

_OGB_URL = 'https://raw.githubusercontent.com/snap-stanford/ogb/master/ogb/nodeproppred/master.csv'
_REDDIT_URL = 'https://drive.google.com/corp/drive/folders/1rq-H0XUM0BIRW9Pq5P4FMC9Xirpdx6zs'


def download_ogb_dataset(name, root):
  """Downloads the OGB dataset."""

  original_root = root
  dir_name = '_'.join(name.split('-'))
  root = os.path.join(original_root, dir_name)

  tempdir = tempfile.mkdtemp()
  master_file_path = os.path.join(tempdir, 'master.csv')

  response = requests.get(_OGB_URL)
  with open(master_file_path, 'wb') as f:
    f.write(response.content)

  master = pd.read_csv(master_file_path, index_col=0)
  if name not in master:
    raise ValueError(f'Invalid dataset name {name}.\n'
                     'Available datasets are as follows:\n'
                     '\n'.join(master.keys()))
  meta_info = master[name]
  download_name = meta_info['download_name']
  url = meta_info['url']

  # Delete the existing output folder.
  try:
    shutil.rmtree(root)
  except FileNotFoundError:
    pass

  if ogb.utils.url.decide_download(url):
    path = ogb.utils.url.download_url(url, original_root)
    ogb.utils.url.extract_zip(path, original_root)
    os.unlink(path)
    shutil.move(os.path.join(original_root, download_name), root)


def download_reddit_dataset(root):
  """Downloads the Reddit dataset."""

  gdown_supported = True

  if not gdown_supported:
    raise ValueError(
        'We do not support automatically downloading the reddit dataset internally. '
        f'Please download the dataset from Google Drive at: {_REDDIT_URL}')

  output = os.path.join(root, 'reddit')

  import gdown  # pylint: disable=g-import-not-at-top
  gdown.download_folder(
      _REDDIT_URL, output=output, quiet=False, use_cookies=False)


def main(unused_argv):
  del unused_argv  # Unused.

  name = _DATASET_NAME.value
  root = _DATASET_ROOT.value

  if name.startswith('ogb'):
    download_ogb_dataset(name, root)

  elif name.startswith('reddit'):
    download_reddit_dataset(root)

  else:
    raise ValueError(f'Unsupported dataset: {name}')


if __name__ == '__main__':
  app.run(main)
