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

"""Download models and associated metadata."""

import logging
import os
from typing import Optional
import urllib

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf

from protenn import utils

_logger = logging.getLogger('protenn')


_INSTALL_ENSEMBLE_FLAG = flags.DEFINE_bool(
    'install_ensemble',
    False,
    'Set to true to install an ensemble of models, not just one. '
    'More ensemble elements takes more time, but tends to be more accurate.',
)
_MODEL_CACHE_PATH_FLAG = flags.DEFINE_string(
    'model_cache_path',
    os.path.join(os.path.expanduser('~'), 'cached_models'),
    'Path to which to store downloaded models and metadata.',
)

METADATA_FILES = (
    'accession_to_description_pfam_35.json',
    'clans_pfam35.tsv',
    'nested_domains_pfam35.txt',
    'vocab_pfam35.tsv',
)


def download_models(
    model_cache_path, num_ensemble_elements = None
):
  """Downloads Pfam, EC and GO models, defaulting to downloading ensembles."""
  _logger.info('Downloading models')

  utils.fetch_oss_pretrained_models(
      model_cache_path, num_ensemble_elements=num_ensemble_elements
  )

  print('\n')  # Because the tqdm bar is position 1, we need to print a newline.


def get_metadata_files(model_cache_path):
  for filename in METADATA_FILES:
    out_path = os.path.join(model_cache_path, filename)
    in_path = os.path.join(
        utils.OSS_ZIPPED_MODELS_ROOT_URL, 'model_metadata', filename
    )
    with tf.io.gfile.GFile(out_path, 'wb') as out_file:
      with urllib.request.urlopen(in_path) as url_contents:
        out_file.write(url_contents.read())


def run(install_ensemble, model_cache_path):
  """Download and untar models and metadata."""
  if install_ensemble:
    _logger.warning('Full installation downloads and unpacks ~10GB of data; '
                    'Download time may take up to a half hour on '
                    'slow internet connections. If you are looking for '
                    'a lighter-weight installation or are a new user, we '
                    'recommend running without the flag --install_ensemble '
                    'set.')
    download_models(
        model_cache_path,
        num_ensemble_elements=utils.MAX_NUM_ENSEMBLE_ELS_FOR_INFERENCE)
  else:
    download_models(model_cache_path, num_ensemble_elements=1)

  get_metadata_files(model_cache_path)


def main(_):
  run(
      install_ensemble=_INSTALL_ENSEMBLE_FLAG.value,
      model_cache_path=_MODEL_CACHE_PATH_FLAG.value,
  )


if __name__ == '__main__':
  _logger.info('Process started.')
  app.run(main)
