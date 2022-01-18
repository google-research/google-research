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

"""Download and cache data necessary to reproduce plots."""
import os

from absl import app
from absl import flags
import requests

FLAGS = flags.FLAGS
flags.DEFINE_string('data_dir', './caltrain/data',
                    'location of the source data')


def download_file(src_url, tgt_filename):
  r = requests.get(src_url, allow_redirects=True)
  open(tgt_filename, 'wb').write(r.content)


def main(_):

  os.makedirs(FLAGS.data_dir, exist_ok=True)

  cache_file_list = [
      'beta_fit_data.p', 'calibration_results.json', 'eece_sece_data.p',
      'glm_fit_data.p'
  ]
  logit_file_list = [
      'probs_densenet161_imgnet_logits.p', 'probs_densenet40_c100_logits.p',
      'probs_densenet40_c10_logits.p', 'probs_lenet5_c100_logits.p',
      'probs_lenet5_c10_logits.p', 'probs_resnet110_SD_c100_logits.p',
      'probs_resnet110_SD_c10_logits.p', 'probs_resnet110_c100_logits.p',
      'probs_resnet110_c10_logits.p', 'probs_resnet152_SD_SVHN_logits.p',
      'probs_resnet152_imgnet_logits.p', 'probs_resnet50_birds_logits.p',
      'probs_resnet_wide32_c100_logits.p', 'probs_resnet_wide32_c10_logits.p'
  ]

  for filename in cache_file_list + logit_file_list:
    src_url = f'https://storage.googleapis.com/caltrain_data/{filename}'
    tgt_filename = os.path.join(FLAGS.data_dir, f'{filename}')
    download_file(src_url, tgt_filename)


if __name__ == '__main__':
  app.run(main)
