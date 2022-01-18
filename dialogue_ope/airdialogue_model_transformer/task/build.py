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

import parlai.core.build_data as build_data
import os
from parlai.core.build_data import DownloadableFile

RESOURCES = [
    DownloadableFile(
        'https://storage.googleapis.com/airdialogue/airdialogue_data.tar.gz',
        'airdialogue.tar.gz',
        'cfce57816f1df881633b3e819ec5c95533e2c4b19064e81d5bdefe207cbab50a',
    )
]


def build(opt):
  dpath = os.path.join(opt['datapath'])
  airdialogue_path = os.path.join(dpath, 'airdialogue_data')
  version = '1.1'

  if not build_data.built(airdialogue_path, version_string=version):
    print('[building data: ' + airdialogue_path + ']')
    if build_data.built(airdialogue_path):
      build_data.remove_dir(airdialogue_path)

    # Download the data.
    for downloadable_file in RESOURCES:
      downloadable_file.download_file(dpath)

    build_data.mark_done(airdialogue_path, version_string=version)
