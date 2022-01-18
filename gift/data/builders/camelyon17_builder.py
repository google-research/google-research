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

r"""TFDS databuilder for camelyon17_v1.0."""

import os
import tensorflow as tf
import tensorflow_datasets as tfds


class Camelyon17(tfds.core.BeamBasedBuilder):
  """TFDS builder for camelyon17_v1.0.

  This dataset is part of the WILDS benchmark.
  """

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  You must manually download and extract camelyon17_v1.0 data from
  () and place them in `manual_dir`.
  """

  VERSION = tfds.core.Version('1.0.0')
  _DOMAINS = ['0', '1', '2', '3', '4']

  def _info(self):

    return tfds.core.DatasetInfo(
        builder=self,
        description=('camelyon17:'),
        features=tfds.features.FeaturesDict({
            'image': tfds.features.Image(),
            'label': tfds.features.ClassLabel(num_classes=2),
        }),
        supervised_keys=('image', 'label'),
        homepage='',
        citation=r"""@article{bandi2018detection,
          title={From detection of individual metastases to classification of lymph node status at the patient level: the camelyon17 challenge},
          author={Bandi, Peter and Geessink, Oscar and Manson, Quirine and Van Dijk, Marcory and Balkenhol, Maschenka and Hermsen, Meyke and Bejnordi, Babak Ehteshami and Lee, Byungjae and Paeng, Kyunghyun and Zhong, Aoxiao and others},
          journal={IEEE transactions on medical imaging},
          volume={38},
          number={2},
          pages={550--560},
          year={2018},
          publisher={IEEE}
                }""",
    )

  def _split_generators(self, dl_manager):
    """Download data and define the splits."""

    image_dirs = {}
    meta_dirs = {}
    for domain in self._DOMAINS:
      image_dirs[domain] = os.path.join(dl_manager.manual_dir, 'patches')
      meta_dirs[domain] = os.path.join(dl_manager.manual_dir, 'metadata.csv')

    splits = []
    for domain in self._DOMAINS:
      gen_kwargs = {
          'data_dir': image_dirs[domain],
          'meta_dir': meta_dirs[domain],
          'domain': domain,
      }
      splits.append(
          tfds.core.SplitGenerator(name=f'{domain}', gen_kwargs=gen_kwargs))

    return splits

  def _build_pcollection(self, pipeline, data_dir, meta_dir, domain):
    """Generate examples as dicts."""
    beam = tfds.core.lazy_imports.apache_beam

    def _process_example(example_description):
      (idx, patient, node, x_coord, y_coord, label, unused_slide, unused_center,
       unused_split) = example_description.split(',')
      image_path = (f'patient_{patient}_node_{node}/'
                    f'patch_patient_{patient}_node_{node}'
                    f'_x_{x_coord}_y_{y_coord}.png')
      return idx, {'image': os.path.join(data_dir, image_path), 'label': label}

    with tf.io.gfile.GFile(meta_dir) as meta_file:
      examples_descriptions = meta_file.readlines()

    return pipeline | beam.Create(examples_descriptions[1:]) | beam.Filter(
        lambda x: x.split(',')[-2] == domain) | beam.Map(_process_example)
