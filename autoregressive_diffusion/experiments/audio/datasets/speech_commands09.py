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

"""SpeechCommands dataset.

Extended label space with zero-nine digits.
"""

import os
import numpy as np

import tensorflow_datasets.public_api as tfds

_CITATION = """
@article{speechcommandsv2,
   author = {{Warden}, P.},
    title = "{Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition}",
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  eprint = {1804.03209},
  primaryClass = "cs.CL",
  keywords = {Computer Science - Computation and Language, Computer Science - Human-Computer Interaction},
    year = 2018,
    month = apr,
    url = {https://arxiv.org/abs/1804.03209},
}
"""

_DESCRIPTION = """
An audio dataset of spoken words designed to help train and evaluate keyword
spotting systems. Its primary goal is to provide a way to build and test small
models that detect when a single word is spoken, from a set of ten target words,
with as few false positives as possible from background noise or unrelated
speech.
"""

_DOWNLOAD_PATH = 'http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz'
_TEST_DOWNLOAD_PATH_ = 'http://download.tensorflow.org/data/speech_commands_test_set_v0.02.tar.gz'

_SPLITS = ['train', 'valid', 'test']

NUMBERS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven',
           'eight', 'nine']
SAMPLE_RATE = 16000


class SpeechCommands09(tfds.core.GeneratorBasedBuilder):
  """The Speech Commands 0-9 dataset for keyword detection."""

  VERSION = tfds.core.Version('0.0.2')

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            'audio':
                tfds.features.Audio(file_format='wav', sample_rate=SAMPLE_RATE),
            'label':
                tfds.features.ClassLabel(names=NUMBERS)
        }),
        supervised_keys=('audio', 'label'),
        # Homepage of the dataset for documentation
        homepage='https://arxiv.org/abs/1804.03209',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    """Returns SplitGenerators."""

    dl_path = dl_manager.download(_DOWNLOAD_PATH)

    train_paths, validation_paths, test_paths = self._split_archive(
        dl_manager.iter_archive(dl_path))

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                'archive': dl_manager.iter_archive(dl_path),
                'file_list': train_paths
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                'archive': dl_manager.iter_archive(dl_path),
                'file_list': validation_paths
            },
        ),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                'archive': dl_manager.iter_archive(dl_path),
                'file_list': test_paths,
                'split_name': 'test'
            },
        ),
    ]

  def _generate_examples(self, archive, file_list, split_name=None):
    """Yields examples."""
    for path, file_obj in archive:
      if file_list is not None and path not in file_list:
        continue
      relpath, wavname = os.path.split(path)
      if split_name == 'test':
        print(path)
      _, word = os.path.split(relpath)
      example_id = '{}_{}'.format(word, wavname)
      if word in NUMBERS:
        label = word
      else:
        # Unlike the original Speech Commands dataset, we throw away all audio
        # samples that are not numbers. We also do not store the silences.
        continue

      try:
        example = {
            'audio':
                np.array(
                    tfds.core.lazy_imports_lib.lazy_imports.pydub.AudioSegment
                    .from_file(file_obj,
                               format='wav').get_array_of_samples()),
            'label':
                label,
        }
        yield example_id, example
      except tfds.core.lazy_imports_lib.lazy_imports.pydub.exceptions.CouldntDecodeError:
        pass

  def _split_archive(self, train_archive):
    train_paths = []
    for path, file_obj in train_archive:
      if 'testing_list.txt' in path:
        test_paths = file_obj.read().strip().splitlines()
        test_paths = [p.decode('ascii') for p in test_paths]
      elif 'validation_list.txt' in path:
        validation_paths = file_obj.read().strip().splitlines()
        validation_paths = [p.decode('ascii') for p in validation_paths]
      elif path.endswith('.wav'):
        train_paths.append(path)

    # The paths for the train set is just whichever paths that do not exist in
    # either the test or validation splits.
    train_paths = (set(train_paths) - set(validation_paths) - set(test_paths))

    return train_paths, validation_paths, test_paths
