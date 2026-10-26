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

"""Computes bigram scores from synthetic data."""

from collections.abc import Sequence
import os

from absl import app
from absl import flags
from absl import logging
import grain
import grain.python as pygrain
import numpy as np
from privacy_auditing import bigram_audit_lib
from privacy_auditing import filesystem
from privacy_auditing import prepare_data

_CANARY_PATH = flags.DEFINE_string(
    'canary_path',
    None,
    'Base path to the canary dataset (without the .seen/.unseen.arrayrecord'
    ' suffix).',
    required=True,
)
_CANARY_NAME = flags.DEFINE_string(
    'canary_name',
    None,
    'Short name for the canary dataset, used as a prefix for output files.',
    required=True,
)
_CANARY_LEN = flags.DEFINE_integer(
    'canary_len',
    60,
    'The number of tokens to score in the canary.',
)
_SYNTHETIC_DATA_PATH = flags.DEFINE_string(
    'synthetic_data_path',
    None,
    'Path to the synthetic data bagz file.',
    required=True,
)
_OUTPUT_DIR = flags.DEFINE_string(
    'output_dir',
    None,
    'The directory to save the scores to.',
    required=True,
)
_CANARIES_PER_SET = flags.DEFINE_integer(
    'canaries_per_set',
    3000,
    'The number of canaries to score per set.',
)


def _extract_text(ex):
  return (
      pygrain.fast_proto.parse_tf_example_experimental(ex)[
          'BODY_NORMALIZED_LIST'
      ]
      .item()
      .decode('utf8')
  )


def _score_canaries(
    bigram_model, seen
):
  """Scores the canaries using the bigram model and saves the scores."""
  seen_str = 'seen' if seen else 'unseen'
  logging.info('Loading %s canaries', seen_str)
  canary_ds = prepare_data.load_canaries(_CANARY_PATH.value, seen=seen)
  if len(canary_ds) < _CANARIES_PER_SET.value:
    raise ValueError(f'Not enough canaries at {_CANARY_PATH.value} to score.')
  canary_ds = grain.DataLoader(
      data_source=canary_ds,
      sampler=grain.samplers.SequentialSampler(_CANARIES_PER_SET.value),
  )
  logging.info('Scoring %s canaries', seen_str)
  scores = [bigram_model.score(_extract_text(ex)) for ex in canary_ds]
  output_file = f'{_CANARY_NAME.value}_{seen_str}_bigram_scores.npy'
  filesystem.makedirs(_OUTPUT_DIR.value)
  output_path = os.path.join(_OUTPUT_DIR.value, output_file)
  logging.info('Saving %s scores to %s', seen_str, output_file)
  with filesystem.open_file(output_path, 'wb') as f:
    np.save(f, scores, allow_pickle=False)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  bigram_model = bigram_audit_lib.BigramModel(
      _SYNTHETIC_DATA_PATH.value, _CANARY_LEN.value
  )
  _score_canaries(bigram_model, seen=True)
  _score_canaries(bigram_model, seen=False)


if __name__ == '__main__':
  app.run(main)
