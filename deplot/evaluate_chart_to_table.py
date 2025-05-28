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

"""Binary to compute table equality metrics."""

from collections.abc import Sequence, Mapping
import csv
import json
import os
import zipfile

from absl import app
from absl import flags
import tensorflow as tf

from deplot import metrics


_PATH = flags.DEFINE_string(
    'path', None, 'Directory containing tables')

_JSONL = flags.DEFINE_string(
    'jsonl', None, 'JSONL directory with predictions')


def _to_markdown(bts):
  reader = csv.reader(bts.decode().splitlines(), delimiter=',')
  parts = ['title |'] + [' | '.join(row) for row in reader]
  return '\n'.join(parts)


def _get_files(suffix):
  with zipfile.ZipFile(tf.io.gfile.GFile(
      f'{_PATH.value}_{suffix}.zip', 'rb')) as f:
    return {os.path.basename(name): f.read(name) for name in f.namelist()
            if name.endswith('.csv')}


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _PATH.value and _JSONL.value:
    raise ValueError('Only one path or value can be specified.')

  targets, predictions = [], []

  if _PATH.value:
    targets_by_id = _get_files('targets')
    predictions_by_id = _get_files('predictions')

    with tf.io.gfile.GFile(_PATH.value + '.jsonl', 'w') as f:
      for k in sorted(targets_by_id.keys()):
        target = _to_markdown(targets_by_id[k])
        prediction = _to_markdown(predictions_by_id[k])
        targets.append([target])
        predictions.append(prediction)
        line = {'input': {'id': k}, 'target': target, 'prediction': prediction}
        f.write(json.dumps(line) + '\n')
  elif _JSONL.value:
    with tf.io.gfile.GFile(_JSONL.value) as f:
      for line in f:
        example = json.loads(line)
        targets.append(example['target'])
        predictions.append(example['prediction'])
  else:
    raise ValueError('No input method specified.')

  metric = {}
  metric.update(metrics.table_datapoints_precision_recall(targets, predictions))
  metric.update(metrics.table_number_accuracy(targets, predictions))
  metric_log = json.dumps(metric, indent=2)
  print(metric_log)
  if _PATH.value:
    with tf.io.gfile.GFile(_PATH.value + '-metrics.json', 'w') as f:
      f.write(metric_log)


if __name__ == '__main__':
  app.run(main)
