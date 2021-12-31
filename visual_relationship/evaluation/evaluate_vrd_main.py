# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Script for evaluating 2.5D visual relationship detection."""

from typing import Sequence

from absl import app
from absl import flags

from visual_relationship.evaluation import evaluate_vrd_lib


# Assumes $PWD == "google_research/visual_relationship/evaluation".
_DEFAULT_VRD_PATH = 'data/within_image_vrd_test.csv'
_DEFAULT_OBJECT_PATH = 'data/within_image_objects_test.csv'


FLAGS = flags.FLAGS

_GROUNDTRUTH_VRD_PATH = flags.DEFINE_string(
    'groundtruth_vrd_path', _DEFAULT_VRD_PATH,
    'Path to the ground truth visual relationships.')
_GROUNDTRUTH_OBJECT_PATH = flags.DEFINE_string(
    'groundtruth_object_path', _DEFAULT_OBJECT_PATH,
    'Path to the ground truth objects.')
_PREDICTION_PATH = flags.DEFINE_string('prediction_path', None,
                                       'Path to the VRD results.')
_OUTPUT_PATH = flags.DEFINE_string('output_path', None,
                                   'Path to the evaluation result output.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  evaluator = evaluate_vrd_lib.VRDEvaluator(
      _GROUNDTRUTH_VRD_PATH.value, _GROUNDTRUTH_OBJECT_PATH.value)

  predictions = evaluate_vrd_lib.load_prediction(_PREDICTION_PATH.value)
  results = evaluator.compute_metrics(predictions)

  if _OUTPUT_PATH.value is not None:
    with open(_OUTPUT_PATH.value, 'w') as f:
      results.to_csv(f, index=False)


if __name__ == '__main__':
  flags.mark_flag_as_required('prediction_path')
  app.run(main)
