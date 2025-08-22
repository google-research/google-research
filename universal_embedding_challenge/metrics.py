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

"""Python module to compute metrics for Universal Embedding Challenge."""


def CalibratedPrecision(predictions, retrieval_solution, k=5):
  """Compute the calibrated_precision@k for each query image and return avg.

  Args:
    predictions: Dict mapping test image ID to a list of strings corresponding
      to index image IDs.
    retrieval_solution: Dict mapping test image ID to list of ground-truth image
      IDs.
    k: denoting the rank to compute precision on, for Universal Embedding
      Challenge set to 5. For query image with n relevant images, where n<k,
      compute precision@n instead of precision@k.

  Returns:
    precision_at_k: Average precision_at_k score for query images (float).

  Raises:
    ValueError: If a test image in `predictions` is not included in
      `retrieval_solutions`, or if a test image has no relevant images.
  """
  # Compute number of test images.
  num_test_images = len(retrieval_solution.keys())

  # Loop over predictions for each query image and compute precision_at_k.
  sum_precision_at_k = 0.0
  for key, prediction in predictions.items():
    if key not in retrieval_solution:
      raise ValueError('Test image %s is not part of retrieval_solution.' % key)
    if not retrieval_solution[key]:
      raise ValueError('Test image %s has no relevant images.' % key)

    # Loop over predicted images, keeping track of those which were already
    # used (duplicates are skipped).
    already_predicted = set()
    num_expected_retrieved = min(len(retrieval_solution[key]), k)
    num_correct = 0
    for i in range(min(len(prediction), num_expected_retrieved)):
      if prediction[i] not in already_predicted:
        if prediction[i] in retrieval_solution[key]:
          num_correct += 1
        already_predicted.add(prediction[i])
    precision_at_k_per_query = num_correct / num_expected_retrieved
    sum_precision_at_k += precision_at_k_per_query

  precision_at_k = sum_precision_at_k / num_test_images

  return precision_at_k
