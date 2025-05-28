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

"""Read solution file for Universal Embedding challenge."""

import csv


def LoadSolution(file_path):
  """Reads retrieval solution from file.

  Args:
    file_path: Path to CSV file with solution. File contains a header. It should
      be formatted as '<image id>,<index ids>,<Usage>', here  '<index ids>' are
      space-separated index ids and 'Usage' is either 'Private' or 'Public'.

  Returns:
    public_solution: Dict mapping test image ID to list of ground-truth IDs of
      relevant index images for the public subset of test images.
    private_solution: Same as `public_solution`, but for the private subset of
      test images.

  Raises:
    ValueError: If 'image_id' is not formatted as a string of size 16, or if
    Usage field is not Public or Private.
  """
  public_solution = {}
  private_solution = {}

  with open(file_path, 'r') as csv_file:
    reader = csv.reader(csv_file)
    next(reader, None)  # Skip header.
    for row in reader:
      if len(row) != 3:
        raise ValueError('Parsed row length is not 3: %s.' % row)
      test_id = row[0]
      ground_truth_ids = []
      for image_id in row[1].split(' '):
        if len(image_id) != 16:
          raise ValueError('Parsed image id: %s is not a string of size 16.' %
                           image_id)
        if image_id.strip() != image_id:
          raise ValueError('Parsed image id: %s contains whitespace.' %
                           image_id)
        ground_truth_ids.append(image_id)

      if row[2] == 'Public':
        public_solution[test_id] = ground_truth_ids
      elif row[2] == 'Private':
        private_solution[test_id] = ground_truth_ids
      else:
        raise ValueError('Test image %s has unrecognized Usage tag %s' %
                         (row[0], row[2]))

  return public_solution, private_solution
