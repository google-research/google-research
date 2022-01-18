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

"""Usage: `python -m r_to_pkl infile.R outfile.pkl 'in1->out1,in2->out2'`."""
# Lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pickle
import sys

import numpy as np
from rpy2 import robjects


def normalize(data):
  data = data.copy()
  for key, val in data.items():
    if np.all(np.floor(val) == val):
      val = val.astype(int)
    if np.prod(val.shape) == 1:
      val = val[0]
    data[key] = val
  return data


def translate_r_dump(filename, input_to_output_mapping):
  """Takes an R dump file and translates its contents to a Python dict.

  Args:
    filename: String giving the path to the .data.R file.
    input_to_output_mapping: string of the form
      `'jj->student_ids,kk->question_ids,y->correct'` controlling the mapping
      from input variable names to output dict keys.

  Returns:
    Python dict mapping from variable names to numpy arrays.
  """
  robjects.r['source'](filename)
  result = {}
  for in_out_pair in input_to_output_mapping.split(','):
    in_name, out_name = in_out_pair.split('->')
    result[out_name] = np.array(robjects.r[in_name])
  return normalize(result)


def main():
  _, fin, fout, input_to_output_mapping = sys.argv  # pylint: disable=unbalanced-tuple-unpacking
  with open(fout, 'wb') as f:
    pickle.dump(translate_r_dump(fin, input_to_output_mapping), f)


if __name__ == '__main__':
  main()
