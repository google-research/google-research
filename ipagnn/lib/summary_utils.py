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

"""Summary utility library used by Learned Interpreter models."""


def _pad(text, length):
  width = _final_line_width(text)
  if width >= length:
    return text
  return text + ' ' * (length - width)


def _final_line_width(text):
  return len(text.split('\n')[-1])


def _str_width(s):
  return max(len(line) for line in s.split('\n'))


def human_readable_texts(example, predicted_outputs, info):
  """Creates human readable text about the example."""
  if info.supervised_keys[-1] == 'error_type':
    hr_key = 'error_type'
    human_readable_predicted_output = [
        info.features['error_type'].int2str(predicted_output[0])
        for predicted_output in predicted_outputs]
  else:
    hr_key = 'human_readable_target_output'
    human_readable_predicted_output = [
        info.features['target_output'].encoder.decode_to_string(
            predicted_output)
        for predicted_output in predicted_outputs]

  rows = [['Code', 'Target', 'Prediction']]
  max_lengths = [len(header) for header in rows[0]]
  for code, target_output, predicted_output in zip(
      example['human_readable_code'],
      example[hr_key],
      human_readable_predicted_output):
    if info.supervised_keys[-1] == 'error_type':
      target = info.features['error_type'].int2str(target_output)
    else:
      target = target_output.decode('utf-8')
    row = [code.decode('utf-8'),
           target,
           predicted_output]
    rows.append(row)
    max_lengths = [
        max(old_max_length, _str_width(row[index]))
        for index, old_max_length in enumerate(max_lengths)]

  lines = []
  for row in rows:
    line = ' '.join(
        _pad(row[index], max_item_length)
        for index, max_item_length in enumerate(max_lengths))
    lines.append(line)
  return '\n\n'.join(lines)
