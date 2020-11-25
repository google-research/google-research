# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Write summaries in joblib files similar to tf.summary which is very slow."""
import collections
import os

import joblib


class SummaryWriter(object):
  """Save data in joblib format for easy processing."""

  def __init__(self, log_dir, available_norm_types):
    """Create a new SummaryWriter.

    Args:
      log_dir: Path to record tfevents files.
      available_norm_types: A list of string of norm types.
    """
    self.log_dir = log_dir
    self.logobj = collections.defaultdict(list)
    self.norm_types = available_norm_types

    os.makedirs(log_dir, exist_ok=True)

  def scalar(self, name, val, step):
    self.logobj[name] += [(step, float(val))]

  def array(self, name, val, step):
    self.logobj[name] += [(step, val)]

  def object(self, name, val):
    self.logobj[name] = val

  def flush(self, filename='log.jb'):
    path = os.path.join(self.log_dir, filename)
    with open(path, 'wb') as file:
      joblib.dump(dict(self.logobj), file)

  def last_scalars_to_str(self, keys):
    """Generates a human-readable string of the last values for given keys."""
    logstr = ''
    for k in keys:
      if k in self.logobj:
        text = tag_to_cmd_text(k, self.norm_types)
        val = self.logobj[k][-1][1]
        # Print the last item of a list of log values, example: largest epsilon
        # in adversarial risk
        if isinstance(val, list) or isinstance(val, tuple):
          val = '(%.2f: %.4f)' % (val[0][-1], val[1][-1])
        else:
          val = '%.4f' % val
        logstr += '%s: %s\t' % (text, val)
    return logstr


def _append_norm(ktl0, norm_types):
  """Generate human-readable labels for multiple norms."""

  ktl = {}
  for norm_type in norm_types:
    for key, val in ktl0.items():
      ktl['%s/%s' % (key, norm_type)] = '%s(%s)' % (val, norm_type)
  return ktl


def tag_to_cmd_text(tag, norm_types):
  """Generate a human-readable text to be printed in standard output logs."""
  # Table of keys to text labels
  ktl0 = {
      'zero_one': '0/1',
      'loss': "Model's loss",
  }

  ktl0.update(_append_norm({'adv': 'Adv'}, norm_types))

  keys_to_label = {}
  for k, v in ktl0.items():
    keys_to_label['risk/train/' + k] = '%s Risk(Train)' % v
    keys_to_label['risk/test/' + k] = '%s Risk(Test)' % v
  ktl0 = {
      'grad/norm': '||d/dw||',
      'weight/norm': '||w||',
      'csim_to_wmin': 'cos_sim(w,w_min)',
  }
  keys_to_label.update(_append_norm(ktl0, norm_types))

  return keys_to_label.get(tag, tag)
