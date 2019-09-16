# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""ResNet Hyperparameter sets.

These hyperparameters are tuned for bringing a ResNet-50 model trained on
ImageNet to convergence. If you are training using different data you almost
certainly will need to adjust these; you can either do so by adding new
parameter sets or by overriding specific parameters using flags; in
resnet_main.py this is accomplished by setting the --param_overrides flag.
"""

import tensorflow as tf
import yaml


def from_file(path):
  """Given a path to a YAML file, read the file and override default parameters.

  Args:
    path: Python string containing path to file. If None, return default params.

  Returns:
    Python dict of hyperparameters.
  """
  params = resnet_imagenet_defaults
  if path is None:
    return params
  with tf.gfile.Open(path, 'r') as f:
    overrides = yaml.safe_load(f)
  for k, v in overrides.iteritems():
    params[k] = v
  return params


def override(params, overrides):
  """Given a dictionary of parameters and a list of overrides, merge them.

  Args:
    params: Python dict containing a base parameter set.
    overrides: Python list of strings. This is a set of k=v overrides for the
      parameters in `params`; if `k=v1` in `params` but `k=v2` in `overrides`,
      the second value wins and the value for `k` is `v2`.

  Returns:
    Python dict containing parameter set.
  """
  if params is None:
    params = resnet_imagenet_defaults
  if not isinstance(params, dict):
    raise ValueError(
        'The base parameter set must be a Python dict, was: {}'.format(params))
  if overrides is None:
    overrides = []
  if isinstance(overrides, str):
    overrides = [overrides]
  if not isinstance(overrides, list):
    raise ValueError(
        'Expected that param_overrides would be None, a single string, or a '
        'list of strings, was: {}'.format(overrides))
  for kv_pair in overrides:
    if not isinstance(kv_pair, str):
      raise ValueError(
          'Expected that param_overrides would contain Python list of strings, '
          'but encountered an item: {}'.format(kv_pair))
    key, value = kv_pair.split('=')
    parser = type(params[key])
    if parser is bool:
      params[key] = value not in ('0', 'False', 'false')
    else:
      params[key] = parser(value)
  return params


def log_hparams_to_model_dir(params, model_dir):  # pylint: disable=unused-argument
  """Given some param_set and the model_dir for training, export params to file.

  Args:
    params: Python dict of model parameters.
    model_dir: Python string of filepath to model_dir where checkpoints for this
      run will be saved.
  """
  if model_dir is None:
    return
  tf.gfile.MakeDirs(model_dir)
  with tf.gfile.GFile(model_dir + '/params.yaml', 'w') as f:
    f.write(yaml.dump(params))


resnet_imagenet_defaults = dict(
    resnet_depth=50,
    train_batch_size=1024,
    eval_batch_size=1024,
    num_train_images=1281167,
    num_eval_images=50000,
    train_steps=112590,
    base_learning_rate=0.1,
    iterations_per_loop=1251,
    use_tpu=True,
    num_cores=8,
    enable_lars=False,
    transpose_input=True,
    precision='bfloat16',
    num_label_classes=1000,
    use_cache=True,
    use_async_checkpointing=False,
    image_size=224,
    momentum=0.9,
    weight_decay=1e-4,
    label_smoothing=0.0,
    poly_rate=0.0,
    skip_host_call=False,
    num_parallel_calls=8,
    dropblock_groups='',
    dropblock_keep_prob=0.9,
    dropblock_size=7,
    data_format='channels_last',
    target_num_classes=500,
    rl_delay_steps=0,
    rl_learning_rate=1e-1,
)
