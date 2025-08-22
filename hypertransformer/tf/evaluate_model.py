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

"""Online model evaluation script."""
import functools
import os

from typing import Callable, Dict, List

from absl import app
from absl import flags

import numpy as np
import tensorflow.compat.v1 as tf

import tf_slim

from hypertransformer.tf import common_flags  # pylint:disable=unused-import
from hypertransformer.tf import eval_model_flags  # pylint:disable=unused-import
from hypertransformer.tf import train

from hypertransformer.tf.core import common
from hypertransformer.tf.core import common_ht
from hypertransformer.tf.core import evaluation_lib as eval_lib
from hypertransformer.tf.core import layerwise
from hypertransformer.tf.core import layerwise_defs  # pylint:disable=unused-import
from hypertransformer.tf.core import util

FLAGS = flags.FLAGS

DS_PRESETS = {
    'miniimagenet':
        {
            'train': 'miniimagenet:0-63',
            'val': 'miniimagenet:64-79',
            'test': 'miniimagenet:80-99'
        },
    'omniglot':
        {
            'train': 'omniglot:0-1149',
            'val': 'omniglot:1150-1199',
            'test': 'omniglot:1200-1622',
        },
    'tieredimagenet':
        {
            'train': 'tieredimagenet:0-350',
            'val': 'tieredimagenet:351-447',
            'test': 'tieredimagenet:448-607',
        },
}

SECONDS_SLEEP = 10
SUMMARY_SUBFOLDER = 'eval'


def get_load_vars():
  """Returns a list of variables to load from the checkpoint."""
  tf.train.get_or_create_global_step()
  load_vars = []
  for v in tf.all_variables():
    if v.name.startswith('augmentation'):
      continue
    if 'train_images' in v.name or 'train_labels' in v.name:
      continue
    load_vars.append(v)
  return load_vars


def run_evaluation(model_config,
                   dataset_configs,
                   make_outputs_fn):
  """Runs model evaluation loop over a set of datasets."""
  is_metadataset = False
  dataset_info = {name: eval_lib.dataset_with_custom_labels(
      model_config, config) for name, config in dataset_configs.items()}

  datasets, custom_labels, assign_ops, outputs = {}, {}, {}, {}
  for name, info in dataset_info.items():
    datasets[name], custom_labels[name] = info
    if not is_metadataset:
      with tf.variable_scope(f'dataset_{name}'):
        images, labels, assign_ops[name] = eval_lib.make_train_samples(
            datasets[name], same_batch=True)
      datasets[name].transformer_images = images
      datasets[name].transformer_labels = labels
    else:
      assign_ops[name] = tf.no_op()
    outputs[name] = make_outputs_fn(model_config=model_config,
                                    dataset=datasets[name])

  # Meta-dataset batches are provided by the library and we can only get one
  # batch per episode (hence 1 evaluation batch).
  eval_config = eval_lib.EvaluationConfig(
      num_task_evals=FLAGS.num_task_evals,
      num_eval_batches=FLAGS.num_eval_batches if not is_metadataset else 1,
      load_vars=get_load_vars())

  summary_writer = util.MultiFileWriter(
      os.path.join(FLAGS.train_log_dir, SUMMARY_SUBFOLDER), graph=None)
  placeholders = {name: tf.placeholder(shape=(), dtype=tf.float32)
                  for name in dataset_info}
  merged_summary = tf.summary.merge([
      tf.summary.scalar(f'eval_accuracy/{name}', placeholders[name])
      for name in dataset_info])

  with tf.Session() as sess:
    global_step = tf.train.get_or_create_global_step()
    saver = tf.train.Saver(eval_config.load_vars)
    sess.run((tf.initializers.global_variables(),
              tf.initializers.local_variables()))

    last_checkpoint = None

    while True:
      checkpoint = tf_slim.evaluation.wait_for_new_checkpoint(
          FLAGS.train_log_dir,
          last_checkpoint,
          seconds_to_sleep=SECONDS_SLEEP)
      last_checkpoint = checkpoint

      saver.restore(sess, checkpoint)
      step = sess.run(global_step)

      feed_dict = {}
      for name in dataset_info:
        _, accs = eval_lib.evaluate_dataset(
            custom_labels=custom_labels[name],
            dataset=datasets[name],
            assign_op=assign_ops[name],
            outputs=outputs[name],
            eval_config=eval_config)
        feed_dict[placeholders[name]] = np.mean(accs)

      serialized = sess.run(merged_summary, feed_dict=feed_dict)
      summary_writer.add_summary(serialized, global_step=step)

      if step >= FLAGS.train_steps:
        break


def evaluate_layerwise(model_config,
                       dataset_configs
                       ):
  """Evaluates a pretrained 'layerwise' model."""
  model = layerwise.build_model(model_config.cnn_model_name,
                                model_config=model_config)

  run_evaluation(model_config=model_config,
                 dataset_configs=dataset_configs,
                 make_outputs_fn=functools.partial(
                     eval_lib.apply_layerwise_model, model=model))


def make_data_configs():
  """Creates dataset config for all datasets that need to be evaluated."""
  eval_datasets = str(FLAGS.eval_datasets)
  if eval_datasets in DS_PRESETS:
    datasets = DS_PRESETS[eval_datasets]
  else:
    raise ValueError(f'Dataset collection {eval_datasets} is not in presets. '
                     'Manual dataset specifications are not currently '
                     'supported.')

  def _builder(name):
    builders = {
        'train': train.make_dataset_config,
        'val': train.make_test_dataset_config,
        'test': train.make_test_dataset_config,
    }
    return builders[name]

  return {name: _builder(name)(dataset)
          for name, dataset in datasets.items()}


def evaluate_pretrained(train_config,
                        optimizer_config,
                        layerwise_model_config
                        ):
  """Evaluates a pre-trained model."""
  del train_config, optimizer_config
  data_configs = make_data_configs()
  evaluate_layerwise(
      model_config=layerwise_model_config,
      dataset_configs=data_configs)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if common_flags.PRETRAIN_SHARED_FEATURE.value:
    # No need to do evaluation if all we do is pretrain the shared feature.
    return

  tf.disable_eager_execution()

  for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

  FLAGS.samples_cnn = FLAGS.eval_batch_size

  evaluate_pretrained(
      train_config=train.make_train_config(),
      optimizer_config=train.make_optimizer_config(),
      layerwise_model_config=train.make_layerwise_model_config())


if __name__ == '__main__':
  app.run(main)
