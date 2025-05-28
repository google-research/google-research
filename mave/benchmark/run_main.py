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

"""Binary to train/eval the BERT sequence tagging model."""
import itertools
import json
import os
import random
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
from clu import platform
import ml_collections
from ml_collections import config_flags
import tensorflow as tf

from mave.benchmark.data import datasets
from mave.benchmark.models import bert_tagger
from mave.benchmark.models import bilstm_crf_tagger
from mave.benchmark.models import etc_tagger
from official.nlp import optimization
from official.nlp.bert import input_pipeline
from official.nlp.bert import model_saving_utils
from official.utils.misc import keras_utils

_CONFIG = config_flags.DEFINE_config_file(
    'config',
    default=None,
    help_string='Training configuration.',
    lock_config=True)
flags.mark_flags_as_required(['config'])

# Connect to remote TensorFlow server.
_USE_TPU = flags.DEFINE_bool('use_tpu', True, 'Whether running on TPU or not.')
_TPU = flags.DEFINE_string(
    'tpu', 'local', "The BNS address of the first TPU worker. 'local' for GPU.")


def _log_info(config: ml_collections.ConfigDict,
              message: str,
              log_file: str,
              mode: str = 'a') -> None:
  log_path = os.path.join(config.train.model_dir, log_file)
  with tf.io.gfile.GFile(log_path, mode) as file:
    print(message, file=file)


def _get_config() -> ml_collections.FrozenConfigDict:
  """Returns a frozen config dict by updating dynamic fields."""
  config = ml_collections.ConfigDict(_CONFIG.value)
  dataset = datasets.DATASETS.get(config.data.version, None)
  if dataset is None:
    raise ValueError(f'Invalid data version: {config.data.version!r}')
  config.train.tf_examples_filespec = dataset.train_tf_records
  config.eval.tf_examples_filespec = dataset.eval_tf_records
  config.eval.num_eval_steps = min(
      config.eval.num_eval_steps,
      dataset.eval_size // config.eval.eval_batch_size)
  config = ml_collections.FrozenConfigDict(config)

  logging.info(config)
  tf.io.gfile.makedirs(config.train.model_dir)
  _log_info(config, json.dumps(config.to_dict(), indent=2), 'config.json', 'w')

  return config


def create_dataset(
    config: ml_collections.FrozenConfigDict,
    name_to_features: Mapping[str, tf.io.FixedLenFeature],
    select_data_from_record_fn: Callable[[Mapping[str, tf.Tensor]], Any],
    is_training: bool = True,
    input_pipeline_context: Optional[tf.distribute.InputContext] = None
) -> tf.data.Dataset:
  """Creates input dataset from (tf)records files for train/eval."""
  if is_training:
    tf_examples_filespec = config.train.tf_examples_filespec
    global_batch_size = config.train.train_batch_size
  else:
    tf_examples_filespec = config.eval.tf_examples_filespec
    global_batch_size = config.eval.eval_batch_size

  filenames = list(
      itertools.chain.from_iterable(
          tf.io.gfile.glob(fp) for fp in tf_examples_filespec.split(',')))
  random.Random(config.train.data_random_seed).shuffle(filenames)

  dataset = tf.data.Dataset.from_tensor_slices(filenames)
  dataset = dataset.interleave(
      tf.data.TFRecordDataset,
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
      deterministic=True,
  )
  dataset = dataset.map(
      lambda record: input_pipeline.decode_record(record, name_to_features),
      num_parallel_calls=tf.data.experimental.AUTOTUNE,
  )

  # The dataset is always sharded by number of hosts.
  # num_input_pipelines is the number of hosts rather than number of cores.
  if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
    dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                            input_pipeline_context.input_pipeline_id)

  if is_training:
    dataset = dataset.shuffle(100, seed=config.train.data_random_seed)
    dataset = dataset.repeat()

  if input_pipeline_context:
    batch_size = input_pipeline_context.get_per_replica_batch_size(
        global_batch_size)
  else:
    batch_size = global_batch_size

  dataset = dataset.map(
      select_data_from_record_fn,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=is_training)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def get_dataset_fn(
    config: ml_collections.FrozenConfigDict,
    is_training: bool = True
) -> Callable[[tf.distribute.InputContext], tf.data.Dataset]:
  """Returns a dataset function."""
  if config.model_type == 'bert':
    name_to_features = {
        'input_ids':
            tf.io.FixedLenFeature([config.model.seq_length], tf.int64),
        'input_mask':
            tf.io.FixedLenFeature([config.model.seq_length], tf.int64),
        'segment_ids':
            tf.io.FixedLenFeature([config.model.seq_length], tf.int64),
        'label_ids':
            tf.io.FixedLenFeature([config.model.seq_length], tf.int64),
    }

    def _select_data_from_record(record):
      x = {
          'input_word_ids': record['input_ids'],
          'input_mask': record['input_mask'],
          'input_type_ids': record['segment_ids'],
      }
      y = tf.expand_dims(record['label_ids'], axis=-1)
      w = tf.cast(record['input_mask'], tf.float32)
      return x, y, w

  elif config.model_type == 'bilstm_crf':
    name_to_features = {
        'input_ids':
            tf.io.FixedLenFeature([config.model.seq_length], tf.int64),
        'input_mask':
            tf.io.FixedLenFeature([config.model.seq_length], tf.int64),
        'label_ids':
            tf.io.FixedLenFeature([config.model.seq_length], tf.int64),
    }

    def _select_data_from_record(record):
      x = {
          'input_word_ids': record['input_ids'],
          'input_mask': record['input_mask'],
      }
      y = record['label_ids']
      return x, y

  elif config.model_type == 'etc':
    name_to_features = {
        'global_token_ids':
            tf.io.FixedLenFeature([config.etc.global_seq_length], tf.int64),
        'global_breakpoints':
            tf.io.FixedLenFeature([config.etc.global_seq_length], tf.int64),
        'global_token_type_ids':
            tf.io.FixedLenFeature([config.etc.global_seq_length], tf.int64),
        'global_label_ids':
            tf.io.FixedLenFeature([config.etc.global_seq_length], tf.int64),
        'long_token_ids':
            tf.io.FixedLenFeature([config.etc.long_seq_length], tf.int64),
        'long_breakpoints':
            tf.io.FixedLenFeature([config.etc.long_seq_length], tf.int64),
        'long_token_type_ids':
            tf.io.FixedLenFeature([config.etc.long_seq_length], tf.int64),
        'long_paragraph_ids':
            tf.io.FixedLenFeature([config.etc.long_seq_length], tf.int64),
        'long_label_ids':
            tf.io.FixedLenFeature([config.etc.long_seq_length], tf.int64),
    }

    def _select_data_from_record(record):
      input_features = [
          'global_token_ids',
          'global_breakpoints',
          'global_token_type_ids',
          'long_token_ids',
          'long_breakpoints',
          'long_token_type_ids',
          'long_paragraph_ids',
      ]
      x = {k: record[k] for k in input_features}
      y = {
          'global': tf.expand_dims(record['global_label_ids'], axis=-1),
          'long': tf.expand_dims(record['long_label_ids'], axis=-1),
      }
      w = {
          'global':
              tf.expand_dims(
                  tf.cast(
                      tf.greater(record['global_token_ids'], 0), tf.float32),
                  axis=-1,
              ),
          'long':
              tf.cast(tf.greater(record['long_token_ids'], 0), tf.float32),
      }
      return x, y, w
  else:
    raise ValueError(f'Invalid config model type {config.model_type!r}.')

  def _dataset_fn(ctx: tf.distribute.InputContext) -> tf.data.Dataset:
    return create_dataset(config, name_to_features, _select_data_from_record,
                          is_training, ctx)

  return _dataset_fn


def _create_default_loss() -> tf.keras.losses.Loss:
  return tf.keras.losses.BinaryCrossentropy(
      reduction=tf.keras.losses.Reduction.SUM)


def _create_default_metrics() -> Sequence[tf.keras.metrics.Metric]:
  return [
      tf.keras.metrics.Precision(),
      tf.keras.metrics.Recall(),
  ]


def build_model(
    config: ml_collections.FrozenConfigDict) -> Tuple[tf.keras.Model, Any, Any]:
  """Returns keras model."""
  if config.model_type == 'bilstm_crf':
    model = bilstm_crf_tagger.build_model(config)
    loss = None
    metrics = _create_default_metrics()
  elif config.model_type == 'bert':
    model = bert_tagger.build_model(config)
    loss = _create_default_loss()
    metrics = _create_default_metrics()
  elif config.model_type == 'etc':
    model = etc_tagger.build_model(config)
    loss = {
        'global': _create_default_loss(),
        'long': _create_default_loss(),
    }
    metrics = {
        'global': _create_default_metrics(),
        'long': _create_default_metrics(),
    }

  else:
    raise ValueError(f'Invalid config model type {config.model_type!r}.')

  return model, loss, metrics


class UseOptimizerIterationAsTrainStep(tf.keras.callbacks.Callback):
  """A callback to sync keras model global batch step."""

  def set_model(self, model):
    self.model = model

  def on_train_begin(self, logs=None):
    self.model._train_counter.assign(self.model.optimizer.iterations)  # pylint: disable=protected-access


def run_train_and_eval(config: ml_collections.FrozenConfigDict,
                       strategy: tf.distribute.Strategy):
  """Runs training and evaluation using Keras compile/fit API."""
  with strategy.scope():
    tagger_model, loss, metrics = build_model(config)

    optimizer = optimization.create_optimizer(
        config.train.initial_learning_rate, config.train.num_train_steps,
        config.train.num_warmup_steps, config.train.end_learning_rate, 'adamw')

    trainable_variables_str = '\n'.join(
        f'{tvar.name:<150}: ({tvar.shape})'
        for tvar in tagger_model.trainable_variables)
    _log_info(config, trainable_variables_str, 'trainable_variables.txt', 'w')

    tagger_model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics,
        steps_per_execution=config.train.steps_per_loop,
    )

    model_summary_file = os.path.join(config.train.model_dir, 'model.summary')
    with tf.io.gfile.GFile(model_summary_file, 'w') as file:
      tagger_model.summary(line_length=186, print_fn=file.write)

    train_dataset = strategy.distribute_datasets_from_function(
        get_dataset_fn(config, is_training=True))
    eval_dataset = strategy.distribute_datasets_from_function(
        get_dataset_fn(config, is_training=False))

    summary_dir = os.path.join(config.train.model_dir, 'summaries')
    summary_callback = tf.keras.callbacks.TensorBoard(
        summary_dir, update_freq=config.train.save_summary_steps)

    checkpoint = tf.train.Checkpoint(model=tagger_model, optimizer=optimizer)
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=config.train.model_dir,
        max_to_keep=config.train.checkpoints_to_keep,
        step_counter=optimizer.iterations,
        checkpoint_interval=config.train.save_checkpoints_steps)
    if checkpoint_manager.latest_checkpoint:
      checkpoint.restore(checkpoint_manager.latest_checkpoint)
      logging.info('Restored from latest checkpoint: %s',
                   checkpoint_manager.latest_checkpoint)
    checkpoint_callback = keras_utils.SimpleCheckpoint(checkpoint_manager)

    time_history_callback = keras_utils.TimeHistory(
        batch_size=config.train.train_batch_size,
        log_steps=config.train.save_summary_steps,
        logdir=config.train.model_dir)

    sync_batch_step_callback = UseOptimizerIterationAsTrainStep()

    initial_epoch = (
        checkpoint.optimizer.iterations.numpy() // config.train.steps_per_epoch)
    tagger_model.fit(
        x=train_dataset,
        steps_per_epoch=config.train.steps_per_epoch,
        epochs=config.train.epochs,
        initial_epoch=initial_epoch,
        validation_data=eval_dataset,
        validation_steps=config.eval.num_eval_steps,
        callbacks=[
            summary_callback,
            checkpoint_callback,
            time_history_callback,
            sync_batch_step_callback,
        ],
        verbose=2)

  model_export_path = os.path.join(config.train.model_dir, 'exported_models')
  model_saving_utils.export_bert_model(model_export_path, model=tagger_model)


def _get_gpu_or_cpu_devices(master: str) -> Optional[Sequence[str]]:
  """Returns the list of GPU or CPU devices (prefering GPUs)."""
  if master != 'local':
    tf.config.experimental_connect_to_host(master)
    logging.warning('Remote eager configured. Remote eager can be slow.')
  for device_type in ['GPU', 'CPU']:
    devices = tf.config.experimental.list_logical_devices(
        device_type=device_type)
    if devices:
      return [d.name for d in devices]
  raise ValueError(
      'Could not find any logical devices of type GPU or CPU, logical devices: '
      f'{tf.config.experimental.list_logical_devices()}')


def main(_):
  config = _get_config()

  tf.config.optimizer.set_jit('autoclustering')

  print(f'jit compile: {tf.config.optimizer.get_jit()}')

  platform.work_unit().create_artifact(
      platform.ArtifactType.URL, f'http://cnsviewer{config.train.model_dir}',
      'Work unit dir')

  devices_str = '\n'.join(repr(d) for d in tf.config.list_physical_devices())
  _log_info(config, devices_str, 'host_devices.txt', 'w')

  master = _TPU.value or 'local'
  if _USE_TPU.value:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=master)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
  else:
    devices = _get_gpu_or_cpu_devices(master=master)
    if len(devices) == 1:
      strategy = tf.distribute.OneDeviceStrategy(devices[0])
    else:
      strategy = tf.distribute.MirroredStrategy(devices)

  run_train_and_eval(config, strategy)


if __name__ == '__main__':
  app.run(main)
