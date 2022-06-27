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

"""Main train and eval loop for SimCLR+linear layer experiments.

Given an existing trained SimCLR model, trains a linear layer on top to predict
the original latents from dsprites dataset.
"""

from absl import app
from absl import flags
from absl import logging

from simclr.tf2 import lars_optimizer as lars
import tensorflow.compat.v2 as tf

import graph_compression.contrastive_learning.data_utils.learning_latents as data_lib  # pylint: disable=unused-import
import graph_compression.contrastive_learning.datasets.learning_latents as datasets_lib
import graph_compression.contrastive_learning.metrics_utils.learning_latents as metrics_lib
import graph_compression.contrastive_learning.models.learning_latents as models_lib


FLAGS = flags.FLAGS

USE_TPU = flags.DEFINE_boolean('use_tpu', False, 'For TPU training.')

TPU_ADDRESS = flags.DEFINE_string('tpu_address', None,
                                  'Manually specify a TPU address.')

MASTER = flags.DEFINE_string('master', '',
                             'Required for compatibility, leave blank.')

LR = flags.DEFINE_float('learning_rate', 1e-1,
                        'Learning rate for linear layer.')

L2 = flags.DEFINE_float('l2_penalty', 1e-4, 'Penalty for L2 regularization.')

T_BATCHSIZE = flags.DEFINE_integer('train_batch_size', 512,
                                   'Batch size for training.')

T_STEPS_PER_LOOP = flags.DEFINE_integer(
    'train_steps_per_loop', 5,
    'How many train steps to run between metrics summaries updates.')

TOTAL_STEPS = flags.DEFINE_integer('total_steps', 1,
                                   'Number of steps to train for.')

DATA_DIR = flags.DEFINE_string('data_dir', None, 'Directory to log data to.')

IMG_SIZE = flags.DEFINE_list(
    'img_size', None,
    'Optional image rescaling (comma separated list representing [new_height, new_width]).'
)

NUM_CHANNELS = flags.DEFINE_integer(
    'num_channels', None, 'Optional image tiling to multiple channels.')

PRETRAINED_MODEL_PATH = flags.DEFINE_string('pretrained_model_path', None,
                                            'Path to saved pretrained model.')

EVAL_SPLIT = flags.DEFINE_float('eval_split', 0.1,
                                'Fraction of dataset to use for eval.')

E_BATCHSIZE = flags.DEFINE_integer('eval_batch_size', 2048,
                                   'Batch size for eval.')

E_FREQ = flags.DEFINE_integer('eval_frequency', 5,
                              'How often to run eval loop.')

SEED = flags.DEFINE_integer('seed', None, 'Specify a random seed.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')


  # set up tpu strategy
  if USE_TPU.value:
    tpu_address = TPU_ADDRESS.value or MASTER.value

    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=tpu_address)
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    strategy = tf.distribute.TPUStrategy(cluster_resolver)

  else:
    # no-op strategy: for debugging / running on one machine
    strategy = tf.distribute.get_strategy()


  # actual run

  with strategy.scope():

    if IMG_SIZE.value is not None:
      img_size = tf.convert_to_tensor([int(v) for v in IMG_SIZE.value])

    # set up dataset
    dataset = datasets_lib.get_standard_dataset(
        name='dsprites',
        img_size=img_size,
        num_channels=NUM_CHANNELS.value,
        eval_split=EVAL_SPLIT.value,
        seed=SEED.value)
    train_df, train_ds, num_train_examples = dataset['train']
    eval_df, eval_ds, num_eval_examples = dataset['eval']

    del train_df, eval_df  ## not used here

    logging.info('Train, eval sets contain %s, %s elements respectively',
                 num_train_examples, num_eval_examples)
    for x in train_ds.take(1):
      num_classes = x['values'].shape[0]
    logging.info('Num classes is %s', num_classes)

    logging.info('Setting up datasets...')
    t_batchsize, e_batchsize = T_BATCHSIZE.value, E_BATCHSIZE.value
    train_ds = train_ds.shuffle(
        buffer_size=t_batchsize * 10, reshuffle_each_iteration=True)
    # drop the final partial batch for tpu reasons
    train_ds_batched = train_ds.batch(t_batchsize, drop_remainder=True)
    eval_ds_batched = eval_ds.batch(e_batchsize, drop_remainder=True)
    # so now we need to update the number of examples to match
    num_train_examples = (num_train_examples // t_batchsize) * t_batchsize
    num_eval_examples = (num_eval_examples // e_batchsize) * e_batchsize

    train_ds_dist = strategy.experimental_distribute_dataset(train_ds_batched)
    eval_ds_dist = strategy.experimental_distribute_dataset(eval_ds_batched)
    logging.info('Datasets set up, setting up model...')

    # instantiate optimizer and regularizer
    optimizer = lars.LARSOptimizer(LR.value)

    # loss has to be handled carefully when distributed on multiple cores;
    # specify no reduction for now and handle reduction manually in train step.
    loss_fn = tf.keras.losses.MeanSquaredError(
        reduction=tf.keras.losses.Reduction.NONE)
    regularizer = tf.keras.regularizers.L2(L2.value)

    # set up model

    model = models_lib.LinearLayerOverPretrainedSimclrModel(
        PRETRAINED_MODEL_PATH.value, optimizer, num_classes)

    logging.info('Optimizer, loss fn, model set up')

  # set up metrics and summary writers for train and eval, if required
  if DATA_DIR.value is not None:
    # metrics need to be within a strategy scope
    with strategy.scope():
      logging.info('Starting on computing y_bar and tss...')
      # TODO(zeef): implement a load from cache option to replace this
      # For testing, just hardcode the values because the computation is slow.
      # These are the values for the dsprites dataset.

      # y_bar, tss = metrics_lib.get_tss_for_r2(strategy, eval_ds_dist,
      #                                         num_classes, num_eval_examples,
      #                                         e_batchsize)

      y_bar = tf.constant([
          0.33251953, 0.33551705, 0.33196342, 0.74878615, 0.50025487,
          0.49955714, 0.5002258
      ],
                          dtype=tf.float32)
      tss = tf.constant([
          16363.938, 16437.312, 16350.195, 2150.051, 6469.37, 6572.2305,
          6550.672
      ],
                        dtype=tf.float32)
      logging.info('CAUTION! Hardcoded values for y_bar and tss!')

      logging.info('y_bar, tss are %s, %s', y_bar, tss)

      train_metrics = metrics_lib.DspritesTrainMetrics(DATA_DIR.value)
      eval_metrics = metrics_lib.DspritesEvalMetrics(DATA_DIR.value, tss)

    logging.info('Metrics set up')

  # define functions for train step, eval step, metrics update step
  @tf.function
  def train_step_loop(iterator, steps_per_loop):

    def step_fn(x):
      with tf.GradientTape() as tape:
        preds = model(x['image'])
        # loss is a tensor of per_example losses of size batch_size/num_replicas
        loss = loss_fn(x['values'], preds)
        loss += tf.reduce_sum(
            [regularizer(w) for w in model.dense_layer.trainable_weights])
        # pass this to metrics first so it gets an accurate count of examples
        if DATA_DIR.value is not None:
          train_metrics.update_metrics(loss, x['values'], preds)
        # now average the loss and then also divide by number of replicas
        # since the gradients from each replica are added together
        loss = tf.reduce_mean(loss) / strategy.num_replicas_in_sync
      dense_layer_weights = model.dense_layer.trainable_weights
      grads = tape.gradient(loss, dense_layer_weights)
      model.optimizer.apply_gradients(zip(grads, dense_layer_weights))

    for _ in tf.range(steps_per_loop):
      strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def eval_step_loop(iterator, steps_per_loop):

    def step_fn(x):
      preds = model(x['image'])
      loss = loss_fn(x['values'], preds)
      # update eval metics
      if DATA_DIR.value is not None:
        eval_metrics.update_metrics(loss, x['values'], preds)
      # no need to worry about scaling the loss here, since no gradients

    for _ in tf.range(steps_per_loop):
      strategy.run(step_fn, args=(next(iterator),))

  def metrics_update_loop(metrics_obj, global_step):
    for k in metrics_obj.writer_names:
      logging.info('Writing metric: %s', k)
      with metrics_obj.summary_writers[k].as_default():
        metrics_obj.write_metrics_to_summary(
            metrics_obj.metrics_dict[k], global_step=global_step)
        metrics_obj.summary_writers[k].flush()
      for metric in metrics_obj.metrics_dict[k]:
        metric.reset_state()

  # training loop

  num_eval_steps = num_eval_examples // e_batchsize
  num_train_steps = num_train_examples // t_batchsize
  num_train_steps_per_loop = T_STEPS_PER_LOOP.value
  num_train_loops_per_eval = E_FREQ.value // num_train_steps_per_loop

  train_iterator_step = 0
  current_step = 0

  logging.info('starting main training loop')
  train_iterator = iter(train_ds_dist)
  while current_step < TOTAL_STEPS.value:
    logging.info('current step %s, global step %s', current_step,
                 optimizer.iterations.numpy())
    for _ in range(num_train_loops_per_eval):
      # check there's enough examples left in the iterator and remake if needed
      # TODO(zeef): rewrite dataset creation to repeat forever?
      if train_iterator_step + num_train_steps_per_loop >= num_train_steps:
        train_iterator = iter(train_ds_dist)
        train_iterator_step = 0

      train_step_loop(train_iterator, num_train_steps_per_loop)


      metrics_update_loop(train_metrics, optimizer.iterations.numpy())

      # keep track of how far through train_iterator we are
      train_iterator_step += num_train_steps_per_loop

    # now run through the entire eval dataset and update eval metrics
    logging.info('Updating eval metrics for step %s',
                 optimizer.iterations.numpy())
    eval_iterator = iter(eval_ds_dist)
    eval_step_loop(eval_iterator, num_eval_steps)
    metrics_update_loop(eval_metrics, optimizer.iterations.numpy())

    # finally update current_step for the while loop to check
    current_step = optimizer.iterations.numpy()


if __name__ == '__main__':
  tf.compat.v1.enable_v2_behavior()
  app.run(main)
