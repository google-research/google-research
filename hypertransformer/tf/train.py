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

"""Training binary."""
import functools
from typing import Any, Dict, List, Optional, Tuple

from absl import app
from absl import flags

import tensorflow.compat.v1 as tf

from hypertransformer.tf import common_flags  # pylint:disable=unused-import
from hypertransformer.tf import eval_model_flags  # pylint:disable=unused-import

from hypertransformer.tf.core import common
from hypertransformer.tf.core import common_ht
from hypertransformer.tf.core import layerwise
from hypertransformer.tf.core import layerwise_defs  # pylint:disable=unused-import
from hypertransformer.tf.core import train_lib
from hypertransformer.tf.core import util

FLAGS = flags.FLAGS


def make_train_config():
  return common.TrainConfig(train_steps=FLAGS.train_steps,
                            steps_between_saves=FLAGS.steps_between_saves)


def make_optimizer_config():
  return common.OptimizerConfig(learning_rate=FLAGS.learning_rate,
                                lr_decay_steps=FLAGS.learning_rate_decay_steps,
                                lr_decay_rate=FLAGS.learning_rate_decay_rate)


def common_model_config():
  """Returns common ModelConfig parameters."""
  return {
      'num_transformer_samples': FLAGS.samples_transformer,
      'num_cnn_samples': FLAGS.samples_cnn,
      'num_labels': FLAGS.num_labels,
      'image_size': FLAGS.image_size,
      'cnn_model_name': FLAGS.cnn_model_name,
      'embedding_dim': FLAGS.embedding_dim,
      'cnn_dropout_rate': FLAGS.cnn_dropout_rate,
      'use_decoder': FLAGS.use_decoder,
      'add_trainable_weights': FLAGS.add_trainable_weights,
      'var_reg_weight': FLAGS.weight_variation_regularization,
      'transformer_activation': FLAGS.transformer_activation,
      'transformer_nonlinearity': FLAGS.transformer_nonlinearity,
      'cnn_activation': FLAGS.cnn_activation,
      'default_num_channels': FLAGS.default_num_channels,
      'shared_fe_dropout': FLAGS.shared_fe_dropout,
      'fe_dropout': FLAGS.fe_dropout,
  }


def make_layerwise_model_config():
  """Makes 'layerwise' model config."""
  if not FLAGS.num_layerwise_features:
    num_features = None
  else:
    num_features = int(FLAGS.num_layerwise_features)
  if FLAGS.lw_weight_allocation == 'spatial':
    weight_allocation = common_ht.WeightAllocation.SPATIAL
  elif FLAGS.lw_weight_allocation == 'output':
    weight_allocation = common_ht.WeightAllocation.OUTPUT_CHANNEL
  else:
    raise ValueError(f'Unknown `lw_weight_allocation` flag value '
                     f'"{FLAGS.lw_weight_allocation}"')
  return common_ht.LayerwiseModelConfig(
      feature_layers=2,
      query_key_dim_frac=FLAGS.lw_key_query_dim,
      value_dim_frac=FLAGS.lw_value_dim,
      internal_dim_frac=FLAGS.lw_inner_dim,
      num_layers=FLAGS.num_layers,
      heads=FLAGS.heads,
      kernel_size=common_flags.KERNEL_SIZE.value,
      stride=common_flags.STRIDE.value,
      dropout_rate=FLAGS.dropout_rate,
      num_features=num_features,
      nonlinear_feature=FLAGS.lw_use_nonlinear_feature,
      weight_allocation=weight_allocation,
      generate_bn=FLAGS.lw_generate_bn,
      generate_bias=FLAGS.lw_generate_bias,
      shared_feature_extractor=FLAGS.shared_feature_extractor,
      shared_features_dim=FLAGS.shared_features_dim,
      separate_bn_vars=FLAGS.separate_evaluation_bn_vars,
      shared_feature_extractor_padding=FLAGS.shared_feature_extractor_padding,
      generator=FLAGS.layerwise_generator,
      train_heads=FLAGS.warmup_steps > 0,
      max_prob_remove_unlabeled=FLAGS.max_prob_remove_unlabeled,
      max_prob_remove_labeled=FLAGS.max_prob_remove_labeled,
      number_of_trained_cnn_layers=(
          common_flags.NUMBER_OF_TRAINED_CNN_LAYERS.value),
      skip_last_nonlinearity=FLAGS.transformer_skip_last_nonlinearity,
      l2_reg_weight=FLAGS.l2_reg_weight,
      logits_feature_extractor=FLAGS.logits_feature_extractor,
      shared_head_weight=common_flags.SHARED_HEAD_WEIGHT.value,
      **common_model_config())


def make_optimizer(optim_config,
                   global_step):
  learning_rate = tf.train.exponential_decay(
      optim_config.learning_rate, global_step, optim_config.lr_decay_steps,
      optim_config.lr_decay_rate)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  return learning_rate, optimizer


def make_train_op(optimizer,
                  loss,
                  train_vars=None):
  global_step = tf.train.get_or_create_global_step()
  return optimizer.minimize(tf.reduce_mean(loss), global_step=global_step,
                            var_list=train_vars)


def make_dataset_config(dataset_spec = ''):
  if not dataset_spec:
    dataset_spec = FLAGS.train_dataset
  dataset, label_set = util.parse_dataset_spec(dataset_spec)
  if label_set is None:
    label_set = list(range(FLAGS.use_labels))
  return common_ht.DatasetConfig(
      dataset_name=dataset,
      use_label_subset=label_set,
      tfds_split='train',
      data_dir=FLAGS.data_dir,
      rotation_probability=FLAGS.rotation_probability,
      smooth_probability=FLAGS.smooth_probability,
      contrast_probability=FLAGS.contrast_probability,
      resize_probability=FLAGS.resize_probability,
      negate_probability=FLAGS.negate_probability,
      roll_probability=FLAGS.roll_probability,
      angle_range=FLAGS.angle_range,
      rotate_by_90=FLAGS.random_rotate_by_90,
      per_label_augmentation=FLAGS.per_label_augmentation,
      cache_path=FLAGS.data_numpy_dir,
      balanced_batches=FLAGS.balanced_batches,
      shuffle_labels_seed=FLAGS.shuffle_labels_seed,
      apply_image_augmentations=FLAGS.apply_image_augmentations,
      augment_individually=FLAGS.augment_images_individually,
      num_unlabeled_per_class=FLAGS.unlabeled_samples_per_class,
  )


def _default(new, default):
  return new if new >= 0 else default


def make_test_dataset_config(dataset_spec = ''):
  if not dataset_spec:
    dataset_spec = FLAGS.test_dataset
  dataset, label_set = util.parse_dataset_spec(dataset_spec)
  if label_set is None:
    raise ValueError('Test dataset should specify a set of labels.')
  return common_ht.DatasetConfig(
      dataset_name=dataset,
      use_label_subset=label_set,
      tfds_split=FLAGS.test_split,
      data_dir=FLAGS.data_dir,
      rotation_probability=_default(FLAGS.test_rotation_probability,
                                    FLAGS.rotation_probability),
      smooth_probability=_default(FLAGS.test_smooth_probability,
                                  FLAGS.smooth_probability),
      contrast_probability=_default(FLAGS.test_contrast_probability,
                                    FLAGS.contrast_probability),
      resize_probability=_default(FLAGS.test_resize_probability,
                                  FLAGS.resize_probability),
      negate_probability=_default(FLAGS.test_negate_probability,
                                  FLAGS.negate_probability),
      roll_probability=_default(FLAGS.test_roll_probability,
                                FLAGS.roll_probability),
      angle_range=_default(FLAGS.test_angle_range, FLAGS.angle_range),
      rotate_by_90=FLAGS.test_random_rotate_by_90,
      per_label_augmentation=FLAGS.test_per_label_augmentation,
      balanced_batches=FLAGS.balanced_batches,
      shuffle_labels_seed=FLAGS.shuffle_labels_seed,
      cache_path=FLAGS.data_numpy_dir,
      apply_image_augmentations=False,
      num_unlabeled_per_class=FLAGS.unlabeled_samples_per_class,
  )


def _make_warmup_loss(loss_heads,
                      loss_prediction,
                      global_step):
  """Uses head losses to build aggregate loss cycling through them."""
  # The warmup period is broken into a set of "head activation periods".
  # Each period, one head weight is linearly growing, while the previous
  # head weight goes down.
  # Basically, each moment of time only two heads are active and the active
  # heads slide towards the final layer.
  num_heads = len(loss_heads)
  steps_per_stage = FLAGS.warmup_steps / num_heads
  loss = 0
  weights = []

  # The following code ends up returning just the true model head loss
  # after `global step` reaches `warmup_steps`.

  for stage, head_loss in enumerate(loss_heads):
    target_steps = stage * steps_per_stage
    norm_step_dist = tf.abs(global_step - target_steps) / steps_per_stage
    # This weight starts at 0 and peaks reaching 1 at `target_steps`. It then
    # decays linearly to 0 and stays 0.
    weight = tf.maximum(0.0, 1.0 - norm_step_dist)
    weights.append(weight)
    loss += weight * head_loss

  target_steps = num_heads * steps_per_stage
  norm_step_dist = 1.0 + (global_step - target_steps) / steps_per_stage
  norm_step_dist = tf.nn.relu(norm_step_dist)
  # Weight for the actual objective linearly grows after the final layer head
  # peaks and then stays equal to 1.
  weight = tf.minimum(1.0, norm_step_dist)
  weights.append(weight)
  loss += weight * loss_prediction
  return loss, weights


def make_loss(labels,
              predictions,
              heads):
  """Makes a full loss including head 'warmup' losses."""
  losses = []
  for head in heads + [predictions]:
    head_loss = tf.losses.softmax_cross_entropy(
        labels, head, label_smoothing=FLAGS.label_smoothing)
    losses.append(head_loss)
  if len(losses) == 1:
    return losses[0], losses, [tf.constant(1.0, dtype=tf.float32)]

  global_step = tf.cast(tf.train.get_or_create_global_step(), tf.float32)
  loss, wamup_weights = _make_warmup_loss(losses[:-1], losses[-1], global_step)
  return loss, losses, wamup_weights


def create_shared_head(shared_features,
                       real_classes,
                       real_class_min,
                       real_class_max
                       ):
  """Creates a real class prediction head from the shared feature."""
  if real_classes is None or shared_features is None:
    return None, None
  if real_class_min is None or real_class_max is None:
    tf.logging.warning('Training classes boundaries are not provided. '
                       'Skippin shared head creation!')
    return None, None
  total_classes = real_class_max - real_class_min + 1
  with tf.variable_scope('shared_head', reuse=tf.AUTO_REUSE):
    fc = tf.layers.Dense(units=total_classes, name='fc')
    predictions = fc(shared_features)
  classes = real_classes - real_class_min
  one_hot_gt = tf.one_hot(classes, depth=total_classes)
  loss = tf.losses.softmax_cross_entropy(one_hot_gt, predictions,
                                         label_smoothing=FLAGS.label_smoothing)
  pred_classes = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
  accuracy = tf.cast(tf.math.equal(classes, pred_classes), tf.float32)
  num_samples = tf.cast(tf.shape(shared_features)[0], tf.float32)
  accuracy = tf.reduce_sum(accuracy) / num_samples
  return loss, accuracy


def create_layerwise_model(
    model_config,
    dataset,
    test_dataset,
    state,
    optim_config):
  """Creates a hierarchichal Transformer-CNN model."""
  tf.logging.info('Building the model')
  global_step = tf.train.get_or_create_global_step()
  model = layerwise.build_model(model_config.cnn_model_name,
                                model_config=model_config)

  with tf.variable_scope('model'):
    weight_blocks = model.train(dataset.transformer_images,
                                dataset.transformer_labels,
                                mask=dataset.transformer_masks,
                                mask_random_samples=True,
                                enable_fe_dropout=True)
    predictions = model.evaluate(dataset.cnn_images,
                                 weight_blocks=weight_blocks,
                                 training=False)
    heads = []
    if model_config.train_heads:
      outputs = model.layer_outputs.values()
      heads = [output[1] for output in outputs if output[1] is not None]

    test_weight_blocks = model.train(test_dataset.transformer_images,
                                     test_dataset.transformer_labels,
                                     mask=test_dataset.transformer_masks)
    test_predictions = model.evaluate(test_dataset.cnn_images,
                                      weight_blocks=test_weight_blocks,
                                      training=False)

  with tf.variable_scope('loss'):
    labels = tf.one_hot(dataset.cnn_labels, depth=model_config.num_labels)
    pred_labels = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    num_cnn_samples = tf.cast(tf.shape(dataset.cnn_labels)[0], tf.float32)

    def _acc(pred):
      accuracy = tf.cast(tf.math.equal(dataset.cnn_labels, pred), tf.float32)
      return tf.reduce_sum(accuracy) / num_cnn_samples

    accuracy = _acc(pred_labels)
    head_preds = [tf.cast(tf.argmax(head, axis=-1), tf.int32) for head in heads]
    head_accs = [_acc(pred) for pred in head_preds]

    test_pred_labels = tf.cast(tf.argmax(test_predictions, axis=-1), tf.int32)
    test_accuracy = tf.cast(
        tf.math.equal(test_dataset.cnn_labels, test_pred_labels), tf.float32)
    test_accuracy = tf.reduce_sum(test_accuracy) / num_cnn_samples

    summaries = []
    reg_losses = tf.losses.get_losses(
        loss_collection=tf.GraphKeys.REGULARIZATION_LOSSES)
    if reg_losses:
      summaries.append(tf.summary.scalar('loss/regularization',
                                         tf.reduce_sum(reg_losses)))

    shared_head_loss, shared_head_acc = create_shared_head(
        weight_blocks.shared_features, dataset.transformer_real_classes,
        dataset.real_class_min, dataset.real_class_max)

    state.loss, _, warmup_weights = make_loss(labels, predictions, heads)
    summaries.append(tf.summary.scalar('loss/ce', state.loss))
    if reg_losses:
      state.loss += tf.reduce_sum(reg_losses)
    _, optimizer = make_optimizer(optim_config, global_step)

  if shared_head_loss is not None:
    if model_config.shared_head_weight > 0.0:
      weighted_head_loss = shared_head_loss * model_config.shared_head_weight
      state.loss += weighted_head_loss
      summaries.append(tf.summary.scalar('loss/shared_head_loss',
                                         shared_head_loss))
      summaries.append(tf.summary.scalar('loss/weighted_shared_head_loss',
                                         weighted_head_loss))

  for head_id, acc in enumerate(head_accs):
    summaries.append(tf.summary.scalar(f'accuracy/head-{head_id+1}', acc))

  for head_id, warmup_weight in enumerate(warmup_weights[:-1]):
    summaries.append(tf.summary.scalar(f'warmup_weights/head-{head_id+1}',
                                       warmup_weight))
  if heads:
    summaries.append(tf.summary.scalar('warmup_weights/main',
                                       warmup_weights[-1]))

  train_op = make_train_op(optimizer, state.loss)

  if shared_head_acc is not None and model_config.shared_head_weight > 0.0:
    summaries.append(tf.summary.scalar('accuracy/shared_head_accuracy',
                                       shared_head_acc))

  return common.TrainState(
      train_op=train_op,
      step_initializer=tf.group(dataset.randomize_op,
                                test_dataset.randomize_op),
      large_summaries=[],
      small_summaries=summaries + [
          tf.summary.scalar('accuracy/accuracy', accuracy),
          tf.summary.scalar('accuracy/test_accuracy', test_accuracy),
          tf.summary.scalar('loss/loss', state.loss)
      ],
  )


def create_shared_feature_model(
    model_config,
    dataset,
    test_dataset,
    state,
    optim_config):
  """Creates an image feature extractor model for pre-training."""
  del test_dataset
  tf.logging.info('Building the model')
  global_step = tf.train.get_or_create_global_step()
  model = layerwise.build_model(model_config.cnn_model_name,
                                model_config=model_config)

  with tf.variable_scope('model'):
    weight_blocks = model.train(dataset.transformer_images,
                                dataset.transformer_labels,
                                mask=dataset.transformer_masks,
                                mask_random_samples=True,
                                enable_fe_dropout=True,
                                only_shared_feature=True)

  with tf.variable_scope('loss'):
    shared_head_loss, shared_head_acc = create_shared_head(
        weight_blocks.shared_features, dataset.transformer_real_classes,
        dataset.real_class_min, dataset.real_class_max)
    assert shared_head_loss is not None
    _, optimizer = make_optimizer(optim_config, global_step)
    state.loss = shared_head_loss

  train_op = make_train_op(optimizer, state.loss)

  return common.TrainState(
      train_op=train_op,
      step_initializer=tf.group(dataset.randomize_op),
      large_summaries=[],
      small_summaries=[
          tf.summary.scalar('loss/shared_head_loss', shared_head_loss),
          tf.summary.scalar('accuracy/shared_head_accuracy', shared_head_acc),
      ],
  )


def _cut_index(name):
  return name.rsplit(':', 1)[0]


def restore_shared_features():
  """Restores shared feature extractor variables from a checkpoint."""
  checkpoint = common_flags.RESTORE_SHARED_FEATURES_FROM.value
  if not checkpoint:
    return None
  all_vars = tf.trainable_variables()
  shared_vars = [v for v in all_vars
                 if v.name.find('model/shared_features') >= 0]
  shared_vars += [v for v in all_vars
                  if v.name.find('loss/shared_head') >= 0]
  var_values = util.load_variables(checkpoint,
                                   [_cut_index(v.name) for v in shared_vars])
  assign_ops = []
  for var in shared_vars:
    assign_ops.append(tf.assign(var, var_values[_cut_index(var.name)]))
  return tf.group(assign_ops)


def train(train_config,
          optimizer_config,
          dataset_config,
          test_dataset_config,
          layerwise_model_config):
  """Main function training the model."""
  state = train_lib.ModelState()
  tf.logging.info('Creating the dataset')
  dataset, dataset_state = train_lib.make_dataset(
      model_config=layerwise_model_config, data_config=dataset_config)
  test_dataset, _ = train_lib.make_dataset(
      model_config=layerwise_model_config, data_config=test_dataset_config,
      dataset_state=dataset_state)
  args = {'dataset': dataset, 'state': state, 'optim_config': optimizer_config,
          'test_dataset': test_dataset}

  if common_flags.PRETRAIN_SHARED_FEATURE.value:
    create_model = functools.partial(create_shared_feature_model,
                                     model_config=layerwise_model_config)
  else:
    create_model = functools.partial(create_layerwise_model,
                                     model_config=layerwise_model_config)

  tf.logging.info('Training')
  train_state = create_model(**args)
  with tf.Session():
    init_op = restore_shared_features()
    restored = common.init_training(train_state)
    if not restored and init_op is not None:
      sess = tf.get_default_session()
      sess.run(init_op)
    common.train(train_config, train_state)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.disable_eager_execution()

  for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

  train(train_config=make_train_config(),
        optimizer_config=make_optimizer_config(),
        dataset_config=make_dataset_config(),
        test_dataset_config=make_test_dataset_config(),
        layerwise_model_config=make_layerwise_model_config())


if __name__ == '__main__':
  app.run(main)
