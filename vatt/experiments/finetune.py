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

# Lint as: python3
"""Main experiments script for pre-training MMV model OR vatt."""

from absl import logging
import numpy as np
import tensorflow as tf

from vatt.data import dataloaders
from vatt.data import processing
from vatt.experiments import base
from vatt.modeling.backbones.audio import factory as aud_factory
from vatt.modeling.backbones.video import factory as vid_factory
from vatt.utils.eval import measures
from vatt.utils.train import optimizers
from vatt.utils.train import restore
from vatt.utils.train import schedules


FeatureNames = dataloaders.FeatureNames
REF_FPS = dataloaders.REF_FPS  # reference fps used during training
REF_SR = dataloaders.REF_SR  # reference sampling rate used during training


class BaseExecutor(base.Executor):
  """Constructs the necessary modules to perform fine-tuning."""

  def __init__(self, params):
    strategy = base.create_strategy(params.strategy_config)
    data = self.construct_data(params)
    with strategy.scope():
      model = self.construct_model(params)
      metrics = base.get_metrics('classification')

    super(BaseExecutor, self).__init__(model=model,
                                       data=data,
                                       params=params,
                                       strategy=strategy,
                                       metrics=metrics)
    self._manual_restore = True

  def prepare_inputs(self, inputs):
    if self.params.mode == 'train':
      return self.prepare_train_inputs(inputs)
    elif self.params.mode == 'eval':
      return self.prepare_eval_inputs(inputs)
    else:
      raise ValueError('Invalid mode!')

  def prepare_train_inputs(self, inputs):
    raise NotImplementedError

  def prepare_eval_inputs(self, inputs):
    raise NotImplementedError

  def construct_data(self, params):

    if params.mode == 'train':
      dataset_id = params.train.input.name
      is_aud_cls = params.train.input.name in dataloaders.AUD_CLS_DS
      if is_aud_cls:
        data = dataloaders.AudioFineTuneLoader(
            dataset_id=dataset_id,
            params=params,
            )
      else:
        data = dataloaders.VisionFineTuneLoader(
            dataset_id=dataset_id,
            params=params,
            )

    elif params.mode == 'eval':
      dataset_id = params.eval.input.name
      is_aud_cls = params.eval.input.name in dataloaders.AUD_CLS_DS
      if is_aud_cls:
        data = dataloaders.AudioEvalLoader(
            dataset_id=dataset_id,
            params=params,
            )
      else:
        data = dataloaders.VisionEvalLoader(
            dataset_id=dataset_id,
            params=params,
            )

    else:
      raise ValueError('Invalid mode!')

    return [data]

  def create_replicated_train_step(self, strategy, model):
    metrics = self.metrics
    optimizer = model.optimizer
    gradient_clip_norm = self.params.train.gradient_clip_norm
    gradient_clip_norm_cls = self.params.train.gradient_clip_norm_cls

    weights_backbone = []
    weights_cls = []
    for w in model.trainable_variables:
      if 'classification' in w.name:
        weights_cls.append(w)
      else:
        weights_backbone.append(w)

    @tf.function
    def _replicated_step(inputs):
      """Replicated training step."""
      replicator = base.Replicator(
          tf.distribute.get_replica_context()
          )
      inputs, labels = self.prepare_inputs(inputs)

      outputs = model(inputs, training=True)

      # update accuracy metrics
      for m in metrics.values():
        m.update_state(labels['one_hot'], outputs['probabilities'])

      # calculate losses
      all_losses = model.loss_fn(labels=labels,
                                 outputs=outputs,
                                 replicator=replicator)
      losses = {}
      for k, v in all_losses.items():
        losses[k] = tf.reduce_mean(v)
        losses[k] = tf.where(tf.math.is_nan(v), 0., v)

      per_replica_loss = losses['total_loss'] / strategy.num_replicas_in_sync

      # apply gradients
      grads_backbone = tf.gradients(per_replica_loss, weights_backbone)
      grads_cls = tf.gradients(per_replica_loss, weights_cls)

      if gradient_clip_norm > 0:
        grads_backbone, _ = tf.clip_by_global_norm(grads_backbone,
                                                   gradient_clip_norm)

      if gradient_clip_norm_cls > 0:
        grads_cls, _ = tf.clip_by_global_norm(grads_cls,
                                              gradient_clip_norm_cls)

      grads = grads_backbone + grads_cls
      weights = weights_backbone + weights_cls

      optimizer.apply_gradients(zip(grads, weights))
      return losses

    return _replicated_step

  def partial_restore(self, params, model):
    """Restore backbone weights from pretrained model checkpoint."""

    ckpt_path = params.checkpoint_path
    if ckpt_path is None:
      logging.info('No pretrained checkpoint provided, '
                   'training with randomly initialized weights.')
      return

    skipped = restore.assign_weight_from_ckpt(model, ckpt_path)
    logging.info(
        'Successfully restored from pretrained checkpoint, while skipping: %s',
        skipped,
        )

  def construct_model(self, params):
    """Build models for train/eval."""

    num_test_samples = 1
    if params.mode == 'train':
      input_params = params.train.input
      ds_name = input_params.name
      is_vid_cls = ds_name in dataloaders.VID_CLS_DS
      is_img_cls = ds_name in dataloaders.IMG_CLS_DS
      is_aud_cls = ds_name in dataloaders.AUD_CLS_DS

    elif params.mode == 'eval':
      input_params = params.eval.input
      ds_name = input_params.name
      is_vid_cls = ds_name in dataloaders.VID_CLS_DS
      is_img_cls = ds_name in dataloaders.IMG_CLS_DS
      is_aud_cls = ds_name in dataloaders.AUD_CLS_DS
      if not is_img_cls:
        num_test_samples = params.eval.input.num_windows_test
        if params.eval.input.multi_crop and not is_aud_cls:
          num_test_samples *= 3

    else:
      raise ValueError('Invalid mode!')

    if is_aud_cls:
      input_shape = processing.get_audio_shape(input_params, REF_FPS, REF_SR)
    elif is_vid_cls:
      space_to_depth = input_params.space_to_depth
      input_shape = processing.get_video_shape(
          input_params, is_space_to_depth=space_to_depth)
    elif is_img_cls:
      input_shape = processing.get_video_shape(input_params)

    if is_img_cls:
      input_shape[0] = params.model_config.temporal_patch_size

    num_classes = dataloaders.CLS_DS[ds_name]['num_classes']

    model_kwargs = {'num_classes': num_classes,
                    'num_test_samples': num_test_samples}
    if is_aud_cls:
      inputs = {'audio': tf.keras.Input(shape=input_shape)}
      model_factory = aud_factory
    else:
      inputs = {'images': tf.keras.Input(shape=input_shape)}
      model_factory = vid_factory

    model = model_factory.build_model(params=params.model_config,
                                      override_params=model_kwargs,
                                      mode='predict')
    outputs = model(inputs, None)
    keras_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    keras_model.loss_fn = model.loss_fn

    if params.mode == 'train':
      self.partial_restore(params, keras_model)

    logging.info('Number of parameters in model: %f M.',
                 keras_model.count_params() / 10.**6)

    learning_rate = schedules.get_learning_rate(
        params.train.optimizer.learning_rate
        )
    keras_model.optimizer = optimizers.get_optimizer(
        learning_rate, params.train.optimizer
        )
    return keras_model


class VisionExecutor(BaseExecutor):
  """Constructs the necessary modules to perform vision fine-tuning."""

  def prepare_train_inputs(self, inputs):
    """Prepares inputs on device to be fed to model in train mode."""
    params = self.params.train.input

    images = inputs[FeatureNames.VISION]
    labels_onehot = inputs[FeatureNames.LABEL_INDEX]

    if params.linearize_vision:
      img_shape = [params.frame_size, params.frame_size, 3]
      if params.name in dataloaders.VID_CLS_DS:
        space_to_depth = params.space_to_depth
        img_shape = processing.get_video_shape(
            params, is_space_to_depth=space_to_depth)
      else:
        img_shape = [1] + img_shape

      img_shape = [-1] + img_shape
      images = tf.reshape(images, img_shape)

    if params.name in dataloaders.IMG_CLS_DS:
      num_replica = self.params.model_config.temporal_patch_size
      images = tf.tile(images, [1, num_replica, 1, 1, 1])

    labels = {'one_hot': labels_onehot}

    inputs = {'images': images}
    return inputs, labels

  def prepare_eval_inputs(self, inputs):
    """Prepares inputs on device to be fed to model in eval mode."""
    params = self.params.eval.input
    images = inputs[FeatureNames.VISION]
    labels_onehot = inputs[FeatureNames.LABEL_INDEX]

    if params.linearize_vision:
      img_shape = [params.frame_size, params.frame_size, 3]
      if params.name in dataloaders.VID_CLS_DS:
        space_to_depth = params.space_to_depth
        img_shape = processing.get_video_shape(
            params, is_space_to_depth=space_to_depth)
      else:
        img_shape = [1] + img_shape

      img_shape = [-1] + img_shape
      images = tf.reshape(images, img_shape)

    if params.name in dataloaders.IMG_CLS_DS:
      num_replica = self.params.model_config.temporal_patch_size
      images = tf.tile(images, [1, num_replica, 1, 1, 1])

    labels = {'one_hot': labels_onehot}

    inputs = {'images': images}

    return inputs, labels


class AudioExecutor(BaseExecutor):
  """Constructs the necessary modules to perform audio fine-tuning."""

  def __init__(self, params):
    super(AudioExecutor, self).__init__(params=params)
    with self.strategy.scope():
      self.metrics = base.get_metrics('ml_classification')

  def prepare_train_inputs(self, inputs):
    """Prepares inputs on device to be fed to model in train mode."""

    params = self.params.train.input

    if params.raw_audio:
      audio = inputs[FeatureNames.AUDIO][:, :, None]
    else:
      audio = inputs[FeatureNames.AUDIO_MEL]

    labels_onehot = inputs[FeatureNames.LABEL_INDEX]

    labels = {'one_hot': labels_onehot}
    inputs = {'audio': audio}

    return inputs, labels

  def prepare_eval_inputs(self, inputs):
    """Prepares inputs on device to be fed to model in eval mode."""

    params = self.params.eval.input

    if params.raw_audio:
      audio = inputs[FeatureNames.AUDIO][:, :, None]
    else:
      audio = inputs[FeatureNames.AUDIO_MEL]
    labels_onehot = inputs[FeatureNames.LABEL_INDEX]

    labels = {'one_hot': labels_onehot}

    inputs = {'audio': audio}

    return inputs, labels

  def evaluation_loop(self):
    """Iterates over data and returns the aggregated metrics."""

    # construct the dataloaders and data iterators
    eval_dataloaders = self.get_dataloaders(self.data, self.strategy)
    assert len(eval_dataloaders) == 1, 'Evaluation only accepts one dataloader!'
    iterator = eval_dataloaders[0]['iterator']

    def outputs_filter(outputs):
      labels = outputs['labels']
      outputs = {'true': labels['one_hot'],
                 'pred': outputs['probabilities']}
      return outputs

    outputs, cnt = self.infer(iterator=iterator, outputs_filter=outputs_filter)

    # aggregate all outputs
    for k in outputs:
      outputs[k] = np.concatenate(outputs[k], axis=0)  # (n_samples, n_classes)

    metrics = measures.compute_map_auc_dprime(outputs['pred'],
                                              outputs['true'],
                                              'sklearn_')
    logging.info('Total evaluation steps: [%d]', cnt)
    logging.info('Evaluation metric = %r', metrics)

    return metrics


def get_executor(params):
  """Returns an instance of the Executor depending on the setting."""

  if params.mode == 'train':
    input_params = params.train.input
  else:
    input_params = params.eval.input

  is_aud_cls = input_params.name in dataloaders.AUD_CLS_DS
  if is_aud_cls:
    return AudioExecutor(params=params)
  else:
    return VisionExecutor(params=params)

