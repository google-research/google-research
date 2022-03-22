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
"""Contrastive Learning training/eval code."""

import os

from absl import app
from absl import flags
import tensorflow.compat.v1 as tf
from tensorflow.compat.v1 import estimator as tf_estimator
import tensorflow.compat.v2 as tf2
import tf_slim as slim

from supcon import enums
from supcon import hparams as hparams_lib
from supcon import hparams_flags
from supcon import inputs
from supcon import losses
from supcon import models
from supcon import preprocessing
from supcon import utils

flags.DEFINE_string(
    'hparams', None,
    'A serialized hparams string representing the hyperparameters to use. If '
    'not set, fall back to using the individual hyperparameter flags defined '
    'in hparams_flags.py')
flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'train_then_eval'],
    'The mode for this job, either "train", "eval", or '
    '"train_then_eval".')
flags.DEFINE_string(
    'model_dir', '', 'Root of the tree containing all files for the current '
    'model.')
flags.DEFINE_string('master', None, 'Address of the TensorFlow runtime.')
flags.DEFINE_integer('summary_interval_steps', 100,
                     'Number of steps in between logging training summaries.')
flags.DEFINE_integer('save_interval_steps', 1000,
                     'Number of steps in between saving model checkpoints.')
flags.DEFINE_integer('max_checkpoints_to_keep', 5,
                     'Maximum number of recent checkpoints to keep.')
flags.DEFINE_float(
    'keep_checkpoint_interval_secs',
    60 * 60 * 1000 * 10,  # 10,000 hours
    'Number of seconds in between permanently retained checkpoints.')
flags.DEFINE_integer(
    'steps_per_loop', 1000,
    'Number of steps to execute on TPU before returning control to the '
    'coordinator. Checkpoints will be taken at least these many steps apart.')
flags.DEFINE_boolean('use_tpu', True, 'Whether this is running on a TPU.')
flags.DEFINE_integer('eval_interval_secs', 60, 'Time interval between evals.')
flags.DEFINE_string(
    'reference_ckpt', '',
    '[Optional] If set, attempt to initialize the model using the latest '
    'checkpoint in this directory.')
flags.DEFINE_string(
    'data_dir', None,
    'The directory that will be passed as the `data_dir` argument to '
    '`tfds.load`.')
tf.flags.DEFINE_string(
    'tpu_name', None,
    'The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 '
    'url.')
tf.flags.DEFINE_string(
    'tpu_zone', None,
    '[Optional] GCE zone where the Cloud TPU is located in. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')
tf.flags.DEFINE_string(
    'gcp_project', None,
    '[Optional] Project name for the Cloud TPU-enabled project. If not '
    'specified, we will attempt to automatically detect the GCE project from '
    'metadata.')

FLAGS = flags.FLAGS

# Learning rate hparam values are defined with respect to this batch size. The
# true learning rate will be scaled by batch_size/BASE_BATCH_SIZE.
BASE_BATCH_SIZE = 256


class ContrastiveTrainer:
  """Encapsulates the train, eval, and inference logic of a contrastive model.

  Upon construction of this class, the model graph is created. In train and eval
  mode, the loss computation graph is also created at construction time.

  Attrs:
    model_inputs: The inputs to the model. These should be Tensors with shape
      [batch_size, side_length, side_length, 3 * views] with values in range
      [-1, 1] and dtype tf.float32 or tf.bfloat16. Currently views must be 1 or
      2.
    labels: The labels corresponding to `model_inputs`. A Tensor of shape
      [batch_size] with integer dtype.
    hparams: A hparams.HParams instance, reflecting the hyperparameters of the
      model and its training.
    mode: An enums.ModelMode value.
    num_classes: The cardinality of the labelset, and also the number of output
      logits of the classification head.
    training_set_size: The number of samples in the training set.
    is_tpu: Whether this is running on a TPU.
  """

  def __init__(self,
               model_inputs,
               labels,
               train_global_batch_size,
               hparams,
               mode,
               num_classes,
               training_set_size,
               is_tpu=False):
    self.model_inputs = model_inputs
    self.labels = labels
    self.train_global_batch_size = train_global_batch_size
    self.hparams = hparams
    self.mode = mode
    assert isinstance(mode, enums.ModelMode)
    self.num_classes = num_classes
    self.training_set_size = training_set_size
    self.is_tpu = is_tpu
    self._summary_dict = {}

    if not self.inference:
      if tf.compat.dimension_at_index(self.model_inputs.shape, -1) != 6:
        raise ValueError(
            'Both train and eval modes must have 2 views provided, '
            'concatenated in the channels dimension.')

    self.data_format = ('channels_first' if not self.inference and
                        tf.config.list_physical_devices('GPU') else
                        'channels_last')

    if self.eval:
      self._summary_update_ops = []

    self.model = self._create_model()

    is_bn_train_mode = (
        # We intentionally run with batch norm in train mode for inference. We
        # call model() a second time with training=False for inference mode
        # below, and include both in the inference graph and SavedModel.
        not self.eval and (not FLAGS.reference_ckpt or
                           self.hparams.warm_start.batch_norm_in_train_mode))
    (self.unnormalized_embedding, self.normalized_embedding, self.projection,
     self.logits) = self._call_model(training=is_bn_train_mode)

    if self.inference:
      (self.unnormalized_embedding_eval, self.normalized_embedding_eval,
       self.projection_eval,
       self.logits_eval) = self._call_model(training=False)
      return

    self._encoder_weights = self._compute_weights('Encoder')
    self._projection_head_weights = self._compute_weights('ProjectionHead')
    self._classification_head_weights = self._compute_weights(
        'ClassificationHead')

    self.contrastive_loss = self._compute_contrastive_loss()
    self.cross_entropy_loss = self._compute_cross_entropy_loss()

  @property
  def train(self):
    return self.mode == enums.ModelMode.TRAIN

  @property
  def eval(self):
    return self.mode == enums.ModelMode.EVAL

  @property
  def inference(self):
    return self.mode == enums.ModelMode.INFERENCE

  def _add_scalar_summary(self, name, tensor):
    """Collects tensors that should be written as summaries in `host_call`."""
    self._summary_dict[name] = tensor

  def _create_model(self):
    """Creates the model, but does not build it or create variables.

    Returns:
      A callable Keras layer that implements the model architecture.
    """
    arch_hparams = self.hparams.architecture
    model = models.ContrastiveModel(
        architecture=arch_hparams.encoder_architecture,
        normalize_projection_head_input=(
            arch_hparams.normalize_projection_head_inputs),
        normalize_classification_head_input=(
            arch_hparams.normalize_classifier_inputs),
        stop_gradient_before_classification_head=(
            arch_hparams.stop_gradient_before_classification_head),
        stop_gradient_before_projection_head=(
            arch_hparams.stop_gradient_before_projection_head),
        encoder_kwargs={
            'depth': arch_hparams.encoder_depth,
            'width': arch_hparams.encoder_width,
            'first_conv_kernel_size': arch_hparams.first_conv_kernel_size,
            'first_conv_stride': arch_hparams.first_conv_stride,
            'data_format': self.data_format,
            'use_initial_max_pool': arch_hparams.use_initial_max_pool,
            'use_global_batch_norm': arch_hparams.use_global_batch_norm,
        },
        projection_head_kwargs={
            'feature_dims':
                arch_hparams.projection_head_layers,
            'normalize_output':
                True,
            'use_batch_norm':
                arch_hparams.projection_head_use_batch_norm,
            'use_batch_norm_beta':
                arch_hparams.projection_head_use_batch_norm_beta,
            'use_global_batch_norm':
                arch_hparams.use_global_batch_norm,
        },
        classification_head_kwargs={
            'num_classes':
                self.num_classes,
            'kernel_initializer': (tf.initializers.zeros()
                                   if arch_hparams.zero_initialize_classifier
                                   else tf.initializers.glorot_uniform)
        })

    return model

  def _call_model(self, training):
    """Passes data through the model.

    Manipulates the input data to get it ready for passing into the model,
    including applying some data augmentation that is more efficient to apply on
    the TPU than on the host. It then passes it into the model, which will first
    build the model and create its variables.

    Args:
      training: Whether the model should be run in training mode.

    Returns:
      A tuple of the model outputs (as Tensors):
      * unnormalized_embedding: The output of the encoder, not including
        normalization, which is sometimes applied before this gets passed into
        the projection and classification heads.
      * normalized_embedding: A normalized version of `unnormalized_embedding`.
      * projection: The output of the projection head.
      * logits: The output of the classification head.
    """
    with tf.name_scope('call_model'):
      model_inputs = self.model_inputs

      # In most cases, the data format NCHW instead of NHWC should be used for a
      # significant performance boost on GPU. NHWC should be used only if the
      # network needs to be run on CPU since the pooling operations are only
      # supported on NHWC. TPU uses XLA compiler to figure out best layout.
      if self.data_format == 'channels_first':
        model_inputs = tf.transpose(model_inputs, [0, 3, 1, 2])

      channels_index = 1 if self.data_format == 'channels_first' else -1
      inputs_are_multiview = tf.compat.dimension_value(
          model_inputs.shape[channels_index]) > 3
      if inputs_are_multiview:
        model_inputs = utils.stacked_multiview_image_channels_to_batch(
            model_inputs, self.data_format)

      # Perform blur augmentations here, since they're faster on TPU than CPU.
      if (self.hparams.input_data.preprocessing.augmentation_type in (
          enums.AugmentationType.SIMCLR,
          enums.AugmentationType.STACKED_RANDAUGMENT) and
          self.hparams.input_data.preprocessing.blur_probability > 0. and
          self.hparams.input_data.preprocessing.defer_blurring and self.train):
        model_inputs = preprocessing.batch_random_blur(
            model_inputs,
            tf.compat.dimension_value(model_inputs.shape[1]),
            blur_probability=(
                self.hparams.input_data.preprocessing.blur_probability))

      with tf.tpu.bfloat16_scope():
        model_outputs = self.model(model_inputs, training)

      if inputs_are_multiview:
        model_outputs = [
            utils.stacked_multiview_embeddings_to_channel(
                tf.cast(x, tf.float32)) if x is not None else x
            for x in model_outputs
        ]

      (unnormalized_embedding, normalized_embedding, projection,
       logits) = model_outputs

      if inputs_are_multiview:
        # If we keep everything in batch dimension then we don't need this. In
        # cross_entropy mode we should just stop generating the 2nd
        # augmentation.
        logits = tf.split(logits, 2, axis=1)[0]

      return unnormalized_embedding, normalized_embedding, projection, logits

  def _compute_cross_entropy_loss(self):
    """Computes and returns the cross-entropy loss on the logits."""
    with tf.name_scope('cross_entropy_loss'):
      one_hot_labels = tf.one_hot(self.labels, self.num_classes)
      cross_entropy = tf.losses.softmax_cross_entropy(
          logits=self.logits,
          onehot_labels=one_hot_labels,
          label_smoothing=(
              self.hparams.loss_all_stages.cross_entropy.label_smoothing),
          reduction=tf.losses.Reduction.NONE)

    if self.train:
      in_top_1 = tf.cast(
          tf.nn.in_top_k(self.logits, self.labels, 1), tf.float32)
      in_top_5 = tf.cast(
          tf.nn.in_top_k(self.logits, self.labels, 5), tf.float32)
      self._add_scalar_summary('top_1_accuracy', in_top_1)
      self._add_scalar_summary('top_5_accuracy', in_top_5)
      self._add_scalar_summary('loss/cross_entropy_loss', cross_entropy)
      cross_entropy = tf.reduce_mean(cross_entropy)

    return cross_entropy

  def _compute_contrastive_loss(self):
    """Computes and returns the contrastive loss on the projection."""
    with tf.name_scope('contrastive_loss'):
      contrastive_params = self.hparams.loss_all_stages.contrastive
      labels = (
          tf.one_hot(self.labels, self.num_classes)
          if contrastive_params.use_labels else None)
      projection = self.projection
      projection_view_1, projection_view_2 = tf.split(projection, 2, axis=-1)
      contrastive_loss = losses.contrastive_loss(
          tf.stack([projection_view_1, projection_view_2], axis=1),
          labels=labels,
          temperature=contrastive_params.temperature,
          contrast_mode=contrastive_params.contrast_mode,
          summation_location=contrastive_params.summation_location,
          denominator_mode=contrastive_params.denominator_mode,
          positives_cap=contrastive_params.positives_cap,
          scale_by_temperature=contrastive_params.scale_by_temperature)

    if self.train:
      self._add_scalar_summary('loss/contrastive_loss', contrastive_loss)
      contrastive_loss = tf.reduce_mean(contrastive_loss)

    return contrastive_loss

  def _compute_stage_weight_decay(self, stage_params, stage_name):
    """Computes and returns the weight decay loss for a single training stage.

    Args:
      stage_params: An instance of hparams.Stage.
      stage_name: A string name for this stage.

    Returns:
      A scalar Tensor representing the weight decay loss for this stage.
    """
    with tf.name_scope(f'{stage_name}_weight_decay'):
      # Don't use independent weight decay with LARS optimizer, since it handles
      # it internally.
      weight_decay_coeff = (
          stage_params.loss.weight_decay_coeff if
          stage_params.training.optimizer is not enums.Optimizer.LARS else 0.)
      weights = (
          self._encoder_weights *
          float(stage_params.loss.use_encoder_weight_decay) +
          self._projection_head_weights *
          float(stage_params.loss.use_projection_head_weight_decay) +
          self._classification_head_weights *
          float(stage_params.loss.use_classification_head_weight_decay))
      weight_decay_loss = weight_decay_coeff * weights
    with tf.name_scope(''):
      self._add_scalar_summary(f'weight_decay/{stage_name}_weight_decay_loss',
                               weight_decay_loss)
    return weight_decay_loss

  def _compute_weights(self, scope_name):
    """Computes the sum of the L2 norms of all kernel weights inside a scope."""

    def is_valid_weight(v):
      if (scope_name in v.name and 'batch_normalization' not in v.name and
          ('bias' not in v.name or
           self.hparams.loss_all_stages.include_bias_in_weight_decay)):
        return True
      return False

    with tf.name_scope(f'sum_{scope_name}_weights'):
      valid_weights = filter(is_valid_weight, tf.trainable_variables())
      sum_of_weights = tf.add_n([tf.nn.l2_loss(v) for v in valid_weights])

    self._add_scalar_summary(f'weight_decay/{scope_name}_weights',
                             sum_of_weights)
    return sum_of_weights

  def train_op(self):
    """Creates the Op for training this network.

    Computes learning rates, builds optimizers, and constructs the train ops to
    minimize the losses.

    Returns:
      A TensorFlow Op that will run one step of training when executed.
    """
    with tf.name_scope('train'):
      batch_size = self.train_global_batch_size
      steps_per_epoch = self.training_set_size / batch_size
      stage_1_epochs = self.hparams.stage_1.training.train_epochs
      stage_1_steps = int(stage_1_epochs * steps_per_epoch)
      stage_2_epochs = self.hparams.stage_2.training.train_epochs
      global_step = tf.train.get_or_create_global_step()
      stage_1_indicator = tf.math.less(global_step, stage_1_steps)
      stage_2_indicator = tf.math.logical_not(stage_1_indicator)

      def stage_learning_rate(stage_training_params, start_epoch, end_epoch):
        schedule_kwargs = {}
        if (stage_training_params.learning_rate_decay in (
            enums.DecayType.PIECEWISE_LINEAR, enums.DecayType.EXPONENTIAL)):
          schedule_kwargs['decay_rate'] = stage_training_params.decay_rate
          if (stage_training_params.learning_rate_decay ==
              enums.DecayType.PIECEWISE_LINEAR):
            schedule_kwargs['boundary_epochs'] = (
                stage_training_params.decay_boundary_epochs)
          if (stage_training_params.learning_rate_decay ==
              enums.DecayType.EXPONENTIAL):
            schedule_kwargs['epochs_per_decay'] = (
                stage_training_params.epochs_per_decay)

        return utils.build_learning_rate_schedule(
            learning_rate=(stage_training_params.base_learning_rate *
                           (batch_size / BASE_BATCH_SIZE)),
            decay_type=stage_training_params.learning_rate_decay,
            warmup_start_epoch=start_epoch,
            max_learning_rate_epoch=(
                start_epoch +
                stage_training_params.learning_rate_warmup_epochs),
            decay_end_epoch=end_epoch,
            global_step=global_step,
            steps_per_epoch=steps_per_epoch,
            **schedule_kwargs)

      stage_1_learning_rate = stage_learning_rate(
          self.hparams.stage_1.training,
          start_epoch=0,
          end_epoch=stage_1_epochs) * tf.cast(stage_1_indicator, tf.float32)
      stage_2_learning_rate = stage_learning_rate(
          self.hparams.stage_2.training,
          start_epoch=stage_1_epochs,
          end_epoch=stage_1_epochs + stage_2_epochs) * tf.cast(
              stage_2_indicator, tf.float32)

      def stage_optimizer(stage_learning_rate, stage_params, stage_name):
        lars_exclude_from_weight_decay = ['batch_normalization']
        if not self.hparams.loss_all_stages.include_bias_in_weight_decay:
          lars_exclude_from_weight_decay.append('bias')
        if not stage_params.loss.use_encoder_weight_decay:
          lars_exclude_from_weight_decay.append('Encoder')
        if not stage_params.loss.use_projection_head_weight_decay:
          lars_exclude_from_weight_decay.append('ProjectionHead')
        if not stage_params.loss.use_classification_head_weight_decay:
          lars_exclude_from_weight_decay.append('ClassificationHead')

        return utils.build_optimizer(
            stage_learning_rate,
            optimizer_type=stage_params.training.optimizer,
            lars_weight_decay=stage_params.loss.weight_decay_coeff,
            lars_exclude_from_weight_decay=lars_exclude_from_weight_decay,
            epsilon=stage_params.training.rmsprop_epsilon,
            is_tpu=self.is_tpu,
            name=stage_name)

      stage_1_optimizer = stage_optimizer(stage_1_learning_rate,
                                          self.hparams.stage_1, 'stage1')
      stage_2_optimizer = stage_optimizer(stage_2_learning_rate,
                                          self.hparams.stage_2, 'stage2')

      def stage_loss(stage_params, stage_name):
        return (
            stage_params.loss.contrastive_weight * self.contrastive_loss +
            stage_params.loss.cross_entropy_weight * self.cross_entropy_loss +
            self._compute_stage_weight_decay(stage_params, stage_name))

      stage_1_loss = stage_loss(self.hparams.stage_1, 'stage1')
      stage_2_loss = stage_loss(self.hparams.stage_2, 'stage2')

      def stage_1_train_op():
        return utils.create_train_op(
            stage_1_loss,
            stage_1_optimizer,
            update_ops=(None if
                        self.hparams.stage_1.training.update_encoder_batch_norm
                        else []))

      def stage_2_train_op():
        return utils.create_train_op(
            stage_2_loss,
            stage_2_optimizer,
            update_ops=(None if
                        self.hparams.stage_2.training.update_encoder_batch_norm
                        else []))

      train_op = tf.cond(stage_1_indicator, stage_1_train_op, stage_2_train_op)

    self._add_scalar_summary('stage_1_learning_rate', stage_1_learning_rate)
    self._add_scalar_summary('stage_2_learning_rate', stage_2_learning_rate)
    self._add_scalar_summary('current_epoch',
                             tf.cast(global_step, tf.float32) / steps_per_epoch)

    return train_op

  def host_call(self, summary_dir):
    """Creates a host call to write summaries."""

    # Ensure that all host_call inputs have batch dimensions, since they get
    # concatenated from all cores along the batch dimension.
    summary_dict = {
        k: tf.expand_dims(v, axis=0) if v.shape.rank == 0 else v
        for k, v in self._summary_dict.items()
    }

    # Pass in the global step, since otherwise we might use a stale copy of the
    # variable from the host.
    global_step_key = 'global_step_is_not_a_summary'
    summary_dict[global_step_key] = tf.expand_dims(
        tf.train.get_or_create_global_step(), axis=0)

    def host_call_fn(**kwargs):
      step = kwargs[global_step_key][0]
      del kwargs[global_step_key]
      writer = tf2.summary.create_file_writer(summary_dir, max_queue=1000)
      always_record = tf2.summary.record_if(True)
      with writer.as_default(), always_record:
        for name, scalar in kwargs.items():
          tf2.summary.scalar(name, tf.reduce_mean(scalar), step)
      return tf.summary.all_v2_summary_ops()

    return host_call_fn, summary_dict

  def eval_metrics(self):
    """Returns eval metric_fn and metrics."""

    def metric_fn(logits, labels, contrastive_loss, cross_entropy_loss):
      metrics = {}
      in_top_1 = tf.cast(tf.nn.in_top_k(logits, labels, 1), tf.float32)
      in_top_5 = tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.float32)
      metrics['top_1_accuracy'] = tf.metrics.mean(in_top_1)
      metrics['top_5_accuracy'] = tf.metrics.mean(in_top_5)
      metrics['loss/contrastive_loss'] = tf.metrics.mean(contrastive_loss)
      metrics['loss/cross_entropy_loss'] = tf.metrics.mean(cross_entropy_loss)
      return metrics

    metrics = {
        'logits': self.logits,
        'labels': self.labels,
        'contrastive_loss': self.contrastive_loss,
        'cross_entropy_loss': self.cross_entropy_loss,
    }

    return metric_fn, metrics

  def signature_def_map(self):
    """Returns a SignatureDef map that can be used to produce SavedModels."""
    signature_def_map = {}
    signature_def_map['contrastive_train'] = {
        'embeddings': self.normalized_embedding,
        'unnormalized_embeddings': self.unnormalized_embedding,
        'projection': self.projection,
        'logits': self.logits,
    }
    signature_def_map['contrastive_eval'] = {
        'embeddings': self.normalized_embedding_eval,
        'unnormalized_embeddings': self.unnormalized_embedding_eval,
        'projection': self.projection_eval,
        'logits': self.logits_eval,
    }
    return signature_def_map

  def scaffold_fn(self):
    """Creates a function that produces a tf.train.Scaffold for custom init.

    When appropriate, it restores all or some of the weights from a checkpoint
    at model initialization.

    Returns:
      A function that produces a tf.train.Scaffold.
    """

    def var_matches_patterns(var, patterns):
      return any(pattern in var.name for pattern in patterns)

    def scaffold_fn():
      """Scaffold function."""
      warm_start_hparams = self.hparams.warm_start
      if FLAGS.reference_ckpt:
        with tf.name_scope('warm_start'):
          include_pattern_list = []
          if warm_start_hparams.warm_start_encoder:
            include_pattern_list.append('ContrastiveModel/Encoder')
          if warm_start_hparams.warm_start_projection_head:
            include_pattern_list.append('ContrastiveModel/ProjectionHead')
          if warm_start_hparams.warm_start_classifier:
            include_pattern_list.append('ContrastiveModel/ClassificationHead')
          # This needs to be updated if new optimizers are added.
          exclude_pattern_list = [
              'Optimizer', 'Momentum', 'RMSProp', 'LARSOptimizer'
          ]
          variables = filter(
              lambda v: var_matches_patterns(v, include_pattern_list),
              tf.global_variables())
          variables = filter(
              lambda v: not var_matches_patterns(v, exclude_pattern_list),
              variables)
          var_init_fn = slim.assign_from_checkpoint_fn(
              tf.train.latest_checkpoint(FLAGS.reference_ckpt),
              list(variables),
              ignore_missing_vars=(
                  warm_start_hparams.ignore_missing_checkpoint_vars),
              reshape_variables=True)

      def init_fn(scaffold, sess):
        del scaffold  # unused.

        if FLAGS.reference_ckpt:
          var_init_fn(sess)

      return tf.train.Scaffold(init_fn=init_fn)

    return scaffold_fn


def model_fn(features, labels, mode, params):
  """Contrastive model function."""

  model_mode = utils.estimator_mode_to_model_mode(mode)
  hparams = params['hparams']

  trainer = ContrastiveTrainer(
      model_inputs=features,
      labels=labels,
      train_global_batch_size=hparams.bs,
      hparams=hparams,
      mode=model_mode,
      num_classes=inputs.get_num_classes(hparams),
      training_set_size=inputs.get_num_train_images(hparams),
      is_tpu=params['use_tpu'])

  if mode == tf_estimator.ModeKeys.PREDICT:
    predictions_map = trainer.signature_def_map()
    exports = {
        k: tf_estimator.export.PredictOutput(v)
        for k, v in predictions_map.items()
    }
    # Export a default SignatureDef to keep the API happy.
    exports[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = (
        exports['contrastive_eval'])
    spec = tf_estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        predictions=predictions_map['contrastive_eval'],
        export_outputs=exports)
    return spec

  # We directly write summaries for the relevant losses, so just hard-code a
  # dummy value to keep the Estimator API happy.
  loss = tf.constant(0.)

  if mode == tf_estimator.ModeKeys.EVAL:
    spec = tf_estimator.tpu.TPUEstimatorSpec(
        mode=mode, loss=loss, eval_metrics=trainer.eval_metrics())
    return spec
  else:  # TRAIN
    spec = tf_estimator.tpu.TPUEstimatorSpec(
        mode=mode,
        train_op=trainer.train_op(),
        loss=loss,
        scaffold_fn=trainer.scaffold_fn(),
        host_call=trainer.host_call(FLAGS.model_dir))
    return spec


def main(_):
  tf.disable_v2_behavior()
  tf.enable_resource_variables()

  if FLAGS.hparams is None:
    hparams = hparams_flags.hparams_from_flags()
  else:
    hparams = hparams_lib.HParams(FLAGS.hparams)

  cluster = None
  if FLAGS.use_tpu and FLAGS.master is None:
    if FLAGS.tpu_name:
      cluster = tf.distribute.cluster_resolver.TPUClusterResolver(
          FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)
    else:
      cluster = tf.distribute.cluster_resolver.TPUClusterResolver()
      tf.config.experimental_connect_to_cluster(cluster)
      tf.tpu.experimental.initialize_tpu_system(cluster)

  session_config = tf.ConfigProto()
  # Workaround for https://github.com/tensorflow/tensorflow/issues/26411 where
  # convolutions (used in blurring) get confused about data-format when used
  # inside a tf.data pipeline that is run on GPU.
  if (tf.test.is_built_with_cuda() and
      not hparams.input_data.preprocessing.defer_blurring):
    # RewriterConfig.OFF = 2
    session_config.graph_options.rewrite_options.layout_optimizer = 2
  run_config = tf_estimator.tpu.RunConfig(
      master=FLAGS.master,
      cluster=cluster,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.save_interval_steps,
      keep_checkpoint_max=FLAGS.max_checkpoints_to_keep,
      keep_checkpoint_every_n_hours=(FLAGS.keep_checkpoint_interval_secs /
                                     (60.0 * 60.0)),
      log_step_count_steps=100,
      session_config=session_config,
      tpu_config=tf_estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.steps_per_loop,
          per_host_input_for_training=True,
          experimental_host_call_every_n_steps=FLAGS.summary_interval_steps,
          tpu_job_name='train_tpu_worker' if FLAGS.mode == 'train' else None,
          eval_training_input_configuration=(
              tf_estimator.tpu.InputPipelineConfig.SLICED if FLAGS.use_tpu else
              tf_estimator.tpu.InputPipelineConfig.PER_HOST_V1)))
  params = {
      'hparams': hparams,
      'use_tpu': FLAGS.use_tpu,
      'data_dir': FLAGS.data_dir,
  }
  estimator = tf_estimator.tpu.TPUEstimator(
      model_fn=model_fn,
      use_tpu=FLAGS.use_tpu,
      config=run_config,
      params=params,
      train_batch_size=hparams.bs,
      eval_batch_size=hparams.eval.batch_size)

  if hparams.input_data.input_fn not in dir(inputs):
    raise ValueError('Unknown input_fn: {hparams.input_data.input_fn}')
  input_fn = getattr(inputs, hparams.input_data.input_fn)

  training_set_size = inputs.get_num_train_images(hparams)
  steps_per_epoch = training_set_size / hparams.bs
  stage_1_epochs = hparams.stage_1.training.train_epochs
  stage_2_epochs = hparams.stage_2.training.train_epochs
  total_steps = int((stage_1_epochs + stage_2_epochs) * steps_per_epoch)

  num_eval_examples = inputs.get_num_eval_images(hparams)
  eval_steps = num_eval_examples // hparams.eval.batch_size

  if FLAGS.mode == 'eval':
    for ckpt_str in tf.train.checkpoints_iterator(
        FLAGS.model_dir,
        min_interval_secs=FLAGS.eval_interval_secs,
        timeout=60 * 60):
      result = estimator.evaluate(
          input_fn=input_fn, checkpoint_path=ckpt_str, steps=eval_steps)
      estimator.export_saved_model(
          os.path.join(FLAGS.model_dir, 'exports'),
          lambda: input_fn(tf_estimator.ModeKeys.PREDICT, params),
          checkpoint_path=ckpt_str)
      if result['global_step'] >= total_steps:
        return
  else:  # 'train' or 'train_then_eval'.
    estimator.train(input_fn=input_fn, max_steps=total_steps)
    if FLAGS.mode == 'train_then_eval':
      result = estimator.evaluate(input_fn=input_fn, steps=eval_steps)
      estimator.export_saved_model(
          os.path.join(FLAGS.model_dir, 'exports'),
          lambda: input_fn(tf_estimator.ModeKeys.PREDICT, params))


if __name__ == '__main__':
  app.run(main)
