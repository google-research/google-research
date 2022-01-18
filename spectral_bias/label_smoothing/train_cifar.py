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

"""Code for training on Cifar10.

   The code builds on the codebase for the paper:
   `AutoAugment: Learning Augmentation Policies from Data`.
   The original code is publicly available here:
   https://github.com/tensorflow/models/blob/238922e98dd0e8254b5c0921b241a1f5a151782f/research/autoaugment

   The CIFAR10 data can be downloaded from:
   https://www.cs.toronto.edu/~kriz/cifar.html.
"""
import contextlib
import os
import time
import data_utils
import helper_utils
import numpy as np
from shake_drop import build_shake_drop_model
from shake_shake import build_shake_shake_model
import six.moves
from six.moves import range
from six.moves import zip
import tensorflow as tf
from tensorflow.contrib import training as contrib_training
from wrn import build_wrn_model

tf.flags.DEFINE_string(
    'model_name', 'wrn_32',
    'The type of model that will be trained. Options are: '
    'wrn_32, wrn_160, shake_shake_32, shake_shake_96, shake_shake_112, '
    'pyramid_net')
tf.flags.DEFINE_string(
    'checkpoint_dir', '/tmp/training',
    'Training Directory where checkpoints and data are '
    'saved.')
tf.flags.DEFINE_integer('dummy_f', 1,
                        'Used to index runs with the same hparams.')
tf.flags.DEFINE_string('dataset', 'cifar10',
                       'Dataset to train with. Either cifar10 or cifar100')
tf.flags.DEFINE_string('extra_dataset', 'cifar10_1',
                       'Extra test dataset. Defaults to cifar10_1.'
                       'Pass None for no extra test set.')
tf.flags.DEFINE_float('frequency', 0.038, 'Frequency of label smoothing noise.')
tf.flags.DEFINE_float('amplitude', 0.0, 'Per-batch average label smoothing'
                      'magnitude.')
tf.flags.DEFINE_integer('use_cpu', 0, '1 if use CPU, else GPU.')
tf.flags.DEFINE_integer('apply_cutout', 1,
                        'Whether to apply cutout after the autoaugment layers.')
tf.flags.DEFINE_string('augment_type', 'none',
                       'Type of data augmentation used: autoaugment, random,'
                       'none, mixup, or image_freq.')
tf.flags.DEFINE_float('mixup_alpha', 0.1, 'Parameter controlling how mixed the'
                      'mixup augmentation is. Paper recommends 0.1 to 0.4'
                      '(smaller alpha -> weaker mixup).')
tf.flags.DEFINE_integer('num_augmentation_layers', 3,
                        'Number of augmentation ops applied in a subpolicy. '
                        'This is only used if `random` augment_type is '
                        'selected.')
tf.flags.DEFINE_integer('num_epochs', 200,
                        'Number of training epochs.')
tf.flags.DEFINE_integer('train_size', 50000,
                        'Number of training samples to train on.')
tf.flags.DEFINE_integer('apply_flip_crop', 1,
                        'Whether to apply flip and crop to input images')
tf.flags.DEFINE_float('augmentation_magnitude', 4.0, 'Global magnitude'
                      'for RandAugment.')
tf.flags.DEFINE_float('augmentation_probability', 1.0, 'Global probabily for'
                      'random augmentation.')
tf.flags.DEFINE_float('freq_augment_amplitude', 0.5, 'Amount of label smoothing'
                      'applied to images that are f or 1/f augmented.')
tf.flags.DEFINE_float('freq_augment_ffrac', 0.5, 'Fraction of frequency'
                      'augmented images to add power spectrum f to, rather than'
                      '1/f.')
tf.flags.DEFINE_string('logdir', None, 'TensorBoard log directory. If None,'
                       'Tensorboard summaries will not be written.')
tf.flags.DEFINE_float('lr', 0.1, 'Global magnitude'
                      'for RandAugment.')
tf.flags.DEFINE_float('weight_decay_rate', 5e-4, 'Global magnitude'
                      'for RandAugment.')
tf.flags.DEFINE_float('is_gan_data', 0, 'Global magnitude'
                      'for RandAugment.')
tf.flags.DEFINE_integer('use_batchnorm', 1, '1 if use batchnorm, else nothing.')
tf.flags.DEFINE_integer('use_fixup', 0, '1 if use fixup, else nothing.')
tf.flags.DEFINE_integer('use_gamma_swish', 0,
                        'Number of training epochs.')
tf.flags.DEFINE_float('init_beta', 0.0, 'Global magnitude'
                      'for RandAugment.')
tf.flags.DEFINE_float('init_gamma', 2.0, 'Global magnitude'
                      'for RandAugment.')
tf.flags.DEFINE_string('noise_type', 'radial', 'Type of label smoothing'
                       'noise, one of: radial, fourier, random, f, 1/f,'
                       'uniform. Radial computes a radial wave based on image'
                       'norm. Fourier computes a sine wave along a Fourier'
                       'basis image direction through the input space. Random'
                       'computes a sine wave along a random direction through'
                       'the input space. f and 1/f are like Fourier, but'
                       'combining all Fourier basis images to have the desired'
                       'power spectrum.')
tf.flags.DEFINE_float('spatial_frequency', 2.0, 'Spatial frequency norm of the'
                      'Fourier basis image direction used for "fourier" label'
                      'smoothing noise.')
tf.flags.DEFINE_integer('noise_seed', 0, 'Seed for choosing random noise'
                        'direction.')
tf.flags.DEFINE_integer('noise_class', -1, 'Only apply label smoothing to'
                        'examples from this class. -1 denotes that label'
                        'smoothing is applied to all classes.')
tf.flags.DEFINE_float('max_accuracy', 1.0, 'Stop training as soon as this'
                      'validation accuracy is achieved.')
tf.flags.DEFINE_float('min_loss', 0.0, 'Stop training as soon as this'
                      'training loss is achieved.')
tf.flags.DEFINE_string('teacher_model', None, 'Path to trained teacher'
                       'model, for distillation.')
tf.flags.DEFINE_float('distillation_alpha', 0.0, 'Relative weighting of the'
                      'original labels compared to the teacher labels for'
                      'distillation.')
tf.flags.DEFINE_bool('normalize_amplitude', True, 'Whether or not the actual'
                     'amplitude of label smoothing wave should be batch'
                     'normalized.')
tf.flags.DEFINE_integer('ckpt_every', 10000, 'How often to save extra'
                        'checkpoints during training.')

FLAGS = tf.flags.FLAGS


def build_model(inputs, num_classes, is_training, hparams):
  """Constructs the vision model being trained/evaled.

  Args:
    inputs: input features/images being fed to the image model build built.
    num_classes: number of output classes being predicted.
    is_training: is the model training or not.
    hparams: additional hyperparameters associated with the image model.

  Returns:
    The logits of the image model.
  """
  scopes = helper_utils.setup_arg_scopes(is_training, hparams)
  with helper_utils.nested(*scopes):
    if hparams.model_name == 'pyramid_net':
      logits, hiddens = build_shake_drop_model(
          inputs, num_classes, is_training)
    elif hparams.model_name == 'wrn':
      logits, hiddens = build_wrn_model(
          inputs, num_classes, hparams)
    elif hparams.model_name == 'shake_shake':
      logits, hiddens = build_shake_shake_model(
          inputs, num_classes, hparams, is_training)
    else:
      print(f'unrecognized hparams.model_name: {hparams.model_name}')
      assert 0
  return logits, hiddens


class CifarModel(object):
  """Builds an image model for Cifar10/Cifar100."""

  def __init__(self, hparams):
    self.hparams = hparams

  def build(self, mode):
    """Construct the cifar model."""
    assert mode in ['train', 'eval']
    self.mode = mode
    self._setup_misc(mode)
    self._setup_images_and_labels()
    self._build_graph(self.images, self.labels, mode)

    self.init = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())

  def _setup_misc(self, mode):
    """Sets up miscellaneous in the cifar model constructor."""
    self.lr_rate_ph = tf.Variable(0.0, name='lrn_rate', trainable=False)
    self.reuse = None if (mode == 'train') else True
    self.batch_size = self.hparams.batch_size
    if mode == 'eval':
      self.batch_size = 256

  def _setup_images_and_labels(self):
    """Sets up image and label placeholders for the cifar model."""
    if FLAGS.dataset in ['cifar10', 'svhn']:
      self.num_classes = 10
    else:
      self.num_classes = 100
    self.images = tf.placeholder(tf.float32, [self.batch_size, 32, 32, 3])
    self.labels = tf.placeholder(tf.float32,
                                 [self.batch_size, self.num_classes])

  def assign_epoch(self, session, epoch_value):
    session.run(self._epoch_update, feed_dict={self._new_epoch: epoch_value})

  def compute_flops_per_example(self):
    options = tf.profiler.ProfileOptionBuilder.float_operation()
    options['output'] = 'none'  # disable printing of ops
    num_flops = tf.profiler.profile(
        tf.get_default_graph(),
        options=options
    ).total_float_ops / self.hparams.batch_size
    tf.logging.info('number of flops: {}'.format(num_flops))

  def _build_graph(self, images, labels, mode):
    """Constructs the TF graph for the cifar model.

    Args:
      images: A 4-D image Tensor
      labels: A 2-D labels Tensor.
      mode: string indicating training mode ( e.g., 'train', 'valid', 'test').
    """
    is_training = 'train' in mode
    if is_training:
      self.global_step = tf.train.get_or_create_global_step()
    if self.hparams.use_gamma_swish:
      for layer in range(13):
        _ = tf.Variable(
            [self.hparams.init_beta],
            trainable=True,
            dtype=tf.float32,
            name='swish_beta_layer_{}'.format(layer))
        _ = tf.Variable(
            [self.hparams.init_gamma],
            trainable=True,
            dtype=tf.float32,
            name='swish_gamma_layer_{}'.format(layer))
    logits, hiddens = build_model(
        images,
        self.num_classes,
        is_training,
        self.hparams)
    self.predictions, self.cost_ = helper_utils.setup_loss(
        logits, labels)
    self.hiddens_norm = tf.reduce_mean(hiddens**2)
    self.logits = logits
    self.logit_norm = tf.reduce_mean(logits**2)
    self.accuracy, self.eval_op = tf.metrics.accuracy(
        tf.argmax(labels, 1), tf.argmax(self.predictions, 1))
    self._calc_num_trainable_params()
    self.compute_flops_per_example()

    # Adds L2 weight decay to the cost
    self.cost = helper_utils.decay_weights(self.cost_,
                                           self.hparams.weight_decay_rate)

    if is_training:
      self._build_train_op()

    # Setup checkpointing for this child model
    # Keep 2 or more checkpoints around during training.
    with tf.device('/cpu:0'):
      self.saver = tf.train.Saver(max_to_keep=2)
      self.ckpt_saver = tf.train.Saver(max_to_keep=100)

    self.init = tf.group(tf.global_variables_initializer(),
                         tf.local_variables_initializer())

  def _calc_num_trainable_params(self):
    self.num_trainable_params = np.sum([
        np.prod(var.get_shape().as_list()) for var in tf.trainable_variables()
    ])
    tf.logging.info('number of trainable params: {}'.format(
        self.num_trainable_params))

  def _build_train_op(self):
    """Builds the train op for the cifar model."""
    hparams = self.hparams
    tvars = tf.trainable_variables()
    grads = tf.gradients(self.cost, tvars)
    if hparams.gradient_clipping_by_global_norm > 0.0:
      grads, norm = tf.clip_by_global_norm(
          grads, hparams.gradient_clipping_by_global_norm)
      tf.summary.scalar('grad_norm', norm)

    # Setup the initial learning rate
    initial_lr = self.lr_rate_ph
    optimizer = tf.train.MomentumOptimizer(
        initial_lr,
        0.9,
        use_nesterov=True)

    self.optimizer = optimizer
    apply_op = optimizer.apply_gradients(
        list(zip(grads, tvars)),
        global_step=self.global_step,
        name='train_step')
    train_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([apply_op]):
      self.train_op = tf.group(*train_ops)


class CifarModelTrainer(object):
  """Trains an instance of the CifarModel class."""

  def __init__(self, hparams):
    self._session = None
    self.hparams = hparams

    self.model_dir = os.path.join(FLAGS.checkpoint_dir, 'model')
    self.log_dir = os.path.join(FLAGS.checkpoint_dir, 'log')
    if not tf.gfile.Exists(self.log_dir):
      tf.gfile.MakeDirs(self.log_dir)
    # Set the random seed to be sure the same validation set
    # is used for each model
    np.random.seed(0)
    self.data_loader = data_utils.DataSet(hparams)
    np.random.seed()  # Put the random seed back to random
    self.data_loader.reset()

  def save_model(self, step=None):
    """Dumps model into the backup_dir.

    Args:
      step: If provided, creates a checkpoint with the given step
        number, instead of overwriting the existing checkpoints.
    """
    model_save_name = os.path.join(self.model_dir, 'model.ckpt')
    if not tf.gfile.IsDirectory(self.model_dir):
      tf.gfile.MakeDirs(self.model_dir)
    self.saver.save(self.session, model_save_name, global_step=step,
                    write_meta_graph=False)
    if step % self.hparams.ckpt_every == 0:
      model_save_name = os.path.join(self.model_dir, 'modelckpt.ckpt')
      self.ckpt_saver.save(self.session, model_save_name, global_step=step,
                           write_meta_graph=False)
    tf.logging.info('Saved child model')

  def extract_model_spec(self):
    """Loads a checkpoint with the architecture structure stored in the name."""
    checkpoint_path = tf.train.latest_checkpoint(self.model_dir)
    if checkpoint_path is not None:
      self.saver.restore(self.session, checkpoint_path)
      tf.logging.info('Loaded child model checkpoint from %s',
                      checkpoint_path)
    else:
      self.save_model(step=0)

  def eval_child_model(self, model, data_loader, mode, only_noise_class=False):
    """Evaluate the child model.

    Args:
      model: image model that will be evaluated.
      data_loader: dataset object to extract eval data from.
      mode: will the model be evalled on train, val or test.
      only_noise_class: If True, evaluate the model only on examples from the
      noised class.

    Returns:
      Accuracy of the model on the specified dataset.
    """
    tf.logging.info('Evaluating child model in mode %s', mode)
    while True:
      try:
        with self._new_session(model):
          accuracy, logit_norm_val, hidden_norm_val, cost = helper_utils.eval_child_model(
              self.session,
              model,
              data_loader,
              mode,
              only_noise_class)
          tf.logging.info('Eval child model accuracy: {}'.format(accuracy))
          # If epoch trained without raising the below errors, break
          # from loop.
          break
      except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
        tf.logging.info('Retryable error caught: %s.  Retrying.', e)

    return accuracy, logit_norm_val, hidden_norm_val, cost

  def eval_training_loss(self, model, data_loader, mode):
    """Evaluate the child model.

    Args:
      model: image model that will be evaluated.
      data_loader: dataset object to extract eval data from.
      mode: will the model be evalled on train, val or test.

    Returns:
      Accuracy of the model on the specified dataset.
    """
    tf.logging.info('Evaluating child model in mode %s', mode)
    while True:
      try:
        with self._new_session(model):
          training_loss = helper_utils.eval_training_loss(
              self.session, model, data_loader, mode)
          tf.logging.info('Eval training loss: {}'.format(training_loss))
          # If epoch trained without raising the below errors, break
          # from loop.
          break
      except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
        tf.logging.info('Retryable error caught: %s.  Retrying.', e)

    return training_loss

  def eval_child_robustness(self, model, mode):
    """Evaluate the child model robustness.

    Args:
      model: image model that will be evaluated.
      mode: will the model be evalled on train, val or test.

    Returns:
      Accuracy of the model on the specified dataset.
    """
    tf.logging.info('Evaluating child robustness in mode %s', mode)
    while True:
      try:
        with self._new_session(model):
          accuracy = helper_utils.eval_child_robustness(
              self.session,
              model,
              mode)
          tf.logging.info('Eval child model robustness: {}'.format(accuracy))
          # If epoch trained without raising the below errors, break
          # from loop.
          break
      except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
        tf.logging.info('Retryable error caught: %s.  Retrying.', e)
    return accuracy

  def eval_preds(self, model, data_loader):
    """Evaluate the child model.

    Args:
      model: image model that will be evaluated.
      data_loader: dataset object to extract eval data from.

    Returns:
      Accuracy of the model on the specified dataset.
    """
    tf.logging.info('Evaluating test predictions')
    while True:
      try:
        with self._new_session(model):
          preds = helper_utils.eval_preds(
              self.session,
              model,
              data_loader
              )
          tf.logging.info('Eval preds shape: {}'.format(preds.shape))
          # If epoch trained without raising the below errors, break
          # from loop.
          break
      except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
        tf.logging.info('Retryable error caught: %s.  Retrying.', e)

    return preds

  @contextlib.contextmanager
  def _new_session(self, m):
    """Creates a new session for model m."""
    # Create a new session for this model, initialize
    # variables, and save / restore from
    # checkpoint.
    self._session = tf.Session(
        '',
        config=tf.ConfigProto(
            allow_soft_placement=True, log_device_placement=False))
    self.session.run(m.init)

    # Load in a previous checkpoint, or save this one
    self.extract_model_spec()
    try:
      yield
    finally:
      tf.Session.reset('')
      self._session = None

  def _build_models(self):
    """Builds the image models for train and eval."""
    # Determine if we should build the train and eval model. When using
    # distributed training we only want to build one or the other and not both.
    with tf.variable_scope('model', use_resource=False):
      m = CifarModel(self.hparams)
      m.build('train')
      self._num_trainable_params = m.num_trainable_params
      self._saver = m.saver
      self._ckpt_saver = m.ckpt_saver
    with tf.variable_scope('model', reuse=True, use_resource=False):
      meval = CifarModel(self.hparams)
      meval.build('eval')
    return m, meval

  def _calc_starting_epoch(self, m):
    """Calculates the starting epoch for model m based on global step."""
    hparams = self.hparams
    batch_size = hparams.batch_size
    steps_per_epoch = int(hparams.train_size / batch_size)
    with self._new_session(m):
      curr_step = self.session.run(m.global_step)
    total_steps = steps_per_epoch * hparams.num_epochs
    epochs_left = (total_steps - curr_step) // steps_per_epoch
    starting_epoch = hparams.num_epochs - epochs_left
    return starting_epoch

  def _run_training_loop(self, m, curr_epoch):
    """Trains the cifar model `m` for one epoch."""
    start_time = time.time()
    while True:
      try:
        with self._new_session(m):
          m.log_dir = self.log_dir
          train_accuracy, train_loss = helper_utils.run_epoch_training(
              self.session, m, self.data_loader, curr_epoch)
          tf.logging.info('Saving model after epoch')
          self.save_model(step=curr_epoch)
          break
      except (tf.errors.AbortedError, tf.errors.UnavailableError) as e:
        tf.logging.info('Retryable error caught: %s.  Retrying.', e)
    tf.logging.info('Finished epoch: {}'.format(curr_epoch))
    epoch_time = time.time() - start_time
    tf.logging.info('Epoch time(min): {}'.format(
        epoch_time / 60.0))
    return train_accuracy, train_loss, epoch_time

  def _compute_final_accuracies(self, meval):
    """Run once training is finished to compute final val/test accuracies."""
    valid_accuracy, _, _, cost = self.eval_child_model(
        meval, self.data_loader, 'val')
    if self.hparams.eval_test:
      test_accuracy, logit_norm_test_f, hidden_norm_test_f, cost = self.eval_child_model(
          meval, self.data_loader, 'test')
    else:
      test_accuracy = 0
    tf.logging.info('Test Accuracy: {}'.format(test_accuracy))
    return valid_accuracy, test_accuracy, logit_norm_test_f, hidden_norm_test_f, cost

  def _compute_final_robustness(self, meval):
    """Run once training is finished to compute final val/test accuracies."""
    if self.hparams.eval_test:
      test_robustness = self.eval_child_robustness(meval, 'test')
    else:
      test_robustness = 0
    tf.logging.info('Test Robustness: {}'.format(test_robustness))
    return test_robustness

  def run_model(self):
    """Trains and evalutes the image model."""
    hparams = self.hparams

    # Build the child graph
    with tf.Graph().as_default(), tf.device(
        '/cpu:0' if FLAGS.use_cpu else '/gpu:0'):
      m, meval = self._build_models()

      # Figure out what epoch we are on
      starting_epoch = self._calc_starting_epoch(m)

      # Run the validation error right at the beginning
      valid_accuracy, valid_logit_norm, valid_hidden_norm, valid_loss = self.eval_child_model(
          meval, self.data_loader, 'val')

      if hparams.validation_size > 0:
        valid_accuracy, valid_logit_norm, valid_hidden_norm, valid_loss = self.eval_child_model(
            meval, self.data_loader, 'val')
      else:
        valid_accuracy, valid_logit_norm, valid_hidden_norm, valid_loss = self.eval_child_model(
            meval, self.data_loader, 'test')

      tf.logging.info(
          'Before Training Epoch: {} valid acc {} valid logit norm {} valid hidden norm {} valid loss {}'
          .format(starting_epoch, valid_accuracy, valid_logit_norm,
                  valid_hidden_norm, valid_loss))

      log_dir_ = os.path.join(FLAGS.checkpoint_dir, 'log')

      for curr_epoch in range(starting_epoch, hparams.num_epochs):

        # Run one training epoch
        training_accuracy, training_loss, epoch_time = self._run_training_loop(
            m, curr_epoch)
        tf.logging.info(
            'At epoch {}: train accuracy {}, train loss {}, epoch time {}'
            .format(curr_epoch, training_accuracy * 100, training_loss,
                    epoch_time))

        if hparams.validation_size > 0:
          valid_accuracy, valid_logit_norm, valid_hidden_norm, valid_loss = self.eval_child_model(
              meval, self.data_loader, 'val')
        else:
          valid_accuracy, valid_logit_norm, valid_hidden_norm, valid_loss = self.eval_child_model(
              meval, self.data_loader, 'test')
        noisy_valid_accuracy, _, _, noisy_valid_loss = self.eval_child_model(
            meval, self.data_loader, 'noisy_test')
        tf.logging.info(
            'At epoch {}: validation accuracy {}, validation loss {}, '
            'noisy validation accuracy {}, noisy validation loss {}, '
            'noise fitting {}'.format(curr_epoch, valid_accuracy * 100,
                                      valid_loss, noisy_valid_accuracy * 100,
                                      noisy_valid_loss,
                                      valid_loss - noisy_valid_loss))

        # Also evaluate on just the noised class
        if hparams.noise_class >= 0:
          if hparams.validation_size > 0:
            nc_valid_accuracy, _, _, nc_valid_loss = self.eval_child_model(
                meval, self.data_loader, 'val', only_noise_class=True)
          else:
            nc_valid_accuracy, _, _, nc_valid_loss = self.eval_child_model(
                meval, self.data_loader, 'test', only_noise_class=True)
          nc_noisy_valid_accuracy, _, _, nc_noisy_valid_loss = self.eval_child_model(
              meval, self.data_loader, 'noisy_test', only_noise_class=True)
          tf.logging.info(
              'At epoch {} for the noised class: validation accuracy {}, '
              'validation loss {}, noisy valid acc {}, noisy valid loss {}, '
              'noise fitting {}'
              .format(curr_epoch, nc_valid_accuracy * m.num_classes * 100,
                      nc_valid_loss * m.num_classes,
                      nc_noisy_valid_accuracy * m.num_classes * 100,
                      nc_noisy_valid_loss * m.num_classes,
                      (nc_valid_loss - nc_noisy_valid_loss) * m.num_classes))

        # Also evaluate on the extra (distribution shift) dataset
        if hparams.extra_dataset == 'cifar10_1':
          extra_accuracy, _, _, extra_loss = self.eval_child_model(
              meval, self.data_loader, 'extra_test')
          tf.logging.info(
              'At epoch {} for the extra dataset: test accuracy {}, '
              'test_loss {}, effective robustness {}'
              .format(curr_epoch, extra_accuracy * 100, extra_loss,
                      helper_utils.effective_robustness(valid_accuracy,
                                                        extra_accuracy) * 100))
        tf.logging.info(
            'Epoch: {} Training Acc: {} Validation Acc: {} valid norm :{} hidden norm :{}'
            .format(curr_epoch, training_accuracy, valid_accuracy,
                    valid_logit_norm, valid_hidden_norm))
        # Early stopping
        if valid_accuracy >= hparams.max_accuracy:
          # The most recent checkpoint is already saved; stop training here
          break
        if training_loss <= hparams.min_loss:
          # The most recent checkpoint is already saved; stop training here
          break
      valid_accuracy, test_accuracy, logit_norm, hidden_norm, valid_loss = self._compute_final_accuracies(
          meval)
      train_accuracy, _, _, train_loss = self.eval_child_model(
          meval, self.data_loader, 'train')
      tf.logging.info(
          'After training, final test accuracy {}, validation loss {}, '
          'training accuracy {}, training loss {}'
          .format(test_accuracy * 100, valid_loss, train_accuracy * 100,
                  train_loss))
      mean_cc_value = 0.0
      with tf.gfile.Open(os.path.join(log_dir_, 'final_acc.bin'), 'w') as f:
        six.moves.cPickle.dump(
            [train_accuracy, valid_accuracy, test_accuracy,
             mean_cc_value, logit_norm, hidden_norm, train_loss],
            f)

    tf.logging.info(
        'Train Acc: {}\tValid Acc: {}\tTest Acc: {}\tTest Robust: {}'.format(
            train_accuracy, valid_accuracy, test_accuracy, mean_cc_value))
    tf.logging.info('Final training loss {}'.format(train_loss))

  @property
  def saver(self):
    return self._saver

  @property
  def ckpt_saver(self):
    return self._ckpt_saver

  @property
  def session(self):
    return self._session

  @property
  def num_trainable_params(self):
    return self._num_trainable_params


def main(_):
  if FLAGS.dataset == 'cifar10':
    data_path = './cifar10_data/'
    assert FLAGS.train_size <= 50000
    validation_size = 50000 - FLAGS.train_size
  elif FLAGS.dataset == 'cifar100':
    data_path = './cifar100_data/'
    assert FLAGS.train_size <= 50000
    validation_size = 50000-FLAGS.train_size
  elif FLAGS.dataset == 'svhn':
    data_path = './svhn_dataset/'
    assert FLAGS.train_size <= 73257
    validation_size = 73257-FLAGS.train_size
  else:
    raise ValueError('Invalid dataset: %s' % FLAGS.dataset)

  hparams = contrib_training.HParams(
      train_size=FLAGS.train_size,
      validation_size=validation_size,
      eval_test=1,
      dataset=FLAGS.dataset,
      extra_dataset=FLAGS.extra_dataset,
      frequency=FLAGS.frequency,
      amplitude=FLAGS.amplitude,
      data_path=data_path,
      batch_size=256,
      gradient_clipping_by_global_norm=5.0,
      dummy_f=FLAGS.dummy_f,
      augment_type=FLAGS.augment_type,
      mixup_alpha=FLAGS.mixup_alpha,
      num_augmentation_layers=FLAGS.num_augmentation_layers,
      augmentation_magnitude=FLAGS.augmentation_magnitude,
      augmentation_probability=FLAGS.augmentation_probability,
      freq_augment_amplitude=FLAGS.freq_augment_amplitude,
      freq_augment_ffrac=FLAGS.freq_augment_ffrac,
      apply_cutout=FLAGS.apply_cutout,
      apply_flip_crop=FLAGS.apply_flip_crop,
      num_epochs=FLAGS.num_epochs,
      weight_decay_rate=FLAGS.weight_decay_rate,
      lr=FLAGS.lr,
      model_name=FLAGS.model_name,
      is_gan_data=FLAGS.is_gan_data,
      use_fixup=FLAGS.use_fixup,
      use_batchnorm=FLAGS.use_batchnorm,
      use_gamma_swish=FLAGS.use_gamma_swish,
      init_beta=FLAGS.init_beta,
      init_gamma=FLAGS.init_gamma,
      noise_type=FLAGS.noise_type,
      spatial_frequency=FLAGS.spatial_frequency,
      noise_seed=FLAGS.noise_seed,
      noise_class=FLAGS.noise_class,
      max_accuracy=FLAGS.max_accuracy,
      min_loss=FLAGS.min_loss,
      teacher_model=FLAGS.teacher_model,
      distillation_alpha=FLAGS.distillation_alpha,
      normalize_amplitude=FLAGS.normalize_amplitude,
      ckpt_every=FLAGS.ckpt_every,
      )
  tf.logging.info('All hparams : {}'.format(hparams))

  if FLAGS.model_name == 'wrn_32':
    setattr(hparams, 'model_name', 'wrn')
    hparams.add_hparam('wrn_size', 32)
  elif FLAGS.model_name == 'wrn_160':
    setattr(hparams, 'model_name', 'wrn')
    hparams.add_hparam('wrn_size', 160)
  elif FLAGS.model_name == 'shake_shake_32':
    setattr(hparams, 'model_name', 'shake_shake')
    hparams.add_hparam('shake_shake_widen_factor', 2)
  elif FLAGS.model_name == 'shake_shake_96':
    setattr(hparams, 'model_name', 'shake_shake')
    hparams.add_hparam('shake_shake_widen_factor', 6)
  elif FLAGS.model_name == 'shake_shake_112':
    setattr(hparams, 'model_name', 'shake_shake')
    hparams.add_hparam('shake_shake_widen_factor', 7)
  elif FLAGS.model_name == 'pyramid_net':
    setattr(hparams, 'model_name', 'pyramid_net')
    hparams.batch_size = 64
  else:
    raise ValueError('Not Valid Model Name: %s' % FLAGS.model_name)
  tf.logging.info('All hparams : {}'.format(hparams))

  cifar_trainer = CifarModelTrainer(hparams)
  cifar_trainer.run_model()

if __name__ == '__main__':
  tf.app.run(main)
