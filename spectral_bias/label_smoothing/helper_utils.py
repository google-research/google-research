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

"""Helper functions used for training AutoAugment models."""

import contextlib
import os

import augmentation_transforms
import custom_ops as ops
import numpy as np
import scipy.special
from shake_drop import build_shake_drop_model
from shake_shake import build_shake_shake_model
from six.moves import xrange
import tensorflow as tf
from tensorflow.contrib import framework as contrib_framework
from wrn import build_wrn_model


arg_scope = contrib_framework.arg_scope


@contextlib.contextmanager
def nested(*contexts):
  with contextlib.ExitStack() as stack:
    for context in contexts:
      stack.enter_context(context)
    yield contexts


def setup_arg_scopes(is_training, hparams):
  """Sets up the argscopes that will be used when building an image model.

  Args:
    is_training: Is the model training or not.
    hparams: Hyper-parameters

  Returns:
    Arg scopes to be put around the model being constructed.
  """

  batch_norm_decay = 0.9
  batch_norm_epsilon = 1e-5
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      'scale': True,
      # collection containing the moving mean and moving variance.
      'is_training': is_training,
      'disable': not hparams.use_batchnorm,
  }

  scopes = []

  scopes.append(arg_scope([ops.maybe_normalize], **batch_norm_params))
  scopes.append(
      arg_scope([ops.conv2d], residual_depth=12 if hparams.use_fixup else None))
  return scopes


def setup_loss(logits, labels):
  """Returns the cross entropy for the given `logits` and `labels`."""
  predictions = tf.nn.softmax(logits)
  cost = tf.losses.softmax_cross_entropy(onehot_labels=labels,
                                         logits=logits)
  return predictions, cost


def decay_weights(cost, weight_decay_rate):
  """Calculates the loss for l2 weight decay and adds it to `cost`."""
  costs = []
  for var in tf.trainable_variables():
    if 'swish_' not in var.name:
      costs.append(tf.nn.l2_loss(var))
    else:
      tf.logging.info('REMOVED VARIABLE WITH NAME {}'.format(var.name))
  cost += tf.multiply(weight_decay_rate, tf.add_n(costs))
  return cost


def effective_robustness(original_acc, new_acc):
  # CIFAR-10.1
  a = 0.8318
  b = -0.4736
  return float(new_acc -
               scipy.special.expit(a * scipy.special.logit(original_acc) + b))


def effective_noise_fitting(clean_test_loss, noisy_test_loss, amplitude,
                            num_classes):
  assert num_classes > 0
  assert amplitude >= 0
  # expected_offset = -np.log(1.0 - amplitude + amplitude / num_classes)
  expected_offset = 0
  return clean_test_loss + expected_offset - noisy_test_loss


def eval_child_model(session, model, data_loader, mode, only_noise_class=False):
  """Evaluates `model` on held out data depending on `mode`.

  Args:
    session: TensorFlow session the model will be run with.
    model: TensorFlow model that will be evaluated.
    data_loader: DataSet object that contains data that `model` will
      evaluate.
    mode: Will `model` either evaluate validation or test data.
    only_noise_class: If True, evaluate the model only on examples from the
      noised class.

  Returns:
    Accuracy of `model` when evaluated on the specified dataset.
    Loss of `model` when evaluated on the specified dataset.

  Raises:
    ValueError: if invalid dataset `mode` is specified.
  """
  if mode == 'val':
    images = data_loader.val_images
    labels = data_loader.val_labels
  elif mode == 'test' or mode == 'noisy_test':
    images = data_loader.test_images
    labels = data_loader.test_labels
  elif mode == 'train':
    images = data_loader.train_images
    labels = data_loader.train_labels
  elif mode == 'extra_test':
    images = data_loader.extra_test_images
    labels = data_loader.extra_test_labels
  else:
    raise ValueError('Not valid eval mode')

  assert len(images) == len(labels)

  tf.logging.info('model.batch_size is {}'.format(model.batch_size))
  eval_batches = int(np.floor(len(images) / model.batch_size))
  logit_norm_vals = []
  hidden_norm_vals = []
  costs = []
  for i in range(eval_batches):
    eval_images = images[i * model.batch_size:(i + 1) * model.batch_size]
    eval_labels = np.copy(labels[i * model.batch_size:(i + 1) *
                                 model.batch_size])
    if only_noise_class:
      # Compute a mask for which examples are not from noise_class
      mask = np.argmax(eval_labels, axis=1) != model.hparams.noise_class
    if mode == 'noisy_test':
      noising_images = eval_images
      if model.hparams.noise_type == 'radial':
        eval_labels = augmentation_transforms.add_radial_noise(
            noising_images, eval_labels,
            model.hparams.frequency, model.hparams.amplitude,
            model.hparams.noise_class, model.hparams.normalize_amplitude)
      elif model.hparams.noise_type == 'random' or model.hparams.noise_type == 'fourier' or model.hparams.noise_type == 'f' or model.hparams.noise_type == '1/f':
        eval_labels = augmentation_transforms.add_sinusoidal_noise(
            noising_images, eval_labels, model.hparams.frequency,
            model.hparams.amplitude, data_loader.direction,
            model.hparams.noise_class, model.hparams.normalize_amplitude)
      elif model.hparams.noise_type == 'uniform':
        eval_labels = augmentation_transforms.add_uniform_noise(
            eval_labels, model.hparams.amplitude, model.hparams.noise_class)
    if only_noise_class:
      # zero out the examples that are from the other classes
      eval_labels[mask] = 0

    _, logit_norm_val, hidden_norm_val, cost = session.run(
        [model.eval_op, model.logit_norm, model.hiddens_norm, model.cost_],
        feed_dict={
            model.images: eval_images,
            model.labels: eval_labels,
        })
    logit_norm_vals.append(logit_norm_val)
    hidden_norm_vals.append(hidden_norm_val)
    costs.append(cost)
  return session.run(model.accuracy), np.mean(logit_norm_vals), np.mean(
      hidden_norm_vals), np.mean(costs)


def eval_training_loss(session, model, data_loader, mode):
  """Evaluates `model` on held out data depending on `mode`.

  Args:
    session: TensorFlow session the model will be run with.
    model: TensorFlow model that will be evaluated.
    data_loader: DataSet object that contains data that `model` will
      evaluate.
    mode: Will `model` either evaluate validation or test data.

  Returns:
    Accuracy of `model` when evaluated on the specified dataset.

  Raises:
    ValueError: if invalid dataset `mode` is specified.
  """
  images = data_loader.train_images
  labels = data_loader.train_labels
  assert len(images) == len(labels)
  assert mode == 'train'
  tf.logging.info('model.batch_size is {}'.format(model.batch_size))
  # assert len(images) % model.batch_size == 0
  eval_batches = int(np.floor(len(images) / model.batch_size))
  training_loss = []
  for i in range(eval_batches):
    eval_images = images[i * model.batch_size:(i + 1) * model.batch_size]
    eval_labels = labels[i * model.batch_size:(i + 1) * model.batch_size]
    loss_tr = session.run(
        model.cost_,
        feed_dict={
            model.images: eval_images,
            model.labels: eval_labels,
        })
    training_loss.append(loss_tr)
  return np.mean(training_loss)

MEANS = [0.49139968, 0.48215841, 0.44653091]
STDS = [0.24703223, 0.24348513, 0.26158784]


def load_cc_data(data_path):
  with tf.gfile.Open(data_path, 'rb') as f:
    data = np.load(f)
  data = data / 255.0
  data = (data - MEANS) / STDS
  return data


def eval_child_robustness(session, model, mode):
  """Evaluates `model` on held out data depending on `mode`.

  Args:
    session: TensorFlow session the model will be run with.
    model: TensorFlow model that will be evaluated.
    mode: Will `model` either evaluate validation or test data.

  Returns:
    Accuracy of `model` when evaluated on the specified dataset.

  Raises:
    ValueError: if invalid dataset `mode` is specified.
  """
  if mode == 'val':
    raise ValueError('Not valid eval mode')
  elif mode == 'test':
    cc_datapath = './common_corruption/CIFAR-10-C/*'
    labels_path = './common_corruption/CIFAR-10-C/labels.npy'
    all_files = [
        f for f in tf.gfile.Glob(cc_datapath)
        if 'extra' not in f and 'labels' not in f
    ]
    all_files.extend(
        tf.gfile.Glob(
            './common_corruption/CIFAR-10-C/extra/*'))
    raw_labels = np.load(tf.gfile.Open(labels_path, 'rb'))
    raw_labels = np.eye(10)[np.array(raw_labels, dtype=np.int32)]
  else:
    raise ValueError('Not valid eval mode')
  tf.logging.info('model.batch_size is {}'.format(model.batch_size))
  mean_accuracies = {}
  for cc_filepath in all_files:
    many_raw_features = load_cc_data(cc_filepath)
    cc_name = cc_filepath.split('/')[-1].replace('.npy', '')
    acc = []
    # each corruption has 5 levels of severity each 10k images in
    # many_raw_features
    for j in range(5):
      severity_name = str(j + 1)
      raw_features = many_raw_features[j * 10000:(j + 1) * 10000]
      eval_batches = int(np.floor(len(raw_features) / model.batch_size))
      for i in range(eval_batches):
        eval_images = raw_features[i * model.batch_size:(i + 1) *
                                   model.batch_size]
        eval_labels = raw_labels[i * model.batch_size:(i + 1) *
                                 model.batch_size]
        preds = session.run(
            model.predictions,
            feed_dict={
                model.images: eval_images,
                model.labels: eval_labels,
            })
        acc.append(
            np.mean(np.argmax(preds, axis=1) == np.argmax(eval_labels, axis=1)))
      mean_accuracies[cc_name + '_' + severity_name] = np.mean(acc)
      tf.logging.info('{} accuracy is {}'.format(
          cc_name + '_' + severity_name,
          mean_accuracies[cc_name + '_' + severity_name]))
  return mean_accuracies


def eval_preds(session, model, data_loader):
  """Evaluates `model` on held out data depending on `mode`.

  Args:
    session: TensorFlow session the model will be run with.
    model: TensorFlow model that will be evaluated.
    data_loader: DataSet object that contains data that `model` will
      evaluate.

  Returns:
    Accuracy of `model` when evaluated on the specified dataset.

  Raises:
    ValueError: if invalid dataset `mode` is specified.
  """
  images = data_loader.test_images
  labels = data_loader.test_labels
  assert len(images) == len(labels)
  tf.logging.info('model.batch_size is {}'.format(model.batch_size))
  eval_batches = int(np.floor(len(images) / model.batch_size))
  preds_recs = []
  for i in range(eval_batches):
    eval_images = images[i * model.batch_size:(i + 1) * model.batch_size]
    eval_labels = labels[i * model.batch_size:(i + 1) * model.batch_size]
    preds = session.run(
        model.logits,
        feed_dict={
            model.images: eval_images,
            model.labels: eval_labels,
        })
    preds_recs.extend(preds)
  return np.array(preds_recs)


def cosine_lr(learning_rate, epoch, iteration, batches_per_epoch, total_epochs):
  """Cosine Learning rate.

  Args:
    learning_rate: Initial learning rate.
    epoch: Current epoch we are one. This is one based.
    iteration: Current batch in this epoch.
    batches_per_epoch: Batches per epoch.
    total_epochs: Total epochs you are training for.

  Returns:
    The learning rate to be used for this current batch.
  """
  t_total = total_epochs * batches_per_epoch
  t_cur = float(epoch * batches_per_epoch + iteration)
  return 0.5 * learning_rate * (1 + np.cos(np.pi * t_cur / t_total))


def get_lr(curr_epoch, hparams, iteration=None):
  """Returns the learning rate during training based on the current epoch."""
  assert iteration is not None
  batches_per_epoch = int(hparams.train_size / hparams.batch_size)
  lr = cosine_lr(hparams.lr, curr_epoch, iteration, batches_per_epoch,
                 hparams.num_epochs)
  return lr


def reload_teacher(session, hparams):
  """Reload a saved model to use as teacher in self-distillation."""
  with tf.variable_scope('teacher', reuse=tf.AUTO_REUSE):
    inputs = tf.placeholder('float', [None, 32, 32, 3])
    scopes = setup_arg_scopes(is_training=False, hparams=hparams)
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE):
      with nested(*scopes):
        if hparams.model_name == 'pyramid_net':
          logits, _ = build_shake_drop_model(
              inputs, num_classes=10, is_training=False)
        elif hparams.model_name == 'wrn':
          logits, _ = build_wrn_model(
              inputs, num_classes=10, hparams=hparams)
        elif hparams.model_name == 'shake_shake':
          logits, _ = build_shake_shake_model(
              inputs, num_classes=10, hparams=hparams, is_training=False)
        else:
          print(f'unrecognized hparams.model_name: {hparams.model_name}')
          assert 0
  ckpt = tf.train.latest_checkpoint(
      os.path.join(hparams.teacher_model, 'model'))
  # Map each variable name in the checkpoint to the variable name to restore
  scopedict = {}
  myvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='teacher')
  for var in myvars:
    scopedict[var.name[8:-2]] = var
  saver = tf.train.Saver(var_list=scopedict)
  saver.restore(session, ckpt)
  return logits, inputs


def run_epoch_training(session,
                       model,
                       data_loader,
                       curr_epoch):
  """Runs one epoch of training for the model passed in.

  Args:
    session: TensorFlow session the model will be run with.
    model: TensorFlow model that will be evaluated.
    data_loader: DataSet object that contains data that `model` will
      evaluate.
    curr_epoch: How many of epochs of training have been done so far.

  Returns:
    The accuracy of 'model' on the training set
    The training loss of 'model' during this epoch
  """
  steps_per_epoch = int(model.hparams.train_size / model.hparams.batch_size)
  tf.logging.info('steps per epoch: {}'.format(steps_per_epoch))
  curr_step = session.run(model.global_step)
  assert curr_step % steps_per_epoch == 0

  # Get the current learning rate for the model based on the current epoch
  curr_lr = get_lr(curr_epoch, model.hparams, iteration=0)
  tf.logging.info('lr of {} for epoch {}'.format(curr_lr, curr_epoch))
  if model.hparams.teacher_model is not None:
    teacher_logits, teacher_inputs = reload_teacher(session, model.hparams)
  costs = []
  for step in xrange(steps_per_epoch):
    curr_lr = get_lr(curr_epoch, model.hparams, iteration=(step + 1))
    # Update the lr rate variable to the current LR.
    model.lr_rate_ph.load(curr_lr, session=session)
    if step % 20 == 0:
      tf.logging.info('Training {}/{}'.format(step, steps_per_epoch))

    train_images, train_labels = data_loader.next_batch()
    # Reload the teacher model inside this session, and use distillation
    if model.hparams.teacher_model is not None:
      teacher_labels = session.run(
          teacher_logits, feed_dict={teacher_inputs: train_images})
      teacher_labels = scipy.special.softmax(teacher_labels, axis=1)
      train_labels = model.hparams.distillation_alpha * train_labels + (
          1.0 - model.hparams.distillation_alpha) * teacher_labels

    _, step, _, cost = session.run(
        [model.train_op, model.global_step, model.eval_op, model.cost_],
        feed_dict={
            model.images: train_images,
            model.labels: train_labels,
        })
    costs.append(cost)

  train_accuracy = session.run(model.accuracy)
  tf.logging.info('Train accuracy: {}'.format(train_accuracy))
  return train_accuracy, np.mean(costs)
