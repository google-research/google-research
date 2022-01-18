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

"""Model training with TensorFlow eager execution."""
import os
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow.compat.v2 as tf

from extreme_memorization import alignment
from extreme_memorization import cifar100_dataset
from extreme_memorization import cifar10_dataset
from extreme_memorization import convnet
from extreme_memorization import mlp
from extreme_memorization import svhn_dataset

FLAGS = flags.FLAGS

flags.DEFINE_integer('log_interval', 10,
                     'batches between logging training status')
# 1k epochs.
flags.DEFINE_integer('train_epochs', 2000,
                     'batches between logging training status')

flags.DEFINE_string('output_dir', '/tmp/tensorflow/generalization/',
                    'Directory to write TensorBoard summaries')

flags.DEFINE_string('model_dir', '/tmp/tensorflow/generalization/checkpoints/',
                    'Directory to write TensorBoard summaries')

flags.DEFINE_string('train_input_files',
                    '/tmp/cifar10/image_cifar10_fingerprint-train*',
                    'Input pattern for training tfrecords.')

flags.DEFINE_string('test_input_files',
                    '/tmp/cifar10/image_cifar10_fingerprint-dev*',
                    'Input pattern for test tfrecords.')

flags.DEFINE_float('learning_rate', 0.01, 'Learning rate.')

flags.DEFINE_bool('no_gpu', False,
                  'disables GPU usage even if a GPU is available')

flags.DEFINE_bool('custom_init', False, 'Use custom initializers for w_1.')

flags.DEFINE_bool('shuffled_labels', False,
                  'Use randomized labels instead of true labels.')

flags.DEFINE_float('stddev', 0.001, 'Stddev for random normal init.')

flags.DEFINE_integer('num_units', 1024, 'Number of hidden units.')

flags.DEFINE_integer(
    'batch_size', 256, 'Batch size for training and evaluation. When using '
    'multiple gpus, this is the global batch size for '
    'all devices. For example, if the batch size is 32 '
    'and there are 4 GPUs, each GPU will get 8 examples on '
    'each step.')

flags.DEFINE_enum(
    'model_type', 'mlp', ['mlp', 'convnet'],
    'Model architecture type. Either a 2-layer MLP or a ConvNet.')

flags.DEFINE_enum(
    'loss_function', 'cross_entropy', ['cross_entropy', 'hinge', 'l2'],
    'Choice of loss functions between cross entropy or multi-class hinge'
    'or squared loss.')

flags.DEFINE_enum(
    'dataset', 'cifar10', ['cifar10', 'cifar100', 'svhn'],
    'Which tf.data.Dataset object to initialize.')

flags.DEFINE_enum(
    'activation', 'relu', ['sin', 'relu', 'sigmoid'],
    'Activation function to be used for MLP model type.')

flags.DEFINE_enum(
    'data_format', None, ['channels_first', 'channels_last'],
    'A flag to override the data format used in the model. '
    'channels_first provides a performance boost on GPU but is not '
    'always compatible with CPU. If left unspecified, the data format '
    'will be chosen automatically based on whether TensorFlow was '
    'built for CPU or GPU.')


def get_dataset():
  if FLAGS.dataset == 'cifar10':
    return cifar10_dataset
  elif FLAGS.dataset == 'cifar100':
    return cifar100_dataset
  elif FLAGS.dataset == 'svhn':
    return svhn_dataset


def get_activation():
  if FLAGS.activation == 'sin':
    return tf.math.sin
  elif FLAGS.activation == 'relu':
    return tf.nn.relu
  elif FLAGS.activation == 'sigmoid':
    return tf.math.sigmoid


def gather_2d(params, indices):
  """Gathers from `params` with a 2D batched `indices` array.

  Args:
    params: [D0, D1, D2 ... Dn] Tensor
    indices: [D0, D1'] integer Tensor

  Returns:
    result: [D0, D1', D2 ... Dn] Tensor, where
      result[i, j, ...] = params[i, indices[i, j], ...]

  Raises:
    ValueError: if more than one entries in [D2 ... Dn] are not known.
  """
  d0 = tf.shape(params)[0]
  d1 = tf.shape(params)[1]
  d2_dn = params.shape.as_list()[2:]
  none_indices = [i for i, s in enumerate(d2_dn) if s is None]
  if none_indices:
    if len(none_indices) > 1:
      raise ValueError(
          'More than one entry in D2 ... Dn not known for Tensor %s.' % params)
    d2_dn[none_indices[0]] = -1
  flatten_params = tf.reshape(params, [d0 * d1] + d2_dn)
  flatten_indices = tf.expand_dims(tf.range(d0) * d1, 1) + tf.cast(
      indices, dtype=tf.int32)
  return tf.gather(flatten_params, flatten_indices)


def hinge_loss(labels, logits):
  """Multi-class hinge loss.

  Args:
    labels: [batch_size] integer Tensor of correct class labels.
    logits: [batch_size, num_classes] Tensor of prediction scores.

  Returns:
    [batch_size] Tensor of the hinge loss value.
  """
  label_logits = gather_2d(logits, tf.expand_dims(labels, 1))
  return tf.reduce_sum(
      tf.math.maximum(0, logits - label_logits + 1.0), axis=1) - 1.0


def get_squared_loss(logits, labels):
  onehot_labels = tf.one_hot(indices=labels, depth=get_dataset().NUM_LABELS)
  diff = logits - tf.to_float(onehot_labels)
  loss_vector = tf.reduce_mean(tf.square(diff), axis=1)
  return tf.reduce_mean(loss_vector), loss_vector


def get_softmax_loss(logits, labels):
  loss_vector = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits=logits, labels=labels)
  return tf.reduce_mean(loss_vector), loss_vector


def get_hinge_loss(logits, labels):
  loss_vector = hinge_loss(labels=labels, logits=logits)
  return tf.reduce_mean(loss_vector), loss_vector


def loss(logits, labels):
  if FLAGS.loss_function == 'cross_entropy':
    return get_softmax_loss(logits, labels)
  elif FLAGS.loss_function == 'hinge':
    return get_hinge_loss(logits, labels)
  elif FLAGS.loss_function == 'l2':
    return get_squared_loss(logits, labels)


def compute_accuracy(logits, labels):
  predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
  labels = tf.cast(labels, tf.int64)
  return tf.reduce_mean(
      tf.cast(tf.equal(predictions, labels), dtype=tf.float32))


def get_image_labels(features, shuffled_labels=False):
  images = features['image/encoded']
  if shuffled_labels:
    labels = features['image/class/shuffled_label']
  else:
    labels = features['image/class/label']
  return images, labels


def train(model, optimizer, dataset, step_counter, log_interval=None):
  """Trains model on `dataset` using `optimizer`."""

  start = time.time()
  for (batch, (features)) in enumerate(dataset):
    images, labels = get_image_labels(features, FLAGS.shuffled_labels)
    # Record the operations used to compute the loss given the input,
    # so that the gradient of the loss with respect to the variables
    # can be computed.
    with tf.GradientTape(persistent=True) as tape:
      tape.watch(images)
      logits = model(images, labels, training=True, step=step_counter)
      tape.watch(logits)
      loss_value, loss_vector = loss(logits, labels)
      loss_vector = tf.unstack(loss_vector)

      tf.summary.scalar('loss', loss_value, step=step_counter)
      tf.summary.scalar(
          'accuracy', compute_accuracy(logits, labels), step=step_counter)

    logit_grad_vector = []
    for i, per_example_loss in enumerate(loss_vector):
      logits_grad = tape.gradient(per_example_loss, logits)
      logit_grad_vector.append(tf.unstack(logits_grad)[i])

    variables = model.trainable_variables
    per_label_grads = {}
    for label in range(get_dataset().NUM_LABELS):
      per_label_grads[label] = []

    per_example_grads = []
    for i, (per_example_loss, label, logit_grad) in enumerate(
        zip(loss_vector, labels, logit_grad_vector)):
      grads = tape.gradient(per_example_loss, variables)
      grads.append(logit_grad)
      per_example_grads.append((grads, label))
      per_label_grads[int(label.numpy())].append(grads)

    for i, var in enumerate(variables + [logits]):
      if i < len(variables):
        var_name = var.name
      else:
        # Last one is logits.
        var_name = 'logits'
      grad_list = [(grads[0][i], grads[1]) for grads in per_example_grads]
      if grad_list[0][0] is None:
        logging.info('grad_list none: %s', var_name)
        continue
      # Okay to restrict this to 10, even for CIFAR100 since this adds a
      # significant compute overhead.
      for label in range(10):
        label_grad_list = [
            grad[0] for grad in grad_list if tf.math.equal(grad[1], label)
        ]

        if not label_grad_list:
          logging.info('label_grad_list none: %s', var_name)
          continue

        label_grad_list = [tf.reshape(grad, [-1]) for grad in label_grad_list]
        if len(label_grad_list) > 1:
          ggmm = alignment.compute_alignment(label_grad_list)
          key = 'grad_alignment/%s/%s' % (label, var_name)
          tf.summary.scalar(key, ggmm, step=step_counter)

    # Compute gradients, only for trainable variables.
    variables = model.trainable_variables
    grads = tape.gradient(loss_value, variables)

    # Tuple is used for py3, where zip is a generator producing values once.
    grads_and_vars = tuple(zip(grads, model.trainable_variables))
    for g, v in grads_and_vars:
      if g is not None:
        tf.summary.scalar(
            'Norm/Grad/%s' % v.name, tf.norm(g), step=step_counter)

    # Chart all variables.
    for v in model.variables:
      tf.summary.scalar('Norm/Var/%s' % v.name, tf.norm(v), step=step_counter)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    if log_interval and batch % log_interval == 0:
      rate = log_interval / (time.time() - start)
      print('Step #%d\tLoss: %.6f (%d steps/sec)' % (batch, loss_value, rate))
      start = time.time()


def test(model, dataset, step_counter):
  """Perform an evaluation of `model` on the examples from `dataset`."""
  avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
  accuracy = tf.keras.metrics.Accuracy('accuracy', dtype=tf.float32)

  for features in dataset:
    images, labels = get_image_labels(features, FLAGS.shuffled_labels)
    logits = model(images, labels, training=False, step=step_counter)
    loss_value, _ = loss(logits, labels)
    avg_loss(loss_value)
    accuracy(
        tf.argmax(logits, axis=1, output_type=tf.int64),
        tf.cast(labels, tf.int64))
  print('Test set: Average loss: %.4f, Accuracy: %4f%%\n' %
        (avg_loss.result(), 100 * accuracy.result()))
  with tf.summary.always_record_summaries():
    tf.summary.scalar('loss', avg_loss.result(), step=step_counter)
    tf.summary.scalar('accuracy', accuracy.result(), step=step_counter)


def run_eager():
  """Run training and eval loop in eager mode."""
  # No need to run tf.enable_eager_execution() since its supposed to be on by
  # default in TF2.0

  # Automatically determine device and data_format
  (device, data_format) = ('/gpu:0', 'channels_first')
  if FLAGS.no_gpu or not tf.test.is_gpu_available():
    (device, data_format) = ('/cpu:0', 'channels_last')
  # If data_format is defined in FLAGS, overwrite automatically set value.
  if FLAGS.data_format is not None:
    data_format = FLAGS.data_format
  print('Using device %s, and data format %s.' % (device, data_format))

  # Its important to set the data format before the model is built, conv layers
  # usually need it.
  tf.keras.backend.set_image_data_format(data_format)

  # Load the datasets
  train_ds = get_dataset().dataset_randomized(
      FLAGS.train_input_files).shuffle(10000).batch(FLAGS.batch_size)
  test_ds = get_dataset().dataset_randomized(FLAGS.test_input_files).batch(
      FLAGS.batch_size)

  # Create the model and optimizer
  if FLAGS.model_type == 'mlp':
    model = mlp.MLP(FLAGS.num_units, FLAGS.stddev, get_activation(),
                    FLAGS.custom_init,
                    get_dataset().NUM_LABELS)
  elif FLAGS.model_type == 'convnet':
    model = convnet.ConvNet(get_dataset().NUM_LABELS)

  optimizer = tf.keras.optimizers.SGD(FLAGS.learning_rate)

  # Create file writers for writing TensorBoard summaries.
  if FLAGS.output_dir:
    # Create directories to which summaries will be written
    # tensorboard --logdir=<output_dir>
    # can then be used to see the recorded summaries.
    train_dir = os.path.join(FLAGS.output_dir, 'train')
    test_dir = os.path.join(FLAGS.output_dir, 'eval')
    tf.io.gfile.mkdir(FLAGS.output_dir)
  else:
    train_dir = None
    test_dir = None
  summary_writer = tf.summary.create_file_writer(train_dir, flush_millis=10000)
  test_summary_writer = tf.summary.create_file_writer(
      test_dir, flush_millis=10000, name='test')

  # Create and restore checkpoint (if one exists on the path)
  checkpoint_prefix = os.path.join(FLAGS.model_dir, 'ckpt')
  checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
  # Restore variables on creation if a checkpoint exists.
  checkpoint.restore(tf.train.latest_checkpoint(FLAGS.model_dir))

  # Train and evaluate for a set number of epochs.
  with tf.device(device):
    for _ in range(FLAGS.train_epochs):
      start = time.time()
      with summary_writer.as_default():
        train(model, optimizer, train_ds, optimizer.iterations,
              FLAGS.log_interval)
      end = time.time()
      print('\nTrain time for epoch #%d (%d total steps): %f' %
            (checkpoint.save_counter.numpy() + 1, optimizer.iterations.numpy(),
             end - start))
      with test_summary_writer.as_default():
        test(model, test_ds, optimizer.iterations)
      checkpoint.save(checkpoint_prefix)


def main(_):
  run_eager()


if __name__ == '__main__':
  app.run(main)
