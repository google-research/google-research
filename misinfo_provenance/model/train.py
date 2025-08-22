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

# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Near-Duplication SSL Train script."""

import io
import os
import time

from absl import app
from absl import flags
from absl import logging
from dataset import cutmix
from matplotlib import pyplot as plt
from ndclr import NDCLR
from ndclr import ssl_local_loss
from ndclr import ssl_loss
import numpy as np
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
from validation import perform_eval

_DEBUG = flags.DEFINE_boolean('debug', False, 'Debug mode.')
_LOGDIR = flags.DEFINE_string('logdir', '/tmp/ssl', 'WithTensorBoard logdir.')

_TRAIN_FILE_PATTERN = flags.DEFINE_string(
    'train_file_pattern', '/tmp/data/train*',
    'File pattern of training dataset files.')
_VALIDATION_FILE_PATTERN = flags.DEFINE_string(
    'validation_file_pattern', '/tmp/data/validation*',
    'File pattern of validation dataset files.')
_SEED = flags.DEFINE_integer('seed', 0, 'Seed to training dataset.')
_INITIAL_LR = flags.DEFINE_float('initial_lr', 0.1, 'Initial learning rate.')
_BATCH_SIZE = flags.DEFINE_integer('batch_size', 32, 'Global batch size.')
_IMAGE_SIZE = flags.DEFINE_integer('image_size', 512,
                                   'Size of each image side to use.')
_LOCAL_TEMPERATURE = flags.DEFINE_float('local_temperature', 0.01,
                                        'Temperature of local loss.')
_INFONCE_TEMPERATURE = flags.DEFINE_float('infonce_temperature', 0.05,
                                          'Temperature of InfoNCE Loss.')
_ENTROPY_WEIGHT = flags.DEFINE_float('entropy_weight', 30, 'Entropy weight.')
_LOSS_TYPE = flags.DEFINE_string('loss_type', 'ssl', 'Loss type.')
_LOSS_WEIGHT = flags.DEFINE_float('loss_weight', 1.0, 'Loss weight.')
_LOCAL_EMB_DIMS = flags.DEFINE_integer('local_emb_dims', 2048,
                                       'Local embedding dimension.')
_BLOCK_LOCAL_DIMS = flags.DEFINE_integer(
    'block_local_dims', 8, 'Block Local dimension (height, width).')
_CONV_OUTPUT_LAYER = flags.DEFINE_string('conv_output_layer',
                                         'conv4_block5_out',
                                         'Local Feat Out Layer.')
_CONV_OUTPUT_LAYER = flags.DEFINE_string('conv_output_layer',
                                         'conv4_block5_out',
                                         'Local Feat Out Layer.')
_DATASET_LENGHT = flags.DEFINE_integer('dataset_lenght', 10000,
                                       'Dataset Lenght.')
_EPOCHS = flags.DEFINE_integer('epochs', 10, 'Epochs.')

N_CLASSES = 1000


def read_labeled_tfrecord(record, image_size=256, block_local_dims=256):
  """Read tfrecords.

  Args:
    record: <tf.Example> The tf records to be read.
    image_size: <int> Image Size to resize the image within the tf records.
    block_local_dims: <int> Number of dimensions of each block extracted from
                      the convolutional layer.

  Returns:
  """
  name_to_features = {
      'height':
          tf.io.FixedLenFeature([], tf.int64),
      'width':
          tf.io.FixedLenFeature([], tf.int64),
      'bands':
          tf.io.FixedLenFeature([], tf.int64),
      'image_raw':
          tf.io.FixedLenFeature([], tf.string),
      't1_image':
          tf.io.FixedLenFeature([], tf.string),
      't2_image':
          tf.io.FixedLenFeature([], tf.string),
      f't1_mask_{block_local_dims}':
          tf.io.FixedLenFeature([block_local_dims * block_local_dims],
                                dtype=tf.float32),
      # f't2_mask_{block_local_dims}':
      # tf.io.FixedLenFeature([block_local_dims * block_local_dims],
      # dtype=tf.float32),
      'label':
          tf.io.FixedLenFeature([], tf.int64),
  }

  record = tf.io.parse_single_example(record, name_to_features)

  image = tf.io.decode_raw(
      record['image_raw'],
      out_type=tf.uint8,
      little_endian=True,
      fixed_length=None,
      name=None)
  height, width, bands = record['height'], record['width'], record['bands']

  t1_image = tf.io.decode_raw(
      record['t1_image'],
      out_type=tf.uint8,
      little_endian=True,
      fixed_length=None,
      name=None)

  t2_image = tf.io.decode_raw(
      record['t2_image'],
      out_type=tf.uint8,
      little_endian=True,
      fixed_length=None,
      name=None)
  label = record['label']
  label = tf.one_hot(label, N_CLASSES)

  fixed_size = 256

  image = tf.reshape(image, (height, width, bands))
  image = tf.image.resize(image, (image_size, image_size))
  image = image / 255.0

  t1_image = tf.reshape(t1_image, (fixed_size, fixed_size, 3))
  t1_image = tf.image.resize(t1_image, (image_size, image_size))
  t1_image = tf.cast(t1_image, tf.float32)
  t1_image = t1_image / 255.0

  t2_image = tf.reshape(t2_image, (fixed_size, fixed_size, 3))
  t2_image = tf.image.resize(t2_image, (image_size, image_size))
  t2_image = tf.cast(t2_image, tf.float32)
  t2_image = t2_image / 255.0

  t1_mask = record[f't1_mask_{block_local_dims}']
  t1_mask = tf.reshape(t1_mask, (block_local_dims, block_local_dims))
  t1_mask = tf.cast(t1_mask, tf.int32)

  # t2_mask = record[f't2_mask_{block_local_dims}']
  # t2_mask = tf.reshape(t2_mask, (block_local_dims, block_local_dims))
  # t2_mask = tf.cast(t2_mask, tf.int32)

  return (image, t1_image, t2_image, t1_mask)


def read_dataset(file_pattern,
                 data_type,
                 batch_size=2):
  """Read TFRecords from file_pattern."""
  # Get Batch of images
  ignore_order = tf.data.Options()
  auto = tf.data.experimental.AUTOTUNE

  dataset = tf.data.Dataset.list_files(file_pattern)
  dataset = dataset.flat_map(tf.data.TFRecordDataset)

  dataset.with_options(ignore_order)
  if data_type == 'train':
    dataset = dataset.map(
        read_labeled_tfrecord,
        num_parallel_calls=auto)
    dataset = dataset.shuffle(2048, seed=_SEED)
  else:
    dataset = dataset.map(
        read_labeled_tfrecord,
        num_parallel_calls=auto)
  dataset = dataset.batch(batch_size)
  dataset = dataset.prefetch(auto)
  dataset = dataset.repeat(-1)
  return dataset


# Read Dataset Directly from imagenet Repo
def read_imagenet_dataset(data_type, batch_size=32):
  """Read TFRecords from file_pattern."""
  # Get Batch of images
  ignore_order = tf.data.Options()
  auto = tf.data.experimental.AUTOTUNE

  imagenet2012 = tfds.builder('imagenet2012')

  dataset = imagenet2012.as_dataset(
      split=data_type,
      decoders={'image': tfds.decode.SkipDecoding()},
      batch_size=None)
  dataset = dataset.map(
      read_labeled_tfrecord,
      num_parallel_calls=auto)
  dataset = dataset.with_options(ignore_order)

  # dataset = dataset.shuffle(2048, seed=_SEED)
  dataset = dataset.batch(batch_size)
  dataset = dataset.repeat(-1)
  return dataset


def output_layer_visualization(model,
                               global_step,
                               image_path='photoshops/17fkpo/c851zwh_0.jpg'):
  """Include a visualization of the output layer into the Tensorboard.

  The visualization is the norm of each block embedding from output layer.
  Args:
    model:
    global_step:
    image_path:
  """
  pil_img = Image.open(image_path, 'rb')
  image = tf.keras.preprocessing.image.img_to_array(pil_img)
  image = tf.image.resize(image, (256, 256))
  image = image / 255.0
  image = tf.expand_dims(image, axis=0)
  conv2_block3 = tf.keras.Model(
      inputs=model.backbone.input,
      outputs=model.backbone.get_layer('conv2_block3_out').output)
  norm_vis = tf.norm(conv2_block3(image)[0], axis=-1)

  plt.figure(figsize=(3, 3), dpi=300)
  plt.imshow(norm_vis)
  plt.axis('off')
  fig = plt.gcf()

  buf = io.BytesIO()
  fig.savefig(buf)
  buf.seek(0)
  img_array = np.array(Image.open(buf).convert('RGB'), dtype=np.uint8)
  img_array = np.expand_dims(img_array, axis=0)
  img_tf = tf.cast(img_array, dtype=tf.uint8)
  # Input image-related summaries.
  tf.summary.image('norm_visualization', img_tf, step=global_step)
  plt.close()


def learning_rate_schedule(global_step_value, max_iters, initial_lr):
  """Calculates learning_rate with linear decay.

  Args:
    global_step_value: int, global step.
    max_iters: int, maximum iterations.
    initial_lr: float, initial learning rate.

  Returns:
    lr: float, learning rate.
  """
  lr = max(initial_lr * (1.0 - global_step_value / (0.2 * max_iters)), 0.001)
  return lr


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  #-------------------------------------------------------------
  # Log flags used.
  logging.info('Running training script with\n')
  logging.info('logdir= %s', _LOGDIR.value)
  logging.info('initial_lr= %f', _INITIAL_LR.value)

  # ------------------------------------------------------------
  # Create the strategy.
  strategy = tf.distribute.MirroredStrategy()
  logging.info('Number of devices: %d', strategy.num_replicas_in_sync)
  if _DEBUG.value:
    print('Number of devices:', strategy.num_replicas_in_sync)

  # Training Parameters
  global_batch_size = _BATCH_SIZE.value
  max_iters = _EPOCHS.value * _DATASET_LENGHT.value // global_batch_size
  num_eval_batches = int(50000 // global_batch_size)
  eval_interval = 3200  # 0.5epoch
  retrieval_eval_interval = eval_interval
  # pylint: disable=invalid-name
  PIR_EVAL = eval_interval
  # pylint: enable=invalid-name
  save_interval = eval_interval
  report_interval = 100

  loss_type = _LOSS_TYPE.value
  local_emb_dims = _LOCAL_EMB_DIMS.value
  block_local_dims = _BLOCK_LOCAL_DIMS.value
  conv_output_layer = _CONV_OUTPUT_LAYER.value

  local_temperature = _LOCAL_TEMPERATURE.value
  infonce_temperature = _INFONCE_TEMPERATURE.value
  entropy_weight = _ENTROPY_WEIGHT.value
  loss_weight = _LOSS_WEIGHT.value

  initial_lr = _INITIAL_LR.value

  if _DEBUG.value:
    tf.config.run_functions_eagerly(True)
    global_batch_size = 4
    max_iters = 100
    save_interval = 200
    report_interval = 10

  # Create the distributed train/validation sets.
  train_dataset = read_dataset(
      file_pattern=_TRAIN_FILE_PATTERN.value,
      data_type='train',
      batch_size=global_batch_size,
  )

  val_dataset = read_dataset(
      file_pattern=_VALIDATION_FILE_PATTERN.value,
      data_type='validation',
      batch_size=global_batch_size,
  )

  train_dist_dataset = strategy.experimental_distribute_dataset(train_dataset)
  val_dist_dataset = strategy.experimental_distribute_dataset(val_dataset)

  train_iter = iter(train_dist_dataset)
  validation_iter = iter(val_dist_dataset)
  logging.info('Dataset ready')

  # Create a checkpoint directory to store the checkpoints.
  checkpoint_prefix = os.path.join(_LOGDIR.value, 'test-chkps')

  # Finally, we do everything in distributed scope.
  with strategy.scope():
    # Compute loss.
    # Set reduction to `none` so we can do the reduction afterwards and divide
    # by global batch size.
    # loss_object = tf.keras.losses.CategoricalCrossentropy(
    #   reduction=tf.keras.losses.Reduction.NONE)

    # Set up metrics.
    total_train_loss = tf.keras.metrics.Mean(name='train_loss')
    ssl_train_loss = tf.keras.metrics.Mean(name='ssl_train_loss')
    ssl_val_loss = tf.keras.metrics.Mean(name='ssl_val_loss')
    infonce_val_loss = tf.keras.metrics.Mean(name='infonce_val_loss')
    ssl_val_entropy_loss = tf.keras.metrics.Mean(name='ssl_val_entropy_loss')
    infonce_train_loss = tf.keras.metrics.Mean(name='infonce_train_loss')
    local_train_loss = tf.keras.metrics.Mean(name='local_train_loss')
    local_val_loss = tf.keras.metrics.Mean(name='local_val_loss')
    ssl_train_entropy_loss = tf.keras.metrics.Mean(
        name='ssl_train_entropy_loss')

    # Create model.
    model = NDCLR(
        batch_size=global_batch_size,
        local_emb_dims=local_emb_dims,
        block_local_dims=block_local_dims,
        conv_output_layer=conv_output_layer)
    logging.info('Model and dataset loaded')

    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    decay_steps = 30 * eval_interval
    lr_decayed_fn = tf.keras.optimizers.schedules.CosineDecay(
        initial_lr, decay_steps, 0.001)

    # Setup summary writer.
    summary_writer = tf.summary.create_file_writer(
        os.path.join(_LOGDIR.value, f'entropy_weights{entropy_weight}'),
        flush_millis=1000)

    # Setup checkpoint directory.
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        checkpoint_prefix,
        max_to_keep=10,
        keep_checkpoint_every_n_hours=3)
    # Restores the checkpoint, if existing.
    checkpoint.restore(manager.latest_checkpoint)

    # Train Step.
    def train_step(inputs):
      """Train one batch."""
      batch_images, t1_images, t2_images, t1_masks = inputs

      apply_cutmix = False
      if np.random.choice([0, 1], p=[0.5, 0.5]):
        apply_cutmix = True
        # include cutmix
        inv_images = tf.reverse(t2_images, axis=[0])
        global_transformed_images = cutmix(t1_images, inv_images)
      else:
        global_transformed_images = t2_images

      # Step number, for summary purposes.
      global_batch_images = tf.concat([t1_images, global_transformed_images],
                                      axis=0)  # N original images

      def _backprop_loss(tape, loss, weights):
        """Backpropogate losses using clipped gradients.

        Args:
          tape: gradient tape.
          loss: scalar Tensor, loss value.
          weights: keras model weights.
        """
        # For some reason, this error:
        # Tensor conversion requested dtype float32 for Tensor with
        # dtype float64:
        # Trying to some with cast
        loss = tf.cast(loss, dtype=tf.float32)
        gradients = tape.gradient(loss, weights)
        clipped, _ = tf.clip_by_global_norm(gradients, clip_norm=model.clip_val)
        optimizer.apply_gradients(zip(clipped, weights))

      with tf.GradientTape() as gradient_tape:
        # Make a forward pass to calculate the embeddings.
        # probs, embeddings = model(batch_images, training=True) #classification

        if 'global' in loss_type:
          embeddings = model(global_batch_images, training=True)

          # Self-Supervised Loss.
          ssl_labels = model.prepare_gt(
              embeddings.shape[0] // 2, mixup=apply_cutmix)
          selfsupervised_loss, infoloss, entropy_loss = ssl_loss(
              embeddings,
              ssl_labels,
              infonce_temperature=infonce_temperature,
              entropy_weight=entropy_weight,
              mixup=apply_cutmix)
        else:
          selfsupervised_loss, infoloss, entropy_loss = 0.0, 0.0, 0.0

        # Extract local features and calculate loss.
        if 'local' in loss_type:
          embs = model.local_feat_layer(batch_images, training=True)
          t_embs = model.local_feat_layer(t1_images, training=True)

          local_loss = ssl_local_loss(
              embs, t_embs, t1_masks, temperature=local_temperature)

        else:
          local_loss = 0

        total_loss = loss_weight * selfsupervised_loss + (
            1 - loss_weight) * local_loss

      # Perform backpropagation through the descriptor.
      _backprop_loss(gradient_tape, total_loss, model.trainable_weights)

      # Record train metrics.
      total_train_loss.update_state(total_loss)
      local_train_loss.update_state(local_loss)
      ssl_train_loss.update_state(selfsupervised_loss)
      infonce_train_loss.update_state(infoloss)
      ssl_train_entropy_loss.update_state(entropy_loss)

      return total_loss

    # Validation Step.
    def validation_step(inputs):
      """Validate one batch."""
      batch_images, t1_images, t2_images, t1_masks = inputs

      global_batch_images = tf.concat([t1_images, t2_images],
                                      axis=0)  # N original images
      # N transformed images.
      if 'global' in loss_type:
        embeddings = model(global_batch_images, training=False)

        # Self-Supervised Loss.
        ssl_labels = model.prepare_gt(embeddings.shape[0] // 2)
        selfsupervised_loss, infoloss, entropy_loss = ssl_loss(
            embeddings,
            ssl_labels,
            infonce_temperature=infonce_temperature,
            entropy_weight=entropy_weight)

      else:
        selfsupervised_loss, infoloss, entropy_loss = 0.0, 0.0, 0.0

      # Extract local features and calculate loss.
      if 'local' in loss_type:
        embs = model.local_feat_layer(batch_images, training=False)
        t_embs = model.local_feat_layer(t1_images, training=False)

        local_loss = ssl_local_loss(
            embs, t_embs, t1_masks, temperature=local_temperature)

      else:
        local_loss = 0

      total_loss = loss_weight * selfsupervised_loss + (
          1 - loss_weight) * local_loss

      # Record Validation.
      local_val_loss.update_state(local_loss)
      infonce_val_loss.update_state(infoloss)
      ssl_val_entropy_loss.update_state(entropy_loss)
      ssl_val_loss.update_state(selfsupervised_loss)

      return total_loss  # selfsupervised_loss

    @tf.function
    def distributed_train_step(dataset_inputs):
      """Get the actual losses."""
      # Each (desc, attn) is a list of 3 losses - crossentropy, reg, total.
      train_loss = strategy.run(train_step, args=(dataset_inputs,))

      # Reduce over the replicas.
      total_loss = strategy.reduce(
          tf.distribute.ReduceOp.SUM, train_loss, axis=None)

      return total_loss  #, attn_global_loss, recon_global_loss

    @tf.function
    def distributed_validation_step(dataset_inputs):
      val_loss = strategy.run(validation_step, args=(dataset_inputs,))
      val_loss = strategy.reduce(
          tf.distribute.ReduceOp.SUM, val_loss, axis=None)
      return val_loss

    # Write Summary.
    logging.info('==== Started Training ====')
    logging.info('==== Training ====')
    with summary_writer.as_default():
      record_cond = lambda: tf.equal(optimizer.iterations % report_interval, 0)
      with tf.summary.record_if(record_cond):
        global_step_value = optimizer.iterations.numpy()

        last_summary_step_value = None
        last_summary_time = None
        while global_step_value < max_iters:
          try:
            # input_batch <- (image, t1_image, t2_image, t1_mask)
            input_batch = next(train_iter)
          except tf.errors.OutOfRangeError:
            # Break if we run out of data in the dataset.
            logging.info('Stopping training at global step %d, no more data',
                         global_step_value)
            break
          except StopIteration:
            logging.info('Stopping training at global step %d, no more data',
                         global_step_value)
            break

          # Set learning rate and run the training step over num_gpu gpus.
          optimizer.learning_rate = lr_decayed_fn(optimizer.iterations.numpy())

          train_loss = distributed_train_step(input_batch)

          # Step number, to be used for summary/logging.
          global_step = optimizer.iterations
          global_step_value = global_step.numpy()

          # losses and metrics summaries.
          tf.summary.scalar(
              'learning_rate', optimizer.learning_rate, step=global_step)
          tf.summary.scalar('loss/train_ssl_loss', train_loss, step=global_step)
          tf.summary.scalar(
              'loss/train_ssl_entropy_loss',
              ssl_train_entropy_loss.result().numpy(),
              step=global_step)
          tf.summary.scalar(
              'loss/train_ssl_infonce_loss',
              infonce_train_loss.result().numpy(),
              step=global_step)

          if 'local' in loss_type:
            tf.summary.scalar(
                'loss/train_local_loss',
                local_train_loss.result().numpy(),
                step=global_step)

          # Summary for number of global steps taken per second.
          if global_step_value % retrieval_eval_interval == 0:
            if global_step_value % PIR_EVAL == 0:
              logging.info('==== Retrieval Validation at step %d ====',
                           global_step_value)
              # PIR DATASET
              # pylint: disable=invalid-name
              uAP, mAP, r_1, r_10, r_100 = perform_eval(
                  model, k=100, dataset_name='PIR')
              # pylint: enable=invalid-name
              logging.info('==== PIR uAP %f ====', uAP)
              logging.info('==== PIR mAP %f ====', mAP)
              tf.summary.scalar('validation/PIR/uAP', uAP, step=global_step)
              tf.summary.scalar('validation/PIR/mAP', mAP, step=global_step)
              tf.summary.scalar('validation/PIR/r1', r_1, step=global_step)
              tf.summary.scalar('validation/PIR/r10', r_10, step=global_step)
              tf.summary.scalar('validation/PIR/r100', r_100, step=global_step)

            # COPYDAYS-10K DATASET
            # pylint: disable=invalid-name
            uAP, mAP, r_1, r_10, r_100 = perform_eval(
                model, k=100, dataset_name='copydays10k')
            # pylint: enable=invalid-name
            logging.info('==== copydays-10k uAP %f ====', uAP)
            logging.info('==== copydays-10k mAP %f ====', mAP)
            tf.summary.scalar(
                'validation/copydays10k/uAP', uAP, step=global_step)
            tf.summary.scalar(
                'validation/copydays10k/mAP', mAP, step=global_step)
            tf.summary.scalar(
                'validation/copydays10k/r1', r_1, step=global_step)
            tf.summary.scalar(
                'validation/copydays10k/r10', r_10, step=global_step)
            tf.summary.scalar(
                'validation/copydays10k/r100', r_100, step=global_step)

            # COPYDAYS-10K STRONG DATASET
            # pylint: disable=invalid-name
            uAP, mAP, r_1, r_10, r_100 = perform_eval(
                model, k=100, dataset_name='copydays10k-strong')
            # pylint: enable=invalid-name
            logging.info('==== copydays-10k-strong uAP %f ====', uAP)
            logging.info('==== copydays-10k-strong MAP %f ====', mAP)
            tf.summary.scalar(
                'validation/copydays10k-strong/uAP', uAP, step=global_step)
            tf.summary.scalar(
                'validation/copydays10k-strong/mAP', mAP, step=global_step)
            tf.summary.scalar(
                'validation/copydays10k-strong/r1', r_1, step=global_step)
            tf.summary.scalar(
                'validation/copydays10k-strong/r10', r_10, step=global_step)
            tf.summary.scalar(
                'validation/copydays10k-strong/r100', r_100, step=global_step)

            output_layer_visualization(model, global_step)

          current_time = time.time()
          if (last_summary_step_value is not None and
              last_summary_time is not None):
            tf.summary.scalar(
                'global_steps_per_sec',
                (global_step_value - last_summary_step_value) /
                (current_time - last_summary_time),
                step=global_step)
          if tf.summary.should_record_summaries().numpy():
            last_summary_step_value = global_step_value
            last_summary_time = current_time

          # Save model
          if (global_step_value % save_interval
              == 0) or (global_step_value >= max_iters):
            save_path = manager.save(checkpoint_number=global_step_value)
            logging.info('Saved (%d) at %s', global_step_value, save_path)

            file_path = '%s/ssl-model' % _LOGDIR.value
            model.save(file_path)
            logging.info('Saved weights (%d) at %s', global_step_value,
                         file_path)

          if global_step_value % eval_interval == 0:
            logging.info('==== Validation at step %d ====', global_step_value)
            val_loss = 0
            for i in range(num_eval_batches):
              try:
                validation_batch = next(validation_iter)
                val_loss += (distributed_validation_step(validation_batch))
              except tf.errors.OutOfRangeError:
                logging.info('Stopping eval at batch %d, no more data', i)
                break

            # Mean across the batch.
            val_loss /= 100
            # Log validation results to tensorboard.
            tf.summary.scalar('loss/val_ssl_loss', val_loss, step=global_step)
            tf.summary.scalar(
                'loss/val_ssl_entropy_loss',
                ssl_val_entropy_loss.result().numpy(),
                step=global_step)
            tf.summary.scalar(
                'loss/val_ssl_infonce_loss',
                infonce_val_loss.result().numpy(),
                step=global_step)
            if 'local' in loss_type:
              tf.summary.scalar(
                  'loss/val_local_loss',
                  local_val_loss.result().numpy(),
                  step=global_step)

          # Reset metrics for next step.
          infonce_val_loss.reset_states()
          infonce_train_loss.reset_states()
          ssl_val_entropy_loss.reset_states()
          ssl_train_entropy_loss.reset_states()
          ssl_val_loss.reset_states()
          ssl_train_loss.reset_states()
          local_val_loss.reset_states()
          local_train_loss.reset_states()

    logging.info('Finished training for %d steps.', max_iters)


if __name__ == '__main__':
  app.run(main)
