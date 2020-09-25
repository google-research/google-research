# coding=utf-8
# Copyright 2020 The Google Research Authors.
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

"""Training loop for object discovery with Slot Attention."""
import datetime
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import slot_attention.data as data_utils
import slot_attention.model as model_utils
import slot_attention.utils as utils


FLAGS = flags.FLAGS
flags.DEFINE_string("model_dir", "/tmp/object_discovery/",
                    "Where to save the checkpoints.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_integer("batch_size", 64, "Batch size for the model.")
flags.DEFINE_integer("num_slots", 7, "Number of slots in Slot Attention.")
flags.DEFINE_integer("num_iterations", 3, "Number of attention iterations.")
flags.DEFINE_float("learning_rate", 0.0004, "Learning rate.")
flags.DEFINE_integer("num_train_steps", 500000, "Number of training steps.")
flags.DEFINE_integer("warmup_steps", 10000,
                     "Number of warmup steps for the learning rate.")
flags.DEFINE_float("decay_rate", 0.5, "Rate for the learning rate decay.")
flags.DEFINE_integer("decay_steps", 100000,
                     "Number of steps for the learning rate decay.")


# We use `tf.function` compilation to speed up execution. For debugging,
# consider commenting out the `@tf.function` decorator.
@tf.function
def train_step(batch, model, optimizer):
  """Perform a single training step."""

  # Get the prediction of the models and compute the loss.
  with tf.GradientTape() as tape:
    preds = model(batch["image"], training=True)
    recon_combined, recons, masks, slots = preds
    loss_value = utils.l2_loss(batch["image"], recon_combined)
    del recons, masks, slots  # Unused.

  # Get and apply gradients.
  gradients = tape.gradient(loss_value, model.trainable_weights)
  optimizer.apply_gradients(zip(gradients, model.trainable_weights))

  return loss_value


def main(argv):
  del argv
  # Hyperparameters of the model.
  batch_size = FLAGS.batch_size
  num_slots = FLAGS.num_slots
  num_iterations = FLAGS.num_iterations
  base_learning_rate = FLAGS.learning_rate
  num_train_steps = FLAGS.num_train_steps
  warmup_steps = FLAGS.warmup_steps
  decay_rate = FLAGS.decay_rate
  decay_steps = FLAGS.decay_steps
  tf.random.set_seed(FLAGS.seed)
  resolution = (128, 128)

  # Build dataset iterators, optimizers and model.
  data_iterator = data_utils.build_clevr_iterator(
      batch_size, split="train", resolution=resolution, shuffle=True,
      max_n_objects=6, get_properties=False, apply_crop=True)

  optimizer = tf.keras.optimizers.Adam(base_learning_rate, epsilon=1e-08)

  model = model_utils.build_model(resolution, batch_size, num_slots,
                                  num_iterations, model_type="object_discovery")

  # Prepare checkpoint manager.
  global_step = tf.Variable(
      0, trainable=False, name="global_step", dtype=tf.int64)
  ckpt = tf.train.Checkpoint(
      network=model, optimizer=optimizer, global_step=global_step)
  ckpt_manager = tf.train.CheckpointManager(
      checkpoint=ckpt, directory=FLAGS.model_dir, max_to_keep=5)
  ckpt.restore(ckpt_manager.latest_checkpoint)
  if ckpt_manager.latest_checkpoint:
    logging.info("Restored from %s", ckpt_manager.latest_checkpoint)
  else:
    logging.info("Initializing from scratch.")

  start = time.time()
  for _ in range(num_train_steps):
    batch = next(data_iterator)

    # Learning rate warm-up.
    if global_step < warmup_steps:
      learning_rate = base_learning_rate * tf.cast(
          global_step, tf.float32) / tf.cast(warmup_steps, tf.float32)
    else:
      learning_rate = base_learning_rate
    learning_rate = learning_rate * (decay_rate ** (
        tf.cast(global_step, tf.float32) / tf.cast(decay_steps, tf.float32)))
    optimizer.lr = learning_rate.numpy()

    loss_value = train_step(batch, model, optimizer)

    # Update the global step. We update it before logging the loss and saving
    # the model so that the last checkpoint is saved at the last iteration.
    global_step.assign_add(1)

    # Log the training loss.
    if not global_step % 100:
      logging.info("Step: %s, Loss: %.6f, Time: %s",
                   global_step.numpy(), loss_value,
                   datetime.timedelta(seconds=time.time() - start))

    # We save the checkpoints every 1000 iterations.
    if not global_step  % 1000:
      # Save the checkpoint of the model.
      saved_ckpt = ckpt_manager.save()
      logging.info("Saved checkpoint: %s", saved_ckpt)


if __name__ == "__main__":
  app.run(main)
