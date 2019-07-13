# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Train model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

from large_margin import margin_loss
from large_margin.mnist import data_provider as mnist
from large_margin.mnist import mnist_config
from large_margin.mnist import mnist_model

flags.DEFINE_integer("num_epochs", 200, "Number of training epochs.")
flags.DEFINE_string("master", "local",
                    "BNS name of the TensorFlow master to use.")
flags.DEFINE_integer("task", 0, "Task id of the replica running the training.")
flags.DEFINE_integer("ps_tasks", 0,
                     "Number of tasks in the ps job. If 0 no ps job is used.")
flags.DEFINE_integer("num_replicas", 0, "Number of workers in the job.")
flags.DEFINE_string("checkpoint_dir", "/tmp/large_margin/train/",
                    "Results directory.")
flags.DEFINE_float("momentum", 0.0,
                   "Momentum value if used momentum optimizer.")
flags.DEFINE_integer("decay_steps", 2000,
                     "Number of steps to decay learning rate")
flags.DEFINE_float("decay_rate", 0.96, "Rate of decay")
flags.DEFINE_integer("batch_size", 256, "Training batch size.")
flags.DEFINE_integer("log_every_steps", 50,
                     "Saving logging frequency in optimization steps.")
flags.DEFINE_integer("save_checkpoint_secs", 2 * 60,
                     "Saving checkpoints frequency in secs.")
flags.DEFINE_integer("save_summaries_secs", 2 * 60,
                     "Saving summaries frequency in secs.")
flags.DEFINE_enum("experiment_type", "mnist", ["mnist"], "Experiment type.")
flags.DEFINE_integer("startup_delay_steps", 15,
                     "Number of training steps between replicas startup.")
flags.DEFINE_string("data_dir", "", "Data directory.")

FLAGS = flags.FLAGS


def train():
  """Training function."""
  is_chief = (FLAGS.task == 0)
  g = tf.Graph()
  with g.as_default():
    with tf.device(tf.train.replica_device_setter(ps_tasks=FLAGS.ps_tasks)):
      if FLAGS.experiment_type == "mnist":
        config = mnist_config.ConfigDict()
        dataset = mnist.MNIST(
            data_dir=FLAGS.data_dir,
            subset="train",
            batch_size=FLAGS.batch_size,
            is_training=True)
        model = mnist_model.MNISTNetwork(config)
        layers_names = [
            "conv_layer%d" % i
            for i in xrange(len(config.filter_sizes_conv_layers))
        ]

      images, labels, num_examples, num_classes = (dataset.images,
                                                   dataset.labels,
                                                   dataset.num_examples,
                                                   dataset.num_classes)
      tf.summary.image("images", images)

      # Build model.
      logits, endpoints = model(images, is_training=True)
      layers_list = [images] + [endpoints[name] for name in layers_names]

      # Define losses.
      l2_loss_wt = config.l2_loss_wt
      xent_loss_wt = config.xent_loss_wt
      margin_loss_wt = config.margin_loss_wt
      gamma = config.gamma
      alpha = config.alpha
      top_k = config.top_k
      dist_norm = config.dist_norm
      with tf.name_scope("losses"):
        xent_loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))
        margin = margin_loss.large_margin(
            logits=logits,
            one_hot_labels=tf.one_hot(labels, num_classes),
            layers_list=layers_list,
            gamma=gamma,
            alpha_factor=alpha,
            top_k=top_k,
            dist_norm=dist_norm,
            epsilon=1e-6,
            layers_weights=[
                np.prod(layer.get_shape().as_list()[1:])
                for layer in layers_list] if np.isinf(dist_norm) else None
            )

        l2_loss = 0.
        for v in tf.trainable_variables():
          tf.logging.info(v)
          l2_loss += tf.nn.l2_loss(v)

        total_loss = 0
        if xent_loss_wt > 0:
          total_loss += xent_loss_wt * xent_loss
        if margin_loss_wt > 0:
          total_loss += margin_loss_wt * margin
        if l2_loss_wt > 0:
          total_loss += l2_loss_wt * l2_loss

        tf.summary.scalar("xent_loss", xent_loss)
        tf.summary.scalar("margin_loss", margin)
        tf.summary.scalar("l2_loss", l2_loss)
        tf.summary.scalar("total_loss", total_loss)

      # Build optimizer.
      init_lr = config.init_lr
      with tf.name_scope("optimizer"):
        global_step = tf.train.get_or_create_global_step()
        if FLAGS.num_replicas > 1:
          num_batches_per_epoch = num_examples // (
              FLAGS.batch_size * FLAGS.num_replicas)
        else:
          num_batches_per_epoch = num_examples // FLAGS.batch_size
        max_iters = num_batches_per_epoch * FLAGS.num_epochs

        lr = tf.train.exponential_decay(init_lr,
                                        global_step,
                                        FLAGS.decay_steps,
                                        FLAGS.decay_rate,
                                        staircase=True,
                                        name="lr_schedule")

        tf.summary.scalar("learning_rate", lr)

        var_list = tf.trainable_variables()
        grad_vars = tf.gradients(total_loss, var_list)
        tf.summary.scalar(
            "grad_norm",
            tf.reduce_mean([tf.norm(grad_var) for grad_var in grad_vars]))
        grad_vars, _ = tf.clip_by_global_norm(grad_vars, 5.0)

        opt = tf.train.RMSPropOptimizer(lr, momentum=FLAGS.momentum,
                                        epsilon=1e-2)
        if FLAGS.num_replicas > 1:
          opt = tf.train.SyncReplicasOptimizer(
              opt,
              replicas_to_aggregate=FLAGS.num_replicas,
              total_num_replicas=FLAGS.num_replicas)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
          opt_op = opt.apply_gradients(
              zip(grad_vars, var_list), global_step=global_step)

      # Compute accuracy.
      top1_op = tf.nn.in_top_k(logits, labels, 1)
      accuracy = tf.reduce_mean(tf.cast(top1_op, dtype=tf.float32))
      tf.summary.scalar("top1_accuracy", accuracy)

      # Prepare optimization.
      vars_to_save = tf.global_variables()
      saver = tf.train.Saver(var_list=vars_to_save, max_to_keep=5, sharded=True)
      merged_summary = tf.summary.merge_all()

      # Hooks for optimization.
      hooks = [tf.train.StopAtStepHook(last_step=max_iters)]
      if not is_chief:
        hooks.append(
            tf.train.GlobalStepWaiterHook(
                FLAGS.task * FLAGS.startup_delay_steps))

      init_op = tf.global_variables_initializer()
      scaffold = tf.train.Scaffold(
          init_op=init_op, summary_op=merged_summary, saver=saver)

      # Run optimization.
      epoch = 0
      with tf.train.MonitoredTrainingSession(
          is_chief=is_chief,
          checkpoint_dir=FLAGS.checkpoint_dir,
          hooks=hooks,
          save_checkpoint_secs=FLAGS.save_checkpoint_secs,
          save_summaries_secs=FLAGS.save_summaries_secs,
          scaffold=scaffold) as sess:
        while not sess.should_stop():
          _, acc, i = sess.run((opt_op, accuracy, global_step))
          epoch = i // num_batches_per_epoch
          if (i % FLAGS.log_every_steps) == 0:
            tf.logging.info("global step %d: epoch %d:\n train accuracy %.3f" %
                            (i, epoch, acc))


def main(argv):
  del argv  # Unused
  # Training.
  train()


if __name__ == "__main__":
  app.run(main)
