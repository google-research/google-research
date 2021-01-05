# coding=utf-8
# Copyright 2021 The Google Research Authors.
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

"""Main script for dense/sparse inference."""
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np

from tensor2tensor.bin import t2t_trainer  # pylint: disable=unused-import
from tensor2tensor.layers import common_layers
from tensor2tensor.utils import hparams_lib
from tensor2tensor.utils import registry
import tensorflow.compat.v1 as tf

from sgk import driver
from sgk.transformer import transformer  # pylint: disable=unused-import

FLAGS = flags.FLAGS

flags.DEFINE_string("runmode", "examples",
                    "Running mode: examples or imagenet.")

flags.DEFINE_string("ckpt_dir", "/tmp/ckpt/", "Checkpoint folders")

flags.DEFINE_string("config", "dense", "Whether to run sparse or dense.")

flags.DEFINE_integer("inner_steps", 20, "Benchmark steps for inner loop.")

flags.DEFINE_integer("outer_steps", 11, "Benchmark steps for outer loop.")

# Disable TF2.
tf.disable_v2_behavior()


# HACK: Get around T2T bug by defaulting to tf.layers.
def layers():
  return tf.layers


common_layers.layers = layers


class InferenceDriver(driver.Driver):
  """Inference driver for Transformer models."""

  def __init__(self, model_name, hparams_set):
    super(InferenceDriver, self).__init__(batch_size=1, image_size=64)
    self.model_name = model_name
    self.hparams_set = hparams_set
    self.problem_name = "image_imagenet64_gen_flat_rev"
    self.num_classes = 256

  def benchmark(self, ckpt_dir, outer_steps=100, inner_steps=1000):
    """Run repeatedly on dummy data to benchmark inference."""
    # Turn off Grappler optimizations.
    options = {"disable_meta_optimizer": True}
    tf.config.optimizer.set_experimental_options(options)

    # Create the model outside the loop body.
    hparams = registry.hparams(self.hparams_set)
    hparams_lib.add_problem_hparams(hparams, self.problem_name)
    model_cls = registry.model(self.model_name)
    model = model_cls(hparams, tf.estimator.ModeKeys.EVAL)

    # Run only the model body (no data pipeline) on device.
    feature_shape = [hparams.batch_size, 3 * self.image_size * self.image_size]
    features = {"targets": tf.zeros(feature_shape, dtype=tf.int32)}

    # Call the model once to initialize the variables. Note that
    # this should never execute.
    with tf.variable_scope(self.model_name) as vso:
      transformed_features = model.bottom(features)
      with tf.variable_scope("body") as vsi:
        body_out = model.body(transformed_features)
      logits = model.top(body_out, features)
      model.loss(logits, features)

    def call_model(features):
      with tf.variable_scope(vso, reuse=tf.AUTO_REUSE):
        transformed_features = model.bottom(features)
        with tf.variable_scope(vsi, reuse=tf.AUTO_REUSE):
          body_out = model.body(transformed_features)
        logits = model.top(body_out, features)
        return model.loss(logits, features)

    # Run the function body in a loop to amortize session overhead.
    loop_index = tf.zeros([], dtype=tf.int32)
    initial_loss = (tf.zeros([]), tf.zeros([]))

    def loop_cond(idx, _):
      return tf.less(idx, tf.constant(inner_steps, dtype=tf.int32))

    def loop_body(idx, _):
      return idx + 1, call_model(features)

    benchmark_op = tf.while_loop(
        loop_cond,
        loop_body, [loop_index, initial_loss],
        parallel_iterations=1,
        back_prop=False)

    session_config = tf.ConfigProto(
        gpu_options=tf.GPUOptions(
            allow_growth=False, per_process_gpu_memory_fraction=0.95))
    run_metadata = tf.RunMetadata()
    with tf.Session(config=session_config) as sess:
      self.restore_model(sess, ckpt_dir)
      tps = []
      for idx in range(outer_steps):
        start_time = time.time()
        sess.run(benchmark_op, run_metadata=run_metadata)
        elapsed_time = time.time() - start_time
        tps.append(inner_steps * hparams.batch_size * (64 * 64 * 3) /
                   elapsed_time)
        logging.error("Iterations %d processed %f TPS.", idx, tps[-1])
      # Skip the first iteration where all the setup and allocation happens.
      tps = np.asarray(tps[1:])
      logging.error("Mean/Std/Max/Min throughput = %f / %f / %f / %f",
                    np.mean(tps), np.std(tps), tps.max(), tps.min())



def main(_):
  logging.set_verbosity(logging.INFO)

  model_name = "sparse_transformer"
  hparams_set = "fast_transformer_imagenet64x64"
  if FLAGS.config == "sparse":
    hparams_set = "fast_sparse_transformer_imagenet64x64"
  drv = InferenceDriver(model_name, hparams_set)

  if FLAGS.runmode == "benchmark":
    drv.benchmark(FLAGS.ckpt_dir, FLAGS.outer_steps, FLAGS.inner_steps)
  else:
    logging.error("Must specify runmode: 'benchmark'")


if __name__ == "__main__":
  app.run(main)
