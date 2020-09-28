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

"""Main script for dense/sparse inference."""
import sys
import time
from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v1 as tf

from sgk import driver
from sgk.mbv1 import config
from sgk.mbv1 import mobilenet_builder

# Crop padding for ImageNet preprocessing.
CROP_PADDING = 32

# Mean & stddev for ImageNet preprocessing.
MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]

FLAGS = flags.FLAGS

flags.DEFINE_string("runmode", "examples",
                    "Running mode: examples or imagenet.")

flags.DEFINE_string("ckpt_dir", "/tmp/ckpt/", "Checkpoint folders")

flags.DEFINE_integer("num_images", 5000, "Number of images to eval.")

flags.DEFINE_string("imagenet_glob", None, "ImageNet eval image glob.")

flags.DEFINE_string("imagenet_label", None, "ImageNet eval label file path.")

flags.DEFINE_float("width", 1.0, "Width for MobileNetV1 model.")

flags.DEFINE_float("sparsity", 0.0, "Sparsity for MobileNetV1 model.")

flags.DEFINE_bool("fuse_bnbr", False, "Whether to fuse batch norm, bias, relu.")

flags.DEFINE_integer("inner_steps", 1000, "Benchmark steps for inner loop.")

flags.DEFINE_integer("outer_steps", 100, "Benchmark steps for outer loop.")

# Disable TF2.
tf.disable_v2_behavior()


class InferenceDriver(driver.Driver):
  """Custom inference driver for MBV1."""

  def __init__(self, cfg):
    super(InferenceDriver, self).__init__(batch_size=1, image_size=224)
    self.num_classes = 1000
    self.cfg = cfg

  def build_model(self, features):
    with tf.device("gpu"):
      # Transpose the input features from NHWC to NCHW.
      features = tf.transpose(features, [0, 3, 1, 2])

      # Apply image preprocessing.
      features -= tf.constant(MEAN_RGB, shape=[3, 1, 1], dtype=features.dtype)
      features /= tf.constant(STDDEV_RGB, shape=[3, 1, 1], dtype=features.dtype)

      logits = mobilenet_builder.build_model(features, cfg=self.cfg)
      probs = tf.nn.softmax(logits)
      return tf.squeeze(probs)

  def preprocess_fn(self, image_bytes, image_size):
    """Preprocesses the given image for evaluation.

    Args:
      image_bytes: `Tensor` representing an image binary of arbitrary size.
      image_size: image size.

    Returns:
      A preprocessed image `Tensor`.
    """
    shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        ((image_size / (image_size + CROP_PADDING)) *
         tf.cast(tf.minimum(image_height, image_width), tf.float32)), tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([
        offset_height, offset_width, padded_center_crop_size,
        padded_center_crop_size
    ])

    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = tf.image.resize_bicubic([image], [image_size, image_size])[0]

    image = tf.reshape(image, [image_size, image_size, 3])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image

  def run_inference(self, ckpt_dir, image_files, labels):
    with tf.Graph().as_default(), tf.Session() as sess:
      images, labels = self.build_dataset(image_files, labels)
      probs = self.build_model(images)
      if isinstance(probs, tuple):
        probs = probs[0]

      self.restore_model(sess, ckpt_dir)

      prediction_idx = []
      prediction_prob = []
      for i in range(len(image_files)):
        # Run inference.
        out_probs = sess.run(probs)

        idx = np.argsort(out_probs)[::-1]
        prediction_idx.append(idx[:5])
        prediction_prob.append([out_probs[pid] for pid in idx[:5]])

        if i % 1000 == 0:
          logging.error("Processed %d images.", i)

      # Return the top 5 predictions (idx and prob) for each image.
      return prediction_idx, prediction_prob

  def imagenet(self, ckpt_dir, imagenet_eval_glob, imagenet_eval_label,
               num_images):
    """Eval ImageNet images and report top1/top5 accuracy.

    Args:
      ckpt_dir: str. Checkpoint directory path.
      imagenet_eval_glob: str. File path glob for all eval images.
      imagenet_eval_label: str. File path for eval label.
      num_images: int. Number of images to eval: -1 means eval the whole
        dataset.

    Returns:
      A tuple (top1, top5) for top1 and top5 accuracy.
    """
    imagenet_val_labels = [int(i) for i in tf.gfile.GFile(imagenet_eval_label)]
    imagenet_filenames = sorted(tf.gfile.Glob(imagenet_eval_glob))
    if num_images < 0:
      num_images = len(imagenet_filenames)
    image_files = imagenet_filenames[:num_images]
    labels = imagenet_val_labels[:num_images]

    pred_idx, _ = self.run_inference(ckpt_dir, image_files, labels)
    top1_cnt, top5_cnt = 0.0, 0.0
    for i, label in enumerate(labels):
      top1_cnt += label in pred_idx[i][:1]
      top5_cnt += label in pred_idx[i][:5]
      if i % 100 == 0:
        print("Step {}: top1_acc = {:4.2f}%  top5_acc = {:4.2f}%".format(
            i, 100 * top1_cnt / (i + 1), 100 * top5_cnt / (i + 1)))
        sys.stdout.flush()
    top1, top5 = 100 * top1_cnt / num_images, 100 * top5_cnt / num_images
    print("Final: top1_acc = {:4.2f}%  top5_acc = {:4.2f}%".format(top1, top5))
    return top1, top5

  def benchmark(self, ckpt_dir, outer_steps=100, inner_steps=1000):
    """Run repeatedly on dummy data to benchmark inference."""
    # Turn off Grappler optimizations.
    options = {"disable_meta_optimizer": True}
    tf.config.optimizer.set_experimental_options(options)

    # Run only the model body (no data pipeline) on device.
    features = tf.zeros([1, 3, self.image_size, self.image_size],
                        dtype=tf.float32)

    # Create the model outside the loop body.
    model = mobilenet_builder.mobilenet_generator(self.cfg)

    # Call the model once to initialize the variables. Note that
    # this should never execute.
    dummy_iteration = model(features)

    # Run the function body in a loop to amortize session overhead.
    loop_index = tf.zeros([], dtype=tf.int32)
    initial_probs = tf.zeros([self.num_classes])

    def loop_cond(idx, _):
      return tf.less(idx, tf.constant(inner_steps, dtype=tf.int32))

    def loop_body(idx, _):
      logits = model(features)
      probs = tf.squeeze(tf.nn.softmax(logits))
      return idx + 1, probs

    benchmark_op = tf.while_loop(
        loop_cond,
        loop_body, [loop_index, initial_probs],
        parallel_iterations=1,
        back_prop=False)

    with tf.Session() as sess:
      self.restore_model(sess, ckpt_dir)
      fps = []
      for idx in range(outer_steps):
        start_time = time.time()
        sess.run(benchmark_op)
        elapsed_time = time.time() - start_time
        fps.append(inner_steps / elapsed_time)
        logging.error("Iterations %d processed %f FPS.", idx, fps[-1])
      # Skip the first iteration where all the setup and allocation happens.
      fps = np.asarray(fps[1:])
      logging.error("Mean, Std, Max, Min throughput = %f, %f, %f, %f",
                    np.mean(fps), np.std(fps), fps.max(), fps.min())


def main(_):
  logging.set_verbosity(logging.ERROR)
  cfg_cls = config.get_config(FLAGS.width, FLAGS.sparsity)
  cfg = cfg_cls(FLAGS.fuse_bnbr)
  drv = InferenceDriver(cfg)

  if FLAGS.runmode == "imagenet":
    drv.imagenet(FLAGS.ckpt_dir, FLAGS.imagenet_glob, FLAGS.imagenet_label,
                 FLAGS.num_images)
  elif FLAGS.runmode == "benchmark":
    drv.benchmark(FLAGS.ckpt_dir, FLAGS.outer_steps, FLAGS.inner_steps)
  else:
    logging.error("Must specify runmode: 'benchmark' or 'imagenet'")


if __name__ == "__main__":
  app.run(main)
