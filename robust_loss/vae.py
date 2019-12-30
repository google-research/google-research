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

"""Code for training a VAE using our adaptive loss on the Celeb-A dataset.

This code is a fork of TensorFlow Probability's example code at
  tensorflow_probability/examples/vae.py
See that code for a deeper documentation of how Variational Auto-Encoders work.
The changes made to that code are:
- Changing the dataset from MNIST to Celeb-A.
- Changing the loss/distribution used to compute distortion from a normal
  distribution on pixels to our adaptive loss on user-controlled image
  representations.
- Changing the CNN architecture in terms of strides and filter widths to
  support input images of size (64, 64, 3) instead of (28, 28).
- Having the posterior sigmas of the VAE be fixed variables instead of the
  output of the decoder, as this lets us use arbitrary image representations.
  rather than just per-pixel ones. This doesn't seem to hurt performance.
- Modernizing the tf.Estimator interface.
- Increasing training time, and delaying the annealing of the learning rate.
This code reproduces the VAE results from "A General and Adaptive Robust Loss
Function", Jonathan T. Barron, https://arxiv.org/abs/1701.03077. Specifically,
the shell script below reproduces the results from all models discussed in that
paper: Adaptive, Normal, and Cauchy distributions used to model RGB pixels,
YUV DCTs, and YUV Wavelets. Running it yourself will require changing the paths
for the output and input, unless you're sitting at barron@'s workstation.

RUN="ipython robust_loss/vae.py -- "
OUTPUT_DIR="--output_dir=/tmp/vae/"
OPTIONS="--celeba_dir=/usr/local/google/home/barron/data/CelebA/ --logtostderr"
WAVELET="--color_space=YUV --representation=CDF9/7"
DCT="--color_space=YUV --representation=DCT"
PIXEL="--color_space=RGB --representation=PIXEL"
ADAPTIVE="--alpha_lo=0 --alpha_hi=3 --alpha_init=1"
STUDENTS="--use_students_t=True"
NORMAL="--alpha_lo=2 --alpha_hi=2"
CAUCHY="--alpha_lo=0 --alpha_hi=0"

${RUN} ${OUTPUT_DIR}adaptive_wavelet/ ${OPTIONS} ${WAVELET} ${ADAPTIVE}
${RUN} ${OUTPUT_DIR}students_wavelet/ ${OPTIONS} ${WAVELET} ${STUDENTS}
${RUN} ${OUTPUT_DIR}normal_wavelet/ ${OPTIONS} ${WAVELET} ${NORMAL}
${RUN} ${OUTPUT_DIR}cauchy_wavelet/ ${OPTIONS} ${WAVELET} ${CAUCHY}

${RUN} ${OUTPUT_DIR}adaptive_dct/ ${OPTIONS} ${DCT} ${ADAPTIVE}
${RUN} ${OUTPUT_DIR}students_dct/ ${OPTIONS} ${DCT} ${STUDENTS}
${RUN} ${OUTPUT_DIR}normal_dct/ ${OPTIONS} ${DCT} ${NORMAL}
${RUN} ${OUTPUT_DIR}cauchy_dct/ ${OPTIONS} ${DCT} ${CAUCHY}

${RUN} ${OUTPUT_DIR}adaptive_pixel/ ${OPTIONS} ${PIXEL} ${ADAPTIVE}
${RUN} ${OUTPUT_DIR}students_pixel/ ${OPTIONS} ${PIXEL} ${STUDENTS}
${RUN} ${OUTPUT_DIR}normal_pixel/ ${OPTIONS} ${PIXEL} ${NORMAL}
${RUN} ${OUTPUT_DIR}cauchy_pixel/ ${OPTIONS} ${PIXEL} ${CAUCHY}
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from absl import flags
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow_probability import distributions as tfd
from robust_loss import adaptive

# Celeb-A images are cropped to the center-most (160, 160) pixel image, and
# then downsampled by 2.5x to be (64, 64)
IMAGE_PRECROP_SIZE = 160
IMAGE_SIZE = 64
VIZ_GRID_SIZE = 8
VIZ_MAX_N_SAMPLES = 3
VIZ_MAX_N_INPUTS = 16

flags.DEFINE_float(
    "learning_rate", default=0.001, help="Initial learning rate.")
flags.DEFINE_integer(
    "max_steps", default=50000, help="Number of training steps to run.")
flags.DEFINE_float(
    "decay_start",
    default=0.8,
    help="The fraction (in [0, 1]) of training at which point the learning "
    "rate should start getting annealed to 0")
flags.DEFINE_integer(
    "latent_size",
    default=16,
    help="Number of dimensions in the latent code (z).")
flags.DEFINE_integer("base_depth", default=32, help="Base depth for layers.")
flags.DEFINE_integer("batch_size", default=32, help="Batch size.")
flags.DEFINE_integer(
    "n_samples",
    default=1,
    help="Number of samples to use in encoding. For accurate importance "
    "weighted ELBOs, make this larger, but for speed make this smaller.")
flags.DEFINE_integer(
    "mixture_components",
    default=100,
    help="Number of mixture components to use in the prior. Each component is "
    "a diagonal normal distribution. The parameters of the components are "
    "intialized randomly, and then learned along with the rest of the "
    "parameters. If `analytic_kl` is True, `mixture_components` must be "
    "set to `1`.")
flags.DEFINE_bool(
    "analytic_kl",
    default=False,
    help="Whether or not to use the analytic version of the KL. When set to "
    "False the E_{Z~q(Z|X)}[log p(Z)p(X|Z) - log q(Z|X)] form of the ELBO "
    "will be used. Otherwise the -KL(q(Z|X) || p(Z)) + "
    "E_{Z~q(Z|X)}[log p(X|Z)] form will be used. If analytic_kl is True, "
    "then you must also specify `mixture_components=1`.")
flags.DEFINE_integer(
    "viz_steps",
    default=5000,
    help="Frequency at which to save visualizations.")
flags.DEFINE_string(
    "celeba_dir", default=None, help="The location of the Celeb-A dataset.")
flags.DEFINE_string(
    "output_dir",
    default=None,
    help="The directory where the output model and stats will be written.")
flags.DEFINE_bool(
    "delete_existing",
    default=False,
    help="If true, deletes existing `FLAGS.output_dir` directory. Use great "
    "caution when setting to True, as this could erase your disk.")
flags.DEFINE_string(
    "color_space",
    default="YUV",
    help="The loss's color_space, see adaptive.image_lossfun().")
flags.DEFINE_string(
    "representation",
    default="CDF9/7",
    help="The loss's image representation, see adaptive.image_lossfun().")
flags.DEFINE_integer(
    "wavelet_scale_base",
    default=1,
    help="The loss's wavelet scaling, see adaptive.image_lossfun().")
flags.DEFINE_integer(
    "wavelet_num_levels",
    default=5,
    help="The loss's wavelet depth, see adaptive.image_lossfun().")
flags.DEFINE_float(
    "alpha_lo",
    default=0,
    help="The lower bound on the loss's alpha, see adaptive.lossfun().")
flags.DEFINE_float(
    "alpha_hi",
    default=2,
    help="The upper bound on the loss's alpha, see adaptive.lossfun().")
flags.DEFINE_float(
    "alpha_init",
    default=None,
    help="The initial value of the loss's alpha, see adaptive.lossfun().")
flags.DEFINE_float(
    "scale_lo",
    default=1e-8,
    help="The lower bound on the loss's scale, see adaptive.lossfun().")
flags.DEFINE_float(
    "scale_init",
    default=1e-2,
    help="The initial value of the loss's scale, see adaptive.lossfun().")
flags.DEFINE_bool(
    "use_students_t",
    default=False,
    help="If true, use the negative log-likelihood of a Generalized Student's "
    "t-distribution as the loss")
# The following flags enable experiments with closed form prior and posterior
# variances and the BILBO objective from the paper:
# "Closed Form Variances for Variational Auto-Encoders"
# http://arxiv.org/abs/1912.10309
flags.DEFINE_bool(
    "unit_posterior",
    default=False,
    help="Whether or not to use constant (identity) posterior variances.")
flags.DEFINE_float(
    "posterior_variance",
    default=1.0,
    help="Posterior variance scaling factor. The posterior variance is the "
    "identity matrix times this value. Only used when `unit_posterior=True`.")
flags.DEFINE_bool(
    "floating_prior",
    default=False,
    help="Whether or not to use a fitted Gaussian for the prior. If "
    "floating_prior is True, then you must also specify `unit_posterior` and "
    "`mixture_components=1`.")
flags.DEFINE_bool(
    "fitted_samples",
    default=False,
    help="Whether or not to use a fitted Gaussian for sampling rather than the "
    "prior, regardless of `floating_prior`. This allows generative samples "
    "from fitted distributions even when lossing with traditional priors and "
    "posteriors. If `fitted_samples` is True, then you must also specify "
    "`mixture_components=1`.")
flags.DEFINE_bool(
    "bilbo",
    default=False,
    help="Whether or not to use the BILBO rather than ELBO for the loss, which "
    "is equivalent to using `unit_posterior` but has a simpler formulation. "
    "Metrics will still be reported as ELBO. If bilbo is True, then you must "
    "also specify `floating_prior` and its prerequisites.")

FLAGS = flags.FLAGS


def _softplus_inverse(x):
  """Helper which computes the function inverse of `tf.nn.softplus`."""
  return tf.math.log(tf.math.expm1(x))


def make_encoder(activation, latent_size, base_depth):
  """Creates the encoder function.

  Args:
    activation: Activation function in hidden layers.
    latent_size: The dimensionality of the encoding.
    base_depth: The lowest depth for a layer.

  Returns:
    encoder: A `callable` mapping a `Tensor` of images to a
      `tfd.Distribution` instance over encodings.
  """
  conv = functools.partial(
      tf.keras.layers.Conv2D, padding="SAME", activation=activation)

  encoder_net = tf.keras.Sequential([
      conv(base_depth, 5, 2),
      conv(base_depth, 5, 2),
      conv(2 * base_depth, 5, 2),
      conv(2 * base_depth, 5, 2),
      conv(4 * base_depth, 5, 2),
      conv(4 * latent_size, 5, 2),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(2 * latent_size, activation=None),
  ])

  def encoder(images):
    images = 2 * tf.cast(images, dtype=tf.float32) - 1
    net = encoder_net(images)
    if FLAGS.unit_posterior:
      scale_diag = tf.ones(latent_size) * FLAGS.posterior_variance
    else:
      scale_diag = tf.nn.softplus(net[Ellipsis, latent_size:] +
                                  _softplus_inverse(1.0))
    return tfd.MultivariateNormalDiag(
        loc=net[Ellipsis, :latent_size], scale_diag=scale_diag, name="code")

  return encoder


def make_decoder(activation, latent_size, output_shape, base_depth):
  """Creates the decoder function.

  Args:
    activation: Activation function in hidden layers.
    latent_size: Dimensionality of the encoding.
    output_shape: The output image shape.
    base_depth: Smallest depth for a layer.

  Returns:
    decoder: A `callable` mapping a `Tensor` of encodings to a
      `tfd.Distribution` instance over images.
  """
  deconv = functools.partial(
      tf.keras.layers.Conv2DTranspose, padding="SAME", activation=activation)
  conv = functools.partial(
      tf.keras.layers.Conv2D, padding="SAME", activation=activation)

  decoder_net = tf.keras.Sequential([
      deconv(4 * base_depth, 5, 2),
      deconv(4 * base_depth, 5, 2),
      deconv(2 * base_depth, 5, 2),
      deconv(2 * base_depth, 5, 2),
      deconv(base_depth, 5, 2),
      deconv(base_depth, 5, 2),
      conv(output_shape[-1], 5, activation=None),
  ])

  def decoder(codes):
    original_shape = tf.shape(codes)
    # Collapse the sample and batch dimension and convert to rank-4 tensor for
    # use with a convolutional decoder network.
    codes = tf.reshape(codes, (-1, 1, 1, latent_size))
    logits = decoder_net(codes)
    logits = tf.reshape(
        logits, shape=tf.concat([original_shape[:-1], output_shape], axis=0))
    mu = tf.nn.sigmoid(logits)
    return mu

  return decoder


def make_mixture_prior(latent_size, mixture_components):
  """Creates the mixture of Gaussians prior distribution.

  Args:
    latent_size: The dimensionality of the latent representation.
    mixture_components: Number of elements of the mixture.

  Returns:
    random_prior: A `tfd.Distribution` instance representing the distribution
      over encodings in the absence of any evidence.
  """
  if mixture_components == 1:
    return tfd.MultivariateNormalDiag(
        loc=tf.zeros([latent_size]), scale_identity_multiplier=1.0)

  loc = tf.get_variable(name="loc", shape=[mixture_components, latent_size])
  raw_scale_diag = tf.get_variable(
      name="raw_scale_diag", shape=[mixture_components, latent_size])
  mixture_logits = tf.get_variable(
      name="mixture_logits", shape=[mixture_components])

  return tfd.MixtureSameFamily(
      components_distribution=tfd.MultivariateNormalDiag(
          loc=loc, scale_diag=tf.nn.softplus(raw_scale_diag)),
      mixture_distribution=tfd.Categorical(logits=mixture_logits),
      name="prior")


def pack_images(images, rows, cols):
  """Helper utility to make a field of images."""
  shape = tf.shape(images)
  width = shape[-3]
  height = shape[-2]
  depth = shape[-1]
  images = tf.reshape(images, (-1, width, height, depth))
  batch = tf.shape(images)[0]
  rows = tf.minimum(rows, batch)
  cols = tf.minimum(batch // rows, cols)
  images = images[:rows * cols]
  images = tf.reshape(images, (rows, cols, width, height, depth))
  images = tf.transpose(images, [0, 2, 1, 3, 4])
  images = tf.reshape(images, [1, rows * width, cols * height, depth])
  images = tf.clip_by_value(images, 0., 1.)
  return images


def image_tile_summary(name, tensor, rows, cols):
  return tf.summary.image(name, pack_images(tensor, rows, cols), max_outputs=1)


def model_fn(features, labels, mode, params, config):
  """Builds the model function for use in an estimator.

  Arguments:
    features: The input features for the estimator.
    labels: The labels, unused here.
    mode: Signifies whether it is train or test or predict.
    params: Some parameters, unused here.
    config: The RunConfig, unused here.

  Returns:
    EstimatorSpec: A tf.estimator.EstimatorSpec instance.
  """
  del labels, params, config

  if FLAGS.analytic_kl and FLAGS.mixture_components != 1:
    raise NotImplementedError(
        "Using `analytic_kl` is only supported when `mixture_components = 1` "
        "since there's no closed form otherwise.")
  if FLAGS.floating_prior and not (FLAGS.unit_posterior and
                                   FLAGS.mixture_components == 1):
    raise NotImplementedError(
        "Using `floating_prior` is only supported when `unit_posterior` = True "
        "since there's a scale ambiguity otherwise, and when "
        "`mixture_components = 1` since there's no closed form otherwise.")
  if FLAGS.fitted_samples and FLAGS.mixture_components != 1:
    raise NotImplementedError(
        "Using `fitted_samples` is only supported when "
        "`mixture_components = 1` since there's no closed form otherwise.")
  if FLAGS.bilbo and not FLAGS.floating_prior:
    raise NotImplementedError(
        "Using `bilbo` is only supported when `floating_prior = True`.")

  activation = tf.nn.leaky_relu
  encoder = make_encoder(activation, FLAGS.latent_size, FLAGS.base_depth)
  decoder = make_decoder(activation, FLAGS.latent_size, [IMAGE_SIZE] * 2 + [3],
                         FLAGS.base_depth)

  approx_posterior = encoder(features)
  approx_posterior_sample = approx_posterior.sample(FLAGS.n_samples)
  decoder_mu = decoder(approx_posterior_sample)

  if FLAGS.floating_prior or FLAGS.fitted_samples:
    posterior_batch_mean = tf.reduce_mean(approx_posterior.mean()**2, [0])
    posterior_batch_variance = tf.reduce_mean(approx_posterior.stddev()**2, [0])
    posterior_scale = posterior_batch_mean + posterior_batch_variance
    floating_prior = tfd.MultivariateNormalDiag(
        tf.zeros(FLAGS.latent_size), tf.sqrt(posterior_scale))
    tf.summary.scalar("posterior_scale", tf.reduce_sum(posterior_scale))

  if FLAGS.floating_prior:
    latent_prior = floating_prior
  else:
    latent_prior = make_mixture_prior(FLAGS.latent_size,
                                      FLAGS.mixture_components)

  # Decode samples from the prior for visualization.
  if FLAGS.fitted_samples:
    sample_distribution = floating_prior
  else:
    sample_distribution = latent_prior

  n_samples = VIZ_GRID_SIZE**2
  random_mu = decoder(sample_distribution.sample(n_samples))

  residual = tf.reshape(features - decoder_mu, [-1] + [IMAGE_SIZE] * 2 + [3])

  if FLAGS.use_students_t:
    nll = adaptive.image_lossfun(
        residual,
        color_space=FLAGS.color_space,
        representation=FLAGS.representation,
        wavelet_num_levels=FLAGS.wavelet_num_levels,
        wavelet_scale_base=FLAGS.wavelet_scale_base,
        use_students_t=FLAGS.use_students_t,
        scale_lo=FLAGS.scale_lo,
        scale_init=FLAGS.scale_init)[0]
  else:
    nll = adaptive.image_lossfun(
        residual,
        color_space=FLAGS.color_space,
        representation=FLAGS.representation,
        wavelet_num_levels=FLAGS.wavelet_num_levels,
        wavelet_scale_base=FLAGS.wavelet_scale_base,
        use_students_t=FLAGS.use_students_t,
        alpha_lo=FLAGS.alpha_lo,
        alpha_hi=FLAGS.alpha_hi,
        alpha_init=FLAGS.alpha_init,
        scale_lo=FLAGS.scale_lo,
        scale_init=FLAGS.scale_init)[0]

  nll = tf.reshape(nll, [tf.shape(decoder_mu)[0],
                         tf.shape(decoder_mu)[1]] + [IMAGE_SIZE] * 2 + [3])

  # Clipping to prevent the loss from nanning out.
  max_val = np.finfo(np.float32).max
  nll = tf.clip_by_value(nll, -max_val, max_val)

  viz_n_inputs = np.int32(np.minimum(VIZ_MAX_N_INPUTS, FLAGS.batch_size))
  viz_n_samples = np.int32(np.minimum(VIZ_MAX_N_SAMPLES, FLAGS.n_samples))

  image_tile_summary("input", tf.to_float(features), rows=1, cols=viz_n_inputs)

  image_tile_summary(
      "recon/mean",
      decoder_mu[:viz_n_samples, :viz_n_inputs],
      rows=viz_n_samples,
      cols=viz_n_inputs)

  img_summary_input = image_tile_summary(
      "input1", tf.to_float(features), rows=viz_n_inputs, cols=1)
  img_summary_recon = image_tile_summary(
      "recon1", decoder_mu[:1, :viz_n_inputs], rows=viz_n_inputs, cols=1)

  image_tile_summary(
      "random/mean", random_mu, rows=VIZ_GRID_SIZE, cols=VIZ_GRID_SIZE)

  distortion = tf.reduce_sum(nll, axis=[2, 3, 4])

  avg_distortion = tf.reduce_mean(distortion)
  tf.summary.scalar("distortion", avg_distortion)

  if FLAGS.analytic_kl:
    rate = tfd.kl_divergence(approx_posterior, latent_prior)
  else:
    rate = (
        approx_posterior.log_prob(approx_posterior_sample) -
        latent_prior.log_prob(approx_posterior_sample))
  avg_rate = tf.reduce_mean(rate)
  tf.summary.scalar("rate", avg_rate)

  elbo_local = -(rate + distortion)

  elbo = tf.reduce_mean(elbo_local)
  tf.summary.scalar("elbo", elbo)

  if FLAGS.bilbo:
    bilbo = -0.5 * tf.reduce_sum(
        tf.log1p(
            posterior_batch_mean / posterior_batch_variance)) - avg_distortion
    tf.summary.scalar("bilbo", bilbo)
    loss = -bilbo
  else:
    loss = -elbo

  importance_weighted_elbo = tf.reduce_mean(
      tf.reduce_logsumexp(elbo_local, axis=0) -
      tf.math.log(tf.to_float(FLAGS.n_samples)))
  tf.summary.scalar("elbo/importance_weighted", importance_weighted_elbo)

  # Perform variational inference by minimizing the -ELBO.
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.cosine_decay(
      FLAGS.learning_rate,
      tf.maximum(
          tf.cast(0, tf.int64),
          global_step - int(FLAGS.decay_start * FLAGS.max_steps)),
      int((1. - FLAGS.decay_start) * FLAGS.max_steps))
  tf.summary.scalar("learning_rate", learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate)

  if mode == tf.estimator.ModeKeys.TRAIN:
    train_op = optimizer.minimize(loss, global_step=global_step)
  else:
    train_op = None

  eval_metric_ops = {}
  eval_metric_ops["elbo"] = tf.metrics.mean(elbo)
  eval_metric_ops["elbo/importance_weighted"] = tf.metrics.mean(
      importance_weighted_elbo)
  eval_metric_ops["rate"] = tf.metrics.mean(avg_rate)
  eval_metric_ops["distortion"] = tf.metrics.mean(avg_distortion)
  # This ugly hackery is necessary to get TF to visualize when running the
  # eval set, apparently.
  eval_metric_ops["img_summary_input"] = (img_summary_input, tf.no_op())
  eval_metric_ops["img_summary_recon"] = (img_summary_recon, tf.no_op())
  eval_metric_ops = {str(k): v for k, v in eval_metric_ops.items()}

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=train_op,
      eval_metric_ops=eval_metric_ops,
  )


def prep_dataset_fn(filenames, splits, is_training):
  """Prepares training/test datasets in the format required by tf.estimator."""
  if is_training:
    filenames = filenames[splits == 0]
  else:
    filenames = filenames[splits == 1]

  def _load_image(filename):
    """Load an image, center-crop it to a square, and then resize it."""
    image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
    # Crop to the centermost (IMAGE_PRECROP_SIZE, IMAGE_PRECROP_SIZE) image.
    idx0 = (tf.shape(image)[0] - IMAGE_PRECROP_SIZE) // 2
    idx1 = (tf.shape(image)[1] - IMAGE_PRECROP_SIZE) // 2
    image_cropped = tf.slice(image, [idx0, idx1, 0],
                             [IMAGE_PRECROP_SIZE] * 2 + [-1])
    # Downsample the image to be (IMAGE_SIZE, IMAGE_SIZE).
    image_resized = tf.image.resize_images(image_cropped, [IMAGE_SIZE] * 2)
    image_resized = tf.to_float(image_resized) / 255.
    return image_resized, 0

  def prep_dataset():
    """Return a TF dataset of images expected by tf.estimator."""
    filenames_tf = tf.constant(filenames)
    dataset = tf.data.Dataset.from_tensor_slices(filenames_tf)
    dataset = dataset.map(_load_image)
    if is_training:
      dataset = dataset.shuffle(50000).repeat()
    dataset = dataset.batch(FLAGS.batch_size)
    return dataset

  return prep_dataset


def main(argv):
  del argv  # Unused.

  if FLAGS.output_dir is None:
    raise ValueError("`output_dir` must be defined")

  if FLAGS.delete_existing and tf.gfile.Exists(FLAGS.output_dir):
    tf.logging.warn("Deleting old log directory at {}".format(
        FLAGS.output_dir))
    tf.gfile.DeleteRecursively(FLAGS.output_dir)
  tf.gfile.MakeDirs(FLAGS.output_dir)

  print("Logging to {}".format(FLAGS.output_dir))

  # Load the training or test split of the Celeb-A filenames.
  if FLAGS.celeba_dir is None:
    raise ValueError("`celeba_dir` must be defined")
  celeba_dataset_path = \
      os.path.join(FLAGS.celeba_dir, "Img/img_align_celeba/")
  celeba_partition_path = \
      os.path.join(FLAGS.celeba_dir, "Eval/list_eval_partition.txt")
  with open(celeba_partition_path, "r") as fid:
    partition = fid.readlines()
  filenames, splits = zip(*[x.split() for x in partition])
  filenames = np.array(
      [os.path.join(celeba_dataset_path, f) for f in filenames])
  splits = np.array([int(x) for x in splits])

  with tf.Graph().as_default():
    train_input_fn = prep_dataset_fn(filenames, splits, is_training=True)
    eval_input_fn = prep_dataset_fn(filenames, splits, is_training=False)
    estimator = tf.estimator.Estimator(
        model_fn,
        config=tf.estimator.RunConfig(
            model_dir=FLAGS.output_dir,
            save_checkpoints_steps=FLAGS.viz_steps,
        ),
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=FLAGS.max_steps)
    # Sad ugly hack here. Setting steps=None should go through all of the
    # validation set, but doesn't seem to, so I'm doing it manually.
    eval_spec = tf.estimator.EvalSpec(
        input_fn=eval_input_fn,
        steps=len(filenames[splits == 1]) // FLAGS.batch_size,
        start_delay_secs=0,
        throttle_secs=0)
    for _ in range(FLAGS.max_steps // FLAGS.viz_steps):
      tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == "__main__":
  tf.app.run()
