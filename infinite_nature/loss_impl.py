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

"""Reference implementation for Infinite Nature's training loss.

These functions are NOT referenced by the demo code and are included
to supplement the training details provided in the paper.
"""
import ops
import tensorflow as tf


def compute_infinite_nature_loss(
    generated_rgbd, gt_rgbd, discriminator, mu_logvar):
  """Computes loss between a generated RGBD sequence and the ground truth.

  Lambda terms are the default values used during the original submission.

  Args:
    generated_rgbd: [B, T, H, W, 4] A batch of T-length RGBD sequences
      produced by the generator. Ranges from (0, 1)
    gt_rgbd: [B, T, H, W, 4] The ground truth sequence from a video.
      Ranges from (0, 1)
    discriminator: a discriminator which accepts an [B, H, W, D] tensor
      and runs a discriminator on multiple scales and returns
      a list of (features, logit) for each scale.
    mu_logvar: ([B, 128], [B, 128]) A tuple of mu, log-variance features
      parameterizing the Gaussian used to sample the variational noise.

  Returns:
    A dictionary of losses. total_loss refers to the final
      loss used to optimize the generator and total_disc_loss refers to the
      loss used by the discriminator.
  """
  _, _, height, width, _ = tf.shape(generated_rgbd).as_list()
  gen_flatten = tf.reshape(generated_rgbd, [-1, height, width, 4])
  gt_flatten = tf.reshape(gt_rgbd, [-1, height, width, 4])

  # discriminator returns:
  # [(scale_1_feats, scale_1_logits), (scale_2_feats, scale_2_logits), ...]
  disc_on_generated = discriminator(gen_flatten)
  generated_features = [f[0] for f in disc_on_generated]
  generated_logits = [f[1] for f in disc_on_generated]
  disc_on_real = discriminator(gt_flatten)
  real_features = [f[0] for f in disc_on_real]
  real_logits = [f[1] for f in disc_on_real]

  disc_loss, _, _ = compute_discriminator_loss(
      real_logits, tf.stop_gradients(generated_logits))
  fool_d_loss = compute_adversarial_loss(generated_logits)

  feature_matching_loss = compute_feature_matching(real_features,
                                                   generated_features)
  kld_loss = compute_kld_loss(mu_logvar[0], mu_logvar[1])

  rgbd_loss = tf.reduce_mean(tf.abs(generated_rgbd - gt_rgbd))
  perceptual_loss = compute_perceptual_loss(generated_rgbd * 255.,
                                            gt_rgbd * 255.)

  loss_dict = {}
  loss_dict["disc_loss"] = disc_loss
  loss_dict["adversarial_loss"] = fool_d_loss
  loss_dict["feature_matching_loss"] = feature_matching_loss
  loss_dict["kld_loss"] = kld_loss
  loss_dict["perceptual_loss"] = perceptual_loss
  loss_dict["reconstruction_loss"] = rgbd_loss

  total_loss = (1e-2 * perceptual_loss +
                10.0 * feature_matching_loss + 0.05 * kld_loss +
                1.5 * fool_d_loss + 0.5 * rgbd_loss)
  total_disc_loss = 1.5 * disc_loss
  loss_dict["total_loss"] = total_loss
  loss_dict["total_disc_loss"] = total_disc_loss
  return loss_dict


def compute_kld_loss(mu, logvar):
  loss = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))
  return loss


def compute_discriminator_loss(real_logit, fake_logit):
  """Computes the discriminator hinge loss given logits.

  Args:
    real_logit: A list of logits produced from the real image
    fake_logit: A list of logits produced from the fake image

  Returns:
    Scalars discriminator loss, adv_loss, patchwise accuracy of discriminator
    at detecting real and fake patches respectively.
  """
  # Multi-scale disc returns a list.
  real_loss, fake_loss = 0, 0
  real_total, fake_total = 0, 0
  real_correct, fake_correct = 0, 0
  for real_l, fake_l in zip(real_logit, fake_logit):
    real_loss += tf.reduce_mean(tf.nn.relu(1 - real_l))
    fake_loss += tf.reduce_mean(tf.nn.relu(1 + fake_l))
    real_total += tf.cast(tf.reduce_prod(tf.shape(real_l)), tf.float32)
    fake_total += tf.cast(tf.reduce_prod(tf.shape(fake_l)), tf.float32)
    real_correct += tf.reduce_sum(tf.cast(real_l >= 0, tf.float32))
    fake_correct += tf.reduce_sum(tf.cast(fake_l < 0, tf.float32))

  # Avg of all outputs.
  real_loss = real_loss / float(len(real_logit))
  fake_loss = fake_loss / float(len(fake_logit))
  real_accuracy = real_correct / real_total
  fake_accuracy = fake_correct / fake_total

  disc_loss = real_loss + fake_loss

  return disc_loss, real_accuracy, fake_accuracy


def compute_adversarial_loss(fake_logit):
  """Computes the adversarial hinge loss to apply to the generator.

  Args:
    fake_logit: list of tensors which correspond to discriminator logits
      on generated images

  Returns:
    A scalar of the loss.
  """
  fake_loss = 0
  for fake_l in fake_logit:
    fake_loss += -tf.reduce_mean(fake_l)

  # average of all.
  fake_loss = fake_loss / float(len(fake_logit))

  return fake_loss


def compute_feature_matching(real_feats, fake_feats):
  """Computes a feature matching loss between real and fake feature pyramids.

  Args:
    real_feats: A list of feature activations of a discriminator on real images
    fake_feats: A list of feature activations on fake images

  Returns:
    A scalar of the loss.
  """
  losses = []
  # Loop for scale
  for real_feat, fake_feat in zip(real_feats, fake_feats):
    losses.append(tf.reduce_mean(tf.abs(real_feat - fake_feat)))

  return tf.reduce_mean(losses)


def compute_perceptual_loss(generated, real):
  """Compute the perceptual loss between a generated and real sample.

  The input to this are RGB images ranging from [0, 255].

  build_vgg19's reference library can be found here:
  https://github.com/CQFIO/PhotographicImageSynthesis/blob/master/demo_1024p.py

  Args:
    generated: [B, H, W, 3] Generated image that we want to be perceptually
      close to real.
    real: [B, H, W, 3] Ground truth image that we want to target.

  Returns:
    A tf scalar corresponding to the perceptual loss.
  """
  # Input is [0, 255.], not necessarily clipped though
  def build_vgg19(*args):
    raise NotImplementedError
  vgg_real_fake = build_vgg19(
      tf.concat([real, generated], axis=0),
      "imagenet-vgg-verydeep-19.mat")

  def compute_l1_loss(key):
    real, fake = tf.split(vgg_real_fake[key], 2, axis=0)
    return tf.reduce_mean(tf.abs(real - fake))

  p0 = tf.zeros([])
  p1 = compute_l1_loss("conv1_2") / 2.6
  p2 = compute_l1_loss("conv2_2") / 4.8
  p3 = compute_l1_loss("conv3_2") / 3.7
  p4 = compute_l1_loss("conv4_2") / 5.6
  p5 = compute_l1_loss("conv5_2") * 10 / 1.5
  return p0 + p1 + p2 + p3 + p4 + p5


def multiscale_discriminator(rgbd_sequence):
  """A reference implementation for the discriminator.

  This is not used by the demo during inference.

  Args:
    rgbd_sequence: [B, H, W, 4] A batch of RGBD images.

  Returns:
    A list of (features, logits) tuples corresponding to two scales.
  """
  features, logit = patch_discriminator(
      rgbd_sequence, scope="spade_discriminator_0")

  x_small = ops.half_size(rgbd_sequence)
  features_small, logit_small = patch_discriminator(
      x_small, scope="spade_discriminator_1")

  # These are lists
  return [features, features_small], [logit, logit_small]


def patch_discriminator(rgbd_sequence, scope="spade_discriminator"):
  """Creates a patch discriminator to process RGBD values.

  Args:
    rgbd_sequence: [B, H, W, 4] A batch of RGBD images.
    scope: (str) variable scope

  Returns:
    (list of features, logits)
  """
  num_channel = 64
  num_layers = 4
  features = []
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    x = ops.sn_conv(
        rgbd_sequence, num_channel, kernel_size=4, stride=2, sn=False)
    channel = num_channel
    for i in range(1, num_layers):
      stride = 1 if i == num_layers - 1 else 2
      channel = min(channel * 2, 512)
      x = ops.sn_conv(
          x, channel, kernel_size=4, stride=stride, sn=True,
          scope="conv_{}".format(i))
      x = ops.instance_norm(x, scope="inst_norm_{}".format(i))
      x = tf.nn.lrelu(x, 0.2)
      features.append(x)

    logit = ops.sn_conv(
        x, 1, kernel_size=4, stride=1,
        sn=False, scope="D_logit")

  return features, logit
