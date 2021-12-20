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

# Lint as: python3
"""Main factory of losses.

   Each class accepts model_configs as its params and has a call function:
     Args:
       labels: a dictionary containing necessary labels
       predictions: a dictionary containing all model outputs
       replicator: a class containing TPU/GPU replicate configs

     Returns:
       loss: a tf.Tensor containing the loss value
       metrics_to_log: a dictionary containing intermediate metrics to be logged
"""

from typing import Optional
from absl import logging
import tensorflow as tf

from vatt.experiments import base


LARGE_NUM = 1e9


def get_shape(x):
  """Deal with dynamic shape in tensorflow cleanly."""
  static = x.shape.as_list()
  dynamic = tf.shape(x)
  return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def mse(x_1, x_2, axis=-1):
  return tf.reduce_mean(tf.square(x_1-x_2), axis=axis)


def cross_replica_gather(tensor, num_replica, batch_dim=0):
  """Cross replica gather of tensors.

  Args:
    tensor: The input tensor to gather from other replica
    num_replica: The total number of replica.
    batch_dim: The batch index of the input tensor.

  Returns:
    The gathered tensor from all replica, where other tensors from other
    replica are concatenated in the batch dimension batch_dim.
  """
  ts_shape = [num_replica] + tensor.shape.as_list()
  group_assignment = [list(range(num_replica))]
  tensor = tf.raw_ops.AllToAll(
      input=tf.broadcast_to(tf.expand_dims(tensor, 0), shape=ts_shape),
      group_assignment=group_assignment,
      concat_dimension=batch_dim + 1,
      split_dimension=0,
      split_count=num_replica,
      name="AllToAllGather",
  )
  return tf.squeeze(tensor, axis=0)


class SymmetricNCE(object):
  """Constructs symmetric NCE / MIL-NCE objective."""

  def __init__(self, params):
    self._vid_txt_weight = params.vid_txt_weight
    self._vid_aud_weight = params.vid_aud_weight
    self._aud_txt_weight = params.aud_txt_weight
    self._temperature = params.temperature
    self.loss_weight = params.loss_weight
    self.name = params.name

    assert (
        self._vid_txt_weight + self._vid_aud_weight + self._aud_txt_weight > 0.
        ), "At least one weight should be non-zero"

  def _calculate_nce(self,
                     modality_1,
                     modality_2,
                     mask_1,
                     mask_2,
                     replicator):
    """Calculate Modality_1 vs. Modality_2 pair-wise similarities."""

    # normalize embeddings
    modality_1 = tf.math.l2_normalize(modality_1, axis=-1)  # (B, N1, D)
    modality_2 = tf.math.l2_normalize(modality_2, axis=-1)  # (B, N2, D)

    logging.info("Using cross replica negative with %d replicas.",
                 replicator.num_replicas)
    modality_1_all = cross_replica_gather(modality_1, replicator.num_replicas)
    modality_2_all = cross_replica_gather(modality_2, replicator.num_replicas)

    # calculate cross-modal similarities for local device -> (B, B, N1, N2)
    m1_vs_m2 = tf.einsum("bmd,cnd->bcmn", modality_1, modality_2)

    # calucalate M1 vs. M2_all similarities -> (B, B_all, N1, N2)
    m1_vs_m2all = tf.einsum("bmd,cnd->bcmn", modality_1, modality_2_all)

    # calucalate M2 vs. M1_all similarities -> (B, B_all, N2, N1)
    m2_vs_m1all = tf.einsum("bmd,cnd->bcmn", modality_2, modality_1_all)

    # apply masks (if any)
    if mask_1 is not None:
      # any similarity based on an invalid modality_1 sample is an outlier
      mask_1_all = cross_replica_gather(mask_1, replicator.num_replicas)
      m1_vs_m2all -= tf.cast(1 - mask_1,
                             dtype=tf.float32)[:, None, None, None] * LARGE_NUM
      m2_vs_m1all -= tf.cast(1 - mask_1_all,
                             dtype=tf.float32)[None, :, None, None] * LARGE_NUM

    if mask_2 is not None:
      # any similarity based on an invalid modality_2 sample is an outlier
      mask_2_all = cross_replica_gather(mask_2, replicator.num_replicas)
      m1_vs_m2all -= tf.cast(1 - mask_2_all,
                             dtype=tf.float32)[None, :, None, None] * LARGE_NUM
      m2_vs_m1all -= tf.cast(1 - mask_2,
                             dtype=tf.float32)[:, None, None, None] * LARGE_NUM

    # reshape similarities to (B, B_all*N1*N2) and (B, B_all*N2*N1)
    m1_vs_m2all = tf.reshape(m1_vs_m2all,
                             tuple(get_shape(m1_vs_m2all)[:1]) + (-1,))
    m2_vs_m1all = tf.reshape(m2_vs_m1all,
                             tuple(get_shape(m2_vs_m1all)[:1]) + (-1,))

    # only keep positive pairs -> (B, B, N1*N2) -> (B, N1*N2)
    m1_vs_m2 = tf.reshape(m1_vs_m2, tuple(get_shape(m1_vs_m2)[:2]) + (-1,))
    sim_pos = tf.einsum("bbn->bn", m1_vs_m2)

    # calculate the log sum exp (numerator) of the NCE loss.
    logsumexp_pos = tf.math.reduce_logsumexp(
        (sim_pos / self._temperature),
        axis=1,
        )  # (B, )

    # calculate the log sum exp (denominator) of the NCE loss.
    logsumexp_all_m1_vs_m2all = tf.math.reduce_logsumexp(
        m1_vs_m2all / self._temperature,
        axis=1,
        )  # (B, )
    logsumexp_all_m2_vs_m1all = tf.math.reduce_logsumexp(
        m2_vs_m1all / self._temperature,
        axis=1,
        )  # (B, )

    # calculate the loss.
    loss_m1_vs_m2 = logsumexp_all_m1_vs_m2all - logsumexp_pos
    loss_m2_vs_m1 = logsumexp_all_m2_vs_m1all - logsumexp_pos

    # scaling loss values if mask_1 or mask_2 are valid
    if mask_1 is not None:
      num_valids = tf.reduce_sum(mask_1)
      scale = tf.math.divide_no_nan(
          tf.cast(tf.size(mask_1), dtype=tf.float32),
          num_valids,
          )
      # note that since the positive pairs are one-by-one, we need to apply the
      # mask and the scale on both m1_vs_m2 and m2_vs_m1
      loss_m1_vs_m2 *= mask_1 * scale
      loss_m2_vs_m1 *= mask_1 * scale

    if mask_2 is not None:
      num_valids = tf.reduce_sum(mask_2)
      scale = tf.math.divide_no_nan(
          tf.cast(tf.size(mask_2), dtype=tf.float32),
          num_valids,
          )
      # note that since the positive pairs are one-by-one, we need to apply the
      # mask and the scale on both m1_vs_m2 and m2_vs_m1
      loss_m1_vs_m2 *= mask_2 * scale
      loss_m2_vs_m1 *= mask_2 * scale

    # if none of the samples are valid, the logsumexp could be NaN, hence
    # we manually set them to zero
    loss_m1_vs_m2 = tf.where(tf.math.is_nan(loss_m1_vs_m2),
                             tf.zeros(loss_m1_vs_m2.shape),
                             tf.identity(loss_m1_vs_m2))
    loss_m2_vs_m1 = tf.where(tf.math.is_nan(loss_m2_vs_m1),
                             tf.zeros(loss_m2_vs_m1.shape),
                             tf.identity(loss_m2_vs_m1))

    # average across batch samples
    loss_m1_vs_m2 = tf.reduce_mean(loss_m1_vs_m2)
    loss_m2_vs_m1 = tf.reduce_mean(loss_m2_vs_m1)

    return loss_m1_vs_m2 + loss_m2_vs_m1

  def _get_empty_logs(self):
    """Dummy loss and logs with same structure as original call."""

    id_losses = []
    metrics_to_log = {}
    if self._vid_txt_weight > 0.:
      id_losses.append("vidtxt_loss")
    if self._vid_aud_weight > 0.:
      id_losses.append("vidaud_loss")
    if self._aud_txt_weight > 0.:
      id_losses.append("audtxt_loss")

    for id_loss in id_losses:
      metrics_to_log[id_loss] = 0.

    return 0., metrics_to_log

  def _reshape_embds(self, embeddings):
    """Reshape all embeddings to (B, N, D)."""

    vid_to_aud = embeddings["video"].get("toaud", None)
    vid_to_txt = embeddings["video"].get("totxt", None)
    if vid_to_aud is not None:
      batch_size = get_shape(vid_to_aud)[0]
    elif vid_to_txt is not None:
      batch_size = get_shape(vid_to_txt)[0]
    else:
      raise ValueError("Could not find batch_size")

    def _reshape_embds(inputs):
      d = get_shape(inputs)[-1]
      return tf.reshape(inputs, [batch_size, -1, d])

    return tf.nest.map_structure(_reshape_embds, embeddings)

  def __call__(self,
               labels,
               predictions,
               training=True,
               replicator = None):
    """Calculates NCE + MIL-NCE loss.

    Args:
      labels: Dictionary containing text masks:
        - audio_mask: Tensor of shape [B, 1] containing indicators of whether
          the audio should be used.
        - text_mask: Tensor of shape [B, 1] containing indicators of whether
          the text should be used.
      predictions: Dictionaries containing all modalidity-specific outputs:
        - video: Dictionaries containing tensor of video embeddings to
          compare against audio and text of shape [B, D] where B is the
          number of video embeddings and D the embedding dimension.
        - audio: Dictionaries containing tensor of audio embeddings to
          compare against video and text of shape [B, D] where B is the
          number of video embeddings and D the embedding dimension.
        - text: Dictionaries containing tensor of text embeddings to
          compare against video and audio of shape [B * L, D] where B is the
          number of text embeddings, D the embedding dimension and L is the
          number of positive candidate narrations for each video clip.
      training: a bool label indicating whether the loss should be considered
        or not.
      replicator: tensorflow replicator used for cross replica negative.

    Returns:
      The computed MIL-NCE loss.
    """

    if not training:
      return self._get_empty_logs()

    predictions = self._reshape_embds(predictions)

    video_embd = predictions["video"]
    audio_embd = predictions["audio"]
    text_embd = predictions["text"]
    video_mask = labels.get("video_mask", None)
    audio_mask = labels.get("audio_mask", None)
    text_mask = labels.get("text_mask", None)

    metrics_to_log = {}
    all_losses = []
    all_weights = []
    if self._vid_txt_weight > 0.:
      vid_txt_loss = self._calculate_nce(
          modality_1=text_embd["tovid"],
          modality_2=video_embd["totxt"],
          mask_1=text_mask,
          mask_2=video_mask,
          replicator=replicator,
          ) * self._vid_txt_weight
      all_losses.append(vid_txt_loss)
      all_weights.append(self._vid_txt_weight)
      metrics_to_log["vidtxt_loss"] = vid_txt_loss

    if self._vid_aud_weight > 0.:
      vid_aud_loss = self._calculate_nce(
          modality_1=audio_embd["tovid"],
          modality_2=video_embd["toaud"],
          mask_1=audio_mask,
          mask_2=video_mask,
          replicator=replicator,
          ) * self._vid_aud_weight
      all_losses.append(vid_aud_loss)
      all_weights.append(self._vid_aud_weight)
      metrics_to_log["vidaud_loss"] = vid_aud_loss

    if self._aud_txt_weight > 0.:
      aud_txt_loss = self._calculate_nce(
          modality_1=text_embd["toaud"],
          modality_2=audio_embd["totxt"],
          mask_1=text_mask,
          mask_2=audio_mask,
          replicator=replicator,
          ) * self._aud_txt_weight
      all_losses.append(aud_txt_loss)
      all_weights.append(self._aud_txt_weight)
      metrics_to_log["audtxt_loss"] = aud_txt_loss

    loss = tf.add_n(all_losses) / tf.add_n(all_weights)

    return loss, metrics_to_log


class AsymmetricNCE(object):
  """Constructs assymetric NCE / MIL-NCE objective."""

  def __init__(self, params):
    self._vid_txt_weight = params.vid_txt_weight
    self._vid_aud_weight = params.vid_aud_weight
    self._aud_txt_weight = params.aud_txt_weight
    self._temperature = params.temperature
    self.loss_weight = params.loss_weight
    self.name = params.name

  def _calculate_similarity(self,
                            embed_1,
                            embed_2,
                            batch_size_1,
                            batch_size_2,
                            temperature=1.0):
    """Calculate cosine similarity between embed_1 and embed_2."""

    embed_1 = tf.math.l2_normalize(embed_1, axis=-1)
    embed_2 = tf.math.l2_normalize(embed_2, axis=-1)

    # Similarities [B_1, B_2*L].
    similarity = tf.matmul(embed_1, embed_2, transpose_b=True)

    # [B_1, B_2, L]
    similarity = tf.reshape(similarity, [batch_size_1, batch_size_2, -1])
    similarity /= temperature

    return similarity

  def _calculate_nce(self,
                     embed_1,
                     embed_2,
                     replicator,
                     embed_2_mask=None):
    """Calculate NCE / MIL-NCE loss between embed_1 and embed_2."""

    n_reps = replicator.num_replicas
    logging.info("Using cross replica negative with %d replicas.", n_reps)
    embed_1_all = cross_replica_gather(embed_1, n_reps)
    embed_2_all = cross_replica_gather(embed_2, n_reps)
    if embed_2_mask is not None:
      embed_2_mask_all = cross_replica_gather(embed_2_mask, n_reps)
    else:
      embed_2_mask_all = None

    # Get number of local videos and number of total videos (across replicas).
    batch_size = embed_1.shape.as_list()[0]  # B
    batch_size_all = embed_1_all.shape.as_list()[0]  # B_all

    # Compute positive similarities by using local2local.
    sim_local2local = self._calculate_similarity(
        embed_1=embed_1,
        embed_2=embed_2,
        batch_size_1=batch_size,
        batch_size_2=batch_size,
        temperature=self._temperature,
        )

    # Set to 0 all non diag entries (to only keep positive candidate scores).
    id_m = tf.eye(batch_size)  # [B, B]
    sim_pos = tf.reduce_sum(sim_local2local * id_m[:, :, None], axis=1)

    # Compute negative similarities by using local2all in a symmetric fashion.
    sim_sp_local2all = self._calculate_similarity(
        embed_1=embed_1,
        embed_2=embed_2_all,
        batch_size_1=batch_size,
        batch_size_2=batch_size_all,
        temperature=self._temperature,
        )

    if embed_2_mask is not None:
      mask_f_all = tf.cast(1 - embed_2_mask_all, dtype=tf.float32)
      sim_sp_local2all = sim_sp_local2all - mask_f_all[None, :, None]*LARGE_NUM

    sim_sp_all2local = self._calculate_similarity(
        embed_1=embed_1_all,
        embed_2=embed_2,
        batch_size_1=batch_size_all,
        batch_size_2=batch_size,
        temperature=self._temperature,
        )

    # Transpose into [B, B_all, L']
    sim_sp_all2local = tf.transpose(sim_sp_all2local, perm=[1, 0, 2])

    if embed_2_mask is not None:
      mask_f = tf.cast(1 - embed_2_mask, dtype=tf.float32)
      sim_sp_all2local = sim_sp_all2local - mask_f[:, None, None] * LARGE_NUM

    # We obtain the symmetric scores (x, y) and (y, x) by concatenation.
    sim_all = tf.concat([sim_sp_local2all, sim_sp_all2local], axis=1)
    sim_all = tf.reshape(sim_all, [batch_size, -1])

    # Compute the log sum exp (numerator) of the NCE loss.
    logsumexp_pos = tf.reduce_logsumexp(sim_pos, axis=1)
    # Compute the log sum exp (denominator) of the NCE loss.
    logsumexp_all = tf.reduce_logsumexp(sim_all, axis=1)
    # Compute the loss.
    loss = logsumexp_all - logsumexp_pos

    if embed_2_mask is not None:
      # Divide by number of examples.
      valid_examples = tf.reduce_sum(embed_2_mask)
      scaling_factor = tf.math.divide_no_nan(tf.cast(tf.size(embed_2_mask),
                                                     dtype=tf.float32),
                                             valid_examples)

      loss = loss * embed_2_mask * scaling_factor

    # If there are no valid examples, logsumexp_all could be a NaN.
    loss = tf.where(tf.math.is_nan(loss),
                    tf.zeros(loss.shape),
                    tf.identity(loss))

    # Average over batch samples.
    loss = tf.reduce_mean(loss)

    return loss

  def _get_empty_logs(self):
    """Dummy loss and logs with same structure as original call."""

    id_losses = []
    metrics_to_log = {}
    if self._vid_txt_weight > 0.:
      id_losses.append("vidtxt_loss")
    if self._vid_aud_weight > 0.:
      id_losses.append("vidaud_loss")
    if self._aud_txt_weight > 0.:
      id_losses.append("audtxt_loss")

    for id_loss in id_losses:
      metrics_to_log[id_loss] = 0.

    return 0., metrics_to_log

  def __call__(self,
               labels,
               predictions,
               training=True,
               replicator = None):
    """Cross modal NCE / MIL-NCE loss as in https://arxiv.org/abs/2006.16228.

    Args:
      labels: Dictionary containing text masks:
        - audio_mask: Tensor of shape [B, 1] containing indicators of whether
          the audio should be used.
        - text_mask: Tensor of shape [B, 1] containing indicators of whether
          the text should be used.
      predictions: Dictionaries containing all modalidity-specific outputs:
        - video: Dictionaries containing tensor of video embeddings to
          compare against audio and text of shape [B, D] where B is the
          number of video embeddings and D the embedding dimension.
        - audio: Dictionaries containing tensor of audio embeddings to
          compare against video and text of shape [B, D] where B is the
          number of video embeddings and D the embedding dimension.
        - text: Dictionaries containing tensor of text embeddings to
          compare against video and audio of shape [B * L, D] where B is the
          number of text embeddings, D the embedding dimension and L is the
          number of positive candidate narrations for each video clip.
      training: a bool label indicating whether the loss should be considered
        or not.
      replicator: tensorflow replicator used for cross replica negative.

    Returns:
      The computed MIL-NCE loss.
    """

    if not training:
      return self._get_empty_logs()

    video_embd = predictions["video"]
    audio_embd = predictions["audio"]
    text_embd = predictions["text"]
    audio_mask = labels.get("audio_mask", None)
    text_mask = labels.get("text_mask", None)

    # Calculate and aggregate NCE / MIL-NCE losses
    all_losses = []
    all_weights = []
    metrics_to_log = {}
    if self._vid_txt_weight > 0.:
      # Get MIL-NCE between video and text.
      vid_txt_loss = self._calculate_nce(
          video_embd["totxt"],
          text_embd["tovid"],
          embed_2_mask=text_mask,
          replicator=replicator
          ) * self._vid_txt_weight
      all_losses.append(vid_txt_loss)
      all_weights.append(self._vid_txt_weight)
      metrics_to_log["vidtxt_loss"] = vid_txt_loss

    if self._vid_aud_weight > 0.:
      # Get NCE between video and audio.
      vid_aud_loss = self._calculate_nce(
          video_embd["toaud"],
          audio_embd["tovid"],
          embed_2_mask=audio_mask,
          replicator=replicator
          ) * self._vid_aud_weight

      all_losses.append(vid_aud_loss)
      all_weights.append(self._vid_aud_weight)
      metrics_to_log["vidaud_loss"] = vid_aud_loss

    if self._aud_txt_weight > 0.:
      # Get MIL-NCE between audio and text.
      aud_txt_loss = self._calculate_nce(
          audio_embd["totxt"],
          text_embd["toaud"],
          embed_2_mask=text_mask,
          replicator=replicator
          ) * self._aud_txt_weight
      all_losses.append(aud_txt_loss)
      all_weights.append(self._aud_txt_weight)
      metrics_to_log["audtxt_loss"] = aud_txt_loss

    loss = tf.add_n(all_losses) / tf.add_n(all_weights)

    return loss, metrics_to_log


def build_loss(params):
  """Initializes and returns an instance of the desired loss object."""

  loss_name = params.name.lower()
  if loss_name == "symmetric_nce":
    loss_class = SymmetricNCE

  elif loss_name == "asymmetric_nce":
    loss_class = AsymmetricNCE

  else:
    raise NotImplementedError

  return loss_class(params)
