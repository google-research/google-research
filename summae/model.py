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

"""Tensorflow modeling functions."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import math

from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf  # tf

from summae import beam_search
from summae import sample
from summae import transformer as trf
from summae import util
from tensorflow.contrib import layers as contrib_layers
from tensorflow.python.ops import inplace_ops  # pylint: disable=g-direct-tensorflow-import

# pylint: disable=invalid-name
# Note on tensor var naming style. We suffix a hint of the shape
# as follows: t_BxD, means tensor of shape (B, D) where B and D are
# shorthand for dimensions defined in comments.
# V = vocab size
# E = embedding dimension
# B = batch size
# S = max sequence length
# H = RNN state size
# Z = latent size

# pylint: disable=g-long-lambda

# For small weights initialization
SMALL_WEIGHTS_SD = 0.02


def prepend_token(x_BxLxE, embed_VxE, token_id):
  """Prepend token_id embedding to a batch of sequence tensors.

  Args:
    x_BxLxE: sequence tensor
    embed_VxE: embedding matrix
    token_id: token to prepend

  Returns:
    x_Bx(L+1)xD tensor
  """
  first_1xE = tf.nn.embedding_lookup(embed_VxE, [token_id])
  first_batched_Bx1xE = tf.tile(
      tf.expand_dims(first_1xE, 1),  # 1x1xE
      [tf.shape(x_BxLxE)[0], 1, 1])
  return tf.concat([first_batched_Bx1xE, x_BxLxE], 1, name='prepend')


def cos_loss(x_BxZ, y_BxZ):
  """Return average cosine distance loss between a batch of 2 vectors."""
  x_n = tf.nn.l2_normalize(x_BxZ, axis=1)
  y_n = tf.nn.l2_normalize(y_BxZ, axis=1)
  # This only works if both x_n and y_n have same shape.
  return tf.reduce_mean(1 - tf.reduce_sum(tf.multiply(x_n, y_n), axis=1))


def avg_cos_dist(x_BxZ, y_BxNxZ):
  """Computes (batch) average cosine distance between a vector and N vectors."""
  # Returns tensor with shape (B)
  xn_BxZ = tf.nn.l2_normalize(x_BxZ, axis=1)
  yn_BxNxZ = tf.nn.l2_normalize(y_BxNxZ, axis=2)
  return 1 - tf.reduce_mean(tf.reduce_sum(tf.multiply(tf.expand_dims(xn_BxZ, 1),
                                                      yn_BxNxZ),
                                          axis=2),
                            axis=1)


def reduce_mean_weighted(x_BxNxZ, b_N):
  """Weighted reduce mean along second dimension."""
  b_N = b_N / tf.reduce_sum(b_N)
  b_BxN = tf.reshape(
      tf.tile(b_N, [tf.shape(x_BxNxZ)[0]]),
      tf.shape(x_BxNxZ)[:-1])
  b_BxNx1 = tf.expand_dims(b_BxN, axis=2)
  # broadcasting the 3rd dimension
  x_avg_BxZ = tf.reduce_sum(x_BxNxZ * b_BxNx1, axis=1)
  return x_avg_BxZ


def norm_loss(x_BxZ, y_BxZ):
  """Returns mean squared (normalized) distance in norms."""
  a = tf.norm(x_BxZ, axis=1)
  b = tf.norm(y_BxZ, axis=1)
  return tf.reduce_mean(
      tf.square((a - b) / tf.maximum(a, b))
  )


def add_eos_2d(ids_BxL):
  """Add 1 to end of batch of id sequences, padded with 0s.

  The second dimension is increased by 1.

  Args:
    ids_BxL: 2d-tensor of int64 representing batch of sequences

  Returns:
    ids with EOS token at end of sequence
  """
  ra = tf.RaggedTensor.from_tensor(ids_BxL, padding=0)
  eos = tf.ones((ra.nrows(), 1), dtype=tf.int64)
  rae = tf.concat([ra, eos], axis=1)
  return rae.to_tensor(default_value=0)


def shift_right_3d(x):
  """Shift the second dimension of x right by one."""
  return tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]


def id_seq_length(x_BxS):
  """Assumes pads are always at the end."""
  return tf.reduce_sum(
      tf.cast(tf.math.not_equal(x_BxS,
                                tf.constant(util.PAD_ID,
                                            dtype=tf.int64)),
              tf.int64),
      axis=1)


def create_perm_label_table(num_s_per_p):
  """Create a table for permutation-label lookup.

  For example, if num_s_per_p is 3, will create a table where keys are strings
  of all possible permunations: {'012', '021', '102', '120', '201', '210'} and
  the values are unique IDs of each permunation: {0, 1, 2, 3, 4, 5}.

  Args:
    num_s_per_p: number of sentences per paragraph, should be an int

  Returns:
    A lookup table that returns an unique class ID given a permutation.
  """
  assert isinstance(num_s_per_p, int), type(num_s_per_p)
  perms = list(itertools.permutations(list(range(num_s_per_p))))
  # For example, when num_s_per_p = 3, perms is as follows:
  # [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)].
  perms_as_str = []
  for perm in perms:
    perms_as_str.append(''.join(str(n) for n in perm))
  # perms_as_str = ['012', '021', '102', '120', '201', '210'].
  num_s_per_p_factorial = math.factorial(num_s_per_p)  # 6
  # This table will map each string in perms_as_str to a unique ID 0 ~ 5.
  return tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(
          tf.convert_to_tensor(perms_as_str),
          tf.range(num_s_per_p_factorial)), -1)


def shuffle_paragraph_with_labels(s_ids_BxNxL, num_s_per_p):
  """Shuffle sentence order and assign unique class ID to each shuffling order.

  Consists of three steps: (1) create a lookup table that maps
  each permutation to a unique class, (2) shuffle the sentences, and (3) pad the
  shuffled sentences to full paragraphs.

  Args:
    s_ids_BxNxL: 3d-tensor of int64 representing batch of sentences
    num_s_per_p: number of sentences per paragraph

  Returns:
    a 2d-tensor of int64 representing the shuffled paragraphs and the
    corresponding permunation class.
  """
  # perm_label_table is a look-up table that given a permutation it returns an
  # unique class ID.
  perm_label_table = create_perm_label_table(num_s_per_p)

  def shuffle_sents(s_ids_NxL):
    """Shuffle one paragraph."""
    shuffle_idx_N = tf.random.shuffle(tf.range(num_s_per_p))
    s_shuffled_ids_NxL = tf.gather(s_ids_NxL, shuffle_idx_N, axis=0)
    shuffle_idx_N = tf.strings.reduce_join(tf.strings.as_string(shuffle_idx_N))
    return s_shuffled_ids_NxL, perm_label_table.lookup(shuffle_idx_N)

  s_shuffled_ids_BxNxL, labels_B = tf.map_fn(
      fn=shuffle_sents, elems=s_ids_BxNxL, dtype=(tf.int64, tf.int32))

  return convert_sents_to_paragraph(s_shuffled_ids_BxNxL, num_s_per_p), labels_B


def convert_sents_to_paragraph(s_ids_BxNxL, num_s_per_p):
  """Convert sentences into a paragraph."""
  max_possible_paragraph_length = num_s_per_p * tf.shape(s_ids_BxNxL)[2]
  # TODO(peterjliu): For TPU pad to max paragraph length.

  def convert_s_to_p(s_ids_NxL):
    """Convert sentences into a paragraph per example."""
    # Substract 1 to get rid of the EOS_ID at the end of each sentence before
    # paddings.
    s_seq_lengths_N = id_seq_length(s_ids_NxL) - 1
    ra = tf.RaggedTensor.from_tensor(s_ids_NxL, lengths=s_seq_lengths_N)
    concat_p = ra.values
    # Add EOS_ID back to the paragraph.
    concat_p = tf.pad(concat_p, [[0, 1]], constant_values=util.EOS_ID)
    # Pad back PAD_ID to the maximum possible paragraph length.
    concat_p = tf.pad(
        concat_p, [[0, max_possible_paragraph_length - tf.shape(concat_p)[0]]],
        constant_values=util.PAD_ID)
    return concat_p

  p_ids_BxS = tf.map_fn(fn=convert_s_to_p, elems=s_ids_BxNxL, dtype=tf.int64)
  # Remove redundant paddings in the end. Won't affect performance though.
  return tf.RaggedTensor.from_tensor(p_ids_BxS,
                                     padding=util.PAD_ID).to_tensor()


def corrupt_paragraph_with_scheme(s_ids_BxNxL, scheme):
  """Corrupt the sentences in a paragraph with the given scheme.

  Options for corrupting a paragraph are: (1) swap the last two sentences,
  (2) swap neighboring sentences at a random position, and (3) randomly
  shuffle the sentence order.

  Args:
    s_ids_BxNxL: a 3d-tensor of int64 representing batch of sentences
    scheme: the paragraph corrupting scheme. Options are 'last_two',
            'neighbor_two', and 'rand_shuffle'

  Returns:
    a 2d-tensor of int64 representing the corrupted paragraphs
  """
  assert scheme in ('last_two', 'neighbor_two', 'rand_shuffle'), scheme
  num_s_per_p = tf.shape(s_ids_BxNxL)[1]

  def swap_sents(s_ids_NxL):
    """Swap one paragraph."""
    if scheme == 'last_two':
      # Swap only the last two sentences.
      gather_idx_N = tf.concat([tf.range(num_s_per_p - 2),
                                [num_s_per_p - 1, num_s_per_p - 2]], axis=0)
    elif scheme == 'neighbor_two':
      idx_to_swap = tf.random.uniform([], minval=1, maxval=num_s_per_p,
                                      dtype=tf.int32)
      gather_idx_N = tf.concat([tf.range(idx_to_swap - 1),
                                [idx_to_swap, idx_to_swap - 1],
                                tf.range(idx_to_swap + 1, num_s_per_p)], axis=0)
    else:  # 'rand_shuffle'
      gather_idx_N = tf.random.shuffle(tf.range(num_s_per_p))
    return tf.gather(s_ids_NxL, gather_idx_N, axis=0)

  s_swapped_ids_BxNxL = tf.map_fn(fn=swap_sents, elems=s_ids_BxNxL,
                                  dtype=tf.int64)
  return convert_sents_to_paragraph(s_swapped_ids_BxNxL, num_s_per_p)


def random_mask_like(x, x_len, mask_rate, not_mask_eos=True):
  """Generate mask tensor (1 for mask, 0 for unmask) as the size of x.

  Generate a boolean tensor for masking input x using IID Bernoulli
  distribution.

  Args:
    x: a tensor of at least 2D
    x_len: length of the element in x.
           The element has the pattern: x_1 x_2 ... x_t <eos> <pad> <pad>...
    mask_rate: rate for masking
    not_mask_eos: if true, <eos> will not be masked

  Returns:
    a boolean tensor where 1 indicates the position to mask.
  """
  x_unif = tf.random.uniform(tf.shape(x), minval=0, maxval=1)
  mask_rand = tf.cast(x_unif > (1 - mask_rate), tf.int64)
  if not_mask_eos:
    # eos is not masked, only masking word tokens in sentence
    x_len = x_len - 1
  mask_len = tf.sequence_mask(x_len, maxlen=tf.shape(x)[1], dtype=tf.int64)
  # zero out for EOS and all PAD_IDs, x_len=sentence_len + 1 (EOS)
  mask = mask_rand * mask_len
  return mask


def mask_ids(x_BxS, x_len_B, mask_rate, mask_id):
  """Randomly (IID) replace ids with the mask id.

  Replace ids with the mask_id with IID Bernoulli distribution with mask_rate.
  PAD IDs can be flipped, but the loss won't be affected because padded
  positions are not counted.

  Args:
    x_BxS: 2d-tensor of int64 representing batch of sequences
    x_len_B: length of sequences
    mask_rate: the probability of being masked at each position
    mask_id: the vocab id for "<MASK>" token

  Returns:
    2d-tensor as the size of x_BxS with positions randomly replaced by mask_id
  """
  assert 0.0 <= mask_rate <= 1.0
  assert mask_id
  # generate mask tensor with probability of being masked = mask_rate
  mask_BxS = random_mask_like(x_BxS, x_len_B, mask_rate)
  # generate tensor with constant MASK_ID
  replace_BxS = tf.ones_like(x_BxS, tf.int64) * mask_id
  # masked positions (mask=1) are replaced by MASK_ID, the rest remains the same
  x_BxS_mask = x_BxS * (1 - mask_BxS) + replace_BxS * mask_BxS
  return x_BxS_mask


def apply_mask_to_embs(x_BxSxE, mask_BxSx1, mask_emb_E):
  """Apply mask tensor to embedding tensor.

  Args:
    x_BxSxE: original tensor, last dimension is for embeddings
    mask_BxSx1: mask boolean tensor of the same size as x_BxSxE except that last
      dimension is 1
    mask_emb_E: the embedding for mask token <MASK>

  Returns:
    the same as the original tensor except for replacing the mask embedding to
    the corresponding mask ids.
  """
  # broadcasting from BxSx1 to BxSxE
  replace_emb_BxSxE = tf.reshape(
      tf.tile(mask_emb_E, [tf.shape(x_BxSxE)[0] * tf.shape(x_BxSxE)[1]]),
      tf.shape(x_BxSxE))
  return x_BxSxE * (1 - mask_BxSx1) + replace_emb_BxSxE * mask_BxSx1


def mask_embs(x_BxSxE, x_len_B, mask_rate, mask_emb_E):
  """Same as mask_embs_with_mask but doesn't not return mask."""
  masked_BxSxE, _ = mask_embs_with_mask(x_BxSxE, x_len_B, mask_rate, mask_emb_E)
  return masked_BxSxE


def mask_embs_with_mask(x_BxSxE, x_len_B, mask_rate, mask_emb_E):
  """Randomly (IID) replace word embeddings with the mask token embedding.

  Replace token embedding with mask token embedding following IID
  bernoulli distribution with mask_rate. PAD IDs can be flipped,
  but loss won't be affected because padded positions are not counted.

  Args:
    x_BxSxE: 3d-tensor representing batch of embedded sentences
    x_len_B: lengths of the sentences in x_BxSxE
    mask_rate: the probability of being masked at each position
    mask_emb_E: the embedding for "<MASK>" token

  Returns:
    3d-tensor as the size of x_BxSxE with embeddings randomly replaced by
    mask token embedding
    2d-tensor representing the mask
  """
  if mask_rate == 0.0:
    return x_BxSxE, tf.zeros([tf.shape(x_BxSxE)[0], tf.shape(x_BxSxE)[1]],
                             dtype=tf.float32)
  else:
    mask_BxS = random_mask_like(
        x_BxSxE[:, :, 0], x_len_B, mask_rate, not_mask_eos=True)
    mask_BxSx1 = tf.cast(tf.expand_dims(mask_BxS, axis=2), tf.float32)
    return apply_mask_to_embs(
        x_BxSxE, mask_BxSx1, mask_emb_E), tf.cast(mask_BxS, dtype=tf.float32)


def where_is_cloze_mask(x_BxSxE, x_len_B, mask_emb_E):
  """Finds the position of masked tokens.

  Args:
    x_BxSxE: a tensor representing the token embeddings
    x_len_B: sequence length of x_BxSxE
    mask_emb_E: embedding of the MASK token to be located

  Returns:
    2d-tensor of shape (B, S) with 1 indicating masked and 0 unmasked
  """
  seq_mask_BxS = tf.sequence_mask(lengths=x_len_B, maxlen=tf.shape(x_BxSxE)[1])
  cloze_mask_BxS = tf.reduce_all(tf.equal(x_BxSxE, mask_emb_E), axis=2)
  return tf.cast(tf.logical_and(seq_mask_BxS, cloze_mask_BxS), tf.float32)


def compute_nsp_pretrain_loss(e_s, s_enc_inputs_YxLxE, s_seq_lengths_Y,
                              pooling, is_training, batch_size,
                              not_next_diff_p_prob=0.0):
  """Computes the loss for NSP (next sentence prediction) pre-training.

  Args:
    e_s: an Encoder object, the sentence encoder
    s_enc_inputs_YxLxE: 3d-tensor representing the input embeddings to e_s
    s_seq_lengths_Y: sequence length of s_enc_inputs_YxLxE
    pooling: the pooling method
    is_training: whether in training mode
    batch_size: batch size
    not_next_diff_p_prob: the percentage of negative sentences that will be
      sampled from a different paragraph

  Returns:
    pretrain_loss: float scalar loss
    logits_2B: output logits
    labels_2B: labels
  """
  assert isinstance(e_s, Encoder)
  # There's always 50% of the sampled s2 is the next sentence of s1. For the
  # rest 50%, the sampled s2 is not the next sentence of s1. How these
  # 'negative' s2 are sampled depends on not_next_diff_p_prob. For example, if
  # not_next_diff_p_prob is 0.25, then 25% of the negative s2 is sampled from a
  # different paragraph as s1, and the rest 75% sampled from the same.
  s_enc_YxZ, _ = e_s.encode(s_enc_inputs_YxLxE, s_seq_lengths_Y, pooling,
                            is_training)
  latent_size = tf.shape(s_enc_YxZ)[1]
  s_enc_BxNxZ = tf.reshape(s_enc_YxZ, [batch_size, -1, latent_size])
  # We take the n-th (n randomly sampled from [0, N - 2]) and (n + 1)-th
  # sentences from each paragraph within a batch as positive (s1, s2) pairs.
  num_s_per_p = tf.shape(s_enc_BxNxZ)[1]
  s_idx_to_take = tf.random.uniform([], maxval=num_s_per_p - 1, dtype=tf.int32)
  pos_idx = s_idx_to_take + 1
  # For example, if s_idx_to_take = 2, pos_idx = 3, then nsp_neg_idx can be
  # 0, 1, or 4 (num_s_per_p = 5).
  neg_idx = tf.mod(
      pos_idx + tf.random.uniform([], minval=1, maxval=num_s_per_p - 1,
                                  dtype=tf.int32), num_s_per_p)
  s_cur_BxZ = tf.gather(s_enc_BxNxZ, s_idx_to_take, axis=1)
  s_next_BxZ = tf.gather(s_enc_BxNxZ, pos_idx, axis=1)
  s_not_next_BxZ = tf.gather(s_enc_BxNxZ, neg_idx, axis=1)
  if not_next_diff_p_prob > 0:
    tf.logging.info('%d of the negative samples are drawn from different '
                    'paragraphs.', not_next_diff_p_prob)
    # Round to the nearest and smaller int.
    num_not_next_diff_p = int(int(batch_size) * not_next_diff_p_prob)
    p_shift = tf.random.uniform([], minval=1, maxval=num_not_next_diff_p,
                                dtype=tf.int32)
    s_enc_rolled_dBxNxZ = tf.roll(s_enc_BxNxZ[:num_not_next_diff_p],
                                  shift=p_shift, axis=0)
    s_not_next_BxZ = tf.concat([
        # from same p
        s_not_next_BxZ[num_not_next_diff_p:],
        # from diff p
        tf.gather(s_enc_rolled_dBxNxZ, s_idx_to_take, axis=1)], axis=0)
  logits_2B = tf.concat([
      tf.reduce_sum(tf.multiply(s_cur_BxZ, s_next_BxZ), axis=1),
      tf.reduce_sum(tf.multiply(s_cur_BxZ, s_not_next_BxZ), axis=1)], axis=0)
  labels_2B = tf.concat([tf.ones(batch_size), tf.zeros(batch_size)], axis=0)
  pretrain_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(
          logits=logits_2B, labels=labels_2B))
  return pretrain_loss, logits_2B, labels_2B


def compute_cpp_pretrain_loss(e_p, s_ids_BxNxL, embedding_VxE, pooling,
                              is_training, scheme='', p_enc_inputs_BxSxE=None,
                              p_seq_lengths_B=None):
  """Computes the loss for CPP (corrupted paragraph prediction) pre-training.

  Args:
    e_p: an Encoder object, the paragraph encoder
    s_ids_BxNxL: sentence token ids
    embedding_VxE: the embedding look-up table
    pooling: the pooling method
    is_training: whether in training mode
    scheme: the scheme to corrupt paragraphs
    p_enc_inputs_BxSxE: 3d-tensor representing the clean paragraph embeddings
      where we sample uncorrupted p from
    p_seq_lengths_B: sequence length of p_enc_inputs_BxSxE

  Returns:
    pretrain_loss: float scalar loss
    logits_B: output logits
    labels_B: labels
  """
  assert isinstance(e_p, Encoder)
  # Take the first half batch to generate corrupted paragraphs.
  batch_size = tf.shape(s_ids_BxNxL)[0]
  hB = batch_size // 2
  s_ids_hBxNxL = s_ids_BxNxL[:hB]
  p_swapped_ids_hBxS1 = corrupt_paragraph_with_scheme(s_ids_hBxNxL, scheme)
  p_swapped_seq_lengths_hB = id_seq_length(p_swapped_ids_hBxS1)
  # TODO(peterjliu): Avoid two embedding_lookup operations.
  p_swapped_enc_inputs_hBxS1xE = tf.nn.embedding_lookup(embedding_VxE,
                                                        p_swapped_ids_hBxS1)
  p_swapped_enc_hBxZ, _ = e_p.encode(p_swapped_enc_inputs_hBxS1xE,
                                     p_swapped_seq_lengths_hB, pooling,
                                     is_training)
  # Create clean paragraph encoding samples by directly taking the second half
  # batch of p_enc_inputs_BxSxE (original clean p embeddings).
  p_clean_enc_inputs_hBxS2xE = p_enc_inputs_BxSxE[hB:]
  p_clean_seq_lengths_hB = p_seq_lengths_B[hB:]
  p_clean_enc_hBxZ, _ = e_p.encode(p_clean_enc_inputs_hBxS2xE,
                                   p_clean_seq_lengths_hB, pooling, is_training)
  # Note that we separately compute the swapped and clean paragraph encodings
  # because S1 and S2 can be different.
  enc_BxZ = tf.concat([p_swapped_enc_hBxZ, p_clean_enc_hBxZ], axis=0)
  with tf.variable_scope('ae_cpp_pretrain'):
    # A simple linear binary classifier.
    logits_B = tf.squeeze(tf.layers.dense(enc_BxZ, 1, activation=None,
                                          name='cpp_kernel'))
  labels_B = tf.concat([tf.ones(hB), tf.zeros(hB + batch_size % 2)], axis=0)
  pretrain_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_B, logits=logits_B))
  return pretrain_loss, logits_B, labels_B


def compute_adversarial_loss(features_2BxZ, labels_2B, disc_fn):
  """Computes and returns both the discriminator loss and the adversarial loss.

  To make samples from a source domain and those from a target domain
  indistinguishable. For example, in critic, the source domain is the paragraph
  data space and the target domain is the sentence data space; in prior critic,
  the source domain is the paragraph/sentence data space and the target domain
  is a prior distribution (e.g., Gaussian).

  Args:
    features_2BxZ: 2d-tensor representing samples from both source and target
                   domains
    labels_2B: 1d-tensor representing the labels
    disc_fn: a function that defines the discriminator; it takes a feature
             tensor and outputs the logits.

  Returns:
    d_loss: discriminator scalar loss
    adv_loss: adversarial scalar loss
    logits_2B: discriminator logits
  """
  batch_size = tf.shape(features_2BxZ)[0] // 2
  logits_2B = disc_fn(features_2BxZ)
  d_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
      logits=logits_2B, labels=labels_2B))

  adv_logits_B = logits_2B[:batch_size]
  adv_loss = tf.reduce_mean(
      tf.nn.sigmoid_cross_entropy_with_logits(logits=adv_logits_B,
                                              labels=tf.zeros([batch_size])))
  return d_loss, adv_loss, logits_2B


def compute_reverse_p_loss(d_p2, p_ids_BxS, p_enc_BxZ, p_enc_inputs_BxSxE,
                           p_seq_lengths_B):
  """Computes the loss for reverse p reconstruction.

  Args:
    d_p2: a Decoder object
    p_ids_BxS: original token ids, used to create reversed token ids to predict
    p_enc_BxZ: 2d-tensor of paragraph encodings
    p_enc_inputs_BxSxE: embedded encoder input tensor, used to create decoder
      input for teacher-forcing
    p_seq_lengths_B: sequence length of p_enc_inputs_BxSxE

  Returns:
    loss: a scalar of cross-entropy loss
    metrics: dict of string to tf.metrics
  """
  assert isinstance(d_p2, Decoder)
  # p_ids: 1 2 3 <eos>
  # p2_ids: 3 2 1 <eos>
  p2_ids_BxS = tf.reverse_sequence(
      p_ids_BxS,
      p_seq_lengths_B - 1,  # Don't include EOS
      seq_axis=1)
  p2_enc_inputs_BxSxE = tf.reverse_sequence(
      p_enc_inputs_BxSxE,
      p_seq_lengths_B - 1,  # Don't include EOS
      seq_axis=1)
  # p2_dec_inputs: <pad> 3 2 1 (embedded)
  p2_dec_inputs_BxSxE = shift_right_3d(p2_enc_inputs_BxSxE)
  p2_logits_BxSxV = d_p2.teacher_force(
      p_enc_BxZ, p2_dec_inputs_BxSxE, p_seq_lengths_B)
  loss, metrics = ce_loss(p2_logits_BxSxV, p2_ids_BxS, p_seq_lengths_B)
  return loss, metrics


def compute_c_avg2_loss(s_enc_YxZ, p_enc_BxZ):
  """Computes the cosine avg. distance loss.

  Args:
    s_enc_YxZ: 2d-tensor representing the sentence encodings
    p_enc_BxZ: 2d-tensor representing the paragraph encodings

  Returns:
    a scalar loss representing the avg. cosine distance
  """
  batch_size = tf.shape(p_enc_BxZ)[0]
  latent_size = tf.shape(p_enc_BxZ)[1]
  s_enc_BxNxZ = tf.reshape(s_enc_YxZ, [batch_size, -1, latent_size])
  return tf.reduce_mean(avg_cos_dist(p_enc_BxZ, s_enc_BxNxZ))


def compute_critic_loss(s_enc_YxZ, p_enc_BxZ, n_hiddens, scope):
  """Computes the critic loss.

  Args:
    s_enc_YxZ: 2d-tensor representing the sentence encodings
    p_enc_BxZ: 2d-tensor representing the paragraph encodings
    n_hiddens: hidden dim. of the MLP discriminator
    scope: scope name for the discriminator

  Returns:
    d_loss: discriminator scalar loss
    adv_loss: adversarial scalar loss
    d_logits_2B: discriminator logits
    d_labels_2B: discriminator labels
  """
  batch_size = tf.shape(p_enc_BxZ)[0]
  s_enc_BxZ = randomly_sample_rows(s_enc_YxZ, batch_size)
  d_features_2BxZ, d_labels_2B = get_features_labels(p_enc_BxZ, s_enc_BxZ)
  d_loss, adv_loss, d_logits_2B = compute_adversarial_loss(
      d_features_2BxZ, d_labels_2B, disc_fn=get_discriminator(n_hiddens, scope))
  return d_loss, adv_loss, d_logits_2B, d_labels_2B


class Encoder(object):
  """Abstract sequence encoder."""

  def encode(self, embs_BxLxE, seq_lengths_B, pooling='mean', is_training=None):
    """Encodes sequence of embeddings to fixed-length vector representation.

    Args:
      embs_BxLxE: sequence embeddings tensor, shape (batch, length, embed_dim)
      seq_lengths_B: sequence lengths
      pooling: pooling method for generating a single vector from encoder output
               features representing the input sequence
      is_training: a bool indicating whether in training mode; useful to
                   determine whether to turn on layers like dropout

    Returns:
      encoding_BxH: tensor shape (B, hidden_size)
      kl_loss: scalar tensor for variational models.
    """
    pass

  def encode_with_output(
      self, embs_BxLxE, seq_lengths_B, pooling='mean', is_training=None):
    """Outputs the encoder features before pooling as well."""
    pass

  def encode_dim(self):
    pass


class TransformerEncoder(Encoder):
  """Transformer-based Encoder."""

  def __init__(self, num_layers, num_heads, hidden_size, filter_size,
               attention_dropout, relu_dropout, postprocess_dropout,
               latent_size, scope=None):
    self.scope = 'transformer_encoder' if scope is None else scope
    self.num_layers = num_layers
    self.num_heads = num_heads
    self.hidden_size = hidden_size
    self.filter_size = filter_size
    self.latent_size = latent_size
    with tf.variable_scope(self.scope):
      self.encoder = trf.TransformerEncoder(
          hidden_size=hidden_size,
          filter_size=filter_size,
          num_layers=num_layers,
          num_heads=num_heads,
          attention_dropout=attention_dropout,
          relu_dropout=relu_dropout,
          postprocess_dropout=postprocess_dropout)

      if hidden_size != latent_size:
        # Projection matrix H -> Z
        self.W_HxZ = tf.get_variable(
            name='W_hz',
            shape=(hidden_size, latent_size),
            initializer=tf.truncated_normal_initializer(
                stddev=SMALL_WEIGHTS_SD))
        self.b_Z = tf.get_variable(name='b_hz', shape=(latent_size),
                                   initializer=tf.zeros_initializer())

  def encode(self, embs_BxLxE, seq_lengths_B, pooling='mean', is_training=None):
    encoded_BxZ, _ = self.encode_with_output(embs_BxLxE, seq_lengths_B, pooling,
                                             is_training)
    return encoded_BxZ, None

  def encode_with_output(self, embs_BxLxE, seq_lengths_B, pooling='mean',
                         is_training=None):
    assert pooling in ('mean', 'first')
    with tf.variable_scope(self.scope):
      # In Transformer, E == H.
      embs_BxLxH = embs_BxLxE
      seq_mask_BxL = tf.sequence_mask(seq_lengths_B, tf.shape(embs_BxLxH)[1],
                                      dtype=tf.float32)
      encoded_BxLxH = self.encoder(embs_BxLxH, 1 - seq_mask_BxL,
                                   training=is_training, cache=None)
      # 'first' pooling simply takes the output representation of the first
      # token while 'mean' pooling averages the representations of the entire
      # output sequence, excluding those that correspond to padded tokens.
      if pooling == 'first':
        encoded_BxH = encoded_BxLxH[:, 0, :]
      else:  # 'mean' pooling
        seq_mask_BxLxH = tf.tile(
            tf.expand_dims(seq_mask_BxL, axis=2), [1, 1, self.hidden_size])
        weighted_sum_BxH = tf.reduce_sum(
            tf.multiply(encoded_BxLxH, seq_mask_BxLxH), axis=1)
        encoded_BxH = tf.divide(
            weighted_sum_BxH,
            tf.tile(tf.expand_dims(
                tf.cast(tf.maximum(seq_lengths_B, 1),  # ensure not divided by 0
                        dtype=weighted_sum_BxH.dtype), axis=1),
                    [1, self.hidden_size]))
      if self.hidden_size == self.latent_size:
        return encoded_BxH, encoded_BxLxH
      encoded_BxZ = tf.nn.bias_add(
          tf.matmul(encoded_BxH, self.W_HxZ), self.b_Z, name='enc_out_proj')
      return encoded_BxZ, encoded_BxLxH


class GruEncoder(Encoder):
  """GRU-based encoder."""

  def __init__(self,
               hidden_size,
               num_layers=1,
               latent_size=0,
               scope=None,
               bidirect_encode=False):
    self.scope = 'gru_encoder' if scope is None else scope
    self.bidirect_encode = bidirect_encode
    with tf.variable_scope(self.scope):
      self.enc_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(
          [tf.nn.rnn_cell.GRUCell(hidden_size) for _ in range(num_layers)])
      if bidirect_encode:
        self.enc_rnn_cell_rev = tf.nn.rnn_cell.MultiRNNCell(
            [tf.nn.rnn_cell.GRUCell(hidden_size) for _ in range(num_layers)])
    self.H = hidden_size * 2 if bidirect_encode else hidden_size
    self.Z = latent_size if latent_size > 0 else hidden_size
    if self.H != self.Z:
      # Projection matrix H -> Z
      with tf.variable_scope(self.scope):
        self.W_HxZ = tf.get_variable(
            name='W_hz',
            shape=(self.H, self.Z),
            initializer=tf.truncated_normal_initializer(
                stddev=SMALL_WEIGHTS_SD))
        self.b_Z = tf.get_variable(
            name='b_hz', shape=(self.Z), initializer=tf.zeros_initializer())

  def encode_dim(self):
    return self.Z

  def encode(self, embs_BxLxE, seq_lengths_B, pooling='mean', is_training=None):
    encoded_BxZ, _ = self.encode_with_output(embs_BxLxE, seq_lengths_B, pooling,
                                             is_training)
    return encoded_BxZ, None

  def encode_with_output(self, embs_BxLxE, seq_lengths_B, pooling='mean',
                         is_training=None):
    assert pooling in ('mean', 'last')
    with tf.variable_scope(self.scope):
      if self.bidirect_encode:
        o_2xBxTxH, s_2xNxBxH = tf.nn.bidirectional_dynamic_rnn(
            self.enc_rnn_cell,
            self.enc_rnn_cell_rev,
            embs_BxLxE,
            sequence_length=seq_lengths_B,
            dtype=tf.float32)
        # s_2xNxBxH is a tuple (not a tensor! the '2' in the name is just for
        # readability) of (output_state_fw, output_state_bw), where
        # output_state_fw and output_state_bw are tensors of shape
        # (num_layers, batch_size, hidden_size). Same rule for o_2xBxTxH.
        s_fw_BxH = (s_2xNxBxH[0][-1] if pooling == 'last'
                    else tf.reduce_mean(o_2xBxTxH[0], axis=1))
        s_bw_BxH = (s_2xNxBxH[1][-1] if pooling == 'last'
                    else tf.reduce_mean(o_2xBxTxH[1], axis=1))
        s_BxH = tf.concat(
            [s_fw_BxH, s_bw_BxH], axis=1)  # H = 2 * H for bidirection
      else:
        o_BxTxH, s_NxBxH = tf.nn.dynamic_rnn(
            self.enc_rnn_cell,
            embs_BxLxE,
            sequence_length=seq_lengths_B,
            dtype=tf.float32)
        # o has shape [B, T, H] and averaging across all hidden states will also
        # include the zero-padded vectors after the true sequence length.
        s_BxH = (s_NxBxH[-1] if pooling == 'last'
                 else tf.reduce_mean(o_BxTxH, axis=1))
    if self.H != self.Z:
      enc_BxZ = tf.nn.bias_add(tf.matmul(s_BxH, self.W_HxZ),
                               self.b_Z, name='enc_out_proj')
      return enc_BxZ, None
    return s_BxH, None


class VGruEncoder(Encoder):
  """Variational encoder."""

  def __init__(self, hidden_size, latent_size, scope=None):
    self.scope = 'vgru_encoder' if scope is None else scope
    with tf.variable_scope(self.scope):
      self.enc_rnn_cell = tf.nn.rnn_cell.GRUCell(hidden_size)
    self.H = hidden_size
    self.Z = latent_size
    self.kl_loss_f = None

  def encode_dim(self):
    return self.Z

  def encode(self, embs_BxLxE, seq_lengths_B, pooling='mean', is_training=None):
    with tf.variable_scope(self.scope):
      _, s_BxH = tf.nn.dynamic_rnn(self.enc_rnn_cell, embs_BxLxE,
                                   sequence_length=seq_lengths_B,
                                   dtype=tf.float32)
    return sample_latent(s_BxH, self.Z)


class Decoder(object):
  """Abstract class which decodes vectors into sequences of symbols.

  Usage:
    d = Decoder(...)

    # Train using teacher-forcing.
    dec_outputs = d.teacher_force(cond_vector, dec_inputs)
    loss = f(dec_outputs, targets)
    ...

    # At test time, decode vector to symbol sequence.
    symbs = d.decode_v(encoder_vector)
    print(detokenize(symbs))
  """

  def teacher_force(self, cond_input_BxZ, dec_inputs_BxSxE, seq_lengths_B):
    """Returns decoded outputs by teacher-forcing (for training)."""
    pass

  def decode_v(self, cond_input_BxZ, method=None, first_token=-1, max_steps=0,
               beam_size=None, alpha=None):
    """Decodes up to max_steps given conditioning vector and initial input.

    Args:
      cond_input_BxZ: conditioning tensor, shape=(batch, latent_size)
      method: one of argmax, random
      first_token: token to be used for initial input, must be reserved
      max_steps: max_steps to decode
      beam_size: beam size for beam search
      alpha: float defining the strength of length normalization. Should be
        between 0 (shorter) and 1 (longer)

    Returns:
      dec_symb_BxM: decoded symbol-ids
    """
    pass

  def hidden_dim(self):
    pass


class TransformerDecoder(Decoder):
  """Transformer-based Decoder."""

  def __init__(self, num_layers, num_heads, hidden_size, filter_size,
               attention_dropout, relu_dropout, postprocess_dropout, embed_VxE,
               vocab_size, max_steps, latent_size, tie_embeddings=False,
               cond_by_addition=False, scope=None):
    self.scope = 'transformer_decoder' if scope is None else scope
    self.hidden_size = hidden_size
    self.embed_VxE = embed_VxE
    self.vocab_size = vocab_size
    self.max_steps = max_steps
    self.latent_size = latent_size
    self.tie_embeddings = tie_embeddings
    self.cond_by_addition = cond_by_addition  # cond_by_addition = (Z == E)

    with tf.variable_scope(self.scope):
      if not tie_embeddings:
        self.out_kernel_HxV = tf.get_variable(
            name='output_kernel_HxV',
            shape=[hidden_size, vocab_size],
            initializer=tf.truncated_normal_initializer(
                stddev=SMALL_WEIGHTS_SD))
        self.out_bias_V = tf.get_variable(
            name='output_bias_V',
            shape=[vocab_size],
            initializer=tf.zeros_initializer())
      if not cond_by_addition:
        # Will do conditioning by concatenation. So we need an affine
        # transformation from dim D = Z + E (which equals Z + H) to H.
        self.input_kernel_DxH = tf.get_variable(
            name='input_kernel_DxH',
            shape=[latent_size + hidden_size, hidden_size],
            initializer=tf.truncated_normal_initializer(
                stddev=SMALL_WEIGHTS_SD))
        self.input_bias_H = tf.get_variable(
            name='input_bias_H',
            shape=[hidden_size],
            initializer=tf.zeros_initializer())
      self.decoder = trf.TransformerDecoderOnly(
          hidden_size=hidden_size,
          filter_size=filter_size,
          num_layers=num_layers,
          num_heads=num_heads,
          attention_dropout=attention_dropout,
          relu_dropout=relu_dropout,
          postprocess_dropout=postprocess_dropout)

  def _input_proj(self, x_BxSxD):
    """Project decoder input to dim of hidden size."""
    if self.cond_by_addition:
      return x_BxSxD
    x_BSxD = tf.reshape(x_BxSxD, [-1, self.hidden_size + self.latent_size])
    x_BSxH = tf.nn.bias_add(tf.matmul(x_BSxD, self.input_kernel_DxH),
                            self.input_bias_H)
    return tf.reshape(x_BSxH, [tf.shape(x_BxSxD)[0], -1, self.hidden_size])

  def _output_proj(self, x_BxH):
    """Project output prediction to logits of vocab."""
    # H == E
    if self.tie_embeddings:
      return tf.matmul(x_BxH, self.embed_VxE, transpose_b=True)
    return tf.nn.bias_add(
        tf.matmul(x_BxH, self.out_kernel_HxV), self.out_bias_V)

  def teacher_force(self, cond_input_BxZ, dec_inputs_BxSxE, seq_lengths_B=None):
    # dec_inputs_BxSxE is already either prepended with special tokens
    # (if tie_decs is True) or right-shifted (if tie_decs is False).
    with tf.variable_scope(self.scope):
      cond_input_BxSxZ = tf.tile(tf.expand_dims(cond_input_BxZ, axis=1),
                                 [1, tf.shape(dec_inputs_BxSxE)[1], 1])
      if self.cond_by_addition:  # E == Z
        dec_inputs_BxSxD = dec_inputs_BxSxE + cond_input_BxSxZ
      else:
        dec_inputs_BxSxD = tf.concat([dec_inputs_BxSxE, cond_input_BxSxZ],
                                     axis=2)  # D = E + Z
      dec_inputs_BxSxH = self._input_proj(dec_inputs_BxSxD)
      dec_outputs_BxSxH = self.decoder(inputs_BxTxH=dec_inputs_BxSxH,
                                       training=True)
      dec_outputs_flat_BSxH = tf.reshape(dec_outputs_BxSxH,
                                         [-1, self.hidden_size])
      logits_BxSxV = tf.reshape(
          self._output_proj(dec_outputs_flat_BSxH),
          [tf.shape(dec_outputs_BxSxH)[0], -1, self.vocab_size])
      return logits_BxSxV

  def _get_symbols_to_logits_fn(self, cond_input_BxMxZ, beam_size):
    """Returns a decoding function that calculates logits of the next tokens."""
    max_steps = tf.shape(cond_input_BxMxZ)[1]
    latent_size = tf.shape(cond_input_BxMxZ)[2]
    cond_input_BKxMxZ = tf.reshape(
        tf.map_fn(lambda x: tf.tile(x, [beam_size, 1]), cond_input_BxMxZ),
        [-1, max_steps, latent_size])

    def symbols_to_logits_fn(ids_BKxI, i, unused_cache):
      """Generate logits for next potential IDs.

      Args:
        ids_BKxI: current decoded sequences, int tensor of shape
          [B x beam_size, i + 1]
        i: looping index
        unused_cache: a dictionary of tensors used in decoding step

      Returns:
        Next token logits of shape [B x beam_size, vocab_size] and the updated
        cache.
      """
      # TODO(peterjliu): Make use of cache for faster decoding as is done here:
      # tensorflow_models/official/transformer/model/transformer.py
      # I = i + 1 denotes the current sequence length, K denotes the beam size.
      input_embs_BKxIxE = tf.nn.embedding_lookup(self.embed_VxE, ids_BKxI)

      cond_input_BKxIxZ = cond_input_BKxMxZ[:, :i + 1, :]
      if self.cond_by_addition:
        input_enc_BKxIxD = input_embs_BKxIxE + cond_input_BKxIxZ
      else:
        input_enc_BKxIxD = tf.concat([input_embs_BKxIxE, cond_input_BKxIxZ],
                                     axis=2)

      input_enc_BKxIxH = self._input_proj(input_enc_BKxIxD)
      outputs_BKxIxH = self.decoder(inputs_BxTxH=input_enc_BKxIxH,
                                    training=False)
      logits_BKxV = self._output_proj(outputs_BKxIxH[:, i, :])
      return logits_BKxV, unused_cache

    return symbols_to_logits_fn

  def decode_v(self, cond_input_BxZ, method='argmax', first_token=-1,
               max_steps=0, beam_size=2, alpha=1.0):
    max_steps = max_steps if max_steps > 0 else self.max_steps
    with tf.variable_scope(self.scope):
      batch_size = cond_input_BxZ.get_shape()[0]
      cond_input_BxMxZ = tf.tile(tf.expand_dims(cond_input_BxZ, axis=1),
                                 [1, max_steps, 1])

      if method == 'beam':
        # TODO(andyyyuan): Remove this constraint. Same for GruDecoder.
        assert first_token >= 0, ('Need token IDs as input, cannot create '
                                  'all-zero vectors.')
        decoded_ids_BxKxM1, _ = beam_search.sequence_beam_search(
            symbols_to_logits_fn=self._get_symbols_to_logits_fn(
                cond_input_BxMxZ, beam_size),
            initial_ids=tf.ones([batch_size], dtype=tf.int32) * first_token,
            # beam_search function takes IDs of type tf.int32.
            initial_cache=dict(),
            vocab_size=self.vocab_size,
            beam_size=beam_size,
            alpha=alpha,
            max_decode_length=max_steps,
            eos_id=util.EOS_ID)
        # Return the top scored sequence for each batch element. Need to cast
        # back to tf.int64
        return tf.cast(decoded_ids_BxKxM1[:, 0, 1:], tf.int64)

      # Tensor to hold decoded symbols, start withempty place-holder.
      dec_symb_BxM = tf.zeros([batch_size, max_steps], dtype=tf.int64,
                              name='dec_symb_array')
      # dec_symb_BxM will also be used as next step input. To do so, we prepend
      # the first token for decoding below.
      first_token = util.PAD_ID if first_token < 0 else first_token
      # TODO(peterjliu,jjren): There's a mismatch of first_token when tie_dics
      # is False. shift_right_3d prepends all-zero embedding while PAD_ID does
      # not necessarily lead to all-zero embedding.
      dec_symb_BxM1 = tf.pad(dec_symb_BxM, [[0, 0], [1, 0]],
                             name='dec_symb_array_as_input',
                             constant_values=first_token)

      i = tf.constant(0, dtype=tf.int32)
      _, dec_symb_BxM1, _ = tf.while_loop(
          cond=self._not_finished,
          body=self._get_decode_v_body(cond_input_BxMxZ),  # loop body
          loop_vars=[
              i,
              dec_symb_BxM1,
              tf.zeros([batch_size], dtype=tf.bool),  # done_B
          ],
          maximum_iterations=max_steps)

      return dec_symb_BxM1[:, 1:]  # BxM shape, remove prepended first_token

  def _get_decode_v_body(self, cond_input_BxMxZ):
    """Returns body for decode_v while loop."""

    def _decode_v_body(i, dec_symb_BxM1, done_B):
      """The decode_v body for tf.while."""
      input_enc_BxMxE = tf.nn.embedding_lookup(self.embed_VxE,
                                               dec_symb_BxM1[:, :-1])
      if self.cond_by_addition:  # Z == E
        input_enc_BxMxD = input_enc_BxMxE + cond_input_BxMxZ
      else:
        input_enc_BxMxD = tf.concat([input_enc_BxMxE, cond_input_BxMxZ], axis=2)
      input_enc_BxMxH = self._input_proj(input_enc_BxMxD)

      outputs_BxMxH = self.decoder(inputs_BxTxH=input_enc_BxMxH, training=False)

      logits_BxV = self._output_proj(outputs_BxMxH[:, i, :])
      next_token_B = tf.argmax(logits_BxV, axis=1)

      is_eos_B = tf.equal(next_token_B, util.EOS_ID)
      done_B = tf.logical_or(done_B, is_eos_B)

      # Write next token to decoded tensor.
      # TODO(peterjliu): Replace alias_inplace_update with
      # tf.tensor_scatter_nd_update.
      dec_symb_BxM1 = tf.transpose(
          inplace_ops.alias_inplace_update(
              tf.transpose(dec_symb_BxM1), i + 1, next_token_B))
      return [i + 1, dec_symb_BxM1, done_B]

    return _decode_v_body

  def _not_finished(self, unused_i, unused_dec_symb, done_B):
    return tf.logical_not(tf.reduce_all(done_B))


class GruDecoder(Decoder):
  """An GRU decoder for decoding fixed-length vectors from an encoder.

  If cond_only_init=True, the conditioning vector is used as the initial state
  for the RNN decoder.
  This case assumes the hidden size is the same as the conditioning vector.

  If cond_only_init=False, the conditioning vector is concatenated to the
  input of RNN at each step and initial state is set to 0.
  """

  def __init__(self,
               hidden_size,
               vocab_size,
               embed_VxE,
               max_steps,
               num_layers=1,
               tie_embeddings=False,
               scope=None,
               cond_only_init=True):
    self.scope = 'gru_decoder' if scope is None else scope
    self.embed_VxE = embed_VxE
    self.max_steps = max_steps  # default max_steps
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.vocab_size = vocab_size
    self.emb_size = embed_VxE.get_shape()[1]
    self.tie_embeddings = tie_embeddings
    # if conditioning tensor only for initial state of RNNs
    self.cond_only_init = cond_only_init

    with tf.variable_scope(self.scope):
      if not tie_embeddings:
        self.out_kernel_HxV = tf.get_variable(
            name='output_kernel_HxV',
            shape=[hidden_size, vocab_size],
            initializer=tf.truncated_normal_initializer(
                stddev=SMALL_WEIGHTS_SD))
        self.out_bias_V = tf.get_variable(
            name='output_bias_V',
            shape=[vocab_size],
            initializer=tf.zeros_initializer())
      elif hidden_size != self.emb_size:
        # if hidenn_size != emb_size, we do additional affine transformation
        # to obtain x_BxE = x_BxH * x_HxE
        self.out_kernel_HxE = tf.get_variable(
            name='output_kernel_HxE',
            shape=[hidden_size, self.emb_size],
            initializer=tf.truncated_normal_initializer(
                stddev=SMALL_WEIGHTS_SD))
        self.out_bias_E = tf.get_variable(
            name='output_bias_E',
            shape=[self.emb_size],
            initializer=tf.zeros_initializer())
      self.cell = tf.nn.rnn_cell.MultiRNNCell(
          [tf.nn.rnn_cell.GRUCell(hidden_size) for _ in range(num_layers)])

  def _output_proj(self, x_BxH):
    """project output prediction to logits of vocab."""
    if self.tie_embeddings:
      if self.hidden_size != self.emb_size:
        x_BxE = tf.nn.bias_add(
            tf.matmul(x_BxH, self.out_kernel_HxE), self.out_bias_E)
      else:  # E == H
        x_BxE = x_BxH
      return tf.matmul(x_BxE, self.embed_VxE, transpose_b=True)
    else:
      return tf.nn.bias_add(
          tf.matmul(x_BxH, self.out_kernel_HxV), self.out_bias_V)

  def hidden_dim(self):
    return self.hidden_size

  # TODO(peterjliu): Reverse sequence tf.reverse_sequence

  def teacher_force(self, cond_input_BxZ, dec_inputs_BxSxE, seq_lengths_B):
    with tf.variable_scope(self.scope):
      if self.cond_only_init:
        assert cond_input_BxZ.get_shape()[1] == self.hidden_size
        init_state_BxH = cond_input_BxZ  # Z = H
        dec_inputs_BxSxI = dec_inputs_BxSxE
      else:
        init_state_BxH = tf.zeros(
            [tf.shape(cond_input_BxZ)[0], self.hidden_size])
        cond_input_BxSxZ = tf.tile(
            tf.expand_dims(cond_input_BxZ, axis=1),
            [1, tf.shape(dec_inputs_BxSxE)[1], 1])
        # I = E + Z
        dec_inputs_BxSxI = tf.concat((dec_inputs_BxSxE, cond_input_BxSxZ),
                                     axis=2)
      # For MultiRNNCell, the accepted and returned states are N-tuples, where N
      # in our case is the number of layers. Here we use the same initial state
      # (i.e., either cond_input_BxZ or zeros_BxZ) for all layers. Same for the
      # decode_v function below.
      dec_outputs_BxSxH, _ = tf.nn.dynamic_rnn(
          self.cell,
          dec_inputs_BxSxI,
          initial_state=(init_state_BxH,) * self.num_layers,
          # initial_state is a self.num_layers-tuple
          sequence_length=seq_lengths_B)
      proj_input_BSxH = tf.reshape(dec_outputs_BxSxH, [-1, self.hidden_size])
      logits_BxSxV = tf.reshape(
          self._output_proj(proj_input_BSxH),
          [tf.shape(dec_outputs_BxSxH)[0], -1, self.vocab_size])
      return logits_BxSxV

  def _get_first_input(self, batch_size, first_token):
    """Get first input embedding for a batch.

    Args:
      batch_size: batch size, int
      first_token: if >=0 token to use as first input, otherwise use all zeros

    Returns:
      Tensor with shape BxE, of first embedding input.
    """
    if first_token >= 0:
      first_input_1xE = tf.nn.embedding_lookup(self.embed_VxE, [first_token])
    else:
      # Default is zero-vector.
      first_input_1xE = tf.zeros((1, tf.shape(self.embed_VxE)[1]),
                                 dtype=tf.float32)
    # shape BxE
    return tf.tile(first_input_1xE, [batch_size, 1])

  def _get_symbols_to_logits_fn(self, cond_input_BxZ, beam_size):
    """Returns a decoding function that calculates logits of the next tokens."""
    latent_size = tf.shape(cond_input_BxZ)[1]
    cond_input_BxKxZ = tf.tile(tf.expand_dims(cond_input_BxZ, 1),
                               [1, beam_size, 1])
    cond_input_BKxZ = tf.reshape(cond_input_BxKxZ, [-1, latent_size])

    def symbols_to_logits_fn(ids_BKxI, unused_i, cache):
      """Generate logits for next potential IDs.

      Args:
        ids_BKxI: current decoded sequences, int tensor of shape
          [B x beam_size, i + 1]
        unused_i: looping index
        cache: a dictionary of tensors used in decoding step

      Returns:
        Next token logits of shape [B x beam_size, vocab_size] and the updated
        cache.
      """
      # beam_size as K, I = i + 1
      # Get the latest generated token IDs.
      input_tokens_BK = ids_BKxI[:, -1]
      input_embs_BKxE = tf.nn.embedding_lookup(self.embed_VxE, input_tokens_BK)

      if self.cond_only_init:
        assert cond_input_BxZ.get_shape()[1] == self.hidden_size
        input_BKxI = input_embs_BKxE  # I = E
      else:
        input_BKxI = tf.concat([input_embs_BKxE, cond_input_BKxZ],
                               axis=1)  # I = E + Z

      # Get the current hidden state from cache.
      state_BKxH = cache.get('state')
      o_BKxH, s_BKxH = self.cell(input_BKxI, state_BKxH)
      logits_BKxV = self._output_proj(o_BKxH)
      # Update cache with the latest hidden state.
      cache.update({'state': s_BKxH})
      return logits_BKxV, cache

    return symbols_to_logits_fn

  def decode_v(self, cond_input_BxZ, method='argmax', first_token=-1,
               max_steps=0, beam_size=2, alpha=1.0):
    max_steps = max_steps if max_steps > 0 else self.max_steps
    with tf.variable_scope(self.scope):
      batch_size = cond_input_BxZ.get_shape()[0]
      init_input_BxE = self._get_first_input(batch_size, first_token)
      if self.cond_only_init:
        assert cond_input_BxZ.get_shape()[1] == self.hidden_size
        cond_input_BxZ = tf.ensure_shape(cond_input_BxZ,
                                         (None, self.hidden_size))
        init_state_BxH = cond_input_BxZ  # H = Z
      else:
        init_state_BxH = tf.zeros([batch_size, self.hidden_size])

      if method == 'beam':
        assert first_token >= 0, ('Need token IDs as input, cannot create '
                                  'all-zero vectors.')
        initial_cache = {'state': (init_state_BxH,) * self.num_layers}
        decoded_ids_BxKxM1, _ = beam_search.sequence_beam_search(
            symbols_to_logits_fn=self._get_symbols_to_logits_fn(cond_input_BxZ,
                                                                beam_size),
            initial_ids=tf.ones([batch_size], dtype=tf.int32) * first_token,
            # beam_search function takes IDs of type tf.int32.
            initial_cache=initial_cache,
            vocab_size=self.vocab_size,
            beam_size=beam_size,
            alpha=alpha,
            max_decode_length=max_steps,
            eos_id=util.EOS_ID)
        # Return the top scored sequence for each batch element. Need to cast
        # back to tf.int64.
        return tf.cast(decoded_ids_BxKxM1[:, 0, 1:], tf.int64)

      dec_symb_MxB = tf.TensorArray(
          dtype=tf.int64,
          size=max_steps,
          dynamic_size=False,
          clear_after_read=False,
          element_shape=batch_size,
          name='dec_symb_array')

      i = tf.constant(0, dtype=tf.int32)
      _, _, _, dec_symb_MxB, _ = tf.while_loop(
          self._not_finished,
          self._get_decode_v_body(cond_input_BxZ, method),  # body
          loop_vars=[
              i,
              (init_state_BxH,) * self.num_layers,  # initial_state
              init_input_BxE,  # first input
              # BxM tensor to hold decoded symbols,
              # start withempty place-holder
              dec_symb_MxB,
              tf.zeros((batch_size), dtype=tf.bool),  # done_B
          ],
          maximum_iterations=max_steps)
      # Remove leading 0 from step 0
      dec_symb_tmp_MxB = dec_symb_MxB.stack()
      dec_symb_BxM = tf.transpose(dec_symb_tmp_MxB, perm=[1, 0])
      return dec_symb_BxM

  def _get_decode_v_body(self, cond_input_BxZ, method):
    """Returns body for decode_v while loop.

    Args:
      cond_input_BxZ: conditioning tensor, shape=(batch, hidden_size)
      method: one of 'argmax', 'random'
    """

    def _decode_v_body(i, state_BxH, input_BxE, dec_symb_MxB, done_B):
      """The decode_v body for tf.while."""

      if self.cond_only_init:
        assert cond_input_BxZ.get_shape()[1] == self.hidden_size
        input_BxI = input_BxE  # I = E
      else:
        input_BxI = tf.concat((input_BxE, cond_input_BxZ), axis=1)  # I = E + Z

      o_BxH, s_BxH = self.cell(input_BxI, state_BxH)
      logits_BxV = self._output_proj(o_BxH)
      # Embed next token and use as next input.
      # Greedy sampling replace with other options.
      if method in ('argmax', 'random'):
        if method == 'argmax':
          next_token_B = tf.argmax(logits_BxV, axis=1)
        elif method == 'random':
          next_token_B = tf.squeeze(tf.random.multinomial(logits_BxV, 1))
        # TODO(peterjliu): Replace with zero if done; nicer but not necessary.
        is_eos_B = tf.equal(next_token_B, util.EOS_ID)
        done_B = tf.logical_or(done_B, is_eos_B)
        # Next token embedding
        next_in_BxE = tf.nn.embedding_lookup(self.embed_VxE, next_token_B)
      else:
        tf.logging.fatal('Method not implemented: %s', method)

      # Write symbols and embeddings into tensorArrays
      dec_symb_MxB = dec_symb_MxB.write(i, next_token_B)
      return [i + 1, s_BxH, next_in_BxE, dec_symb_MxB, done_B]

    return _decode_v_body

  def _not_finished(self, unused_i, unused_state, unused_input, unused_dec_symb,
                    done_B):  # loop variables
    """Condition for tf.while for decode_v."""
    return tf.logical_not(tf.reduce_all(done_B))

  def decode_v_gumbel(self, cond_input_BxZ, first_token=-1, max_steps=0,
                      temperature=2.0):
    """Similar to decode_v, but using gumbel-softmax.

    Args:
      cond_input_BxZ: conditioning tensor, shape=(batch, hidden_size)
      first_token: token to be used for initial input
      max_steps: override default max_steps
      temperature: gumbel-softmax temperature

    Returns:
      dec_symb_BxM: decoded symbol ids
      dec_emb_BxMxE: embeddings corresponding to dec_symb_BxM
          if with_embeddings, otherwise None
    """
    max_steps = max_steps if max_steps > 0 else self.max_steps
    with tf.variable_scope(self.scope):
      # Differentiable decoding, also returns embeddings.
      batch_size = cond_input_BxZ.get_shape()[0]
      # Decoded embeddings corresponding to decoded symbols.
      # Initially, Bx1xE, to which we append M embeddings.
      # M = length of decoded sequence.
      init_input_BxE = self._get_first_input(batch_size, first_token)

      if self.cond_only_init:
        assert cond_input_BxZ.get_shape()[1] == self.hidden_size
        init_state_BxH = cond_input_BxZ  # H = Z
      else:
        init_state_BxH = tf.zeros([batch_size, self.hidden_size])

      dec_emb_MxBxE = tf.TensorArray(
          dtype=tf.float32,
          size=max_steps,  # 0th position is for init_input
          dynamic_size=False,
          clear_after_read=False,
          element_shape=(batch_size, self.emb_size),
          name='dec_emb_gumbel_array')

      dec_symb_MxB = tf.TensorArray(
          dtype=tf.int64,
          size=max_steps,
          dynamic_size=False,
          clear_after_read=False,
          element_shape=batch_size,
          name='dec_symb_gumbel_array')

      i = tf.constant(0, dtype=tf.int32)
      _, _, _, dec_emb_MxBxE, dec_symb_MxB, _ = tf.while_loop(
          self._not_finished_gumbel,  # while condition
          self._get_decode_v_gumbel_body(cond_input_BxZ,
                                         temperature),  # body
          loop_vars=[
              i,
              (init_state_BxH,) * self.num_layers,  # initial_state
              init_input_BxE,  # initial_input
              dec_emb_MxBxE,  # decoded embeddings
              # BxM tensor to hold decoded symbols,
              # start withempty place-holder
              dec_symb_MxB,
              # done_B, says whether EOS encountered in each sequence
              tf.zeros((batch_size), dtype=tf.bool),
          ],
          maximum_iterations=max_steps,
      )

      # convert tensorArray to tensor
      dec_emb_temp_MxBxE = dec_emb_MxBxE.stack()
      dec_emb_BxMxE = tf.transpose(dec_emb_temp_MxBxE, perm=[1, 0, 2])

      dec_symb_temp_MxB = dec_symb_MxB.stack()
      dec_symb_BxM = tf.transpose(dec_symb_temp_MxB, perm=[1, 0])
      return dec_symb_BxM, dec_emb_BxMxE

  def _get_decode_v_gumbel_body(self, cond_input_BxZ, temperature):
    """Returns body for decode_v_gumbel while loop.

    Args:
      cond_input_BxZ: conditioning tensor, shape=(batch, hidden_size)
      temperature: GS temperature
    """

    def _decode_v_gumbel_body(i, state_BxH, input_BxE, dec_emb_MxBxE,
                              dec_symb_MxB, done_B):
      """Body for decode_v_gumbel tf.while."""
      if self.cond_only_init:
        assert cond_input_BxZ.get_shape()[1] == self.hidden_size
        input_BxI = input_BxE  # I = E
      else:
        input_BxI = tf.concat((input_BxE, cond_input_BxZ), axis=1)
      o_BxH, s_BxH = self.cell(input_BxI, state_BxH)
      logits_BxV = self._output_proj(o_BxH)
      # TODO(peterjliu): hparam, should be annealed during training
      # Sample next token (embedding) and use as next input.
      gumbel_samples_BxV = sample.gumbel_softmax(
          logits_BxV, temperature, hard=True)
      gumbel_samples_BxV = tf.identity(
          gumbel_samples_BxV, name='gumbel_samples')
      next_in_BxE = tf.matmul(
          tf.cast(gumbel_samples_BxV, dtype=tf.float32),
          self.embed_VxE)  # differentiable
      next_token_B = tf.argmax(gumbel_samples_BxV, 1)  # Not differentiable

      is_eos_B = tf.equal(next_token_B, util.EOS_ID)
      done_B = tf.logical_or(done_B, is_eos_B)

      # Write symbols and embeddings into tensorArrays
      dec_emb_MxBxE = dec_emb_MxBxE.write(i, next_in_BxE)
      dec_symb_MxB = dec_symb_MxB.write(i, next_token_B)
      return [i + 1, s_BxH, next_in_BxE, dec_emb_MxBxE, dec_symb_MxB, done_B]

    return _decode_v_gumbel_body

  def _not_finished_gumbel(self, unused_i, unused_state, unused_input,
                           unused_dec_emb, unused_dec_symb,
                           done_B):  # loop variables
    """Condition for tf.while for decode_v."""
    return tf.logical_not(tf.reduce_all(done_B))


def autoencode(ids_BxS,
               seq_lengths_B,
               encoder,
               decoder,
               embedding_VxE,
               mask_rate=0.0,
               mask_id=None,
               mask_decode=False,
               enc_drop_rate=0.0,
               eval_keep_rate=1.0,
               pooling='mean',
               is_training=None):
  """Returns losses and encoding from autoencoder formed from encoder-decoder.

  It is trained using teacher-forcing.

  Args:
    ids_BxS: tensor of ids with shape (batch_size, sequence length)
    seq_lengths_B: sequence lengths for ids
    encoder: Encoder that encodes to vector with dim H
    decoder: Decoder that decode vectors of dim H
    embedding_VxE: (vocab, embedding dim) embedding tensor
    mask_rate: probability for masking
    mask_id: vocab id for mask token "<MASK>"
    mask_decode: if True, use masked input for decoder, otherwise use original
    enc_drop_rate: if >0, randomly dropout elements in encoded vector
    eval_keep_rate: proportion of tokens are used for ce_loss evalaution.
    pooling: pooling method for generating a single vector from encoder output
             features representing the input sequence
    is_training: a bool indicating whether in training mode; useful to
                 determine whether to turn on layers like dropout

  Returns:
    loss: reconstruction loss scalar tensor
    kl_loss: for variational models, a scalar for kl loss
    metrics: dict of sequence metrics
    encoding: vector encoding, BxH tensor
  """
  enc_inputs_BxSxE = tf.nn.embedding_lookup(embedding_VxE, ids_BxS)
  dec_inputs_BxSxE = shift_right_3d(enc_inputs_BxSxE)
  enc_BxZ, kl_loss_f = encoder.encode(
      enc_inputs_BxSxE, seq_lengths_B, pooling=pooling, is_training=is_training)

  if is_training:
    if mask_rate > 0.0:
      # randomly mask tokens in the input
      assert mask_id  # mask_id should not be None
      ids_mask_BxS = mask_ids(ids_BxS, seq_lengths_B, mask_rate, mask_id)
      enc_inputs_BxSxE = tf.nn.embedding_lookup(embedding_VxE, ids_mask_BxS)
      enc_BxZ, kl_loss_f = encoder.encode(
          enc_inputs_BxSxE,
          seq_lengths_B,
          pooling=pooling,
          is_training=is_training)

      if mask_decode:
        # decode with masked input
        dec_inputs_BxSxE = shift_right_3d(enc_inputs_BxSxE)

    if enc_drop_rate > 0.0:
      # add dropout to enc
      enc_BxZ = tf.nn.dropout(enc_BxZ, rate=enc_drop_rate)

  logits_BxSxV = decoder.teacher_force(enc_BxZ, dec_inputs_BxSxE, seq_lengths_B)
  reconstruction_loss_f, metrics = ce_loss(
      logits_BxSxV, ids_BxS, seq_lengths_B, eval_keep_rate=eval_keep_rate)

  return reconstruction_loss_f, kl_loss_f, metrics, enc_BxZ


def mask_random_pos(ids_BxS, keep_rate):
  """Randomly mask positions to avoid evaluation."""
  unifs = tf.random.uniform(
      tf.shape(ids_BxS), minval=0, maxval=1, dtype=tf.float32)
  mask = tf.cast(unifs < keep_rate, dtype=tf.float32)
  return mask


def ce_loss(logits_BxSxV, ids_BxS, seq_lengths_B, eval_keep_rate=1.0):
  """Returns cross-entropy sequence loss and auxillary metrics.

  Args:
    logits_BxSxV: logits corresponding to ids
    ids_BxS: token ids, shape (batch, sequence-length)
    seq_lengths_B: sequence lengths for ids_BxS
    eval_keep_rate: proportion of tokens are kept for evaluation

  Returns:
    loss: cross-entropy loss, a float tensor
    metrics: dict of string to tf.metrics
  """
  vocab_size = tf.shape(logits_BxSxV)[2]
  with tf.variable_scope('cross_entropy_loss'):
    weights_BxS = tf.sequence_mask(seq_lengths_B,
                                   maxlen=tf.shape(ids_BxS)[1],
                                   dtype=tf.float32)
    if eval_keep_rate < 1.0:
      mask_pos_BxS = mask_random_pos(ids_BxS, eval_keep_rate)
      weights_BxS = tf.multiply(weights_BxS, mask_pos_BxS)

    # TODO(peterjliu): Add label smoothing "smoothing_cross_entropy"
    logits_flat_BSxV = tf.reshape(logits_BxSxV, [-1, vocab_size])
    targets_flat_BS = tf.reshape(ids_BxS, [-1])
    ce_BS = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=targets_flat_BS, logits=logits_flat_BSxV)
    weights_flat_BS = tf.reshape(weights_BxS, [-1], name='flattened_ce_weights')
    ce_weighted = tf.multiply(ce_BS, weights_flat_BS)

    # Average across timesteps and across batch
    accuracy_weighted = tf.metrics.accuracy(
        labels=targets_flat_BS, predictions=tf.argmax(logits_flat_BSxV, axis=1),
        weights=weights_flat_BS)
    metrics = {'accuracy': accuracy_weighted}
    return tf.identity(
        tf.reduce_sum(ce_weighted) / tf.reduce_sum(weights_flat_BS),
        name='reconstruction_loss'), metrics


def sample_latent(s_BxH, latent_dim):
  """Samples latent vectors from a hidden state (for variational models).

  Args:
    s_BxH: batch of state vectors
    latent_dim: int, dimensionality of latent z

  Returns:
    z_BxL: latent z tensor
    kl_loss: KL loss for VAE
  """
  [batch_size, hdim] = s_BxH.get_shape().as_list()

  # Using notation in https://arxiv.org/pdf/1704.03477.pdf,
  # project h vector to (mu, sigmah) vectors both of dimension L=latent_dim
  # B = batch_size
  # H = hdim
  # L = latent_dim, N_z

  # Eq (2)
  with tf.variable_scope('variational'):
    Wmu_HxL = tf.get_variable(name='W_mu', shape=(hdim, latent_dim),
                              initializer=tf.truncated_normal_initializer(
                                  stddev=SMALL_WEIGHTS_SD))
    bmu_L = tf.get_variable(name='b_mu', shape=(latent_dim),
                            initializer=tf.zeros_initializer())
    Wsigma_HxL = tf.get_variable(name='W_sigma', shape=(hdim, latent_dim),
                                 initializer=tf.truncated_normal_initializer(
                                     stddev=SMALL_WEIGHTS_SD))
    bsigma_L = tf.get_variable(name='b_sigma', shape=(latent_dim),
                               initializer=tf.zeros_initializer())
    mu_BxL = tf.nn.bias_add(
        tf.matmul(s_BxH, Wmu_HxL),
        bmu_L, name='mu')
    sigmah_BxL = tf.nn.bias_add(
        tf.matmul(s_BxH, Wsigma_HxL),
        bsigma_L, name='sigmah')

    exp_sigmah_BxL = tf.exp(sigmah_BxL)
    # exp(x/2) = sqrt(exp(x)), avoids doing two exp operations
    sigma_BxL = tf.sqrt(exp_sigmah_BxL, name='sigma')
    z_BxL = tf.add(mu_BxL,
                   tf.multiply(sigma_BxL,
                               tf.random.normal((batch_size, latent_dim))),
                   name='z')

    # Eq (10)
    kl_loss_B = -0.5/latent_dim * (1 + sigmah_BxL -
                                   tf.pow(mu_BxL, 2) - exp_sigmah_BxL)
    return z_BxL, tf.reduce_mean(kl_loss_B, name='kl_loss')


def minimize(loss, optimizer, global_step, clip_norm):
  tvars = tf.trainable_variables()
  grads = tf.gradients(loss, tvars)

  if clip_norm > 0:
    (grads, _) = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
  train_op = optimizer.apply_gradients(
      list(zip(grads, tvars)), global_step=global_step)
  return train_op


def get_adam(learning_rate, use_tpu=False):
  opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
  if use_tpu:
    opt = tf.tpu.CrossShardOptimizer(opt)
  return opt


def minimize_with_adam(loss, learning_rate, global_step,
                       var_list=None, use_tpu=False):
  opt = get_adam(learning_rate, use_tpu)
  return opt.minimize(loss, global_step=global_step, var_list=var_list)


def swap_neighboring_rows_5(x_5xL):
  p = tf.constant(
      [[0, 1, 2, 3, 4],
       [1, 0, 2, 3, 4],
       [0, 2, 1, 3, 4],
       [0, 1, 3, 2, 4],
       [0, 1, 2, 4, 3]])
  rand_row = tf.random.uniform([], minval=0, maxval=5, dtype=tf.int32)
  return tf.gather(x_5xL, p[rand_row, :])


def randomly_sample_rows(x_RxC, n):
  perm = tf.random.shuffle(tf.range(0, tf.shape(x_RxC)[0]))
  return tf.gather(x_RxC, perm[0:n])


def get_features_labels(pos_BxZ, neg_BxZ):
  """Generate features and labels for adversarial training."""
  batch_size = tf.shape(pos_BxZ)[0]
  features_2BxZ = tf.concat([pos_BxZ, neg_BxZ], axis=0)
  labels_2B = tf.concat([tf.ones([batch_size]), tf.zeros([batch_size])], axis=0)
  return features_2BxZ, labels_2B


def get_discriminator(n_hiddens, scope):
  def discriminator(x_BxZ):
    """Train a basic discriminator."""
    with tf.variable_scope(scope):
      tf.logging.info('Add discriminator (MLP) with %d hiddens.', n_hiddens)
      h_BxH = tf.layers.dense(x_BxZ, n_hiddens, activation=tf.nn.relu)
      logits_B = tf.squeeze(tf.layers.dense(h_BxH, 1, activation=None))
      return logits_B
  return discriminator


def dense_layer_with_proj(x_BxSxI, proj_HxO, scope=None, name=''):
  """MLP (provided with a projection matrix) that outputs logits."""
  input_size = x_BxSxI.get_shape()[-1]
  hidden_size = proj_HxO.get_shape()[0]
  output_size = proj_HxO.get_shape()[1]
  with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
    x_BSxI = tf.reshape(x_BxSxI, [-1, input_size])
    x_BSxH = tf.layers.dense(  # with bias by default
        inputs=x_BSxI,
        units=hidden_size,
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(
            stddev=SMALL_WEIGHTS_SD),
        name=name + '_kernel')
    x_BSxH = contrib_layers.layer_norm(
        x_BSxH, begin_norm_axis=-1, begin_params_axis=-1)
    logits_bias_O = tf.get_variable(
        name=name + '_output_bias',
        shape=[output_size],
        initializer=tf.zeros_initializer())
  logits_BxSxO = tf.reshape(
      tf.nn.bias_add(
          tf.matmul(x_BSxH, proj_HxO),
          logits_bias_O),
      [tf.shape(x_BxSxI)[0], -1, output_size])
  return logits_BxSxO
