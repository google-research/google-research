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

"""Transformations for pairwise sequence alignment."""

import collections
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

from absl import logging

import gin
import numpy as np
import tensorflow as tf

from dedal import pairs
from dedal import vocabulary
from dedal.data import transforms


Example = Dict[str, tf.Tensor]
InMemoryDataset = Dict[str, Union[tf.Tensor, tf.RaggedTensor]]


@gin.configurable
class RandomPairing(transforms.DatasetTransform):
  """Samples two examples totally at random."""

  def __init__(self,
               index_keys,
               suffixes = ('_1', '_2')):
    del index_keys, suffixes
    super().__init__()

  def call(self, ds):

    def rename(x, y):
      """Rename the keys of the two tensors."""
      length = tf.minimum(
          tf.shape(x['sequence'])[0], tf.shape(y['sequence'])[0])
      x['sequence'] = x['sequence'][:length]
      y['sequence'] = y['sequence'][:length]

      def tag(x, i):
        return {f'{k}_{i}': v for k, v in x.items()}

      return {**tag(x, 1), **tag(y, 2)}

    return tf.data.Dataset.zip((ds, ds.shuffle(1024))).map(rename)


def dataset_to_memory(ds):
  """Stores a dataset in memory, stacking components as Tensor/RaggedTensor."""
  examples = collections.defaultdict(list)
  n_examples = 0
  for ex in ds.prefetch(tf.data.AUTOTUNE):
    for k, v in ex.items():
      examples[k].append(v)
    n_examples += 1
  examples = {k: tf.ragged.stack(v) for k, v in examples.items()}
  return examples, n_examples


def smooth_logit(w, a = 1.0):
  """Transforms weights into (smoothed) logits."""
  w = tf.cast(w, tf.float64)
  return tf.where(w > 0.0, a * tf.math.log(w), -float('inf'))


@gin.configurable
class StratifiedSamplingPairing(transforms.DatasetTransform):
  """Dataset-level transform to iterate over pairs of examples."""

  def __init__(self,
               index_keys,
               branch_key,
               smoothing = None,
               suffixes = ('_1', '_2')):
    self._index_keys = index_keys
    self._branch_key = branch_key
    self._smoothing = ((1.0,) * len(index_keys) if smoothing is None
                       else smoothing)
    self._suffixes = suffixes

    self._n_levels = len(index_keys)
    self._branch_idx = index_keys.index(branch_key)

  def _build_sampling_structs(
      self,
      keys,
  ):
    """Precomputes key index and sampling logits."""
    # Hashes key tuples for each example, treating nesting levels as "digits".
    # This hierarchically clusters examples according to the (int) value of each
    # key in index_keys, in order.
    keys = tf.cast(keys, tf.int64)  # Prevents overflow when ci_x keys are uniq.
    key_ranges = tf.reduce_max(keys, 0) + 1
    key_hash_multipliers = tf.cast(
        tf.reduce_prod(key_ranges) / tf.math.cumprod(key_ranges), tf.int64)
    hashes = tf.reduce_sum(tf.cast(keys, tf.int64) * key_hash_multipliers, -1)

    # At this point, the order of examples is random, depending on the loader's
    # PRNGs. This sorts examples by hash, making examples in same cluster
    # consecutive.
    index = tf.argsort(hashes)
    hashes = tf.gather(hashes, index)

    # Builds a cluster index, structured as nested lists with len(index_keys)
    # levels of nesting.
    for i in range(self._n_levels - 1, -1, -1):
      hashes //= key_hash_multipliers[i]
      hashes, _, counts = tf.unique_with_counts(hashes)
      hashes *= key_hash_multipliers[i]
      index = tf.RaggedTensor.from_row_lengths(index, counts)

    # Computes cluster weights at all levels in the hierarchy. The weight of a
    # cluster is given by the number of distinct elements it can produce during
    # sampling. For clusters at or after the branching point, this is simply the
    # total number of examples in the cluster. For clusters before the branching
    # this is the number of pairs of elements they contain belonging to
    # different clusters at the branching level.
    weights = {self._index_keys[-1]: index.row_lengths(axis=self._n_levels)}
    for i in range(self._n_levels - 2, -1, -1):
      w = weights[self._index_keys[i + 1]]
      weights[self._index_keys[i]] = (
          tf.reduce_sum(w, -1) if i != (self._branch_idx - 1)
          else (tf.reduce_sum(w, -1) ** 2 - tf.reduce_sum(w ** 2, -1)) // 2)
      weights[self._index_keys[i + 1]] = tf.ragged.map_flat_values(
          smooth_logit, w, self._smoothing[i + 1])
    weights[self._index_keys[0]] = tf.ragged.map_flat_values(
        smooth_logit, weights[self._index_keys[0]], self._smoothing[0])

    return index, weights

  def call(self, ds):
    """Returns a new tf.data.Dataset that yields pairs of examples.

    Example pairs will be randomly generated following the constraints
      ex1[index_keys[0]] == ex2[index_keys[0]],
      ex1[index_keys[1]] != ex2[index_keys[1]].

    The probability of drawing a pair of examples with ex[index_keys[0]] == v is
    proportional to w_v ^ smoothings[0], where w_v is the number of example
    pairs such that
      ex1[index_keys[0]] == v, ex2[index_keys[0]] == v,
      ex1[index_keys[1]] != ex2[index_keys[1]].

    Conditional on ex[index_keys[0]] == v, the probability of sampling a pair of
    examples with ex1[index_keys[1]] == v1 and ex2[index_keys[1]] == v2 and
    v1 != v2 is proportional to (z_{v1} * z_{v2}) ^ smoothings[1], where z_y is
    the number of examples such that
      ex[index_keys[0]] == y,
      ex[index_keys[1]] == y.

    Note that setting smoothings[0] = smoothings[1] = 1 is equivalent to
    sampling example pairs uniformly at random from the set of example pairs
    satisfying the constraints. Likewise, if smoothings[0] = smoothings[1] = 0
    is equivalent to sampling key values uniformly at random at each step, thus
    inflating the probability of sampling pairs from rare keys.

    Args:
      ds: A tf.data.Dataset with elements assumed to be Dict[str, tf.Tensor].
        Each element must contain all `index_keys`, and these are assumed to be
        scalar int values.

    Returns:
      A tf.data.Dataset that yields elements formed by pairs of elements of the
      input dataset ds, sampled randomly as described above.
    """
    examples, n_examples = dataset_to_memory(ds)
    logging.info('PairExamples: %s examples cached.', n_examples)

    index, logits = self._build_sampling_structs(
        tf.stack([examples[k] for k in self._index_keys], 1))
    logging.info('PairExamples: built sampling data structures.')

    def random_stratified_pair(seed):
      n_seeds = self._branch_idx + 2 * (self._n_levels - self._branch_idx + 1)
      seeds = tf.random.experimental.stateless_split(seed, num=n_seeds)
      seed_idx = 0

      def random_category(logits, seed):
        """Samples a single categorical variable from logits."""
        logits = tf.reshape(logits, [1, -1])
        return tf.reshape(tf.random.stateless_categorical(logits, 1, seed), ())

      def random_exclusive_categories(
          logits, seeds):
        """Samples a pair of distinct categorical variables from logits."""
        logits = tf.reshape(logits, [1, -1])
        idx1 = random_category(logits, seeds[0])
        logits = tf.tensor_scatter_nd_update(
            logits, [[0, idx1]], [-float('inf')])
        idx2 = random_category(logits, seeds[1])
        return idx1, idx2

      def random_key(keys, seed):
        """Samples a key from keys uniformly at random."""
        maxval = tf.shape(keys)[0]
        return keys[tf.random.stateless_uniform(
            shape=(), seed=seed, maxval=maxval, dtype=tf.int32)]

      def lookup_examples(ind1, ind2):
        """Combines two dataset elements into a single Example."""
        ex = {}
        for k, v in examples.items():
          for ind, suffix in zip((ind1, ind2), self._suffixes):
            ex[f'{k}{suffix}'] = v[ind]
        return ex

      indices = []

      for i in range(self._branch_idx):
        key = self._index_keys[i]
        idx = random_category(logits[key][indices], seeds[seed_idx])
        indices.append(idx)
        seed_idx += 1

      key = self._index_keys[self._branch_idx]
      idx1, idx2 = random_exclusive_categories(
          logits[key][indices], seeds[seed_idx:seed_idx+2])
      indices1, indices2 = indices.copy(), indices.copy()
      indices1.append(idx1)
      indices2.append(idx2)
      seed_idx += 2

      for i in range(self._branch_idx + 1, self._n_levels):
        key = self._index_keys[i]
        idx1 = random_category(logits[key][indices1], seeds[seed_idx])
        idx2 = random_category(logits[key][indices2], seeds[seed_idx + 1])
        indices1.append(idx1)
        indices2.append(idx2)
        seed_idx += 2

      ind1 = random_key(index[indices1], seeds[seed_idx])
      ind2 = random_key(index[indices2], seeds[seed_idx + 1])
      return lookup_examples(ind1, ind2)

    ds = tf.data.experimental.RandomDataset().batch(2)
    ds = ds.map(random_stratified_pair, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


@gin.configurable
class ProjectMSARows(transforms.Transform):
  """Projects a pair of rows from a MSA, with gaps represented by token."""

  def __init__(self,
               token = '-',
               **kwargs):
    super().__init__(**kwargs)
    self._token = token

  def call(
      self,
      seq1,
      seq2,
      match_cols = None,
  ):
    masks = [self._vocab.compute_mask(seq, self._token) for seq in (seq1, seq2)]
    keep_indices = tf.reduce_any(masks, 0)
    if match_cols is not None:
      keep_indices = tf.logical_and(keep_indices, match_cols)
    keep_indices = tf.reshape(tf.where(keep_indices), [-1])
    return tf.gather(seq1, keep_indices), tf.gather(seq2, keep_indices)


@gin.configurable
class PID(transforms.Transform):
  """Computes Percent IDentity for two MSA rows, with gaps given by token.

  Attributes:
    definition: whether to use PID1 (1) or PID3 (3). The former defines PID as
      the number of identical matches divided by the alignment length whereas
      the latter divides by the length of the shorter sequence. Thus, PID1
      ignores stretches of sequence before (resp. after) the first (resp. last)
      match whereas PID3 does not.
    token: the character used to represent gaps in the `Vocabulary`.
  """

  def __init__(self, definition = 3, token = '-', **kwargs):
    super().__init__(**kwargs)
    self._definition = definition
    self._token = token

  def call(self, seq1, seq2):
    masks = [self._vocab.compute_mask(seq, self._token) for seq in (seq1, seq2)]

    keep_indices = tf.reshape(tf.where(tf.reduce_all(masks, 0)), [-1])
    seq1, seq2 = tf.gather(seq1, keep_indices), tf.gather(seq2, keep_indices)
    n_matches = tf.reduce_sum(tf.cast(seq1 == seq2, tf.int32))

    if self._definition == 1:
      den = tf.cast(keep_indices[-1] - keep_indices[0] + 1, tf.int32)
    elif self._definition == 3:
      den = tf.minimum(*[tf.reduce_sum(tf.cast(m, tf.int32)) for m in masks])
    else:
      raise ValueError(f'PID{self._definition} not yet supported.')

    return n_matches / den


@gin.configurable
class CreateAlignmentTargets(transforms.Transform):
  """Creates targets for pairwise sequence alignment task."""
  # Constants for (integer) encoding of alignment states.
  _GAP_IN_X = -1
  _MATCH = 0
  _GAP_IN_Y = 1
  _START = 2
  # Integer-encoding for special initial transition.
  _INIT_TRANS = 0

  def __init__(self,
               gap_token = '-',
               n_prepend_tokens = 0,
               **kwargs):
    super().__init__(**kwargs)
    self._gap_token = gap_token
    self._n_prepend_tokens = n_prepend_tokens

    # Transition look-up table (excluding special initial transition).
    look_up = {
        (self._MATCH, self._MATCH): 1,
        (self._GAP_IN_X, self._MATCH): 2,
        (self._GAP_IN_Y, self._MATCH): 3,
        (self._MATCH, self._GAP_IN_X): 4,
        (self._GAP_IN_X, self._GAP_IN_X): 5,
        (self._GAP_IN_Y, self._GAP_IN_X): 9,  # "forbidden" transition.
        (self._MATCH, self._GAP_IN_Y): 6,
        (self._GAP_IN_X, self._GAP_IN_Y): 7,
        (self._GAP_IN_Y, self._GAP_IN_Y): 8,
    }
    # Builds data structures for efficiently encoding transitions.
    self._hash_fn = lambda d0, d1: 3 * (d1 + 1) + (d0 + 1)
    hashes = [self._hash_fn(d0, d1) for (d0, d1) in look_up]
    self._trans_encoder = tf.scatter_nd(indices=[[x] for x in hashes],
                                        updates=list(look_up.values()),
                                        shape=[max(hashes) + 1])
    self._trans_encoder = tf.cast(self._trans_encoder, tf.int32)
    self._init_trans = tf.convert_to_tensor([self._INIT_TRANS], dtype=tf.int32)

  def call(self, seq1, seq2):
    """Creates targets for pairwise sequence alignment task from proj. MSA rows.

    Given a pair of projected rows from an MSA (i.e., with positions at which
    both rows have a gap removed), the ground-truth alignment targets are
    obtained by:
    1) Each position in the projected MSA is classified as _MATCH, _GAP_IN_X or
       _GAP_IN_Y.
    2) The positions of match states are retrieved, as well as the starting
       position of each sequence in the ground-truth (local) alignment.
    3) Positions before the first match state or after the last match state are
       discarded, as these do not belong to the local ground-truth alignment.
    4) For each pair of consecutive match states, where consecutive here is to
       be understood when ignoring non-match states, it is checked whether there
       are BOTH _GAP_IN_X and _GAP_IN_Y states in between.
    5) For each pair of consecutive match states with both _GAP_IN_X and
       _GAP_IN_Y states in between, these states are canonically sorted to
       ensure all _GAP_IN_X states occur first, being followed by all _GAP_IN_Y
       states.
    6) We encode transitions, that is, ordered tuples (s_old, s_new) of states
       using the 9 hidden state model described in `look_up` (c.f. `init`), with
       initial transition (_START, _MATCH) encoded as in `self._init_trans`.
    7) Given the new sequence of states, we reconstructed the positions in each
       sequence where those states would occur.
    8) Finally, optionally, if any special tokens are to be prepended to the
       sequences after this transformation, the ground-truth alignment targets
       will be adjusted accordingly. Note, however, that tokens being appended
       require no further modification.

    Args:
      seq1: A tf.Tensor<int>[len], representing the first proj. row of the MSA.
      seq2: A tf.Tensor<int>[len], representing the second proj. row of the MSA.

    Returns:
      A tf.Tensor<int>[3, tar_len] with three stacked tf.Tensor<int>[tar_len],
      pos1, pos2 and enc_trans, such that (pos1[i], pos2[i], enc_trans[i])
      represents the i-th transition in the ground-truth alignment. For example,
        (pos1[0], pos2[0], enc_trans[0]) = (1, 1, 3)
      would represent that the first transition in the ground-truth alignment is
      from the start state _START to the _MATCH(1,1) state whereas
        (pos1[2], pos2[2], enc_trans[2]) = (2, 5, 4)
      would represent that the third transition in the ground-truth alignment is
      from the match state _MATCH(2, 4) to the gap in X state _GAP_IN_X(2, 5).
      Both pos1 and pos2 use one-based indexing, reserving the use of the value
      zero for padding. In rare cases where the sequence pair has no aligned
      characters, tar_len will be zero.
    """
    keep_indices1 = tf.cast(
        self._vocab.compute_mask(seq1, self._gap_token), tf.int32)
    keep_indices2 = tf.cast(
        self._vocab.compute_mask(seq2, self._gap_token), tf.int32)
    states = keep_indices1 - keep_indices2
    m_states = tf.cast(
        tf.reshape(tf.where(states == self._MATCH), [-1]), tf.int32)
    n_matches = len(m_states)
    if n_matches == 0:
      return tf.zeros([3, 0], tf.int32)
    start, end = m_states[0], m_states[-1]
    offset1 = tf.reduce_sum(keep_indices1[:start])
    offset2 = start - offset1
    offset1 += self._n_prepend_tokens
    offset2 += self._n_prepend_tokens
    states = states[start:end + 1]
    keep_indices1 = keep_indices1[start:end + 1]
    keep_indices2 = keep_indices2[start:end + 1]
    m_states -= start
    segment_ids = tf.cumsum(tf.scatter_nd(
        m_states[1:, tf.newaxis],
        tf.ones(n_matches - 1, dtype=tf.int32),
        shape=[len(states)]))
    aux1 = tf.math.segment_sum(1 - keep_indices1, segment_ids)[:-1]
    aux2 = tf.math.segment_max(1 - keep_indices2, segment_ids)[:-1]
    gap_gap_trans_m_states_indices = tf.reshape(tf.where(aux1 * aux2), [-1])
    if len(gap_gap_trans_m_states_indices) > 0:  # pylint: disable=g-explicit-length-test
      for idx in gap_gap_trans_m_states_indices:
        s_i, e_i = m_states[idx] + 1, m_states[idx + 1]
        m_i = s_i + aux1[idx]
        v_x = tf.fill([aux1[idx]], self._GAP_IN_X)
        v_y = tf.fill([e_i - m_i], self._GAP_IN_Y)
        states = tf.raw_ops.TensorStridedSliceUpdate(
            input=states, begin=[s_i], end=[m_i], strides=[1], value=v_x)
        states = tf.raw_ops.TensorStridedSliceUpdate(
            input=states, begin=[m_i], end=[e_i], strides=[1], value=v_y)
    # Builds transitions.
    enc_trans = tf.gather(
        self._trans_encoder, self._hash_fn(states[:-1], states[1:]))
    enc_trans = tf.concat([self._init_trans, enc_trans], 0)
    # Positions such that (pos1[i], pos2[i]) for i = 0, ..., align_len - 1
    # describes the alignment "path".
    pos1 = offset1 + tf.cumsum(tf.cast(states >= self._MATCH, tf.int32))
    pos2 = offset2 + tf.cumsum(tf.cast(states <= self._MATCH, tf.int32))
    return tf.stack([pos1, pos2, enc_trans])


@gin.configurable
class CreateHomologyTargets(transforms.Transform):
  """Creates targets for pairwise homology detection task."""

  def __init__(self,
               process_negatives = True,
               **kwargs):
    super().__init__(**kwargs)
    self._process_negatives = process_negatives

  def call(self, values):
    def get_vals(indices):
      vals = tf.gather(values, indices)
      return tf.cast(vals[:, 0] == vals[:, 1], tf.int32)

    pos_indices = pairs.consecutive_indices(values)
    neg_indices = pairs.roll_indices(pos_indices)
    targets = [get_vals(pos_indices)]
    if self._process_negatives:
      targets.append(get_vals(neg_indices))
    return tf.concat(targets, 0)[:, tf.newaxis]  # [batch, 1]


@gin.configurable
class CreateBatchedWeights(transforms.Transform):
  """Adds sample weights based on targets."""

  def single_call(self, targets):
    return tf.ones(tf.shape(targets)[0], tf.float32)


@gin.configurable
class PadNegativePairs(transforms.Transform):
  """Pads tensor with identical all zeroes copy along batch axis."""

  def __init__(self,
               value = 0,
               **kwargs):
    super().__init__(**kwargs)
    self._value = value

  def single_call(self, tensor):
    shape, dtype = tf.shape(tensor), tensor.dtype
    padding = tf.fill(shape, tf.convert_to_tensor(self._value, dtype))
    return tf.concat([tensor, padding], 0)


@gin.configurable
class AddRandomTails(transforms.Transform):
  """Left and right pads sequence pair with random background sequence.

  Attributes:
    max_len: the maximum sequence length supported by the encoder.
    len_increase_ratio: limits the length of random sequence added (prefix plus
      suffix) to be no more than `len_increase_ratio` times the length of the
      original sequence.
    logits: (unnormalized) logits representing the background distribution over
      amino acid tokens. Order must coincide with that of `self._vocab`.
    gap_token: the character used to represent gaps in the `Vocabulary`.
  """
  # Logits for Pfam 34.0, estimated on 2,000,000 training sequences. Ordering
  # corresponds to `vocabulary.sequin`.
  PFAM_LOGITS = [-2.39965096, -4.32255327, -2.88551883, -2.74512196,
                 -3.20157522, -2.59228165, -3.83989081, -2.75582294,
                 -2.95477551, -2.28018693, -3.79231868, -3.3071618,
                 -3.17908159, -3.36927306, -2.86781839, -2.84114913,
                 -2.94084287, -2.59451421, -4.40920176, -3.47693793,
                 -15.38847123, -18.02752856, -14.79870241, -11.6010401,
                 -np.inf]

  def __init__(self,
               max_len = 512,
               len_increase_ratio = 2.0,
               logits = None,
               gap_token = '-',
               **kwargs):
    super().__init__(**kwargs)
    self._max_len = max_len
    self._len_increase_ratio = len_increase_ratio
    self._sampler = vocabulary.Sampler(
        vocab=self._vocab,
        logits=self.PFAM_LOGITS if logits is None else logits)
    self._gap_token = gap_token
    self._gap_code = self._vocab.get(self._gap_token)

  def sample_prefix_and_suffix_len(
      self,
      sequence,
  ):
    # Computes `seq_len` ignoring any gaps and stores it as float.
    mask = self._vocab.compute_mask(sequence, self._gap_token)
    seq_len = tf.cast(tf.reduce_sum(tf.cast(mask, tf.int32)), tf.float32)

    max_pad_len = tf.minimum(self._max_len - seq_len,
                             self._len_increase_ratio * seq_len)
    pad_len = tf.random.uniform((), maxval=max_pad_len)
    left_pad_len = tf.random.uniform((), maxval=pad_len)
    pad_len = tf.cast(pad_len, tf.int64)
    left_pad_len = tf.cast(left_pad_len, tf.int64)
    return left_pad_len, pad_len - left_pad_len

  def sample_tails(
      self,
      prefix_len,
      suffix_len,
      dtype,
  ):
    pad_seq = tf.cast(self._sampler.sample([prefix_len + suffix_len]), dtype)
    return pad_seq[:prefix_len], pad_seq[prefix_len:]

  def call(
      self,
      sequence_1,
      sequence_2,
  ):
    # Randomly samples prefix and suffix length for each sequence.
    prefix_len_1, suffix_len_1 = self.sample_prefix_and_suffix_len(sequence_1)
    prefix_len_2, suffix_len_2 = self.sample_prefix_and_suffix_len(sequence_2)
    # Samples prefix and suffix from univariate background distribution (iid).
    prefix_1, suffix_1 = self.sample_tails(
        prefix_len_1, suffix_len_1, sequence_1.dtype)
    prefix_2, suffix_2 = self.sample_tails(
        prefix_len_2, suffix_len_2, sequence_2.dtype)
    # Pads prefixes and suffixes to indicate they are unaligned.
    gap_code_1 = tf.cast(self._gap_code, sequence_1.dtype)
    prefix_1 = tf.concat([prefix_1, tf.fill([prefix_len_2], gap_code_1)], 0)
    suffix_1 = tf.concat([suffix_1, tf.fill([suffix_len_2], gap_code_1)], 0)
    gap_code_2 = tf.cast(self._gap_code, sequence_2.dtype)
    prefix_2 = tf.concat([tf.fill([prefix_len_1], gap_code_2), prefix_2], 0)
    suffix_2 = tf.concat([tf.fill([suffix_len_1], gap_code_2), suffix_2], 0)
    # Prepends and appends prefixes and suffixes, respectively.
    sequence_1 = tf.concat([prefix_1, sequence_1, suffix_1], 0)
    sequence_2 = tf.concat([prefix_2, sequence_2, suffix_2], 0)
    return sequence_1, sequence_2


@gin.configurable
class AddAlignmentContext(transforms.Transform):
  """Adds (unaligned) prefix / suffix from UniprotKB to Pfam-A seed MSAs.

  For each Pfam-A seed sequence, a random amount of prefix / suffix will be
  preprended / appended from the corresponding UniprotKB entry. This data
  augmentation increases the diversity of start / end positions for ground-truth
  alignments in Pfam-A seed, which would otherwise be dominated by alignments
  that are de facto global.

  For each sequence, the length of prefix plus suffix to be added is uniformly
  distributed between zero and the maximum possible amount given the maximum
  length supported by the encoder and the length of the UniprotKB protein
  sequence containing the Pfam-A seed entry. This amount is subsequently split
  between prefix and suffix uniformly as well.

  The transform modifies the MSA rows, prepending / appending the
  prefixes / suffixes as unaligned columns (i.e. the prefix of sequence_1 will
  be aligned to gaps in sequence_2 and viceversa).

  Attributes:
    max_len: the maximum sequence length supported by the encoder.
    gap_token: the character used to represent gaps in the `Vocabulary`.
  """

  def __init__(self,
               max_len = 512,
               gap_token = '-',
               **kwargs):
    super().__init__(**kwargs)
    self._max_len = max_len
    self._gap_token = gap_token
    self._gap_code = self._vocab.get(self._gap_token)

  def sample_prefix_and_suffix_len(
      self,
      sequence,
      full_sequence,
      start,
      end,
  ):
    # Computes length of Pfam-A seed `sequence` (excluding padding and gaps) and
    # UniprotKB context `full_sequence`.
    seq_len = end - start + 1
    full_seq_len = tf.cast(tf.shape(full_sequence)[0], tf.int64)
    # Computes length of prefix and suffix such that
    #   `concat([prefix, sequence, suffix]) == full_sequence`.
    full_prefix_len = start - 1  # `start` uses one-based indexing.
    full_suffix_len = full_seq_len - seq_len - full_prefix_len
    # Computes maximum amount of context that could be added, accounting for
    # encoder's maximum length restriction.
    full_ctx_len = tf.minimum(full_seq_len, self._max_len) - seq_len
    # To increase data diversity, uniformly samples amount of context to be
    # added between zero and the maximum possible.
    ctx_len = tf.random.uniform((), maxval=full_ctx_len + 1, dtype=tf.int64)
    # To increase data diversity, randomly splits the randomly sampled context
    # length between prefix and suffix.
    min_prefix_len = tf.maximum(ctx_len - full_suffix_len, 0)
    max_prefix_len = tf.minimum(full_prefix_len, ctx_len)
    prefix_len = tf.random.uniform(
        (), minval=min_prefix_len, maxval=max_prefix_len + 1, dtype=tf.int64)
    suffix_len = ctx_len - prefix_len
    return prefix_len, suffix_len

  def call(
      self,
      sequence_1,
      sequence_2,
      full_sequence_1,
      full_sequence_2,
      start_1,
      start_2,
      end_1,
      end_2,
  ):
    # Randomly samples prefix and suffix length independent for each sequence in
    # the aligned pair.
    prefix_len_1, suffix_len_1 = self.sample_prefix_and_suffix_len(
        sequence_1, full_sequence_1, start_1, end_1)
    prefix_len_2, suffix_len_2 = self.sample_prefix_and_suffix_len(
        sequence_2, full_sequence_2, start_2, end_2)
    # Fetches the prefix and suffix from UniprotKB context sequences.
    prefix_1 = full_sequence_1[start_1 - 1 - prefix_len_1:start_1 - 1]
    prefix_2 = full_sequence_2[start_2 - 1 - prefix_len_2:start_2 - 1]
    suffix_1 = full_sequence_1[end_1:end_1 + suffix_len_1]
    suffix_2 = full_sequence_2[end_2:end_2 + suffix_len_2]
    # Pads prefixes and suffixes to indicate they are unaligned.
    gap_code_1 = tf.cast(self._gap_code, full_sequence_1.dtype)
    prefix_1 = tf.concat([prefix_1, tf.fill([prefix_len_2], gap_code_1)], 0)
    suffix_1 = tf.concat([suffix_1, tf.fill([suffix_len_2], gap_code_1)], 0)
    gap_code_2 = tf.cast(self._gap_code, full_sequence_2.dtype)
    prefix_2 = tf.concat([tf.fill([prefix_len_1], gap_code_2), prefix_2], 0)
    suffix_2 = tf.concat([tf.fill([suffix_len_1], gap_code_2), suffix_2], 0)
    # Prepends and appends prefixes and suffixes, respectively.
    sequence_1 = tf.concat([prefix_1, sequence_1, suffix_1], 0)
    sequence_2 = tf.concat([prefix_2, sequence_2, suffix_2], 0)
    return sequence_1, sequence_2


@gin.configurable
class TrimAlignment(transforms.Transform):
  """Randomly trims Pfam-A seed MSAs for data augmentation purposes.

  Given a pair of MSA rows, this transform will, with probability `p_trim`, trim
  a random amount of prefix and suffix from each row by substitutying any amino
  acids in those streches of sequence by gaps. If new columns consisting of only
  gaps are created in the MSA as a result of this, they will be eliminated from
  the output.

  To coin toss determining whether trimming occurs is performed independently
  for each of the two MSA rows. Thus, for small `p_trim`, if trimming occurs
  w.h.p. it will affect only one of the two MSA rows.

  Attributes:
    max_trim_ratio: if trimming occurs, the length to be trimmed will be
      uniformly distributed between zero and `max_trim_ratio` times the length
      of the alignment (stretch of sequence between first and last matches).
    p_trim: the probability that trimming will be applied to each MSA row. These
      will be modelled as independent coin tosses.
    gap_token: the character used to represent gaps in the `Vocabulary`.
  """

  def __init__(self,
               max_trim_ratio = 0.5,
               p_trim = 0.0,
               gap_token = '-',
               **kwargs):
    super().__init__(**kwargs)
    self._max_trim_ratio = tf.convert_to_tensor(max_trim_ratio, tf.float32)
    self._p_trim = tf.convert_to_tensor(p_trim, tf.float32)
    self._gap_token = gap_token
    self._gap_code = self._vocab.get(self._gap_token)

  def maybe_trim_sequence(self,
                          sequence,
                          first,
                          last):
    msa_len = tf.cast(tf.shape(sequence)[0], tf.int64)
    alignment_len = last - first + 1
    alignment_len = tf.cast(alignment_len, self._max_trim_ratio.dtype)

    # Trims MSA with probability `self._p_trim`. If MSA is to be trimmed, the
    # length of sequence trimmed is uniformly distributed between zero and at
    # most `floor(self._max_trim_ratio * alignment_len)`.
    max_trim_len = tf.cast(self._max_trim_ratio * alignment_len, tf.int64)
    max_trim_len = tf.where(tf.random.uniform(()) > 1.0 - self._p_trim,
                            max_trim_len, 0)
    trim_len = tf.random.uniform((), maxval=max_trim_len + 1, dtype=tf.int64)
    # Randomly splits amount to be trimmed between the start and end of the
    # alignment. Note: we might want to increase the probability of trimming
    # only one tail for additional diversity.
    prefix_trim_len = tf.random.uniform((), maxval=trim_len + 1, dtype=tf.int64)
    suffix_trim_len = trim_len - prefix_trim_len

    # If MSA is to be trimmed, substitutes the `sequence` prefix up to
    # `first + prefix_trim_len` and suffix from `last - suffix_trim_len`, both
    # non-inclusive, by the gap token. This simulates a shorter query than the
    # ground-truth.
    erase_until = tf.where(prefix_trim_len > 0,
                           first + prefix_trim_len, 0)
    erase_from = tf.where(suffix_trim_len > 0,
                          last - suffix_trim_len, msa_len - 1)
    indices = tf.range(msa_len)
    mask = tf.logical_and(indices >= erase_until, indices <= erase_from)
    return tf.where(mask, sequence, tf.cast(self._gap_code, sequence.dtype))

  def call(
      self,
      sequence_1,
      sequence_2,
  ):
    # Finds the positions of the first and last match states in the MSA.
    gap_mask_1 = self._vocab.compute_mask(sequence_1, self._gap_token)
    gap_mask_2 = self._vocab.compute_mask(sequence_2, self._gap_token)
    matches = tf.logical_and(gap_mask_1, gap_mask_2)
    m_states = tf.cast(tf.reshape(tf.where(matches), [-1]), tf.int64)
    first, last = m_states[0], m_states[-1]

    # Trims sequences, independently at random. Note that, in practice, it is
    # intended for `self._p_trim` to be small (e.g. < 0.1). Hence, w.h.p., only
    # one of the two sequences will be actually trimmed.
    sequence_1 = self.maybe_trim_sequence(sequence_1, first, last)
    sequence_2 = self.maybe_trim_sequence(sequence_2, first, last)

    # Trimming might create new columns consisting of gaps only which need to be
    # removed prior to being fed to a `CreateAlignmentTargets` transform.
    trimmed_gap_mask_1 = self._vocab.compute_mask(sequence_1, self._gap_token)
    trimmed_gap_mask_2 = self._vocab.compute_mask(sequence_2, self._gap_token)
    keep_indices = tf.logical_or(trimmed_gap_mask_1, trimmed_gap_mask_2)
    keep_indices = tf.reshape(tf.where(keep_indices), [-1])
    sequence_1 = tf.gather(sequence_1, keep_indices)
    sequence_2 = tf.gather(sequence_2, keep_indices)
    return sequence_1, sequence_2
