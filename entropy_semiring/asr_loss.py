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

"""Neural Speech Recognition Losses."""

from lingvo import compat as tf
from lingvo.core import py_utils
import semiring
import utils

SeqLen = utils.SeqLen


def interleave_with_blank(x, blank,
                          axis):
  """Interleaves x with blanks along axis.

  E.g. if input is AAA, we want the output to be bAbAbAb.

  Args:
     x: Tensor of shape [..., U, ...].
     blank: Tensor of same shape as x except U is now 1.
     axis: Axis of input that U corresponds to.

  Returns:
    output: Tensor of same shape as x except U is now 2*U+1.
  """
  input_shape = tf.shape(x)
  input_rank = tf.rank(x)
  u = input_shape[axis]

  # Create a blanks tensor of same shape as x.
  blanks = tf.broadcast_to(blank, input_shape)  # [..., U, ...]

  # Interleave x with blanks.
  interleaved_dims = input_shape + tf.one_hot(
      axis, depth=input_rank, dtype=tf.int32) * u
  interleaved = tf.reshape(
      tf.stack([blanks, x], axis=axis + 1), interleaved_dims)  # [..., 2*U, ...]

  # Add an extra blank at the end.
  interleaved = tf.concat([interleaved, blank], axis=axis)  # [..., 2*U+1, ...]

  return interleaved


def ctc(input_logits, output_labels,
        input_seq_len, output_seq_len):
  """CTC loss.

  B: Batch size.
  T: Input sequence dimension.
  U: Output sequence dimension.
  V: Vocabulary size.

  Alex Graves's CTC paper can be found at:
  https://www.cs.toronto.edu/~graves/icml_2006.pdf

  The following Distill article is also a helpful reference:
  https://distill.pub/2017/ctc/

  Args:
    input_logits: Logits for input sequence of shape [B, T, V]. We assume the
      0th token in the vocabulary represents the blank.
    output_labels: Labels for output sequence of shape [B, U].
    input_seq_len: Sequence lengths for input sequence of shape [B].
    output_seq_len: Sequence lengths for output sequence of shape [B].

  Returns:
    CTC loss, which is a tf.Tensor of shape B.
  """
  log_sum = ctc_semiring(
      sr=semiring.LogSemiring(),
      sr_inputs=(input_logits,),
      output_labels=output_labels,
      input_seq_len=input_seq_len,
      output_seq_len=output_seq_len)[0]
  return -log_sum


def rnnt(s1_logits,
         s2_logits,
         s1_seq_len,
         s2_seq_len):
  """RNN-T loss for two sequences.

  B: Batch size.
  D: Loop skewing dimension, i.e. the sum of s1 and s2.
  S1: Sequence 1 dimension (canonically, the input).
  S2: Sequence 2 dimension (canonically, the output).

  At each step, we consume/produce from each of the two sequences until all the
  tokens have been used. We abide by the convention that the last token is
  always from sequence 1. The RNN-T loss is a dynamic programming algorithm that
  sums the probability of every possible such sequence.

  The dynamic programming equation is:
    alpha[s1, s2] = alpha[s1-1, s2] * s1_logits[s1-1, s2] +
                    alpha[s1, s2-1] * s2_logits[s1, s2-1].

  The boundary condition is:
    loss = alpha[S1, S2] * s1_logits[S1, S2].

  In practice, we work in the log space for numerical stability. Instead of
  two nested for loops, we do loop skewing by iterating through the diagonal
  line, D.

  Loop skewing for RNN-T is discussed here:
    T. Bagby, K. Rao and K. C. Sim, "Efficient Implementation of Recurrent
    Neural Network Transducer in Tensorflow," 2018 IEEE Spoken Language
    Technology Workshop (SLT), Athens, Greece, 2018, pp. 506-512.

  The following wikipedia article is also a helpful reference:
  https://en.wikipedia.org/wiki/Polytope_model

  Args:
    s1_logits: Logits for sequence 1 of shape [B, S1, S2]. We always end with a
      token from sequence 1.
    s2_logits: Logits for sequence 2 of shape [B, S1, S2].
    s1_seq_len: Sequence lengths for sequence 1 of shape [B].
    s2_seq_len: Sequence lengths for sequence 2 of shape [B].

  Returns:
    RNN-T loss, which is a tf.Tensor of shape B.
  """
  log_sum = rnnt_semiring(
      sr=semiring.LogSemiring(),
      s1_inputs=(s1_logits,),
      s2_inputs=(s2_logits,),
      s1_seq_len=s1_seq_len,
      s2_seq_len=s2_seq_len)[0]
  return -log_sum


def ctc_semiring(sr,
                 sr_inputs, output_labels,
                 input_seq_len,
                 output_seq_len):
  """CTC loss for an arbitrary semiring.

  The CTC dynamic programming graph stays the same, but the addition and
  multiplication operations are now given by the semiring.

  B: Batch size.
  T: Input sequence dimension.
  U: Output sequence dimension.
  V: Vocabulary size.

  Alex Graves's CTC paper can be found at:
  https://www.cs.toronto.edu/~graves/icml_2006.pdf

  The following Distill article is also a helpful reference:
  https://distill.pub/2017/ctc/

  Args:
    sr: Semiring object where each state is a tuple of tf.Tensors of shape [B,
      T, V].
    sr_inputs: Input to the CTC graph.
    output_labels: Labels for output sequence of shape [B, U].
    input_seq_len: Sequence lengths for input sequence of shape [B].
    output_seq_len: Sequence lengths for output sequence of shape [B].

  Returns:
    Output of the CTC graph, which is a tuple of tf.Tensors of shape [B].
  """
  tf.debugging.assert_shapes([(state, ['B', 'T', 'V']) for state in sr_inputs])
  tf.debugging.assert_shapes([
      (output_labels, ['B', 'U']),
      (input_seq_len, ['B']),
      (output_seq_len, ['B']),
  ])

  # Convert inputs to tensors.
  sr_inputs = tuple(tf.convert_to_tensor(i) for i in sr_inputs)  # [B, T, V]
  output_labels = tf.convert_to_tensor(output_labels)  # [B, U]
  input_seq_len = tf.convert_to_tensor(input_seq_len)  # [B]
  output_seq_len = tf.convert_to_tensor(output_seq_len)  # [B]

  b, t, v = py_utils.GetShape(sr_inputs[0])
  u = py_utils.GetShape(output_labels)[1]
  dtype = sr_inputs[0].dtype

  # Create a bitmask for the labels that returns True for the start of every
  # contiguous segment of repeating characters.
  # E.g. For AABBB, we have TFTFF. After padding with blanks (which are filled
  # with False), i.e. bAbAbBbBbBb, we have FTFFFTFFFFF.
  is_label_distinct = tf.not_equal(output_labels[:, :-1],
                                   output_labels[:, 1:])  # [B, U-1]
  is_label_distinct = tf.pad(
      is_label_distinct, [[0, 0], [1, 0]], constant_values=True)  # [B, U]
  is_label_distinct = interleave_with_blank(
      is_label_distinct, tf.zeros([b, 1], dtype=tf.bool), axis=1)  # [B, 2*U+1]

  # Create CTC tables of shape [B, 2*U+1, T].
  ctc_state_tables = []
  onehot_labels = tf.one_hot(output_labels, depth=v, dtype=dtype)  # [B, U, V]
  for state in sr_inputs:
    ctc_state_table = tf.einsum('buv, btv -> but', onehot_labels,
                                state)  # [B, U, T]
    blank = tf.transpose(state[:, :, :1], [0, 2, 1])  # [B, 1, T]
    ctc_state_table = interleave_with_blank(
        ctc_state_table, blank, axis=1)  # [B, 2*U+1, T]
    ctc_state_tables.append(ctc_state_table)
  ctc_state_tables = sr.convert_logits(tuple(ctc_state_tables))  # [B, 2*U+1, T]

  # Mask out invalid starting states, i.e. all but the first two.
  start_mask = tf.concat(
      [tf.ones([2], dtype=tf.bool),
       tf.zeros([2 * u - 1], dtype=tf.bool)],
      axis=0)[:, tf.newaxis]  # [2*U+1, 1]
  start_mask = tf.pad(
      start_mask, [[0, 0], [0, t - 1]], constant_values=True)  # [2*U+1, T]
  start_mask = tf.tile(start_mask[tf.newaxis], [b, 1, 1])  # [B, 2*U+1, T]

  additive_identity = sr.additive_identity(
      shape=(b, 2 * u + 1, t), dtype=dtype)  # [B, 2*U+1, T]
  ctc_state_tables = [
      tf.where(start_mask, cst, ai)
      for cst, ai in zip(ctc_state_tables, additive_identity)
  ]  # [B, 2*U+1, T]

  # Iterate through the CTC tables.
  ctc_state_tables = tuple(
      tf.transpose(cst, [2, 0, 1]) for cst in ctc_state_tables)  # [T, B, 2*U+1]

  def _generate_transitions(acc, ai):
    """Generates CTC state transitions."""
    plus_zero = acc  # [B, 2*U+1]
    plus_one = tf.pad(
        acc[:, :-1], [[0, 0], [1, 0]], constant_values=ai[0, 0])  # [B, 2*U+1]
    plus_two = tf.pad(
        acc[:, :-2], [[0, 0], [2, 0]], constant_values=ai[0, 0])  # [B, 2*U+1]
    plus_two = tf.where(is_label_distinct, plus_two, ai)  # [B, 2*U+1]
    return [plus_zero, plus_one, plus_two]

  def _step(acc, x):
    additive_identity = sr.additive_identity(
        shape=(b, 2 * u + 1), dtype=dtype)  # [B, 2*U+1]
    path_sum = tuple(
        _generate_transitions(acc_i, ai)
        for acc_i, ai in zip(acc, additive_identity))  # [B, 2*U+1]
    path_sum = sr.add_list(utils.tuple_to_list(path_sum))  # [B, 2*U+1]
    new_acc = sr.multiply(path_sum, x)  # [B, 2*U+1]
    return new_acc

  ctc_state_tables = tf.scan(_step, ctc_state_tables)  # [T, B, 2*U+1]

  # Sum up the final two states.
  indices_final_state = tf.stack([
      input_seq_len - 1,
      tf.range(b),
      2 * output_seq_len,
  ],
                                 axis=1)  # [B, 3]
  indices_penultimate_state = tf.stack([
      input_seq_len - 1,
      tf.range(b),
      2 * output_seq_len - 1,
  ],
                                       axis=1)  # [B, 3]

  final_state = tuple(
      tf.gather_nd(cst, indices_final_state) for cst in ctc_state_tables)  # [B]
  penultimate_state = tuple(
      tf.gather_nd(cst, indices_penultimate_state)
      for cst in ctc_state_tables)  # [B]
  result = sr.add(final_state, penultimate_state)  # [B]

  # Zero out invalid losses.
  return tuple(
      tf.where(tf.math.is_inf(r), tf.zeros_like(r), r) for r in result)  # [B]


def rnnt_semiring(
    sr,
    s1_inputs,
    s2_inputs,
    s1_seq_len,
    s2_seq_len):
  """RNN-T loss for an arbitrary semiring.

  The RNN-T dynamic programming graph stays the same, but the addition (+) and
  multiplication (*) operations are now given by the semiring.

  B: Batch size.
  D: Loop skewing dimension, i.e. the sum of s1 and s2.
  S1: Sequence 1 dimension (canonically, the input).
  S2: Sequence 2 dimension (canonically, the output).

  At each step, we consume/produce from each of the two sequences until all the
  tokens have been used. We abide by the convention that the last token is
  always from sequence 1. The RNN-T loss is a dynamic programming algorithm that
  sums the probability of every possible such sequence.

  The dynamic programming equation is:
    alpha[s1, s2] = alpha[s1-1, s2] (*) s1_inputs[s1-1, s2] (+)
                    alpha[s1, s2-1] (*) s2_inputs[s1, s2-1].

  The boundary condition is:
    loss = alpha[S1, S2] (*) s1_inputs[S1, S2].

  Instead of two nested for loops, we do loop skewing by iterating through the
  diagonal line, D.

  Loop skewing for RNN-T is discussed here:
    T. Bagby, K. Rao and K. C. Sim, "Efficient Implementation of Recurrent
    Neural Network Transducer in Tensorflow," 2018 IEEE Spoken Language
    Technology Workshop (SLT), Athens, Greece, 2018, pp. 506-512.

  The following wikipedia article is also a helpful reference:
  https://en.wikipedia.org/wiki/Polytope_model

  Args:
    sr: Semiring object where each state is a tuple of tf.Tensors of shape [B,
      S1, S2].
    s1_inputs: Sequence 1 inputs to the RNN-T graph.
    s2_inputs: Sequence 2 inputs to the RNN-T graph.
    s1_seq_len: Sequence lengths for sequence 1 of shape [B].
    s2_seq_len: Sequence lengths for sequence 2 of shape [B].

  Returns:
    Output of the RNN-T graph, which is a tuple of tf.Tensors of shape [B].
  """
  tf.debugging.assert_shapes([(state, ['B', 'S1', 'S2']) for state in s1_inputs
                             ])
  tf.debugging.assert_shapes([(state, ['B', 'S1', 'S2']) for state in s2_inputs
                             ])
  tf.debugging.assert_shapes([
      (s1_seq_len, ['B']),
      (s2_seq_len, ['B']),
  ])

  # Convert inputs to tensor.
  s1_inputs = tuple(tf.convert_to_tensor(i) for i in s1_inputs)  # [B, S1, S2]
  s2_inputs = tuple(tf.convert_to_tensor(i) for i in s2_inputs)  # [B, S1, S2]
  s1_seq_len = tf.convert_to_tensor(s1_seq_len)  # [B]
  s2_seq_len = tf.convert_to_tensor(s2_seq_len)  # [B]

  # Convert inputs to semiring inputs.
  s1_inputs = sr.convert_logits(s1_inputs)  # [B, S1, S2]
  s2_inputs = sr.convert_logits(s2_inputs)  # [B, S1, S2]

  b, s1, s2 = py_utils.GetShape(s1_inputs[0])
  d = s1 + s2 - 1
  dtype = s1_inputs[0].dtype

  # Mask invalid logit states.
  s1_mask = tf.sequence_mask(
      s1_seq_len, maxlen=s1)[:, :, tf.newaxis]  # [B, S1, 1]
  s1_mask = tf.broadcast_to(s1_mask, [b, s1, s2])  # [B, S1, S2]
  additive_identity = sr.additive_identity(
      shape=(b, s1, s2), dtype=dtype)  # [B, S1, S2]
  s2_inputs = tuple(
      tf.where(s1_mask, s2_state, ai)
      for s2_state, ai in zip(s2_inputs, additive_identity))  # [B, S1, S2]

  # Skew the RNN-T table.
  def _skew(y, ai_scalar):
    """Skew the loop along the dimension of sequence 1."""
    # [B, S1, S2] => [D, B, S2]
    y = tf.transpose(y, [0, 2, 1])  # [B, S2, S1]
    y = tf.pad(
        y, [[0, 0], [0, 0], [0, s2]],
        constant_values=ai_scalar[0])  # [B, S2, S1+S2]
    y = tf.reshape(y, [b, s2 * (s1 + s2)])  # [B, S2*(S1+S2)]
    y = y[:, :s2 * d]  # [B, S2*(S1+S2-1)]
    y = tf.reshape(y, [b, s2, d])  # [B, S2, S1+S2-1]
    y = tf.transpose(y, [2, 0, 1])  # [D, B, S2]
    return y

  additive_identity_scalar = sr.additive_identity(
      shape=(1,), dtype=dtype)  # [D, B, S2]
  r_s1 = tuple(
      _skew(s1_i, ai)
      for s1_i, ai in zip(s1_inputs, additive_identity_scalar))  # [D, B, S2]
  r_s2 = tuple(
      _skew(s2_i, ai)
      for s2_i, ai in zip(s2_inputs, additive_identity_scalar))  # [D, B, S2]

  # Iterate through the RNN-T table.
  def _shift_down_s2(y, ai):
    """Shift the sequence 2 inputs down by one time step."""
    return tf.pad(
        y[:, :-1], [[0, 0], [1, 0]], constant_values=ai[0, 0])  # [B, S2]

  def _step(alpha_d, x):
    s1_d, s2_d = x  # [B, S2]
    additive_identity = sr.additive_identity(
        shape=(b, s2), dtype=dtype)  # [B, S2]
    a_s1_d = sr.multiply(alpha_d, s1_d)  # [B, S2]
    a_s2_d = sr.multiply(alpha_d, s2_d)  # [B, S2]
    a_s2_d = tuple(
        _shift_down_s2(as2d_i, ai)
        for as2d_i, ai in zip(a_s2_d, additive_identity))  # [B, S2]
    path_sum = sr.add(a_s1_d, a_s2_d)  # [B, S2]
    return path_sum

  # Initial condition for the loop accumulator.
  multiplicative_identity = sr.multiplicative_identity(
      shape=(b, 1), dtype=dtype)  # [B, 1]
  additive_identity = sr.additive_identity(
      shape=(b, s2 - 1), dtype=dtype)  # [B, S2-1]
  init_d = tuple(
      tf.concat([mi, ai], axis=1)
      for mi, ai in zip(multiplicative_identity, additive_identity))  # [B, S2]

  # Compute the RNN-T loss with loop skewing.
  r_alpha = tf.scan(
      _step,
      (r_s1, r_s2),
      init_d,
  )  # [D+1, B, S2]
  indices = tf.stack([s1_seq_len + s2_seq_len - 1,
                      tf.range(b), s2_seq_len],
                     axis=1)  # [B, 3]
  result = tuple(tf.gather_nd(r_a, indices) for r_a in r_alpha)  # [B]

  # Zero out invalid losses from length zero sequences. These losses are invalid
  # because empty sequences do not produce an alignment path.
  valid_seqs = tf.math.logical_and(s1_seq_len > 0, s2_seq_len > 0)  # [B]
  return tuple(tf.where(valid_seqs, r, tf.zeros_like(r)) for r in result)  # [B]
