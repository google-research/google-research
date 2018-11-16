# coding=utf-8
# Copyright 2018 The Google Research Authors.
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

"""Finds sufficient input subsets for an input and black-box function.

This module implements the sufficient input subsets (SIS) procedure published
in [1]. The goal of this procedure is to interpret black-box functions by
identifying minimal sets of input features whose observed values alone suffice
for the same decision to be reached, even with all other input values missing.

More precisely, presuming the function's value at an input x exceeds a
pre-specified threshold (f(x) >= threshold), this procedure identifies a
collection of sparse subsets of features in x,
SIS-collection = [sis_1, sis_2, ...] where each sis_i satisfies
f(x_sis_i) >= threshold, and x_sis_i is a variant of x where all positions
except for those in the SIS are masked.

The authors of the SIS paper [1] recommend that the threshold be selected based
on the application, e.g. by precision/recall considerations in the case f is a
classifier. Note that as the threshold is increased, the SIS become larger.
The mask is likewise pre-specified and also highly application-dependent.
In the SIS paper, the authors mask values by using a mean feature value (e.g.
a mean word embedding in natural language applications, or a mean pixel value
in image classification). Other possible masking values could include <UNK>
tokens or zero values. Regardless of choice, one should check that the
function's prediction on the fully-masked input is uninformative.

Note: this procedure allows for interpreting of any arbitrary function, not
just those stemming from machine learning applications!


  Typical usage example:

    In this example, suppose f returns the L_2 norm of its inputs. With a
    threshold of 1, the two SIS identified are [1] and [2] (where the 1 and 2
    are indices into the original input), such that if we select just these
    values (and mask all others, with the supplied all-zero mask), we have
    f([0, 10, 0]) >= 1 and f([0, 0, 5]) >= 1.

  f_l2 = lambda batch_coords: np.linalg.norm(batch_coords, ord=2, axis=-1)
  threshold = 1.0
  initial_input = np.array([0.1, 10, 5])
  fully_masked_input = np.array([0, 0, 0])
  collection = sis_collection(f_l2, threshold, initial_input,
                              fully_masked_input)


  See docstring of sis_collection for more-detailed usage information.
  Additional usage examples can be found in tests for sis_collection.


References:

[1] Carter, B., Mueller, J., Jain, S., & Gifford, D. (2018). What made you do
    this? Understanding black-box decisions with sufficient input subsets.
    arXiv preprint arXiv:1810.03805. https://arxiv.org/abs/1810.03805
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np


class SISResult(
    collections.namedtuple(
        'SISResult',
        [
            'sis', 'ordering_over_entire_backselect',
            'values_over_entire_backselect', 'mask'
        ],
    )):
  """Specifies a single SIS identified by the find_sis procedure.

  Fields:
    sis: Array of idxs into the mask which define the sufficient input subset.
      These idxs describe the *unmasked positions* in the input. This array
      has shape (k x idx.shape), where k is the length of the SIS
      and idx is an idx into the mask. Note that in case of any ties between
      elements during backward selection, lower indices appear later in this
      array (see docstring for find_sis).
    ordering_over_entire_backselect: Array of shape (m x idx.shape), containing
      the order of idxs masked during backward selection while identifying this
      SIS, where 1 <= m <= d (and d is the max number of maskable positions).
      Later elements in this list were masked later during backward selection.
      If this is the first SIS extracted for this input, the m = d.
      Otherwise, m < d (as elements in earlier SIS are not considered again when
      extracting additional SIS in the sis_collection procedure).
      In particular, m + the total number of elements in all previous SIS = d.
    values_over_entire_backselect: Array of floats of shape (m,) containing the
      values found during backward selection, corresponding to the idxs in
      ordering_over_entire_backselect. At each position, the value is the value
      of f *after* that corresponding position is masked. The length m is
      defined in the same way as in ordering_over_entire_backselect.
    mask: Boolean array of shape M that corresponds to this SIS. Applying this
      mask to the original input produces a version of the input where all
      values are masked except for those in the SIS. The mask and input may have
      different shape, as long as the mask is broadcastable over the input (see
      docstring of sis_collection for details/example).
  """
  __slots__ = ()

  def __len__(self):
    """Defines len of SISResult as number of elements in the SIS."""
    return self.sis.shape[0]

  def __hash__(self):
    return NotImplemented

  def __eq__(self, other):
    """Checks equality between this SISResult and another SISResult.

    Check that all fields are the exactly equal (including orderings).

    Args:
      other: A SISResult instance.

    Returns:
      True if self and other are equal, and False otherwise.
    """
    if not isinstance(other, SISResult):
      return False

    return (np.array_equal(self.sis, other.sis) and
            np.array_equal(self.ordering_over_entire_backselect,
                           other.ordering_over_entire_backselect) and
            np.array_equal(self.values_over_entire_backselect,
                           other.values_over_entire_backselect) and
            np.array_equal(self.mask, other.mask))

  def approx_equal(self, other, rtol=1e-05, atol=1e-08):
    """Checks that this SISResult and another SISResult are approximately equal.

    SISResult.{sis, mask, ordering_over_entire_backselect} are compared exactly,
    while SISResult.values_over_entire_backselect are compared with slight
    tolerance (using np.allclose with provided rtol and atol). This is intended
    to check equality allowing for small differences due to floating point
    representations.

    Args:
      other: A SISResult instance.
      rtol: Float, the relative tolerance parameter used when comparing
        `values_over_entire_backselect` (see documentation for np.allclose).
      atol: Float, the absolute tolerance parameter used when comparing
        `values_over_entire_backselect` (see documentation for np.allclose).

    Returns:
      True if self and other are approximately equal, and False otherwise.
    """
    if not isinstance(other, SISResult):
      return False

    # SISResult.{sis, ordering_over_entire_backselect, mask} compared exactly.
    # SISResult.values_over_entire_backselect compared with slight tolerance.
    return (np.array_equal(self.sis, other.sis) and
            np.array_equal(self.ordering_over_entire_backselect,
                           other.ordering_over_entire_backselect) and
            np.allclose(
                self.values_over_entire_backselect,
                other.values_over_entire_backselect,
                rtol=rtol,
                atol=atol) and np.array_equal(self.mask, other.mask))

  def __ne__(self, other):
    return not self == other


def make_empty_boolean_mask(shape):
  """Creates empty boolean mask (no values are masked) given shape.

  Args:
    shape: A tuple of array dimensions (as in numpy.ndarray.shape).

  Returns:
    ndarray of given shape and boolean type, all values are True (not masked).
  """
  return np.full(shape, True, dtype=np.bool)


def make_empty_boolean_mask_broadcast_over_axis(shape, axis):
  """Creates empty boolean mask that is broadcastable over specified axes.

  Usage example:

    Given an input of shape (2, 3):

    - A broadcastable mask over columns (to mask entire columns at a time during
    the SIS procedure) has shape (1, 3) and is created using
    make_empty_boolean_mask_broadcast_over_axis((2, 3), 0).

    - A broadcastable mask over rows (to mask entire rows at a time during SIS)
    has shape (2, 1) and is created using
    make_empty_boolean_mask_broadcast_over_axis((2, 3), 1).

  Args:
    shape: Shape (a tuple of array dimensions, as in numpy.ndarray.shape) of the
      underlying input to be masked.
    axis: An integer, or tuple of integers, specifying the axis (or axes) to
      broadcast over.

  Returns:
    ndarray of boolean type (all values are True) and shape S, where S is the
      same as the provided shape, but with value 1 along each of the provided
      axes (see usage example above).
  """
  new_shape = np.copy(shape)
  new_shape[np.asarray(axis)] = 1
  return make_empty_boolean_mask(tuple(new_shape))


def _assert_sis_collection_disjoint(collection):
  """Asserts that all SIS in a SIS-collection are disjoint.

  Args:
    collection: A list of SISResult objects representing a SIS-collection.

  Raises:
    AssertionError if any of the sis attributes of the SISResults contains the
      same element as some other sis attribute in the collection.
  """
  all_seen_idxs = set()
  for sis_result in collection:
    sis_idxs = set((tuple(idx) for idx in sis_result.sis))
    if all_seen_idxs.intersection(sis_idxs):
      raise AssertionError(
          'SIS-collection is not disjoint. Got: %s' % (str(collection)))
    all_seen_idxs.update(sis_idxs)


def _transform_index_array_into_indexer(idx_array):
  """Transforms an array of index arrays into tuple for index those elements."""
  return tuple(np.asarray(idx_array).T)


def _transform_next_masks_index_array_into_tuple(idx_array):
  """Transforms array of mask idxs into tuple of column arrays for indexing.

  This transformation is needed in _produce_next_masks for indexing into
  next_masks, where one position in each of the next_masks is modified.

  For example, if idx_array is [[0, 1], [1, 1], [1, 2]] (contains three indices
  into a 2-dimensional mask), this function first augments the indices with
  an additional column of 0-indexed increasing integers (corresponding to which
  of the next_masks will be modified at the specified index) to produce:
  [[0, 0, 1], [1, 1, 1], [2, 1, 2]]. Then, to use this as a valid index into
  next_masks (as in _produce_next_masks), this array is sliced by column and
  cast as a tuple.

  See tests for additional examples.

  Args:
    idx_array: Array of shape (B x C) containing B coordinates, each of shape C.
      B and C must be >= 1 (i.e. the input array cannot be flat).

  Returns:
    A tuple of the row-augmented transformed indices. The tuple contains C+1
    arrays, each of shape (B,). The first element of the tuple is np.arange(B),
    and elements 1, ..., C+1 are column slices along each column of idx_array.
    If idx_array is empty,

  Raises:
    TypeError if idx_array is not 2-dimensional.
  """
  if len(idx_array.shape) != 2:
    raise TypeError('idx_array must be 2-dimensional.')

  return _transform_index_array_into_indexer(
      np.hstack((np.expand_dims(np.arange(idx_array.shape[0]), 1), idx_array)))


def _produce_next_masks(current_mask):
  """Produces all possible next masks starting from the current_mask.

  Each possible next mask is defined by masking a single unmasked position in
  the current mask. A position is considered masked when its value in the mask
  is False, and unmasked when the value in the mask in True.

  For example, if current_mask is [False, True, True], the two possible next
    masks are [False, False, True], and [False, True, False].

  Args:
    current_mask: Array of shape D containing the current mask. D may be
      multi-dimensional.

  Returns:
    Tuple containing (next_masks, masked_indices), where:

    next_masks is an array of all possible next masks, with shape (B x D), or an
      empty array ([]) if all positions are already masked (i.e. all values in
      current_mask are False), where B is the number of possible next masks
      (i.e. the number of True values in current_mask), and D is the dimension
      of the mask.

    next_masks_idxs is an array with shape (B x len(D.shape)), where each
      element is an index into to the position masked in each of the
      corresponding next_masks, or an empty array ([]) if no possible next mask.
      The idxs always appear in increasing (or in the multi-dimensional case,
      row-major) order.
  """
  current_mask = np.asarray(current_mask)

  next_masks_idxs = np.transpose(np.nonzero(current_mask))

  if next_masks_idxs.size == 0:
    next_masks = np.array([])
    next_masks_idxs = np.array([])
  else:
    next_masks = np.repeat(
        np.expand_dims(current_mask, axis=0),
        next_masks_idxs.shape[0],
        axis=0,
    )
    next_masks[_transform_next_masks_index_array_into_tuple(
        next_masks_idxs)] = False

  return next_masks, next_masks_idxs


def produce_masked_inputs(input_to_mask, fully_masked_input, batch_of_masks):
  """Applies masks to an input to produce the corresponding masked inputs.

  Args:
    input_to_mask: Array of shape D to be masked. Note that D may be
      multi-dimensional.
    fully_masked_input: The fully masked version of input_to_mask, also an array
      of shape D.
    batch_of_masks: Array of shape (B x D), a batch of masks to apply to
      input_to_mask, and B is at least 1.

  Returns:
    An array of masked inputs of shape (B x D), where each mask in
      batch_of_masks is applied to input_to_mask, and the masked values are
      taken from fully_masked_input.

    The order of masked inputs in the output corresponds to the order of masks
      in batch_of_masks.

  Raises:
    TypeError if shape of batch_of_masks does not have 1 more dimension than
      shape of input_to_mask.
  """
  input_to_mask = np.asarray(input_to_mask)
  fully_masked_input = np.asarray(fully_masked_input)
  batch_of_masks = np.asarray(batch_of_masks)

  # Check that batch_of_masks includes batch dimension.
  if len(batch_of_masks.shape) != len(input_to_mask.shape) + 1:
    raise TypeError('batch_of_masks must include batch dimension.')

  return np.where(batch_of_masks, input_to_mask, fully_masked_input)


def _backselect(f, current_input, current_mask, fully_masked_input):
  """Applies backward selection to a given input.

  Implements the BackSelect procedure in the SIS paper [1].

  Args:
    f: A function mapping an array of shape (B x D), containing a batch of B
      D-dimensional inputs to an array of scalar values with shape (B,).
    current_input: Array of shape D on which to apply the SIS procedure. D may
      be multi-dimensional. If any positions are already masked, these must be
      specified in current_mask.
    current_mask: Boolean array of shape M corresponding to already-masked
      positions in current_input. If no values are masked, this is an empty mask
      (i.e. all values in the mask == True).
    fully_masked_input: Array of shape D (same as current_input), in which all
      positions hold their masked value. If the mask and input are not the same
      shape (M != D), the mask must be broadcastable over the input. This
      enables masking entire rows or columns at a time. For example, for an
      input of shape (2, 3), using a mask of shape (1, 3) will mask entire
      columns at the same time during backward selection, and a mask of shape
      (2, 1) will mask entire rows at a time.

  Returns:
    List containing (idx, value) tuples, where idx is an array of shape
      (len(M.shape),) that indexes into the mask to identify the position
      masked, and value is the corresponding value with that position
      additionally masked during backward selection. Later tuples in the list
      correspond to positions masked later during backward selection.
    Note that if masking multiple positions leads to the same optimal value
      at any step during backward selection, the tie is broken by masking the
      lowest index first.
    If there are no more positions that can be masked from current_mask, returns
      empty list.
  """
  backselect_stack = []  # List of (idx, value) tuples during backselect.
  next_masks, next_masks_idxs = _produce_next_masks(current_mask)

  while next_masks_idxs.size > 0:
    next_masked_inputs = produce_masked_inputs(current_input,
                                               fully_masked_input, next_masks)
    next_masked_values = f(next_masked_inputs)
    optimal_batch_idx = np.argmax(next_masked_values)
    optimal_value = next_masked_values[optimal_batch_idx]
    optimal_idx_to_mask = next_masks_idxs[optimal_batch_idx]

    backselect_stack.append((optimal_idx_to_mask, optimal_value))

    current_mask = next_masks[optimal_batch_idx]
    next_masks, next_masks_idxs = _produce_next_masks(current_mask)

  return backselect_stack


def _find_sis_from_backselect(backselect_stack, threshold):
  """Constructs SIS using result of backward selection.

  Implements the FindSIS procedure in the SIS paper [1].

  Args:
    backselect_stack: List containing (idx, value) tuples, where idx identifies
      a position masked during backward selection (an array type), and value is
      the corresponding value after that position is masked. Later tuples in the
      list correspond to idxs masked later during backward selection. (This list
      is usually the output of _backselect.)
    threshold: A scalar, the threshold to use for identifying a SIS. Assumes
      that a SIS exists in the backselect_stack (i.e. some value exceeds the
      threshold).

  Returns:
    List containing SIS elements (defined by idx in backselect_stack tuples).
      These elements are ordered such that elements toward the top of the
      backselect_stack (added later to stack) appear earlier, i.e. the final
      element added to the backselect stack corresponds to the first position in
      the returned list.

    Assumes that there exists a SIS. Since the backselect_stack contains values
      after each position is masked, it cannot be certain that the prediction on
      all features is >= threshold. If there is no value in the backselect_stack
      that is >= threshold, then the SIS contains all idxs in the stack.

    If the value at the top of the backselect_stack is >= threshold, returns
      empty list (since value >= threshold with all positions masked).

  Raises:
    ValueError, if backselect_stack is empty, i.e. there is no valid SIS.
  """
  if not backselect_stack:
    raise ValueError('backselect_stack cannot be empty.')

  sis = []
  stack_iter = reversed(backselect_stack)
  i, value = next(stack_iter)
  if value < threshold:
    sis.append(i)
  for i, value in stack_iter:
    if value >= threshold:
      break
    else:
      sis.append(i)

  return sis


def find_sis(f, threshold, current_input, current_mask, fully_masked_input):
  """Returns a single SIS from one (possibly partially-masked) input.

  This method combines both the BackSelect and FindSIS procedures as defined
  in the SIS paper [1].

  Args:
    f: A function mapping an array of shape (B x D), containing a batch of B
      D-dimensional inputs to an array of scalar values with shape (B,).
    threshold: A scalar, used as threshold in SIS procedure. Corresponds to tau
      in the SIS paper [1].
    current_input: Array (or type convertible to array) of shape D on which to
      apply the SIS procedure. D may be multi-dimensional. If any positions are
      already masked, these must be specified in current_mask.
    current_mask: Boolean array (or type convertible to array) of shape M
      corresponding to already-masked positions in current_input. If no values
      are masked, this is an empty mask (i.e. all values in the mask == True).
    fully_masked_input: Array (or type convertible to array) of shape D (same as
      current_input), in which all positions hold their masked value. If the
      mask and input are not the same shape (M != D), the mask must be
      broadcastable over the input. This enables masking entire rows or columns
      at a time. For example, for an input of shape (2, 3), using a mask of
      shape (1, 3) will mask entire columns at the same time during backward
      selection, and a mask of shape (2, 1) will mask entire rows at a time.

  Returns:
    A SISResult corresponding to the identified SIS (see docstring for
      SISResult), or None if no SIS is identified, which occurs only when the
      prediction on the initially provided input is below the threshold, i.e.
      f(current_input) < threshold, or if all positions are given as masked in
      current_mask.

    The SIS values are sorted so that the earlier elements in the SIS were
      masked later during backward selection (see docstring of SISResult).

    Note that in the case of value ties during backward selection, the first of
      the positions is masked first (see docstring for _backselect). This means
      that if both elements end up in the SIS, the one with the larger index
      appears first in the SIS (since the SIS is built by adding elements from
      the backselect_stack in reverse order).
  """
  current_input = np.asarray(current_input)
  current_mask = np.asarray(current_mask)
  fully_masked_input = np.asarray(fully_masked_input)

  starting_prediction = f(np.asarray([current_input]))

  if starting_prediction < threshold:
    return None

  # Backward selection of unmasked inputs (BackSelect)
  backselect_stack = _backselect(f, current_input, current_mask,
                                 fully_masked_input)
  if not backselect_stack:  # all positions masked in current_mask
    return None

  # Find minimal SIS after backward selection (FindSIS)
  sis_idxs = _find_sis_from_backselect(backselect_stack, threshold)

  ordering_over_entire_backselect, values_over_entire_backselect = zip(
      *backselect_stack)

  # Create mask that selects only SIS elements
  mask = ~(make_empty_boolean_mask(current_mask.shape))
  mask[_transform_index_array_into_indexer(sis_idxs)] = True

  sis_result = SISResult(
      sis=np.array(sis_idxs, dtype=np.int_),
      ordering_over_entire_backselect=np.array(
          ordering_over_entire_backselect, dtype=np.int_),
      values_over_entire_backselect=np.array(
          values_over_entire_backselect, dtype=np.float_),
      mask=mask,
  )

  return sis_result


def sis_collection(f,
                   threshold,
                   initial_input,
                   fully_masked_input,
                   initial_mask=None):
  """Identifies the entire collection of SIS for an input.

  Implements the SIScollection procedure in the SIS paper [1].

  Args:
    f: A function mapping an array of shape (B x D), containing a batch of B
      D-dimensional inputs to an array of scalar values with shape (B,).
    threshold: A scalar, used as threshold in SIS procedure. Corresponds to tau
      in the SIS paper [1].
    initial_input: Array of shape D (or type convertible to array) on which to
      apply the SIS procedure. D may be multi-dimensional.
    fully_masked_input: Array (or type convertible to array) of shape D (same
      shape as initial_input), in which all positions hold their masked value.
    initial_mask: Optional. Boolean array (or type convertible to array) of
      shape M to define how input is masked. Default value is None, in which
      case a mask is created with the same shape as initial_input. If the mask
      and input are not the same shape (M != D), the mask must be broadcastable
      over the input. This enables masking entire rows or columns at a time. For
      example, for an input of shape (2, 3), using a mask of shape (1, 3) will
      mask entire columns at the same time during backward selection, and a mask
      of shape (2, 1) will mask entire rows at a time. (See
      make_empty_boolean_mask_broadcast_over_axis, which can construct
      broadcastable masks.)

  Returns:
    A list of SISResult objects, containing the entire SIS-collection for the
      initial_input. If no SIS exists (i.e. f(initial_input) < threshold),
      returns an empty list.

    Note that we follow the convention in the SIS paper [1], where a SIS only
      exists if f(initial_input) >= threshold. If f(initial_input) < threshold,
      but there exists a subset of features on which f(subset) >= threshold, we
      do not consider this a valid SIS.

    The order of SISResults in this list corresponds to the order of the SIS as
      they are found -- the first element is the first SIS found, and so on.
      Earlier SIS are masked while finding later SIS, so all the SIS in the
      SIS-collection are disjoint (as in the SIS paper [1]).
  """
  fully_masked_input = np.asarray(fully_masked_input)

  current_input = np.copy(initial_input)

  if initial_mask is None:
    current_mask = make_empty_boolean_mask(initial_input.shape)
  else:
    current_mask = np.copy(initial_mask)

  all_sis = []

  while True:
    sis_result = find_sis(f, threshold, current_input, current_mask,
                          fully_masked_input)
    if sis_result is None:
      break
    else:
      all_sis.append(sis_result)
      current_input = np.copy(
          produce_masked_inputs(current_input, fully_masked_input,
                                np.asarray([~sis_result.mask]))[0])
      # Update mask by AND with NOT SIS mask.
      current_mask = np.logical_and(current_mask, ~sis_result.mask)

  _assert_sis_collection_disjoint(all_sis)

  return all_sis
