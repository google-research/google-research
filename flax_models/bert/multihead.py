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

"""JAX transformations for creating batching attention across examples/heads."""
from typing import Any

from flax import struct
import jax
import jax.numpy as jnp


def counter(start, size):
  """Similar to jnp.arange(start, start + size, dtype=jnp.int32).

  The size must be an python int. This function is used for slices so that jax
  can infer that the slice shape is a compile-time constant.

  Args:
    start: ...
    size: ...

  Returns:
    ...
  """
  return start + jax.lax.iota(jnp.int32, size)


def split_join_nondiff_fns(sample_data):
  """Construct a pair of functions to split/join differentiable values."""
  treedef = jax.tree_structure(sample_data)
  def is_differentiable(x):
    return isinstance(x, jnp.ndarray) and jnp.issubdtype(x.dtype, jnp.inexact)
  inputs_is_differentiable = jax.tree_map(is_differentiable, sample_data)
  def split_differentiable(xs):
    differentiable_xs = jax.tree_multimap(
        lambda x, is_differentiable: x if is_differentiable else None,
        xs, inputs_is_differentiable)
    non_differentiable_xs = jax.tree_multimap(
        lambda x, is_differentiable: None if is_differentiable else x,
        xs, inputs_is_differentiable)
    return differentiable_xs, non_differentiable_xs
  def join_differentiable(differentiable_xs, non_differentiable_xs):
    """Reconstitute inputs pytree from differentiable/non-d. partitions."""
    differentiable_leaves = list(jax.tree_leaves(differentiable_xs))
    non_differentiable_leaves = list(jax.tree_leaves(non_differentiable_xs))
    leaves = []
    for is_differentiable in jax.tree_leaves(inputs_is_differentiable):
      if is_differentiable:
        leaves.append(differentiable_leaves.pop(0))
      else:
        leaves.append(non_differentiable_leaves.pop(0))
    assert not differentiable_leaves
    assert not non_differentiable_leaves
    return jax.tree_unflatten(treedef, leaves)
  return split_differentiable, join_differentiable


@struct.dataclass
class BatchedMultiHeadBundle:
  """A collection of values with info on how to map across examples/heads."""

  # Per-example values include inputs, outputs, and gradients wrt the inputs
  per_example: Any = None
  # Per-head values include parameters and gradients wrt the parameters
  per_head: Any = None
  # Per-example and per-head values include state
  per_example_per_head: Any = None
  # Singleton values include non-array arguments to a module's apply method
  singleton: Any = None

  def take(self, examples=None, heads=None):
    """Returns a new bundle with a subset of examples and heads."""
    # TODO(marcvanzee): Remove this when b/162398046 is fixed.
    # pytype: disable=attribute-error
    if examples is not None and heads is not None:
      return self.replace(
          per_example=jax.tree_map(lambda x: x[examples], self.per_example),
          per_head=jax.tree_map(lambda x: x[heads], self.per_head),
          per_example_per_head=jax.tree_map(
              lambda x: x[examples, heads], self.per_example_per_head))
    elif examples is not None:
      return self.replace(
          per_example=jax.tree_map(lambda x: x[examples], self.per_example),
          per_example_per_head=jax.tree_map(
              lambda x: x[examples], self.per_example_per_head))
    elif heads is not None:
      return self.replace(
          per_head=jax.tree_map(lambda x: x[heads], self.per_head),
          per_example_per_head=jax.tree_map(
              lambda x: x[:, heads], self.per_example_per_head))
    else:
      return self
    # pytype: enable=attribute-error

  def put(self, other, examples, heads=None):
    """Returns a new bundle with updated values for some examples and heads."""
    assert not jax.tree_leaves(other.singleton)
    if heads is not None:
      idxs_e = jax.ops.index[examples]
      per_example_fn = lambda x, y: jax.ops.index_add(x, idxs_e, y)
      per_head_fn = lambda x, y: jax.ops.index_add(x, jax.ops.index[heads], y)
      idxs_eh = jax.ops.index[examples, heads]
      per_example_per_head_fn = lambda x, y: jax.ops.index_update(x, idxs_eh, y)
    else:
      # Apply update to all heads
      idxs_e = jax.ops.index[examples]
      per_example_fn = lambda x, y: jax.ops.index_update(x, idxs_e, y)
      per_head_fn = lambda x, y: x + y
      per_example_per_head_fn = per_example_fn
    # pytype: disable=attribute-error
    return self.replace(
        per_example=jax.tree_multimap(
            per_example_fn, self.per_example, other.per_example),
        per_head=jax.tree_multimap(per_head_fn, self.per_head, other.per_head),
        per_example_per_head=jax.tree_multimap(
            per_example_per_head_fn, self.per_example_per_head,
            other.per_example_per_head))
    # pytype: enable=attribute-error

  def call_vmapped_over_heads(self, f):
    """Applies a function f to all heads in the bundle."""
    def wrapped_f(*args):
      """Wraps f to use tuples because vmap doesn't support bundle objects."""
      bundle_in = BatchedMultiHeadBundle(*args)
      bundle_out = f(bundle_in)
      return (bundle_out.per_example,
              bundle_out.per_head,
              bundle_out.per_example_per_head,
              bundle_out.singleton)
    vmapped_f = jax.vmap(
        wrapped_f, in_axes=(None, 0, 0, None), out_axes=(0, 0, 0, None))
    per_example, per_head, per_example_per_head, singleton = vmapped_f(
        self.per_example, self.per_head, self.per_example_per_head,
        self.singleton)
    return BatchedMultiHeadBundle(
        per_example=jax.tree_map(lambda x: x.sum(0), per_example),
        per_head=per_head,
        per_example_per_head=per_example_per_head,
        singleton=singleton)

  def call_vmapped_over_examples_and_heads(self, f):
    """Applies a function f to all examples and heads in the bundle."""
    def wrapped_f(*args):
      """Runs f for a single example, using tuples for input/output."""
      bundle_in = BatchedMultiHeadBundle(*args)
      bundle_out = bundle_in.call_vmapped_over_heads(f)
      return (bundle_out.per_example,
              bundle_out.per_head,
              bundle_out.per_example_per_head,
              bundle_out.singleton)
    vmapped_f = jax.vmap(
        wrapped_f, in_axes=(0, None, 0, None), out_axes=(0, None, 0, None))
    return BatchedMultiHeadBundle(*vmapped_f(
        self.per_example, self.per_head, self.per_example_per_head,
        self.singleton))

  def call_on_chunk(self, f, examples, heads=None, chunk_in=None):
    """Applies a function f to a subset of examples/heads in the bundle."""
    if chunk_in is None:
      chunk_in = self.take(examples, heads)
    if jnp.ndim(examples) == 0:
      if jnp.ndim(heads) == 0:
        return f(chunk_in)
      else:
        return chunk_in.call_vmapped_over_heads(f)
    else:
      assert heads is None, (
          "Slices with multiple examples must include all heads.")
      return chunk_in.call_vmapped_over_examples_and_heads(f)

  @staticmethod
  def pack_unpack_fns(sample_data, has_batch_dim, has_head_dim):
    """Create a pair of functions for packing/unpacking data into bundles."""
    def expand_structure(src, target):
      return jax.tree_unflatten(
          jax.tree_structure(target),
          [src for leaf in jax.tree_leaves(target)])
    has_batch_dim = jax.tree_multimap(
        expand_structure, has_batch_dim, sample_data)
    has_head_dim = jax.tree_multimap(
        expand_structure, has_head_dim, sample_data)
    treedef = jax.tree_structure(sample_data)
    has_batch_dim = jax.tree_leaves(has_batch_dim)
    has_head_dim = jax.tree_leaves(has_head_dim)

    def pack(data):
      per_example = []
      per_head = []
      per_example_per_head = []
      singleton = []
      for x, is_per_example, is_per_head in zip(jax.tree_leaves(data),
                                                has_batch_dim, has_head_dim):
        if is_per_example and not is_per_head:
          per_example.append(x)
        elif not is_per_example and is_per_head:
          per_head.append(x)
        elif is_per_example and is_per_head:
          per_example_per_head.append(x)
        else:
          singleton.append(x)
      return BatchedMultiHeadBundle(
          per_example=per_example,
          per_head=per_head,
          per_example_per_head=per_example_per_head,
          singleton=singleton)

    def unpack(bundle):
      per_example = list(bundle.per_example)
      per_head = list(bundle.per_head)
      per_example_per_head = list(bundle.per_example_per_head)
      singleton = list(bundle.singleton)
      leaves = []
      for is_per_example, is_per_head in zip(has_batch_dim, has_head_dim):
        if is_per_example and not is_per_head:
          leaves.append(per_example.pop(0))
        elif not is_per_example and is_per_head:
          leaves.append(per_head.pop(0))
        elif is_per_example and is_per_head:
          leaves.append(per_example_per_head.pop(0))
        else:
          leaves.append(singleton.pop(0))
      assert (not per_example) and (not per_head)
      assert (not per_example_per_head) and (not singleton)
      return jax.tree_unflatten(treedef, leaves)

    return pack, unpack


def chunked_multihead_map(
    fun,
    in_has_batch_dim, in_has_head_dim, out_has_batch_dim, out_has_head_dim,
    num_parallel_heads=None,
    use_python_loop=False,
    grad=None):
  """Map a function over examples and heads, and run it in chunks."""
  def fori_loop_python(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
      val = body_fun(i, val)
    return val

  def mapped_fun(*args):
    pack_inputs, unpack_inputs = BatchedMultiHeadBundle.pack_unpack_fns(
        args, in_has_batch_dim, in_has_head_dim)
    bundle_in = pack_inputs(args)
    per_example_per_head = jax.tree_leaves(bundle_in.per_example_per_head)
    if per_example_per_head:
      batch_size = per_example_per_head[0].shape[0]
      num_heads = per_example_per_head[0].shape[1]
    else:
      batch_size = jax.tree_leaves(bundle_in.per_example)[0].shape[0]
      num_heads = jax.tree_leaves(bundle_in.per_head)[0].shape[0]

    # Set up output value accumulators and packing/unpacking and accumulators
    # TODO(kitaev): Use jax.eval_shape or similar. The main challenge is that
    # python scalars are used to denote hyperparameters, while jax.eval_shape
    # treats them as arrays.
    in_shapes_bundle = bundle_in.take(0, 0)
    in_shapes = unpack_inputs(in_shapes_bundle)
    out_shapes = fun(*in_shapes)
    pack_outputs, unpack_outputs = BatchedMultiHeadBundle.pack_unpack_fns(
        out_shapes, out_has_batch_dim, out_has_head_dim)
    out_shapes_bundle = pack_outputs(out_shapes)
    assert not jax.tree_leaves(out_shapes_bundle.singleton), (
        "Only mapped outputs are supported.")
    initial_bundle_out = BatchedMultiHeadBundle(
        per_example=jax.tree_map(
            lambda x: jnp.zeros((batch_size,) + x.shape, x.dtype),
            out_shapes_bundle.per_example),
        per_head=jax.tree_map(
            lambda x: jnp.zeros((num_heads,) + x.shape, x.dtype),
            out_shapes_bundle.per_head),
        per_example_per_head=jax.tree_map(
            lambda x: jnp.zeros((batch_size, num_heads) + x.shape, x.dtype),
            out_shapes_bundle.per_example_per_head),
        singleton=[])

    # Adjust degree of parallelism based on the batch size.
    num_total_heads = batch_size * num_heads
    if num_parallel_heads is None or num_parallel_heads > num_total_heads:
      chunk_size = num_total_heads
    else:
      chunk_size = num_parallel_heads

    # Split processing examples and heads across chunks
    assert (batch_size * num_heads) % chunk_size == 0
    num_chunks = (batch_size * num_heads) // chunk_size
    if num_chunks == 1 or use_python_loop:
      fori_loop = fori_loop_python
    else:
      fori_loop = jax.lax.fori_loop
    def calculate_examples_and_heads(chunk_idx):
      if chunk_size == 1:
        # Run attention for a single example and a single head
        examples = chunk_idx // num_heads
        heads = chunk_idx % num_heads
      elif chunk_size < num_heads:
        # Run attention for a single example, but multiple heads.
        assert num_heads % chunk_size == 0
        idx = chunk_idx * chunk_size
        examples = idx // num_heads
        heads = counter(idx % num_heads, chunk_size)
      else:
        # Run attention for all heads for one or more examples.
        assert chunk_size % num_heads == 0
        idx = chunk_idx * chunk_size
        examples = counter(idx // num_heads, chunk_size // num_heads)
        heads = None  # All heads
      return examples, heads

    def run_single_example_and_head(bundle_in):
      return pack_outputs(fun(*unpack_inputs(bundle_in)))

    split_inputs, join_inputs = split_join_nondiff_fns(bundle_in)
    diff_in, nondiff_in = split_inputs(bundle_in)

    @jax.custom_vjp
    def f(diff_in):
      bundle_in = join_inputs(diff_in, nondiff_in)
      def run_chunk(chunk_idx, bundle_out):
        examples, heads = calculate_examples_and_heads(chunk_idx)
        chunk_out = bundle_in.call_on_chunk(run_single_example_and_head,
                                            examples, heads)
        bundle_out = bundle_out.put(chunk_out, examples, heads)
        return bundle_out
      bundle_out = fori_loop(0, num_chunks, run_chunk, initial_bundle_out)
      return bundle_out

    def f_fwd(diff_in):
      return f(diff_in), diff_in

    def f_bwd(diff_in, bundle_grad_out):
      def run_chunk(chunk_idx, diff_in_grad):
        examples, heads = calculate_examples_and_heads(chunk_idx)
        chunk_diff_in = diff_in.take(examples, heads)
        chunk_nondiff_in = nondiff_in.take(examples, heads)
        chunk_grad_out = bundle_grad_out.take(examples, heads)
        def run_chunk_inner(chunk_diff_in):
          chunk_in = join_inputs(chunk_diff_in, chunk_nondiff_in)
          chunk_out = bundle_in.call_on_chunk(
              run_single_example_and_head, examples, heads, chunk_in=chunk_in)
          return chunk_out
        _, vjpfun = jax.vjp(run_chunk_inner, chunk_diff_in)
        chunk_diff_in_grad, = vjpfun(chunk_grad_out)
        diff_in_grad = diff_in_grad.put(chunk_diff_in_grad, examples, heads)
        return diff_in_grad

      initial_diff_in_grad = jax.tree_map(jnp.zeros_like, diff_in)
      diff_in_grad = fori_loop(0, num_chunks, run_chunk, initial_diff_in_grad)
      return (diff_in_grad,)

    def f_fwd_and_bwd(diff_in, bundle_grad_out):
      bundle_in = join_inputs(diff_in, nondiff_in)
      def run_chunk(chunk_idx, loop_val):
        bundle_out, diff_in_grad = loop_val
        examples, heads = calculate_examples_and_heads(chunk_idx)
        chunk_diff_in = diff_in.take(examples, heads)
        chunk_nondiff_in = nondiff_in.take(examples, heads)
        chunk_grad_out = bundle_grad_out.take(examples, heads)
        def run_chunk_inner(chunk_diff_in):
          chunk_in = join_inputs(chunk_diff_in, chunk_nondiff_in)
          chunk_out = bundle_in.call_on_chunk(
              run_single_example_and_head, examples, heads, chunk_in=chunk_in)
          return chunk_out
        chunk_out, vjpfun = jax.vjp(run_chunk_inner, chunk_diff_in)
        chunk_diff_in_grad, = vjpfun(chunk_grad_out)
        diff_in_grad = diff_in_grad.put(chunk_diff_in_grad, examples, heads)
        bundle_out = bundle_out.put(chunk_out, examples, heads)
        return (bundle_out, diff_in_grad)

      initial_diff_in_grad = jax.tree_map(jnp.zeros_like, diff_in)
      bundle_out, diff_in_grad = fori_loop(
          0, num_chunks, run_chunk, (initial_bundle_out, initial_diff_in_grad))
      return bundle_out, diff_in_grad

    f.defvjp(f_fwd, f_bwd)

    if grad is None:
      return unpack_outputs(f(diff_in))
    else:
      bundle_out, diff_in_grad = f_fwd_and_bwd(diff_in, pack_outputs(grad))
      nondiff_in_dummy_grad = jax.tree_map(
          lambda x: "non-differentiable", nondiff_in)
      return (unpack_outputs(bundle_out),
              unpack_inputs(join_inputs(diff_in_grad, nondiff_in_dummy_grad)))

  return mapped_fun
