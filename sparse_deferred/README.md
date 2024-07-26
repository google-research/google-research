# Sparse Deferred

Sparse Deferred (SD) is a backend-agnostic framework that defines a number of
algorithms and machine learning models. It implements a number of Graph Neural
Network models, graph algorithms, linear algebra routines (e.g., SVD).
Importantly, SD expresses these implementations on the *computation graph* of
data matrices, such as, the adjacency and feature matrices.
This gives practical advantage of expressing dense-looking models (e.g.,DeepWalk, MixHop),
```
sd.prod([adj, adj, adj]) @ embedding_matrix
```
but without computing the dense `M = adj ** 3`. Instead, the DAG representing `M` will be passed to algorithms, which may only need to compute (multiplication) `M`-times-Tensor, without needing explicit values of `M`. The multiplication will be pushed towards the leaves (`adj`) of the computation graph.

As the framework is backend-agnostic, it is independent from computation engines
such as JAX, TensorFlow,
PyTorch, Numpy, scipy, etc. Nonetheless, we integrate with some of these
engines by implementing facade interface `matrix.ComputeEngine`.

## Primitive: `Matrix`

The primitive object offered by SD is `Matrix`, which must encapsulate a rank-2
Matrix. Each instance can either:

1. directly store the entries (e.g., as dense, sparse), or,
1. be defined as a **deferred** linear composition of other `Matrix` operations.

### Implementations of `Matrix`

There are several `Matrix` instantiations, e.g.,

1. One wraps `numpy` arrays, e.g.,
   
   ```
   import sparse_deferred.np as sdnp
   mat = sdnp.NumpyMatrix(np.array([[1, 2, 0], [0, -1, 3]]))
   ```
1. One that wraps `tf.Tensor`, e.g.,
   
   ```
   import sparse_deferred.tf as sdtf
   mat = sdtf.DenseMatrix(tf.constant([[1, 2, 0], [0, -1, 3]]))
   ```
1. One that wraps `tf.sparse.SparseTensor`, e.g.,
   
   ```
   import sparse_deferred.tf as sdtf
   mat = sdtf.SparseMatrix(
       tf.sparse.from_dense(
           tf.constant([[1, 2, 0], [0, -1, 3]])))
   ```
1. Wrapper for graph data matrices. The class `graph_struct.GraphStruct` can
   hold graphs (graphs, nodes, edges, and their features).

### **deferred** VS **computed**

#### **computed**
Calling `__matmul__` or `__rmatmul__` **computes** a product of
`Matrix` against another (numpy, tf, jax, ...) `Tensor`. Specifically:

```
import sparse_deferred as sd

mat: sd.Matrix = ... # (e.g., per above)

# suppose mat.shape == [2, 5]
tensor = tf.ones([5, 3, 7])
prod = mat @ tensor
assert isinstance(prod, tf.Tensor)  # i.e., computed already!
assert prod.shape == (2, 3, 7)
```

NOTE: It must be that the `engine` of `mat` is TensorFlow, for the above to
work. This is the job of whoever created the `Matrix` instance. For instance,
if the engine was numpy, then the middle line should be replaced by `np.ones`.
To write backend-agnostic code, one could use:
```
tensor = mat.engine.ones([5, 3, 7])
```

#### **deferred**
On the other hand, instructions exposed by `sd` are deferred:

```
import sparse_deferred as sd

mat1: sd.Matrix = ... # (e.g., per above), assume rectangular
mat2: sd.Matrix = ... # (e.g., per above), assume square

mat3 = sd.product([mat1, mat2])

assert isinstance(mat3, sd.Matrix)  # i.e., deferred.

mat4 = sd.product([mat2, mat2, mat2])  # deferred MatrixPower(mat2, 3)
```

In addition, all methods defined on `Matrix` class (except for
`@ == {__matmul__, __rmatmul__}`) also return a deferred expression:

```
right_stochastic_mat = mat2.normalize_right()
assert isinstance(right_stochastic_mat, sd.Matrix)  # i.e., deferred

# Above is equivalent to `right_stochastic_mat_equiv`, defined next:
rowsums = mat2.rowsums(1.0)
# rowsums must be Tensor of tensorflow, numpy, jax, etc.

diag_inv_row_sums = mat2.engine.deferred_diag(1.0 / rowsums)

right_stochastic_mat_equiv = sd.prod([right_stochastic_mat, diag_inv_row_sums])
assert isinstance(right_stochastic_mat_equiv, sd.Matrix)  # i.e., deferred
```

Finally, if you want to **materialize** (i.e., compute) the deferred matrix,
into a real matrix, you can multiply by the identity matrix, from either side.

```
materialized_option1 = mat @ mat.engine.eye(mat.shape[0])
materialized_option2 = mat.engine.eye(mat.shape[1]) @ mat

mat.engine.assert_equal(materialized_option1, materialized_option2)
```
However, the above is **not advised** if `mat` has large dimensions. In fact,
many algorithms and models **do not need** to materialize `mat`, including,
computing its SVD, topologically-ordering nodes, or calculating DeepWalk-like
embeddings under Frobenius Norm objective.

## Utility

This section will be written as we add code to repo.

## `ComputeEngine` Backends

1. Numpy.
1. Tensorflow (we also offer functions to import from TF-GNN formats).
1. JAX.

## About us
This Framework is developed to satisfy research and production use-cases that
our team cares about. The primary contributors to this framework are:
* Sami Abu-el-Haija (haija@google.com)
* Hasan Awais (hasanawais@google.com)
* Aditya Mishra (mishraaditya@google.com)
* Mangpo Phothilimthana (mangpo@google.com)
* Bryan Perozzi (bperozzi@google.com)

## Disclaimer

This is not an official Google product.
