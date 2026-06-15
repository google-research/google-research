# coding=utf-8
# Copyright 2026 The Google Research Authors.
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

"""K-norm mechanism implementations.

All the code in this file is taken directly from
https://github.com/google-research/google-research/tree/master/k_norm
"""


import numpy as np
from scipy import special
from scipy import stats


def random_signs(vector):
  """Returns vector with each coordinate's sign randomly set.

  Args:
    vector: Numpy array whose signs should be randomly set.
  """
  signs = -np.random.randint(2, size=len(vector)) * 2 + 1
  return signs * vector


def compute_eulerian_numbers(d):
  """Returns A where A[i, j] = Eulerian number A(i, j).

  Args:
    d: Integer such that the returned matrix has d+1 rows and columns.
  """
  eulerian_numbers = np.zeros((d + 1, d + 1))
  eulerian_numbers[:, 0] = np.ones(d + 1)
  for row in range(2, d + 1):
    for k in range(1, row + 1):
      eulerian_numbers[row, k] = (row - k) * eulerian_numbers[
          row - 1, k - 1
      ] + (k + 1) * eulerian_numbers[row - 1, k]
  return eulerian_numbers


def compute_add_ascent_indices(eulerian_numbers, k):
  """Returns array where array[i] = 1 <==> i + 1 adds an ascent when inserted.

  Args:
    eulerian_numbers: Pre-computed ndarray of Eulerian numbers, of size at least
      (k + 1) x (k + 1).
    k: Integer number of ascents.

  Returns:
    An array specifying the values which add an ascent upon insertion when
    iteratively constructing a permutation in S_d. For example, [0, 1, 1, 0]
    means that 2 and 3 add ascents when inserted, and this corresponds to
    permutations (4123), (1423), and (1243).
  """
  d = len(eulerian_numbers) - 1
  add_ascents = np.zeros(d)
  current_k = k
  for current_d in range(1, d + 1)[::-1]:
    left_mass = (current_d - current_k) * eulerian_numbers[
        current_d - 1, current_k - 1
    ]
    right_mass = (current_k + 1) * eulerian_numbers[current_d - 1, current_k]
    if np.random.uniform() < left_mass / (left_mass + right_mass):
      add_ascents[current_d - 1] = 1
      current_k = current_k - 1
    else:
      add_ascents[current_d - 1] = 0
  return add_ascents


def sample_permutation_with_ascents(eulerian_numbers, k):
  """Returns a uniform random permutation on [d] with k ascents.

  Args:
    eulerian_numbers: Pre-computed ndarray of Eulerian numbers, of size at least
      (k + 1) x (k + 1).
    k: Integer number of ascents.
  """
  add_ascents = compute_add_ascent_indices(eulerian_numbers, k)
  d = len(add_ascents)
  permutation = np.ones(1)
  # i ranges over numbers to insert
  for i in range(2, d + 1):
    if add_ascents[i - 1]:
      insertion_options = list(
          np.where(permutation[1:] < permutation[:-1])[0] + 1
      ) + [i - 1]
    else:
      insertion_options = list(
          np.where(permutation[1:] > permutation[:-1])[0] + 1
      ) + [0]
    insert_idx = np.random.choice(insertion_options, 1)[0]
    permutation = np.insert(permutation, min(len(permutation), insert_idx), i)
  return permutation.astype(int)


def phi(x):
  """Returns phi(x), where phi is Stanley's bijection.

  Args:
    x: Float in (0, 1).

  Returns:
    phi(x), where phi is the function described on the first page of
    https://math.mit.edu/~rstan/pubs/pubfiles/34a.pdf.
  """
  d = len(x)
  new_x = np.zeros(d + 1)
  new_x[1:] = x
  x_diffs = new_x[:-1] - new_x[1:]
  y = x_diffs + (x_diffs < 0)
  return y


def sample_slice_index(eulerian_numbers, k):
  """Returns a slice index of the cube, sampled proportionally to its volume.

  Args:
    eulerian_numbers: Pre-computed matrix of Eulerian numbers, of size at least
      (k + 1) x (k + 1).
    k: Integer number of ascents.
  """
  slices = eulerian_numbers[-1, :k]
  weights = slices / np.sum(slices)
  return np.random.choice(len(slices), 1, p=weights)[0]


def sample_fundamental_simplex(d):
  """Returns a uniform random sample from the fundamental simplex.

  Args:
    d: Integer dimension of the fundamental simplex.
  """
  convex_combination_weights = stats.dirichlet.rvs([1] * (d + 1))
  fundamental_simplex_vertices = np.zeros((d + 1, d))
  for i in range(1, d + 1):
    fundamental_simplex_vertices[i - 1, -i:] = 1
  return np.sum(
      np.transpose(convex_combination_weights) * fundamental_simplex_vertices,
      axis=0,
  )


def sample_sum_ball(eulerian_numbers, k):
  """Returns a uniform sample from the induced norm ball for sum.

  Args:
    eulerian_numbers: Pre-computed matrix of Eulerian numbers, of size at least
      (k + 1) x (k + 1).
    k: Integer l_0 bound.
  """
  d = len(eulerian_numbers) - 1
  if d == 0:
    return np.random.uniform()
  slice_index = sample_slice_index(eulerian_numbers, k)
  sigma = sample_permutation_with_ascents(eulerian_numbers, slice_index)
  fundamental_simplex_sample = sample_fundamental_simplex(d)
  permuted_fundamental_simplex_sample = fundamental_simplex_sample[sigma - 1]
  return random_signs(phi(permuted_fundamental_simplex_sample))


def sum_mechanism(vector, eulerian_numbers, k, epsilon, num_samples):
  """Returns num_samples samples from the epsilon-DP induced norm mechanism for sum.

  Args:
    vector: The output will be a noisy version of Numpy array vector.
    eulerian_numbers: Pre-computed matrix of Eulerian numbers, of size at least
      (k + 1) x (k + 1).
    k: l_0 bound.
    epsilon: The output will be epsilon-DP for float epsilon. Assumes vector is
      1-sensitive with respect to the sum ball.
    num_samples: Number of samples to return.

  Returns:
    num_samples samples from the K-norm mechanism, as described in Section 4 of
    https://arxiv.org/abs/0907.3754, instantiated with the norm induced by sum.
  """
  d = len(vector)
  radii = np.random.gamma(shape=d + 1, scale=1 / epsilon, size=num_samples)
  samples = np.asarray(
      [sample_sum_ball(eulerian_numbers, k) for _ in range(num_samples)]
  )
  noises = radii[:, np.newaxis] * samples
  return vector + noises


def sample_orthant_num_positive(eulerian_numbers, k):
  """Samples an orthant class, as number of positive entries, by total volume.

  Args:
    eulerian_numbers: Pre-computed matrix of Eulerian numbers, of size at least
      (k + 1) x (k + 1).
    k: Integer l_0 bound.

  Returns:
    Integer number of positive entries in the sampled orthant.
  """
  d = len(eulerian_numbers) - 1
  eulerian_row_sums = np.sum(eulerian_numbers[:, :k], axis=1)
  factorials = special.factorial(np.arange(d + 1))
  weights = np.divide(
      np.multiply(eulerian_row_sums, eulerian_row_sums[::-1]),
      np.multiply(factorials, factorials[::-1]),
  )
  normalized_weights = weights / np.sum(weights)
  return np.random.choice(
      d + 1, p=np.array(normalized_weights, dtype="float64")
  )


def sample_from_orthant(eulerian_numbers, num_positive, k):
  """Samples from count ball orthant with num_positive positive entries.

  Args:
    eulerian_numbers: Pre-computed matrix of Eulerian numbers, of size at least
      (k + 1) x (k + 1).
    num_positive: Integer number of positive entries in orthant.
    k: Integer l_0 bound.

  Returns:
    Numpy array representing the sampled point.
  """
  d = len(eulerian_numbers) - 1
  if num_positive == 0:
    sample = -np.abs(sample_sum_ball(eulerian_numbers, k))
    return sample
  if num_positive == d:
    sample = np.abs(sample_sum_ball(eulerian_numbers, k))
    return sample
  combination_weight = stats.beta.rvs(a=num_positive, b=d - num_positive + 1)
  sample = np.zeros(d)
  negative_sample = -np.abs(
      sample_sum_ball(eulerian_numbers[: (d - num_positive) + 1], k)
  )
  sample[num_positive:] = (1 - combination_weight) * negative_sample
  if num_positive == 1:
    sample[0] = combination_weight
  else:
    w_weights = np.zeros(num_positive + 1)
    factorial_term = np.power(
        combination_weight, num_positive - 1
    ) / special.factorial(num_positive - 1)
    w_weights[:num_positive] = factorial_term * np.sum(
        eulerian_numbers[num_positive - 1, : k - 1]
    )
    # if num_positive > k, we may sample from the l1 = k hyperplane
    if num_positive > k:
      w_weights[num_positive] = (
          factorial_term * k * eulerian_numbers[num_positive - 1, k - 1]
      )
    normalized_w_weights = w_weights / np.sum(w_weights)
    idx = (
        np.random.choice(
            num_positive + 1, p=np.array(normalized_w_weights, dtype="float64")
        )
        + 1
    )
    if idx > num_positive:
      sigma = sample_permutation_with_ascents(
          eulerian_numbers[:num_positive], k - 1
      )
      fundamental_simplex_sample = sample_fundamental_simplex(num_positive - 1)
      permuted_fundamental_simplex_sample = fundamental_simplex_sample[
          sigma - 1
      ]
      positive_sample = np.abs(phi(permuted_fundamental_simplex_sample))
      sample[num_positive - 1] = combination_weight * (
          k - np.sum(positive_sample)
      )
    else:
      positive_sample = np.abs(
          sample_sum_ball(eulerian_numbers[:num_positive], k - 1)
      )
      sample[num_positive - 1] = combination_weight
    sample[: num_positive - 1] = combination_weight * positive_sample
  return sample[np.random.permutation(d)]


def sample_count_ball(eulerian_numbers, k):
  """Returns a uniform sample from the induced norm ball for count.

  Args:
    eulerian_numbers: Pre-computed matrix of Eulerian numbers, of size at least
      (k + 1) x (k + 1).
    k: Integer l_0 bound.
  """
  num_positive = sample_orthant_num_positive(eulerian_numbers, k)
  return sample_from_orthant(eulerian_numbers, num_positive, k)


def count_mechanism(vector, eulerian_numbers, k, epsilon, num_samples):
  """Returns a sample from the epsilon-DP induced norm mechanism for sum.

  Args:
    vector: The output will be a noisy version of Numpy array vector.
    eulerian_numbers: Pre-computed matrix of Eulerian numbers, of size at least
      (k + 1) x (k + 1).
    k: l_0 bound.
    epsilon: The output will be epsilon-DP for float epsilon. Assumes vector is
      1-sensitive with respect to the sum ball.
    num_samples: Number of samples to return.

  Returns:
    A sample from the K-norm mechanism, as described in Section 4 of
    https://arxiv.org/abs/0907.3754, instantiated with the norm induced by sum.
  """
  d = len(vector)
  radii = np.random.gamma(shape=d + 1, scale=1 / epsilon, size=num_samples)
  samples = np.asarray(
      [sample_count_ball(eulerian_numbers, k) for _ in range(num_samples)]
  )
  noises = radii[:, np.newaxis] * samples
  return vector + noises


def sample_from_simplex(vertices):
  """Returns point sampled uniformly from simplex given by vertices.

  Args:
    vertices: Shape-(d, d) ndarray of vertices of the simplex.

  Returns:
    Shape-(d,) ndarray for a single point sampled uniformly from the simplex.
  """
  d = len(vertices)
  convex_combination_weights = stats.dirichlet.rvs([1] * d)[0]
  return np.matmul(convex_combination_weights, vertices)


def external_to_internal_direct_sum(
    subpermutohedron_1, subpermutohedron_2, x_and_y
):
  """Returns the embedded version of vertex x_and_y.

  Args:
    subpermutohedron_1: Coordinates where the values in x will appear.
    subpermutohedron_2: Coordinates where the values in y will appear.
    x_and_y: List [x, y] where x is a list of numbers that represents a vertex
      of subpermutohedron_1 and y is a list of numbers that represents a vertex
      of subpermutohedron_2. len(x[0]) + len(y[0]) = len(subpermutohedron_1) +
      len(subpermutohedron_2) = d.

  Returns:
    Shape-(d,) ndarray of the embedded version of x_and_y where the values of x
    appear in subpermutohedron_1 and the values of y appear in
    subpermutohedron_2.
  """
  d = len(x_and_y[0]) + len(x_and_y[1])
  embedded = np.zeros(d)
  for i, index_1 in enumerate(subpermutohedron_1):
    embedded[index_1] = x_and_y[0][i]
  for i, index_2 in enumerate(subpermutohedron_2):
    embedded[index_2] = x_and_y[1][i]
  return embedded


def compute_face_class_weights(d):
  """Returns the face class weights associated with dimension d.

  Args:
    d: int for the dimension of the ball.

  Returns:
    Shape-(d,) ndarray of face class weights, each of which is proportional to
    the combined volumes of the pyramids in that class, except for the 0th
    index, which is left empty.
  """
  # num_faces, small_face_volume, altitude, and pyramid_weight_vector
  # respectively correspond to the variables M, V, H, and W used in the weight
  # computation in the paper.
  num_faces = np.zeros(d)
  small_face_volumes = np.zeros(d)
  pyramid_heights = np.zeros(d)
  face_class_weights = np.zeros(d)
  for j in range(1, d):
    num_faces[j] = special.binom(d, j)
    small_face_volumes[j] = (j ** (j - 1.5)) * ((d - j) ** (d - j - 1.5))
    pyramid_heights[j] = 0.5 * np.sqrt(j * (d - j) ** 2 + (d - j) * (j**2))
    face_class_weights[j] = (
        num_faces[j] * small_face_volumes[j] * pyramid_heights[j]
    )
  normalized_face_class_weights = face_class_weights / np.sum(
      face_class_weights
  )
  return normalized_face_class_weights


def sample_simplex_from_star_decomposition(d):
  """Returns simplex sampled from the star decomposition of the permutohedron.

  Args:
    d: int for the dimension of the ball.

  Returns:
    Shape-(d,d) ndarray of vertices of (d-1)-simplex sampled from the star
    decomposition, describing a simplex in a (d-1)-dimensional affine subspace.
  """
  if d == 1:
    return [[0]]
  face_class_weights = compute_face_class_weights(d)
  simplex_1, simplex_2, subpermutohedron_1, subpermutohedron_2 = (
      sample_simplices_and_embedding(face_class_weights)
  )
  simplex_3 = sample_simplex_3(
      simplex_1, simplex_2, subpermutohedron_1, subpermutohedron_2
  )
  center_of_chpd = (d - 1) / 2 * np.ones(d)
  simplex_3.append(center_of_chpd)
  return np.asarray(simplex_3)


def sample_simplices_and_embedding(face_class_weights):
  """Returns simplices and subpermutohedra relevant to a sampled face class.

  Args:
    face_class_weights: Shape-(d,) ndarray of the (normalized) face class
      weights. This is assumed to be generated by compute_face_class_weights and
      thus be 1-indexed, i.e., normalized_face_class_weights[0] = 0 is a dummy
      entry.

  Returns:
    simplex_1: ndarray of vertices of (|subpermutohedron_1|-1)-simplex sampled
      from the first subpermutohedron.
    simplex_2: ndarray of vertices of (d-|subpermutohedron_1|-1)-simplex sampled
      from the second subpermutohedron.
    subpermutohedron_1: ndarray of the coordinates of the first
      subpermutohedron.
    subpermutohedron_2: ndarray of the coordinates of the second
      subpermutohedron.
  """
  d = len(face_class_weights)
  subpermutohedron_1_size = (
      np.random.choice(d - 1, 1, p=face_class_weights[1:])[0] + 1
  )
  subpermutohedron_2_size = d - subpermutohedron_1_size
  random_permutation = np.random.permutation(d)
  subpermutohedron_1 = random_permutation[:subpermutohedron_1_size]
  subpermutohedron_2 = random_permutation[subpermutohedron_1_size:]
  simplex_1 = sample_simplex_from_star_decomposition(subpermutohedron_1_size)
  simplex_2 = sample_simplex_from_star_decomposition(subpermutohedron_2_size)
  subpermutohedron_1_identity = np.ones(len(subpermutohedron_1))
  simplex_1 = (
      simplex_1 + (d - len(subpermutohedron_1)) * subpermutohedron_1_identity
  )
  return simplex_1, simplex_2, subpermutohedron_1, subpermutohedron_2


def sample_simplex_3(
    simplex_1, simplex_2, subpermutohedron_1, subpermutohedron_2
):
  """Returns simplex sampled from a triangulation of simplex_1 and simplex_2.

  Args:
    simplex_1: ndarray of vertices of simplex sampled from the first
      subpermutohedron. Each vertex has the same dimension as
      subpermutohedron_1.
    simplex_2: ndarray of vertices of simplex sampled from the second
      subpermutohedron. Each vertex has the same dimension as
      subpermutohedron_2.
    subpermutohedron_1: Ndarray of the coordinates of the first
      subpermutohedron.
    subpermutohedron_2: Ndarray of the coordinates of the second
      subpermutohedron. subpermutohedron_1 has k coordinates, and
      subpermutohedron_2 has d-k coordinates.

  Returns:
    List of vertices of (d-2)-simplex sampled from the triangulation.
  """
  d = len(subpermutohedron_1) + len(subpermutohedron_2)
  # 0 in type_vector means t_simplex_1, 1 means t_simplex_2
  type_vector = np.zeros(d - 2)
  indices_of_t_simplex_2 = np.random.permutation(d - 2)[
      len(subpermutohedron_1) - 1 :
  ]
  type_vector[indices_of_t_simplex_2] = 1
  simplex_3 = [[simplex_1[0], simplex_2[0]]]
  simplex_1_index = 0
  simplex_2_index = 0
  for i in range(d - 2):
    if type_vector[i] == 0:
      simplex_1_index += 1
    else:
      simplex_2_index += 1
    simplex_3.append([simplex_1[simplex_1_index], simplex_2[simplex_2_index]])
  simplex_3_embedded = []
  for x_and_y in simplex_3:
    simplex_3_embedded.append(
        external_to_internal_direct_sum(
            subpermutohedron_1, subpermutohedron_2, x_and_y
        )
    )
  return simplex_3_embedded


def sample_vote_ball(d):
  """Returns point sampled uniformly from the d-dimensional vote unit ball.

  Args:
    d: int for the dimension of the ball.

  Returns:
    A list of length d representing a single point sampled uniformly from the
    vote unit ball, i.e., the cylinder of the permutohedron.
  """
  sampled_simplex = sample_simplex_from_star_decomposition(d)
  sample_from_cylinder_end = sample_from_simplex(sampled_simplex)
  cylinder_shift = (d - 1) * np.ones(d)
  reflected_sample_from_cylinder_end = sample_from_cylinder_end - cylinder_shift
  t = np.random.uniform()
  sampled_point_on_cylinder = (
      t * sample_from_cylinder_end
      + (1 - t) * reflected_sample_from_cylinder_end
  )
  return sampled_point_on_cylinder


def vote_mechanism(vector, epsilon):
  """Returns a sample from the epsilon-DP induced norm mechanism for vote.

  Args:
    vector: The output will be a noisy version of Numpy array vector.
    epsilon: The output will be epsilon-DP for float epsilon. Assumes vector is
      1-sensitive with respect to the vote ball.

  Returns:
    A sample from the K-norm mechanism, as described in Section 4 of
    https://arxiv.org/abs/0907.3754, instantiated with the norm induced by vote.
  """
  d = len(vector)
  radius = np.random.gamma(shape=d + 1, scale=1 / epsilon)
  sample = sample_vote_ball(d)
  noise = radius * sample
  return vector + noise
