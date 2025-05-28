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

"""Deterministic input pipeline."""

import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf

from mesh_diffusion.latents import dataset as ds
from mesh_diffusion.latents import io_utils as mio

field_conv_args = ['hei_mass_', 'hei_neigh_', 'logs_', 'xport_', 'bases_']
field_pool_args = ['unpool_adj_', 'unpool_vals_']
field_clip_flags = [0, 1, 1, 1, 0]


def get_geom_data(config):
  """Get geom data."""
  geom_record_path = config.geom_record_path + '/' + config.obj_name + '/'
  geom_data, shape_keys = ds.load_shape_tfrecords(geom_record_path + 'train/')
  gtest_data, tshape_keys = ds.load_shape_tfrecords(geom_record_path + 'test/')
  geom_data = geom_data.map(
      ds.parser(shape_keys), num_parallel_calls=tf.data.AUTOTUNE
  )
  gtest_data = gtest_data.map(
      ds.parser(tshape_keys), num_parallel_calls=tf.data.AUTOTUNE
  )
  return geom_data, gtest_data


def get_decoder_data(geom):
  """Get decoder data."""
  vert_inds = jnp.asarray(geom['vert_map'].numpy()).astype(jnp.int32)
  pix_tris = jnp.asarray(geom['pix_tris'].numpy()).astype(jnp.int32)
  pix_bary = jnp.asarray(geom['pix_bary'].numpy()).astype(jnp.float32)
  pix_logs = jnp.asarray(geom['pix_logs'].numpy()).astype(jnp.float32)
  valid_mask = jnp.asarray(geom['valid_ind'].numpy()).astype(jnp.float32)
  geom_i = jnp.asarray(geom['I'].numpy()).astype(jnp.int32)
  geom_j = jnp.asarray(geom['J'].numpy()).astype(jnp.int32)

  if 'tex_dim' in geom:
    tex_dim = jnp.asarray(geom['tex_dim'].numpy()).astype(jnp.int32)
  else:
    tex_dim = jnp.asarray([1024, 1024], dtype=jnp.int32)
    if jnp.ndim(geom_i) > 1:
      tex_dim = jnp.tile(tex_dim[None, Ellipsis], (geom_i.shape[0], 1))
  return (vert_inds, pix_tris, pix_bary, pix_logs,
          valid_mask, geom_i, geom_j, tex_dim)


def compose_labels(var_a, var_b):
  """Compose."""
  return var_a[var_b]


def get_conv_data(config, geom, labels=None):
  """Get conv data."""
  num_levels = config.num_levels

  # Get DDM model (unet) args
  k_conv = config.k_conv

  conv_args = field_conv_args
  pool_args = field_pool_args
  clip_flags = field_clip_flags

  geom_data = []

  for a in range(len(conv_args)):
    hei_list = []
    ag = conv_args[a]
    if not config.use_bases and ag == 'bases_':
      continue
    for l in range(num_levels):
      if clip_flags[a] == 0:
        hei_list.append(jnp.asarray(geom[ag + '{}'.format(l)].numpy()))
      else:
        if ag == 'logs_':
          hei_list.append(
              jnp.asarray(geom[ag + '{}'.format(l)].numpy()[Ellipsis, :k_conv, :])
          )
        else:
          hei_list.append(
              jnp.asarray(geom[ag + '{}'.format(l)].numpy()[Ellipsis, :k_conv])
          )

    geom_data.append(tuple(hei_list))

  for a in pool_args:
    hei_list = []
    for l in range(num_levels - 1):
      hei_list.append(jnp.asarray(geom[a + '{}_{}'.format(l, l + 1)].numpy()))
    geom_data.append(tuple(hei_list))

  if labels is not None:
    lab_pyr = []
    for l in range(num_levels):
      hei_ind = jnp.asarray(geom['hei_{}'.format(l)].numpy())
      lab_pyr.append(jax.vmap(compose_labels, (0, 0), 0)(labels, hei_ind))

    geom_data.append(tuple(lab_pyr))

  return tuple(geom_data)


def get_encoder_data(geom):
  """Get encoder data."""
  ring_logs = jnp.asarray(geom['ring_logs'].numpy()).astype(jnp.float32)
  ring_vals = jnp.asarray(geom['ring_vals'].numpy()).astype(jnp.float32)

  return (ring_logs, ring_vals)


def barycentric(p, a, b, c):
  """Barycentric."""

  v0 = b - a
  v1 = c - a
  v2 = p - a

  d00 = np.sum(v0 * v0, axis=-1)
  d01 = np.sum(v0 * v1, axis=-1)
  d11 = np.sum(v1 * v1, axis=-1)
  d20 = np.sum(v2 * v0, axis=-1)
  d21 = np.sum(v2 * v1, axis=-1)

  denom = d00 * d11 - d01 * d01

  v = (d11 * d20 - d01 * d21) / denom
  w = (d00 * d21 - d01 * d20) / denom
  u = 1.0 - v - w

  return u, v, w


def regular_samples(n):
  """Generate regular samples."""
  ng = ((n + 1) * (n + 2)) // 2
  t = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
  tg = np.zeros((ng, 2))

  p = 0

  for i in range(0, n + 1):
    for j in range(0, n + 1 - i):

      tg[p, 0] = (
          float(i) * t[0, 0] + float(j) * t[1, 0] + float(n - i - j) * t[2, 0]
      ) / float(n)

      tg[p, 1] = (
          float(i) * t[0, 1] + float(j) * t[1, 1] + float(n - i - j) * t[2, 1]
      ) / float(n)
      p = p + 1

  v1 = np.asarray([0.0, 0.0, 0.0])
  v2 = np.asarray([1.0, 0.0, 0.0])
  v3 = np.asarray([0.0, 1.0, 0.0])

  bcoords = []

  for j in range(tg.shape[0]):
    pt = np.asarray([tg[j, 0], tg[j, 1], 0.0])
    a, b, c = barycentric(pt, v1, v2, v3)
    bcoords.append([a, b, c])

  return np.asarray(bcoords, dtype=np.float32)


def map_mask(v, f, m, nodes, n=4):
  """Map mask."""
  uvw = regular_samples(n)
  fv = np.sum(uvw[None, Ellipsis, None] * v[f[:, None, :], :], axis=2)
  mv = np.tile(m[:, None], (1, fv.shape[1]))

  fv = np.reshape(fv, (-1, 3))
  mv = np.reshape(mv, (-1,))

  nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fv)

  _, nn_ind = nbrs.kneighbors(nodes)

  nn_ind = np.squeeze(nn_ind)

  return mv[nn_ind]


def map_signal(v, f, s, nodes, n=4):
  """Map signal."""
  uvw = regular_samples(n)

  fv = np.sum(uvw[None, Ellipsis, None] * v[f[:, None, :], :], axis=2)
  sv = np.sum(uvw[None, Ellipsis, None] * s[f[:, None, :], :], axis=2)

  fv = np.reshape(fv, (-1, 3))
  sv = np.reshape(sv, (-1, sv.shape[-1]))

  nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(fv)
  _, nn_ind = nbrs.kneighbors(nodes)
  nn_ind = np.squeeze(nn_ind)

  return sv[nn_ind]


def load_and_map_mask(config, nodes, inpaint=False):
  """Load and map mask."""
  if inpaint:
    v, f, m = mio.load_masked_ply(config.inpaint_labels)
    nz_ind = np.flatnonzero(m)
    m[nz_ind] = np.ones_like(m[nz_ind])
  else:
    v, f, m = mio.load_masked_ply(config.obj_labels)
  return map_mask(v, f, m, nodes)


def load_spec(config):
  """Load spec from config."""
  spec_data = mio.load_npz(config.spec)
  v = spec_data['nodes']
  spec_freq = spec_data['spec_freq']
  spec_bases = spec_data['spec_bases']

  return v, spec_freq, spec_bases


def compute_eig_embed(spec_freq, spec_bases, top_k):
  """Compute eig embed."""
  return spec_bases[:, 1 : (top_k + 1)] / (
      np.sqrt(np.abs(spec_freq))[None, 1 : (top_k + 1)] + 1.0e-6
  )


def compute_hks(spec_freq, spec_bases, features):
  """Compute hks."""
  scale = jnp.logspace(-2.0, 0.0, num=features, dtype=spec_bases.dtype)

  return jnp.sum(
      jnp.exp(-1.0 * scale[:, None] * spec_freq[None, :])[None, Ellipsis]
      * (spec_bases * spec_bases)[:, None, :],
      axis=-1,
  )


def load_and_map_spec(config, nodes):
  """Load and map spec."""
  features = config.spec_features
  v, spec_freq, spec_bases = load_spec(config)
  if config.spec_type == 'hks':
    signal = compute_hks(spec_freq, spec_bases, features)
  else:
    signal = compute_eig_embed(spec_freq, spec_bases, features)

  nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(v)

  _, nn_ind = nbrs.kneighbors(nodes)

  nn_ind = np.squeeze(nn_ind)

  return signal[nn_ind, :]


def load_cluster_map_spec(config, nodes):
  """Load cluster map spec."""
  v, spec_freq, spec_bases = load_spec(config)
  signal = compute_eig_embed(spec_freq, spec_bases, 8)
  # K-means
  kmeans = KMeans(
      n_clusters=config.num_clust, init='random', n_init=10, verbose=True
  ).fit(signal)
  labels = kmeans.labels_
  centroids = kmeans.cluster_centers_

  nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(v)
  _, nn_ind = nbrs.kneighbors(nodes)
  nn_ind = np.squeeze(nn_ind)

  return labels[nn_ind], centroids


def get_nodes(geom):
  """Get nodes."""
  nodes = jnp.asarray(geom['nodes'].numpy()).astype(jnp.float32)

  return nodes
