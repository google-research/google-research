# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from . import nn
from . import utils

from absl import logging
import numpy as np
import tensorflow as tf


class MlpBlock(nn.Module):

  def __init__(self, dim, hdim_factor, init_scale, name=None):
    super(MlpBlock, self).__init__(name=name)
    self.dim = dim
    hdim = dim * hdim_factor

    with self.name_scope:
      self.proj_in = nn.Dense(dim, hdim, name='proj_in')
      self.proj_out = nn.Dense(
          hdim, dim, init_scale=init_scale, name='proj_out')

  @nn.Module.with_name_scope
  def __call__(self, x):
    in_shape = x.shape
    assert in_shape[-1] == self.dim
    x = self.proj_in(x)
    x = nn.nonlinearity(x)
    x = self.proj_out(x)
    assert x.shape == in_shape
    return x


class AttnBlock(nn.Module):

  def __init__(self, dim, axis, num_heads, masked, init_scale, name=None):
    super(AttnBlock, self).__init__(name=name)
    self.dim = dim
    self.axis = axis
    self.masked = masked
    with self.name_scope:
      self.q = nn.Dense(dim, [num_heads, dim // num_heads], name='q')
      self.k = nn.Dense(dim, [num_heads, dim // num_heads], name='k')
      self.v = nn.Dense(dim, [num_heads, dim // num_heads], name='v')
      self.proj_out = nn.Dense(dim, dim, init_scale=init_scale, name='proj_out')

  @nn.Module.with_name_scope
  def __call__(self, x):
    in_shape = x.shape
    assert in_shape[-1] == self.dim
    q = self.q(x)
    k = self.k(x)
    v = self.v(x)
    a = nn.attn_nd(
        q, k, v, time_axis=self.axis, feat_axis=-1, masked=self.masked)
    assert a.shape == q.shape == k.shape == v.shape
    x = self.proj_out(tf.reshape(a, in_shape))
    assert x.shape == in_shape
    return x


class TransformerBlock(nn.Module):

  def __init__(self,
               dim,
               attn_axis,
               num_heads,
               masked,
               hdim_factor,
               res_init_scale,
               name=None):
    super(TransformerBlock, self).__init__(name=name)
    self.dim = dim
    with self.name_scope:
      self.attn = AttnBlock(
          dim=dim,
          axis=attn_axis,
          num_heads=num_heads,
          masked=masked,
          init_scale=res_init_scale,
          name='attn')
      self.mlp = MlpBlock(
          dim=dim,
          hdim_factor=hdim_factor,
          init_scale=res_init_scale,
          name='mlp')
      self.layernorm_attn = nn.LayerNorm(dim=dim, name='layernorm_attn')
      self.layernorm_mlp = nn.LayerNorm(dim=dim, name='layernorm_mlp')

  @nn.Module.with_name_scope
  def __call__(self, x, dropout):
    in_shape = x.shape
    assert in_shape[-1] == self.dim
    x += tf.nn.dropout(self.attn(self.layernorm_attn(x)), rate=dropout)
    x += tf.nn.dropout(self.mlp(self.layernorm_mlp(x)), rate=dropout)
    assert x.shape == in_shape
    return x


def emb_init(std):

  def fn(shape, dtype=None, partition_info=None):
    dim = int(shape[-1])
    return tf.random.normal(
        shape, mean=0.0, stddev=std / np.sqrt(dim), dtype=dtype)

  return fn


class Transformer3d(nn.Module):

  def __init__(self,
               img_height,
               img_width,
               img_channels,
               num_embs,
               emb_dim,
               emb_init_scale,
               num_heads,
               hdim_factor,
               res_init_scale,
               num_exterior_layers,
               num_outer_layers,
               num_inner_layers,
               logits_init_scale=1e-10,
               name=None):
    super(Transformer3d, self).__init__(name=name)

    self.img_channels = img_channels
    self.emb_dim = emb_dim
    self.num_embs = num_embs

    kwargs = dict(
        dim=emb_dim,
        num_heads=num_heads,
        hdim_factor=hdim_factor,
        res_init_scale=res_init_scale)
    make_col_block = functools.partial(TransformerBlock, attn_axis=1, **kwargs)
    make_row_block = functools.partial(TransformerBlock, attn_axis=2, **kwargs)

    with self.variable_scope:
      self.pos_embs_h = tf.get_variable(
          'pos_embs_h',
          shape=[img_height, emb_dim],
          initializer=emb_init(emb_init_scale))
      self.pos_embs_w = tf.get_variable(
          'pos_embs_w',
          shape=[img_width, emb_dim],
          initializer=emb_init(emb_init_scale))

    with self.name_scope:

      self.exterior_input_conv = nn.Conv2d(
          img_channels * 2, emb_dim, name='exterior_input_conv')

      # unmasked attention for previous slices
      self.exterior_layers = []
      assert num_exterior_layers % 2 == 0
      for i in range(num_exterior_layers // 2):
        logging.info('creating exterior layer {}'.format(2 * i))
        self.exterior_layers.append(
            make_row_block(masked=False, name='exterior_row_{}'.format(i)))
        logging.info('creating exterior layer {}'.format(2 * i + 1))
        self.exterior_layers.append(
            make_col_block(masked=False, name='exterior_col_{}'.format(i)))

      # AR model for the current slice
      self.transformer2d = Transformer2d(
          img_height=img_height,
          img_width=img_width,
          num_embs=num_embs,
          emb_dim=emb_dim,
          emb_init_scale=emb_init_scale,
          num_heads=num_heads,
          hdim_factor=hdim_factor,
          res_init_scale=res_init_scale,
          num_outer_layers=num_outer_layers,
          num_inner_layers=num_inner_layers,
          logits_init_scale=logits_init_scale,
          name='transformer2d')

  def sample_fast(self, noise, cond, dropout):
    return self._sample(noise=noise, cond=cond, dropout=dropout, fast=True)

  def sample_slow(self, noise, cond, dropout):
    return self._sample(noise=noise, cond=cond, dropout=dropout, fast=False)

  @nn.Module.with_name_scope
  def _sample(self, noise, cond, dropout, fast):
    B, H, W, C, K = (int(noise.shape[0]), self.transformer2d.img_height,
                     self.transformer2d.img_width, self.img_channels,
                     self.transformer2d.num_embs)
    assert noise.shape == [B, H, W, C, K]

    def loop_body(i, x_bhwc):
      chn_mask = tf.equal(tf.range(C, dtype=tf.int32), i)
      noise_slice = tf.reduce_sum(
          noise * tf.cast(chn_mask[None, None, None, :, None], noise.dtype),
          axis=3)
      assert noise_slice.shape == [B, H, W, K]
      sampled_slice = self.sample_slice(
          noise_bhwk=noise_slice,
          x_bhwc=x_bhwc,
          slice_inds_b=tf.fill([B], i),
          cond=cond,
          dropout=dropout,
          fast=fast)
      assert sampled_slice.shape == [B, H, W]
      assert sampled_slice.dtype == tf.int32
      chn_mask_111c = tf.cast(chn_mask[None, None, None, :], x_bhwc.dtype)
      new_x_bhwc = ((1 - chn_mask_111c) * x_bhwc +
                    chn_mask_111c * sampled_slice[:, :, :, None])
      assert new_x_bhwc.shape == x_bhwc.shape and new_x_bhwc.dtype == tf.int32
      return [i + 1, new_x_bhwc]

    i0 = tf.constant(0, dtype=tf.int32)
    img0 = tf.zeros([B, H, W, C], dtype=tf.int32)
    _, img_final = tf.while_loop(  # loop over channels
        cond=lambda i, _: i < C,
        body=loop_body,
        loop_vars=[i0, img0],
        shape_invariants=[i0.shape, img0.shape],
        back_prop=False)
    assert img_final.shape == noise.shape[:-1]
    assert img_final.dtype == tf.int32
    return img_final

  def compute_logits(self, x_bhwc, cond, dropout):
    return self.compute_all_slice_logits(x_bhwc, cond=cond, dropout=dropout)

  @nn.Module.with_name_scope
  def compute_all_slice_logits(self, x_bhwc, cond, dropout):
    B, H, W, C, K = (int(x_bhwc.shape[0]), self.transformer2d.img_height,
                     self.transformer2d.img_width, self.img_channels,
                     self.transformer2d.num_embs)

    def one_slice_logits(slice_index):
      return self.compute_slice_logits(
          x_bhwc, tf.fill([B], slice_index), cond=cond, dropout=dropout)

    logits_cbhwk = tf.map_fn(
        one_slice_logits,
        tf.range(self.img_channels, dtype=tf.int32),
        dtype=tf.float32)
    assert logits_cbhwk.shape == [C, B, H, W, K]
    logits_bhwck = tf.transpose(logits_cbhwk, [1, 2, 3, 0, 4])
    assert logits_bhwck.shape == [B, H, W, C, K]
    return logits_bhwck

  @nn.Module.with_name_scope
  def compute_random_slice_nll(self,
                               x_bhwc,
                               cond,
                               dropout,
                               rand_slice_range=None):
    B, H, W, C, K = (int(x_bhwc.shape[0]), self.transformer2d.img_height,
                     self.transformer2d.img_width, self.img_channels,
                     self.transformer2d.num_embs)

    if rand_slice_range is None:
      rand_slice_range = [0, C]
    else:
      assert len(rand_slice_range) == 2
    rand_slice_inds = tf.random.uniform(
        shape=[B],
        minval=rand_slice_range[0],
        maxval=rand_slice_range[1],
        dtype=tf.int32)

    logits = self.compute_slice_logits(
        x_bhwc, rand_slice_inds, cond=cond, dropout=dropout)
    labels = self._extract_cur_chn_slice(x_bhwc, rand_slice_inds)
    assert logits.shape == labels.shape + [K] == [B, H, W, K]
    nlls = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=labels)
    assert nlls.shape == [B, H, W]
    return nlls

  def compute_slice_logits(self, x_bhwc, slice_inds_b, cond, dropout):
    B, H, W, C, K = (int(x_bhwc.shape[0]), self.transformer2d.img_height,
                     self.transformer2d.img_width, self.img_channels,
                     self.transformer2d.num_embs)
    assert x_bhwc.shape == [B, H, W, C]
    # condition
    cond = self._compute_conditioning(
        x_bhwc, slice_inds_b, cond=cond, dropout=dropout)
    # process current slice
    logits = self.transformer2d.compute_logits(
        self._extract_cur_chn_slice(x_bhwc, slice_inds_b),
        cond=cond,
        dropout=dropout)
    assert logits.shape == [B, H, W, K]
    return logits

  def sample_slice(self, noise_bhwk, x_bhwc, slice_inds_b, cond, dropout, fast):
    cond = self._compute_conditioning(
        x_bhwc, slice_inds_b, cond=cond, dropout=dropout)
    return (self.transformer2d.sample_fast
            if fast else self.transformer2d.sample_slow)(
                noise_bhwk, cond=cond, dropout=dropout)

  def _compute_conditioning(self, x_bhwc, slice_inds_b, cond, dropout):
    B, H, W, C, D = (int(x_bhwc.shape[0]), self.transformer2d.img_height,
                     self.transformer2d.img_width, self.img_channels,
                     self.transformer2d.emb_dim)
    assert x_bhwc.shape == [B, H, W, C]
    assert x_bhwc.dtype in [tf.int32, tf.int64]

    # Embed previous slices
    h = self._mask_prev_chn_slices(x_bhwc, slice_inds_b, fill_value=0)
    assert h.shape == [B, H, W, C] and h.dtype == tf.int32
    h = tf.cast(h, tf.float32) / 255.0

    mask_bc = tf.equal(
        tf.range(C, dtype=tf.int32)[None, :], slice_inds_b[:, None])
    mask_bhwc = tf.cast(mask_bc[:, None, None, :], tf.float32) * tf.ones_like(h)

    h = self.exterior_input_conv(tf.concat([h, mask_bhwc], axis=-1))
    assert h.shape == [B, H, W, D]

    # Position embeddings
    h += self.pos_embs_h[None, :, None, :]
    h += self.pos_embs_w[None, None, :, :]
    assert h.shape == [B, H, W, D]

    # Conditioning info
    if cond is not None:
      assert cond.shape == h.shape
      h += cond

    # Attention layers
    for layer in self.exterior_layers:
      h = layer(h, dropout=dropout)
    assert h.shape == [B, H, W, D]

    return h

  def _extract_cur_chn_slice(self, x_bhwc, slice_inds_b):
    # x_bhwc[:,:,:,slice_inds_b]
    B, H, W, C = (int(x_bhwc.shape[0]), self.transformer2d.img_height,
                  self.transformer2d.img_width, self.img_channels)
    assert x_bhwc.dtype in [tf.int32, tf.int64] and x_bhwc.shape == [B, H, W, C]
    assert slice_inds_b.dtype == tf.int32 and slice_inds_b.shape == [B]

    mask_bc = tf.equal(
        tf.range(C, dtype=tf.int32)[None, :], slice_inds_b[:, None])
    assert mask_bc.shape == [B, C]
    mask_bc = tf.cast(mask_bc, x_bhwc.dtype)

    x_sliced_bhw = tf.reduce_sum(x_bhwc * mask_bc[:, None, None, :], axis=-1)
    assert x_sliced_bhw.shape == [B, H, W]
    return x_sliced_bhw

  def _mask_prev_chn_slices(self, x_bhwc, slice_inds_b, fill_value):
    B, H, W, C = (int(x_bhwc.shape[0]), self.transformer2d.img_height,
                  self.transformer2d.img_width, self.img_channels)
    assert x_bhwc.dtype in [tf.int32, tf.int64] and x_bhwc.shape == [B, H, W, C]
    assert slice_inds_b.dtype == tf.int32 and slice_inds_b.shape == [B]

    prevmask_bc = tf.less(
        tf.range(C, dtype=tf.int32)[None, :], slice_inds_b[:, None])
    assert prevmask_bc.shape == [B, C]
    prevmask_b11c = tf.cast(prevmask_bc, x_bhwc.dtype)[:, None, None, :]
    assert len(prevmask_b11c.shape) == len(x_bhwc.shape)

    return prevmask_b11c * x_bhwc + (1 - prevmask_b11c) * fill_value


class Transformer2d(nn.Module):
  """Single-channel images only."""

  def __init__(self,
               img_height,
               img_width,
               num_embs,
               emb_dim,
               emb_init_scale,
               num_heads,
               hdim_factor,
               res_init_scale,
               num_outer_layers,
               num_inner_layers,
               logits_init_scale=1e-10,
               name=None):
    super(Transformer2d, self).__init__(name=name)

    self.img_height = img_height
    self.img_width = img_width
    self.num_embs = num_embs
    self.emb_dim = emb_dim

    with self.variable_scope:
      # Embeddings
      self.embs = tf.get_variable(
          'embs',
          shape=[num_embs, emb_dim],
          initializer=emb_init(emb_init_scale))
      # Position embeddings
      self.pos_embs_h = tf.get_variable(
          'pos_embs_h',
          shape=[img_height, emb_dim],
          initializer=emb_init(emb_init_scale))
      self.pos_embs_w = tf.get_variable(
          'pos_embs_w',
          shape=[img_width, emb_dim],
          initializer=emb_init(emb_init_scale))

    with self.name_scope:
      # Transformer layers
      kwargs = dict(
          dim=emb_dim,
          num_heads=num_heads,
          hdim_factor=hdim_factor,
          res_init_scale=res_init_scale)
      make_col_block = functools.partial(
          TransformerBlock, attn_axis=1, **kwargs)
      make_row_block = functools.partial(
          TransformerBlock, attn_axis=2, **kwargs)

      self.outer_layers = []
      assert num_outer_layers % 2 == 0
      for i in range(num_outer_layers // 2):
        logging.info('creating outer layer {}'.format(2 * i))
        self.outer_layers.append(
            make_row_block(masked=False, name='outer_row_{}'.format(i)))
        logging.info('creating outer layer {}'.format(2 * i + 1))
        self.outer_layers.append(
            make_col_block(masked=True, name='outer_col_{}'.format(i)))

      self.inner_layers = []
      for i in range(num_inner_layers):
        logging.info('creating inner layer {}'.format(i))
        self.inner_layers.append(
            make_row_block(masked=True, name='inner_row_{}'.format(i)))
      self.final_ln = nn.LayerNorm(dim=emb_dim, name='final_ln')
      self.final_hdim = emb_dim
      self.final_dense = nn.Dense(
          emb_dim, num_embs, init_scale=logits_init_scale, name='final_dense')

  @nn.Module.with_name_scope
  def compute_logits(self, x, cond, dropout):
    """Computes logits for each pixel in the image `x`."""
    assert x.dtype in [tf.int32, tf.int64]
    assert x.shape[1:] == [self.img_height, self.img_width]
    bs = x.shape[0]
    pos_embs = self.get_pos_embs()[None]
    # image -> embeddings
    h = tf.gather(self.embs, x)
    assert h.shape == [bs, self.img_height, self.img_width, self.emb_dim]
    # embeddings -> last hidden layer
    u = self._upper_context(h, pos_embs=pos_embs, cond=cond, dropout=dropout)
    h = self._row_autoregressive(
        h, u=u, pos_embs=pos_embs, cond=cond, dropout=dropout)
    assert h.shape == [bs, self.img_height, self.img_width, self.final_hdim]
    # last hidden layer -> logits
    logits = self._final_to_logits(h)
    assert logits.shape == [bs, self.img_height, self.img_width, self.num_embs]
    return logits

  # === Sampling ===

  @nn.Module.with_name_scope
  def sample_slow(self, noise, cond, dropout):
    """Naive sampling implementation."""
    assert noise.shape[1:] == [self.img_height, self.img_width, self.num_embs]
    bs = noise.shape[0]
    h, w = self.img_height, self.img_width

    def _pixel_loop_body(i, img_bhw):
      """Loop over all pixels: `i` indexes into the flattened image."""
      r = i // w
      c = i % w
      # sample this one pixel
      logits_bhwk = self.compute_logits(img_bhw, cond=cond, dropout=dropout)
      assert logits_bhwk.shape == noise.shape
      samples_b = tf.argmax(
          logits_bhwk[:, r, c, :] + noise[:, r, c, :],
          axis=1,
          output_type=tf.int32)
      assert samples_b.shape == [bs]
      # mask of ones at (:,row,col), zeros elsewhere
      mask_1hw = tf.reshape(tf.equal(tf.range(h * w), c + r * w), [1, h, w])
      mask_1hw = tf.cast(mask_1hw, tf.int32)
      # set img_bhw[:,row,col] <- samples_b
      newimg_bhw = ((1 - mask_1hw) * img_bhw +
                    mask_1hw * samples_b[:, None, None])
      return [i + 1, newimg_bhw]

    i0 = tf.constant(0, dtype=tf.int32)
    img0 = tf.zeros([bs, self.img_height, self.img_width], tf.int32)
    _, img_final = tf.while_loop(
        cond=lambda i, _: i < h * w,
        body=_pixel_loop_body,
        loop_vars=[i0, img0],
        shape_invariants=[i0.shape, img0.shape],
        back_prop=False)
    assert img_final.shape == noise.shape[:-1]
    assert img_final.dtype == tf.int32
    return img_final

  @nn.Module.with_name_scope
  def sample_fast(self, noise, cond, dropout):
    """Faster sampling implementation via caching upper context."""
    assert noise.shape[1:] == [self.img_height, self.img_width, self.num_embs]
    bs = int(noise.shape[0])
    h, w = self.img_height, self.img_width

    pos_embs = self.get_pos_embs()

    def _sample_one_row_slow(row_noise, row_u, row_pos_embs, row_cond):
      """Samples one row."""
      assert row_noise.shape == [bs, 1, self.img_width, self.num_embs]
      if row_cond is not None:
        assert len(row_cond.shape) == 4 and row_cond.shape[:3] == [bs, 1, w]

      def _col_loop_body(c, row_b1w):
        """Loop over columns of a fixed row (c is the current column)."""
        c_mask = tf.equal(tf.range(w), c)  # column indicator for slicing
        # compute logits for this row
        row_logits_b1wk = self._final_to_logits(
            self._row_autoregressive(
                tf.gather(self.embs, row_b1w),
                u=row_u,
                pos_embs=row_pos_embs,
                cond=row_cond,
                dropout=dropout))
        assert row_logits_b1wk.shape == row_noise.shape
        # sample this one pixel in this row
        noisy_logits_b1wk = row_logits_b1wk + row_noise
        c_mask_11w1 = tf.cast(c_mask[None, None, :, None], tf.float32)
        noisy_logits_bk = tf.reduce_sum(
            noisy_logits_b1wk * c_mask_11w1, axis=2)[:, 0, :]
        # note: noisy_logits_bk ==
        #   row_logits_b1wk[:,0,c,:] + row_noise[:,0,c,:]
        samples_b = tf.argmax(noisy_logits_bk, axis=1, output_type=tf.int32)
        assert samples_b.shape == [bs]
        # set row[:,0,col] <- samples_b
        c_mask_11w = tf.cast(c_mask[None, None, :], tf.int32)
        newrow_b1w = ((1 - c_mask_11w) * row_b1w +
                      c_mask_11w * samples_b[:, None, None])
        return [c + 1, newrow_b1w]

      c0 = tf.constant(0, dtype=tf.int32)
      row0 = tf.zeros([bs, 1, self.img_width], tf.int32)
      _, row_final = tf.while_loop(
          cond=lambda c, _: c < w,
          body=_col_loop_body,
          loop_vars=[c0, row0],
          shape_invariants=[c0.shape, row0.shape],
          back_prop=False)
      assert row_final.shape == row_noise.shape[:-1]
      assert row_final.dtype == tf.int32
      return row_final

    def _row_loop_body(r, img_bhw):
      # sample one row: compute upper context once per row
      u = self._upper_context(
          tf.gather(self.embs, img_bhw),
          cond=cond,
          pos_embs=pos_embs[None],
          dropout=dropout)
      # conditioned on u, sample this row
      r_mask = tf.equal(tf.range(h), r)  # row indicator for slicing
      r_mask_1h11 = tf.cast(r_mask[None, :, None, None], tf.float32)
      # slice_row(a) is the same as a[:, r, None, :, :] but works on TPUs
      slice_row = lambda a_: tf.reduce_sum(  # pylint: disable=g-long-lambda
          a_ * r_mask_1h11,
          axis=1,
          keepdims=True)
      sampled_row_b1w = _sample_one_row_slow(
          row_noise=slice_row(noise),
          row_u=slice_row(u),
          row_pos_embs=slice_row(pos_embs[None]),
          row_cond=slice_row(cond) if cond is not None else None)
      assert sampled_row_b1w.shape == [bs, 1, w]
      # fill in this new row
      r_mask_1h1 = tf.cast(r_mask[None, :, None], tf.int32)
      newimg_bhw = ((1 - r_mask_1h1) * img_bhw + r_mask_1h1 * sampled_row_b1w)
      assert newimg_bhw.shape == img_bhw.shape
      return [r + 1, newimg_bhw]

    # loop over rows
    r0 = tf.constant(0, dtype=tf.int32)
    img0 = tf.zeros([bs, self.img_height, self.img_width], tf.int32)
    _, img_final = tf.while_loop(
        cond=lambda r, _: r < h,
        body=_row_loop_body,
        loop_vars=[r0, img0],
        shape_invariants=[r0.shape, img0.shape],
        back_prop=False)
    assert img_final.shape == noise.shape[:-1]
    assert img_final.dtype == tf.int32
    return img_final

  def get_pos_embs(self):
    return self.pos_embs_h[:, None, :] + self.pos_embs_w[None, :, :]

  def _upper_context(self, h, pos_embs, cond, dropout):
    """Summarize information above the current pixel (outer layers)."""
    assert len(h.shape) == len(pos_embs.shape)
    assert h.shape[1:] == [self.img_height, self.img_width, self.emb_dim]
    assert cond is None or cond.shape == h.shape

    u = h + pos_embs
    if cond is not None:
      u += cond
    for block in self.outer_layers:
      u = block(u, dropout=dropout)
    u = nn.shift_down(u)

    assert u.shape == h.shape
    return u

  def _row_autoregressive(self, h, u, pos_embs, cond, dropout):
    """Autoregressive over columns, with no mixing over rows."""
    assert len(h.shape) == len(pos_embs.shape)
    assert h.shape[-1] == self.emb_dim and u.shape == h.shape
    assert cond is None or cond.shape == h.shape

    # aggregate context above and the pixel to the immediate left
    r = nn.shift_right(h) + u + pos_embs
    if cond is not None:
      r += cond

    # masked row attention layers
    for block in self.inner_layers:
      r = block(r, dropout=dropout)

    assert r.shape == h.shape[:3] + [self.final_hdim]
    return r

  def _final_to_logits(self, h):
    """Converts the final Transformer layer to logits."""
    assert len(h.shape) == 4 and h.shape[-1] == self.final_hdim
    h = self.final_ln(h)
    logits = self.final_dense(h)
    assert logits.shape == h.shape[:3] + [self.num_embs]
    return logits


class MultiChannelTransformer2d(nn.Module):

  def __init__(self, img_height, img_width, img_channels, name=None, **kwargs):
    super(MultiChannelTransformer2d, self).__init__(name=name)

    self.img_height = img_height
    self.img_width = img_width
    self.img_channels = img_channels

    with self.name_scope:
      self.model = Transformer2d(
          img_height=img_height * img_channels,
          img_width=img_width,
          name='transformer2d',
          **kwargs)

  @staticmethod
  def _to_single_channel(x):
    return utils.chans_to_rows(x)

  @staticmethod
  def _from_single_channel(x, channels):
    return utils.rows_to_chans(x, channels=channels)

  @property
  def num_embs(self):
    return self.model.num_embs

  @property
  def emb_dim(self):
    return self.model.emb_dim

  def _duplicate_cond_over_channels(self, cond):
    if cond is not None:
      assert len(cond.shape) == 4
      assert cond.shape[1:3] == [self.img_height, self.img_width]
      cond = self._to_single_channel(
          tf.tile(cond[:, :, :, None, :], [1, 1, 1, self.img_channels, 1]))
    return cond

  def compute_logits(self, x, cond, dropout):
    assert x.shape[1:] == [self.img_height, self.img_width, self.img_channels]
    logits = self._from_single_channel(
        self.model.compute_logits(
            self._to_single_channel(x[Ellipsis, None])[Ellipsis, 0],
            cond=self._duplicate_cond_over_channels(cond),
            dropout=dropout),
        channels=self.img_channels)
    assert logits.shape == x.shape + [self.model.num_embs]
    return logits

  @nn.Module.with_name_scope
  def _sample(self, noise, cond, dropout, fast):
    bs = noise.shape[0]
    x_shape = [bs, self.img_height, self.img_width, self.img_channels]
    assert noise.shape == x_shape + [self.model.num_embs]
    if cond is not None:
      assert len(cond.shape) == 4 and cond.shape[:3] == x_shape[:3]
    samples = self._from_single_channel(
        (self.model.sample_fast if fast else self.model.sample_slow)(
            noise=self._to_single_channel(noise),
            cond=self._duplicate_cond_over_channels(cond),
            dropout=dropout)[Ellipsis, None],
        channels=self.img_channels)[Ellipsis, 0]
    assert samples.shape == x_shape
    return samples

  def sample_fast(self, noise, cond, dropout):
    return self._sample(noise, cond, dropout, fast=True)

  def sample_slow(self, noise, cond, dropout):
    return self._sample(noise, cond, dropout, fast=False)
