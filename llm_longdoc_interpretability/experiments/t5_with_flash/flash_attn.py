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

"""Triton implementation of Flash Attention for T5 Models.

The entire subrepository is heavily inspired by the following two
implementations:
https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py
https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py


"flash_attn_v1" = vanilla FA;
"flash_attn_v2" = vanilla FA with virtual bias matrix; encoder-encoder self
attention
"flash_attn_v6" = block-sparse (FiD) FA; encoder-encoder self attention
"flash_attn_v7" = block-sparse (FiD) FA; encoder-decoder cross attention
"""

import torch
import triton
import triton.language as tl


@triton.jit
def max_fn(x, y):
  return tl.math.max(x, y)


# pylint: disable=invalid-name
@triton.heuristics({
    "EVEN_M": lambda args: args["M_CTX"] % args["BLOCK_M"] == 0,
    "EVEN_N": lambda args: args["N_CTX"] % args["BLOCK_N"] == 0,
})
@triton.jit
def _fwd_kernel_v1(
    Q,
    K,
    V,
    sm_scale,
    Bias,
    L,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    H,
    M_CTX,
    N_CTX,
    BLOCK_M,
    BLOCK_DMODEL,
    BLOCK_N,
    IS_CAUSAL,
    BIAS_TYPE,
    EVEN_M,
    EVEN_N,
):
  """Fwd kernel v1."""
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // H  # batch index
  off_h = off_hz % H  # head index

  q_offset = off_z * stride_qz + off_h * stride_qh
  k_offset = off_z * stride_kz + off_h * stride_kh
  v_offset = off_z * stride_vz + off_h * stride_vh

  offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_n = tl.arange(0, BLOCK_N)
  offs_d = tl.arange(0, BLOCK_DMODEL)

  q_ptrs = (
      Q + q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
  )
  k_ptrs = (
      K + k_offset + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk)
  )
  v_ptrs = (
      V + v_offset + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
  )

  b_ptrs = None  # avoid cider"s error
  if BIAS_TYPE == "matrix":
    b_ptrs = (
        Bias
        + off_z * stride_bz
        + off_h * stride_bh
        + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    )

  # initialize offsets
  offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_n = tl.arange(0, BLOCK_N)
  # initialize pointer to m and l
  m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
  acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
  # scale sm_scale by log_2(e) and use
  # 2^x instead of exp in the loop because CSE and LICM
  # don"t work as expected with `exp` in the loop
  qk_scale = sm_scale * 1.44269504

  if EVEN_M:
    q = tl.load(q_ptrs)
  else:
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < M_CTX), other=0.0)
  q = (q * qk_scale).to(tl.float16)

  # loop over k, v and update accumulator
  lo = 0
  hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX) if IS_CAUSAL else N_CTX
  for start_n in range(lo, hi, BLOCK_N):
    if EVEN_N:
      k = tl.load(k_ptrs + start_n * stride_kn)
      v = tl.load(v_ptrs + start_n * stride_vn)
    else:
      k = tl.load(
          k_ptrs + start_n * stride_kn,
          mask=(start_n + offs_n)[None, :] < N_CTX,
          other=0.0,
      )
      v = tl.load(
          v_ptrs + start_n * stride_vn,
          mask=(start_n + offs_n)[:, None] < N_CTX,
          other=0.0,
      )

    # -- compute qk ---
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)

    # needed to fix edge cases for softmax computation
    qk += tl.where((start_n + offs_n)[None, :] < N_CTX, 0, float("-inf"))
    if IS_CAUSAL:  # dao does afterwards
      qk += tl.where(
          offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf")
      ).to(tl.float16)

    if BIAS_TYPE != "none":
      bias = tl.load(
          b_ptrs + start_n * stride_bn,
          mask=(offs_m[:, None] < M_CTX)
          & ((start_n + offs_n)[None, :] < N_CTX),
          other=0.0,
      ).to(tl.float32)
      qk += bias * qk_scale

    # -- compute scaling constant ---
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))  # dao "lse_i" = trit "m_i"
    alpha = tl.math.exp2(m_i - m_i_new)  # dao "m_ij"  = trit "m_i_new"
    p = tl.math.exp2(qk - m_i_new[:, None])

    # -- scale and update acc --
    acc_scale = l_i * 0 + alpha  # workaround some compiler bug
    acc *= acc_scale[:, None]
    acc += tl.dot(
        p.to(tl.float16), v
    )  # while testing original float implementation

    # -- update m_i and l_i --
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new

  # write back l and m
  acc = acc / l_i[:, None]
  l_ptrs = L + off_hz * M_CTX + offs_m
  tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=(offs_m < M_CTX))

  out_ptrs = (
      Out
      + off_z * stride_oz
      + off_h * stride_oh
      + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
  )
  if EVEN_M:
    tl.store(out_ptrs, acc.to(tl.float16))
  else:
    tl.store(out_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M_CTX))


class _attention_v1(torch.autograd.Function):
  """Attention v1."""

  @staticmethod
  def forward(
      ctx, q, k, v, bias=None, causal=False, sm_scale=None, verbose=False
  ):
    """Computes transformer attention value.

    Args:
      ctx:
      q: matrix of queries
      k: matrix of keys
      v: matrix of values
      bias: bias tensor, possibly in different shapes according to type
      causal: boolean, using triangular causal attention or not
      sm_scale: what to scale the SM attention temperature by
      verbose: whether to print out relevant computing information

    Returns:
      computed transformer attention value
    """
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    q, k, v = [
        x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]
    ]  # this line from dao
    o = torch.empty_like(q)
    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    L = torch.empty(
        (q.shape[0] * q.shape[1], q.shape[2]),
        device=q.device,
        dtype=torch.float32,
    )

    if verbose:
      print("q", q.shape)
      print("k", k.shape)
      print("BLOCK_M", BLOCK_M, "BLOCK_N", BLOCK_N, "BLOCK_DMODEL", Lk)

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
      assert bias.dtype in [q.dtype, torch.float]
      assert bias.is_cuda
      assert bias.dim() == 4
      if bias.stride(-1) != 1:
        bias = bias.contiguous()
      bias_type = "matrix"

    bias_strides = (
        (bias.stride(0), bias.stride(1), bias.stride(2), bias.stride(3))
        if has_bias
        else (0, 0, 0, 0)
    )

    num_warps = 4 if Lk <= 64 else 8
    _fwd_kernel_v1[grid](
        q,
        k,
        v,
        sm_scale,
        bias,
        L,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        *bias_strides,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        q.shape[0],
        q.shape[1],
        q.shape[2],
        k.shape[2],
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=causal,
        BIAS_TYPE=bias_type,
        num_warps=num_warps,
        num_stages=4,
    )
    ctx.save_for_backward(q, k, v, o, L, bias)
    ctx.grid = grid
    ctx.sm_scale = sm_scale
    ctx.BLOCK_DMODEL = Lk
    ctx.causal = causal
    return o

  @staticmethod
  def backward(ctx, do):
    raise NotImplementedError("no backwards support yet")


@triton.heuristics({
    "EVEN_M": lambda args: args["M_CTX"] % args["BLOCK_M"] == 0,
    "EVEN_N": lambda args: args["N_CTX"] % args["BLOCK_N"] == 0,
})
@triton.jit
def _fwd_kernel_v2(
    Q,
    K,
    V,
    sm_scale,
    Bias,
    L,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    H,
    M_CTX,
    N_CTX,
    BLOCK_M,
    BLOCK_DMODEL,
    BLOCK_N,
    IS_CAUSAL,
    BIAS_TYPE,
    EVEN_M,
    EVEN_N,
):
  """Forward kernel v2."""
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // H  # batch index
  off_h = off_hz % H  # head index

  q_offset = off_z * stride_qz + off_h * stride_qh
  k_offset = off_z * stride_kz + off_h * stride_kh
  v_offset = off_z * stride_vz + off_h * stride_vh

  offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_n = tl.arange(0, BLOCK_N)
  offs_d = tl.arange(0, BLOCK_DMODEL)

  q_ptrs = (
      Q + q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
  )
  k_ptrs = (
      K + k_offset + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk)
  )
  v_ptrs = (
      V + v_offset + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
  )

  b_ptrs = None  # avoid cider"s error
  bias = None
  if BIAS_TYPE == "matrix":
    b_ptrs = (
        Bias
        + off_z * stride_bz
        + off_h * stride_bh
        + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    )

  # initialize offsets
  offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_n = tl.arange(0, BLOCK_N)
  # initialize pointer to m and l
  m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
  acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
  # scale sm_scale by log_2(e) and use
  # 2^x instead of exp in the loop because CSE and LICM
  # don"t work as expected with `exp` in the loop
  qk_scale = sm_scale * 1.44269504

  if EVEN_M:
    q = tl.load(q_ptrs)
  else:
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < M_CTX), other=0.0)
  q = (q * qk_scale).to(tl.float16)

  # loop over k, v and update accumulator
  lo = 0
  hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX) if IS_CAUSAL else N_CTX
  for start_n in range(lo, hi, BLOCK_N):
    # -- load k, v --
    if EVEN_N:
      k = tl.load(k_ptrs + start_n * stride_kn)
      v = tl.load(v_ptrs + start_n * stride_vn)
    else:
      k = tl.load(
          k_ptrs + start_n * stride_kn,
          mask=(start_n + offs_n)[None, :] < N_CTX,
          other=0.0,
      )
      v = tl.load(
          v_ptrs + start_n * stride_vn,
          mask=(start_n + offs_n)[:, None] < N_CTX,
          other=0.0,
      )

    # -- compute qk ---
    qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    qk += tl.dot(q, k)
    qk += tl.where((start_n + offs_n)[None, :] < N_CTX, 0, float("-inf"))

    if IS_CAUSAL:  # dao does afterwards
      qk += tl.where(
          offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf")
      ).to(tl.float16)

    if BIAS_TYPE != "none":
      if BIAS_TYPE == "matrix":
        bias = tl.load(
            b_ptrs + start_n * stride_bn,
            mask=(offs_m[:, None] < M_CTX)
            & ((start_n + offs_n)[None, :] < N_CTX),
            other=0.0,
        ).to(tl.float32)

      elif BIAS_TYPE == "virtual_matrix":
        rel_pos_ptrs = (-offs_m[:, None] + (start_n + offs_n)[None, :]) + (
            M_CTX - 1
        )
        b_ptrs = (
            Bias + 0 * stride_bz + off_h * stride_bh + rel_pos_ptrs * stride_bm
        )

        bias = tl.load(
            b_ptrs,
            mask=(offs_m[:, None] < M_CTX)
            & ((start_n + offs_n)[None, :] < N_CTX),
            other=0.0,
        ).to(tl.float32)
      qk += bias * qk_scale

    # -- compute scaling constant ---
    m_i_new = tl.maximum(m_i, tl.max(qk, 1))  # dao "lse_i" = trit "m_i"
    alpha = tl.math.exp2(m_i - m_i_new)  # dao "m_ij"  = trit "m_i_new"
    p = tl.math.exp2(qk - m_i_new[:, None])

    # -- scale and update acc --
    acc_scale = l_i * 0 + alpha  # workaround some compiler bug
    acc *= acc_scale[:, None]
    acc += tl.dot(
        p.to(tl.float16), v
    )  # while testing original float implementation

    # -- update m_i and l_i --
    l_i = l_i * alpha + tl.sum(p, 1)
    m_i = m_i_new

  # write back l and m
  acc = acc / l_i[:, None]
  l_ptrs = L + off_hz * M_CTX + offs_m
  tl.store(
      l_ptrs, m_i + tl.math.log2(l_i), mask=(offs_m < M_CTX)
  )  # okay this mask should fix it

  out_ptrs = (
      Out
      + off_z * stride_oz
      + off_h * stride_oh
      + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
  )
  if EVEN_M:
    tl.store(out_ptrs, acc.to(tl.float16))
  else:
    tl.store(out_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M_CTX))


class _attention_v2(torch.autograd.Function):
  """Attention v2."""

  @staticmethod
  def forward(
      ctx, q, k, v, bias=None, causal=False, sm_scale=None, verbose=False
  ):
    """Computes transformer attention value.

    Args:
      ctx:
      q: matrix of queries
      k: matrix of keys
      v: matrix of values
      bias: bias tensor, possibly in different shapes according to type
      causal: boolean, using triangular causal attention or not
      sm_scale: what to scale the SM attention temperature by
      verbose: whether to print out relevant computing information

    Returns:
      computed transformer attention value
    """
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    seqlen_k = k.shape[-2]
    seqlen_q = q.shape[-2]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    q, k, v = [
        x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]
    ]  # this line from dao
    o = torch.empty_like(q)
    BLOCK_M = 128
    BLOCK_N = 64
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    L = torch.empty(
        (q.shape[0] * q.shape[1], q.shape[2]),
        device=q.device,
        dtype=torch.float32,
    )

    if verbose:
      print("q", q.shape)
      print("k", k.shape)
      print("BLOCK_M", BLOCK_M, "BLOCK_N", BLOCK_N, "BLOCK_DMODEL", Lk)

    has_bias = bias is not None
    bias_type = "none"
    bias_strides = (0, 0, 0, 0)
    if has_bias:
      assert bias.dtype in [q.dtype, torch.float]
      assert bias.is_cuda
      if bias.stride(-1) != 1:
        bias = bias.contiguous()
      if bias.shape[2:] == (seqlen_q, seqlen_k):
        bias_type = "matrix"
        bias_strides = (
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            bias.stride(3),
        )
      elif len(bias.shape) == 3 and bias.shape[2] == (seqlen_q + seqlen_k - 1):
        bias_type = "virtual_matrix"
        bias_strides = (
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            -1,
        )

    num_warps = 4 if Lk <= 64 else 8
    _fwd_kernel_v2[grid](
        q,
        k,
        v,
        sm_scale,
        bias,
        L,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        *bias_strides,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        q.shape[0],
        q.shape[1],
        q.shape[2],
        k.shape[2],
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=causal,
        BIAS_TYPE=bias_type,
        num_warps=num_warps,
        num_stages=4,
    )

    ctx.save_for_backward(q, k, v, o, L, bias)
    ctx.grid = grid
    ctx.sm_scale = sm_scale
    ctx.BLOCK_DMODEL = Lk
    ctx.causal = causal
    return o

  @staticmethod
  def backward(ctx, do):
    raise NotImplementedError("no backwards support yet")


@triton.heuristics({
    "EVEN_M": lambda args: args["M_CTX"] % args["BLOCK_M"] == 0,
    "EVEN_N": lambda args: args["N_CTX"] % args["BLOCK_N"] == 0,
})
@triton.jit
def _fwd_kernel_v6(
    Q,
    K,
    V,
    sm_scale,
    Bias,
    L,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Sparsity,
    stride_sm,
    stride_sn,
    BiasShift,
    stride_tm,
    stride_tn,
    LocalLength,
    stride_ln,
    H,
    M_CTX,
    N_CTX,
    BLOCK_M,
    BLOCK_DMODEL,
    BLOCK_N,
    IS_CAUSAL,
    BIAS_TYPE,
    SPARSITY_TYPE,
    BIAS_SHIFT_TYPE,
    EVEN_M,
):
  """Forward kernel v6."""

  pass
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)

  off_z = off_hz // H  # batch index
  off_h = off_hz % H  # head index

  q_offset = off_z * stride_qz + off_h * stride_qh
  k_offset = off_z * stride_kz + off_h * stride_kh
  v_offset = off_z * stride_vz + off_h * stride_vh

  offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_n = tl.arange(0, BLOCK_N)
  offs_d = tl.arange(0, BLOCK_DMODEL)

  q_ptrs = (
      Q + q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
  )
  k_ptrs = (
      K + k_offset + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk)
  )
  v_ptrs = (
      V + v_offset + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
  )

  b_ptrs = None  # avoid cider"s error
  bias = None
  if BIAS_TYPE == "matrix":
    b_ptrs = (
        Bias
        + off_z * stride_bz
        + off_h * stride_bh
        + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    )

  # initialize offsets
  offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_n = tl.arange(0, BLOCK_N)
  # initialize pointer to m and l
  m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
  acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
  # scale sm_scale by log_2(e) and use
  # 2^x instead of exp in the loop because CSE and LICM
  # don"t work as expected with `exp` in the loop
  qk_scale = sm_scale * 1.44269504

  if EVEN_M:
    q = tl.load(q_ptrs)
  else:
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < M_CTX), other=0.0)
  q = (q * qk_scale).to(tl.float16)

  # loop over k, v and update accumulator
  lo = 0
  hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX) if IS_CAUSAL else N_CTX
  for start_n in range(lo, hi, BLOCK_N):

    good_to_include = True
    if SPARSITY_TYPE != "none":
      sp_ptr = Sparsity + start_m * stride_sm + start_n // BLOCK_N * stride_sn
      sp = tl.load(sp_ptr)
      if sp == 0.0:
        good_to_include = False

    bias_shift = 0
    if BIAS_SHIFT_TYPE != "none":
      bs_ptr = BiasShift + start_m * stride_tm + start_n // BLOCK_N * stride_tn
      bias_shift = tl.load(bs_ptr)

    n_loc_ptr = LocalLength + start_n // BLOCK_N * stride_ln
    N_LOC = tl.load(n_loc_ptr)

    if good_to_include:
      # -- load k, v --
      k = tl.load(
          k_ptrs + start_n * stride_kn,
          mask=(start_n + offs_n)[None, :] < N_LOC,
          other=0.0,
      )
      v = tl.load(
          v_ptrs + start_n * stride_vn,
          mask=(start_n + offs_n)[:, None] < N_LOC,
          other=0.0,
      )

      # -- compute qk ---
      qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
      qk += tl.dot(q, k)
      qk += tl.where((start_n + offs_n)[None, :] < N_LOC, 0, float("-inf"))

      if IS_CAUSAL:  # dao does afterwards
        qk += tl.where(
            offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf")
        ).to(tl.float16)

      if BIAS_TYPE != "none":
        if BIAS_TYPE == "matrix":
          bias = tl.load(
              b_ptrs + start_n * stride_bn,
              mask=(offs_m[:, None] < M_CTX)
              & ((start_n + offs_n)[None, :] < N_CTX),
              other=0.0,
          ).to(tl.float32)

        elif BIAS_TYPE == "virtual_matrix":
          rel_pos_ptrs = (
              (-offs_m[:, None] + (start_n + offs_n)[None, :])
              + (M_CTX - 1)
              + bias_shift
          )  # bias shift for FID
          b_ptrs = (
              Bias
              + 0 * stride_bz
              + off_h * stride_bh
              + rel_pos_ptrs * stride_bm
          )

          bias = tl.load(
              b_ptrs,
              mask=(offs_m[:, None] < M_CTX)
              & ((start_n + offs_n)[None, :] < N_CTX),
              other=0.0,
          ).to(tl.float32)
        qk += bias * qk_scale

      # -- compute scaling constant ---
      m_i_new = tl.maximum(m_i, tl.max(qk, 1))  # dao "lse_i" = trit "m_i"
      alpha = tl.math.exp2(m_i - m_i_new)  # dao "m_ij"  = trit "m_i_new"
      p = tl.math.exp2(qk - m_i_new[:, None])

      # -- scale and update acc --
      acc_scale = l_i * 0 + alpha  # workaround some compiler bug
      acc *= acc_scale[:, None]
      acc += tl.dot(
          p.to(tl.float16), v
      )  # while testing original float implementation

      # -- update m_i and l_i --
      l_i = l_i * alpha + tl.sum(p, 1)
      m_i = m_i_new

  # write back l and m
  acc = acc / l_i[:, None]
  l_ptrs = L + off_hz * M_CTX + offs_m
  tl.store(l_ptrs, m_i + tl.math.log2(l_i), mask=(offs_m < M_CTX))

  out_ptrs = (
      Out
      + off_z * stride_oz
      + off_h * stride_oh
      + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
  )
  if EVEN_M:
    tl.store(out_ptrs, acc.to(tl.float16))
  else:
    tl.store(out_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M_CTX))


# adding block sparse version
class _attention_v6(torch.autograd.Function):
  """Attention v6."""

  @staticmethod
  def forward(
      ctx,
      q,
      k,
      v,
      bias=None,
      causal=False,
      sm_scale=None,
      sparsity=None,
      bias_shift=None,
      local_length=None,
      BLOCK_MN=None,
      verbose=False,
  ):
    """Computes transformer attention value.

    Args:
      ctx:
      q: matrix of queries
      k: matrix of keys
      v: matrix of values
      bias: bias tensor, possibly in different shapes according to type
      causal: boolean, using triangular causal attention or not
      sm_scale: what to scale the SM attention temperature by
      sparsity: sparsity matrix choosing which blocks to compute
      bias_shift: shifts of the bias matrix for white space inside of the Flash
        Attention blocks, (when passages are not exactly 128 or 256 tokens long)
      local_length: running total of total actual length, not counting 'white
        space' from reshaping to fit inside flash attention's triton blocks
      BLOCK_MN:  preferred sizes of GPU blocks, defaults to None
      verbose: whether to print out relevant computing information

    Returns:
      computed transformer attention value
    """
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    seqlen_k = k.shape[-2]
    seqlen_q = q.shape[-2]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    # this line from dao:
    tup = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
    q, k, v = tup
    o = torch.empty_like(q)
    if BLOCK_MN is None:
      BLOCK_M = 128
      BLOCK_N = 128
    else:
      BLOCK_M = BLOCK_MN[0]
      BLOCK_N = BLOCK_MN[1]
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1], 1)
    L = torch.empty(
        (q.shape[0] * q.shape[1], q.shape[2]),
        device=q.device,
        dtype=torch.float32,
    )

    if verbose:
      print("q", q.shape)
      print("k", k.shape)
      print("BLOCK_M", BLOCK_M, "BLOCK_N", BLOCK_N, "BLOCK_DMODEL", Lk)

    has_bias = bias is not None
    bias_type = "none"
    bias_strides = (0, 0, 0, 0)
    if has_bias:
      assert bias.dtype in [q.dtype, torch.float]
      assert bias.is_cuda
      if bias.stride(-1) != 1:
        bias = bias.contiguous()
      if bias.shape[2:] == (seqlen_q, seqlen_k):
        bias_type = "matrix"
        bias_strides = (
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            bias.stride(3),
        )
      elif len(bias.shape) == 3 and bias.shape[2] == (seqlen_q + seqlen_k - 1):
        bias_type = "virtual_matrix"
        # NOTE: "bm" and "bn" become misnomer
        bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2), -1)

    has_sparsity = sparsity is not None
    sparsity_type = "none"
    sparsity_strides = (0, 0)
    if has_sparsity:
      assert sparsity.dim() == 2
      # just get the sparsity matrix and return the two strides
      sparsity_type = "matrix"
      sparsity_strides = (sparsity.stride(0), sparsity.stride(1))

    has_bias_shift = bias_shift is not None
    bias_shift_type = "none"
    bias_shift_strides = (0, 0)
    if has_bias_shift:
      assert bias_shift.dim() == 2
      bias_shift_type = "matrix"
      bias_shift_strides = (bias_shift.stride(0), bias_shift.stride(1))

    local_length_strides = (0,)
    if local_length is not None:
      local_length_strides = (local_length.stride(0),)

    num_warps = 4 if Lk <= 64 else 8
    _fwd_kernel_v6[grid](
        q,
        k,
        v,
        sm_scale,
        bias,
        L,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        *bias_strides,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        sparsity,
        *sparsity_strides,
        bias_shift,
        *bias_shift_strides,
        local_length,
        *local_length_strides,
        q.shape[0],
        q.shape[1],
        q.shape[2],
        k.shape[2],
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=causal,
        BIAS_TYPE=bias_type,
        SPARSITY_TYPE=sparsity_type,
        BIAS_SHIFT_TYPE=bias_shift_type,
        num_warps=num_warps,
        num_stages=4,
    )

    ctx.save_for_backward(q, k, v, o, L, bias)
    ctx.grid = grid
    ctx.sm_scale = sm_scale
    ctx.BLOCK_DMODEL = Lk
    ctx.causal = causal
    return o

  @staticmethod
  def backward(ctx, do):
    raise NotImplementedError("no backwards support yet")


@triton.heuristics({
    "EVEN_M": lambda args: args["M_CTX"] % args["BLOCK_M"] == 0,
    "EVEN_N": lambda args: args["N_CTX"] % args["BLOCK_N"] == 0,
})
@triton.jit
def _fwd_kernel_v7(
    Q,
    K,
    V,
    sm_scale,
    Bias,
    L,
    Out,
    stride_qz,
    stride_qh,
    stride_qm,
    stride_qk,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_kk,
    stride_vz,
    stride_vh,
    stride_vn,
    stride_vk,
    stride_bz,
    stride_bh,
    stride_bm,
    stride_bn,
    stride_oz,
    stride_oh,
    stride_om,
    stride_ok,
    Sparsity,
    stride_sz,
    stride_sm,
    stride_sn,
    BiasShift,
    stride_tm,
    stride_tn,
    LocalLength,
    stride_ln,
    H,
    SS,
    M_CTX,
    N_CTX,
    BLOCK_M,
    BLOCK_DMODEL,
    BLOCK_N,
    IS_CAUSAL,
    BIAS_TYPE,
    SPARSITY_TYPE,
    BIAS_SHIFT_TYPE,
    EVEN_M,
):
  """Forward kernel v7."""
  # M index, batchXhead index, decoder batch index (only for sparsity2 matrix)
  start_m = tl.program_id(0)
  off_hz = tl.program_id(1)
  off_z2 = tl.program_id(2)

  off_z = off_hz // H  # batch index
  off_h = off_hz % H  # head index
  off_z3 = off_z2 + off_z * SS  # decoder batch index

  q_offset = off_z3 * stride_qz + off_h * stride_qh
  k_offset = off_z * stride_kz + off_h * stride_kh
  v_offset = off_z * stride_vz + off_h * stride_vh

  offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_n = tl.arange(0, BLOCK_N)
  offs_d = tl.arange(0, BLOCK_DMODEL)

  q_ptrs = (
      Q + q_offset + (offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk)
  )
  k_ptrs = (
      K + k_offset + (offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kk)
  )
  v_ptrs = (
      V + v_offset + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk)
  )

  b_ptrs = None  # avoid cider"s error
  bias = None
  if BIAS_TYPE == "matrix":
    b_ptrs = (
        Bias
        + off_z * stride_bz
        + off_h * stride_bh
        + (offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn)
    )

  # initialize offsets
  offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
  offs_n = tl.arange(0, BLOCK_N)
  # initialize pointer to m and l
  m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
  acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
  # scale sm_scale by log_2(e) and use
  # 2^x instead of exp in the loop because CSE and LICM
  # don"t work as expected with `exp` in the loop
  qk_scale = sm_scale * 1.44269504

  if EVEN_M:
    q = tl.load(q_ptrs)
  else:
    q = tl.load(q_ptrs, mask=(offs_m[:, None] < M_CTX), other=0.0)
  q = (q * qk_scale).to(tl.float16)

  # loop over k, v and update accumulator
  lo = 0
  hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX) if IS_CAUSAL else N_CTX
  for start_n in range(lo, hi, BLOCK_N):

    good_to_include = True
    if SPARSITY_TYPE != "none":
      sp_ptr = (
          Sparsity
          + start_m * stride_sm
          + start_n // BLOCK_N * stride_sn
          + off_z3 * stride_sz
      )  # v7 version
      sp = tl.load(sp_ptr)
      if sp == 0.0:  # Not really sure if this works fine in triton compile
        good_to_include = False

    bias_shift = 0
    if BIAS_SHIFT_TYPE != "none":
      bs_ptr = BiasShift + start_m * stride_tm + start_n // BLOCK_N * stride_tn
      bias_shift = tl.load(bs_ptr)

    n_loc_ptr = LocalLength + start_n // BLOCK_N * stride_ln
    N_LOC = tl.load(n_loc_ptr)

    if good_to_include:
      k = tl.load(
          k_ptrs + start_n * stride_kn,
          mask=(start_n + offs_n)[None, :] < N_LOC,
          other=0.0,
      )
      v = tl.load(
          v_ptrs + start_n * stride_vn,
          mask=(start_n + offs_n)[:, None] < N_LOC,
          other=0.0,
      )

      # -- compute qk ---
      qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
      qk += tl.dot(q, k)
      qk += tl.where((start_n + offs_n)[None, :] < N_LOC, 0, float("-inf"))

      if IS_CAUSAL:  # dao does afterwards
        qk += tl.where(
            offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf")
        ).to(tl.float16)
      if BIAS_TYPE != "none":
        if BIAS_TYPE == "matrix":
          bias = tl.load(
              b_ptrs + start_n * stride_bn,
              mask=(offs_m[:, None] < M_CTX)
              & ((start_n + offs_n)[None, :] < N_CTX),
              other=0.0,
          ).to(tl.float32)
        elif BIAS_TYPE == "virtual_matrix":
          rel_pos_ptrs = (
              (-offs_m[:, None] + (start_n + offs_n)[None, :])
              + (M_CTX - 1)
              + bias_shift
          )  # bias shift for FID
          b_ptrs = (
              Bias
              + 0 * stride_bz
              + off_h * stride_bh
              + rel_pos_ptrs * stride_bm
          )

          bias = tl.load(
              b_ptrs,
              mask=(offs_m[:, None] < M_CTX)
              & ((start_n + offs_n)[None, :] < N_CTX),
              other=0.0,
          ).to(tl.float32)
        qk += bias * qk_scale

      # -- compute scaling constant ---
      m_i_new = tl.maximum(m_i, tl.max(qk, 1))  # dao "lse_i" = trit "m_i"
      alpha = tl.math.exp2(m_i - m_i_new)  # dao "m_ij"  = trit "m_i_new"
      p = tl.math.exp2(qk - m_i_new[:, None])

      # -- scale and update acc --
      acc_scale = l_i * 0 + alpha  # workaround some compiler bug
      acc *= acc_scale[:, None]
      acc += tl.dot(
          p.to(tl.float16), v
      )  # while testing original float implementation

      # -- update m_i and l_i --
      l_i = l_i * alpha + tl.sum(p, 1)
      m_i = m_i_new

  # write back l and m
  acc = acc / l_i[:, None]
  l_ptrs = L + (off_z3 * H + off_h) * M_CTX + offs_m
  # probably necessary for "v7 cross attention" (dont need bc no gradients tho)
  tl.store(
      l_ptrs, m_i + tl.math.log2(l_i), mask=(offs_m < M_CTX)
  )  # okay this mask should fix it

  out_ptrs = (
      Out
      + off_z3 * stride_oz
      + off_h * stride_oh
      + (offs_m[:, None] * stride_om + offs_d[None, :] * stride_ok)
  )
  if EVEN_M:
    tl.store(out_ptrs, acc.to(tl.float16))
  else:
    tl.store(out_ptrs, acc.to(tl.float16), mask=(offs_m[:, None] < M_CTX))


# adding block sparse version
class _attention_v7(torch.autograd.Function):
  """Attention v7."""

  @staticmethod
  def forward(
      ctx,
      q,
      k,
      v,
      bias=None,
      causal=False,
      sm_scale=None,
      sparsity=None,
      bias_shift=None,
      local_length=None,
      BLOCK_MN=None,
      verbose=False,
  ):
    """Computes transformer attention value.

    Args:
      ctx:
      q: matrix of queries
      k: matrix of keys
      v: matrix of values
      bias: bias tensor, possibly in different shapes according to type
      causal: boolean, using triangular causal attention or not
      sm_scale: what to scale the SM attention temperature by
      sparsity: sparsity matrix choosing which blocks to compute
      bias_shift: shifts of the bias matrix for white space inside of the Flash
        Attention blocks, (when passages are not exactly 128 or 256 tokens long)
      local_length: running total of total actual length, not counting 'white
        space' from reshaping to fit inside flash attention's triton blocks
      BLOCK_MN:  preferred sizes of GPU blocks, defaults to None
      verbose: whether to print out relevant computing information

    Returns:
      computed transformer attention value
    """
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    seqlen_k = k.shape[-2]
    seqlen_q = q.shape[-2]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]
    o = torch.empty_like(q)
    if BLOCK_MN is None:
      BLOCK_M = 128
      BLOCK_N = 128
    else:
      BLOCK_M = BLOCK_MN[0]
      BLOCK_N = BLOCK_MN[1]
    assert q.shape[0] % k.shape[0] == 0  # v7 style decoding
    SS = (
        q.shape[0] // k.shape[0]
    )  # sparsity rate of decoding samples (per each sample) "blow-up factor"
    grid = (triton.cdiv(q.shape[2], BLOCK_M), k.shape[0] * q.shape[1], SS)
    L = torch.empty(
        (q.shape[0] * q.shape[1], q.shape[2]),
        device=q.device,
        dtype=torch.float32,
    )

    if verbose:
      print("q", q.shape)
      print("k", k.shape)
      print("BLOCK_M", BLOCK_M, "BLOCK_N", BLOCK_N, "BLOCK_DMODEL", Lk)

    has_bias = bias is not None
    bias_type = "none"
    bias_strides = (0, 0, 0, 0)
    if has_bias:
      assert bias.dtype in [q.dtype, torch.float]
      assert bias.is_cuda
      if bias.stride(-1) != 1:
        bias = bias.contiguous()
      if bias.shape[2:] == (seqlen_q, seqlen_k):
        bias_type = "matrix"
        bias_strides = (
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            bias.stride(3),
        )
      elif len(bias.shape) == 3 and bias.shape[2] == (seqlen_q + seqlen_k - 1):
        bias_type = "virtual_matrix"
        bias_strides = (
            bias.stride(0),
            bias.stride(1),
            bias.stride(2),
            -1,
        )

    has_sparsity = sparsity is not None
    sparsity_type = "none"
    sparsity_strides = (0, 0, 0)
    if has_sparsity:
      assert sparsity.dim() == 3
      sparsity_type = "matrix"
      sparsity_strides = (
          sparsity.stride(0),
          sparsity.stride(1),
          sparsity.stride(2),
      )

    has_bias_shift = bias_shift is not None
    bias_shift_type = "none"
    bias_shift_strides = (0, 0)
    if has_bias_shift:
      assert bias_shift.dim() == 2
      bias_shift_type = "matrix"
      bias_shift_strides = (bias_shift.stride(0), bias_shift.stride(1))

    local_length_strides = (0,)
    if local_length is not None:
      local_length_strides = (local_length.stride(0),)

    num_warps = 4 if Lk <= 64 else 8
    _fwd_kernel_v7[grid](
        q,
        k,
        v,
        sm_scale,
        bias,
        L,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        *bias_strides,
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        sparsity,
        *sparsity_strides,
        bias_shift,
        *bias_shift_strides,
        local_length,
        *local_length_strides,
        q.shape[0],
        q.shape[1],
        SS,
        q.shape[2],
        k.shape[2],
        BLOCK_M=BLOCK_M,
        BLOCK_DMODEL=Lk,
        BLOCK_N=BLOCK_N,
        IS_CAUSAL=causal,
        BIAS_TYPE=bias_type,
        SPARSITY_TYPE=sparsity_type,
        BIAS_SHIFT_TYPE=bias_shift_type,
        num_warps=num_warps,
        num_stages=4,
    )
    ctx.save_for_backward(q, k, v, o, L, bias)
    ctx.grid = grid
    ctx.sm_scale = sm_scale
    ctx.BLOCK_DMODEL = Lk
    ctx.causal = causal
    return o

  @staticmethod
  def backward(ctx, do):
    raise NotImplementedError("no backwards support yet")


attention_v1 = _attention_v1.apply
attention_v2 = _attention_v2.apply
attention_v6 = _attention_v6.apply
attention_v7 = _attention_v7.apply
