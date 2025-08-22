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

# pylint: skip-file
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

# from byol.spatial_avg_group_linear import spatial_avg_group_linear_v2


def l2_normalize(x, axis=-1, epsilon=1e-12):
    """l2 normalize a tensor on an axis with numerical stability."""
    square_sum = jnp.sum(jnp.square(x), axis=axis, keepdims=True)
    x_inv_norm = jax.lax.rsqrt(jnp.maximum(square_sum, epsilon))
    return x * x_inv_norm


l2_norm_grad = jax.grad(
    lambda x, dy: jnp.sum(l2_normalize(x, axis=-1) * dy, axis=[0, 1]))


def _spatial_avg_norm_repeated_dot_product_cross_entropy_pmap(
        x, y, w, b, temp, labels_idx):
    """npgc,mc->npgm"""
    LARGE_NUM = 1e9
    N, P, G, C = x.shape
    x_avg = jnp.mean(x, axis=1)
    x_grp = jnp.reshape(x_avg, [N, -1])
    x2 = jnp.einsum('nc,cd->nd', x_grp, w) + b

    y_avg = jnp.mean(y, axis=1)
    y_grp = jnp.reshape(y_avg, [N, -1])
    y2 = jnp.einsum('nc,cd->nd', y_grp, w) + b

    x2_norm = l2_normalize(x2, axis=-1)
    y2_norm = l2_normalize(y2, axis=-1)
    D = x2_norm.shape[1]
    parallel = jax.device_count() > 1

    if parallel:
        # [N,D] -> [MN,D]
        x2_all = jnp.reshape(jax.lax.all_gather(x2_norm, axis_name='i'),
                             [-1, D])
        y2_all = jnp.reshape(jax.lax.all_gather(y2_norm, axis_name='i'),
                             [-1, D])
    else:
        x2_all = x2_norm
        y2_all = y2_norm

    Nall = x2_all.shape[0]
    masks = jax.nn.one_hot(labels_idx, Nall)
    labels = jax.nn.one_hot(labels_idx, Nall * 2)

    # [N, M]
    logits_aa = jnp.einsum('nc,mc->nm', x2_norm, x2_all) / temp
    logits_ab = jnp.einsum('nc,mc->nm', x2_norm, y2_all) / temp
    if parallel:
        logits_ba = jnp.einsum('nc,mc->nm', y2_norm, x2_all) / temp
    else:
        logits_ba = jnp.transpose(logits_ab)
    logits_bb = jnp.einsum('nc,mc->nm', y2_norm, y2_all) / temp
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = logits_bb - masks * LARGE_NUM

    # [N, 2M]
    logits_a = jnp.concatenate([logits_ab, logits_aa], axis=1)
    logits_b = jnp.concatenate([logits_ba, logits_bb], axis=1)
    logits_a = logits_a - logsumexp(logits_a, axis=-1, keepdims=True)
    logits_b = logits_b - logsumexp(logits_b, axis=-1, keepdims=True)

    # [N]
    loss_a = -jnp.sum(logits_a * labels, axis=-1)
    loss_b = -jnp.sum(logits_b * labels, axis=-1)
    loss = jnp.concatenate([loss_a, loss_b], axis=0)
    return jnp.tile(jnp.reshape(loss, [N * 2, 1, 1]), [1, P, G])


@jax.custom_vjp
def spatial_avg_norm_repeated_dot_product_cross_entropy_pmap_custom_vjp(
        x, y, w, b, temp, labels_idx):
    return _spatial_avg_norm_repeated_dot_product_cross_entropy_pmap(
        x, y, w, b, temp, labels_idx)


def spatial_avg_norm_repeated_dot_product_cross_entropy_pmap_fwd_(
        x, y, w, b, temp, labels_idx):
    LARGE_NUM = 1e9
    N, P, G, C = x.shape
    x_avg = jnp.mean(x, axis=1)
    x_grp = jnp.reshape(x_avg, [N, -1])
    x2 = jnp.einsum('nc,cd->nd', x_grp, w) + b

    y_avg = jnp.mean(y, axis=1)
    y_grp = jnp.reshape(y_avg, [N, -1])
    y2 = jnp.einsum('nc,cd->nd', y_grp, w) + b

    x2_norm = l2_normalize(x2, axis=-1)
    y2_norm = l2_normalize(y2, axis=-1)
    D = x2_norm.shape[1]
    parallel = jax.device_count() > 1

    if parallel:
        # [N,D] -> [MN,D]
        x2_all = jnp.reshape(jax.lax.all_gather(x2_norm, axis_name='i'),
                             [-1, D])
        y2_all = jnp.reshape(jax.lax.all_gather(y2_norm, axis_name='i'),
                             [-1, D])
    else:
        x2_all = x2_norm
        y2_all = y2_norm

    Nall = x2_all.shape[0]
    masks = jax.nn.one_hot(labels_idx, Nall)
    labels = jax.nn.one_hot(labels_idx, Nall * 2)

    # [N, M]
    logits_aa = jnp.einsum('nc,mc->nm', x2_norm, x2_all) / temp
    logits_ab = jnp.einsum('nc,mc->nm', x2_norm, y2_all) / temp
    if parallel:
        logits_ba = jnp.einsum('nc,mc->nm', y2_norm, x2_all) / temp
    else:
        logits_ba = jnp.transpose(logits_ab)
    logits_bb = jnp.einsum('nc,mc->nm', y2_norm, y2_all) / temp
    logits_aa = logits_aa - masks * LARGE_NUM
    logits_bb = logits_bb - masks * LARGE_NUM

    # [N, 2M]
    logits_a = jnp.concatenate([logits_ab, logits_aa], axis=1)
    logits_b = jnp.concatenate([logits_ba, logits_bb], axis=1)
    logits_a = logits_a - logsumexp(logits_a, axis=-1, keepdims=True)
    logits_b = logits_b - logsumexp(logits_b, axis=-1, keepdims=True)

    # [N]
    loss_a = -jnp.sum(logits_a * labels, axis=-1)
    loss_b = -jnp.sum(logits_b * labels, axis=-1)
    loss = jnp.concatenate([loss_a, loss_b], axis=0)
    return jnp.tile(jnp.reshape(loss, [N * 2, 1, 1]),
                    [1, P, G]), (x_grp, y_grp, x2, y2, x2_all, y2_all,
                                 logits_a, logits_b, temp, labels, w, P, G)


def spatial_avg_norm_repeated_dot_product_cross_entropy_pmap_bwd_(res, g):
    x_grp, y_grp, x2, y2, x2_all, y2_all, logits_a, logits_b, temp, labels, w, P, G = res
    N, GC = x_grp.shape
    C = GC // G

    # [N, P, G] -> [N]
    g_a = g[:N, 0, 0:1]  # [N, 1]
    g_b = g[N:, 0, 0:1]  # [N, 1]
    dlogits_a = g_a * (jax.nn.softmax(logits_a, axis=-1) - labels)  # [N, 2M]
    dlogits_b = g_b * (jax.nn.softmax(logits_b, axis=-1) - labels)  # [N, 2M]
    Nall = x2_all.shape[0]
    dlogits_ab, dlogits_aa = dlogits_a[:, :Nall], dlogits_a[:, Nall:]
    dlogits_ba, dlogits_bb = dlogits_b[:, :Nall], dlogits_b[:, Nall:]
    dx2 = (jnp.einsum('nm,mc->nc', dlogits_ab, y2_all) +
           jnp.einsum('nm,mc->nc', dlogits_aa, x2_all)) / temp
    dy2 = (jnp.einsum('nm,mc->nc', dlogits_ba, x2_all) +
           jnp.einsum('nm,mc->nc', dlogits_bb, y2_all)) / temp

    # NC
    dx2 = l2_norm_grad(x2, dx2)
    dy2 = l2_norm_grad(y2, dy2)

    db = (jnp.sum(dx2, axis=[0]) + jnp.sum(dy2, axis=[0])) * float(P * G)
    dw = (jnp.einsum('nc,nd->cd', x_grp, dx2) +
          jnp.einsum('nc,nd->cd', y_grp, dy2)) * float(P * G)
    w_ = jnp.reshape(w, [G, C, -1])
    dx = jnp.einsum('nd,gcd->ngc', dx2, w_) / float(P)
    dy = jnp.einsum('nd,gcd->ngc', dy2, w_) / float(P)
    dx = jnp.tile(dx[:, None, :, :], [1, P, 1, 1])
    dy = jnp.tile(dy[:, None, :, :], [1, P, 1, 1])
    return dx, dy, dw, db, None, None


spatial_avg_norm_repeated_dot_product_cross_entropy_pmap_custom_vjp.defvjp(
    spatial_avg_norm_repeated_dot_product_cross_entropy_pmap_fwd_,
    spatial_avg_norm_repeated_dot_product_cross_entropy_pmap_bwd_)


@jax.custom_jvp
def spatial_avg_norm_repeated_dot_product_cross_entropy_pmap_custom_jvp(
        x, y, w, b, temp, labels_idx):
    return _spatial_avg_norm_repeated_dot_product_cross_entropy_pmap(
        x, y, w, b, temp, labels_idx)


def spatial_avg_norm_repeated_dot_product_cross_entropy_pmap_jvp_(
        primals, tangents):
    x, y, w, b, temp, labels_idx = primals
    parallel = jax.device_count() > 1
    LARGE_NUM = 1e9
    N, P, G, C = x.shape

    dx, dy, dw, db, _, _ = tangents
    w_ = jnp.reshape(w, [G, C, -1])
    b = jnp.reshape(b, [-1])

    dx_avg = dx / float(P)
    x_avg = jnp.mean(x, axis=1)
    x_grp = jnp.reshape(x_avg, [x_avg.shape[0], -1])

    dy_avg = dy / float(P)
    y_avg = jnp.mean(y, axis=1)
    y_grp = jnp.reshape(y_avg, [y_avg.shape[0], -1])

    # [N, 1, 1, D]
    x2 = jnp.einsum('nc,cd->nd', x_grp, w) + b
    y2 = jnp.einsum('nc,cd->nd', y_grp, w) + b

    dx2 = jnp.einsum('npgc,gcd->npgd', dx_avg, w_) + jnp.einsum(
        'nc,cd->nd', x_grp, dw)[:, None, None, :] + db
    dy2 = jnp.einsum('npgc,gcd->npgd', dy_avg, w_) + jnp.einsum(
        'nc,cd->nd', y_grp, dw)[:, None, None, :] + db

    eps = 1e-12

    # [N,1,1,D]
    x_sq_sum = jnp.sum(jnp.square(x2), axis=-1, keepdims=True)
    y_sq_sum = jnp.sum(jnp.square(y2), axis=-1, keepdims=True)
    x_sq_sum = jnp.maximum(x_sq_sum, eps)
    y_sq_sum = jnp.maximum(y_sq_sum, eps)
    x_inv_norm = jax.lax.rsqrt(x_sq_sum)
    y_inv_norm = jax.lax.rsqrt(y_sq_sum)
    x_inv_norm_ = x_inv_norm[:, None, None, :]
    y_inv_norm_ = y_inv_norm[:, None, None, :]

    dx2 = dx2 * x_inv_norm_ - (
        x2 * (x_inv_norm**3) *
        (x_sq_sum > eps).astype(jnp.float32))[:, None, None, :] * jnp.sum(
            x2[:, None, None, :] * dx2, axis=-1, keepdims=True)
    dy2 = dy2 * y_inv_norm_ - (
        y2 * (y_inv_norm**3) *
        (y_sq_sum > eps).astype(jnp.float32))[:, None, None, :] * jnp.sum(
            y2[:, None, None, :] * dy2, axis=-1, keepdims=True)
    x2_norm = x2 * x_inv_norm
    y2_norm = y2 * y_inv_norm
    D = x2_norm.shape[1]

    if parallel:
        # [N,D] -> [MN,D]
        x2_all = jnp.reshape(jax.lax.all_gather(x2_norm, axis_name='i'),
                             [-1, D])
        y2_all = jnp.reshape(jax.lax.all_gather(y2_norm, axis_name='i'),
                             [-1, D])
    else:
        x2_all = x2_norm
        y2_all = y2_norm

    Nall = x2_all.shape[0]
    masks = jax.nn.one_hot(labels_idx, Nall)
    labels = jax.nn.one_hot(labels_idx, Nall * 2)

    # [N, M]
    logits_ab = jnp.einsum('nc,mc->nm', x2_norm, y2_all) / temp
    logits_aa = jnp.einsum('nc,mc->nm', x2_norm, x2_all) / temp
    logits_aa = logits_aa - masks * LARGE_NUM
    if parallel:
        logits_ba = jnp.einsum('nc,mc->nm', y2_norm, x2_all) / temp
    else:
        logits_ba = jnp.transpose(logits_ab)
    logits_bb = jnp.einsum('nc,mc->nm', y2_norm, y2_all) / temp
    logits_bb = logits_bb - masks * LARGE_NUM
    # [N, M]
    logits_a = jnp.concatenate([logits_ab, logits_aa], axis=1)
    logits_b = jnp.concatenate([logits_ba, logits_bb], axis=1)
    logits_a = logits_a - logsumexp(logits_a, axis=-1, keepdims=True)
    logits_b = logits_b - logsumexp(logits_b, axis=-1, keepdims=True)
    # [N]
    loss_a = -jnp.sum(logits_a * labels, axis=-1)
    loss_b = -jnp.sum(logits_b * labels, axis=-1)
    loss = jnp.concatenate([loss_a, loss_b], axis=0)
    loss = jnp.tile(jnp.reshape(loss, [N * 2, 1, 1]), [1, P, G])

    dlogits_bwd_a = jax.nn.softmax(logits_a, axis=-1) - labels  # [N, 2M]
    dlogits_bwd_b = jax.nn.softmax(logits_b, axis=-1) - labels  # [N, 2M]
    yx2 = jnp.concatenate([y2_all, x2_all], axis=0)  # [2M, D]
    xy2 = jnp.concatenate([x2_all, y2_all], axis=0)  # [2M, D]
    dlogits_fwd_a = jnp.einsum('npgc,mc,nm->npg', dx2, yx2,
                               dlogits_bwd_a) / temp
    dlogits_fwd_b = jnp.einsum('npgc,mc,nm->npg', dy2, xy2,
                               dlogits_bwd_b) / temp
    dloss = jnp.concatenate([dlogits_fwd_a, dlogits_fwd_b], axis=0)
    return loss, dloss


def spatial_avg_norm_repeated_dot_product_cross_entropy_pmap_jvp_new_(
        primals, tangents):
    x, y, w, b, temp, labels_idx = primals
    parallel = jax.device_count() > 1
    LARGE_NUM = 1e9
    N, P, G, C = x.shape

    dx, dy, dw, db, _, _ = tangents
    w_ = jnp.reshape(w, [G, C, -1])
    b = jnp.reshape(b, [-1])

    dx_avg = dx / float(P)
    x_avg = jnp.mean(x, axis=1)
    x_grp = jnp.reshape(x_avg, [x_avg.shape[0], -1])

    dy_avg = dy / float(P)
    y_avg = jnp.mean(y, axis=1)
    y_grp = jnp.reshape(y_avg, [y_avg.shape[0], -1])

    # [N, 1, 1, D]
    x2 = jnp.einsum('nc,cd->nd', x_grp, w) + b
    y2 = jnp.einsum('nc,cd->nd', y_grp, w) + b

    # [N,1,1,D]
    x2_norm = l2_normalize(x2, axis=-1)
    y2_norm = l2_normalize(y2, axis=-1)
    D = x2_norm.shape[1]

    if parallel:
        # [N,D] -> [MN,D]
        x2_all = jnp.reshape(jax.lax.all_gather(x2_norm, axis_name='i'),
                             [-1, D])
        y2_all = jnp.reshape(jax.lax.all_gather(y2_norm, axis_name='i'),
                             [-1, D])
    else:
        x2_all = x2_norm
        y2_all = y2_norm

    Nall = x2_all.shape[0]
    masks = jax.nn.one_hot(labels_idx, Nall)
    labels = jax.nn.one_hot(labels_idx, Nall * 2)

    # [N, M]
    logits_ab = jnp.einsum('nc,mc->nm', x2_norm, y2_all) / temp
    logits_aa = jnp.einsum('nc,mc->nm', x2_norm, x2_all) / temp
    logits_aa = logits_aa - masks * LARGE_NUM
    if parallel:
        logits_ba = jnp.einsum('nc,mc->nm', y2_norm, x2_all) / temp
    else:
        logits_ba = jnp.transpose(logits_ab)
    logits_bb = jnp.einsum('nc,mc->nm', y2_norm, y2_all) / temp
    logits_bb = logits_bb - masks * LARGE_NUM
    # [N, M]
    logits_a = jnp.concatenate([logits_ab, logits_aa], axis=1)
    logits_b = jnp.concatenate([logits_ba, logits_bb], axis=1)
    logits_a = logits_a - logsumexp(logits_a, axis=-1, keepdims=True)
    logits_b = logits_b - logsumexp(logits_b, axis=-1, keepdims=True)
    # [N]
    loss_a = -jnp.sum(logits_a * labels, axis=-1)
    loss_b = -jnp.sum(logits_b * labels, axis=-1)
    loss = jnp.concatenate([loss_a, loss_b], axis=0)
    loss = jnp.tile(jnp.reshape(loss, [N * 2, 1, 1]), [1, P, G])

    dlogits_bwd_a = jax.nn.softmax(logits_a, axis=-1) - labels  # [N, 2M]
    dlogits_bwd_b = jax.nn.softmax(logits_b, axis=-1) - labels  # [N, 2M]
    yx2 = jnp.concatenate([y2_all, x2_all], axis=0)
    xy2 = jnp.concatenate([x2_all, y2_all], axis=0)
    dlogits_fwd_a = jnp.einsum('mc,nm->nc', yx2, dlogits_bwd_a) / temp
    dlogits_fwd_b = jnp.einsum('mc,nm->nc', xy2, dlogits_bwd_b) / temp

    dx2_norm = l2_norm_grad(x2, dlogits_fwd_a)
    dy2_norm = l2_norm_grad(y2, dlogits_fwd_b)
    dloss_x = jnp.einsum(
        'npgc,gcd,nd->npg', dx_avg, w_, dx2_norm) + jnp.einsum(
            'nd,nd->n',
            (jnp.einsum('nc,cd->nd', x_grp, dw) + db), dx2_norm)[:, None, None]
    dloss_y = jnp.einsum(
        'npgc,gcd,nd->npg', dy_avg, w_, dy2_norm) + jnp.einsum(
            'nd,nd->n',
            (jnp.einsum('nc,cd->nd', y_grp, dw) + db), dy2_norm)[:, None, None]
    dloss = jnp.concatenate([dloss_x, dloss_y], axis=0)
    return loss, dloss


spatial_avg_norm_repeated_dot_product_cross_entropy_pmap_custom_jvp.defjvp(
    spatial_avg_norm_repeated_dot_product_cross_entropy_pmap_jvp_new_)
