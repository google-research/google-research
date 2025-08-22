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

import jax.numpy as jnp
from jax import lax


def define_pose(rot=None, trans=None):
  """Construct pose matrix."""
  assert(rot is not None or trans is not None)
  if rot is None:
    rot = jnp.repeat(jnp.eye(3)[None, Ellipsis],
                     repeats=trans.shape[0], axis=0)  # Bx3x3
  if trans is None:
    trans = jnp.zeros(rot.shape[:-1])  # Bx3
  pose = jnp.concatenate([rot, trans[Ellipsis, None]], axis=-1)  # [...,3,4]
  assert pose.shape[-2:] == (3, 4)
  return pose


def invert_pose(pose):
  """Invert pose."""
  # invert a camera pose
  rot, trans = pose[Ellipsis, :3], pose[Ellipsis, 3:]
  r_inv = jnp.swapaxes(rot, -1, -2)
  # t_inv = (-r_inv@t)[..., 0]
  t_inv = (-jnp.matmul(r_inv, trans, precision=lax.Precision.HIGHEST))[Ellipsis, 0]
  pose_inv = define_pose(rot=r_inv, trans=t_inv)
  return pose_inv


def compose(pose_list):
  """Compose sequence of poses."""
  # compose a sequence of poses together
  # pose_new(x) = poseN o ... o pose2 o pose1(x)
  pose_new = pose_list[0]
  for pose in pose_list[1:]:
    pose_new = compose_pair(pose_new, pose)
  return pose_new


def compose_pair(pose_a, pose_b):
  """Compose poses."""
  # pose_new(x) = pose_b o pose_a(x)
  r_a, t_a = pose_a[Ellipsis, :3], pose_a[Ellipsis, 3:]
  r_b, t_b = pose_b[Ellipsis, :3], pose_b[Ellipsis, 3:]
  # r_new = r_b@r_a
  r_new = jnp.matmul(r_b, r_a, precision=lax.Precision.HIGHEST)
  # t_new = (r_b@t_a+t_b)[...,0]
  t_new = (jnp.matmul(r_b, t_a, precision=lax.Precision.HIGHEST)+t_b)[Ellipsis, 0]
  pose_new = define_pose(rot=r_new, trans=t_new)
  return pose_new


def se3_exp(wu):  # [..., 3]
  """Exponential map."""
  w_vec, u_vec = jnp.split(wu, 2, axis=-1)

  # import pdb; pdb.set_trace()
  theta = jnp.linalg.norm(w_vec, axis=-1)[Ellipsis, None, None]
  wx = skew_symmetric(w_vec)
  i_mat = jnp.eye(3)
  a_mat = taylor_a(theta)
  b_mat = taylor_b(theta)
  c_mat = taylor_c(theta)
  r_mat = i_mat + a_mat*wx + jnp.matmul(b_mat * wx, wx,
                                        precision=lax.Precision.HIGHEST)
  v_mat = i_mat + b_mat*wx + jnp.matmul(c_mat * wx, wx,
                                        precision=lax.Precision.HIGHEST)
  rt = jnp.concatenate(
      [r_mat, (jnp.matmul(v_mat, u_vec[Ellipsis, None],
                          precision=lax.Precision.HIGHEST))], axis=-1)  # Bx3x4
  return rt


def skew_symmetric(w_vec):
  """Skew symmetric matrix from vector."""
  w0, w1, w2 = jnp.split(w_vec, 3, axis=-1)
  w0 = w0.squeeze()
  w1 = w1.squeeze()
  w2 = w2.squeeze()
  zeros = jnp.zeros(w0.shape, jnp.float32)
  wx = jnp.stack([jnp.stack([zeros, -w2, w1], axis=-1),
                  jnp.stack([w2, zeros, -w0], axis=-1),
                  jnp.stack([-w1, w0, zeros], axis=-1)], axis=-2)
  return wx


def taylor_a(x_val, nth=10):
  """Taylor expansion of sin(x)/x."""
  ans = jnp.zeros_like(x_val)
  denom = 1.
  ans = ans+x_val**0
  for i in jnp.arange(1, nth+1):
    denom *= (2*i)*(2*i+1)
    ans = ans+(-1)**i*x_val**(2*i)/denom
  return ans


def taylor_b(x_val, nth=10):
  """Taylor expansion of (1-cos(x))/x**2."""
  ans = jnp.zeros_like(x_val)
  denom = 1.
  for i in jnp.arange(nth+1):
    denom *= (2*i+1)*(2*i+2)
    ans = ans+(-1)**i*x_val**(2*i)/denom
  return ans


def taylor_c(x_val, nth=10):
  """Taylor expansion of (x-sin(x))/x**3."""
  ans = jnp.zeros_like(x_val)
  denom = 1.
  for i in jnp.arange(nth+1):
    denom *= (2*i+2)*(2*i+3)
    ans = ans+(-1)**i*x_val**(2*i)/denom
  return ans


def to_hom(x_mat):
  """Homogeneous coordinates."""
  x_hom = jnp.concatenate([x_mat, jnp.ones_like(x_mat[Ellipsis, :1])], axis=-1)
  return x_hom


def world2cam(x_mat, pose):  # [B,N,3]
  """Transform between coordinate systems."""
  x_hom = to_hom(x_mat)
  return jnp.matmul(x_hom, jnp.swapaxes(pose, -1, -2),
                    precision=lax.Precision.HIGHEST)


def cam2img(x_mat, cam_intr):
  """Transform between coordinate systems."""
  return jnp.matmul(x_mat, jnp.swapaxes(cam_intr, -1, -2),
                    precision=lax.Precision.HIGHEST)


def img2cam(x_mat, cam_intr):
  """Transform between coordinate systems."""
  return jnp.matmul(x_mat, jnp.swapaxes(jnp.linalg.inv(cam_intr), -1, -2),
                    precision=lax.Precision.HIGHEST)


def cam2world(x_mat, pose):
  """Rotation matrix from angle."""
  x_hom = to_hom(x_mat)
  pose_inv = invert_pose(pose)
  # return x_hom@jnp.swapaxes(pose_inv, -1, -2)
  return jnp.matmul(x_hom, jnp.swapaxes(pose_inv, -1, -2),
                    precision=lax.Precision.HIGHEST)


def angle_to_rotation_matrix(angle, axis):
  """Rotation matrix from angle."""
  # get the rotation matrix from Euler angle around specific axis
  roll = dict(X=1, Y=2, Z=0)[axis]
  zeros = jnp.zeros_like(angle)
  eye = jnp.ones_like(angle)
  m_mat = jnp.stack([jnp.stack([angle.cos(), -angle.sin(), zeros], axis=-1),
                     jnp.stack([angle.sin(), angle.cos(), zeros], axis=-1),
                     jnp.stack([zeros, zeros, eye], axis=-1)], axis=-2)
  m_mat = m_mat.roll((roll, roll), axis=(-2, -1))
  return m_mat


def rotation_distance(r1, r2, eps=1e-7):
  """Rotation distance."""
  # http://www.boris-belousov.net/2016/12/01/quat-dist/
  r_diff = jnp.matmul(r1, jnp.swapaxes(r2, -2, -1),
                      precision=lax.Precision.HIGHEST)
  trace = r_diff[Ellipsis, 0, 0]+r_diff[Ellipsis, 1, 1]+r_diff[Ellipsis, 2, 2]
  angle = jnp.arccos(((trace-1)/2).clip(-1+eps, 1-eps))
  return angle


def procrustes_analysis(x0, x1):  # [N,3]
  """Procrustes."""
  # translation
  t0 = x0.mean(axis=0, keepdims=True)
  t1 = x1.mean(axis=0, keepdims=True)
  x0c = x0-t0
  x1c = x1-t1
  # scale
  s0 = jnp.sqrt((x0c**2).sum(axis=-1).mean())
  s1 = jnp.sqrt((x1c**2).sum(axis=-1).mean())
  x0cs = x0c/s0
  x1cs = x1c/s1
  # rotation (use double for SVD, float loses precision)
  # NOTE the return format of jnp.linalg.svd and pytorch svd is different.
  u_mat, _, v_mat = jnp.linalg.svd(jnp.matmul(
      x0cs.transpose(), x1cs, precision=lax.Precision.HIGHEST),
                                   full_matrices=False)
  rot = jnp.matmul(u_mat, v_mat, precision=lax.Precision.HIGHEST)
  # if jnp.linalg.det(R) < 0: R[2]*=-1
  neg_r = rot.at[2].multiply(-1)
  # align x1 to x0: x1to0 = (x1-t1)/s1@R.t()*s0+t0
  sim3 = {'t0': t0[0], 't1': t1[0], 's0': s0, 's1': s1,
          'R': jnp.where(jnp.linalg.det(rot) < 0, neg_r, rot)}
  return sim3


def prealign_cameras(poses_pred, poses_gt):
  """Prealign cameras."""
  center = jnp.zeros((1, 1, 3))
  centers_pred = cam2world(center, poses_pred)[:, 0]
  centers_gt = cam2world(center, poses_gt)[:, 0]
  sim3 = procrustes_analysis(centers_gt, centers_pred)
  centers_aligned = jnp.matmul(
      (centers_pred-sim3['t1'])/sim3['s1'], sim3['R'].transpose(),
      precision=lax.Precision.HIGHEST)*sim3['s0']+sim3['t0']
  r_aligned = jnp.matmul(poses_pred[Ellipsis, :3, :3],
                         sim3['R'].transpose(), precision=lax.Precision.HIGHEST)
  t_aligned = jnp.matmul(-r_aligned, centers_aligned[Ellipsis, None],
                         precision=lax.Precision.HIGHEST)[Ellipsis, 0]
  poses_aligned = define_pose(rot=r_aligned, trans=t_aligned)
  return poses_aligned, sim3


def evaluate_camera_alignment(pose_aligned, pose_gt):
  """Measure errors in rotation and translation."""
  r_aligned, t_aligned = pose_aligned.split([3,], axis=-1)
  r_gt, t_gt = pose_gt.split([3,], axis=-1)
  r_error = rotation_distance(r_aligned, r_gt)
  t_error = jnp.linalg.norm((t_aligned-t_gt)[Ellipsis, 0], axis=-1)
  return r_error, t_error


def evaluate_camera(pose, pose_ref):
  """Evaluate the distance between pose and pose_ref; pose_ref is fixed."""
  pose_aligned, sim3 = prealign_cameras(pose, pose_ref)
  return evaluate_camera_alignment(pose_aligned, pose_ref)
