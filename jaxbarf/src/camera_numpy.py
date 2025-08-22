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

import numpy as np


def generate_camera_noise(num_poses=100, noise_std=0.15):
  """Generate camera noise."""
  se3_noise = np.random.randn(num_poses, 6) * noise_std
  return se3_noise


def add_camera_noise(poses, se3_noise):
  """Add camera noise."""
  num_poses = len(poses)
  se3_noise = se3_exp(se3_noise[:num_poses])
  noisy_poses = compose([se3_noise, poses[:, :3]])
  new_poses = np.copy(poses)
  new_poses[:, :3] = noisy_poses
  return new_poses


def define_pose(rot=None, trans=None):
  """Construct pose matrix."""
  assert(rot is not None or trans is not None)
  if rot is None:
    rot = np.repeat(np.eye(3)[None, Ellipsis], repeats=trans.shape[0], axis=0)
  if trans is None:
    trans = np.zeros(rot.shape[:-1])  # Bx3
  pose = np.concatenate([rot, trans[Ellipsis, None]], axis=-1)  # [..., 3, 4]
  assert pose.shape[-2:] == (3, 4)
  return pose


def invert_pose(pose):
  """Invert pose."""
  # invert a camera pose
  rot, trans = pose[Ellipsis, :3], pose[Ellipsis, 3:]
  r_inv = np.swapaxes(rot, -1, -2)
  t_inv = (-r_inv@trans)[Ellipsis, 0]
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
  r_new = r_b@r_a
  t_new = (r_b@t_a+t_b)[Ellipsis, 0]
  pose_new = define_pose(rot=r_new, trans=t_new)
  return pose_new


def se3_exp(wu):  # [..., 3]
  """Exponential map."""
  w_vec, u_vec = np.split(wu, 2, axis=1)
  wx = skew_symmetric(w_vec)
  theta = np.linalg.norm(w_vec, axis=-1)[Ellipsis, None, None]
  i_mat = np.eye(3)
  a_mat = taylor_a(theta)
  b_mat = taylor_b(theta)
  c_mat = taylor_c(theta)
  r_mat = i_mat+a_mat*wx+b_mat*wx@wx
  v_mat = i_mat+b_mat*wx+c_mat*wx@wx
  rt = np.concatenate([r_mat, (v_mat@u_vec[Ellipsis, None])], axis=-1)  # Bx3x4
  return rt


def skew_symmetric(w_vec):
  """Skey symmetric matrix from vectors."""
  w0, w1, w2 = np.split(w_vec, 3, axis=-1)
  w0 = w0.squeeze()
  w1 = w1.squeeze()
  w2 = w2.squeeze()
  zeros = np.zeros(w0.shape, np.float32)
  wx = np.stack([np.stack([zeros, -w2, w1], axis=-1),
                 np.stack([w2, zeros, -w0], axis=-1),
                 np.stack([-w1, w0, zeros], axis=-1)], axis=-2)
  return wx


def taylor_a(x_val, nth=10):
  """Taylor expansion."""
  # Taylor expansion of sin(x)/x
  ans = np.zeros_like(x_val)
  denom = 1.
  for i in range(nth+1):
    if i > 0:
      denom *= (2*i)*(2*i+1)
    ans = ans+(-1)**i*x_val**(2*i)/denom
  return ans


def taylor_b(x_val, nth=10):
  """Taylor expansion."""
  # Taylor expansion of (1-cos(x))/x**2
  ans = np.zeros_like(x_val)
  denom = 1.
  for i in range(nth+1):
    denom *= (2*i+1)*(2*i+2)
    ans = ans+(-1)**i*x_val**(2*i)/denom
  return ans


def taylor_c(x_val, nth=10):
  """Taylor expansion."""
  # Taylor expansion of (x-sin(x))/x**3
  ans = np.zeros_like(x_val)
  denom = 1.
  for i in range(nth+1):
    denom *= (2*i+2)*(2*i+3)
    ans = ans+(-1)**i*x_val**(2*i)/denom
  return ans


def to_hom(x_mat):
  """Homogenous coordinates."""
  # get homogeneous coordinates of the input
  x_hom = np.concatenate([x_mat, np.ones_like(x_mat[Ellipsis, :1])], axis=-1)
  return x_hom


def world2cam(x_mat, pose):  # [B, N, 3]
  """Coordinate transformations."""
  x_hom = to_hom(x_mat)
  return x_hom@np.swapaxes(pose, -1, -2)


def cam2img(x_mat, cam_intr):
  """Coordinate transformations."""
  return x_mat@np.swapaxes(cam_intr, -1, -2)


def img2cam(x_mat, cam_intr):
  """Coordinate transformations."""
  return x_mat@np.swapaxes(np.linalg.inv(cam_intr), -1, -2)


def cam2world(x_mat, pose):
  """Coordinate transformations."""
  x_hom = to_hom(x_mat)
  pose_inv = invert_pose(pose)
  return x_hom@np.swapaxes(pose_inv, -1, -2)


def angle_to_rotation_matrix(angle, axis):
  """Angle to rotation matrix."""
  # get the rotation matrix from Euler angle around specific axis
  roll = dict(X=1, Y=2, Z=0)[axis]
  zeros = np.zeros_like(angle)
  eye = np.ones_like(angle)
  mat = np.stack([np.stack([angle.cos(), -angle.sin(), zeros], axis=-1),
                  np.stack([angle.sin(), angle.cos(), zeros], axis=-1),
                  np.stack([zeros, zeros, eye], axis=-1)], axis=-2)
  mat = mat.roll((roll, roll), axis=(-2, -1))
  return mat


def rotation_distance(r1, r2, eps=1e-7):
  """Rotation distance."""
  # http://www.boris-belousov.net/2016/12/01/quat-dist/
  r_diff = r1@np.swapaxes(r2, -2, -1)
  trace = r_diff[Ellipsis, 0, 0]+r_diff[Ellipsis, 1, 1]+r_diff[Ellipsis, 2, 2]
  angle = np.arccos(((trace-1)/2).clip(-1+eps, 1-eps))
  return angle


def procrustes_analysis(x0, x1):  # [N, 3]
  """Procrustes."""
  # translation
  t0 = x0.mean(axis=0, keepdims=True)
  t1 = x1.mean(axis=0, keepdims=True)
  x0c = x0-t0
  x1c = x1-t1
  # scale
  s0 = np.sqrt((x0c**2).sum(axis=-1).mean())
  s1 = np.sqrt((x1c**2).sum(axis=-1).mean())
  x0cs = x0c/s0
  x1cs = x1c/s1
  # rotation (use double for SVD, float loses precision)
  # TODO do we need float64?
  u_mat, _, v_mat = np.linalg.svd((x0cs.transpose()@x1cs).astype(np.float64),
                                  full_matrices=False)
  rot = (u_mat@v_mat).astype(np.float32)
  if np.linalg.det(rot) < 0:
    rot[2] *= -1
  # align X1 to X0: X1to0 = (X1-t1)/s1@R.t()*s0+t0
  sim3 = {"t0": t0[0], "t1": t1[0], "s0": s0, "s1": s1, "R": rot}
  return sim3


def prealign_cameras(poses_pred, poses_gt):
  """Prealign cameras."""
  center = np.zeros((1, 1, 3))
  centers_pred = cam2world(center, poses_pred)[:, 0]
  centers_gt = cam2world(center, poses_gt)[:, 0]
  sim3 = procrustes_analysis(centers_gt, centers_pred)
  centers_aligned = (centers_pred-sim3["t1"])/(
      sim3["s1"]@sim3["R"].transpose()*sim3["s0"]+sim3["t0"])
  r_aligned = poses_pred[Ellipsis, :3]@sim3["R"].transpose()
  t_aligned = (-r_aligned@centers_aligned[Ellipsis, None])[Ellipsis, 0]
  poses_aligned = define_pose(rot=r_aligned, trans=t_aligned)
  return poses_aligned, sim3


def refine_test_cameras(poses_init, sim3):
  """Refine test cameras."""
  center = np.zeros((1, 1, 3))
  centers = cam2world(center, poses_init)[:, 0]
  centers_aligned = (
      centers-sim3["t0"])/sim3["s0"]@sim3["R"]*sim3["s1"]+sim3["t1"]
  r_aligned = poses_init[Ellipsis, :3]@sim3["R"]
  t_aligned = (-r_aligned@centers_aligned[Ellipsis, None])[Ellipsis, 0]
  poses_aligned = define_pose(rot=r_aligned, trans=t_aligned)
  return poses_aligned


def evaluate_camera(pose, pose_gt):
  """Evaluate cameras."""
  # evaluate the distance between pose and pose_ref; pose_ref is fixed
  pose_aligned, _ = prealign_cameras(pose, pose_gt)
  r_aligned, t_aligned = np.split(pose_aligned, [3,], axis=-1)
  r_gt, t_gt = np.split(pose_gt, [3,], axis=-1)
  r_error = rotation_distance(r_aligned, r_gt)
  t_error = np.linalg.norm((t_aligned-t_gt)[Ellipsis, 0], axis=-1)
  return r_error, t_error


def evaluate_aligned_camera(pose_aligned, pose_gt):
  """Evaluate aligned cameras."""
  # evaluate the distance between pose and pose_ref; pose_ref is fixed
  r_aligned, t_aligned = np.split(pose_aligned, [3,], axis=-1)
  r_gt, t_gt = np.split(pose_gt, [3,], axis=-1)
  r_error = rotation_distance(r_aligned, r_gt)
  t_error = np.linalg.norm((t_aligned-t_gt)[Ellipsis, 0], axis=-1)
  return r_error, t_error
