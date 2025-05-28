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

"""BEAHVE Data Utils."""
# pylint: disable=missing-function-docstring
# pylint: disable=unused-variable
# pylint: disable=missing-class-docstring
# pylint: disable=g-builtin-op
# pylint: disable=invalid-name
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=dangerous-default-value
# pylint: disable=unused-import
# pylint: disable=g-multiple-import
# pylint: disable=g-complex-comprehension
# pylint: disable=using-constant-test
# pylint: disable=g-explicit-length-test
# pylint: disable=undefined-variable
# pylint: disable=self-assigning-variable
# pylint: disable=g-import-not-at-top
# pylint: disable=line-too-long

## load object and mesh in the same frame?
## sample points in the 3D vol of the object ## Sample points on the mesh (SMPL)
## Get correspondences between the SMPL model and 3D points.

from collections import abc
import json
import os
import os.path as osp
import pickle as pkl
import sys

import cv2
from data.frame_data import FrameDataReader
from data.kinect_transform import KinectTransform
from data.seq_utils import SeqInfo
from etils import epath
from loguru import logger
import numpy as np
from oci.datasets import behave_utils
from oci.utils import geometry_utils, rigid_transform, smpl_utils, transformations
from oci.vibe_lib import geometry
from psbody.mesh import Mesh
from scipy.spatial.transform import Rotation
from sklearn.utils import shuffle
import smplx
import torch
from torch.utils.data.dataloader import default_collate
from viz.pyt3d_wrapper import Pyt3DWrapper

## these imports are from behave code

###


class MeshSampler:

  def __init__(self,):

    return


class MeshVertexSampler(MeshSampler):

  def __init__(self, canonical_mesh, num_points, sample_all_verts):
    super(MeshVertexSampler, self).__init__()
    vertices = canonical_mesh.v
    num_verts = len(vertices)
    self.sample_all_verts = sample_all_verts
    self.selected_inds = np.random.choice(num_verts, num_points)

  def sample_from_mesh(self, mesh):
    vertices = mesh.v
    sampled_verts = vertices[self.selected_inds]

    if self.sample_all_verts:
      return vertices * 1
    return sampled_verts


class CanonicalPoseEstimator:

  def __init__(self, canonical_mesh):
    self.canonical_mesh = canonical_mesh
    verts = canonical_mesh.v
    self.sampled_inds = np.random.choice(len(verts), 50)

  def compute_transformation(self, mesh):
    ref_verts = self.canonical_mesh.v[self.sampled_inds]
    trg_verts = mesh.v[self.sampled_inds]
    RT = rigid_transform.estimate_rigid_transform(ref_verts, trg_verts)
    return RT

  def get_canonical_mesh(self):
    return self.canonical_mesh


class BehaveDataset:

  def __init__(self, opts):

    self.opts = opts
    self.datadir = opts.datadir
    self.smpl_fit_name = opts.smpl_fit_name
    self.obj_fit_name = opts.obj_fit_name

    seq_folder = epath.Path(opts.seq_folder)
    filter_key = opts.seq_filter
    sequences = [
        seq for seq in seq_folder.iterdir() if filter_key in str(seq.name)
    ]
    if opts.single_sequence:
      sequences = sequences[0:1]
    frame_readers = [
        FrameDataReader(os.fspath(seq_folder)) for seq_folder in sequences
    ]
    kinect_transforms = [
        KinectTransform(
            os.fspath(seq_folder), kinect_count=frame_reader.kinect_count)
        for frame_reader, seq_folder in zip(frame_readers, sequences)
    ]
    seq_infos = [SeqInfo(os.fspath(seq_folder)) for seq_folder in sequences]
    self.num_input_frames = opts.num_input_frames
    self.frame_items = self.create_frames_data(
        num_input_frames=self.num_input_frames,
        sequences=sequences,
        frame_readers=frame_readers,
    )

    if opts.shuffle:
      self.frame_items = shuffle(self.frame_items)

    self.kinect_transforms = kinect_transforms
    self.frame_reader = frame_readers
    self.seq_infos = seq_infos

    # self.frame_reader = FrameDataReader(opts.seq_folder)
    # self.kinect_transform = KinectTransform(opts.seq_folder,
    # kinect_count=self.frame_reader.kinect_count)
    self.smpl_native = opts.smpl.native_path

    # self.seq_end = self.frame_reader.cvt_end(opts.seq_end if opts.seq_end > 0
    # else None)
    # self.frame_ids = [i for i in range(opts.seq_start, self.seq_end)]
    self.gender = "male"
    self.smplh_models = {}

    (
        self.smplh_models["mesh_male"],
        self.smplh_models["model_male"],
    ) = smpl_utils.smplh_canonical(
        gender="male",
        smpl_model_path=self.smpl_native,
        use_pca=False,
        flat_hand_mean=True,
    )
    (
        self.smplh_models["mesh_female"],
        self.smplh_models["model_female"],
    ) = smpl_utils.smplh_canonical(
        gender="female",
        smpl_model_path=self.smpl_native,
        use_pca=False,
        flat_hand_mean=True,
    )
    self.num_sampled_points = opts.num_sampled_points
    self.sample_all_verts = opts.sample_all_verts

    if False:
      ## deprecated code that only works with one sequence
      self.seq_info = SeqInfo(os.fspath(sequence[0]))
      self.object_cat = self.seq_info.get_obj_name(convert=True)

      object_obj_path = osp.join(
          self.datadir,
          "objects",
          self.seq_info.info["cat"],
          "{}.obj".format(self.seq_info.info["cat"]),
      )
      object_mesh = Mesh()
      object_mesh.load_from_obj(object_obj_path)
      object_mesh.v -= object_mesh.v.mean(0)[None, :]
      self.object_mesh = object_mesh
      self.seq_folder = opts.seq_folder
    # canonical_mesh = self.get_canonical_mesh()
    # self.canonical_pose_estimator = CanonicalPoseEstimator(canonical_mesh)
    logger.info(f"Number of sequences {len(sequences)}")
    logger.info(f"Number of frames {len(self.frame_items)}")
    return

  @staticmethod
  def create_frames_data(num_input_frames, sequences, frame_readers):
    ## create a dictonary for sequnece path index and frame_idx
    items = []
    for sx, (seq_folder,
             frame_reader) in enumerate(zip(sequences, frame_readers)):
      seq_end = frame_reader.cvt_end(None)
      frame_ids = list(range(num_input_frames, seq_end - 1))
      for frame_id in frame_ids:
        meta_data = {
            "seq_folder": seq_folder,
            "seq_index": sx,
            "frame_id": frame_id,
        }
        items.append(meta_data)
    return items

  def get_canonical_mesh(self,):
    return self.object_mesh

  def compute_transform_to_canonical(self, frame_data):
    angles = frame_data["obj_params"]["angle"]  ## this is encoded as rot_vec
    trans = frame_data["obj_params"]["translation"]
    rMatrix = np.eye(4)
    rMatrix[:3, :3] = Rotation.from_rotvec(angles).as_matrix()
    tMatrix = geometry_utils.get_T_from(trans)
    rtMatrix = np.matmul(tMatrix, rMatrix)
    return rtMatrix

  def __len__(self):
    return len(self.frame_items) - 1

  @staticmethod
  def read_frame(frame_reader, frame_id, smpl_name, obj_name):
    frame_data = behave_utils.read_frame(frame_reader, frame_id, smpl_name,
                                         obj_name)
    return frame_data

  def __getitem__(self, index):
    opts = self.opts
    meta_data = self.frame_items[index]
    seq_index = meta_data["seq_index"]
    seq_folder = meta_data["seq_folder"].name

    frame_reader = self.frame_reader[seq_index]
    kinect_transforms = self.kinect_transforms[seq_index]

    frame_id = meta_data["frame_id"]

    prev_frame_ids = []
    for i in range(0, self.num_input_frames):
      prev_frame_ids.append(frame_id - i)
    prev_frame_ids = prev_frame_ids[::-1]

    prev_frame_data = [
        self.read_frame(
            frame_reader,
            frame_id=fid,
            smpl_name=self.smpl_fit_name,
            obj_name=self.obj_fit_name,
        ) for fid in prev_frame_ids
    ]
    future_frame_id = frame_id + 1

    curr_frame_data = prev_frame_data[-1]

    gender = self.seq_infos[seq_index].get_gender()
    gender_id = 0
    if gender == "male":
      gender_id = 1
      smplh_canonical_mesh = self.smplh_models["mesh_male"]
      smplh_model = self.smplh_models["model_male"]
    elif gender == "female":
      gender_id = 2
      smplh_canonical_mesh = self.smplh_models["mesh_female"]
      smplh_model = self.smplh_models["model_female"]

    if False:
      gender = self.seq_infos[seq_index].get_gender()
      smpl_mesh, smpl_model = smplh_canonical(
          gender=gender, smpl_model_path=opts.smpl.native_path, use_pca=False)
      smpl_model.pose_mean *= 0

      betas = torch.FloatTensor(curr_frame_data["smpl_params"]["beta"])[None,]
      pose = torch.FloatTensor(curr_frame_data["smpl_params"]["pose"])[None,]
      num_body_joints = 21
      global_pose = pose[:, 0:3]
      body_pose = pose[:, 3:21 * 3 + 3]
      left_hand_pose = pose[:, 22 * 3:(22 + 15) * 3]
      right_hand_pose = pose[:, (22 + 15) * 3:]
      transl = torch.FloatTensor(curr_frame_data["smpl_params"]["trans"])[None,]
      smplh_out = smpl_model.forward(
          betas=betas,
          body_pose=body_pose,
          transl=transl,
          left_hand_pose=left_hand_pose,
          right_hand_pose=right_hand_pose,
          global_orient=global_pose,
      )
      temp_mesh = Mesh(smplh_out.vertices[0].data.numpy(), smpl_model.faces)

      temp_mesh.write_ply("curr_smpl.ply")
      curr_frame_data["smpl_mesh"].write_ply("curr_mesh.ply")

    # curr_frame_data = self.read_frame(frame_reader, frame_id=frame_id,  smpl_name=self.smpl_fit_name, obj_name=self.obj_fit_name)
    future_frame_data = self.read_frame(
        frame_reader,
        frame_id=future_frame_id,
        smpl_name=self.smpl_fit_name,
        obj_name=self.obj_fit_name,
    )

    num_verts = smplh_canonical_mesh.v.shape[0]

    assert (curr_frame_data["smpl_mesh"].v.shape[0] == num_verts
           ), "canonical and smpl model shapes should match."

    if False:
      objCan2World = self.compute_transform_to_canonical(curr_frame_data)
      objWorld2Can = np.linalg.inv(objCan2World)

    if False:
      objCan2World = self.compute_transform_to_canonical(curr_frame_data)
      og_can = self.get_canonical_mesh()
      og_can.write_ply("og_can.ply")
      object_verts = og_can.v * 1
      object_verts = geometry_utils.transform_points(objCan2World, object_verts)
      world_obj = Mesh(object_verts, og_can.f)

    objCan2World = self.compute_transform_to_canonical(curr_frame_data)
    objWorld2Can = np.linalg.inv(objCan2World)
    point_sampler = MeshVertexSampler(
        smplh_canonical_mesh,
        num_points=self.num_sampled_points,
        sample_all_verts=self.sample_all_verts,
    )

    canonical_points = point_sampler.sample_from_mesh(smplh_canonical_mesh)
    current_input_pts = [
        point_sampler.sample_from_mesh(fd["smpl_mesh"])
        for fd in prev_frame_data
    ]

    current_pts = point_sampler.sample_from_mesh(curr_frame_data["smpl_mesh"])
    future_pts = point_sampler.sample_from_mesh(future_frame_data["smpl_mesh"])

    ## Transform points from World Frame to Obj Can Frame. Can world frame?
    points_lst = [current_pts, future_pts]
    can_points_lst = self.transform_points_lst(points_lst, objWorld2Can)
    can_input_pts = self.transform_points_lst(current_input_pts, objWorld2Can)

    if False:
      world_obj_mesh = curr_frame_data["obj_mesh"]
      world_obj_mesh.write_ply("world_obj.ply")
      new_verts = self.transform_points_lst([world_obj_mesh.v], objWorld2Can)[0]
      tfs_can_mesh = Mesh(new_verts, world_obj_mesh.f * 1)
      tfs_can_mesh.write_ply("world2can_obj.ply")
      # can_mesh = self.canonical_pose_estimator.get_canonical_mesh()
      # can_mesh.write_ply('can.ply')

    elem = {}
    elem["can_points_curr"] = can_points_lst[0].astype(np.float32)  ## N x 3
    elem["can_points_futr"] = can_points_lst[1].astype(np.float32)  ## N x 3
    elem["can_input_pts"] = np.array(can_input_pts).astype(np.float32)

    # if opts.relative_targets:
    #   elem["can_points_futr"] -= elem["can_points_curr"]
    elem["can_points_smpl"] = canonical_points.astype(np.float32)

    elem["smpl_pose"] = torch.stack(
        [fd["smpl_params"]["pose"] for fd in prev_frame_data])
    elem["smpl_beta"] = torch.stack(
        [fd["smpl_params"]["beta"] for fd in prev_frame_data])

    elem["smpl_transl"] = torch.stack(
        [fd["smpl_params"]["trans"] for fd in prev_frame_data])
    ## subtract the translation of object to get in the frame.

    for ix in range(len(prev_frame_data)):
      _beta, _pose, _transl = smpl_utils.transform_smplh_parameters(
          smplh_model=smplh_model,
          beta=elem["smpl_beta"][ix][None,],
          pose=elem["smpl_pose"][ix][None],
          transl=elem["smpl_transl"][ix][None,],
          RT=objWorld2Can,
      )
      elem["smpl_beta"][ix], elem["smpl_pose"][ix], elem["smpl_transl"][ix] = (
          _beta[0],
          _pose[0],
          _transl[0],
      )

    if False:
      obj_mesh = curr_frame_data["obj_mesh"]
      verts = self.transform_points_lst([obj_mesh.v], objWorld2Can)[0]
      obj_mesh = Mesh(verts, obj_mesh.f)

      obj_mesh.write_ply("obj_can.ply")
      smpl_can_mesh = Mesh(elem["can_points_curr"], smplh_canonical_mesh.f)

      smpl_can_mesh.write_ply("can_mesh.ply")
      gender = self.seq_infos[seq_index].get_gender()

      smpl_mesh, smpl_model = smplh_canonical(
          gender=gender,
          smpl_model_path=opts.smpl.native_path,
          use_pca=False,
          flat_hand_mean=True,
      )

      betas = torch.FloatTensor(curr_frame_data["smpl_params"]["beta"])[None,]
      pose = torch.FloatTensor(curr_frame_data["smpl_params"]["pose"])[None,]
      num_body_joints = 21
      # objWorld2Can = np.eye(4)
      R1 = torch.FloatTensor(objWorld2Can[:3, :3])
      t1 = torch.FloatTensor(objWorld2Can[:3, 3])

      transl = torch.FloatTensor(curr_frame_data["smpl_params"]["trans"])[None,]

      # transl *= 0
      global_pose = pose[:, 0:3]
      global_pose = global_pose

      global_rot_matrix = geometry.angle_axis_to_rotation_matrix(global_pose)
      gpose = torch.matmul(torch.FloatTensor(R1), global_rot_matrix)
      gpose_rod = geometry.rotation_matrix_to_angle_axis(gpose)

      body_pose = pose[:, 3:21 * 3 + 3]
      left_hand_pose = pose[:, 22 * 3:(22 + 15) * 3]
      right_hand_pose = pose[:, (22 + 15) * 3:]

      new_gpose, new_transl = self._transform_smplh_parameters_helper(
          smpl_model,
          betas=betas,
          body_pose=body_pose,
          left_hand_pose=left_hand_pose,
          right_hand_pose=right_hand_pose,
          global_pose=global_pose,
          transl=transl,
          RT=objWorld2Can,
      )

      smplh_out = smpl_model.forward(
          betas=betas,
          body_pose=body_pose,
          transl=new_transl,
          left_hand_pose=left_hand_pose,
          right_hand_pose=right_hand_pose,
          global_orient=new_gpose,
      )

      temp_mesh = Mesh(smplh_out.vertices[0].data.numpy(), smpl_model.faces)
      temp_mesh.write_ply("can_mesh2.ply")
      if False:

        smplh_out = smpl_model.forward(
            betas=betas,
            body_pose=body_pose,
            transl=transl * 0,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            global_orient=global_pose * 0,
        )
        pelvis = smplh_out.joints[0, 0]
        smplh_out = smpl_model.forward(
            betas=betas,
            body_pose=body_pose,
            transl=transl,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            global_orient=global_pose,
        )
        vertices = smplh_out.vertices[0].data.numpy()
        vertices = self.transform_points_lst([vertices], objWorld2Can)[0]
        temp_mesh = Mesh(vertices, smpl_model.faces)
        temp_mesh.write_ply("can_mesh.ply")

        # pelvis = smplh_out.joints[0,0]
        transl2 = torch.matmul(R1, (pelvis + transl[0])) + t1 - pelvis
        transl2 = transl2[None,]
        smplh_out = smpl_model.forward(
            betas=betas,
            body_pose=body_pose,
            transl=transl2,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            global_orient=gpose_rod,
        )

        temp_mesh = Mesh(smplh_out.vertices[0].data.numpy(), smpl_model.faces)
        temp_mesh.write_ply("can_mesh2.ply")

      if False:
        smplh_out = smpl_model.forward(
            betas=betas,
            body_pose=body_pose,
            transl=t1[None],
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            global_orient=gpose_rod,
        )

        temp_mesh = Mesh(smplh_out.vertices[0].data.numpy(), smpl_model.faces)
        temp_mesh.write_ply("can_mesh3.ply")
      if False:

        smplh_out = smpl_model.forward(
            betas=betas,
            body_pose=body_pose,
            transl=transl * 0,
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
            global_orient=global_pose,
        )
        smplh_out = self.transform_points_lst([smplh_out.vertices[0]],
                                              objWorld2Can)[0]
        temp_mesh = Mesh(smplh_out, smpl_model.faces)

        temp_mesh.write_ply("can_mesh4.ply")

    elem["futr_smpl_beta"] = future_frame_data["smpl_params"]["beta"]
    elem["futr_smpl_pose"] = future_frame_data["smpl_params"]["pose"]
    elem["futr_smpl_transl"] = future_frame_data["smpl_params"]["trans"]
    _beta, _pose, _transl = self.transform_smplh_parameters(
        smplh_model=smplh_model,
        beta=elem["futr_smpl_beta"][None,],
        pose=elem["futr_smpl_pose"][None],
        transl=elem["futr_smpl_transl"][None,],
        RT=objWorld2Can,
    )
    elem["futr_smpl_beta"], elem["futr_smpl_pose"], elem["futr_smpl_transl"] = (
        _beta[0],
        _pose[0],
        _transl[0],
    )

    obj_mesh = curr_frame_data["obj_mesh"]

    elem["can2world"] = objCan2World.astype(np.float32)
    elem["empty"] = False
    elem["seq_folder"] = seq_folder
    elem["frame_id"] = frame_id
    elem["smpl_mesh_v"] = smplh_canonical_mesh.v.astype(np.float32)
    elem["smpl_mesh_f"] = smplh_canonical_mesh.f.astype(int)
    elem["obj_mesh_v"] = self.transform_points_lst(obj_mesh.v, objWorld2Can)
    elem["obj_mesh_f"] = obj_mesh.f
    # mesh = Mesh(v=elem['obj_mesh_v'], f= elem['obj_mesh_f'])
    # mesh.write_ply('test.ply')
    # breakpoint()
    elem["gender"] = gender_id
    elem["index"] = index
    return elem

  @staticmethod
  def transform_points_lst(points_array_lst, RT):

    if isinstance(points_array_lst) == list:
      output_array_lst = []
      for points_array in points_array_lst:
        points_array = geometry_utils.transform_points(
            RT=RT, points=points_array, return_homo=False)
        output_array_lst.append(points_array)
      return output_array_lst
    else:
      points_array = geometry_utils.transform_points(
          RT=RT, points=points_array_lst, return_homo=False)
      return points_array


def behave_dataoader(opts, shuffle_d=False):

  dataset = BehaveDataset(opts)
  # shuffle = opts.DATALOADER.SPLIT == 'train' or shuffle
  dataloader = torch.utils.data.DataLoader(
      dataset,
      batch_size=opts.batch_size,
      shuffle=shuffle_d,
      num_workers=opts.num_workers,
      collate_fn=collate_fn,
      worker_init_fn=worker_init_fn,
      drop_last=True,
  )
  return dataloader


def recursive_convert_to_torch(elem):
  if torch.is_tensor(elem):
    return elem
  elif type(elem).__module__ == "numpy":
    if elem.size == 0:
      return torch.zeros(elem.shape).type(torch.DoubleTensor)
    else:
      return torch.from_numpy(elem)
  elif isinstance(elem, int):
    return torch.LongTensor([elem])
  elif isinstance(elem, float):
    return torch.DoubleTensor([elem])
  elif isinstance(elem, abc.Mapping):
    return {key: recursive_convert_to_torch(elem[key]) for key in elem}
  elif isinstance(elem, abc.Sequence):
    return [recursive_convert_to_torch(samples) for samples in elem]
  elif elem is None:
    return elem
  else:
    return elem


def collate_fn(batch):
  """Globe data collater.

  Assumes each instance is a dict.
  Applies different collation rules for each field.
  Args:
    batch: List of loaded elements via Dataset.__getitem__
  """
  collated_batch = {"empty": True}
  new_batch = []
  for b in batch:
    if not b["empty"]:
      new_batch.append(b)

  if len(new_batch) > 0:
    for key in new_batch[0]:
      # print(key)

      if key == "mesh" or key == "obj_mesh":
        collated_batch[key] = recursive_convert_to_torch(
            [elem[key] for elem in new_batch])
      elif key == "ray_events":
        collated_batch[key] = [elem[key] for elem in new_batch]
      else:
        collated_batch[key] = default_collate([elem[key] for elem in new_batch])
    collated_batch["empty"] = False
  return collated_batch


def worker_init_fn(worker_id):
  ppid = os.getppid()
  np.random.seed(ppid + worker_id)
