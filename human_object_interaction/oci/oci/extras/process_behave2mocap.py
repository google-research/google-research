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

"""Process Behave2Mocap."""
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
# pylint: disable=g-importing-member
# pylint: disable=g-too-many-blank-lines
import argparse
import collections
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
from oci.utils import (
    geometry_utils,
    rigid_transform,
    smpl_utils,
    tensor_utils,
    transformations,
)
from psbody.mesh import Mesh
from scipy.spatial.transform import Rotation
from sklearn.utils import shuffle
import torch
from torch.utils.data.dataloader import default_collate
from viz.pyt3d_wrapper import Pyt3DWrapper

sys.path.append("/mnt/data/Research/behave_code")

c2c = tensor_utils.copy2cpu
n2t = tensor_utils.numpy2torch

SMPLH_PATH = '/mnt/data/Research/object_interactions/smplx/clean_models2/"'


class BehaveDataset:

  def __init__(self, datadir, save_dir, smpl_path, seq_filter="chairwood_sit"):

    self.smpl_fit_name = "fit02"
    self.obj_fit_name = "fit01"
    seq_folder = epath.Path(datadir)
    self.smpl_path = smpl_path
    filter_key = seq_filter
    sequences = [
        seq for seq in seq_folder.iterdir() if filter_key in str(seq.name)
    ]
    logger.info(f"Number of sequences {len(sequences)}")
    self.save_dir = save_dir
    for seq_folder in sequences:
      if "Date02_Sub02_chairwood_sit" not in str(seq_folder) and False:
        continue
      self.process_sequence2mocap(seq_folder)
    return

  def process_sequence2mocap(self, seq_folder):
    seq_info = SeqInfo(os.fspath(seq_folder))
    frame_reader = FrameDataReader(os.fspath(seq_folder))
    kinect_transform = KinectTransform(
        os.fspath(seq_folder), kinect_count=frame_reader.kinect_count)
    seq_end = frame_reader.cvt_end(None)
    bdata = {}
    seq_name = seq_folder.name
    save_path = osp.join(self.save_dir, f"{seq_name}_poses.npz")

    # bdata = np.load(input_file_path)
    # gender = np.array(bdata['gender'], ndmin=1)[0]
    # gender = str(gender,'utf-8') if isinstance(gender, bytes) else str(gender)
    # fps = bdata['mocap_framerate']
    # num_frames = bdata['poses'].shape[0]
    # trans = bdata['trans'][:]            # global translation
    # root_orient = bdata['poses'][:, :3]  # global root orientation (1 joint)
    # pose_body = bdata['poses'][:, 3:66]  # body joint rotations (21 joints)
    # pose_hand = bdata['poses'][:, 66:]   # finger articulation joint rotations
    # betas = bdata['betas'][:]

    bdata["poses"] = []
    bdata["gender"] = seq_info.info["gender"]
    bdata["mocap_framerate"] = []
    bdata["trans"] = []

    bdata["mocap_framerate"] = 1.0

    bdata["obj_vertices"] = []
    bdata["obj_angle"] = []
    bdata["obj_translation"] = []

    ## the behave model has mocap parameters such that the y-axis is the floor
    ## plane, while the AMASS dataset is based on the fact that
    ## z axis the floor.

    ## rotate about x-axis by 90 degrees to match with AMASS config.
    if bdata["gender"] == "male":
      _, smplh_model = smpl_utils.smplh_canonical(
          gender="male",
          smpl_model_path=self.smpl_path,
          use_pca=False,
          flat_hand_mean=True,
      )
    else:
      _, smplh_model = smpl_utils.smplh_canonical(
          gender="female",
          smpl_model_path=self.smpl_path,
          use_pca=False,
          flat_hand_mean=True,
      )

    RT1 = transformations.euler_matrix(
        -1 * np.pi / 2,
        0,
        0,
    )
    RT = RT1

    for ix in range(0, seq_end):
      frame_data = behave_utils.read_frame(frame_reader, ix, self.smpl_fit_name,
                                           self.obj_fit_name)
      _beta, _pose, _transl = smpl_utils.transform_smplh_parameters(
          smplh_model=smplh_model,
          beta=frame_data["smpl_params"]["beta"][None,],
          pose=frame_data["smpl_params"]["pose"][None],
          transl=frame_data["smpl_params"]["trans"][None,],
          RT=RT,
      )
      repo_obj_mesh_can, repo_obj_mesh_world = behave_utils.read_object_mesh(
          frame_reader, ix, self.obj_fit_name, from_repository=True)
      if False:
        obj_vertices = frame_data["obj_mesh"].v
        obj_faces = frame_data["obj_mesh"].f
        obj_mesh = Mesh(v=obj_vertices, f=obj_faces)
        obj_mesh.write_ply("frame_reader_obj.ply")
        repo_obj_mesh_world.write_ply("obj_watertight_world.ply")
      if False:
        smplh_model = smpl_utils.SMPL2Mesh(
            smpl_model_path=self.smpl_path, gender=bdata["gender"])
        body_mesh_outs = smplh_model.forward(
            _pose=_pose,
            _transl=_transl,
            _beta=_beta,
        )
        body_mesh = Mesh(
            v=c2c(body_mesh_outs.vertices[0]), f=body_mesh_outs.faces)
        body_mesh.write_ply("body2.ply")
        obj_vertices = frame_data["obj_mesh"].v
        obj_vertices = geometry_utils.transform_points(
            RT=RT, points=obj_vertices, return_homo=False)
        obj_faces = frame_data["obj_mesh"].f
        obj_mesh = Mesh(v=obj_vertices, f=obj_faces)
        obj_mesh.write_ply("obj2.ply")

      if False:
        obj_vertices = frame_data["obj_mesh"].v
        obj_faces = frame_data["obj_mesh"].f
      if True:
        obj_vertices = repo_obj_mesh_world.v
        obj_faces = repo_obj_mesh_world.f

      obj_vertices = geometry_utils.transform_points(
          RT=RT, points=obj_vertices, return_homo=False)
      bdata["obj_vertices"].append(obj_vertices)
      bdata["obj_faces"] = obj_faces
      bdata["obj_angle"].append(frame_data["obj_params"]["angle"])
      bdata["obj_translation"].append(frame_data["obj_params"]["translation"])

      bdata["poses"].append(_pose[0])
      bdata["trans"].append(_transl[0])
      bdata["betas"] = frame_data["smpl_params"]["beta"]
    bdata["obj_data"] = True
    bdata["obj_vertices"] = np.stack(bdata["obj_vertices"])
    bdata["obj_angle"] = np.stack(bdata["obj_angle"])
    bdata["obj_translation"] = np.stack(bdata["obj_translation"])
    bdata["poses"] = np.stack(bdata["poses"])
    bdata["trans"] = np.stack(bdata["trans"])
    bdata["betas"] = np.stack(bdata["betas"])
    np.savez(save_path, **bdata)
    return


def main(cfg):
  dataset = BehaveDataset(
      datadir=cfg.behave_datadir,
      save_dir=cfg.savedir,
      smpl_path=cfg.smpl_path,
      seq_filter=cfg.filter_key,
  )
  print("Done processing behave dataset")
  return


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--filter-key", type=str, default="chairwood_sit")
  parser.add_argument(
      "--behave-datadir",
      type=str,
      default="/mnt/data/Research/data/BEHAVE/sequences/",
      help="Behave dataset sequence folder",
  )
  parser.add_argument(
      "--rawmocap-savedir",
      dest="savedir",
      type=str,
      default="/mnt/data/Research/BEHAVE/raw_mocap_wt_obj/",
      help="Behave dataset mocap outdir",
  )
  parser.add_argument(
      "--smplh-path",
      dest="smpl_path",
      type=str,
      default="/mnt/data/Research/object_interactions/smplx/clean_models2/",
      help="SMPLH model path",
  )
  config = parser.parse_known_args()
  config = config[0]
  main(config)
