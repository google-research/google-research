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

"""Predicting future trajectory (points) given the current shape as input."""

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
# pylint: disable=using-constant-test

import os.path as osp

import cv2
from data.frame_data import FrameDataReader
from data.kinect_transform import KinectTransform
from data.seq_utils import SeqInfo
import imageio
from matplotlib import cm
import numpy as np
from oci.utils import geometry_utils, mesh_utils
from psbody.mesh import Mesh
from viz.pyt3d_wrapper import Pyt3DWrapper

## Behave dataloder functions

## oci imports

colormap = cm.get_cmap("plasma")


class VisualizeModelOutputs:
  """seq_folder : gt sequence vis_mesh: visualize mesh or just points.

  In case of vis_mesh.

  Connectivity needs to be given as input
  """

  def __init__(self, seq_folder, vis_mesh=False, faces=None):
    frame_reader = FrameDataReader(seq_folder)
    kinect_transform = KinectTransform(
        seq_folder, kinect_count=frame_reader.kinect_count)

    self.frame_reader = frame_reader
    self.kinect_transform = kinect_transform
    self.pyt3d_wrapper = Pyt3DWrapper(image_size=1200)
    self.vis_mesh = vis_mesh
    self.faces = faces
    return

  def visualize_outputs(self, inputs, predictions):
    frame_reader = self.frame_reader
    kinect_transform = self.kinect_transform
    pyt3d_wrapper = self.pyt3d_wrapper
    return self._visualize_model_outputs(
        frame_reader,
        kinect_transform,
        pyt3d_wrapper,
        inputs,
        predictions,
        can_frame=True,
        vis_mesh=self.vis_mesh,
        mesh_faces=self.faces,
    )

  def visualize_mesh_ouputs(self,
                            frame_id,
                            can2WorldRT,
                            obj_mesh,
                            smpl_mesh,
                            can_frame=True,
                            kids=[1, 2]):
    frame_reader = self.frame_reader
    kinect_transform = self.kinect_transform
    pyt3d_wrapper = self.pyt3d_wrapper

    if can_frame:
      # can2worldRT = can2worldRT
      ## transform points to world frame for visualization.
      smpl_mesh = self.transform_mesh(mesh=smpl_mesh, RT=can2WorldRT)
      obj_mesh = self.transform_mesh(mesh=obj_mesh, RT=can2WorldRT)

    outputs = self._visualize_mesh_ouputs(
        frame_id,
        frame_reader=frame_reader,
        kinect_transform=kinect_transform,
        pyt3d_wrapper=pyt3d_wrapper,
        smpl_mesh=smpl_mesh,
        obj_mesh=obj_mesh,
        kids=kids,
    )
    return outputs

  @staticmethod
  def transform_mesh(mesh, RT):
    mesh_v = geometry_utils.transform_points(RT, mesh.v)
    return Mesh(mesh_v, f=mesh.f)

  @staticmethod
  def _visualize_mesh_ouputs(
      frame_id,
      frame_reader,
      kinect_transform,
      pyt3d_wrapper,
      smpl_mesh,
      obj_mesh,
      kids=[1, 2],
  ):
    # kids = [1, 2]  # choose which kinect id to visualize

    imgs_all = frame_reader.get_color_images(frame_id, frame_reader.kids)
    image_size = 640
    w, h = image_size, int(image_size * 0.75)
    imgs_resize = [cv2.resize(x, (w, h)) for x in imgs_all]
    visuals = {"imgs": [imgs_resize[k] for k in kids]}
    selected_imgs = [imgs_resize[x] for x in kids]
    fit_meshes = [obj_mesh, smpl_mesh]
    rend_imgs = []

    for orig, kid in zip(selected_imgs, kids):
      rend_img = VisualizeModelOutputs._render_mesh(
          fit_meshes,
          pyt3d_wrapper=pyt3d_wrapper,
          kinect_transform=kinect_transform,
          kid=kid,
      )
      h, w = orig.shape[:2]
      rend_img = cv2.resize((rend_img * 255).astype(np.uint8), (w, h))
      rend_imgs.append(rend_img)
    visuals["rend_imgs"] = rend_imgs
    return visuals

  @staticmethod
  def _visualize_model_outputs(
      frame_reader,
      kinect_transform,
      pyt3d_wrapper,
      inputs,
      predictions,
      can_frame,
      vis_mesh,
      mesh_faces=None,
  ):
    image_size = 640
    w, h = image_size, int(image_size * 0.75)

    frame_id = inputs["seq_frame_id"]
    smpl_name = "fit02"
    obj_name = "fit01"
    smpl_fit = frame_reader.get_smplfit(frame_id, smpl_name)
    obj_fit = frame_reader.get_objfit(frame_id, obj_name)

    curr_frame_pts = inputs["can_points_curr"]
    futr_frame_pts = inputs["can_points_futr"]

    pred_futr_frame_pts = predictions["points_futr"]
    if can_frame:
      can2worldRT = inputs["can2world"]
      ## transform points to world frame for visualization.
      curr_frame_pts = geometry_utils.transform_points(can2worldRT,
                                                       curr_frame_pts)
      futr_frame_pts = geometry_utils.transform_points(can2worldRT,
                                                       futr_frame_pts)

      pred_futr_frame_pts = geometry_utils.transform_points(
          can2worldRT, pred_futr_frame_pts)

    if vis_mesh:
      curr_frame_mesh = mesh_utils.convert_verts_to_mesh(
          curr_frame_pts, faces=mesh_faces)
      futr_frame_mesh = mesh_utils.convert_verts_to_mesh(
          futr_frame_pts, faces=mesh_faces)
      pred_futr_frame_mesh = mesh_utils.convert_verts_to_mesh(
          pred_futr_frame_pts, faces=mesh_faces)

    else:
      curr_frame_mesh = mesh_utils.convert_points_to_mesh(
          curr_frame_pts, return_ps_mesh=True)
      futr_frame_mesh = mesh_utils.convert_points_to_mesh(
          futr_frame_pts, return_ps_mesh=True)
      pred_futr_frame_mesh = mesh_utils.convert_points_to_mesh(
          pred_futr_frame_pts, return_ps_mesh=True)

    kids = [1, 2]  # choose which kinect id to visualize
    imgs_all = frame_reader.get_color_images(frame_id, frame_reader.kids)

    imgs_resize = [cv2.resize(x, (w, h)) for x in imgs_all]
    visuals = {"img": imgs_resize[1]}

    selected_imgs = [imgs_resize[x] for x in kids]
    fit_meshes = [obj_fit, curr_frame_mesh]

    for orig, kid in zip(selected_imgs, kids):
      rend_img = VisualizeModelOutputs._render_mesh(
          fit_meshes,
          pyt3d_wrapper=pyt3d_wrapper,
          kinect_transform=kinect_transform,
          kid=kid,
      )
      h, w = orig.shape[:2]
      rend_img = cv2.resize((rend_img * 255).astype(np.uint8), (w, h))
      visuals[f"curr_frame_{kid}"] = rend_img

    fit_meshes = [obj_fit, futr_frame_mesh]
    for orig, kid in zip(selected_imgs, kids):
      rend_img = VisualizeModelOutputs._render_mesh(
          fit_meshes,
          pyt3d_wrapper=pyt3d_wrapper,
          kinect_transform=kinect_transform,
          kid=kid,
      )
      h, w = orig.shape[:2]
      rend_img = cv2.resize((rend_img * 255).astype(np.uint8), (w, h))
      visuals[f"futr_frame_{kid}"] = rend_img

    fit_meshes = [obj_fit, pred_futr_frame_mesh]
    for orig, kid in zip(selected_imgs, kids):
      rend_img = VisualizeModelOutputs._render_mesh(
          fit_meshes,
          pyt3d_wrapper=pyt3d_wrapper,
          kinect_transform=kinect_transform,
          kid=kid,
      )
      h, w = orig.shape[:2]
      rend_img = cv2.resize((rend_img * 255).astype(np.uint8), (w, h))
      visuals[f"pred_futr_frame_{kid}"] = rend_img

    return visuals

  @staticmethod
  def _render_mesh(fit_meshes, pyt3d_wrapper, kinect_transform, kid):
    fit_meshes_local = kinect_transform.world2local_meshes(fit_meshes, kid)
    rend = pyt3d_wrapper.render_meshes(fit_meshes_local, viz_contact=False)
    return rend


class SequenceVisualizer:

  def __init__(self, seq_folder, start_frame_id, can2WorldRT):
    self.visualizer = VisualizeModelOutputs(seq_folder=seq_folder)
    self.start_frame_id = start_frame_id
    self.can2WorldRT = can2WorldRT
    self.kids = [1, 2]
    return

  def forward(self, save_dir, mesh_sequences, object_meshes):
    num_steps = len(mesh_sequences)
    frame_id = self.start_frame_id * 1
    can2worldRT = self.can2WorldRT

    sequence_visuals = []
    for i in range(num_steps):
      smpl_mesh = mesh_sequences[i]
      obj_mesh = object_meshes[i]
      visuals_i = self.visualizer.visualize_mesh_ouputs(
          frame_id=frame_id,
          can2WorldRT=can2worldRT,
          obj_mesh=obj_mesh,
          smpl_mesh=smpl_mesh,
          can_frame=True,
          kids=self.kids,
      )
      visuals_i["kids"] = self.kids
      visuals_i["frame_id"] = frame_id - self.start_frame_id
      frame_id += 1
      sequence_visuals.append(visuals_i)

    rend_video_path = osp.join(save_dir, "out.mp4")
    video_writer = None

    if False:

      for i in range(num_steps):
        comb = self.montage_visuals(sequence_visuals[i])
        if video_writer is None:
          ch, cw = comb.shape[:2]
          # fourcc = cv2.VideoWriter_fourcc(*'H264')
          # fourcc = 0x00000021
          fourcc = cv2.VideoWriter_fourcc(*"mp4v")
          video_writer = cv2.VideoWriter(rend_video_path, fourcc, 1, (cw, ch))
        sequence_visuals[i]["combined"] = comb
        video_writer.write(cv2.cvtColor(comb, cv2.COLOR_RGB2BGR))
      video_writer.release()
    if True:
      video_writer = None
      for i in range(num_steps):
        comb = self.montage_visuals(sequence_visuals[i])
        if video_writer is None:
          video_writer = imageio.get_writer(rend_video_path, fps=1)
        video_writer.append_data(comb)
        sequence_visuals[i]["combined"] = comb
      video_writer.close()

    return rend_video_path

  def montage_visuals(self, visuals):
    overlaps = []
    overlaps.append(visuals["imgs"][0])
    kids = visuals["kids"]
    fid = visuals["frame_id"]
    for i in range(len(kids)):
      kid = kids[i]
      rend_img = visuals["rend_imgs"][i]
      h, w = rend_img.shape[:2]
      rend_img = cv2.putText(
          rend_img,
          f"kinect {kid}, frame {fid}",
          (w // 3, 30),
          cv2.FONT_HERSHEY_PLAIN,
          2,
          (0, 255, 255),
          2,
      )
      overlaps.append(rend_img)

    comb = np.concatenate(overlaps, 1)
    return comb
