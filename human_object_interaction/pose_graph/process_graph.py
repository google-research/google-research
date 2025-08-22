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

# pylint: disable=line-too-long
"""Process the pose graph.

Sample command python pose_graph/process_graph.py --humor_out_path
/mnt/data/Research/humor/out/humor_test_reverse_sampling_behave_tree_colored_big/eval_tree_sampling/
"""

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
# pylint: disable=g-importing-member
# pylint: disable=g-explicit-length-test
# pylint: disable=unexpected-keyword-arg

import argparse
import os
import os.path as osp
from pathlib import Path
import pdb
import pickle as pkl
import sys

from body_model.utils import SMPLH_PATH, SMPL_JOINTS
from humor.body_model.body_model import BodyModel
from humor.utils import torch as torch_utils
from humor.utils import transforms
import imageio
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PIL import ImageColor
import proplot as pplt
from psbody.mesh import Mesh
import ray
import torch
import trimesh

# from . import meshviewer

colorlist = {
    "red": "#FF0000",
    "cyan": "#00FFFF",
    "blue": "#0000FF",
    "dark_blue": "#00008B",
    "light_blue": "#ADD8E6",
    "purple": "#800080",
    "yellow": "#FFFF00",
    "lime": "#00FF00",
    "magenta": "#FF00FF",
    "pink": "#FFC0CB",
    "grey": "#808080",
    "orange": "#FFA500",
    "brown": "#A52A2A",
    "maroon": "#800000",
    "green": "#008000",
    "olive": "#808000",
    "aquamarine": "#7FFFD4",
    "chocolate": "#D2691E",
}

sequential_color_maps = colornames = list(colorlist.keys())
pro_colormaps = {}
for key in colornames:
  color = colorlist[key]
  pro_colormaps[key] = pplt.Colormap(
      color, l=100, name=f"linear_{key}", space="hpl")

sys.path.append("/mnt/data/Research/humor")
sys.path.append("/mnt/data/Research/humor/humor")

os.environ["PYOPENGL_PLATFORM"] = "osmesa"

NUM_BETAS = 10
n2t = torch_utils.numpy2torch
c2c = torch_utils.copy2cpu
root_orient = n2t(np.array([0, 0, 0])[None,])


def get_body_directions(root_or, pelvis_xyz, length=1, projection=True):
  B, _ = root_or.shape
  zdir = length * np.array([0, 0, 1])
  zdir = zdir[None,] * np.ones((B, 1))
  zdir = zdir[Ellipsis, None]
  root_mats = transforms.convert_to_rotmat(n2t(root_or[:, None]))
  root_mats = root_mats.reshape(B, 3, 3)

  directions = np.matmul(c2c(root_mats), zdir)[Ellipsis, 0]

  directions = directions + pelvis_xyz
  if projection:
    return directions[:, 0:2]
  return directions


def color_mesh(mesh, mesh_col):

  # mesh_color = trimesh.visual.ColorVisuals(mesh, vertex_colors=color)
  # # mesh.visual = mesh_colore
  # mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, visuals=mesh_color)
  # breakpoint()

  nverts = len(mesh.vertices)
  if len(mesh_col.shape) == 1:
    vertex_colors = mesh_col[None,] * np.ones((nverts, 1))
  else:
    vertex_colors = mesh_col * 1
    trimesh.visual.color.ColorVisuals(mesh, vertex_colors=vertex_colors)

  mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, vertex_colors=vertex_colors)
  return mesh, vertex_colors


def convert_seqeunce_to_mesh(sequence_path, colormap_name, body_mesh=True):
  colormap = pro_colormaps[colormap_name]
  # colormap = cm.get_cmap(colormap_name)
  sequence_file = osp.join(sequence_path, "seq_output.pkl")
  with open(sequence_file, "rb") as f:
    sequence_data = pkl.load(f)

  sequence_len = len(sequence_data["vertices"])
  if body_mesh:
    vertices = sequence_data["vertices"]
    faces = sequence_data["faces"]
  else:
    vertices = sequence_data["skel_vertices"]
    faces = sequence_data["skel_faces"]

  sequence_mesh = []
  vertex_colors = []
  for sx in range(0, sequence_len, 10):
    mesh = trimesh.Trimesh(vertices=vertices[sx], faces=faces)
    sx_color = colormap(1.0 - (sx * 1.0) / sequence_len)
    sx_color = np.array(sx_color)
    sx_color[-1] = 0.25 + (sx * 1.0) * 0.75 / sequence_len
    sx_color = (sx_color * 255).astype(np.uint8)
    mesh, colors = color_mesh(mesh, sx_color)
    vertex_colors.append(colors)
    sequence_mesh.append(mesh)
    # sequence_mesh += mesh
  sequence_mesh = trimesh.util.concatenate(sequence_mesh)
  return sequence_mesh


def compute_intersection_with_object(query_pts, smpl_mesh):
  proxmity_struct = trimesh.proximity.ProximityQuery(smpl_mesh)
  sdf_values = proxmity_struct.signed_distance(query_pts)
  return sdf_values


@ray.remote(num_cpus=1)
def sdf_ray_workers(smpl_mesh_vertices, smpl_mesh_faces, query_pts):
  smpl_mesh = trimesh.Trimesh(
      vertices=smpl_mesh_vertices, faces=smpl_mesh_faces)
  sdf_values = compute_intersection_with_object(query_pts, smpl_mesh)
  return sdf_values


def check_if_mesh_bbox_intersect(smpl_mesh, object_mesh):
  contains = np.array(
      trimesh.bounds.contains(smpl_mesh.bounds, object_mesh.vertices))
  return contains.sum() > 0


def write_mesh(mesh, filepath):
  trimesh.exchange.export.export_mesh(mesh, filepath)
  return


INSIDE_VERTICES_THRESHOLD = 200


def draw_tree_graph_with_node_pos(tree_data, graph, save_path):
  node_ids = list(graph.nodes())
  node_transl = tree_data["transl"]
  node_tree_depth = tree_data["tree_depth"]
  pos = {ix: node_transl[ix, 0, 0:2] for ix in node_ids}
  fig, ax = plt.subplots(figsize=(15, 10))
  ax.grid(True, which="both")
  ax.set_xlim(np.min(node_transl[:, 0, 0]), np.max(node_transl[:, 0, 0]))
  ax.set_ylim(np.min(node_transl[:, 0, 1]), np.max(node_transl[:, 0, 1]))

  node_colormap = cm.get_cmap("plasma")
  node_color_depth = node_colormap(node_tree_depth / (max(node_tree_depth) + 1))
  node_color_depth = node_color_depth[np.array(node_ids)]

  nx.draw(
      graph,
      pos=pos,
      with_labels=False,
      ax=ax,
      alpha=0.4,
      node_size=5,
      node_color=node_color_depth,
  )
  plt.savefig(save_path)


def draw_tree_nodes(tree_data, graph, save_path):
  node_ids = list(graph.nodes())
  node_transl = tree_data["transl"]
  node_tree_depth = tree_data["tree_depth"]
  pos = {ix: node_transl[ix, 0, 0:2] for ix in node_ids}
  fig, ax = plt.subplots(figsize=(15, 10))
  ax.grid(True, which="both")
  ax.set_xlim(np.min(node_transl[::-1, 0, 0]), np.max(node_transl[:, 0, 0]))
  ax.set_ylim(np.min(node_transl[::-1, 0, 1]), np.max(node_transl[:, 0, 1]))
  node_colormap = cm.get_cmap("plasma")
  node_color_depth = node_colormap(node_tree_depth / (max(node_tree_depth) + 1))
  node_color_depth = node_color_depth[np.array(node_ids)]
  nx.draw_networkx_nodes(
      graph,
      pos=pos,
      nodelist=node_ids,
      ax=ax,
      alpha=0.4,
      node_size=10,
      node_color=node_color_depth,
  )
  # plt.savefig('nodes.svg')
  plt.savefig(save_path)


def draw_tree_nodes_with_direction(tree_data, graph, save_path):
  node_ids = np.array(list(graph.nodes()))
  node_transl = tree_data["transl"]
  node_global_orient = tree_data["global_orient"]
  node_tree_depth = tree_data["tree_depth"]
  directions = get_body_directions(
      node_global_orient[:, 0], pelvis_xyz=node_transl[:, 0], length=0.05)
  node_positions = node_transl[:, 0, 0:2]
  fig, ax = plt.subplots(figsize=(15, 10))
  node_ordering = node_ids[::-1]
  # node_ordering = [k for k in range(10)][::-1]

  arrow_start_location = node_transl[:, 0] * 1
  arrow_end_locations = directions
  node_colormap = cm.get_cmap("plasma")
  node_color_depth = node_colormap(node_tree_depth / (max(node_tree_depth) + 1))

  for ix in node_ordering:
    color_ix = node_color_depth[ix]
    xy = node_positions[ix]
    end = arrow_end_locations[ix]
    ax.scatter(xy[0], xy[1], color=color_ix, marker=".")
    ax.arrow(
        xy[0],
        xy[1],
        end[0] - xy[0],
        end[1] - xy[1],
        color=color_ix,
        head_width=0.005)

  # plt.savefig('node_with_arrow.svg')
  plt.savefig(save_path)


def convert_sequences_to_pose_graph(tree_data_file,
                                    outdir,
                                    outname="tree_sample_processed.pkl",
                                    overwrite=False):

  # tree_data_file = osp.join(sequence_path, "tree_sample.pkl")
  with open(tree_data_file, "rb") as f:
    tree_data = pkl.load(f)

  vertices = tree_data["vertices"][:, 0]
  faces = tree_data["faces"]

  Jtrs = tree_data["Jtr"]

  skel_vertices = tree_data["skel_vertices"]
  skel_faces = tree_data["skel_faces"]

  axes_vertices = tree_data["axes_vertices"]
  axes_faces = tree_data["axes_faces"]
  axes_vertex_color = tree_data["axes_vertex_colors"]
  tree_depth = tree_data["tree_depth"]
  connectivity = tree_data["connectivity"]
  nodes_lst = tree_data["nodes_lst"]
  graph = nx.DiGraph()

  for connect in connectivity:
    graph.add_edge(connect[0], connect[1])

  if True:
    ## write the object mesh to file.
    object_mesh_path = osp.join(outdir, "object_mesh.ply")
    obj_mesh = trimesh.Trimesh(tree_data["obj_vertices"],
                               tree_data["obj_faces"])
    write_mesh(obj_mesh, object_mesh_path)

  sdf_file_path = osp.join(outdir, "sdf_values.npz")
  num_nodes = vertices.shape[0]
  overwrite = True
  if not osp.exists(sdf_file_path) or overwrite:
    sdf_results = []
    easy_sdf = []
    for ix in range(num_nodes):
      smpl_mesh = trimesh.Trimesh(vertices[ix], faces)
      contains = check_if_mesh_bbox_intersect(smpl_mesh, obj_mesh)
      easy_sdf.append(contains)

    obj_vertices = tree_data["obj_vertices"]
    sdf_obj_values = np.zeros((num_nodes, len(obj_vertices)))

    indexes = []
    for ix in range(num_nodes):
      if easy_sdf[ix]:
        smpl_mesh_vertices = vertices[ix]
        smpl_mesh_faces = faces
        sdf_results.append(
            sdf_ray_workers.remote(
                smpl_mesh_vertices=smpl_mesh_vertices,
                smpl_mesh_faces=smpl_mesh_faces,
                query_pts=obj_vertices,
            ))
        indexes.append(ix)

    all_sdf_values = ray.get(sdf_results)
    all_sdf_values = np.stack(all_sdf_values)
    for ind, sdf_ind_value in zip(indexes, all_sdf_values):
      sdf_obj_values[ind] = sdf_ind_value

    all_sdf_values = sdf_obj_values
    np.savez(sdf_file_path, sdf_values=all_sdf_values)
  else:
    all_sdf_values = np.load(sdf_file_path)["sdf_values"]

  ## SDF values that are positive means those are inside
  vertices_inside_smpl_mask = all_sdf_values > 0
  vertex_count_intersecting = vertices_inside_smpl_mask.sum(1)

  intersecting_pose_nodes_ids = np.where(
      vertex_count_intersecting > INSIDE_VERTICES_THRESHOLD)[0]
  intersect_counts = vertex_count_intersecting[intersecting_pose_nodes_ids]
  non_intersecting_pose_nodes_ids = np.where(
      vertex_count_intersecting <= INSIDE_VERTICES_THRESHOLD)[0]
  non_intersect_counts = vertex_count_intersecting[
      non_intersecting_pose_nodes_ids]

  tree_data["sdf_values"] = all_sdf_values
  # tree_data["invalid_nodes"] = intersecting_pose_nodes_ids

  invalid_nodes = []

  nodes_to_check = list(intersecting_pose_nodes_ids)
  invalid_nodes.extend(nodes_to_check)
  nodes_to_check = set(nodes_to_check)

  while len(nodes_to_check) > 0:
    node_id = nodes_to_check.pop()
    children = list(graph.neighbors(node_id))
    invalid_nodes.extend(list(children))
    for c in children:
      nodes_to_check.add(c)

  invalid_nodes = set(invalid_nodes)

  keep_nodes = set()
  for c in connectivity.reshape(-1):
    if c not in invalid_nodes:
      keep_nodes.add(c)

  processed_tree_file = osp.join(outdir, outname)

  new_graph = nx.DiGraph()

  for connect in connectivity:
    if (connect[0] not in invalid_nodes) and (connect[1] not in invalid_nodes):
      new_graph.add_edge(connect[0], connect[1])

  new_nodes = set(new_graph.nodes())
  old_nodes = set(graph.nodes())
  tree_data["invalid_nodes"] = np.array(list(invalid_nodes))
  if False:
    draw_tree_nodes_with_direction(
        tree_data,
        new_graph,
        save_path=osp.join(outdir, "new_nodes_heading.svg"))
    draw_tree_graph_with_node_pos(
        tree_data, graph, save_path=osp.join(outdir, "old_graph.svg"))
    draw_tree_graph_with_node_pos(
        tree_data, new_graph, save_path=osp.join(outdir, "new_graph.svg"))
    draw_tree_nodes(
        tree_data, graph, save_path=osp.join(outdir, "old_nodes.svg"))
    draw_tree_nodes(
        tree_data, new_graph, save_path=osp.join(outdir, "new_nodes.svg"))
    draw_tree_nodes_with_direction(
        tree_data, graph, save_path=osp.join(outdir, "old_nodes_heading.svg"))

  og_leaves = [v for v, d in graph.out_degree() if d == 0]
  paths = list(range(len(og_leaves)))
  new_leaves = [v for v, d in new_graph.out_degree() if d == 0]

  path_mapping = {nid: paths[i] for i, nid in enumerate(og_leaves)}
  tree_data["path_mapping"] = path_mapping
  with open(processed_tree_file, "wb") as f:
    pkl.dump(tree_data, f)

  mesh_txt_file = osp.join(outdir, "good_processed_mesh.txt")

  with open(mesh_txt_file, "w") as f:
    for k, value in path_mapping.items():
      if k in new_leaves:
        f.write(f"body_mesh_{value}.ply skel_mesh_{value}.ply\n")

  mesh_txt_file = osp.join(outdir, "bad_processed_mesh.txt")

  with open(mesh_txt_file, "w") as f:
    for k, value in path_mapping.items():
      if k not in new_leaves:
        f.write(f"body_mesh_{value}.ply skel_mesh_{value}.ply\n")
  return


def main(cfg):
  base_dir = Path(cfg.humor_out_path)

  sequence_prefixes = [
      k for k in os.listdir(base_dir) if "sequence_viz" not in k
  ]

  if cfg.out_path is None:
    outdir = osp.join(base_dir.parent, Path("pose_graph_processed"))
  else:
    outdir = Path(cfg.out_path)

  for sequence_base in sequence_prefixes:
    sequence_path = osp.join(base_dir, sequence_base)
    club_outdir = osp.join(outdir, sequence_base)
    os.makedirs(club_outdir, exist_ok=True)
    tree_data_file = osp.join(sequence_path, "tree_sample.pkl")
    convert_sequences_to_pose_graph(
        tree_data_file=tree_data_file,
        outdir=club_outdir,
        overwrite=config.overwrite)
  return


if __name__ == "__main__":
  ray.init(num_cpus=10)
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--humor_out_path",
      type=str,
      default="/mnt/data/Research/humor/out/humor_test_reverse_sampling_behave_tree_colored/eval_tree_sampling/",
      help="Root directory for humor generated tree",
  )
  parser.add_argument(
      "--out-path",
      type=str,
      default=None,
      help="Root directory for humor samples")
  parser.add_argument(
      "--overwrite",
      action="store_true",
      default=None,
      help="Recompute SDF data")

  config = parser.parse_known_args()
  config = config[0]

  main(config)
  ray.shutdown()
