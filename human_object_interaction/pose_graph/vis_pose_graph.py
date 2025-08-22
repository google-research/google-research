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

"""Vis pose graph.

sample command python vis_pose_graph.py --humor_out_path
/mnt/data/Research/humor/out/humor_test_reverse_sampling_behave_tree/eval_tree_sampling
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
# pylint: disable=line-too-long
# pylint: disable=g-importing-member
# pylint: disable=redefined-outer-name

import argparse
import os
import os.path as osp
from pathlib import Path
import pickle as pkl
import sys

import imageio
import matplotlib
from matplotlib import cm
import networkx as nx
import numpy as np
from PIL import ImageColor
import proplot as pplt
import trimesh

sys.path.append("/mnt/data/Research/humor/humor")

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
    color_sx = colormap(1.0 - (sx * 1.0) / sequence_len)
    color_sx = np.array(color_sx)
    color_sx[-1] = 0.25 + (sx * 1.0) * 0.75 / sequence_len
    color_sx = (color_sx * 255).astype(np.uint8)
    mesh, colors = color_mesh(mesh, color_sx)
    vertex_colors.append(colors)
    sequence_mesh.append(mesh)
    # sequence_mesh += mesh
  sequence_mesh = trimesh.util.concatenate(sequence_mesh)
  # write_mesh(sequence_mesh, 'test.ply')
  # breakpoint()
  # vertex_colors = np.concatenate(vertex_colors)
  # sequence_mesh = trimesh.Trimesh(vertices=sequence_mesh.vertices, faces=sequence_mesh.faces, vertex_colors=vertex_colors)
  # breakpoint()
  return sequence_mesh


def write_mesh(mesh, filepath):
  trimesh.exchange.export.export_mesh(mesh, filepath)
  return


def convert_sequences_to_pose_graph(
    sequence_path,
    outdir,
):

  tree_data_file = osp.join(sequence_path, "tree_sample.pkl")
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

  graph = nx.DiGraph()

  for connect in connectivity:
    graph.add_edge(connect[0], connect[1])

  max_td = np.max(tree_depth).astype(int)
  ## output as levels
  colormap = pro_colormaps["red"]

  if True:
    for tx in range(max_td + 1):
      tx_inds = np.where(tree_depth == tx)[0]
      color_tx = colormap(1.0 - (tx * 1.0) / max_td)
      color_tx = np.array(color_tx)
      color_tx = (color_tx * 255).astype(np.uint8)
      tx_meshes = []
      for ind in tx_inds:
        mesh = trimesh.Trimesh(
            vertices=vertices[ind,],
            faces=faces,
        )
        mesh, colors = color_mesh(mesh, color_tx)
        tx_meshes.append(mesh)

      tx_mesh = trimesh.util.concatenate(tx_meshes)
      write_mesh(tx_mesh, filepath=osp.join(outdir, f"level_{tx}.ply"))

  if True:
    ## write the object mesh to file.
    object_mesh_path = osp.join(outdir, "object_mesh.ply")
    obj_mesh = trimesh.Trimesh(tree_data["obj_vertices"],
                               tree_data["obj_faces"].astype(int))
    write_mesh(obj_mesh, object_mesh_path)

  body_mesh_out_dir = osp.join(outdir, "body_meshes")
  os.makedirs(body_mesh_out_dir, exist_ok=True)

  skel_mesh_out_dir = osp.join(outdir, "skel_meshes")
  os.makedirs(skel_mesh_out_dir, exist_ok=True)

  axes_mesh_out_dir = osp.join(outdir, "axes_meshes")
  os.makedirs(axes_mesh_out_dir, exist_ok=True)

  num_colors = len(colorlist)
  num_nodes = len(graph.nodes())
  edges = list(graph.edges)
  num_edges = len(edges)
  node_mesh_path_lst = [None] * num_nodes
  skel_mesh_path_lst = [None] * num_nodes
  axes_mesh_path_lst = [None] * num_nodes
  if False:
    for ix in range(num_nodes):
      node_mesh_path = osp.join("body_meshes", f"body_mesh_{ix:04d}.ply")
      node_mesh = trimesh.Trimesh(vertices=vertices[ix], faces=faces)
      td = int(tree_depth[ix])
      cx = td % num_colors
      colorname = colornames[cx]
      color_tx = ImageColor.getcolor(colorlist[colorname], "RGB")
      color_tx = np.array(color_tx)
      node_mesh, colors = color_mesh(node_mesh, color_tx)
      write_mesh(node_mesh, filepath=osp.join(outdir, node_mesh_path))
      node_mesh_path_lst[ix] = node_mesh_path

      skel_mesh_path = osp.join("skel_meshes", f"skel_mesh_{ix:04d}.ply")

      skel_mesh = trimesh.Trimesh(vertices=skel_vertices[ix], faces=skel_faces)
      skel_mesh, colors = color_mesh(skel_mesh, color_tx)
      write_mesh(skel_mesh, filepath=osp.join(outdir, skel_mesh_path))
      skel_mesh_path_lst[ix] = skel_mesh_path

      axes_mesh_path = osp.join("axes_meshes", f"axis_mesh_{ix:04d}.ply")
      axes_mesh = trimesh.Trimesh(
          vertices=axes_vertices[ix], faces=axes_faces, process=False)
      axes_mesh, _ = color_mesh(axes_mesh, axes_vertex_color[ix, :, 0:3])
      write_mesh(axes_mesh, filepath=osp.join(outdir, axes_mesh_path))
      axes_mesh_path_lst[ix] = axes_mesh_path

    ## creates nodes.txt
    graph_file = osp.join(outdir, "graph.txt")
    with open(graph_file, "w") as f:
      f.write(f"N {num_nodes}\n")
      f.write(f"E {num_edges}\n")

      for ix in range(num_nodes):
        f.write(
            f"{ix} {node_mesh_path_lst[ix]} {skel_mesh_path_lst[ix]} {axes_mesh_path_lst[ix]} \n"
        )

      for edge in edges:
        f.write(f"{edge[0]} {edge[1]}\n")

    axes_locations_file = osp.join(outdir, "axes.txt")
    with open(axes_locations_file, "w") as f:
      trans = Jtrs[:, 0]
      root_orient = tree_data["global_orient"][:, 0]
      for ix in range(num_nodes):
        pose_str = f"{trans[ix][0]} {trans[ix][1]} {trans[ix][2]} {root_orient[ix][0]} {root_orient[ix][1]} {root_orient[ix][2]}\n"
        f.write(pose_str)
  return


def main(cfg):
  base_dir = Path(cfg.humor_out_path)

  sequence_prefixes = [
      k for k in os.listdir(base_dir) if "sequence_viz" not in k
  ]

  if cfg.viz_out_path is None:
    outdir = osp.join(base_dir.parent, Path("pose_graph"))
  else:
    outdir = Path(cfg.viz_out_path)

  for sequence_base in sequence_prefixes:
    sequence_path = osp.join(base_dir, sequence_base)
    club_outdir = osp.join(outdir, sequence_base)
    os.makedirs(club_outdir, exist_ok=True)
    convert_sequences_to_pose_graph(
        sequence_path=sequence_path, outdir=club_outdir)
  return


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--humor_out_path",
      type=str,
      default="/mnt/data/Research/humor/out/humor_test_reverse_sampling_behave_tree_colored_big/eval_tree_sampling/",
      help="Root directory for humor generated tree",
  )
  parser.add_argument(
      "--viz_out_path",
      type=str,
      default=None,
      help="Root directory for humor samples",
  )
  parser.add_argument(
      "--sequence-prefix", type=str, default=None, help="sequence prefix")

  cfg1 = parser.parse_known_args()
  cfg1 = cfg1[0]

  main(cfg1)
