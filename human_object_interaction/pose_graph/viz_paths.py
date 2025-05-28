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

"""Viz paghs in the tree."""
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
# pylint: disable=pointless-string-statement
# pylint: disable=g-importing-member
# pylint: disable=redefined-outer-name

import argparse
import os
import os.path as osp
from pathlib import Path
import pdb
import pickle as pkl
import sys

import matplotlib
from matplotlib import cm
import numpy as np
import proplot as pplt
import trimesh

# from . import meshviewer

# sequential_color_maps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds']

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


def club_sequences_by_samples(base_sample_dir):

  ## this directory contains various samples generated from sequences, the oragnization is by folder.

  all_sample_dirs_names = [
      f for f in os.listdir(base_sample_dir)
      if osp.isdir(osp.join(base_sample_dir, f))
  ]

  base_dirs = set()
  prefix = "path_"
  for sample_dir in all_sample_dirs_names:
    sample_breakindex = sample_dir.find(prefix)
    base_dir = sample_dir[:sample_breakindex]
    base_dirs.add(base_dir)

  sequences = {}
  for base_dir in base_dirs:
    sequences[base_dir] = []
    for sample_dir_name in all_sample_dirs_names:
      if sample_dir_name.startswith(base_dir):
        if osp.exists(
            osp.join(base_sample_dir, sample_dir_name, "seq_output.pkl")):
          sequences[base_dir].append(sample_dir_name)

    paths = list(sequences[base_dir])
    numbers = np.argsort(
        np.array([int(s.replace(prefix, "")) for s in sequences[base_dir]]))
    paths = [paths[ix] for ix in numbers]

  return sequences


def write_mesh(mesh, filepath):
  trimesh.exchange.export.export_mesh(mesh, filepath)
  return


def color_mesh(mesh, mesh_col):

  # mesh_color = trimesh.visual.ColorVisuals(mesh, vertex_colors=color)
  # # mesh.visual = mesh_colore
  # mesh = trimesh.Trimesh(mesh.vertices, mesh.faces, visuals=mesh_color)
  # breakpoint()
  nverts = len(mesh.vertices)
  vertex_colors = mesh_col[None,] * np.ones((nverts, 1))
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
    color_tx = colormap(1.0 - (sx * 1.0) / sequence_len)
    color_tx = np.array(color_tx)
    color_tx[-1] = 0.25 + (sx * 1.0) * 0.75 / sequence_len
    color_tx = (color_tx * 255).astype(np.uint8)
    mesh, colors = color_mesh(mesh, color_tx)
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


def convert_multiple_sequences_to_mesh(sequence_path_lst, outdir, indices=None):
  all_sample_meshes = []
  body_paths = []
  skel_paths = []
  if indices is None:
    indices = range(0, len(sequence_path_lst))

  for sx in indices:
    # assert sx < len(sequential_color_maps), 'can only handle {} sequences per state'.format(len(sequential_color_maps))
    sequence_path = sequence_path_lst[sx]
    color_sx = sx % len(sequential_color_maps)
    colormap_name = sequential_color_maps[color_sx]
    body_sequence_colored_mesh = convert_seqeunce_to_mesh(
        sequence_path=sequence_path, colormap_name=colormap_name)
    output_path = osp.join(outdir, f"body_mesh_{sx}.ply")
    write_mesh(body_sequence_colored_mesh, output_path)
    body_paths.append(output_path)

    skel_sequence_colored_mesh = convert_seqeunce_to_mesh(
        sequence_path=sequence_path,
        colormap_name=colormap_name,
        body_mesh=False)
    output_path = osp.join(outdir, f"skel_mesh_{sx}.ply")
    write_mesh(skel_sequence_colored_mesh, output_path)
    skel_paths.append(output_path)
    if sx % 100 == 0:
      print(f"{sx} / {len(sequence_path_lst)} ")

  mesh_file_index = osp.join(outdir, f"mesh_{len(indices)}.txt")
  with open(mesh_file_index, "w") as f:
    for p1, p2 in zip(body_paths, skel_paths):
      f.write(f"{osp.basename(p1)} {osp.basename(p2)}\n")
  return mesh_file_index
  #   all_sample_meshes.append(sequence_colored_mesh)

  # all_sample_sequence_mesh = trimesh.util.concatenate(all_sample_meshes)
  # write_mesh(all_sample_sequence_mesh, 'test.ply')
  # breakpoint()
  # return all_sample_sequence_mesh


def main(config):
  base_dir = Path(config.humor_out_path)

  tree_prefixes = os.listdir(config.humor_out_path)

  clubbed_sequences = {}
  for tree_prefix in tree_prefixes:
    temp = club_sequences_by_samples(
        osp.join(config.humor_out_path, tree_prefix))

    clubbed_sequences[tree_prefix] = temp[""]

  if config.viz_out_path is None:
    outdir = osp.join(base_dir.parent, Path("sequence_viz"))
  else:
    outdir = Path(config.viz_out_path)

  input_sequence_prefix = config.sequence_prefix
  sequence_prefixes = list(clubbed_sequences.keys())

  indices = None
  if config.sample_index_file is not None:
    index_path = Path(config.sample_index_file)
    with open(index_path) as f:
      indices = [int(l.strip()) for l in f.readlines()]

  max_n_seq = 1000

  for club_base in sequence_prefixes[:1]:
    sequences = clubbed_sequences[club_base]
    num_sequences = len(sequences)
    if True:
      if len(sequences) > max_n_seq:
        sequences = [
            sequences[i] for i in np.random.choice(num_sequences, max_n_seq)
        ]

    sequence_path_lst = [
        osp.join(base_dir, club_base, spath) for spath in sequences
    ]

    club_outdir = osp.join(outdir, club_base)
    os.makedirs(club_outdir, exist_ok=True)
    output_mesh = convert_multiple_sequences_to_mesh(
        sequence_path_lst=sequence_path_lst,
        outdir=club_outdir,
        indices=indices)

  return


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--humor_out_path",
      type=str,
      default="/mnt/data/Research/humor/out/humor_test_sampling_behave_4/eval_sampling",
      help="Root directory for humor samples",
  )
  parser.add_argument(
      "--viz_out_path",
      type=str,
      default=None,
      help="Root directory for humor samples",
  )
  # parser.add_argument('--smplh-root', type=str, default='./body_models/smplh', help='Root directory of the SMPL+H body model.')
  parser.add_argument(
      "--sequence-prefix", type=str, default=None, help="sequence prefix")
  parser.add_argument(
      "--sample-index-file", type=str, default=None, help="sample_indices_file")

  config = parser.parse_known_args()
  config = config[0]

  main(config)
"""Sample command
python humor_viz/humor_sample_vis.py
--humor_out_path=/mnt/data/Research/humor/out/humor_test_reverse_sampling_behave_2/eval_sampling/
--sequence-prefix=behave_processed_mocap_Date02_Sub02_chairwood_sit_poses_46_frames_1_fps_b0seq0
--sample-index-file=/mnt/data/Research/humor/out/humor_test_reverse_sampling_behave_2/sequence_viz/behave_processed_mocap_Date02_Sub02_chairwood_sit_poses_46_frames_1_fps_b0seq0.index


python humor_viz/humor_sample_vis.py
--humor_out_path=/mnt/data/Research/humor/out/humor_test_reverse_sampling_behave_2/eval_sampling/
--sequence-prefix=behave_processed_mocap_Date02_Sub02_chairwood_sit_poses_46_frames_1_fps_b0seq0
--sample-index-file=/mnt/data/Research/humor/out/humor_test_reverse_sampling_behave_2/sequence_viz/behave_processed_mocap_Date02_Sub02_chairwood_sit_poses_46_frames_1_fps_b0seq0.

"""
