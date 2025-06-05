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

from collections import namedtuple
import glob
import re
import stat
from typing import cast

WASHINGTON_OSM_DIR = "WASHINGTON-PROCESSED DIRECTORY NAME"
BADEN_OSM_DIR = "BADEN-PROCESSED DIRECTORY NAME"
# OSM_DIR = "*-PROCESSED DIRECTORY NAME"

# PENALTY_MODE = "most_important_arc_only_simple_version_A"
PENALTY_MODE = "important_arcs_only_simple_version_A"
# PENALTY_MODE = "whole_path"

PENALTY_MULTIPLIER = 1.1 if (PENALTY_MODE == "whole_path") else 100000
OTHER_MULTIPLIER = 2
PATH_ONLY_OTHER_MULTIPLIER = 1
NUM_PATHS = 10
T_UPPER_BOUND = 10
USE_NORMALIZED_OBJECTIVE = True
ALL_QUERY_SETS = [
    (
        WASHINGTON_OSM_DIR,
        "seattle_medium_queries_random_300_part_1.tsv",
        "Seattle",
    ),
    (WASHINGTON_OSM_DIR, "short_queries_random_100.tsv", "WA Short"),
    (WASHINGTON_OSM_DIR, "medium_queries_random_100.tsv", "WA Medium"),
    (WASHINGTON_OSM_DIR, "long_queries_random_100.tsv", "WA Long"),
    (BADEN_OSM_DIR, "short_queries_random_100.tsv", "Baden Short"),
    (BADEN_OSM_DIR, "medium_queries_random_100.tsv", "Baden Medium"),
    (BADEN_OSM_DIR, "long_queries_random_100.tsv", "Baden Long"),
]

Args = namedtuple(
    "Args",
    [
        "osm_dir",
        "penalty_mode",
        "penalty_multiplier",
        "other_multiplier",
        "path_only_other_multiplier",
        "num_paths",
        "t_upper_bound",
        "use_normalized_objective",
    ],
)
DEFAULT_ARGS_PARTIAL = Args(
    osm_dir="NONE",
    penalty_mode=PENALTY_MODE,
    penalty_multiplier=PENALTY_MULTIPLIER,
    other_multiplier=OTHER_MULTIPLIER,
    path_only_other_multiplier=PATH_ONLY_OTHER_MULTIPLIER,
    num_paths=NUM_PATHS,
    t_upper_bound=T_UPPER_BOUND,
    use_normalized_objective=USE_NORMALIZED_OBJECTIVE,
)


# helper functions


def paths_subdir(args):
  return (
      args.osm_dir
      + "/"
      + "scenario_path_store__"
      + args.penalty_mode
      + "__penalty_"
      + str(args.penalty_multiplier)
  )


def suffix(args):
  return (
      args.penalty_mode
      + "__p_"
      + str(args.penalty_multiplier)
      + ("_uno" if args.use_normalized_objective else "")
      + "_tub_"
      + str(args.t_upper_bound)
      + "_om_"
      + str(args.other_multiplier)
      + "_poom_"
      + str(args.path_only_other_multiplier)
      + "_npp_"
      + str(args.num_paths)
  )


def cuts_subdir(args):
  return args.osm_dir + "/" + "scenario_cut_store__" + suffix(args)


def penalized_arcs_filename(args, source, target, i):
  return (
      paths_subdir(args)
      + "/"
      + "penalized_arcs__src_"
      + str(source)
      + "__dst_"
      + str(target)
      + "__num_"
      + str(i)
      + ".tsv"
  )


def read_paths(args, source, target):
  paths = []
  for i in range(args.num_paths):
    filename = (
        paths_subdir(args)
        + "/"
        + "path__src_"
        + str(source)
        + "__dst_"
        + str(target)
        + "__num_"
        + str(i)
        + ".tsv"
    )
    with open(filename, "r") as f:
      lines = f.readlines()

    path = []
    for line in lines[1:-1]:
      (is_forward, node, index) = line.split("\t")
      assert int(is_forward) == 1
      path.append((int(node), int(index)))
    paths.append(path)
  assert len(paths) == args.num_paths
  return paths


def read_penalized_arcs(args, source, target):
  all_penalized_arcs = []
  for i in range(1, args.num_paths):
    filename = penalized_arcs_filename(args, source, target, i)
    with open(filename, "r") as f:
      lines = f.readlines()

    penalized_arcs_i = []
    for line in lines[1:]:
      (_, node, index) = line.split("\t")
      penalized_arcs_i.append((int(node), int(index)))
    all_penalized_arcs.append(penalized_arcs_i)
  return all_penalized_arcs


def get_paths_arc_set(args, source, target):
  paths = read_paths(args, source, target)[
      : (args.num_paths - 1)
  ]  # leave out last path, which is path to explain
  arc_set = set()
  for path in paths:
    for arc in path:
      arc_set.add(arc)
  return arc_set


def has_explanation(args, source, target):
  regex = (
      cuts_subdir(args)
      + "/"
      + "cycle__src_"
      + str(source)
      + "__dst_"
      + str(target)
      + "__cycle_*.tsv"
  )
  files = glob.glob(regex)
  assert len(files) <= 1
  return len(files) == 1


def read_explanation(args, source, target):
  regex = (
      cuts_subdir(args)
      + "/"
      + "cycle__src_"
      + str(source)
      + "__dst_"
      + str(target)
      + "__cycle_*.tsv"
  )
  files = glob.glob(regex)
  assert len(files) == 1
  filename = files[0]
  with open(filename, "r") as f:
    lines = f.readlines()

  penalized_arcs = set()
  for line in lines[1:]:
    (node, index, _) = line.split("\t")
    penalized_arcs.add((int(node), int(index)))
  return penalized_arcs


def read_valid_queries(args, filename):
  with open(filename, "r") as f:
    lines = f.readlines()

  queries = []
  for line in lines[1:]:
    (source, target, _, _, _, _, _) = line.split("\t")
    if has_explanation(args, source, target) and (
        len(read_explanation(args, source, target)) > 0
    ):
      queries.append((int(source), int(target)))
  return queries


def get_stats(values):
  sorted_values = sorted(values)
  DIGITS = 3
  minval = sorted_values[0]
  twoth = sorted_values[int(0.02 * len(sorted_values))]
  tenth = sorted_values[int(0.1 * len(sorted_values))]
  median = sorted_values[int(0.5 * len(sorted_values))]
  ninetieth = sorted_values[int(0.9 * len(sorted_values))]
  ninetyeighth = sorted_values[int(0.98 * len(sorted_values))]
  maxval = sorted_values[-1]
  numl1 = len([v for v in values if v < 1.0])
  fracl1 = numl1 / len(values)

  minval = round(minval, DIGITS)
  twoth = round(twoth, DIGITS)
  tenth = round(tenth, DIGITS)
  median = round(median, DIGITS)
  ninetieth = round(ninetieth, DIGITS)
  ninetyeighth = round(ninetyeighth, DIGITS)
  maxval = round(maxval, DIGITS)
  fracl1 = round(fracl1, DIGITS)

  Stats = namedtuple(
      "Stats",
      [
          "count",
          "min",
          "twoth",
          "tenth",
          "median",
          "ninetieth",
          "ninetyeighth",
          "max",
          "numl1",
          "fracl1",
      ],
  )
  return Stats(
      len(values),
      minval,
      twoth,
      tenth,
      median,
      ninetieth,
      ninetyeighth,
      maxval,
      numl1,
      fracl1,
  )


# statistics-generating methods


def get_frac_exp_of_paths(args, queries_fname):
  queries = read_valid_queries(args, queries_fname)
  ratios = []
  for source, target in queries:
    paths_arc_set = get_paths_arc_set(args, source, target)
    explanation_arcs = read_explanation(args, source, target)
    ratio = len(explanation_arcs) / len(paths_arc_set)
    ratios.append(ratio)
  return get_stats(ratios)


def get_frac_exp_in_paths(args, queries_fname):
  queries = read_valid_queries(args, queries_fname)
  ratios = []
  for source, target in queries:
    paths_arc_set = get_paths_arc_set(args, source, target)
    explanation_arcs = read_explanation(args, source, target)
    ratio = len(explanation_arcs.intersection(paths_arc_set)) / len(
        explanation_arcs
    )
    ratios.append(ratio)
  return get_stats(ratios)


def get_frac_exp_in_penalized(args, queries_fname):
  queries = read_valid_queries(args, queries_fname)
  ratios = []
  for source, target in queries:
    penalized_arc_set = set()
    for l in read_penalized_arcs(args, source, target):
      for arc in l:
        penalized_arc_set.add(arc)
    explanation_arcs = read_explanation(args, source, target)
    exp_arcs_in_penalized = explanation_arcs.intersection(penalized_arc_set)
    ratio = len(exp_arcs_in_penalized) / len(explanation_arcs)
    ratios.append(ratio)
  return get_stats(ratios)


def get_num_penalty_stretches_hit(args, queries_fname):
  queries = read_valid_queries(args, queries_fname)
  intersection_counts = []
  for source, target in queries:
    penalized_stretches = [
        set(l) for l in read_penalized_arcs(args, source, target)
    ]
    paths_arc_set = get_paths_arc_set(args, source, target)
    num_intersections = sum(
        [(len(s.intersection(paths_arc_set)) > 0) for s in penalized_stretches]
    )
    intersection_counts.append(num_intersections)
  return get_stats(intersection_counts)


# main methods


def whole_path_stats():
  print("FRAC_EXP_IN_PATHS")
  for region, queries_fname, region_name in ALL_QUERY_SETS:
    queries_path = region + "/" + queries_fname
    print(f"{region_name=}")
    args = Args(**{**DEFAULT_ARGS_PARTIAL._asdict(), "osm_dir": region})
    print(get_frac_exp_in_paths(args, queries_path))
  print("FRAC_EXP_OF_PATHS")
  for region, queries_fname, region_name in ALL_QUERY_SETS:
    queries_path = region + "/" + queries_fname
    print(f"{region_name=}")
    args = Args(**{**DEFAULT_ARGS_PARTIAL._asdict(), "osm_dir": region})
    print(get_frac_exp_of_paths(args, queries_path))


def non_whole_path_stats():
  print("NUM_PENALTY_STRETCHES_HIT")
  for region, queries_fname, region_name in ALL_QUERY_SETS:
    queries_path = region + "/" + queries_fname
    print(f"{region_name=}")
    args = Args(**{**DEFAULT_ARGS_PARTIAL._asdict(), "osm_dir": region})
    print(get_num_penalty_stretches_hit(args, queries_path))
  print("FRAC_EXP_IN_PENALIZED")
  for region, queries_fname, region_name in ALL_QUERY_SETS:
    queries_path = region + "/" + queries_fname
    print(f"{region_name=}")
    args = Args(**{**DEFAULT_ARGS_PARTIAL._asdict(), "osm_dir": region})
    print(get_frac_exp_in_penalized(args, queries_path))


# if PENALTY_MODE == "whole_path":
#   whole_path_stats()
# else:
#   non_whole_path_stats()

# table-generating methods


def print_important_arcs_only_table(poom):
  print("\\begin{tabular}{|l|c|c|c|c|c|}")
  print("\\hline")
  print(
      "\\% (\\textbf{exp} $\\subseteq$ \\textbf{deleted})  & \\%"
      " \\textbf{valid} & \\multicolumn{2}{c|}{\\textbf{SVE}} &"
      " \\multicolumn{2}{c|}{\\textbf{PBE}}\\\\ \\hline"
  )
  print("\\# paths & $\\le 10$ & 2 & 10 & 2 & 10 \\\\ \\hline")

  pbe_value = 100 if poom == 1 else 0

  for region, queries_fname, region_name in ALL_QUERY_SETS:
    queries_path = region + "/" + queries_fname
    args_2 = Args(**{
        **DEFAULT_ARGS_PARTIAL._asdict(),
        "osm_dir": region,
        "num_paths": 2,
        "penalty_multiplier": 100000,
        "penalty_mode": "important_arcs_only_simple_version_A",
        "path_only_other_multiplier": poom,
    })
    stats_2 = get_frac_exp_in_penalized(args_2, queries_path)
    args_10 = Args(**{
        **DEFAULT_ARGS_PARTIAL._asdict(),
        "osm_dir": region,
        "num_paths": 10,
        "penalty_multiplier": 100000,
        "penalty_mode": "important_arcs_only_simple_version_A",
        "path_only_other_multiplier": poom,
    })
    stats_10 = get_frac_exp_in_penalized(args_10, queries_path)
    print(
        f"{region_name} & {stats_10.count}\\% & {100*(1.0-stats_2.fracl1)}\\% &"
        f" {100*(1.0-stats_10.fracl1)}\\% & {pbe_value}\\% & {pbe_value}\\%"
        " \\\\ \\hline"
    )
  print("\\end{tabular}")


def print_whole_path_table():
  print("\\begin{tabular}{|l|c|c|c|c|c|}")
  print("\\hline")
  print(
      " & \\textbf{\\% valid} & \\textbf{\\parbox{1.5cm}{\\% exp in paths}} &"
      " \\multicolumn{3}{c|}{\\parbox{2cm}{\\textbf{\\centering\\# exp arcs /"
      " \\# path arcs}}}\\\\ \\hline"
  )
  print("percentile &  & min & 50\\% & 90\\% & max \\\\ \\hline")

  for region, queries_fname, region_name in ALL_QUERY_SETS:
    queries_path = region + "/" + queries_fname
    args = Args(**{
        **DEFAULT_ARGS_PARTIAL._asdict(),
        "osm_dir": region,
        "num_paths": 10,
        "penalty_multiplier": 1.1,
        "penalty_mode": "whole_path",
    })
    in_stats = get_frac_exp_in_paths(args, queries_path)
    of_stats = get_frac_exp_of_paths(args, queries_path)
    print(
        f"{region_name} & {in_stats.count}\\% & {100*in_stats.min}\\% &"
        f" {of_stats.median} & {of_stats.ninetieth} & {of_stats.max} \\\\"
        " \\hline"
    )
  print("\\end{tabular}")


def print_all_tables():
  print()
  print("POOM 1 IMPORTANT ARCS ONLY")
  print_important_arcs_only_table(1)
  print()
  print("POOM 2 IMPORTANT ARCS ONLY")
  print_important_arcs_only_table(2)
  print()
  print("WHOLE PATH")
  print_whole_path_table()


print_all_tables()
