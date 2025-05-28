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

"""This file contains the warm-start Ford-Fulkerson algorithm implmentation, the image sequence experiments

and the subroutines used by the experiment.
"""

import numpy as np
from queue import Queue
from collections import defaultdict
from copy import deepcopy
import time
from imagesegmentation import parseArgs, imageSegmentation
from augmentingPath import augmentingPath
import os
"""
One iteration of feasibility projection.
Algorithm: given a node u, BFS until find an excess-deficit path in residual graph
Prioritize paths that do not contain s or t, if no such path exists, allow using s or t.
Such a path must be found.
"""


def FeasRestoreIter(p, rGraph, excesses, s, t):

  def FindPath(chain, node, excesses, s, t, out_dir=True):
    # here into_dir = True denotes the path should carry flow OUT OF v.
    v, path = node, [node]
    path_flow = abs(excesses[v]) if v not in [s, t] else float('inf')
    u = chain[v]
    while u >= 0:
      path.append(u)
      (head, tail) = (u, v) if out_dir else (v, u)
      path_flow = min(rGraph[head][tail], path_flow)
      v = u
      u = chain[v]
    if v not in [s, t]:
      path_flow = min(path_flow, abs(excesses[v]))
    if out_dir:
      path.reverse()
    return path, path_flow

  if excesses[p] == 0:
    return [p], 0
  direc = (excesses[p] > 0)
  chain = defaultdict(int)
  q = Queue()
  q.put(p)
  visited = {p}
  chain[p] = -1

  while not q.empty():
    u = q.get()

    for v in list(rGraph[u].keys()) + [s, t]:
      (head, tail) = (u, v) if direc else (v, u)
      if v not in visited and rGraph[head][tail] > 0:
        q.put(v)
        chain[v] = u
        visited.add(v)
        if excesses[p] * excesses[v] < 0:
          return FindPath(chain, v, excesses, s, t, direc)

  if t in visited:
    return FindPath(chain, t, excesses, s, t, direc)
  if s in visited:
    return FindPath(chain, s, excesses, s, t, direc)

  return None, None


"""
Given a flow and the original graph, build the residual graph in the same dictionary format.
"""


def BuildRdGraph(flows, graph):
  rGraph = deepcopy(graph)
  for i in graph:
    for j in graph[i]:
      assert flows[i][j] <= graph[i][j]
      rGraph[i][j] -= flows[i][j]
      rGraph[j][i] += flows[i][j]
  return rGraph


"""
Given an infeasible flow, project it to a feasible one satisfying capacity constraints on the new graph
Iteratively calling the FeasRestoreIter() subroutine.
"""


def FeasProj(flows, graph, excesses, V, s, t):
  assert len(excesses) == V
  num_paths = 0
  total_path_len = 0
  rGraph = BuildRdGraph(flows, graph)
  proj_flows = deepcopy(flows)

  begin = time.time()
  for i in range(V):
    if i in [s, t] or excesses[i] == 0:
      continue
    while excesses[i] != 0:
      path, pathFlow = FeasRestoreIter(i, rGraph, excesses, s, t)
      if pathFlow is None:
        raise Exception(
            "Code has a bug. Didn't find a path during feasibility projection.")

      num_paths += 1
      total_path_len += (len(path) - 1)

      for j in range(len(path) - 1):
        u, v = path[j], path[j + 1]
        rGraph[u][v] -= pathFlow
        rGraph[v][u] += pathFlow
        proj_flows[u][v] += max(0, pathFlow - proj_flows[v][u])
        proj_flows[v][u] = max(0, proj_flows[v][u] - pathFlow)

      if path[0] not in [s, t]:
        excesses[path[0]] -= pathFlow
      if path[-1] not in [s, t]:
        excesses[path[-1]] += pathFlow
  end = time.time()

  print('feasiblity projection time:', end - begin)

  if num_paths > 0:
    feas_proj_avg_len = float(total_path_len) / num_paths
  else:
    feas_proj_avg_len = 0
  print('# of paths: %d, average length: %f' % (num_paths, feas_proj_avg_len))
  return proj_flows, rGraph, end - begin, num_paths, float(
      total_path_len) / num_paths


"""
The main function for warm-starting Ford-Fulkerson with a potentially infeasible flow.
  - First perform feasibility projection.
  - Then, based on the feasible flow, keep finding augmenting path and increasing the flow until termination.
"""


def WarmStartFlow(inf_flows, graph, excesses, V, s, t):
  begin = time.time()
  proj_flows, rGraph, proj_time, num_paths_proj, avg_len_proj = FeasProj(
      inf_flows, graph, excesses, V, s, t)
  proj_value = sum([proj_flows[s][v] for v in proj_flows[s]])
  max_flows, cuts, num_paths_aug, avg_len_aug = augmentingPath(
      graph, V, s, t, proj_flows, rGraph)
  end = time.time()
  print('warmstart total time: ', end - begin)
  print(
      'finding maximum flow # of augmenting path: %d, average augmenting path length: %f'
      % (num_paths_aug, avg_len_aug))
  return max_flows, cuts, proj_value, proj_time, end - begin, num_paths_proj, avg_len_proj, num_paths_aug, avg_len_aug


"""
Given another flow conserving flow and graph, round down the flow on every arc if it exceeds the capacity.
Returns the rounded flow and the excess at every node.
This function is not needed for warm-start itself but mostly for the experiment setting.
"""


def RoundDown(flows, graph, V, s, t):
  excesses = np.zeros(V, dtype=int)
  for i in range(V):
    for j in graph[i]:
      diff = max(0, flows[i][j] - graph[i][j])
      if i != s:
        excesses[i] += diff
      if j != t:
        excesses[j] -= diff
      flows[i][j] -= diff
  return excesses


"""
This function prints out the excesses/deficits on the graph so that we can see where they are.
"""


def ExcessGraph(excesses, size):
  excess_graph = np.zeros((size, size))
  for u in range(size * size):
    i, j = u // size, u % size
    excess_graph[i][j] = excesses[u]
  print(excess_graph)
  return


"""
The main function for running the full experiment with an existing set of seeds. The experiment does the following from image to image:
  - Construct the graph on the new image.
  - Take the old max flow solution from the last image, and round down according to the new capacities, thus generating excesses/deficits.
  - Run warm-start Ford-Fulkerson on the resulting infeasible flow.
"""


def Exp(args):
  folder, group, size, algo, loadseed = args.folder, args.group, args.size, args.algo, args.loadseed
  image_dir = folder + '/' + group + '_cropped'
  image_list = os.listdir(image_dir)
  V = size * size + 2
  SOURCE, SINK = V - 2, V - 1

  result_dir = folder + '/' + group + '_results' + '/'
  if not os.path.exists(result_dir):
    os.makedirs(result_dir)
  time_dir = result_dir + str(size) + '_time.txt'
  path_dir = result_dir + str(size) + '_path.txt'

  time_f = open(time_dir, 'w')
  time_f.write('image_name\tff\twarm_start\tfeas_proj\n')
  path_f = open(path_dir, 'w')
  path_f.write(
      'image_name\tflow_value\texcess\trecoverd_flow\tnum_aug_path\tavg_path_len\tnum_proj_path\tavg_proj_len\tnum_warm_start_path\tavg_warm_start_len\n'
  )

  num_images = len(image_list)
  old_flows = None
  for i in range(num_images):
    new_image = image_list[i]
    new_flows, min_cut, path_count, average_path_len, new_graph, ff_time = imageSegmentation(
        new_image, folder, group, (size, size), algo, loadseed)
    if old_flows is None:
      old_flows = deepcopy(new_flows)
      continue
    excesses = RoundDown(old_flows, new_graph, V, SOURCE, SINK)
    total_excess = max(
        np.sum(np.maximum(excesses, 0)), np.sum(np.maximum(-excesses, 0)))
    print('total excess/deficit to round down:' + str(total_excess))
    warmstart_flows, warmstart_cuts, proj_value, proj_time, warmstart_time, num_paths_proj, avg_len_proj, num_paths_aug, avg_len_aug = WarmStartFlow(
        old_flows, new_graph, excesses, V, SOURCE, SINK)

    # check that the warm start algo reaches optimality
    opt_flow_value = sum([new_flows[SOURCE][v] for v in new_flows[SOURCE]])
    ws_flow_value = sum(
        [warmstart_flows[SOURCE][v] for v in warmstart_flows[SOURCE]])
    assert opt_flow_value == ws_flow_value
    old_flows = deepcopy(new_flows)

    # write the results
    time_f.write(new_image.split('.')[0] + '\t')
    time_f.write('{}\t{}\t{}\n'.format(ff_time, warmstart_time, proj_time))
    time_f.flush()
    path_f.write(new_image.split('.')[0] + '\t')
    path_f.write('{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n'.format(
        opt_flow_value, total_excess, proj_value, path_count,
        round(average_path_len, 4), num_paths_proj, round(avg_len_proj, 4),
        num_paths_aug, round(avg_len_aug, 4)))
    path_f.flush()

  time_f.close()
  path_f.close()
  return


if __name__ == '__main__':
  args = parseArgs()
  Exp(args)
