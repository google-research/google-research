// Copyright 2021 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gflags/gflags.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cmath>

#include "common.h"           // NOLINT
#include "graph.h"            // NOLINT
#include "intersect_edges.h"  // NOLINT
#include "mmapped_vector.h"   // NOLINT

using edge_t = unsigned long long_t;
using node_t = uint32_t;

using Graph = GraphT<node_t, edge_t>;

DEFINE_string(filename, "", "Input filename. Omit for stdin.");
DEFINE_string(format, "nde",
              "Input file format. Valid values are nde, bin and sbin.");
DEFINE_string(exact_exe, "", "Executable for the exact algorithm.");
DEFINE_string(storage_dir, "",
              "Directory for storage of working variables. Leave empty to use "
              "main memory.");
DEFINE_int64(edge_limit, 0,
             "Number of edges at which the algorithm should switch to the "
             "exact algorithm.");
DEFINE_int64(minimum_trussness, 0,
             "Minimum value of trussness to start processing from.");
DEFINE_double(eps, 0.1,
              "Approximation factor will be guaranteed to be at most 3+eps.");

void SwitchToExact(Graph *__restrict__ g, size_t *res) {
  fprintf(stderr, "Switching to exact algorithm...\n");

  // Prepare arguments and pipes for running the
  int exit_code;
  int cin_pipe[2];
  int cerr_pipe[2];
  posix_spawn_file_actions_t action;

  CHECK_M(pipe(cin_pipe) == 0, strerror(errno));
  CHECK_M(pipe(cerr_pipe) == 0, strerror(errno));

  posix_spawn_file_actions_init(&action);
  posix_spawn_file_actions_addclose(&action, cin_pipe[1]);
  posix_spawn_file_actions_addclose(&action, cerr_pipe[0]);
  posix_spawn_file_actions_adddup2(&action, cin_pipe[0], fileno(stdin));
  posix_spawn_file_actions_adddup2(&action, cerr_pipe[1], fileno(stderr));

  posix_spawn_file_actions_addclose(&action, cin_pipe[0]);
  posix_spawn_file_actions_addclose(&action, cerr_pipe[1]);

  char *args[] = {&FLAGS_exact_exe[0], nullptr};

  pid_t pid;
  CHECK_M(posix_spawnp(&pid, args[0], &action, NULL, &args[0], NULL) == 0,
          strerror(errno));

  close(cin_pipe[0]);
  close(cerr_pipe[1]);

  // Write graph to stdin of exact algorithm.
  {
    FastWriter out(cin_pipe[1]);
    out.Write(g->N());
    out.Write('\n');
    out.Flush();
    for (size_t i = 0; i < g->N(); i++) {
      out.Write(i);
      out.Write(' ');
      out.Write(g->adj_start[i + 1] - g->adj_start[i]);
      out.Write('\n');
    }

    for (size_t i = 0; i < g->N(); i++) {
      for (size_t j = g->adj_start[i]; j < g->adj_start[i + 1]; j++) {
        out.Write(i);
        out.Write(' ');
        out.Write(g->adj[j]);
        out.Write('\n');
      }
    }
    g->adj_start.clear();
    g->adj.clear();
    out.Flush();
  }

  // Parse output from exact algorithm.
  FILE *in = fdopen(cerr_pipe[0], "r");
  CHECK_M(in, strerror(errno));
  char *line = nullptr;
  size_t len = 0;
  ssize_t nread;
  size_t edg = 0;
  size_t trs = 0;
  while ((nread = getline(&line, &len, in)) != -1) {
    fputs(line, stderr);
    if (sscanf(line, "Iterations for %zu-truss (%zu g->adj)", &trs, &edg) ==
        2) {
      if (edg > 0) {
        if (*res + 2 < trs) {
          *res = trs - 2;
        }
      }
    }
  }
  free(line);
  fclose(in);

  CHECK_NE(waitpid(pid, &exit_code, 0), -1);
}

size_t CountTriangles(
    Graph *__restrict__ g,
    mmapped_vector<std::atomic<node_t>> *__restrict__ edge_triangles_count) {
  size_t triangles = 0;
#pragma omp parallel for schedule(dynamic) reduction(+ : triangles)
  for (node_t i = 0; i < g->N(); i++) {
    std::vector<node_t> local_counts(g->adj_start[i + 1] - g->adj_start[i]);
    for (edge_t cnt = g->adj_start[i]; cnt < g->adj_start[i + 1]; cnt++) {
      const node_t j = g->adj[cnt];
      IntersectEdges(g, cnt + 1, g->adj_start[i + 1], g->adj_start[j],
                     g->adj_start[j + 1], [&](edge_t k, edge_t l) {
                       local_counts[cnt - g->adj_start[i]]++;
                       local_counts[k - g->adj_start[i]]++;
                       triangles++;
                       (*edge_triangles_count)[l].fetch_add(
                           1, std::memory_order_relaxed);
                       return true;
                     });
    }
    for (edge_t cnt = g->adj_start[i]; cnt < g->adj_start[i + 1]; cnt++) {
      (*edge_triangles_count)[cnt].fetch_add(
          local_counts[cnt - g->adj_start[i]], std::memory_order_relaxed);
    }
  }
  FullFence();
  return triangles;
}

void ApproximateTrussness(Graph *__restrict__ g, float eps) {
  mmapped_vector<std::atomic<node_t>> new_degree;  // O(N)

  mmapped_vector<std::atomic<node_t>> edge_triangles_count;  // O(M)
  edge_triangles_count.init(
      FileFromStorageDir(FLAGS_storage_dir, "triangles.bin"), g->adj.size());

  int iter = 0;
  size_t trussness_bound = FLAGS_minimum_trussness;
  size_t upper_bound = 0;
  double c = 3 + eps;
  while (g->adj.size() > 0) {
    iter++;

    // Compute lower bound on the max truss of the graph: number of triangles
    // divided by number of edges.
    size_t m = g->adj.size();
    auto iter_start = std::chrono::high_resolution_clock::now();
    size_t T = CountTriangles(g, &edge_triangles_count);
    auto iter_tri = std::chrono::high_resolution_clock::now();
    if (std::ceil(1.0 * T / m) > trussness_bound) {
      trussness_bound = std::ceil(1.0 * T / m);
    }

    // We know that the trussness is at most 3 * T / m. If c > 3, `upper_bound`
    // is guaranteed to be greater than the trussness.
    upper_bound = std::ceil(c * trussness_bound);

    // If c <= 3 (ie. eps <= 0), there is no guarantee that the upper bound is
    // indeed correct. If all edges have at least `upper_bound` triangles, then
    // the upper bound is wrong and we need to update it.
    if (eps <= 0) {
      size_t min_support = std::numeric_limits<size_t>::max();
#pragma omp parallel for reduction(min : min_support)
      for (size_t i = 0; i < edge_triangles_count.size(); i++) {
        if (edge_triangles_count[i] < min_support) {
          min_support = edge_triangles_count[i];
        }
      }
      if (min_support > upper_bound) {
        trussness_bound = min_support;
        upper_bound = min_support;
      }
    }

    // Remove all edges with a lower number of triangles than our upper bound.
    // TODO: parallelize.
    size_t edge_counter = 0;
    size_t min_support = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < g->N(); i++) {
      size_t initial_counter = edge_counter;
      for (size_t j = g->adj_start[i]; j < g->adj_start[i + 1]; j++) {
        if (edge_triangles_count[j] > upper_bound) {
          edge_triangles_count[edge_counter] = 0;
          g->adj[edge_counter++] = g->adj[j];
        }
        if (edge_triangles_count[j] < min_support) {
          min_support = edge_triangles_count[j];
        }
      }
      g->adj_start[i] = initial_counter;
    }
    if (min_support > trussness_bound) {
      trussness_bound = min_support;
    }

    g->adj.resize(edge_counter);
    g->adj.shrink();
    edge_triangles_count.resize(edge_counter);
    edge_triangles_count.shrink();
    g->adj_start[g->N()] = edge_counter;

    auto iter_end = std::chrono::high_resolution_clock::now();
    fprintf(stderr,
            "time for iter %3d (%14lu M, %14lu T, t >= %5lu): %9lums (%9lums "
            "tr. cnt.)\n",
            iter, m, T, trussness_bound + 2,
            std::chrono::duration_cast<std::chrono::milliseconds>(iter_end -
                                                                  iter_start)
                .count(),
            std::chrono::duration_cast<std::chrono::milliseconds>(iter_tri -
                                                                  iter_start)
                .count());

    // If the graph is sufficiently small, switch to exact algorithm.
    if (FLAGS_edge_limit > g->adj.size()) {
      SwitchToExact(g, &trussness_bound);
      fprintf(stderr, "Trussness values above %lu are accurate\n",
              upper_bound + 2);
      if (upper_bound < trussness_bound) upper_bound = trussness_bound;
      break;
    }
  }

  fprintf(stderr, "Approx factor: %f\n", upper_bound * 1.0 / trussness_bound);
  fprintf(stderr, "%6lu <= t <= %6lu\n", trussness_bound + 2, upper_bound + 2);
}

void Run() {
  double eps = FLAGS_eps;
  auto start = std::chrono::high_resolution_clock::now();

  Graph g;

  std::chrono::high_resolution_clock::time_point reading;
  std::chrono::high_resolution_clock::time_point permutation;

  g.adj_start.init(FileFromStorageDir(FLAGS_storage_dir, "adj_start.bin"));
  g.adj.init(FileFromStorageDir(FLAGS_storage_dir, "edges.bin"));

  g.Read(FLAGS_filename, ParseGraphFormat(FLAGS_format),
         GraphPermutation::kDegreeOrder,
         /*forward_only=*/true, &reading, &permutation);
  size_t edge_count = g.adj_start.back();

  fprintf(stderr, "Permute done\n");
  auto permute = std::chrono::high_resolution_clock::now();

  ApproximateTrussness(&g, eps);

  auto end = std::chrono::high_resolution_clock::now();

  fprintf(stderr, "time for reading: %lums\n",
          std::chrono::duration_cast<std::chrono::milliseconds>(reading - start)
              .count());
  fprintf(stderr, "time for computing permutation: %lums\n",
          std::chrono::duration_cast<std::chrono::milliseconds>(permutation -
                                                                reading)
              .count());
  fprintf(stderr, "time for permuting: %lums\n",
          std::chrono::duration_cast<std::chrono::milliseconds>(permute -
                                                                permutation)
              .count());
  fprintf(stderr, "time for computing: %lums\n",
          std::chrono::duration_cast<std::chrono::milliseconds>(end - permute)
              .count());
  printf("NumEdges\tExecutionTime(s)\tRate\n");
  double executionTime =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - reading)
          .count() /
      1000.0;
  printf("%lu\t%f\t%f\n", edge_count, executionTime,
         edge_count / executionTime);
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_edge_limit != 0 && FLAGS_exact_exe == "") {
    fprintf(stderr,
            "Invalid flags: must specify exact executable if specifying an "
            "edge limit.\n");
    return 1;
  }
  if (FLAGS_eps < -2) {
    fprintf(stderr,
            "Invalid flags: eps must not be smaller than -2. Running time is "
            "guaranteed only for eps > 0.\n");
    return 1;
  }
  if (FLAGS_edge_limit < 0) {
    fprintf(stderr, "Invalid flags: edge_limit must be non-negative.\n");
    return 1;
  }
  if (FLAGS_minimum_trussness < 0) {
    fprintf(stderr, "Invalid flags: minimum_trussness must be non-negative.\n");
    return 1;
  }
  Run();
}
