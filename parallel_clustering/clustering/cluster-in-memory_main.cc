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

#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "clustering/clusterers/parallel-affinity.h"
#include "clustering/clusterers/parallel-correlation-clusterer.h"
#include "clustering/config.pb.h"
#include "clustering/in-memory-clusterer.h"
#include "clustering/status_macros.h"
#include "external/gbbs/gbbs/edge_map_blocked.h"
#include "google/protobuf/text_format.h"

ABSL_FLAG(std::string, clusterer_name, "",
          "Name of a clusterer (ParallelAffinityClusterer or "
          "ParallelCorrelationClusterer.");

ABSL_FLAG(std::string, clusterer_config, "",
          "Text-format research_graph.in_memory.ClustererConfig proto.");

ABSL_FLAG(std::string, input_graph, "",
          "Input file pattern of a graph. Should be in edge list format "
          "(SNAP format).");

ABSL_FLAG(std::string, output_clustering, "",
          "Output filename of a clustering.");

ABSL_FLAG(bool, is_symmetric_graph, true,
          "Without this flag, the program expects the edge list to represent "
          "an undirected graph (each edge needs to be given in both "
          "directions). With this flag, the program symmetrizes the graph.");

ABSL_FLAG(bool, float_weighted, false,
          "Use this flag if the edge list is weighted with 32-bit floats. If "
          "this flag is not set, then the graph is assumed to be unweighted, "
          "and edge weights are automatically set to 1.");

namespace research_graph {
namespace in_memory {
namespace {

double DoubleFromWeight(pbbslib::empty weight) { return 1; }
double DoubleFromWeight(double weight) { return weight; }

template <class Graph>
absl::Status GbbsGraphToInMemoryClustererGraph(InMemoryClusterer::Graph* graph,
                                               Graph& gbbs_graph) {
  using weight_type = typename Graph::weight_type;
  for (std::size_t i = 0; i < gbbs_graph.n; i++) {
    auto vertex = gbbs_graph.get_vertex(i);
    std::vector<std::pair<int32_t, double>> outgoing_edges(
        vertex.getOutDegree());
    std::size_t index = 0;
    auto add_outgoing_edge = [&](gbbs::uintE, const gbbs::uintE neighbor,
                                 weight_type wgh) {
      outgoing_edges[index] =
          std::make_pair(static_cast<int32_t>(neighbor), DoubleFromWeight(wgh));
      index++;
    };
    vertex.mapOutNgh(i, add_outgoing_edge, false);
    InMemoryClusterer::Graph::AdjacencyList adjacency_list{
        static_cast<InMemoryClusterer::NodeId>(i), 1,
        std::move(outgoing_edges)};
    RETURN_IF_ERROR(graph->Import(adjacency_list));
  }
  RETURN_IF_ERROR(graph->FinishImport());
  return absl::OkStatus();
}

template <typename Weight>
absl::StatusOr<std::size_t> WriteEdgeListAsGraph(
    InMemoryClusterer::Graph* graph,
    const std::vector<gbbs::gbbs_io::Edge<Weight>>& edge_list,
    bool is_symmetric_graph) {
  std::size_t n = 0;
  if (is_symmetric_graph) {
    auto gbbs_graph{gbbs::gbbs_io::edge_list_to_symmetric_graph(edge_list)};
    n = gbbs_graph.n;
    auto status = GbbsGraphToInMemoryClustererGraph<
        gbbs::symmetric_graph<gbbs::symmetric_vertex, Weight>>(graph,
                                                               gbbs_graph);
    RETURN_IF_ERROR(status);
    gbbs_graph.del();
  } else {
    auto gbbs_graph{gbbs::gbbs_io::edge_list_to_asymmetric_graph(edge_list)};
    n = gbbs_graph.n;
    auto status = GbbsGraphToInMemoryClustererGraph<
        gbbs::asymmetric_graph<gbbs::asymmetric_vertex, Weight>>(graph,
                                                                 gbbs_graph);
    RETURN_IF_ERROR(status);
    gbbs_graph.del();
  }
  return n;
}

absl::Status WriteClustering(const char* filename,
                             InMemoryClusterer::Clustering clustering) {
  std::ofstream file{filename};
  if (!file.is_open()) {
    return absl::NotFoundError("Unable to open file.");
  }
  for (int64_t i = 0; i < clustering.size(); i++) {
    for (auto node_id : clustering[i]) {
      file << node_id << std::endl;
    }
    file << std::endl;
  }
  return absl::OkStatus();
}

struct FakeGraph {
  std::size_t n;
};

absl::Status Main() {
  std::string clusterer_name = absl::GetFlag(FLAGS_clusterer_name);

  ClustererConfig config;
  std::string clusterer_config = absl::GetFlag(FLAGS_clusterer_config);
  if (!google::protobuf::TextFormat::ParseFromString(clusterer_config,
                                                     &config)) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Cannot parse --clusterer_config as a text-format "
                        "research_graph.in_memory.ClustererConfig proto: %s",
                        clusterer_config));
  }

  std::unique_ptr<InMemoryClusterer> clusterer;
  if (clusterer_name == "ParallelAffinityClusterer") {
    clusterer.reset(new ParallelAffinityClusterer);
  } else if (clusterer_name == "ParallelCorrelationClusterer") {
    clusterer.reset(new ParallelCorrelationClusterer);
  } else {
    return absl::UnimplementedError(
        "ParallelAffinityClusterer and ParallelCorrelationClusterer are the "
        "only supported clusterers.");
  }

  std::string input_file = absl::GetFlag(FLAGS_input_graph);
  bool is_symmetric_graph = absl::GetFlag(FLAGS_is_symmetric_graph);
  bool float_weighted = absl::GetFlag(FLAGS_float_weighted);
  std::size_t n = 0;
  if (float_weighted) {
    const auto edge_list{
        gbbs::gbbs_io::read_weighted_edge_list<float>(input_file.c_str())};
    ASSIGN_OR_RETURN(n, WriteEdgeListAsGraph(clusterer->MutableGraph(),
                                             edge_list, is_symmetric_graph));
  } else {
    const auto edge_list{
        gbbs::gbbs_io::read_unweighted_edge_list(input_file.c_str())};
    ASSIGN_OR_RETURN(n, WriteEdgeListAsGraph(clusterer->MutableGraph(),
                                             edge_list, is_symmetric_graph));
  }

  // Must initialize the list allocator for GBBS, to support parallelism.
  // The list allocator seeds using the number of vertices in the input graph.
  FakeGraph fake_graph{n};
  gbbs::alloc_init(fake_graph);

  InMemoryClusterer::Clustering clustering;
  ASSIGN_OR_RETURN(clustering, clusterer->Cluster(config));

  gbbs::alloc_finish();

  std::string output_file = absl::GetFlag(FLAGS_output_clustering);
  WriteClustering(output_file.c_str(), clustering);

  return absl::OkStatus();
}

}  // namespace
}  // namespace in_memory
}  // namespace research_graph

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  auto status = research_graph::in_memory::Main();
  if (!status.ok()) {
    std::cerr << status << std::endl;
    return EXIT_FAILURE;
  }
}
