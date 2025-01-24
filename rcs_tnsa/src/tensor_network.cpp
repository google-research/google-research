#include "tensor_network.h"  // NOLINT(build/include)

#include <cmath>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>

////// Public methods //////

TensorNetwork::TensorNetwork(const std::vector<std::vector<size_t>>& edges,
                             const std::vector<double>& log2_bds) {
  // These two have to have the same length.
  edges_ = edges;
  log2_bds_ = log2_bds;
}

TensorNetwork::TensorNetwork(const std::string filename) {
  InitFromGraph(filename);
  InitEncoding();
}

size_t const TensorNetwork::GetNumEdges() { return num_edges_; }

size_t const TensorNetwork::GetNumNodes() { return num_nodes_; }

std::vector<size_t> const& TensorNetwork::GetContractingEdgeNums() {
  return contracting_edge_nums_;
}

std::set<size_t> const& TensorNetwork::GetEnvironmentEdgeNums() {
  return environment_edge_nums_;
}

std::vector<std::vector<std::vector<size_t>>>
TensorNetwork::ContractionBreakdown(const std::vector<size_t>& ordering) {
  // Initializing returning vectors.
  std::vector<std::vector<size_t>> left_edge_nums_vector(
      num_contracting_nodes_ - 1);
  std::vector<std::vector<size_t>> middle_edge_nums_vector(
      num_contracting_nodes_ - 1);
  std::vector<std::vector<size_t>> right_edge_nums_vector(
      num_contracting_nodes_ - 1);

  // Mock contraction.
  size_t i = 0;
  size_t num_tree_nodes = num_contracting_nodes_ * 2 - 1;
  std::vector<std::set<size_t>> tree_node_to_edge_nums(num_tree_nodes);
  std::vector<std::vector<size_t>> tree_node_to_nodes(num_tree_nodes);
  std::vector<size_t> node_to_tree_node(num_contracting_nodes_);
  for (size_t n = 0; n < num_contracting_nodes_; n++) {
    tree_node_to_edge_nums[n] = node_to_edge_nums_[n];
    tree_node_to_nodes[n] = std::vector<size_t>({n});
    node_to_tree_node[n] = n;
  }
  size_t tn = num_contracting_nodes_;
  for (auto en : ordering) {
    size_t m0 = edges_[en][0];
    size_t m1 = edges_[en][1];
    size_t n0 = node_to_tree_node[m0];
    size_t n1 = node_to_tree_node[m1];

    if (n0 == n1) continue;

    const std::set<size_t>& n0_edge_nums = tree_node_to_edge_nums[n0];
    const std::set<size_t>& n1_edge_nums = tree_node_to_edge_nums[n1];
    std::set<size_t> left_edge_nums, middle_edge_nums, right_edge_nums;
    SetDifference(n0_edge_nums, n1_edge_nums, left_edge_nums);
    SetIntersection(n0_edge_nums, n1_edge_nums, middle_edge_nums);
    SetDifference(n1_edge_nums, n0_edge_nums, right_edge_nums);

    size_t tree_node_to_nodes_n0_size = tree_node_to_nodes[n0].size();
    tree_node_to_nodes[tn] = std::vector<size_t>();
    VectorUnion(tree_node_to_nodes[n0], tree_node_to_nodes[n1],
                tree_node_to_nodes[tn]);

    for (auto m : tree_node_to_nodes[tn]) {
      node_to_tree_node[m] = tn;
    }
    tree_node_to_edge_nums[tn] = std::set<size_t>();
    SetUnion(left_edge_nums, right_edge_nums, tree_node_to_edge_nums[tn]);
    left_edge_nums_vector[i] =
        std::vector<size_t>(left_edge_nums.begin(), left_edge_nums.end());
    middle_edge_nums_vector[i] =
        std::vector<size_t>(middle_edge_nums.begin(), middle_edge_nums.end());
    right_edge_nums_vector[i] =
        std::vector<size_t>(right_edge_nums.begin(), right_edge_nums.end());

    i++;
    tn++;
  }

  // Preparing returning variable.
  std::vector<std::vector<std::vector<size_t>>> ret_vectors;
  ret_vectors.push_back(left_edge_nums_vector);
  ret_vectors.push_back(middle_edge_nums_vector);
  ret_vectors.push_back(right_edge_nums_vector);
  return ret_vectors;
}

double TensorNetwork::Log2Flops(
    const std::vector<std::vector<std::vector<size_t>>>& breakdown) {
  double flops = 0.;
  for (size_t i = 0; i < breakdown[0].size(); i++) {
    double exponent = 0.;
    for (size_t j = 0; j < 3; j++) {
      for (size_t k = 0; k < breakdown[j][i].size(); k++) {
        exponent += log2_bds_[breakdown[j][i][k]];
      }
    }
    flops += std::pow(2., exponent);
  }
  return std::log2(flops);
}

double TensorNetwork::SlicedLog2FlopsGroupedSlicesSparseOutput(
    const std::vector<std::vector<std::vector<size_t>>>& breakdown,
    const std::vector<std::vector<size_t>>& sliced_grouped_edge_nums,
    size_t num_output_confs) {
  double log2_environment_bd = std::log2(num_output_confs);

  double log2_prefactor = 0.;
  for (auto g : sliced_grouped_edge_nums) {
    log2_prefactor += log2_bds_[g[0]];
  }

  std::vector<size_t> sliced_edge_nums_vector;
  for (size_t i = 0; i < sliced_grouped_edge_nums.size(); i++) {
    sliced_edge_nums_vector.insert(sliced_edge_nums_vector.end(),
                                   sliced_grouped_edge_nums[i].begin(),
                                   sliced_grouped_edge_nums[i].end());
  }
  std::set<size_t> sliced_edge_nums(sliced_edge_nums_vector.begin(),
                                    sliced_edge_nums_vector.end());

  double flops = 0.;
  for (size_t i = 0; i < breakdown[0].size(); i++) {
    double exponent = 0.;
    std::vector<size_t> local_environment_edge_nums;
    for (size_t j = 0; j < 3; j++) {
      for (size_t k = 0; k < breakdown[j][i].size(); k++) {
        if (sliced_edge_nums.find(breakdown[j][i][k]) ==
            sliced_edge_nums.end()) {
          if (environment_edge_nums_.find(breakdown[j][i][k]) ==
              environment_edge_nums_.end()) {
            exponent += log2_bds_[breakdown[j][i][k]];
          } else {
            local_environment_edge_nums.push_back(breakdown[j][i][k]);
          }
        }
      }
    }
    double log2_local_environment_bd = 0.;
    for (auto en : local_environment_edge_nums) {
      log2_local_environment_bd += log2_bds_[en];
    }
    exponent += std::min(log2_environment_bd, log2_local_environment_bd);

    flops += std::pow(2., exponent);
  }

  return std::log2(flops) + log2_prefactor;
}

double TensorNetwork::SlicedMemoizedLog2FlopsGroupedSlicesSparseOutput(
    const std::vector<std::vector<std::vector<size_t>>>& breakdown,
    const std::vector<std::vector<size_t>>& sliced_grouped_edge_nums,
    size_t num_output_confs) {
  double log2_environment_bd = std::log2(num_output_confs);

  // Has the intersection of this slice been reached?
  std::unordered_map<size_t, bool> slice_reached;
  for (auto g : sliced_grouped_edge_nums) {
    slice_reached[g[0]] = false;
  }
  std::unordered_map<size_t, size_t> slice_to_group;
  for (auto g : sliced_grouped_edge_nums) {
    for (auto s : g) {
      slice_to_group[s] = g[0];
    }
  }

  std::vector<double> log2_prefactors(sliced_grouped_edge_nums.size() + 1, 0.);
  for (size_t i = 1; i < sliced_grouped_edge_nums.size() + 1; i++) {
    log2_prefactors[i] =
        log2_prefactors[i - 1] + log2_bds_[sliced_grouped_edge_nums[i - 1][0]];
  }

  std::vector<size_t> sliced_edge_nums_vector;
  for (size_t i = 0; i < sliced_grouped_edge_nums.size(); i++) {
    sliced_edge_nums_vector.insert(sliced_edge_nums_vector.end(),
                                   sliced_grouped_edge_nums[i].begin(),
                                   sliced_grouped_edge_nums[i].end());
  }
  std::set<size_t> sliced_edge_nums(sliced_edge_nums_vector.begin(),
                                    sliced_edge_nums_vector.end());

  std::vector<double> flopss(sliced_grouped_edge_nums.size() + 1, 0.);

  size_t sg = 0;  // Enumerating slice groups.
  for (size_t i = 0; i < breakdown[0].size(); i++) {
    double exponent = 0.;
    std::vector<size_t> local_environment_edge_nums;
    for (size_t j = 0; j < 3; j++) {
      for (size_t k = 0; k < breakdown[j][i].size(); k++) {
        if (sliced_edge_nums.find(breakdown[j][i][k]) ==
            sliced_edge_nums.end()) {
          if (environment_edge_nums_.find(breakdown[j][i][k]) ==
              environment_edge_nums_.end()) {
            exponent += log2_bds_[breakdown[j][i][k]];
          } else {
            local_environment_edge_nums.push_back(breakdown[j][i][k]);
          }
        } else {
          if (!slice_reached[slice_to_group[breakdown[j][i][k]]]) {
            sg++;
            slice_reached[slice_to_group[breakdown[j][i][k]]] =
                true;  // Dangerous: assumming unique representative. Groups
                       // overlap. Works for RCS circuits.
          }
        }
      }
    }
    double log2_local_environment_bd = 0.;
    for (auto en : local_environment_edge_nums) {
      log2_local_environment_bd += log2_bds_[en];
    }
    exponent += std::min(log2_environment_bd, log2_local_environment_bd);

    flopss[sg] += std::pow(2., exponent);
  }

  double flops = 0.;
  for (size_t sg = 0; sg < sliced_grouped_edge_nums.size() + 1; sg++) {
    flops += flopss[sg] * std::pow(2., log2_prefactors[sg]);
  }
  double log2_flops = std::log2(flops);

  // If this sum didn't work then forget about memoization and compute the usual
  // one.
  if (std::isnan(flops)) {
    double slice_flops = 0.;
    for (size_t sg = 0; sg < sliced_grouped_edge_nums.size() + 1; sg++) {
      slice_flops += flopss[sg];
    }
    log2_flops = (std::log2(slice_flops) +
                  log2_prefactors[sliced_grouped_edge_nums.size()]);
  }

  return log2_flops;
}

double TensorNetwork::Width(
    const std::vector<std::vector<std::vector<size_t>>>& breakdown) {
  double width = 0.;
  for (size_t i = 0; i < breakdown[0].size(); i++) {
    double log2_size = 0.;
    for (size_t j = 0; j < 3; j++) {
      if (j == 1) continue;
      for (size_t k = 0; k < breakdown[j][i].size(); k++) {
        log2_size += log2_bds_[breakdown[j][i][k]];
      }
    }
    width = std::max(width, log2_size);
  }
  return width;
}

double TensorNetwork::SlicedWidthSparseOutput(
    const std::vector<std::vector<std::vector<size_t>>>& breakdown,
    const std::set<size_t>& sliced_edge_nums, size_t num_output_confs) {
  double log2_environment_bd = std::log2(num_output_confs);

  double width = 0.;
  for (size_t i = 0; i < breakdown[0].size(); i++) {
    double log2_size = 0.;
    std::vector<size_t> local_environment_edge_nums;
    for (size_t j = 0; j < 3; j++) {
      if (j == 1) continue;
      for (size_t k = 0; k < breakdown[j][i].size(); k++) {
        if (sliced_edge_nums.find(breakdown[j][i][k]) ==
            sliced_edge_nums.end()) {
          if (environment_edge_nums_.find(breakdown[j][i][k]) ==
              environment_edge_nums_.end()) {
            log2_size += log2_bds_[breakdown[j][i][k]];
          } else {
            local_environment_edge_nums.push_back(breakdown[j][i][k]);
          }
        }
      }
    }
    double log2_local_environment_bd = 0.;
    for (auto en : local_environment_edge_nums) {
      log2_local_environment_bd += log2_bds_[en];
    }
    log2_size += std::min(log2_environment_bd, log2_local_environment_bd);

    width = std::max(width, log2_size);
  }

  return width;
}

double TensorNetwork::SlicedFullMemoryGroupedSlicesSparseOutput(
    const std::vector<std::vector<std::vector<size_t>>>& breakdown,
    const std::vector<std::vector<size_t>>& sliced_grouped_edge_nums,
    size_t num_output_confs) {
  double log2_environment_bd = std::log2(num_output_confs);
  std::vector<double> list_memory(
      breakdown[0].size());  // Fill in with memory consumption for every step.
  double memory = 0.;

  // Has the intersection of this slice been reached?
  std::unordered_map<size_t, bool> slice_reached;
  for (auto g : sliced_grouped_edge_nums) {
    slice_reached[g[0]] = false;
  }
  std::unordered_map<size_t, size_t> slice_to_group;
  for (auto g : sliced_grouped_edge_nums) {
    for (auto s : g) {
      slice_to_group[s] = g[0];
    }  // Dangerous: assumes unique representatives. Groups overlap. Works for
       // RCS circuits.
  }
  // Set of sliced edge nums.
  std::set<size_t> sliced_edge_nums_set;
  for (auto g : sliced_grouped_edge_nums)
    for (auto f : g) {
      sliced_edge_nums_set.insert(f);
    }

  size_t sg = 0;  // Enumerating slice groups.
  std::vector<size_t> bifurcations;
  for (size_t i = 0; i < breakdown[0].size(); i++) {
    auto left_edges = breakdown[0][i];
    auto middle_edges = breakdown[1][i];
    auto right_edges = breakdown[2][i];
    // Result.
    double log2_result_size = 0.;
    double log2_result_environment = 0.;
    // Left.
    double log2_left_size = 0.;
    double log2_left_environment = 0.;
    for (auto e : left_edges) {
      // If edge is not sliced.
      if (sliced_edge_nums_set.find(e) == sliced_edge_nums_set.end()) {
        // If edge is not environment.
        if (environment_edge_nums_.find(e) == environment_edge_nums_.end()) {
          log2_left_size += log2_bds_[e];
          log2_result_size += log2_bds_[e];
        } else {
          log2_left_environment += log2_bds_[e];
          log2_result_environment += log2_bds_[e];
        }
      } else {
        if (!slice_reached[slice_to_group[e]]) {
          bifurcations.push_back(i);
          sg++;
          slice_reached[slice_to_group[e]] = true;
        }
      }
    }
    if (log2_left_environment > log2_environment_bd) {
      log2_left_environment = log2_environment_bd;
    }
    log2_left_size += log2_left_environment;
    // Right.
    double log2_right_size = 0.;
    double log2_right_environment = 0.;
    for (auto e : right_edges) {
      // If edge is not sliced.
      if (sliced_edge_nums_set.find(e) == sliced_edge_nums_set.end()) {
        // If edge is not environment.
        if (environment_edge_nums_.find(e) == environment_edge_nums_.end()) {
          log2_right_size += log2_bds_[e];
          log2_result_size += log2_bds_[e];
        } else {
          log2_right_environment += log2_bds_[e];
          log2_result_environment += log2_bds_[e];
        }
      } else {
        if (!slice_reached[slice_to_group[e]]) {
          bifurcations.push_back(i);
          sg++;
          slice_reached[slice_to_group[e]] = true;
        }
      }
    }
    if (log2_right_environment > log2_environment_bd) {
      log2_right_environment = log2_environment_bd;
    }
    log2_right_size += log2_right_environment;
    // Middle.
    double log2_middle_size = 0.;
    for (auto e : middle_edges) {
      // If edge is not sliced.
      if (sliced_edge_nums_set.find(e) == sliced_edge_nums_set.end()) {
        log2_right_size += log2_bds_[e];
      } else {
        if (!slice_reached[slice_to_group[e]]) {
          bifurcations.push_back(i);
          sg++;
          slice_reached[slice_to_group[e]] = true;
        }
      }
    }
    // Result environment decision.
    if (log2_result_environment > log2_environment_bd) {
      log2_result_environment = log2_environment_bd;
    }
    log2_result_size += log2_result_environment;

    // Accumulate sizes.
    if (i == 0) {
      memory = std::pow(2., log2_result_size);
      list_memory[i] = memory;
    } else {
      memory += std::pow(2., log2_result_size);
      list_memory[i] = memory;
      memory -= std::pow(2., log2_left_size + log2_middle_size);
      memory -= std::pow(2., log2_right_size + log2_middle_size);
    }
  }
  std::vector<double> additions(bifurcations.size());
  if (bifurcations.size() > 0) {
    additions[0] = list_memory[bifurcations[0]];
  }
  for (size_t i = 1; i < bifurcations.size(); i++) {
    additions[i] = list_memory[bifurcations[i]] + additions[i - 1];
  }
  // Updating list_memory. The way the usage of memory is defined, it is
  // non-monotonic in the number of slices. For this reason, we need to replace
  // the binary search for a search from the beginning (no slices) until a
  // better viable solution is found.
  size_t pos = 0;
  for (size_t i = 0; i < bifurcations.size(); i++) {
    for (size_t j = pos; j <= bifurcations[i]; j++) {
      list_memory[j] += additions[i];
    }
  }
  if (bifurcations.size() > 0) {
    for (size_t j = bifurcations[bifurcations.size() - 1];
         j < list_memory.size(); j++) {
      list_memory[j] += additions[additions.size() - 1];
    }
  }

  double max_memory = 0.;
  for (auto v : list_memory) {
    max_memory = std::max(v, max_memory);
  }
  return max_memory;
}

size_t TensorNetwork::NumSlicesGivenWidthGroupedSlicesSparseOutput(
    const std::vector<std::vector<std::vector<size_t>>>& breakdown,
    size_t max_width,
    const std::vector<std::vector<size_t>>& slices_groups_ordering,
    size_t num_confs) {
  size_t low = 0, high = slices_groups_ordering.size(), mid;
  double low_width = 0., high_width = 0., mid_width = 0.;
  while (low <= high) {
    mid = (high + low) / 2;
    std::vector<size_t> sliced_edge_nums_vector;
    for (size_t i = 0; i < mid; i++) {
      sliced_edge_nums_vector.insert(sliced_edge_nums_vector.end(),
                                     slices_groups_ordering[i].begin(),
                                     slices_groups_ordering[i].end());
    }
    std::set<size_t> sliced_edge_nums(sliced_edge_nums_vector.begin(),
                                      sliced_edge_nums_vector.end());
    mid_width = SlicedWidthSparseOutput(breakdown, sliced_edge_nums, num_confs);
    if (mid_width > max_width)
      low = mid + 1;
    else if (mid_width <= max_width)
      high = mid - 1;
  }
  if (mid_width > max_width) mid++;
  return mid;
}

size_t TensorNetwork::NumSlicesGivenFullMemoryGroupedSlicesSparseOutput(
    const std::vector<std::vector<std::vector<size_t>>>& breakdown,
    double log2_max_memory,
    const std::vector<std::vector<size_t>>& slices_groups_ordering,
    size_t num_output_confs) {
  size_t pos = 0;
  double memory;
  do {
    std::vector<std::vector<size_t>> sliced_grouped_edge_nums;
    for (size_t i = 0; i < pos; i++) {
      sliced_grouped_edge_nums.push_back(slices_groups_ordering[i]);
    }
    memory = SlicedFullMemoryGroupedSlicesSparseOutput(
        breakdown, sliced_grouped_edge_nums, num_output_confs);
    pos++;
  } while (std::log2(memory) > log2_max_memory);
  return pos - 1;
}

////// Private methods //////

void TensorNetwork::InitFromGraph(const std::string filename) {
  edges_.clear();
  log2_bds_.clear();

  auto graph_stream = std::ifstream(filename);
  std::string line;
  while (std::getline(graph_stream, line)) {
    auto line_stream = std::istringstream(line);
    std::vector<std::string> tokens;
    std::string token;
    while (getline(line_stream, token, ' ')) {
      tokens.push_back(token);
    }
    double w = std::stod(tokens[0]);
    std::vector<size_t> edge;
    for (size_t i = 1; i < tokens.size(); i++) {
      edge.push_back(std::stoi(tokens[i]));
    }
    edges_.push_back(edge);
    log2_bds_.push_back(std::log2(w));
  }

  // Check if there is environment.
  // If there is, then assign the environment the next natural node name.
  std::set<int> nodes;
  for (auto e : edges_) {
    for (auto n : e) {
      nodes.insert(n);
    }
  }
  if (nodes.find(-1) != nodes.end()) {
    environment_ = true;
    environment_num_ = nodes.size() - 1;
    for (size_t i = 0; i < edges_.size(); i++) {
      bool include_edge = true;
      for (size_t j = 0; j < edges_[i].size(); j++) {
        if (edges_[i][j] == -1) {
          include_edge = false;
          edges_[i][j] = environment_num_;
        }
      }
      if (include_edge)
        contracting_edge_nums_.push_back(i);
      else
        environment_edge_nums_.insert(i);
    }
  } else {
    environment_ = false;
    contracting_edge_nums_.resize(edges_.size());
    for (size_t i = 0; i < edges_.size(); i++) {
      contracting_edge_nums_[i] = i;
    }
  }
}

void TensorNetwork::InitEncoding() {
  for (auto e : edges_) {
    for (auto n : e) {
      nodes_.insert(n);
    }
  }
  num_nodes_ = nodes_.size();
  num_contracting_nodes_ = num_nodes_ - (environment_ ? 1 : 0);
  num_edges_ = edges_.size();
  num_contracting_edges_ = contracting_edge_nums_.size();
  node_to_edge_nums_.resize(nodes_.size());
  for (auto n : nodes_) {
    node_to_edge_nums_[n] = std::set<size_t>();
  }
  for (size_t i = 0; i < num_edges_; i++) {
    auto e = edges_[i];
    for (auto n : e) {
      node_to_edge_nums_[n].insert(i);
    }
  }
}

//////// EXTERNAL METHODS ////////

std::vector<size_t> LoadOrdering(const std::string ordering_filename) {
  std::vector<size_t> ordering;
  auto ordering_stream = std::ifstream(ordering_filename);
  std::string token;
  while (getline(ordering_stream, token)) {
    ordering.push_back(std::stoi(token));
  }
  return ordering;
}

std::vector<size_t> LoadSlicedEdges(const std::string slicing_filename) {
  std::vector<size_t> sliced_edges;
  auto slicing_stream = std::ifstream(slicing_filename);
  std::string token;
  while (getline(slicing_stream, token)) {
    sliced_edges.push_back(std::stoi(token));
  }
  return sliced_edges;
}

std::vector<size_t> GetSlicedEdges(const std::vector<size_t>& slicing_ordering,
                                   size_t num_sliced_edges) {
  return std::vector<size_t>({slicing_ordering.cbegin(),
                              slicing_ordering.cbegin() + num_sliced_edges});
}

// Insert elements to result in the next three methods.
template <typename T>
void SetUnion(const std::set<T>& s0, const std::set<T>& s1,
              std::set<T>& result) {
  for (auto x0 : s0) {
    result.insert(x0);
  }
  for (auto x1 : s1) {
    result.insert(x1);
  }
}

template <typename T>
void SetIntersection(const std::set<T>& s0, const std::set<T>& s1,
                     std::set<T>& result) {
  for (auto x0 : s0) {
    if (s1.find(x0) != s1.end()) {
      result.insert(x0);
    }
  }
}

template <typename T>
void SetDifference(const std::set<T>& s0, const std::set<T>& s1,
                   std::set<T>& result) {
  for (auto x0 : s0) {
    if (s1.find(x0) == s1.end()) {
      result.insert(x0);
    }
  }
}

template <typename T>
void VectorUnion(const std::vector<T>& s0, const std::vector<T>& s1,
                 std::vector<T>& result) {
  size_t s0_size = s0.size();
  size_t s1_size = s1.size();
  result.resize(s0_size + s1_size);
  for (size_t i = 0; i < s0_size; i++) {
    result[i] = s0[i];
  }
  for (size_t i = 0; i < s1_size; i++) {
    result[s0_size + i] = s1[i];
  }
}
