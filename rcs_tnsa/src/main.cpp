#include <algorithm>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <vector>

#include "sa_optimizer.h"    // NOLINT(build/include)
#include "tensor_network.h"  // NOLINT(build/include)
#include "utils.h"           // NOLINT(build/include)

int main(int argc, char** argv) {
  // Argument handling.
  int mode;
  std::string graph_filename;
  std::string groups_filename;
  std::string output_filename;
  int random_seed;
  double m;  // Memory constraint.
  int num_output_confs;
  int num_steps;
  double T0;
  double T1;
  std::string message =
      "Wrong arguments. Run `./optimize -h` for help and instructions.";

  // Help instructions begin.
  std::string instructions =
      ""
      "########################################################################"
      "########\n"
      "Tensor network contraction optimizer. "
      "This program can be run in three modes:\n"
      "  1) Mode 1: optimize contraction ordering without memory constraints.\n"
      "  2) Mode 2: optimize contraction ordering with width constraint, "
      "grouped slices, and sparse outputs.\n"
      "  3) Mode 3: optimize contraction ordering with full memory footprint "
      "constraint, grouped slices, and sparse outputs.\n\n"
      "Usage:\n"
      "  1) ./optimize 1 <graph filename> <output filename> <random seed> <num "
      "SA steps> <initial SA temperature T0>  <final SA temperature T1>\n"
      "  2) ./optimize 2 <graph filename> <groups filename> <output filename> "
      "<random seed> <width> <num sparse configurations> <num SA steps> "
      "<initial SA temperature T0>  <final SA temperature T1>\n"
      "  3) ./optimize 3 <graph filename> <groups filename> <output filename> "
      "<random seed> <log2(memory footprint (num scalars))> <num sparse "
      "configurations> <num SA steps> <initial SA temperature T0> "
      "<final SA temperature T1>\n\n"
      "For more details and examples on the format for the `.graph` and "
      "`.groups` files see "
      "https://github.com/google-research/google-research/tree/master/rcs_tnsa "
      "and https://www.nature.com/articles/s41586-024-07998-6.\n"
      "########################################################################"
      "########\n";
  // Help instructions end.

  if (argc == 2) {
    std::string arg = argv[1];
    if (arg == "-h" || arg == "--help") {
      std::cout << instructions << std::endl;
      return 2;
    }
  } else if (8 <= argc && argc <= 11) {
    mode = std::stoi(argv[1]);
    if (mode == 1)  // Non-memory constrained.
    {
      if (argc != 8) {
        std::cout << message << std::endl;
        return 2;
      }
      graph_filename = argv[2];
      output_filename = argv[3];
      random_seed = std::stoi(argv[4]);
      num_steps = std::stoi(argv[5]);
      T0 = std::stod(argv[6]);
      T1 = std::stod(argv[7]);
    }
    if (mode == 2 || mode == 3)  // Memory constrained.
    {
      if (argc != 11) {
        std::cout << message << std::endl;
        return 2;
      }
      graph_filename = argv[2];
      groups_filename = argv[3];
      output_filename = argv[4];
      random_seed = std::stoi(argv[5]);
      m = std::stod(argv[6]);
      num_output_confs = std::stoi(argv[7]);
      num_steps = std::stoi(argv[8]);
      T0 = std::stod(argv[9]);
      T1 = std::stod(argv[10]);
    }
  } else {
    std::cout << message << std::endl;
    return 2;
  }

  TensorNetwork tn(graph_filename);
  std::vector<size_t> ordering(tn.GetContractingEdgeNums());
  std::seed_seq seed({random_seed});
  std::mt19937 g(random_seed);
  // std::random_shuffle(ordering.begin(), ordering.end());
  std::shuffle(ordering.begin(), ordering.end(), g);
  SAOptimizer sao(tn);
  sao.SetSeed(random_seed);
  if (mode == 1) {
    double cost = sao.Optimize(T0, T1, num_steps, ordering);
    std::ofstream output_file;
    output_file.open(output_filename);
    output_file << "Info:\n";
    output_file
        << "  Mode 1: optimize contraction ordering without memory constraints."
        << "\n";
    output_file << "  Graph file: " << graph_filename << "\n";
    output_file << "  Seed: " << random_seed << "\n";
    output_file << "  Num steps: " << num_steps << "\n";
    output_file << "  T0: " << T0 << "\n";
    output_file << "  T1: " << T1 << "\n";
    output_file << "Cost (Log2(FLOPs)):\n";
    output_file << cost << "\n";
    output_file << "Ordering:\n";
    for (auto v : ordering) {
      output_file << v << " ";
    }
    output_file << "\n";
  }
  if (mode == 2) {
    std::vector<std::vector<size_t>> slices_ordering =
        utils::ReadGroups(groups_filename);
    // std::random_shuffle(slices_ordering.begin(), slices_ordering.end());
    std::shuffle(slices_ordering.begin(), slices_ordering.end(), g);
    auto [cost, num_slices] = sao.SlicedOptimizeGroupedSlicesSparseOutput(
        T0, T1, num_steps, m, num_output_confs, ordering, slices_ordering);
    cost *= utils::SCALAR_FACTOR;
    std::ofstream output_file;
    output_file.open(output_filename);
    output_file << "Info:\n";
    output_file << "  Mode 2: optimize contraction ordering with width "
                   "constraint, grouped slices, and sparse outputs."
                << "\n";
    output_file << "  Graph file: " << graph_filename << "\n";
    output_file << "  Groups file: " << groups_filename << "\n";
    output_file << "  Seed: " << random_seed << "\n";
    output_file << "  Width: " << m << "\n";
    output_file << "  Num output configurations: " << num_output_confs << "\n";
    output_file << "  Num steps: " << num_steps << "\n";
    output_file << "  T0: " << T0 << "\n";
    output_file << "  T1: " << T1 << "\n";
    output_file << "Cost (Log2(FLOPs)):\n";
    output_file << cost << "\n";
    output_file << "Ordering:\n";
    for (auto v : ordering) {
      output_file << v << " ";
    }
    output_file << "\n";
    output_file << "Sliced groups (" << num_slices << "):\n";
    for (size_t i = 0; i < num_slices; i++) {
      for (auto v : slices_ordering[i]) {
        output_file << v << " ";
      }
      output_file << "\n";
    }
  }
  if (mode == 3) {
    std::vector<std::vector<size_t>> slices_ordering =
        utils::ReadGroups(groups_filename);
    // std::random_shuffle(slices_ordering.begin(), slices_ordering.end());
    std::shuffle(slices_ordering.begin(), slices_ordering.end(), g);
    auto [cost, num_slices] =
        sao.SlicedOptimizeFullMemoryGroupedSlicesSparseOutput(
            T0, T1, num_steps, m, num_output_confs, ordering, slices_ordering);
    cost *= utils::SCALAR_FACTOR;
    std::ofstream output_file;
    output_file.open(output_filename);
    output_file << "Info:\n";
    output_file << "  Mode 3: optimize contraction ordering with memory "
                   "footprint constraint, grouped slices, and sparse outputs."
                << "\n";
    output_file << "  Graph file: " << graph_filename << "\n";
    output_file << "  Groups file: " << groups_filename << "\n";
    output_file << "  Seed: " << random_seed << "\n";
    output_file << "  Log2(memory footprint in num scalars): " << m << "\n";
    output_file << "  Num output configurations: " << num_output_confs << "\n";
    output_file << "  Num steps: " << num_steps << "\n";
    output_file << "  T0: " << T0 << "\n";
    output_file << "  T1: " << T1 << "\n";
    output_file << "Cost (Log2(FLOPs)):\n";
    output_file << cost << "\n";
    output_file << "Ordering:\n";
    for (auto v : ordering) {
      output_file << v << " ";
    }
    output_file << "\n";
    output_file << "Sliced groups (" << num_slices << "):\n";
    for (size_t i = 0; i < num_slices; i++) {
      for (auto v : slices_ordering[i]) {
        output_file << v << " ";
      }
      output_file << "\n";
    }
  }

  return 0;
}
