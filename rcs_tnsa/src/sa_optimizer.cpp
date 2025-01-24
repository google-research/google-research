#include "sa_optimizer.h"  // NOLINT(build/include)

////// PUBLIC METHODS //////

SAOptimizer::SAOptimizer(const TensorNetwork& tn) { tn_ = tn; }

void SAOptimizer::SetSeed(unsigned int seed) {
  // We use the same seed for both.
  seed_ = seed;
  std::srand(seed);
  dre_ = std::default_random_engine(seed);
}

double SAOptimizer::Optimize(double t0, double t1, size_t num_ts,
                             std::vector<size_t>& ordering) {
  // Get schedule of temperatures.
  std::vector<double> ts = LinInvSchedule(t0, t1, num_ts);

  // Initialize SA variables.
  std::vector<size_t> ord(ordering);
  std::vector<size_t> proposed_ord(ordering);
  double cost, proposed_cost;
  auto breakdown = tn_.ContractionBreakdown(ord);
  cost = tn_.Log2Flops(breakdown);

  // Run SA.
  for (size_t i = 0; i < ts.size(); i++) {
    double t = ts[i];
    // Move to proposed ordering.
    proposed_ord = ord;
    SwapMove(proposed_ord);

    // Compute cost for proposed_ord.
    breakdown = tn_.ContractionBreakdown(proposed_ord);
    proposed_cost = tn_.Log2Flops(breakdown);

    // Compare to decide whether to accept.
    double coin = (double)std::rand() / (RAND_MAX);
    double ratio = std::pow(2., (cost - proposed_cost) / t);
    bool accept = coin < ratio ? true : false;

    // Update variables if accepted.
    if (accept) {
      cost = proposed_cost;
      ord = proposed_ord;
    }
  }

  ordering = ord;
  return cost;
}

std::tuple<double, size_t> SAOptimizer::SlicedOptimizeGroupedSlicesSparseOutput(
    double t0, double t1, size_t num_ts, double max_width,
    size_t num_output_confs, std::vector<size_t>& ordering,
    std::vector<std::vector<size_t>>& slices_groups_ordering) {
  // Get schedule of temperatures.
  std::vector<double> ts = LinInvSchedule(t0, t1, num_ts);

  // Initialize SA variables.
  std::vector<size_t> ord(ordering);
  std::vector<std::vector<size_t>> slices_ord(slices_groups_ordering);
  std::vector<size_t> proposed_ord(ordering);
  std::vector<std::vector<size_t>> proposed_slices_ord(slices_groups_ordering);
  double cost, proposed_cost;
  size_t num_slices, proposed_num_slices;
  auto breakdown = tn_.ContractionBreakdown(ord);
  num_slices = tn_.NumSlicesGivenWidthGroupedSlicesSparseOutput(
      breakdown, max_width, slices_ord, num_output_confs);
  std::vector<std::vector<size_t>> sliced_grouped_edge_nums(
      slices_ord.begin(), slices_ord.begin() + num_slices);
  std::vector<std::vector<size_t>> proposed_sliced_edge_nums;
  cost = tn_.SlicedMemoizedLog2FlopsGroupedSlicesSparseOutput(
      breakdown, sliced_grouped_edge_nums, num_output_confs);

  // Run SA.
  for (size_t i = 0; i < ts.size(); i++) {
    double t = ts[i];
    // Move to proposed orderings.
    proposed_ord = ord;
    proposed_slices_ord = slices_ord;
    if (i % 10 == 0)
      SlicesMove(proposed_slices_ord, num_slices);
    else
      SwapMove(proposed_ord);

    // Compute cost for proposed_ord.
    breakdown = tn_.ContractionBreakdown(proposed_ord);
    proposed_num_slices = tn_.NumSlicesGivenWidthGroupedSlicesSparseOutput(
        breakdown, max_width, proposed_slices_ord, num_output_confs);
    proposed_sliced_edge_nums = std::vector<std::vector<size_t>>(
        proposed_slices_ord.begin(),
        proposed_slices_ord.begin() + proposed_num_slices);
    proposed_cost = tn_.SlicedMemoizedLog2FlopsGroupedSlicesSparseOutput(
        breakdown, proposed_sliced_edge_nums, num_output_confs);

    // Compare to decide whether to accept.
    double coin = (double)std::rand() / (RAND_MAX);
    double ratio = std::pow(2., (cost - proposed_cost) / t);
    bool accept = coin < ratio ? true : false;

    // Update variables if accepted.
    if (accept) {
      cost = proposed_cost;
      ord = proposed_ord;
      slices_ord = proposed_slices_ord;
      num_slices = proposed_num_slices;
    }
  }

  ordering = ord;
  slices_groups_ordering = slices_ord;

  return {cost, num_slices};
}

std::tuple<double, size_t>
SAOptimizer::SlicedOptimizeFullMemoryGroupedSlicesSparseOutput(
    double t0, double t1, size_t num_ts, double log2_max_memory,
    size_t num_output_confs, std::vector<size_t>& ordering,
    std::vector<std::vector<size_t>>& slices_groups_ordering) {
  // Get schedule of temperatures.
  std::vector<double> ts = LinInvSchedule(t0, t1, num_ts);

  // Initialize SA variables.
  std::vector<size_t> ord(ordering);
  std::vector<std::vector<size_t>> slices_ord(slices_groups_ordering);
  std::vector<size_t> proposed_ord(ordering);
  std::vector<std::vector<size_t>> proposed_slices_ord(slices_groups_ordering);
  double cost, proposed_cost;
  size_t num_slices, proposed_num_slices;
  auto breakdown = tn_.ContractionBreakdown(ord);
  num_slices = tn_.NumSlicesGivenFullMemoryGroupedSlicesSparseOutput(
      breakdown, log2_max_memory, slices_ord, num_output_confs);
  std::vector<std::vector<size_t>> sliced_grouped_edge_nums(
      slices_ord.begin(), slices_ord.begin() + num_slices);
  std::vector<std::vector<size_t>> proposed_sliced_edge_nums;
  cost = tn_.SlicedMemoizedLog2FlopsGroupedSlicesSparseOutput(
      breakdown, sliced_grouped_edge_nums, num_output_confs);

  // Run SA.
  for (size_t i = 0; i < ts.size(); i++) {
    double t = ts[i];
    // Move to proposed orderings.
    proposed_ord = ord;
    proposed_slices_ord = slices_ord;
    if (i % 10 == 0)
      SlicesMove(proposed_slices_ord, num_slices);
    else
      SwapMove(proposed_ord);

    // Compute cost for proposed_ord.
    breakdown = tn_.ContractionBreakdown(proposed_ord);
    proposed_num_slices = tn_.NumSlicesGivenFullMemoryGroupedSlicesSparseOutput(
        breakdown, log2_max_memory, proposed_slices_ord, num_output_confs);
    proposed_sliced_edge_nums = std::vector<std::vector<size_t>>(
        proposed_slices_ord.begin(),
        proposed_slices_ord.begin() + proposed_num_slices);
    proposed_cost = tn_.SlicedMemoizedLog2FlopsGroupedSlicesSparseOutput(
        breakdown, proposed_sliced_edge_nums, num_output_confs);

    // Compare to decide whether to accept.
    double coin = (double)std::rand() / (RAND_MAX);
    double ratio = std::pow(2., (cost - proposed_cost) / t);
    bool accept = coin < ratio ? true : false;

    // Update variables if accepted.
    if (accept) {
      cost = proposed_cost;
      ord = proposed_ord;
      slices_ord = proposed_slices_ord;
      num_slices = proposed_num_slices;
    }
  }

  ordering = ord;
  slices_groups_ordering = slices_ord;

  return {cost, num_slices};
}

////// EXTERNAL METHODS //////

void SwapMove(std::vector<size_t>& ordering) {
  size_t idx_0 = std::rand() % ordering.size();
  size_t idx_1;
  do {
    idx_1 = std::rand() % ordering.size();
  } while (idx_0 == idx_1);
  size_t aux = ordering[idx_0];
  ordering[idx_0] = ordering[idx_1];
  ordering[idx_1] = aux;
}

template <typename T>
void SlicesMove(std::vector<T>& slices_ordering, size_t num_slices) {
  size_t idx_0 = std::rand() % num_slices;
  size_t idx_1 = std::rand() % (slices_ordering.size() - num_slices);
  T aux = slices_ordering[idx_0];
  slices_ordering[idx_0] = slices_ordering[idx_1];
  slices_ordering[idx_1] = aux;
}

std::vector<double> LinInvSchedule(double t0, double t1, double num_ts) {
  std::vector<double> ts(num_ts);
  double inv_t0 = 1. / t0;
  double inv_t1 = 1. / t1;
  double d_inv_t = (inv_t1 - inv_t0) / (num_ts - 1);
  for (size_t i = 0; i < num_ts; i++) {
    ts[i] = 1. / (inv_t0 + d_inv_t * i);
  }

  return ts;
}
