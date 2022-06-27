// Copyright 2022 The Google Research Authors.
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

// This file contains the definitions of parallel sequence primitives
// which exploit multi-core parallelism.
//
// WARNING: The interfaces in this file are experimental and are liable to
// change.

#ifndef RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_SEQUENCE_OPS_H_
#define RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_SEQUENCE_OPS_H_

#include "absl/types/span.h"
#include "external/gbbs/gbbs/bridge.h"
#include "external/gbbs/gbbs/macros.h"
#include "external/gbbs/pbbslib/counting_sort.h"
#include "external/gbbs/pbbslib/integer_sort.h"
#include "external/gbbs/pbbslib/monoid.h"
#include "external/gbbs/pbbslib/quicksort.h"
#include "external/gbbs/pbbslib/sample_sort.h"
#include "external/gbbs/pbbslib/seq.h"
#include "external/gbbs/pbbslib/sequence_ops.h"
#include "external/gbbs/pbbslib/utilities.h"
#include "parallel/parallel-utils.h"

namespace research_graph {
namespace parallel {

// ======== Parallel reduction, scan, filtering, and packing methods ======== //

// Given an array of A's, In, and an associative f: A -> A -> A, computes the
// "sum" of all elements in In with respect to f.
template <class A>
A Reduce(absl::Span<const A> In, const std::function<A(A, A)>& f, A zero) {
  auto get_in = [&](size_t i) { return In.at(i); };
  auto seq_in = pbbs::delayed_seq<A>(In.size(), get_in);
  auto monoid = pbbs::make_monoid(f, zero);
  return pbbs::reduce(seq_in, monoid);
}

// Given an array of A's, In, computes the sum (using the operator "+") of all
// elements in In.
template <class A>
A ReduceAdd(absl::Span<const A> In) {
  auto get_in = [&](size_t i) { return In.at(i); };
  auto seq_in = pbbs::delayed_seq<A>(In.size(), get_in);
  return pbbslib::reduce_add(seq_in);
}

// Given an array of A's, In, computes the sum (using "std::max") of all
// elements in In.
template <class A>
A ReduceMax(absl::Span<const A> In) {
  auto get_in = [&](size_t i) { return In.at(i); };
  auto seq_in = pbbs::delayed_seq<A>(In.size(), get_in);
  return pbbslib::reduce_max(seq_in);
}

// Given an array of A's, In, computes the sum (using "std::min") of all
// elements in In.
template <class A>
A ReduceMin(absl::Span<const A> In) {
  auto get_in = [&](size_t i) { return In.at(i); };
  auto seq_in = pbbs::delayed_seq<A>(In.size(), get_in);
  return pbbslib::reduce_min(seq_in);
}

// Given an array of A's, In, an
// associative function, f : A -> A -> A, and an identity, zero, computes the
// prefix sum of A with respect to f and zero. That is, the output is [zero,
// f(zero, In[0]), f(f(zero, In[0]),n[1]), ..]. If inclusive is true, the output
// is [f(zero, In[0]), f(f(zero, In[0]),n[1]), ... ], that is the output of
// f(..), including the element at index i, is written to the i'th element of
// the output. The output is a pbbs::sequence of the scan'd vector, and the
// overall sum.
template <class A>
std::pair<pbbs::sequence<A>, A> Scan(absl::Span<const A> In,
                                     const std::function<A(A, A)>& f, A zero,
                                     bool inclusive = false) {
  auto get_in = [&](size_t i) { return In.at(i); };
  auto seq_in = pbbs::delayed_seq<A>(In.size(), get_in);
  auto fl = (inclusive) ? pbbs::fl_scan_inclusive : pbbs::no_flag;
  auto monoid = pbbs::make_monoid(f, zero);
  return pbbs::scan(seq_in, monoid, fl);
}

// Computes the prefix-sum of In returns it using + as f, and 0 as
// zero (see the description for scan above for more details).
template <class A>
std::pair<pbbs::sequence<A>, A> ScanAdd(absl::Span<const A> In,
                                        bool inclusive = false) {
  auto get_in = [&](size_t i) { return In.at(i); };
  auto seq_in = pbbs::delayed_seq<A>(In.size(), get_in);
  auto fl = (inclusive) ? pbbs::fl_scan_inclusive : pbbs::no_flag;
  return pbbs::scan(seq_in, pbbs::addm<A>(), fl);
}

// Given an array of A's, In, and a predicate function, Flags, indicating the
// indices corresponding to elements that should be in the output, Pack computes
// an array of A's containing only elements, In[i] in In s.t. Flags[i] == true.
// Flags must be a function that is callable on indices in the range [0, n).
template <class A>
pbbs::sequence<A> Pack(absl::Span<const A> In,
                       const std::function<bool(size_t)>& Flags) {
  auto get_in = [&](size_t i) { return In.at(i); };
  auto seq_in = pbbs::delayed_seq<A>(In.size(), get_in);
  auto bool_seq = pbbs::delayed_seq<bool>(In.size(), Flags);
  return pbbs::pack(seq_in, bool_seq, pbbs::no_flag);
}

// Returns an array of Idx_Type (e.g., size_t's) containing only those indices i
// s.t. Flags[i] == true. Flags must be a function that is callable on each
// element of the range [0, n).
template <class Idx_Type>
pbbs::sequence<Idx_Type> PackIndex(const std::function<bool(size_t)>& Flags,
                                   size_t num_elements) {
  auto flags_sequence = pbbs::delayed_seq<bool>(num_elements, Flags);
  return pbbs::pack_index<Idx_Type>(flags_sequence);
}

// Given an array of A's, In, and a predicate function, p : A -> bool, computes
// an array of A's containing only elements, e in In s.t. p(e) == true.
template <class A>
pbbs::sequence<A> Filter(absl::Span<const A> In,
                         const std::function<bool(A)>& p) {
  auto get_in = [&](size_t i) { return In.at(i); };
  auto seq_in = pbbs::delayed_seq<A>(In.size(), get_in);
  return pbbs::filter(seq_in, p, pbbs::no_flag);
}

template <class A>
std::vector<A> FilterOut(absl::Span<const A> In,
                         const std::function<bool(A)>& p) {
  std::vector<A> out(In.size());
  auto seq_out = pbbslib::make_sequence(out.data(), out.size());
  auto get_in = [&](size_t i) { return In.at(i); };
  auto seq_in = pbbs::delayed_seq<A>(In.size(), get_in);
  size_t size_out = pbbs::filter_out(seq_in, seq_out, p, pbbs::no_flag);
  out.resize(size_out);
  out.shrink_to_fit();
  return out;
}

// Given a number of keys and a key_eq_func that takes as input two indices and
// outputs whether the corresponding keys are equal, create a vector consisting
// of all i such that 1 <= i < num_keys and key(i) != key(i-1), sorted in
// increasing order. An additional element is appended to the end of the vector,
// with value num_keys.
// Note that num_keys must be less than std::numeric_limits<A>::max().
template <class A>
std::vector<A> GetBoundaryIndices(
    std::size_t num_keys,
    const std::function<bool(std::size_t, std::size_t)>& key_eq_func) {
  std::vector<A> mark_keys(num_keys);
  auto null_key = std::numeric_limits<A>::max();
  pbbs::parallel_for(0, num_keys, [&](std::size_t i) {
    if (i != 0 && key_eq_func(i, i - 1))
      mark_keys[i] = null_key;
    else
      mark_keys[i] = i;
  });
  std::vector<A> filtered_mark_keys =
      FilterOut<A>(absl::Span<const A>(mark_keys.data(), mark_keys.size()),
                   [&null_key](A x) -> bool { return x != null_key; });
  filtered_mark_keys.push_back(num_keys);
  return filtered_mark_keys;
}

// Given ids (class A) and a function indicating which indices (class B) to
// process, re-arrange the indices such that they are grouped by id, and return
// a vector of these groups.
template <class A, class B>
std::vector<std::vector<B>> OutputIndicesById(
    const std::vector<A>& index_ids,
    const std::function<B(B)>& get_indices_func, B num_indices) {
  // Sort all vertices by cluster id
  auto indices_sort = pbbs::sample_sort(
      pbbs::delayed_seq<B>(num_indices, get_indices_func),
      [&](B a, B b) { return index_ids[a] < index_ids[b]; }, true);

  // Obtain the boundary indices where cluster ids differ
  std::vector<B> filtered_mark_ids = GetBoundaryIndices<B>(
      indices_sort.size(), [&](std::size_t i, std::size_t j) {
        return index_ids[indices_sort[i]] == index_ids[indices_sort[j]];
      });
  std::size_t num_filtered_mark_ids = filtered_mark_ids.size() - 1;

  // Boundary indices indicate sections corresponding to clusters
  std::vector<std::vector<B>> finished_indices(num_filtered_mark_ids);
  pbbs::parallel_for(0, num_filtered_mark_ids, [&](std::size_t i) {
    B start_id_index = filtered_mark_ids[i];
    B end_id_index = filtered_mark_ids[i + 1];
    finished_indices[i] = std::vector<B>(indices_sort.begin() + start_id_index,
                                         indices_sort.begin() + end_id_index);
  });

  return finished_indices;
}

// ===================   Parallel sorting methods ========================= //

// Takes a span of A's, In, a function get_key which specifies the integer key
// for each element, and val_bits which specifies how many bits of the keys to
// use. Sorts In using a top-down recursive radix sort with respect to the keys
// provided by get_key. The result is returned as a new sorted sequence.
template <typename A>
pbbs::sequence<A> ParallelIntegerSort(
    absl::Span<A> In, const std::function<size_t(size_t)>& get_key,
    size_t val_bits) {
  auto seq_in = pbbs::make_range(In.data(), In.data() + In.size());
  return pbbs::integer_sort(seq_in, get_key, val_bits);
}

// Takes a span of A's, In, and a comparison function f. Sorts In with respect
// to f using a sample sort. The result is returned as a new sorted sequence.
template <class A>
pbbs::sequence<A> ParallelSampleSort(absl::Span<A> In,
                                     const std::function<bool(A, A)>& f) {
  auto seq_in = pbbs::make_range(In.data(), In.data() + In.size());
  return pbbs::sample_sort(seq_in, f, true);
}

// Takes a span of A's, In, a function keys which takes the index of an element
// and returns the integer key (the bucket) for the element, and the total
// number of buckets (the size of the range of keys). In is sorted with respect
// to the buckets given by keys.
template <typename A>
pbbs::sequence<A> ParallelCountSort(
    absl::Span<A> In, const std::function<size_t(size_t)>& get_key,
    size_t num_buckets) {
  auto seq_in = pbbs::make_range(In.data(), In.data() + In.size());
  auto seq_out = pbbs::sequence<A>(In.size());
  auto key_seq = pbbs::delayed_seq<size_t>(In.size(), get_key);
  pbbs::count_sort(seq_in, seq_out.slice(), key_seq, num_buckets);
  return seq_out;
}

}  // namespace parallel
}  // namespace research_graph

#endif  // RESEARCH_GRAPH_IN_MEMORY_PARALLEL_PARALLEL_SEQUENCE_OPS_H_
