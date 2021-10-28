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

// Implementation of "Efficient Large Scale Generalized Voting for Geometric
// Vision Problems", Dror Aiger, Jan Hosang, Bernhard Zeisl, Simon Lynen.
#ifndef EXPERIMENTAL_USERS_AIGERD_GENERAL_VOTING_ICCV_21_TOWARD_OPEN_SOURCE_GENERAL_VOTER_H_
#define EXPERIMENTAL_USERS_AIGERD_GENERAL_VOTING_ICCV_21_TOWARD_OPEN_SOURCE_GENERAL_VOTER_H_

#include <algorithm>
#include <cstddef>
#include <functional>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/barrier.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "concurrency.h"
#include "status_macros.h"

namespace large_scale_voting {

template <typename Scalar, int D>
absl::Span<const Scalar> EigenVectorToSpan(
    const Eigen::Matrix<Scalar, D, 1>& vector) {
  return absl::MakeConstSpan(vector);
}

// A hack to access the underlying container of a priority_queue.
template <class T, class S, class C>
const S& Container(const std::priority_queue<T, S, C>& q) {
  struct HackedQueue : private std::priority_queue<T, S, C> {
    static const S& Container(const std::priority_queue<T, S, C>& q) {
      return q.*&HackedQueue::c;
    }
  };
  return HackedQueue::Container(q);
}

template <typename T, typename Comperator>
class BoundedPriorityQueue {
 public:
  using const_iterator = typename std::vector<T>::const_iterator;

  void set_max_size(size_t max_size) { max_size_ = max_size; }
  bool full() const { return prio_queue_.size() == max_size_; }
  bool empty() const { return prio_queue_.empty(); }
  const T& top() const { return prio_queue_.top(); }
  void push(T item);

  const const_iterator begin() const { return Container(prio_queue_).begin(); }
  const const_iterator end() const { return Container(prio_queue_).end(); }

 private:
  size_t max_size_;
  std::priority_queue<T, std::vector<T>, Comperator> prio_queue_;
};

template <typename T, typename Comperator>
void BoundedPriorityQueue<T, Comperator>::push(T item) {
  if (prio_queue_.size() >= max_size_) {
    if (!Comperator()(item, top())) {
      return;
    }
    prio_queue_.pop();
  }
  prio_queue_.push(std::move(item));
}

// A D-dimensional hypercube.
template <size_t D>
struct HyperOct {
  using VectorDf = Eigen::Matrix<float, D, 1>;

  VectorDf min = VectorDf::Zero();
  VectorDf max = VectorDf::Zero();

  HyperOct() = default;
  VectorDf mid() const { return (min + max) / 2.0; }
  VectorDf side_length() const { return max - min; }

  HyperOct<D> extend(const VectorDf& eps) const {
    return {min - eps, max + eps};
  }
  bool Contain(const VectorDf& point) const {
    for (int i = 0; i < D; ++i) {
      if (point(i) < min(i) || point(i) > max(i)) return false;
    }
    return true;
  }
};

// Iterator over all children HyperOcts with half the side length of the parent.
template <size_t D>
class HyperOctIterator {
 public:
  using VectorDf = Eigen::Matrix<float, D, 1>;

  explicit HyperOctIterator(const HyperOct<D>& parent,
                            const VectorDf& min_side_length)
      : parent_(parent), index_(0), mid_(parent.mid()) {
    const VectorDf side_length = parent.max - parent.min;
    subdivision_dimensions_.reserve(D);
    for (size_t d = 0; d < D; ++d) {
      if (side_length[d] > min_side_length[d]) {
        subdivision_dimensions_.push_back(d);
      }
    }
  }

  bool valid() const { return index_ < (1 << subdivision_dimensions_.size()); }
  void advance() { index_ += 1; }

  HyperOct<D> operator*() {
    HyperOct<D> child = parent_;
    for (size_t i = 0; i < subdivision_dimensions_.size(); ++i) {
      const int dim = subdivision_dimensions_[i];
      // Should we pick the first or second half in this dimensions?
      const bool first_half = (index_ & (1 << i)) == 0;
      if (first_half) {
        child.max[dim] = mid_[dim];
      } else {
        child.min[dim] = mid_[dim];
      }
    }
    return child;
  }

 private:
  const HyperOct<D> parent_;
  int index_;
  const VectorDf mid_;
  std::vector<int> subdivision_dimensions_;
};

// NumDimensions: Number of dimensions of the voting space (d in the paper).
// NumEssentialDimensions: k in the paper.
// NumEssentialParameters: l in the paper.
// ScalarT: Scalar type of the parameters and for performing the rounding
// operations.
template <size_t NumDimensions, size_t NumEssentialDimensions,
          size_t NumEssentialParameters, typename ScalarT>
class Surface {
  static_assert(NumDimensions > NumEssentialDimensions,
                "Surfaces need at least one essential and one free dimension.");

 public:
  // Note that the number of free dimensions is the same as the number of free
  // parameters, which is not the case for essential dimensions and parameters.
  static constexpr size_t kNumFreeParameters =
      NumDimensions - NumEssentialDimensions;
  using EssentialVector = Eigen::Matrix<ScalarT, NumEssentialDimensions, 1>;
  using FreeVector = Eigen::Matrix<ScalarT, kNumFreeParameters, 1>;
  using EssentialParameterVector =
      Eigen::Matrix<ScalarT, NumEssentialParameters, 1>;
  using TypedSurface = Surface<NumDimensions, NumEssentialDimensions,
                               NumEssentialParameters, ScalarT>;

  using Oct = HyperOct<NumDimensions>;
  using SurfaceFn = std::function<FreeVector(const EssentialVector&,
                                             const EssentialParameterVector&)>;
  using DoesIntersectFn =
      std::function<bool(const EssentialParameterVector&, const FreeVector&,
                         const HyperOct<NumDimensions>&)>;

  // |surface_fn|: The F in "x_dependent = F(x;t) + f" in the paper, i.e. not
  // including the "+ f".
  Surface(EssentialParameterVector essential_parameters,
          FreeVector free_parameters, SurfaceFn surface_fn,
          DoesIntersectFn does_intersect_fn)
      : surface_fn_(std::move(surface_fn)),
        does_intersect_fn_(std::move(does_intersect_fn)),
        essential_parameters_(std::move(essential_parameters)),
        free_parameters_(std::move(free_parameters)) {}

  void Round(const Oct& oct, EssentialParameterVector epsilon_essential,
             FreeVector epsilon_free) {
    const ScalarT delta = oct.side_length().maxCoeff();
    const EssentialParameterVector rounded_essential_parameters =
        RoundVector<NumEssentialParameters>(essential_parameters_,
                                            epsilon_essential / delta);

    const Eigen::Matrix<ScalarT, NumDimensions, 1> xi =
        oct.min.template cast<ScalarT>();
    const EssentialVector xi_essential =
        xi.template head<NumEssentialDimensions>();
    const FreeVector xi_free = xi.template tail<kNumFreeParameters>();
    // First translate the function so |xi| is the new origin, then round, then
    // translate back. That means on the translation back, we need to use the
    // rounded essential parameters.
    free_parameters_ =
        RoundVector<kNumFreeParameters>(
            free_parameters_ +
                surface_fn_(xi_essential, essential_parameters_) - xi_free,
            epsilon_free) -
        surface_fn_(xi_essential, rounded_essential_parameters) + xi_free;
    essential_parameters_ = rounded_essential_parameters;
  }

  bool DoesIntersect(const Oct& oct) const {
    return does_intersect_fn_(essential_parameters_, free_parameters_, oct);
  }

  bool operator==(const TypedSurface& other) const {
    return essential_parameters_ == other.essential_parameters_ &&
           free_parameters_ == other.free_parameters_;
  }

  // We collapse surfaces that have been rounded to the same canonical surface,
  // by using the surface as a key in a hash map.
  template <typename H>
  friend H AbslHashValue(H h, const TypedSurface& s) {
    return H::combine(std::move(h), EigenVectorToSpan(s.essential_parameters_),
                      EigenVectorToSpan(s.free_parameters_));
  }

  const EssentialParameterVector& essential_parameters() const {
    return essential_parameters_;
  }
  const FreeVector& free_parameters() const { return free_parameters_; }

 private:
  // Round |vector| to the closest integer multiple of |step|.
  template <int D>
  Eigen::Matrix<ScalarT, D, 1> RoundVector(
      const Eigen::Matrix<ScalarT, D, 1>& vector,
      const Eigen::Matrix<ScalarT, D, 1>& step) const {
    return ((vector.array() / step.array()).round() * step.array()).matrix();
  }

  const SurfaceFn surface_fn_;
  const DoesIntersectFn does_intersect_fn_;
  EssentialParameterVector essential_parameters_ =
      EssentialParameterVector::Zero();
  FreeVector free_parameters_ = FreeVector::Zero();
};

// Efficient version of voting: In naive voting we render a set of surfaces into
// D-dimensional space and look for one/multiple cells that receive most votes.
// Instead this search uses a successive subdivision of space, while rounding
// the surfaces to reduce the number of intersection computations.
//
// TODO(b/145888020): Add functor for finishing up the work for few surfaces.
template <size_t D, typename Surface>
class GeneralVoter {
 public:
  using Oct = HyperOct<D>;
  using VectorDf = Eigen::Matrix<float, D, 1>;
  using EssentialParameterVector = typename Surface::EssentialParameterVector;
  using FreeVector = typename Surface::FreeVector;
  using ScoreHypothesisFn = std::function<float(const Oct&, float score)>;
  using VerificationFunction = std::function<std::vector<int>(
      const Oct&, const std::vector<int>& inliers)>;

  struct Maximum {
    float score = 0.0;
    int num_intersections = 0;
    std::vector<int> surface_indices;
    Oct oct;

    bool operator<(const Maximum& other) const { return score < other.score; }
    bool operator>(const Maximum& other) const { return score > other.score; }
  };

  // |minimal_oct_side_length| is the epsilon in the paper.
  explicit GeneralVoter(
      VectorDf minimal_oct_side_length, int num_threads, int max_num_results,
      EssentialParameterVector epsilon_essential, FreeVector epsilon_free,
      int min_num_surfaces, int max_num_surfaces, float ext_factor,
      float max_time_sec, ScoreHypothesisFn score_hypothesis_fn,
      VerificationFunction verification_fn, bool enable_rounding)
      : minimal_oct_side_length_(minimal_oct_side_length),
        num_threads_(num_threads),
        max_num_results_(max_num_results),
        epsilon_essential_(epsilon_essential),
        epsilon_free_(epsilon_free),
        min_num_surfaces_(min_num_surfaces),
        max_num_surfaces_(max_num_surfaces),
        ext_factor_(ext_factor),
        max_time_sec_(max_time_sec),
        score_hypothesis_fn_(score_hypothesis_fn),
        verification_fn_(verification_fn),
        enable_rounding_(enable_rounding) {}

  // Executes the search and returns |max_num_results| largest voting octs.
  absl::StatusOr<std::vector<Maximum>> Vote(
      const std::vector<Surface>& surfaces, const Oct& start_oct) {
    start_time_ = absl::Now();
    std::vector<WrappedSurface> wrapped_surfaces;
    wrapped_surfaces.reserve(surfaces.size());
    for (int i = 0; i < static_cast<int>(surfaces.size()); ++i) {
      wrapped_surfaces.push_back(WrappedSurface{
          .surface = surfaces[i], .original_surface_indices = {i}});
    }
    return Vote(wrapped_surfaces, start_oct);
  }

 private:
  struct WrappedSurface {
    Surface surface;
    std::vector<int> original_surface_indices;
  };
  struct OctAndSurfaces {
    double priority = 0.0;
    Oct oct;
    std::vector<WrappedSurface> surfaces;

    bool operator<(const OctAndSurfaces& other) const {
      return priority < other.priority;
    }
  };
  using WorkItemWriteFn = std::function<void(OctAndSurfaces)>;

  const VectorDf minimal_oct_side_length_;
  const int num_threads_;
  const int max_num_results_;
  const EssentialParameterVector epsilon_essential_;
  const FreeVector epsilon_free_;
  const int min_num_surfaces_;
  const int max_num_surfaces_;
  const float ext_factor_;
  const float max_time_sec_;
  const ScoreHypothesisFn score_hypothesis_fn_;
  const VerificationFunction verification_fn_;
  const bool enable_rounding_;

  // Executes the voting after the input is prepared.
  absl::StatusOr<std::vector<Maximum>> Vote(
      const std::vector<WrappedSurface>& start_surfaces, const Oct& start_oct);

  // Does one subdivision step of the divide-and-conquer-style search.
  std::vector<OctAndSurfaces> Subdivide(
      const Oct& oct, std::vector<WrappedSurface> surfaces,
      const std::vector<WrappedSurface>& start_surfaces);
  absl::Mutex done_mutex_;
  bool done_ = false;
  absl::Time start_time_;
  absl::Mutex max_queue_mutex_;
  BoundedPriorityQueue<Maximum, std::greater<Maximum>> max_queue_;
};

template <size_t D, typename Surface>
absl::StatusOr<std::vector<typename GeneralVoter<D, Surface>::Maximum>>
GeneralVoter<D, Surface>::Vote(
    const std::vector<WrappedSurface>& start_surfaces, const Oct& start_oct) {
  // Collect all results in a bounded size priority queue.
  max_queue_.set_max_size(max_num_results_);

  const OctAndSurfaces start_work_item{.oct = start_oct,
                                       .surfaces = start_surfaces};
  RETURN_IF_ERROR(ProcessRecursivelyDFS<OctAndSurfaces>(
      num_threads_, start_work_item,
      [&](const OctAndSurfaces& item)
          -> absl::StatusOr<std::vector<OctAndSurfaces>> {
        auto work_items = Subdivide(item.oct, item.surfaces, start_surfaces);
        {
          absl::MutexLock l(&max_queue_mutex_);
          if (max_queue_.full() && max_queue_.top().score > max_num_surfaces_) {
            // All results have a score of at least `max_num_surfaces_`, so we
            // can abort.
            absl::MutexLock l(&done_mutex_);
            done_ = true;
            return std::vector<OctAndSurfaces>();
          }
        }
        if (work_items.size() > 1) {
          // Swap max to front.
          // We could do this when creating the work items, but we pay linear
          // complexity either way. Going with this assuming swaps are
          // dominating runtime.
          auto max_it = std::max_element(work_items.begin(), work_items.end());
          using std::swap;
          std::swap(*max_it, work_items[0]);
        }
        return work_items;
      }));

  std::vector<Maximum> results{max_queue_.begin(), max_queue_.end()};
  return results;
}

template <size_t D, typename Surface>
std::vector<typename GeneralVoter<D, Surface>::OctAndSurfaces>
GeneralVoter<D, Surface>::Subdivide(
    const Oct& oct, std::vector<WrappedSurface> surfaces,
    const std::vector<WrappedSurface>& start_surfaces) {
  float score = 0;
  {
    for (const WrappedSurface& wrapped_surface : surfaces) {
      score += wrapped_surface.original_surface_indices.size();
    }
    {
      // Early aborts due to
      // * all solutions good enough `done_`,
      // * timeout,
      // * upper bound on score is lower than `min_num_surfaces_`,
      // * upper bound on score is lower than worst current result.
      absl::MutexLock l(&max_queue_mutex_);
      absl::MutexLock l2(&done_mutex_);
      if (done_ || absl::Now() - start_time_ > absl::Seconds(max_time_sec_) ||
          score < min_num_surfaces_ ||
          (!max_queue_.empty() && score <= max_queue_.top().score)) {
        return {};
      }
    }

    // Stop because we have reached the minimum oct size.
    if ((oct.side_length().array() <= minimal_oct_side_length_.array()).all()) {
      std::vector<int> surface_indices;
      for (const WrappedSurface& wrapped_surface : surfaces) {
        surface_indices.insert(surface_indices.end(),
                               wrapped_surface.original_surface_indices.begin(),
                               wrapped_surface.original_surface_indices.end());
      }

      surface_indices = verification_fn_(oct, surface_indices);
      score = surface_indices.size();
      const int num_intersections = surface_indices.size();
      absl::MutexLock l(&max_queue_mutex_);
      // if (max_queue_.empty() || score > max_queue_.top().score) {
      //   std::cout << "final score: " << surface_indices.size()
      //             << " pose: " << oct.mid().transpose()
      //             << " time: " << absl::Now() - start_time_ << std::endl;
      // }
      max_queue_.push(Maximum{.score = score,
                              .num_intersections = num_intersections,
                              .surface_indices = std::move(surface_indices),
                              .oct = oct});
      return {};
    }
  }

  std::vector<WrappedSurface> canonical_surfaces;
  if (enable_rounding_) {
    // Collapse all surfaces that round to the same parameters.

    // Since the surface is contained in the wrapped_surface, we could use a
    // hash_set instead and use the surface hash function as the wrapped surface
    // hash function.
    absl::flat_hash_map<Surface, WrappedSurface> surface_to_wrapped_surface;
    for (WrappedSurface& wrapped_surface : surfaces) {
      wrapped_surface.surface.Round(oct, epsilon_essential_, epsilon_free_);
      auto [it, inserted] = surface_to_wrapped_surface.emplace(
          wrapped_surface.surface, wrapped_surface);
      if (!inserted) {
        it->second.original_surface_indices.insert(
            it->second.original_surface_indices.end(),
            wrapped_surface.original_surface_indices.begin(),
            wrapped_surface.original_surface_indices.end());
      }
    }

    surface_to_wrapped_surface.reserve(surface_to_wrapped_surface.size());
    for (auto& [unused_surface, wrapped_surface] : surface_to_wrapped_surface) {
      canonical_surfaces.push_back(std::move(wrapped_surface));
    }
  } else {
    canonical_surfaces.swap(surfaces);
  }

  std::vector<OctAndSurfaces> work_items;
  for (HyperOctIterator<D> oct_iter(oct, minimal_oct_side_length_);
       oct_iter.valid(); oct_iter.advance()) {
    const Oct sub_oct = *oct_iter;
    const Oct extended_sub_oct =
        sub_oct.extend(minimal_oct_side_length_ * ext_factor_);
    std::vector<WrappedSurface> intersecting_surfaces;
    // This is overestimating the number of surfaces, potentially by a lot.
    intersecting_surfaces.reserve(canonical_surfaces.size());
    for (const WrappedSurface& wrapped_surface : canonical_surfaces) {
      if (wrapped_surface.surface.DoesIntersect(extended_sub_oct)) {
        // We need to copy the wrapped surface because it may intersect multiple
        // subdivisions.
        intersecting_surfaces.push_back(wrapped_surface);
      }
    }
    if (!intersecting_surfaces.empty()) {
      work_items.push_back(
          OctAndSurfaces{.priority = score_hypothesis_fn_(sub_oct, score),
                         .oct = sub_oct,
                         .surfaces = std::move(intersecting_surfaces)});
    }
  }
  return work_items;
}

}  // namespace large_scale_voting

#endif  // EXPERIMENTAL_USERS_AIGERD_GENERAL_VOTING_ICCV_21_TOWARD_OPEN_SOURCE_GENERAL_VOTER_H_
