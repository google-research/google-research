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

// Implements the two black boxes, surface definition and surface-box
// intersection predicate, (required for the general voter), for 6DOF posing
// problem, used in our paper:
//
// Dror Aiger, Simon Lynen, Jan Hosang, Bernhard Zeisl:
// Efficient Large Scale Inlier Voting for Geometric Vision Problems. iccv
// (2021)

#include "general_voter_6dof.h"

#include <fstream>
#include <utility>

#include "Eigen/Geometry"
#include "absl/container/flat_hash_map.h"
#include "absl/flags/flag.h"
#include "absl/time/time.h"
#include "general_voter.h"
#include "intersection_tools.h"

// Optionally, controls the implementation of the surface-box intersection. We
// use a simpler (sub-optimal, found as a good approximation) version of the
// general intersection algorithm from the paper. The algorithm finds
// intersections of the given surface with subspaces of the input box
// boundaries.
ABSL_FLAG(int, intersection_bits, 7, "");

namespace large_scale_voting {
namespace {

using Vector2f = Eigen::Matrix<float, 2, 1>;
using Surface6f = Surface<6, 4, 4, float>;

// The general voter uses a user-defined (provided via the interface)
// arbitrary verification function. This verification is applied to each final
// minimum oct box and is used as the final score for this hypothetical pose.
// Here, we use the number of inliers computed by counting the number of matches
// with at most some max projection distance (angles of bearing vectors to
// camera-point vector). Note that the general voter always uses euclidean
// distance so we can use the min oct box as an upper bound and switch to angle
// distance in the end).
std::vector<int> VerifyPose(const Vector6f& pose,
                            const std::vector<int>& inliers,
                            const std::vector<Vector6f>& matches,
                            float max_proj_distance,
                            float min_distance_from_point) {
  if (max_proj_distance < 0) return inliers;
  std::vector<int> ret;
  absl::flat_hash_set<std::pair<float, float>> keys;
  absl::flat_hash_set<std::tuple<float, float, float>> lms;
  const float sq_dist = max_proj_distance * max_proj_distance;
  const float sq_dist_from_point =
      min_distance_from_point * min_distance_from_point;
  const Vector6f pose_for_project(pose(0), pose(1), pose(2), pose(4), pose(5),
                                  pose(3));
  // Count inliers, enforcing proj distance, minimum camera-point distance and
  // 1-1 matching.
  for (auto i : inliers) {
    const auto& ep = matches[i];
    const Eigen::Vector3f loc(ep(2), ep(0), ep(1));
    if ((loc - pose.tail<3>()).squaredNorm() < sq_dist_from_point) continue;
    // Enforce 1-1 matching of 2D to 3D points.
    std::pair<float, float> key(ep(3) / ep(5), ep(4) / ep(5));
    std::tuple<float, float, float> lm(ep(0), ep(1), ep(2));
    if (keys.find(key) != keys.end() || lms.find(lm) != lms.end()) continue;
    const auto p = Project(pose_for_project, matches[i].head<3>());
    if (p(2) * ep(5) < 0) continue;
    if ((Eigen::Vector2f(ep(3) / ep(5), ep(4) / ep(5)) - p.head<2>())
            .squaredNorm() < sq_dist) {
      ret.push_back(i);
      keys.insert(key);
      lms.insert(lm);
    }
  }
  return ret;
}

// The first black box: surface definition. We swap the x,y,z coordinates to
// z,x,y so that we can have x,y as a function of the rest. Here we have a
// 4-dimensional surface in the 6-space of parameters.
Vector2f MySurface(const Vector4f& v, const Vector4f& essential_parameters) {
  const float sa = sin(v(0));
  const float ca = cos(v(0));
  const float sb = sin(v(1));
  const float cb = cos(v(1));
  const float sc = sin(v(2));
  const float cc = cos(v(2));

  const float z = v(3);
  const float w3 = essential_parameters(0);
  const float xi = essential_parameters(1) / essential_parameters(3);
  const float eta = essential_parameters(2) / essential_parameters(3);
  const float denom = (xi * ca * cb + eta * cb * sa - sb);
  const float x = ((cb * cc + ca * (xi * cc * sb - eta * sc) +
                    sa * (eta * cc * sb + xi * sc)) *
                   (z - w3)) /
                  denom;
  const float y = ((-(xi * cc * sa) + (cb + eta * sa * sb) * sc +
                    ca * (eta * cc + xi * sb * sc)) *
                   (z - w3)) /
                  denom;
  return {x, y};
}

// Implements a set of surface-boundary intersections, as part of the full
// surface-box intersection algorithm. We found that a subset of the box
// boundaries is sufficient to approximate intersection, though this is
// suboptimal theoretically. Here, A,B,C are the Euler angles (in radians) and
// X,Y,Z are the translations.

// Intersections with the X-A subspace.
std::vector<Vector2f> IntersectXA(const Vector6f& v,
                                  const Vector4f& essential_parameters,
                                  const Vector2f& free_parameters) {
  std::vector<std::pair<float, float>> x_a =
      ComputeXA(v, essential_parameters, free_parameters);
  return {Vector2f(x_a[0].first, x_a[0].second),
          Vector2f(x_a[1].first, x_a[1].second)};
}

// Intersections with the Y-B subspace.
std::vector<Vector2f> IntersectYB(const Vector6f& v,
                                  const Vector4f& essential_parameters,
                                  const Vector2f& free_parameters) {
  std::vector<std::pair<float, float>> y_b =
      ComputeYB(v, essential_parameters, free_parameters);
  return {Vector2f(y_b[0].first, y_b[0].second),
          Vector2f(y_b[1].first, y_b[1].second)};
}

// Intersections with the X-Y subspace.
Vector2f IntersectXY(const Vector6f& v, const Vector4f& essential_parameters,
                     const Vector2f& free_parameters) {
  const float sa = sin(v(0));
  const float ca = cos(v(0));
  const float sb = sin(v(1));
  const float cb = cos(v(1));
  const float sc = sin(v(2));
  const float cc = cos(v(2));

  const float z = v(3);
  const float w3 = essential_parameters(0);
  const float xi = essential_parameters(1) / essential_parameters(3);
  const float eta = essential_parameters(2) / essential_parameters(3);
  const float denom = (xi * ca * cb + eta * cb * sa - sb);
  const float x = ((cb * cc + ca * (xi * cc * sb - eta * sc) +
                    sa * (eta * cc * sb + xi * sc)) *
                   (z - w3)) /
                  denom;
  const float y = ((-(xi * cc * sa) + (cb + eta * sa * sb) * sc +
                    ca * (eta * cc + xi * sb * sc)) *
                   (z - w3)) /
                  denom;
  return {x, y};
}

// Intersections with the Z-C subspace.
std::vector<Vector2f> IntersectZC(const Vector6f& v,
                                  const Vector4f& essential_parameters,
                                  const Vector2f& free_parameters) {
  std::vector<std::pair<float, float>> z_c =
      ComputeZC(v, essential_parameters, free_parameters);
  return {Vector2f(z_c[0].first, z_c[0].second),
          Vector2f(z_c[1].first, z_c[1].second)};
}

// Intersections with the X-Z subspace.
Vector2f IntersectXZ(const Vector6f& v, const Vector4f& essential_parameters,
                     const Vector2f& free_parameters) {
  const float sa = sin(v(0));
  const float ca = cos(v(0));
  const float sb = sin(v(1));
  const float cb = cos(v(1));
  const float sc = sin(v(2));
  const float cc = cos(v(2));

  const float y = v(5);
  const float w1 = free_parameters(0);
  const float w2 = free_parameters(1);
  const float w3 = essential_parameters(0);
  const float xi = essential_parameters(1) / essential_parameters(3);
  const float eta = essential_parameters(2) / essential_parameters(3);
  const float denom = (-(xi * cc * sa) + (cb + eta * sa * sb) * sc +
                       ca * (eta * cc + xi * sb * sc));
  const float x = w1 + ((cb * cc + ca * (xi * cc * sb - eta * sc) +
                         sa * (eta * cc * sb + xi * sc)) *
                        (y - w2)) /
                           denom;
  const float z = (y * (xi * ca * cb + eta * cb * sa - sb) +
                   (-(xi * ca * cb) - eta * cb * sa + sb) * w2 +
                   (-(xi * cc * sa) + (cb + eta * sa * sb) * sc +
                    ca * (eta * cc + xi * sb * sc)) *
                       w3) /
                  denom;

  return {x, z};
}

// Intersections with the Y-Z subspace.
Vector2f IntersectYZ(const Vector6f& v, const Vector4f& essential_parameters,
                     const Vector2f& free_parameters) {
  const float sa = sin(v(0));
  const float ca = cos(v(0));
  const float sb = sin(v(1));
  const float cb = cos(v(1));
  const float sc = sin(v(2));
  const float cc = cos(v(2));

  const float x = v(4);
  const float w1 = free_parameters(0);
  const float w2 = free_parameters(1);
  const float w3 = essential_parameters(0);
  const float xi = essential_parameters(1) / essential_parameters(3);
  const float eta = essential_parameters(2) / essential_parameters(3);
  const float denom = (cb * cc + ca * (xi * cc * sb - eta * sc) +
                       sa * (eta * cc * sb + xi * sc));
  const float y = (x * (-(xi * cc * sa) + (cb + eta * sa * sb) * sc +
                        ca * (eta * cc + xi * sb * sc)) -
                   (-(xi * cc * sa) + (cb + eta * sa * sb) * sc +
                    ca * (eta * cc + xi * sb * sc)) *
                       w1 +
                   (cb * cc + ca * (xi * cc * sb - eta * sc) +
                    sa * (eta * cc * sb + xi * sc)) *
                       w2) /
                  denom;
  const float z = (x * (xi * ca * cb + eta * cb * sa - sb) +
                   (-(xi * ca * cb) - eta * cb * sa + sb) * w1 +
                   (cb * cc + xi * ca * cc * sb + eta * cc * sa * sb -
                    eta * ca * sc + xi * sa * sc) *
                       w3) /
                  denom;
  return {y, z};
}

// The full intersection predicate.
bool DoesIntersectOct(const Vector4f& essential_parameters,
                      const Vector2f& free_parameters, const HyperOct<6>& oct) {
  Vector6f v;
  const float delta = 1.0;
  // Controls the number of subspaces we use on the boundaries.
  const uint flag = absl::GetFlag(FLAGS_intersection_bits);
  if (flag & 1) {
    for (float i0 = 0; i0 <= 1; i0 += delta) {
      v(0) = oct.min(0) + i0 * (oct.max(0) - oct.min(0));
      for (float i1 = 0; i1 <= 1; i1 += delta) {
        v(1) = oct.min(1) + i1 * (oct.max(1) - oct.min(1));
        for (float i2 = 0; i2 <= 1; i2 += delta) {
          v(2) = oct.min(2) + i2 * (oct.max(2) - oct.min(2));
          for (float i5 = 0; i5 <= 1; i5 += delta) {
            v(5) = oct.min(5) + i5 * (oct.max(5) - oct.min(5));
            Vector2f x_z =
                IntersectXZ(v, essential_parameters, free_parameters);
            if (x_z(0) >= oct.min(4) && x_z(0) <= oct.max(4) &&
                x_z(1) >= oct.min(3) && x_z(1) <= oct.max(3)) {
              return true;
            }
          }
        }
      }
    }
  }
  if (flag & 2) {
    for (float i0 = 0; i0 <= 1; i0 += delta) {
      v(0) = oct.min(0) + i0 * (oct.max(0) - oct.min(0));
      for (float i1 = 0; i1 <= 1; i1 += delta) {
        v(1) = oct.min(1) + i1 * (oct.max(1) - oct.min(1));
        for (float i2 = 0; i2 <= 1; i2 += delta) {
          v(2) = oct.min(2) + i2 * (oct.max(2) - oct.min(2));
          for (float i4 = 0; i4 <= 1; i4 += delta) {
            v(4) = oct.min(4) + i4 * (oct.max(4) - oct.min(4));
            Vector2f y_z =
                IntersectYZ(v, essential_parameters, free_parameters);
            if (y_z(0) >= oct.min(5) && y_z(0) <= oct.max(5) &&
                y_z(1) >= oct.min(3) && y_z(1) <= oct.max(3)) {
              return true;
            }
          }
        }
      }
    }
  }
  if (flag & 4) {
    for (float i0 = 0; i0 <= 1; i0 += delta) {
      v(0) = oct.min(0) + i0 * (oct.max(0) - oct.min(0));
      for (float i1 = 0; i1 <= 1; i1 += delta) {
        v(1) = oct.min(1) + i1 * (oct.max(1) - oct.min(1));
        for (float i2 = 0; i2 <= 1; i2 += delta) {
          v(2) = oct.min(2) + i2 * (oct.max(2) - oct.min(2));
          for (float i3 = 0; i3 <= 1; i3 += delta) {
            v(3) = oct.min(3) + i3 * (oct.max(3) - oct.min(3));
            Vector2f x_y =
                IntersectXY(v, essential_parameters, free_parameters);
            if (x_y(0) >= oct.min(4) && x_y(0) <= oct.max(4) &&
                x_y(1) >= oct.min(5) && x_y(1) <= oct.max(5)) {
              return true;
            }
          }
        }
      }
    }
  }
  if (flag & 8) {
    for (float i4 = 0; i4 <= 1; i4 += delta) {
      v(4) = oct.min(4) + i4 * (oct.max(4) - oct.min(4));
      for (float i0 = 0; i0 <= 1; i0 += delta) {
        v(0) = oct.min(0) + i0 * (oct.max(0) - oct.min(0));
        for (float i1 = 0; i1 <= 1; i1 += delta) {
          v(1) = oct.min(1) + i1 * (oct.max(1) - oct.min(1));
          for (float i5 = 0; i5 <= 1; i5 += delta) {
            v(5) = oct.min(5) + i5 * (oct.max(5) - oct.min(5));
            const auto z_c_v =
                IntersectZC(v, essential_parameters, free_parameters);
            for (int k = 0; k < 2; ++k) {
              Vector2f z_c = z_c_v[k];
              if (z_c(0) >= oct.min(3) && z_c(0) <= oct.max(3) &&
                  z_c(1) >= oct.min(2) && z_c(1) <= oct.max(2)) {
                return true;
              }
            }
          }
        }
      }
    }
  }
  if (flag & 16) {
    for (float i5 = 0; i5 <= 1; i5 += delta) {
      v(5) = oct.min(5) + i5 * (oct.max(5) - oct.min(5));
      for (float i1 = 0; i1 <= 1; i1 += delta) {
        v(1) = oct.min(1) + i1 * (oct.max(1) - oct.min(1));
        for (float i2 = 0; i2 <= 1; i2 += delta) {
          v(2) = oct.min(2) + i2 * (oct.max(2) - oct.min(2));
          for (float i3 = 0; i3 <= 1; i3 += delta) {
            v(3) = oct.min(3) + i3 * (oct.max(3) - oct.min(3));
            const auto x_a_v =
                IntersectXA(v, essential_parameters, free_parameters);
            for (int k = 0; k < 2; ++k) {
              Vector2f x_a = x_a_v[k];
              if (x_a(0) >= oct.min(4) && x_a(0) <= oct.max(4) &&
                  x_a(1) >= oct.min(0) && x_a(1) <= oct.max(0)) {
                return true;
              }
            }
          }
        }
      }
    }
  }
  if (flag & 32) {
    for (float i4 = 0; i4 <= 1; i4 += delta) {
      v(4) = oct.min(4) + i4 * (oct.max(4) - oct.min(4));
      for (float i0 = 0; i0 <= 1; i0 += delta) {
        v(0) = oct.min(0) + i0 * (oct.max(0) - oct.min(0));
        for (float i2 = 0; i2 <= 1; i2 += delta) {
          v(2) = oct.min(2) + i2 * (oct.max(2) - oct.min(2));
          for (float i3 = 0; i3 <= 1; i3 += delta) {
            v(3) = oct.min(3) + i3 * (oct.max(3) - oct.min(3));
            const auto y_b_v =
                IntersectYB(v, essential_parameters, free_parameters);
            for (int k = 0; k < 2; ++k) {
              Vector2f y_b = y_b_v[k];
              if (y_b(0) >= oct.min(5) && y_b(0) <= oct.max(5) &&
                  y_b(1) >= oct.min(1) && y_b(1) <= oct.max(1)) {
                return true;
              }
            }
          }
        }
      }
    }
  }
  return false;
}

// Computes axis parallel bounding box with k outliers.
Eigen::AlignedBox3f ComputeRobustBoundingBox(
    const std::vector<Vector6f>& points, int k) {
  if (2 * k > points.size()) return {};
  std::vector<float> x_coords(points.size());
  std::vector<float> y_coords(points.size());
  std::vector<float> z_coords(points.size());
  for (int i = 0; i < points.size(); ++i) {
    x_coords[i] = points[i](0);
    y_coords[i] = points[i](1);
    z_coords[i] = points[i](2);
  }
  absl::c_sort(x_coords);
  absl::c_sort(y_coords);
  absl::c_sort(z_coords);
  Eigen::Vector3f min(x_coords[k], y_coords[k], z_coords[k]);
  Eigen::Vector3f max(x_coords[points.size() - 1 - k],
                      y_coords[points.size() - 1 - k],
                      z_coords[points.size() - 1 - k]);
  const int margin = 1;
  Eigen::AlignedBox3f ret(min - Eigen::Vector3f(margin, margin, margin),
                          max + Eigen::Vector3f(margin, margin, margin));
  return ret;
}

// The pose computation. Creates the parameters and data and calls the general
// voting.
std::vector<PoseAndInliers> Compute6DofPoseLimited(
    const Options& options, const std::vector<Vector6f>& matches) {
  // Normalize all points to [0,1]^3.
  Eigen::AlignedBox3f bbox = ComputeRobustBoundingBox(matches, 0);
  std::cout << "BBOX: " << bbox.min().transpose() << " "
            << bbox.max().transpose() << std::endl;
  const float max_axis = (bbox.max() - bbox.min()).maxCoeff();
  std::vector<Vector6f> norm_matches = matches;
  for (auto& p : norm_matches) {
    p(0) = (p(0) - bbox.min()(0)) / max_axis;
    p(1) = (p(1) - bbox.min()(1)) / max_axis;
    p(2) = (p(2) - bbox.min()(2)) / max_axis;
  }
  // Create a set of surfaces, one for each match.
  std::vector<Surface6f> surfaces;
  surfaces.reserve(norm_matches.size());
  for (size_t i = 0; i < norm_matches.size(); ++i) {
    auto surface = Surface6f(
        /*essential_parameters=*/Surface6f::EssentialParameterVector(
            norm_matches[i](2), norm_matches[i](3), norm_matches[i](4),
            norm_matches[i](5)),
        /*free_parameters=*/
        Surface6f::FreeVector(norm_matches[i](0), norm_matches[i](1)),
        /* surface definition */ MySurface,
        /* intersection predicate*/ DoesIntersectOct);
    surfaces.push_back(std::move(surface));
  }
  // (x,y,z) -> (z,x,y) in our internal pose representation.
  Options coptions = options;
  coptions.prior(3) = (options.prior(5) - bbox.min()(2)) / max_axis;
  coptions.prior(4) = (options.prior(3) - bbox.min()(0)) / max_axis;
  coptions.prior(5) = (options.prior(4) - bbox.min()(1)) / max_axis;
  coptions.ranges(3) = options.ranges(5) / max_axis;
  coptions.ranges(4) = options.ranges(3) / max_axis;
  coptions.ranges(5) = options.ranges(4) / max_axis;
  coptions.min_oct(3) /= max_axis;
  coptions.min_oct(4) /= max_axis;
  coptions.min_oct(5) /= max_axis;
  coptions.min_distance_to_landmark /= max_axis;
  const auto max_norm = (bbox.max() - bbox.min()) / max_axis;

  // The search space.
  const HyperOct<6> space{
      .min = Vector6f(
          std::max<float>(-kPi / 2, coptions.prior(0) - coptions.ranges(0)),
          std::max<float>(-kPi, coptions.prior(1) - coptions.ranges(1)),
          std::max<float>(-kPi / 2, coptions.prior(2) - coptions.ranges(2)),
          std::max(0.0f, coptions.prior(3) - coptions.ranges(3)),
          std::max(0.0f, coptions.prior(4) - coptions.ranges(4)),
          std::max(0.0f, coptions.prior(5) - coptions.ranges(5))),
      .max = Vector6f(
          std::min<float>(kPi / 2, coptions.prior(0) + coptions.ranges(0)),
          std::min<float>(kPi, coptions.prior(1) + coptions.ranges(1)),
          std::min<float>(kPi / 2, coptions.prior(2) + coptions.ranges(2)),
          std::min(max_norm(2), coptions.prior(3) + coptions.ranges(3)),
          std::min(max_norm(0), coptions.prior(4) + coptions.ranges(4)),
          std::min(max_norm(1), coptions.prior(5) + coptions.ranges(5)))};

  // Canonization parameters.
  const double kEpsilonEssential = options.epsilon_essential;
  const double kEpsilonFree = options.epsilon_free;
  typename Surface6f::EssentialParameterVector essential_vector(
      kEpsilonEssential, kEpsilonEssential, kEpsilonEssential,
      kEpsilonEssential);
  typename Surface6f::FreeVector free_vector(kEpsilonFree, kEpsilonFree);

  const bool enable_rounding = true;
  using VoterT = GeneralVoter<6, Surface6f>;
  using ScoreFn = VoterT::ScoreHypothesisFn;
  using Maximum = VoterT::Maximum;

  // Optionally, one can use prior to order the search. This does not change the
  // results but may effect greatly the runtime if the actual pose is close to
  // the prior. We don't use it here.
  const ScoreFn score_by_prior = [prior = coptions.prior](
                                     const HyperOct<6>& oct, float score) {
    return -(oct.mid() - prior).squaredNorm();
  };
  // Default (no-prior) order by the number of intersections.
  const ScoreFn score_by_score = [](const HyperOct<6>& oct, float score) {
    return static_cast<float>(score);
  };
  const ScoreFn scoring_function =
      options.use_prior ? score_by_prior : score_by_score;
  const double max_proj_distance = coptions.max_proj_distance;
  const double min_distance_from_point = coptions.min_distance_to_landmark;
  // Define the verification function. We can verify using the inlier set found
  // by the general voter or, using all matches. The latter is better if we miss
  // some intersections with suboptimal (but much faster) algorithm.
  const auto verification_function = [&](const HyperOct<6>& oct,
                                         const std::vector<int>& inliers) {
    if (coptions.use_all_in_verification) {
      std::vector<int> inliers(norm_matches.size());
      std::iota(inliers.begin(), inliers.end(), 0);
      return VerifyPose(oct.mid(), inliers, norm_matches, max_proj_distance,
                        min_distance_from_point);
    }
    return VerifyPose(oct.mid(), inliers, norm_matches, max_proj_distance,
                      min_distance_from_point);
  };

  // Create the general voter.
  VoterT voter(coptions.min_oct,
               /*num_threads=*/1,
               /*max_num_results=*/options.num_poses, essential_vector,
               free_vector, options.min_intersections,
               options.max_intersections, options.ext_factor,
               options.max_time_sec, scoring_function, verification_function,
               enable_rounding);
  // And call it and return a pose(s).
  const std::vector<Maximum> results = *voter.Vote(surfaces, space);
  std::vector<PoseAndInliers> ret;
  std::cout << "num results: " << results.size() << std::endl;
  // (x,x,y) -> (x,y,z) in our internal pose representation.
  for (size_t i = 0; i < results.size(); ++i) {
    PoseAndInliers pose_item;
    pose_item.pose = Vector6f(results[i].oct.mid()(0), results[i].oct.mid()(1),
                              results[i].oct.mid()(2), results[i].oct.mid()(4),
                              results[i].oct.mid()(5), results[i].oct.mid()(3));
    pose_item.pose(3) = bbox.min()(0) + pose_item.pose(3) * max_axis;
    pose_item.pose(4) = bbox.min()(1) + pose_item.pose(4) * max_axis;
    pose_item.pose(5) = bbox.min()(2) + pose_item.pose(5) * max_axis;
    pose_item.inlier_indices = results[i].surface_indices;
    ret.push_back(std::move(pose_item));
  }
  return ret;
}

}  // namespace

// Projection function used.
// NOTE: this projection is applied to final data (not normalized and swapped as
// we use in our internal representation for the voter), so we already have Here
// (x,y,z) and not (z,x,y).
Eigen::Vector3f Project(const Vector6f& v, const Eigen::Vector3f& point) {
  const float sa = sin(v(0));
  const float ca = cos(v(0));
  const float sb = sin(v(1));
  const float cb = cos(v(1));
  const float sc = sin(v(2));
  const float cc = cos(v(2));

  const float x = v(3);
  const float y = v(4);
  const float z = v(5);
  const float w1 = point(0);
  const float w2 = point(1);
  const float w3 = point(2);
  float denom = (-(x * cb * cc) + z * sb - y * cb * sc + cb * cc * w1 +
                 cb * sc * w2 - sb * w3);
  const float xi =
      (-(z * ca * cb) + y * cc * sa - x * ca * cc * sb - x * sa * sc -
       y * ca * sb * sc + (ca * cc * sb + sa * sc) * w1 +
       (-(cc * sa) + ca * sb * sc) * w2 + ca * cb * w3) /
      denom;
  const float eta =
      (y * ca * cc + z * cb * sa + x * cc * sa * sb - x * ca * sc +
       y * sa * sb * sc + (-(cc * sa * sb) + ca * sc) * w1 -
       (ca * cc + sa * sb * sc) * w2 - cb * sa * w3) /
      -denom;
  return {xi, eta, denom};
}

std::pair<Eigen::Matrix3f, Eigen::Vector3f> MatricesFromPose(
    const Vector6f& pose) {
  Eigen::Matrix3f R;
  R = Eigen::AngleAxisf(pose(2), Eigen::Vector3f::UnitZ()) *
      Eigen::AngleAxisf(pose(1), Eigen::Vector3f::UnitY()) *
      Eigen::AngleAxisf(pose(0), Eigen::Vector3f::UnitX());
  return std::make_pair(R, Eigen::Vector3f(pose(3), pose(4), pose(5)));
}

std::vector<PoseAndInliers> Compute6DofPose(
    const Options& options, const std::vector<Vector6f>& matches) {
  return Compute6DofPoseLimited(options, matches);
}

}  // namespace large_scale_voting
