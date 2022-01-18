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

// Computes best poses from a given set of 3d-2d matches using the generalized
// voter.
#ifndef EXPERIMENTAL_USERS_AIGERD_GENERAL_VOTING_ICCV_21_TOWARD_OPEN_SOURCE_GENERAL_VOTER_6DOF_H_
#define EXPERIMENTAL_USERS_AIGERD_GENERAL_VOTING_ICCV_21_TOWARD_OPEN_SOURCE_GENERAL_VOTER_6DOF_H_

#include <iostream>
#include <string>
#include <vector>

#include "Eigen/Core"
#include "absl/base/optimization.h"

namespace large_scale_voting {

constexpr double kPi = 3.14159265358979323846;

// 4-tuple.
using Vector4f = Eigen::Matrix<float, 4, 1>;
// 5-tuple represents a match 3d-2d.
using Vector5f = Eigen::Matrix<float, 5, 1>;
// 6-tuple represents a pose (alpha,beta,gamma,z,x,y).
using Vector6f = Eigen::Matrix<float, 6, 1>;

// Stores pose and corresponding inlier indices.
struct PoseAndInliers {
  Vector6f pose;
  std::vector<int> inlier_indices;
};

struct Options {
  Vector6f prior = Vector6f(0, 0, 0, 0, 0, 0);
  Vector6f min_oct = Vector6f(0, 0, 0, 0, 0, 0);
  Vector6f ranges = Vector6f(0, 0, 0, 0, 0, 0);
  float epsilon_essential = 0.0002;
  float epsilon_free = 0.001;
  int num_poses = 1;
  int min_intersections = 3;
  int max_intersections = 100000;
  float min_distance_to_landmark = 0.0;
  bool use_prior = false;
  float ext_factor = 0.5;
  float max_time_sec = 1e9;
  float max_proj_distance = 0.05;
  bool use_all_in_verification = false;
  void Print() const {
    std::cout << "prior: " << prior.transpose() << std::endl;
    std::cout << "min_oct: " << min_oct.transpose() << std::endl;
    std::cout << "ranges: " << ranges.transpose() << std::endl;
    std::cout << "epsilon_essential: " << epsilon_essential << std::endl;
    std::cout << "epsilon_free: " << epsilon_free << std::endl;
    std::cout << "num_poses: " << num_poses << std::endl;
    std::cout << "min_intersections: " << min_intersections << std::endl;
    std::cout << "max_intersections: " << max_intersections << std::endl;
    std::cout << "min_distance_to_landmark: " << min_distance_to_landmark
              << std::endl;
    std::cout << "use_prior: " << use_prior << std::endl;
    std::cout << "ext_factor: " << ext_factor << std::endl;
    std::cout << "max_time_sec: " << max_time_sec << std::endl;
    std::cout << "max_proj_distance: " << max_proj_distance << std::endl;
    std::cout << "use_all_in_verification: " << use_all_in_verification << std::endl;
  }
};

inline double NormalizeRadians(double angle_radians) {
  constexpr double low = -kPi;
  constexpr double high = kPi;
  constexpr double range = high - low;
  if (angle_radians < low) angle_radians += range;
  if (angle_radians >= high) angle_radians -= range;
  if (ABSL_PREDICT_TRUE(low <= angle_radians && angle_radians < high))
    return angle_radians;
  return angle_radians - range * std::floor((angle_radians - low) / range);
}

inline double NormalizeDegrees(double angle_degrees) {
  constexpr double low = -180;
  constexpr double high = 180;
  constexpr double range = high - low;
  if (angle_degrees < low) angle_degrees += range;
  if (angle_degrees >= high) angle_degrees -= range;
  if (ABSL_PREDICT_TRUE(low <= angle_degrees && angle_degrees < high))
    return angle_degrees;
  return angle_degrees - range * std::floor((angle_degrees - low) / range);
}

inline double RadToDeg(double radians) { return radians * (180.0 / kPi); }

// Computes the best k poses and their corresponding inliers (indices to the
// input).
std::vector<PoseAndInliers> Compute6DofPose(
    const Options& options, const std::vector<Vector6f>& matches);

// Projects a 3d point into the frame for a given pose.
Eigen::Vector3f Project(const Vector6f& v, const Eigen::Vector3f& point);

void DrawDebug(const std::string& prefix, const Vector6f& pose,
               const std::vector<Vector6f>& matches, float max_proj_distance);

std::vector<int> CountAccurateInliers(const Vector6f& pose,
                                      const std::vector<Vector6f>& matches,
                                      float max_proj_distance);

std::pair<Eigen::Matrix3f, Eigen::Vector3f> MatricesFromPose(
    const Vector6f& pose);

Vector6f Invert(const Vector6f& pose);

}  // namespace large_scale_voting

#endif  // EXPERIMENTAL_USERS_AIGERD_GENERAL_VOTING_ICCV_21_TOWARD_OPEN_SOURCE_GENERAL_VOTER_6DOF_H_
