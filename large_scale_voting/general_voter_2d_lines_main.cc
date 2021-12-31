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
// intersection predicate, (required for the general voter), for the toy example
// of 2d line fitting used in our paper:
//
// Dror Aiger, Simon Lynen, Jan Hosang, Bernhard Zeisl:
// Efficient Large Scale Inlier Voting for Geometric Vision Problems. iccv
// (2021)

#include <stdio.h>

#include <fstream>
#include <iostream>

#include "Eigen/Core"
#include "Eigen/Geometry"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/random/distributions.h"
#include "absl/random/random.h"
#include "absl/time/time.h"
#include "opencv2/core.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "general_voter.h"

ABSL_FLAG(int, num_points, 100000, "Number of points");
ABSL_FLAG(bool, use_canon, true, "Use canonization");
ABSL_FLAG(bool, use_ransac, false, "Use ransac");
ABSL_FLAG(double, inlier_fraction, 0.01, "Inlier fraction");

namespace general_voter_2d_line_fitting {

using Vector1f = Eigen::Matrix<float, 1, 1>;

// Surface function for a 2d line.
Vector1f LineSurface(const Vector1f& x, const Vector1f& essential_parameters) {
  return x * essential_parameters;
}

// Intersection function for a 2d line.
bool DoesLineIntersectOct(const Vector1f& essential_parameters,
                          const Vector1f& free_parameters,
                          const large_scale_voting::HyperOct<2>& oct) {
  const float y1 = oct.min.x() * essential_parameters(0) + free_parameters(0);
  const float y2 = oct.max.x() * essential_parameters(0) + free_parameters(0);
  const float y_min = std::min(y1, y2);
  const float y_max = std::max(y1, y2);
  const bool y_range_outside = y_max < oct.min.y() || y_min > oct.max.y();
  return !y_range_outside;
}

Eigen::Vector2d LineParameters(const Eigen::Vector2d& p0,
                               const Eigen::Vector2d& p1) {
  const float a = -(-p0.y() + p1.y()) / (p0.x() - p1.x());
  const float b = -(p1.x() * p0.y() - p0.x() * p1.y()) / (p0.x() - p1.x());
  return Eigen::Vector2d(a, b);
}

int CountNearest(Eigen::Vector2d& line,
                 const std::vector<Eigen::Vector2d>& points, float distance) {
  int ret = 0;
  for (const auto& p : points) {
    const float vertical_distance = std::fabs(line(0) * p(0) + line(1) - p(1));
    if (vertical_distance < distance) ret++;
  }
  return ret;
}

// A quick ransac implementation.
std::pair<int, Eigen::Vector2d> Ransac(
    const std::vector<Eigen::Vector2d>& points, float inliers_fraction,
    float distance) {
  std::pair<int, Eigen::Vector2d> ret(0, {});
  std::default_random_engine reng(1962);
  std::uniform_int_distribution<int> uniform_line(0, points.size() - 1);
  const int k =
      std::log(0.001) / std::log(1.0 - (inliers_fraction * inliers_fraction));
  for (int i = 0; i < k; ++i) {
    const auto id1 = uniform_line(reng);
    const auto id2 = uniform_line(reng);
    if (id1 != id2) {
      Eigen::Vector2d line = LineParameters(points[id1], points[id2]);
      const auto count = CountNearest(line, points, distance);
      if (count > ret.first) {
        ret.first = count;
        ret.second = line;
      }
    }
  }
  return ret;
}

int Run() {
  std::vector<Eigen::Vector2d> points(absl::GetFlag(FLAGS_num_points));
  // Create input data by sampling from a given true line (with noise) and add
  // outliers as random points. We use here only lines with slope in [-1,1] to
  // satisfy the gradient requirement of the surfaces (for canonization and
  // duality, see the paper). This however is no restriction as we can divide
  // the entire space of parameters into two and run the second half similarly
  // after an appropriate rotation, making sure we eval all posible minimum oct
  // results from both (the complexity does not change).
  std::default_random_engine reng(1962);
  std::uniform_real_distribution<float> uniform_line(-1.0, 1.0);
  std::uniform_real_distribution<float> uniform_point(0, 1.0);
  std::normal_distribution<float> gaussian_noise{0, 0.002};
  const double line_a = 1;
  const double line_b = 0.2;
  const float kInlierFraction = absl::GetFlag(FLAGS_inlier_fraction);
  std::cout << "true line: " << line_a << " " << line_b << std::endl;
  for (auto& p : points) {
    if (uniform_point(reng) < kInlierFraction) {
      const double x = uniform_point(reng);
      p = Eigen::Vector2d(x, x * line_a + line_b + gaussian_noise(reng));
    } else {
      p = Eigen::Vector2d(uniform_point(reng), uniform_point(reng));
    }
  }

  constexpr double kEpsilon = 0.002;

  // Use ransac.
  if (absl::GetFlag(FLAGS_use_ransac)) {
    absl::Time t1 = absl::Now();
    std::pair<int, Eigen::Vector2d> ransac =
        Ransac(points, kInlierFraction, kEpsilon);
    std::cout << "time: " << absl::ToDoubleSeconds(absl::Now() - t1) << " "
              << "line: " << ransac.first << " " << ransac.second.transpose()
              << " "
              << "errors: " << std::fabs(ransac.second(0) - line_a) << " "
              << std::fabs(ransac.second(1) - line_b) << std::endl;
    return 0;
  }

  // Our surface is a 1d surface embedded in the 2-space.
  using Surface2f = large_scale_voting::Surface<2, 1, 1, float>;

  constexpr double kSpaceSize = 2.0;

  absl::Time t1 = absl::Now();
  std::vector<Surface2f> surfaces;
  // Create a surface for each point, using the dual of the point as a 2d line.
  for (const auto& p : points) {
    const float a = p.x();
    const float b = -p.y();
    surfaces.push_back(Surface2f(
        /*essential_parameters=*/Surface2f::EssentialParameterVector(a),
        /*free_parameters=*/Surface2f::FreeVector(b),
        /*surface definition*/ LineSurface,
        /*surface-box intersection predicate*/ DoesLineIntersectOct));
  }

  // The parameter space.
  const large_scale_voting::HyperOct<2> space{
      .min = Eigen::Vector2f(-kSpaceSize / 2.0, -kSpaceSize / 2.0),
      .max = Eigen::Vector2f(kSpaceSize / 2.0, kSpaceSize / 2.0)};
  const Eigen::Vector2f min_oct_side_length(kEpsilon, kEpsilon);

  // Canonization parameters found empirically.
  typename Surface2f::EssentialParameterVector essential_vector(0.005);
  typename Surface2f::FreeVector free_vector(0.001);

  using VoterT = large_scale_voting::GeneralVoter<2, Surface2f>;
  using ScoreFn = VoterT::ScoreHypothesisFn;
  using Maximum = VoterT::Maximum;

  // We don't have priors so we just use the score order (of search).
  const ScoreFn score_by_score = [](const large_scale_voting::HyperOct<2>& oct,
                                    float score) {
    return static_cast<float>(score);
  };

  // No verification function required.
  const auto verification_function =
      [&](const large_scale_voting::HyperOct<2>& oct,
          const std::vector<int>& inliers) { return inliers; };

  // Define the general voter.
  VoterT voter(min_oct_side_length, 1,
               /*max_num_results=*/1, essential_vector, free_vector,
               /*minimum number of intersections*/ 3,
               /*maximum number of intersections*/ 1e9,
               /*ext_factor*/ 0.5, /*maximum time seconds*/ 1e9, score_by_score,
               verification_function, absl::GetFlag(FLAGS_use_canon));

  // Run the voter and obtain results.
  const std::vector<Maximum> results = *voter.Vote(surfaces, space);
  const Eigen::Vector2d result_line(results[0].oct.mid()(0),
                                    -results[0].oct.mid()(1));
  std::cout << "time: " << absl::ToDoubleSeconds(absl::Now() - t1) << " "
            << "line: " << results[0].num_intersections << " "
            << result_line.transpose() << " "
            << "errors: " << std::fabs(result_line(0) - line_a) << " "
            << std::fabs(result_line(1) - line_b) << std::endl;

  // Draw debug image.
  cv::Mat image = cv::Mat::ones(cv::Size(500, 500), CV_8UC3);
  for (const auto& p : points) {
    const cv::Point2d p2_draw(p(0), p(1));
    cv::circle(image, p2_draw * 500, 1, cv::Scalar(255, 255, 255));
  }
  cv::imwrite("/tmp/points.png", image);
  for (const auto& i : results[0].surface_indices) {
    const Eigen::Vector2d p2 = points[i];
    const cv::Point2d p2_draw(p2(0), p2(1));
    cv::circle(image, p2_draw * 500, 3, cv::Scalar(0, 255, 0), 2);
  }
  cv::imwrite("/tmp/points_line.png", image);
  return 0;
}
}  // namespace general_voter_2d_line_fitting

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  return general_voter_2d_line_fitting::Run();
}
