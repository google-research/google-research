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

// A comman-line tool to run the general voter for a 6DOF posing problem.
// Reads data from input 2D-3D matches (and ground truth poses for evaluation)
// or uses random data. Used to reproduce the specific results from the paper
// (table 2, table 3). The data is from [1,2] and was used in [3] to which we
// compare our results.
//
// Our paper:
// Dror Aiger, Simon Lynen, Jan Hosang, Bernhard Zeisl:
// Efficient Large Scale Inlier Voting for Geometric Vision Problems. iccv
// (2021)
//
// [1] A. R. Zamir I. Armeni, A. Sax and S. Savarese. Joint 2d-3d semantic data
// for indoor scene understanding. In ArXiv e-prints, Feb. 2017. [Online].
// Available: http://adsabs.harvard.edu/abs/2017arXiv170201105A, 2017. [2] M.
// Salzmann S. T. Namin, M. Najafi and L. Petersson. A multimodal graphical
// model for scene analysis. In Winter Conf. Appl. Comput. Vis., page 1006–1013,
// 2015. [3] Dylan Campbell, Lars Petersson, Laurent Kneip, and Hongdong Li.
// Globally-optimal inlier set maximisation for camera pose and correspondence
// estimation. PAMI, 42(2):328–342, 2020.

#include <fstream>
#include <iostream>
#include <random>

#include "Eigen/Geometry"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "general_voter_6dof.h"

ABSL_FLAG(std::vector<float>, gt, std::vector<float>({0, 0, 0, 0, 0, 0}),
          "comma-separated list of 6 floats for GT (alpha, beta,gamma,z,x,y) "
          "for random data");
ABSL_FLAG(std::string, match_file, "", "Match file name (prefix), if exists.");
ABSL_FLAG(int, num_matches, 1000, "Number of random matches (random data).");
ABSL_FLAG(std::string, debug_file, "/tmp/pose_debug",
          "debug file (obj files to view with e.g. meshlab).");

ABSL_FLAG(int, table, -1, "Table to use (-1 for random data)");
ABSL_FLAG(int, min_intersections, 3,
          "Minimum number of intersections (surfaces-box)");
ABSL_FLAG(int, num_poses, 1, "Number of (best) poses.");
ABSL_FLAG(int, max_intersections, 1e9,
          "Maximum number of intersection (terminates if all poses have this "
          "number of intersection)");
ABSL_FLAG(double, ext_factor, 11,
          "Margin added to each box during the recursion (to avoid missing "
          "intersection close to the box).");
ABSL_FLAG(double, max_time_sec, 1e9,
          "Maximum time (terminates after this time).");
ABSL_FLAG(double, angle_min_oct, 0.001,
          "Minimum oct box for angles (radians).");
ABSL_FLAG(double, min_distance_to_landmark, 0.8,
          "Minimum distance from camera to a point.");
ABSL_FLAG(double, spatial_min_oct, 0.08,
          "Minimum oct box for translation (in input metric)");
ABSL_FLAG(double, angle_range, 1e9, "Search range for angles.");
ABSL_FLAG(double, spatial_range, 1e9, "Search range for translation.");
ABSL_FLAG(double, epsilon_essential, 0.0001,
          "Canonization accuracy: essential parameters.");
ABSL_FLAG(double, epsilon_free, 0.001,
          "Canonization accuracy: free parameters.");
ABSL_FLAG(bool, use_prior, false,
          "Use prior (Used to order the search if exists).");
ABSL_FLAG(bool, use_all_in_verification, false,
          "Use all input point for the verification function (applied to each "
          "final minimum oct box.");
ABSL_FLAG(double, max_proj_distance, 1e9,
          "Maximum projection distance in verification.");

namespace std {
bool AbslParseFlag(absl::string_view s, std::vector<float>* dst, std::string*) {
  dst->clear();
  if (s.empty()) {
    return true;
  }
  for (absl::string_view v : absl::StrSplit(s, ',', absl::AllowEmpty())) {
    dst->push_back(0.0f);
    if (!absl::SimpleAtof(v, &dst->back())) return false;
  }
  return true;
}
string AbslUnparseFlag(std::vector<float> v) { return absl::StrJoin(v, ","); }
}  // namespace std

namespace large_scale_voting {

// Draws debug data into .obj files.
void DrawCameraAndPoints(const std::string& points_name,
                         const std::string& proj_name, const Vector6f& camera,
                         const std::vector<int>& inliers,
                         const std::vector<Vector6f>& matches) {
  std::fstream f(points_name, std::ios::out);
  std::pair<Eigen::Matrix3f, Eigen::Vector3f> T = MatricesFromPose(camera);
  std::string camera_color = "1 0 0";
  // Camera shape.
  std::vector<Eigen::Vector3f> cam_shape = {Eigen::Vector3f(0, 0, 0),
                                            Eigen::Vector3f(1, 0, -0.2),
                                            Eigen::Vector3f(1, 0, 0.2)};
  for (int i = 0; i < 3; ++i) {
    const auto p = T.first * cam_shape[i] + T.second;
    f << "v " << p.transpose() << " " << camera_color << std::endl;
  }
  f << "f " << 1 << " " << 2 << " " << 3 << std::endl;
  // 3d points.
  std::string points_color = "0 0 1";
  for (const auto& m : matches) {
    const Eigen::Vector3f p = m.head<3>();
    f << "v " << p.transpose() << " " << points_color << std::endl;
  }
  f.close();
  // Projected 2d points in the camera plane.
  f.open(proj_name, std::ios::out);
  const float scale = 2.0;
  absl::flat_hash_set<int> inliers_set(inliers.begin(), inliers.end());
  for (int i = 0; i < matches.size(); ++i) {
    const auto& m = matches[i];
    if (inliers_set.contains(i)) {
      f << "v " << 0 << " " << scale * m(4) / m(5) << " " << scale * m(3) / m(5)
        << " "
        << "0 0 1" << std::endl;
      const auto p = Project(camera, m.head<3>());
      f << "v " << 0 << " " << scale * p(1) << " " << scale * p(0) << " "
        << "1 0 0" << std::endl;
    }
  }
  f.close();
}

int Run() {
  std::vector<Vector6f> matches;
  Vector6f gt_pose;
  const std::string match_file = absl::GetFlag(FLAGS_match_file);
  // Reads matches and ground truth from a file.
  if (!match_file.empty()) {
    std::fstream f(absl::StrCat(match_file, "_gt.txt"), std::ios::in);
    if (!f || f.fail() || f.eof()) {
      std::cout << "Failed to open " << absl::StrCat(match_file, "_gt.txt")
                << "\n";
      return {};
    }
    Eigen::Matrix3f gt_R;
    char s[512];
    float r1;
    float r2;
    float r3;
    for (int i = 0; i < 3; ++i) {
      f.getline(s, 512);
      sscanf(s, "%f %f %f", &r1, &r2, &r3);
      gt_R(i, 0) = r1;
      gt_R(i, 1) = r2;
      gt_R(i, 2) = r3;
    }
    // For convinient, we aleays transform the input so that the ground truth
    // pose is (0,0,0,0,0,0). This helps to eliminated coordinate system
    // conventions issues.
    Eigen::Matrix3f mr;
    mr = Eigen::AngleAxisf(kPi / 2.0, Eigen::Vector3f::UnitY());
    gt_R = mr * gt_R;
    f.getline(s, 512);
    sscanf(s, "%f %f %f", &r1, &r2, &r3);
    Eigen::Vector3f gt_translation(r1, r2, r3);
    f.close();
    // GT is always (0,0,0,0,0,0) since we transformed the input to its GT.
    gt_pose = Vector6f(0, 0, 0, 0, 0, 0);
    f.open(absl::StrCat(match_file, "_matches.txt"), std::ios::in);
    if (!f || f.fail()) {
      std::cout << "Failed to open " << absl::StrCat(match_file, "_matches.txt")
                << "\n";
      return -1;
    }
    while (!f.eof() && !f.fail()) {
      char s[512];
      f.getline(s, 512);
      if (s[0] == 0) break;
      float x;
      float y;
      float z;
      float b1;
      float b2;
      float b3;
      // Points and bearing vectors.
      const int n = sscanf(s, "%f %f %f %f %f %f", &x, &y, &z, &b1, &b2, &b3);
      if (n == 6) {
        Eigen::Vector3f location(x, y, z);
        location = gt_R * (location - gt_translation);
        Vector6f p(location.x(), location.y(), location.z(), -b1, b2, b3);
        matches.push_back(std::move(p));
      }
    }
    f.close();
  } else {
    // Random 2D-3D matches (with a given inlier fraction and gaussian noise).
    std::default_random_engine reng(1962);
    std::uniform_real_distribution<float> uniform_dist_inlier(0, 1);
    std::uniform_real_distribution<float> uniform_dist_z(-3, 3);
    std::uniform_real_distribution<float> uniform_dist_x(-3, 3);
    std::uniform_real_distribution<float> uniform_dist_y(-1, 2);
    std::uniform_real_distribution<float> uniform_dist_proj(-1, 1);
    std::normal_distribution<float> gaussian_noise{0, 0.02};
    const auto gt_vect = absl::GetFlag(FLAGS_gt);
    gt_pose = Vector6f(gt_vect[0], gt_vect[1], gt_vect[2], gt_vect[3],
                       gt_vect[4], gt_vect[5]);
    for (int i = 0; i < absl::GetFlag(FLAGS_num_matches); ++i) {
      Eigen::Vector3f point(uniform_dist_x(reng), uniform_dist_y(reng),
                            uniform_dist_z(reng));
      Eigen::Vector3f kp;
      if (uniform_dist_inlier(reng) < 0.5) {
        // Inlier.
        kp = Project(gt_pose, point) + Eigen::Vector3f(gaussian_noise(reng),
                                                       gaussian_noise(reng),
                                                       gaussian_noise(reng));
      } else {
        // Outlier.
        kp = Eigen::Vector3f(uniform_dist_proj(reng), uniform_dist_proj(reng),
                             uniform_dist_proj(reng));
        kp.normalize();
      }
      Vector6f match(point(0), point(1), point(2), kp(0) * kp(2), kp(1) * kp(2),
                     kp(2));
      matches.push_back(std::move(match));
    }
  }
  std::vector<int> inliers(matches.size());
  std::iota(inliers.begin(), inliers.end(), 0);
  DrawCameraAndPoints("/tmp/random_input.obj", "/tmp/gt_proj.obj", gt_pose,
                      inliers, matches);
  if (matches.empty()) exit(0);
  std::cout << "file: " << match_file << " num matches: " << matches.size()
            << std::endl;
  Options options;
  options.min_intersections = absl::GetFlag(FLAGS_min_intersections);
  options.max_intersections = absl::GetFlag(FLAGS_max_intersections);
  options.prior = gt_pose;
  options.use_prior = absl::GetFlag(FLAGS_use_prior);
  options.min_distance_to_landmark =
      absl::GetFlag(FLAGS_min_distance_to_landmark);
  options.num_poses = absl::GetFlag(FLAGS_num_poses);
  options.ext_factor = absl::GetFlag(FLAGS_ext_factor);
  options.max_time_sec = absl::GetFlag(FLAGS_max_time_sec);
  options.max_proj_distance = absl::GetFlag(FLAGS_max_proj_distance);
  options.min_oct(0) = absl::GetFlag(FLAGS_angle_min_oct);
  options.min_oct(1) = absl::GetFlag(FLAGS_angle_min_oct);
  options.min_oct(2) = absl::GetFlag(FLAGS_angle_min_oct);
  options.min_oct(3) = absl::GetFlag(FLAGS_spatial_min_oct);
  options.min_oct(4) = absl::GetFlag(FLAGS_spatial_min_oct);
  options.min_oct(5) = absl::GetFlag(FLAGS_spatial_min_oct);
  options.ranges(0) = absl::GetFlag(FLAGS_angle_range);
  options.ranges(1) = absl::GetFlag(FLAGS_angle_range);
  options.ranges(2) = absl::GetFlag(FLAGS_angle_range);
  options.use_all_in_verification =
      absl::GetFlag(FLAGS_use_all_in_verification);

  // NOTE: These are specific ranges (used in [3] so we fairly compare them) for
  // the tables in the paper.
  if (absl::GetFlag(FLAGS_table) == 2) {
    options.ranges(3) = std::min(25.5, absl::GetFlag(FLAGS_spatial_range));
    options.ranges(4) = std::min(2.5, absl::GetFlag(FLAGS_spatial_range));
    options.ranges(5) = std::min(2.5, absl::GetFlag(FLAGS_spatial_range));
  } else if (absl::GetFlag(FLAGS_table) == 3) {
    options.ranges(3) = absl::GetFlag(FLAGS_spatial_range);
    options.ranges(4) = 0;
    options.ranges(5) = absl::GetFlag(FLAGS_spatial_range);
  } else {
    options.ranges(3) = absl::GetFlag(FLAGS_spatial_range);
    options.ranges(4) = absl::GetFlag(FLAGS_spatial_range);
    options.ranges(5) = absl::GetFlag(FLAGS_spatial_range);
  }
  // Canonization accuracy.
  options.epsilon_essential = absl::GetFlag(FLAGS_epsilon_essential);
  options.epsilon_free = absl::GetFlag(FLAGS_epsilon_free);
  absl::Time t1 = absl::Now();
  // Compute the pose(s).
  std::vector<PoseAndInliers> results = Compute6DofPose(options, matches);
  const double overall_time = absl::ToDoubleSeconds(absl::Now() - t1);
  if (results.empty()) {
    std::cout << "no result!!!" << std::endl;
    std::cout << "\033[1;" << 31 << "m"
              << "errors: " << 100 << " " << 100 << "\033[0m" << std::endl;
    return 0;
  }
  std::cout << "found pose: " << results[0].pose.transpose() << " "
            << results[0].inlier_indices.size() << std::endl;
  const auto matrices = MatricesFromPose(results[0].pose);
  std::cout << "pose matrices:" << std::endl;
  std::cout << matrices.first << std::endl << std::endl;
  std::cout << matrices.second.transpose() << std::endl << std::endl;
  const Eigen::Vector3f gt_location(gt_pose(3), gt_pose(4), gt_pose(5));
  const Eigen::Vector3f pose_location(results[0].pose(3), results[0].pose(4),
                                      results[0].pose(5));
  std::cout << "gt location: " << gt_location.transpose() << std::endl;
  std::cout << "pose location: " << pose_location.transpose() << std::endl;
  const float pos_error = (gt_location - pose_location).norm();
  const float degrees_error_alpha = std::fabs(
      NormalizeDegrees(RadToDeg(std::fabs(gt_pose(0) - results[0].pose(0)))));
  const float degrees_error_beta = std::fabs(
      NormalizeDegrees(RadToDeg(std::fabs(gt_pose(1) - results[0].pose(1)))));
  const float degrees_error_gamma = std::fabs(
      NormalizeDegrees(RadToDeg(std::fabs(gt_pose(2) - results[0].pose(2)))));
  const float degrees_error = std::max(
      degrees_error_alpha, std::max(degrees_error_beta, degrees_error_gamma));
  const bool ok = pos_error <= 1.0 && degrees_error <= 5.729;  // 0.1rad
  const int color = ok ? 32 : 31;
  std::cout << "\033[1;" << color << "m"
            << "errors: " << pos_error << " " << degrees_error << "\033[0m"
            << " overall_time: " << overall_time << std::endl;
  DrawCameraAndPoints(absl::GetFlag(FLAGS_debug_file) + ".obj",
                      absl::GetFlag(FLAGS_debug_file) + "_proj.obj",
                      results[0].pose, results[0].inlier_indices, matches);
  return 0;
}

}  // namespace large_scale_voting

int main(int argc, char* argv[]) {
  absl::ParseCommandLine(argc, argv);
  return large_scale_voting::Run();
}
