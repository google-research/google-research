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

// Implementation of complex (computed analytically with wolfram Mathematica)
// intersection functions that determine the intersection of a given surface
// with subspaces of the boundaries of a given box in 6-space for our example
// (6DOF posing). One can possibly create faster functions using a numerical
// equation system solver. These functions are a subset of what is required by
// the general surface-box intersection algorithm (see the paper). We did not
// implement all possible intersections so this is suboptimal theoretically
// (but very close to in practice). These functions are not general and only
// valid for the example problem.

#ifndef EXPERIMENTAL_USERS_AIGERD_GENERAL_VOTING_ICCV_21_TOWARD_OPEN_SOURCE_INTERSECTION_TOOLS_H_
#define EXPERIMENTAL_USERS_AIGERD_GENERAL_VOTING_ICCV_21_TOWARD_OPEN_SOURCE_INTERSECTION_TOOLS_H_

#include <cmath>
#include <iostream>
#include <vector>

#include "Eigen/Core"

using Vector4f = Eigen::Matrix<float, 4, 1>;
using Vector2f = Eigen::Matrix<float, 2, 1>;
using Vector6f = Eigen::Matrix<float, 6, 1>;

inline float Power(float x, int p) {
  float ret = x;
  for (int i = 1; i < p; ++i) {
    ret *= x;
  }
  return ret;
}

inline float Sqrt(float x) { return std::sqrt(std::fabs(x)); }
inline float ArcTan(float x, float y) { return std::atan2(y, x); }

std::vector<std::pair<float, float>> ComputeXA(
    const Vector6f& v, const Vector4f& essential_parameters,
    const Vector2f& free_parameters) {
  const float b = v(1);
  const float c = v(2);
  const float y = v(5);
  const float z = v(3);
  const float w1 = free_parameters(0);
  const float w2 = free_parameters(1);
  const float w3 = essential_parameters(0);
  const float xi = essential_parameters(1) / essential_parameters(3);
  const float eta = essential_parameters(2) / essential_parameters(3);
  const float sb = sin(b);
  const float cb = cos(b);
  const float sc = sin(c);
  const float cc = cos(c);
  const float eta2 = Power(eta, 2);
  const float eta3 = Power(eta, 3);
  const float xi2 = Power(xi, 2);
  const float xi3 = Power(xi, 3);
  const float c2b = cos(2 * b);
  const float c2c = cos(2 * c);
  const float s2b = sin(2 * b);
  const float s2c = sin(2 * c);
  const float y2 = Power(y, 2);
  const float z2 = Power(z, 2);
  const float w2_2 = Power(w2, 2);
  const float w3_2 = Power(w3, 2);

  const float x1 =
      ((-6 + 2 * eta2 + 2 * xi2 + 2 * (1 + eta2 + xi2) * c2b +
        (1 + eta2 + xi2) * cos(2 * (b - c)) + 2 * c2c + 2 * eta2 * c2c +
        2 * xi2 * c2c + cos(2 * (b + c)) + eta2 * cos(2 * (b + c)) +
        xi2 * cos(2 * (b + c))) *
           w1 -
       (2 *
        (-4 * z2 * xi * Power(cb, 3) * Power(cc, 2) * sb -
         4 * z2 * eta2 * xi * Power(cb, 3) * Power(cc, 2) * sb -
         4 * z2 * xi3 * Power(cb, 3) * Power(cc, 2) * sb -
         4 * y * z * eta * Power(cb, 2) * Power(cc, 3) * sb -
         4 * y * z * eta3 * Power(cb, 2) * Power(cc, 3) * sb -
         4 * y * z * eta * xi2 * Power(cb, 2) * Power(cc, 3) * sb -
         4 * z2 * xi * cb * Power(cc, 2) * Power(sb, 3) -
         4 * z2 * eta2 * xi * cb * Power(cc, 2) * Power(sb, 3) -
         4 * z2 * xi3 * cb * Power(cc, 2) * Power(sb, 3) +
         4 * y * z * xi * Power(cb, 2) * Power(cc, 2) * sc +
         4 * y * z * eta2 * xi * Power(cb, 4) * Power(cc, 2) * sc +
         4 * y * z * xi3 * Power(cb, 4) * Power(cc, 2) * sc +
         4 * y2 * eta * cb * Power(cc, 3) * sc +
         4 * y2 * eta3 * Power(cb, 3) * Power(cc, 3) * sc +
         4 * y2 * eta * xi2 * Power(cb, 3) * Power(cc, 3) * sc +
         4 * y * z * xi * Power(cc, 2) * Power(sb, 2) * sc -
         4 * y * z * xi * Power(cb, 2) * Power(cc, 2) * Power(sb, 2) * sc +
         4 * y * z * eta2 * xi * Power(cb, 2) * Power(cc, 2) * Power(sb, 2) *
             sc +
         4 * y * z * xi3 * Power(cb, 2) * Power(cc, 2) * Power(sb, 2) * sc -
         4 * y2 * eta * cb * Power(cc, 3) * Power(sb, 2) * sc -
         4 * y * z * xi * Power(cc, 2) * Power(sb, 4) * sc -
         4 * y * z * eta * cc * sb * Power(sc, 2) -
         4 * y * z * eta * Power(cb, 2) * cc * sb * Power(sc, 2) -
         8 * y * z * eta3 * Power(cb, 2) * cc * sb * Power(sc, 2) -
         8 * y * z * eta * xi2 * Power(cb, 2) * cc * sb * Power(sc, 2) +
         4 * y * z * eta * cc * Power(sb, 3) * Power(sc, 2) +
         4 * y2 * eta * cb * cc * Power(sc, 3) +
         4 * y2 * eta3 * Power(cb, 3) * cc * Power(sc, 3) +
         4 * y2 * eta * xi2 * Power(cb, 3) * cc * Power(sc, 3) -
         4 * y2 * eta * cb * cc * Power(sb, 2) * Power(sc, 3) +
         z2 * eta * sb * s2b * s2c + z2 * eta3 * sb * s2b * s2c +
         z2 * eta * xi2 * sb * s2b * s2c +
         2 * eta * (1 + eta2 + xi2) * Power(cb, 3) * s2c * w2_2 +
         2 * (1 + eta2 + xi2) * cb * cc *
             (4 * z * xi * cc * sb + 2 * z * eta * Power(cb, 2) * sc +
              z * eta * (-3 + c2b) * sc -
              y * cb * (eta * (-3 + c2c) * sb + xi * s2c)) *
             w3 +
         2 * (1 + eta2 + xi2) * cc * s2b * (-(xi * cc) + eta * sb * sc) * w3_2 -
         2 * (1 + eta2 + xi2) * Power(cb, 2) * cc * w2 *
             (z * eta * (-3 + c2c) * sb +
              2 * (2 * y * eta * cb + z * xi * cc) * sc -
              (eta * (-3 + c2c) * sb + xi * s2c) * w3) +
         Sqrt(2) *
             Sqrt(
                 Power(y * eta * cb +
                           z * xi * cc - z * eta * sb * sc - eta * cb * w2 +
                           (-(xi * cc) + eta * sb * sc) * w3,
                       2) *
                 (-4 * y2 - 2 * z2 + 4 * y2 * eta2 + 6 * z2 * eta2 +
                  4 * y2 * xi2 + 6 * z2 * xi2 + 4 * y2 * c2b - 2 * z2 * c2b +
                  4 * y2 * eta2 * c2b - 2 * z2 * eta2 * c2b +
                  4 * y2 * xi2 * c2b - 2 * z2 * xi2 * c2b +
                  z2 *
                      cos(2 * (b - c)) +
                  z2 *
                      eta2 *
                      cos(2 * (b - c)) +
                  z2 *
                      xi2 *
                      cos(2 * (b - c)) -
                  4 * y *
                      z *
                      cos(2 * b - c) -
                  4 * y *
                      z *
                      eta2 *
                      cos(2 * b - c) -
                  4 * y *
                      z *
                      xi2 *
                      cos(2 * b - c) +
                  2 * z2 *
                      c2c +
                  2 * z2 *
                      eta2 *
                      c2c +
                  2 * z2 *
                      xi2 *
                      c2c +
                  z2 *
                      cos(2 * (b + c)) +
                  z2 *
                      eta2 *
                      cos(2 * (b + c)) +
                  z2 *
                      xi2 *
                      cos(2 * (b + c)) +
                  4 * y *
                      z *
                      cos(2 * b + c) +
                  4 * y *
                      z *
                      eta2 *
                      cos(2 * b + c) +
                  4 * y *
                      z *
                      xi2 *
                      cos(2 * b + c) +
                  4 * (-1 + eta2 + xi2 + (1 + eta2 + xi2) * c2b) * w2_2 -
                  2 * (-2 * z + 6 * z * eta2 + 6 * z * xi2 - 2 * z * (1 + eta2 + xi2) * c2b + z * (1 + eta2 + xi2) * cos(2 * (b - c)) - 2 * y * cos(2 * b - c) - 2 * y * eta2 * cos(2 * b - c) - 2 * y * xi2 * cos(2 * b - c) + 2 * z * c2c + 2 * z * eta2 * c2c + 2 * z * xi2 * c2c + z * cos(2 * (b + c)) + z * eta2 * cos(2 * (b + c)) + z * xi2 * cos(2 * (b + c)) + 2 * y * cos(2 * b + c) + 2 * y * eta2 * cos(2 * b + c) + 2 * y * xi2 * cos(2 * b + c)) *
                      w3 +
                  (-2 + 6 * eta2 + 6 * xi2 - 2 * (1 + eta2 + xi2) * c2b +
                   (1 + eta2 + xi2) * cos(2 * (b - c)) + 2 * c2c +
                   2 * eta2 * c2c + 2 * xi2 * c2c + cos(2 * (b + c)) +
                   eta2 * cos(2 * (b + c)) + xi2 * cos(2 * (b + c))) *
                      w3_2 -
                  4 * w2 *
                      (-3 * y + y * eta2 + y * xi2 +
                       2 * y * (1 + eta2 + xi2) * Power(cb, 2) + y * c2b +
                       y * eta2 * c2b + y * xi2 * c2b - 2 * z * s2b * sc -
                       2 * z * eta2 * s2b * sc - 2 * z * xi2 * s2b * sc +
                       2 * (1 + eta2 + xi2) * s2b * sc * w3))))) /
           (y * eta * cb + z * xi * cc - z * eta * sb * sc - eta * cb * w2 +
            (-(xi * cc) + eta * sb * sc) * w3)) /
      (8. * ((eta2 + xi2) * Power(cb, 2) * Power(cc, 2) -
             Power(cc, 2) * Power(sb, 2) - Power(sc, 2)));
  const float a1 = ArcTan(
      (4 * xi * s2b * w2_2 +
       8 * w2 *
           (z * eta * cc * sb - xi * (y * s2b + z * c2b * sc) +
            (-(eta * cc * sb) + xi * c2b * sc) * w3) -
       4 * (-2 * y2 * xi * cb * sb + z2 * xi * cb * sb +
            2 * y * z * eta * cc * sb - z2 * xi * cb * Power(cc, 2) * sb -
            2 * y * z * xi * Power(cb, 2) * sc +
            2 * y * z * xi * Power(sb, 2) * sc +
            z2 * xi * cb * sb * Power(sc, 2) + z2 * eta * cb * s2c -
            2 * z * xi * cb * sb * w3 - 2 * y * eta * cc * sb * w3 +
            z * xi * Power(cc, 2) * s2b * w3 +
            2 * y * xi * Power(cb, 2) * sc * w3 -
            4 * z * eta * cb * cc * sc * w3 -
            2 * y * xi * Power(sb, 2) * sc * w3 -
            2 * z * xi * cb * sb * Power(sc, 2) * w3 + xi * cb * sb * w3_2 -
            xi * cb * Power(cc, 2) * sb * w3_2 +
            xi * cb * sb * Power(sc, 2) * w3_2 + eta * cb * s2c * w3_2 +
            std::sqrt(
                Power(y * eta * cb + z * xi * cc - z * eta * sb * sc -
                          eta * cb * w2 + (-(xi * cc) + eta * sb * sc) * w3,
                      2) *
                (-2 * y2 - z2 + 2 * y2 * eta2 + 3 * z2 * eta2 + 2 * y2 * xi2 +
                 3 * z2 * xi2 + 2 * y2 * c2b - z2 * c2b + 2 * y2 * eta2 * c2b -
                 z2 * eta2 * c2b + 2 * y2 * xi2 * c2b - z2 * xi2 * c2b +
                 z2 * c2c + z2 * eta2 * c2c + z2 * xi2 * c2c + z2 * c2b * c2c +
                 z2 * eta2 * c2b * c2c + z2 * xi2 * c2b * c2c -
                 4 * y * z * s2b * sc - 4 * y * z * eta2 * s2b * sc -
                 4 * y * z * xi2 * s2b * sc +
                 2 * (-1 + eta2 + xi2 + (1 + eta2 + xi2) * c2b) * w2_2 -
                 2 * (z * (1 + eta2 + xi2) * c2c - 2 * y * (1 + eta2 + xi2) * s2b * sc - z * (1 - 3 * eta2 - 3 * xi2 + 2 * (1 + eta2 + xi2) * c2b * Power(sc, 2))) *
                     w3 +
                 (-1 + 3 * eta2 + 3 * xi2 + (1 + eta2 + xi2) * c2c -
                  2 * (1 + eta2 + xi2) * c2b * Power(sc, 2)) *
                     w3_2 -
                 2 * w2 *
                     (-3 * y + y * eta2 + y * xi2 +
                      2 * y * (1 + eta2 + xi2) * Power(cb, 2) + y * c2b +
                      y * eta2 * c2b + y * xi2 * c2b - 2 * z * s2b * sc -
                      2 * z * eta2 * s2b * sc - 2 * z * xi2 * s2b * sc +
                      2 * (1 + eta2 + xi2) * s2b * sc * w3))))) /
          ((eta2 + xi2) *
           (4 * y2 + 6 * z2 + 4 * y2 * c2b - 2 * z2 * c2b +
            z2 * cos(2 * b - 2 * c) - 4 * y * z * cos(2 * b - c) +
            2 * z2 * c2c + z2 * cos(2 * (b + c)) + 4 * y * z * cos(2 * b + c) +
            8 * Power(cb, 2) * w2_2 -
            2 * (6 * z - 2 * z * c2b + z * cos(2 * b - 2 * c) - 2 * y * cos(2 * b - c) + 2 * z * c2c + z * cos(2 * (b + c)) + 2 * y * cos(2 * b + c)) *
                w3 +
            (6 - 2 * c2b + cos(2 * b - 2 * c) + 2 * c2c + cos(2 * (b + c))) *
                w3_2 -
            16 * cb * w2 * (y * cb - z * sb * sc + sb * sc * w3))),
      (
          4 * Power(y, 3) * eta2 * sb + 2 * y * z2 * eta2 * sb +
          8 * y * z2 * xi2 * sb + 12 * Power(y, 3) * eta2 * Power(cb, 2) * sb -
          18 * y * z2 * eta2 * Power(cb, 2) * sb +
          32 * y2 * z * eta * xi * cb * cc * sb -
          8 * Power(z, 3) * eta * xi * cb * cc * sb -
          2 * y * z2 * eta2 * Power(cc, 2) * sb +
          8 * y * z2 * xi2 * Power(cc, 2) * sb +
          18 * y * z2 * eta2 * Power(cb, 2) * Power(cc, 2) * sb +
          8 * Power(z, 3) * eta * xi * cb * Power(cc, 3) * sb -
          4 * Power(y, 3) * eta2 * Power(sb, 3) +
          6 * y * z2 * eta2 * Power(sb, 3) -
          6 * y * z2 * eta2 * Power(cc, 2) * Power(sb, 3) +
          4 * y2 * z * eta2 * cb * sc + 3 * Power(z, 3) * eta2 * cb * sc +
          4 * Power(z, 3) * xi2 * cb * sc +
          12 * y2 * z * eta2 * Power(cb, 3) * sc -
          3 * Power(z, 3) * eta2 * Power(cb, 3) * sc +
          32 * y * z2 * eta * xi * Power(cb, 2) * cc * sc -
          3 * Power(z, 3) * eta2 * cb * Power(cc, 2) * sc +
          12 * Power(z, 3) * xi2 * cb * Power(cc, 2) * sc +
          3 * Power(z, 3) * eta2 * Power(cb, 3) * Power(cc, 2) * sc -
          36 * y2 * z * eta2 * cb * Power(sb, 2) * sc +
          9 * Power(z, 3) * eta2 * cb * Power(sb, 2) * sc -
          32 * y * z2 * eta * xi * cc * Power(sb, 2) * sc -
          9 * Power(z, 3) * eta2 * cb * Power(cc, 2) * Power(sb, 2) * sc +
          2 * y * z2 * eta2 * sb * Power(sc, 2) -
          8 * y * z2 * xi2 * sb * Power(sc, 2) -
          18 * y * z2 * eta2 * Power(cb, 2) * sb * Power(sc, 2) +
          6 * y * z2 * eta2 * Power(sb, 3) * Power(sc, 2) +
          Power(z, 3) * eta2 * cb * Power(sc, 3) -
          4 * Power(z, 3) * xi2 * cb * Power(sc, 3) -
          Power(z, 3) * eta2 * Power(cb, 3) * Power(sc, 3) +
          3 * Power(z, 3) * eta2 * cb * Power(sb, 2) * Power(sc, 3) -
          6 * Power(z, 3) * eta * xi * s2b * sc * s2c -
          12 * y2 * eta2 * sb * w2 - 2 * z2 * eta2 * sb * w2 -
          8 * z2 * xi2 * sb * w2 - 36 * y2 * eta2 * Power(cb, 2) * sb * w2 +
          18 * z2 * eta2 * Power(cb, 2) * sb * w2 -
          64 * y * z * eta * xi * cb * cc * sb * w2 +
          2 * z2 * eta2 * Power(cc, 2) * sb * w2 -
          8 * z2 * xi2 * Power(cc, 2) * sb * w2 -
          18 * z2 * eta2 * Power(cb, 2) * Power(cc, 2) * sb * w2 +
          12 * y2 * eta2 * Power(sb, 3) * w2 -
          6 * z2 * eta2 * Power(sb, 3) * w2 +
          6 * z2 * eta2 * Power(cc, 2) * Power(sb, 3) * w2 -
          8 * y * z * eta2 * cb * sc * w2 -
          24 * y * z * eta2 * Power(cb, 3) * sc * w2 -
          32 * z2 * eta * xi * Power(cb, 2) * cc * sc * w2 +
          72 * y * z * eta2 * cb * Power(sb, 2) * sc * w2 +
          32 * z2 * eta * xi * cc * Power(sb, 2) * sc * w2 -
          2 * z2 * eta2 * sb * Power(sc, 2) * w2 +
          8 * z2 * xi2 * sb * Power(sc, 2) * w2 +
          18 * z2 * eta2 * Power(cb, 2) * sb * Power(sc, 2) * w2 -
          6 * z2 * eta2 * Power(sb, 3) * Power(sc, 2) * w2 +
          12 * y * eta2 * sb * w2_2 + 36 * y * eta2 * Power(cb, 2) * sb * w2_2 +
          32 * z * eta * xi * cb * cc * sb * w2_2 -
          12 * y * eta2 * Power(sb, 3) * w2_2 + 4 * z * eta2 * cb * sc * w2_2 +
          12 * z * eta2 * Power(cb, 3) * sc * w2_2 -
          36 * z * eta2 * cb * Power(sb, 2) * sc * w2_2 -
          4 * eta2 * sb * Power(w2, 3) -
          12 * eta2 * Power(cb, 2) * sb * Power(w2, 3) +
          4 * eta2 * Power(sb, 3) * Power(w2, 3) - 4 * y * z * eta2 * sb * w3 -
          16 * y * z * xi2 * sb * w3 +
          36 * y * z * eta2 * Power(cb, 2) * sb * w3 -
          32 * y2 * eta * xi * cb * cc * sb * w3 +
          24 * z2 * eta * xi * cb * cc * sb * w3 +
          4 * y * z * eta2 * Power(cc, 2) * sb * w3 -
          16 * y * z * xi2 * Power(cc, 2) * sb * w3 -
          36 * y * z * eta2 * Power(cb, 2) * Power(cc, 2) * sb * w3 -
          24 * z2 * eta * xi * cb * Power(cc, 3) * sb * w3 -
          12 * y * z * eta2 * Power(sb, 3) * w3 +
          12 * y * z * eta2 * Power(cc, 2) * Power(sb, 3) * w3 -
          4 * y2 * eta2 * cb * sc * w3 - 9 * z2 * eta2 * cb * sc * w3 -
          12 * z2 * xi2 * cb * sc * w3 -
          12 * y2 * eta2 * Power(cb, 3) * sc * w3 +
          9 * z2 * eta2 * Power(cb, 3) * sc * w3 -
          64 * y * z * eta * xi * Power(cb, 2) * cc * sc * w3 +
          9 * z2 * eta2 * cb * Power(cc, 2) * sc * w3 -
          36 * z2 * xi2 * cb * Power(cc, 2) * sc * w3 -
          9 * z2 * eta2 * Power(cb, 3) * Power(cc, 2) * sc * w3 +
          36 * y2 * eta2 * cb * Power(sb, 2) * sc * w3 -
          27 * z2 * eta2 * cb * Power(sb, 2) * sc * w3 +
          64 * y * z * eta * xi * cc * Power(sb, 2) * sc * w3 +
          27 * z2 * eta2 * cb * Power(cc, 2) * Power(sb, 2) * sc * w3 -
          4 * y * z * eta2 * sb * Power(sc, 2) * w3 +
          16 * y * z * xi2 * sb * Power(sc, 2) * w3 +
          36 * y * z * eta2 * Power(cb, 2) * sb * Power(sc, 2) * w3 +
          72 * z2 * eta * xi * cb * cc * sb * Power(sc, 2) * w3 -
          12 * y * z * eta2 * Power(sb, 3) * Power(sc, 2) * w3 -
          3 * z2 * eta2 * cb * Power(sc, 3) * w3 +
          12 * z2 * xi2 * cb * Power(sc, 3) * w3 +
          3 * z2 * eta2 * Power(cb, 3) * Power(sc, 3) * w3 -
          9 * z2 * eta2 * cb * Power(sb, 2) * Power(sc, 3) * w3 +
          4 * z * eta2 * sb * w2 * w3 + 16 * z * xi2 * sb * w2 * w3 -
          36 * z * eta2 * Power(cb, 2) * sb * w2 * w3 +
          64 * y * eta * xi * cb * cc * sb * w2 * w3 -
          4 * z * eta2 * Power(cc, 2) * sb * w2 * w3 +
          16 * z * xi2 * Power(cc, 2) * sb * w2 * w3 +
          36 * z * eta2 * Power(cb, 2) * Power(cc, 2) * sb * w2 * w3 +
          12 * z * eta2 * Power(sb, 3) * w2 * w3 -
          12 * z * eta2 * Power(cc, 2) * Power(sb, 3) * w2 * w3 +
          8 * y * eta2 * cb * sc * w2 * w3 +
          24 * y * eta2 * Power(cb, 3) * sc * w2 * w3 +
          64 * z * eta * xi * Power(cb, 2) * cc * sc * w2 * w3 -
          72 * y * eta2 * cb * Power(sb, 2) * sc * w2 * w3 -
          64 * z * eta * xi * cc * Power(sb, 2) * sc * w2 * w3 +
          4 * z * eta2 * sb * Power(sc, 2) * w2 * w3 -
          16 * z * xi2 * sb * Power(sc, 2) * w2 * w3 -
          36 * z * eta2 * Power(cb, 2) * sb * Power(sc, 2) * w2 * w3 +
          12 * z * eta2 * Power(sb, 3) * Power(sc, 2) * w2 * w3 -
          32 * eta * xi * cb * cc * sb * w2_2 * w3 -
          4 * eta2 * cb * sc * w2_2 * w3 -
          12 * eta2 * Power(cb, 3) * sc * w2_2 * w3 +
          36 * eta2 * cb * Power(sb, 2) * sc * w2_2 * w3 +
          2 * y * eta2 * sb * w3_2 + 8 * y * xi2 * sb * w3_2 -
          18 * y * eta2 * Power(cb, 2) * sb * w3_2 -
          24 * z * eta * xi * cb * cc * sb * w3_2 -
          2 * y * eta2 * Power(cc, 2) * sb * w3_2 +
          8 * y * xi2 * Power(cc, 2) * sb * w3_2 +
          18 * y * eta2 * Power(cb, 2) * Power(cc, 2) * sb * w3_2 +
          24 * z * eta * xi * cb * Power(cc, 3) * sb * w3_2 +
          6 * y * eta2 * Power(sb, 3) * w3_2 -
          6 * y * eta2 * Power(cc, 2) * Power(sb, 3) * w3_2 +
          9 * z * eta2 * cb * sc * w3_2 + 12 * z * xi2 * cb * sc * w3_2 -
          9 * z * eta2 * Power(cb, 3) * sc * w3_2 +
          32 * y * eta * xi * Power(cb, 2) * cc * sc * w3_2 -
          9 * z * eta2 * cb * Power(cc, 2) * sc * w3_2 +
          36 * z * xi2 * cb * Power(cc, 2) * sc * w3_2 +
          9 * z * eta2 * Power(cb, 3) * Power(cc, 2) * sc * w3_2 +
          27 * z * eta2 * cb * Power(sb, 2) * sc * w3_2 -
          32 * y * eta * xi * cc * Power(sb, 2) * sc * w3_2 -
          27 * z * eta2 * cb * Power(cc, 2) * Power(sb, 2) * sc * w3_2 +
          2 * y * eta2 * sb * Power(sc, 2) * w3_2 -
          8 * y * xi2 * sb * Power(sc, 2) * w3_2 -
          18 * y * eta2 * Power(cb, 2) * sb * Power(sc, 2) * w3_2 -
          72 * z * eta * xi * cb * cc * sb * Power(sc, 2) * w3_2 +
          6 * y * eta2 * Power(sb, 3) * Power(sc, 2) * w3_2 +
          3 * z * eta2 * cb * Power(sc, 3) * w3_2 -
          12 * z * xi2 * cb * Power(sc, 3) * w3_2 -
          3 * z * eta2 * Power(cb, 3) * Power(sc, 3) * w3_2 +
          9 * z * eta2 * cb * Power(sb, 2) * Power(sc, 3) * w3_2 -
          2 * eta2 * sb * w2 * w3_2 - 8 * xi2 * sb * w2 * w3_2 +
          18 * eta2 * Power(cb, 2) * sb * w2 * w3_2 +
          2 * eta2 * Power(cc, 2) * sb * w2 * w3_2 -
          8 * xi2 * Power(cc, 2) * sb * w2 * w3_2 -
          18 * eta2 * Power(cb, 2) * Power(cc, 2) * sb * w2 * w3_2 -
          6 * eta2 * Power(sb, 3) * w2 * w3_2 +
          6 * eta2 * Power(cc, 2) * Power(sb, 3) * w2 * w3_2 -
          32 * eta * xi * Power(cb, 2) * cc * sc * w2 * w3_2 +
          32 * eta * xi * cc * Power(sb, 2) * sc * w2 * w3_2 -
          2 * eta2 * sb * Power(sc, 2) * w2 * w3_2 +
          8 * xi2 * sb * Power(sc, 2) * w2 * w3_2 +
          18 * eta2 * Power(cb, 2) * sb * Power(sc, 2) * w2 * w3_2 -
          6 * eta2 * Power(sb, 3) * Power(sc, 2) * w2 * w3_2 +
          8 * eta * xi * cb * cc * sb * Power(w3, 3) -
          8 * eta * xi * cb * Power(cc, 3) * sb * Power(w3, 3) -
          3 * eta2 * cb * sc * Power(w3, 3) - 4 * xi2 * cb * sc * Power(w3, 3) +
          3 * eta2 * Power(cb, 3) * sc * Power(w3, 3) +
          3 * eta2 * cb * Power(cc, 2) * sc * Power(w3, 3) -
          12 * xi2 * cb * Power(cc, 2) * sc * Power(w3, 3) -
          3 * eta2 * Power(cb, 3) * Power(cc, 2) * sc * Power(w3, 3) -
          9 * eta2 * cb * Power(sb, 2) * sc * Power(w3, 3) +
          9 * eta2 * cb * Power(cc, 2) * Power(sb, 2) * sc * Power(w3, 3) -
          eta2 * cb * Power(sc, 3) * Power(w3, 3) +
          4 * xi2 * cb * Power(sc, 3) * Power(w3, 3) +
          eta2 * Power(cb, 3) * Power(sc, 3) * Power(w3, 3) -
          3 * eta2 * cb * Power(sb, 2) * Power(sc, 3) * Power(w3, 3) +
          6 * eta * xi * s2b * sc * s2c * Power(w3, 3) +
          4 * Sqrt(2) * y * xi * cb *
              std::sqrt(
                  Power(y * eta * cb + z * xi * cc - z * eta * sb * sc -
                            eta * cb * w2 + (-(xi * cc) + eta * sb * sc) * w3,
                        2) *
                  (-4 * y2 - 2 * z2 + 4 * y2 * eta2 + 6 * z2 * eta2 +
                   4 * y2 * xi2 + 6 * z2 * xi2 + 4 * y2 * c2b - 2 * z2 * c2b +
                   4 * y2 * eta2 * c2b - 2 * z2 * eta2 * c2b +
                   4 * y2 * xi2 * c2b - 2 * z2 * xi2 * c2b +
                   z2 * cos(2 * b - 2 * c) + z2 * eta2 * cos(2 * b - 2 * c) +
                   z2 * xi2 * cos(2 * b - 2 * c) - 4 * y * z * cos(2 * b - c) -
                   4 * y * z * eta2 * cos(2 * b - c) -
                   4 * y * z * xi2 * cos(2 * b - c) + 2 * z2 * c2c +
                   2 * z2 * eta2 * c2c + 2 * z2 * xi2 * c2c +
                   z2 * cos(2 * (b + c)) + z2 * eta2 * cos(2 * (b + c)) +
                   z2 * xi2 * cos(2 * (b + c)) + 4 * y * z * cos(2 * b + c) +
                   4 * y * z * eta2 * cos(2 * b + c) +
                   4 * y * z * xi2 * cos(2 * b + c) +
                   4 * (-1 + eta2 + xi2 + (1 + eta2 + xi2) * c2b) * w2_2 -
                   2 * (-2 * z + 6 * z * eta2 + 6 * z * xi2 - 2 * z * (1 + eta2 + xi2) * c2b + z * (1 + eta2 + xi2) * cos(2 * b - 2 * c) - 2 * y * cos(2 * b - c) - 2 * y * eta2 * cos(2 * b - c) - 2 * y * xi2 * cos(2 * b - c) + 2 * z * c2c + 2 * z * eta2 * c2c + 2 * z * xi2 * c2c + z * cos(2 * (b + c)) + z * eta2 * cos(2 * (b + c)) + z * xi2 * cos(2 * (b + c)) + 2 * y * cos(2 * b + c) + 2 * y * eta2 * cos(2 * b + c) + 2 * y * xi2 * cos(2 * b + c)) *
                       w3 +
                   (-2 + 6 * eta2 + 6 * xi2 - 2 * (1 + eta2 + xi2) * c2b +
                    (1 + eta2 + xi2) * cos(2 * b - 2 * c) + 2 * c2c +
                    2 * eta2 * c2c + 2 * xi2 * c2c + cos(2 * (b + c)) +
                    eta2 * cos(2 * (b + c)) + xi2 * cos(2 * (b + c))) *
                       w3_2 -
                   8 * w2 *
                       (y * (1 + eta2 + xi2) * Power(cb, 2) +
                        (y * (-3 + eta2 + xi2 + (1 + eta2 + xi2) * c2b)) / 2. -
                        z * (1 + eta2 + xi2) * s2b * sc +
                        (1 + eta2 + xi2) * s2b * sc * w3))) -
          4 * Sqrt(2) * z * eta * cc *
              std::
                  sqrt(
                      Power(
                          y *
                                  eta * cb +
                              z * xi * cc - z * eta * sb * sc - eta * cb * w2 +
                              (-(xi * cc) + eta * sb * sc) * w3,
                          2) *
                      (-4 * y2 -
                       2 * z2 + 4 * y2 * eta2 + 6 * z2 * eta2 + 4 * y2 * xi2 + 6 * z2 * xi2 + 4 * y2 * c2b - 2 * z2 * c2b + 4 * y2 * eta2 * c2b - 2 * z2 * eta2 * c2b + 4 * y2 * xi2 * c2b - 2 * z2 * xi2 * c2b + z2 * cos(2 * b - 2 * c) + z2 * eta2 * cos(2 * b - 2 * c) + z2 * xi2 * cos(2 * b - 2 * c) - 4 * y * z * cos(2 * b - c) - 4 * y * z * eta2 * cos(2 * b - c) - 4 * y * z * xi2 * cos(2 * b - c) + 2 * z2 * c2c + 2 * z2 * eta2 * c2c + 2 * z2 * xi2 * c2c + z2 * cos(2 * (b + c)) + z2 * eta2 * cos(2 * (b + c)) + z2 * xi2 * cos(2 * (b + c)) + 4 * y * z * cos(2 * b + c) + 4 * y * z * eta2 * cos(2 * b + c) + 4 * y * z * xi2 * cos(2 * b + c) + 4 * (-1 + eta2 + xi2 + (1 + eta2 + xi2) * c2b) * w2_2 - 2 * (-2 * z + 6 * z * eta2 + 6 * z * xi2 - 2 * z * (1 + eta2 + xi2) * c2b + z * (1 + eta2 + xi2) * cos(2 * b - 2 * c) - 2 * y * cos(2 * b - c) - 2 * y * eta2 * cos(2 * b - c) - 2 * y * xi2 * cos(2 * b - c) + 2 * z * c2c + 2 * z * eta2 * c2c + 2 * z * xi2 * c2c + z * cos(2 * (b + c)) + z * eta2 * cos(2 * (b + c)) + z * xi2 * cos(2 * (b + c)) + 2 * y * cos(2 * b + c) + 2 * y * eta2 * cos(2 * b + c) + 2 * y * xi2 * cos(2 * b + c)) * w3 +
                       (-2 + 6 * eta2 + 6 * xi2 - 2 * (1 + eta2 + xi2) * c2b +
                        (1 + eta2 + xi2) * cos(2 * b - 2 * c) + 2 * c2c +
                        2 * eta2 * c2c + 2 * xi2 * c2c + cos(2 * (b + c)) +
                        eta2 * cos(2 * (b + c)) + xi2 * cos(2 * (b + c))) *
                           w3_2 -
                       8 * w2 *
                           (y * (1 + eta2 + xi2) * Power(cb, 2) +
                            (y * (-3 + eta2 + xi2 + (1 + eta2 + xi2) * c2b)) /
                                2. -
                            z * (1 + eta2 + xi2) * s2b * sc +
                            (1 + eta2 + xi2) * s2b * sc * w3))) -
          4 * Sqrt(2) * z * xi * sb * sc *
              std::sqrt(Power(y * eta * cb + z * xi * cc - z * eta * sb * sc -
                                  eta * cb * w2 +
                                  (-(xi * cc) + eta * sb * sc) * w3,
                              2) *
                        (-4 * y2 - 2 * z2 + 4 * y2 * eta2 + 6 * z2 * eta2 +
                         4 * y2 * xi2 + 6 * z2 * xi2 + 4 * y2 * c2b -
                         2 * z2 * c2b + 4 * y2 * eta2 * c2b -
                         2 * z2 * eta2 * c2b + 4 * y2 * xi2 * c2b -
                         2 * z2 * xi2 * c2b + z2 * cos(2 * b - 2 * c) +
                         z2 * eta2 * cos(2 * b - 2 * c) +
                         z2 * xi2 * cos(2 * b - 2 * c) -
                         4 * y * z * cos(2 * b - c) -
                         4 * y * z * eta2 * cos(2 * b - c) -
                         4 * y * z * xi2 * cos(2 * b - c) + 2 * z2 * c2c +
                         2 * z2 * eta2 * c2c + 2 * z2 * xi2 * c2c +
                         z2 * cos(2 * (b + c)) + z2 * eta2 * cos(2 * (b + c)) +
                         z2 * xi2 * cos(2 * (b + c)) +
                         4 * y * z * cos(2 * b + c) +
                         4 * y * z * eta2 * cos(2 * b + c) +
                         4 * y * z * xi2 * cos(2 * b + c) +
                         4 * (-1 + eta2 + xi2 + (1 + eta2 + xi2) * c2b) * w2_2 -
                         2 * (-2 * z + 6 * z * eta2 + 6 * z * xi2 - 2 * z * (1 + eta2 + xi2) * c2b + z * (1 + eta2 + xi2) * cos(2 * b - 2 * c) - 2 * y * cos(2 * b - c) - 2 * y * eta2 * cos(2 * b - c) - 2 * y * xi2 * cos(2 * b - c) + 2 * z * c2c + 2 * z * eta2 * c2c + 2 * z * xi2 * c2c + z * cos(2 * (b + c)) + z * eta2 * cos(2 * (b + c)) + z * xi2 * cos(2 * (b + c)) + 2 * y * cos(2 * b + c) + 2 * y * eta2 * cos(2 * b + c) + 2 * y * xi2 * cos(2 * b + c)) *
                             w3 +
                         (-2 + 6 * eta2 + 6 * xi2 - 2 * (1 + eta2 + xi2) * c2b +
                          (1 + eta2 + xi2) * cos(2 * b - 2 * c) + 2 * c2c +
                          2 * eta2 * c2c + 2 * xi2 * c2c + cos(2 * (b + c)) +
                          eta2 * cos(2 * (b + c)) + xi2 * cos(2 * (b + c))) *
                             w3_2 -
                         8 * w2 *
                             (y * (1 + eta2 + xi2) * Power(cb, 2) +
                              (y * (-3 + eta2 + xi2 + (1 + eta2 + xi2) * c2b)) /
                                  2. -
                              z * (1 + eta2 + xi2) * s2b * sc +
                              (1 + eta2 + xi2) * s2b * sc * w3))) -
          4 * Sqrt(2) * xi * cb * w2 *
              Sqrt(Power(y * eta * cb + z * xi * cc - z * eta * sb * sc -
                             eta * cb * w2 + (-(xi * cc) + eta * sb * sc) * w3,
                         2) *
                   (-4 * y2 - 2 * z2 + 4 * y2 * eta2 + 6 * z2 * eta2 +
                    4 * y2 * xi2 + 6 * z2 * xi2 + 4 * y2 * c2b - 2 * z2 * c2b +
                    4 * y2 * eta2 * c2b - 2 * z2 * eta2 * c2b +
                    4 * y2 * xi2 * c2b - 2 * z2 * xi2 * c2b +
                    z2 * cos(2 * b - 2 * c) + z2 * eta2 * cos(2 * b - 2 * c) +
                    z2 * xi2 * cos(2 * b - 2 * c) - 4 * y * z * cos(2 * b - c) -
                    4 * y * z * eta2 * cos(2 * b - c) -
                    4 * y * z * xi2 * cos(2 * b - c) + 2 * z2 * c2c +
                    2 * z2 * eta2 * c2c + 2 * z2 * xi2 * c2c +
                    z2 * cos(2 * (b + c)) + z2 * eta2 * cos(2 * (b + c)) +
                    z2 * xi2 * cos(2 * (b + c)) + 4 * y * z * cos(2 * b + c) + 4 * y * z * eta2 * cos(2 * b + c) + 4 * y * z * xi2 * cos(2 * b + c) + 4 * (-1 + eta2 + xi2 + (1 + eta2 + xi2) * c2b) * w2_2 - 2 * (-2 * z + 6 * z * eta2 + 6 * z * xi2 - 2 * z * (1 + eta2 + xi2) * c2b + z * (1 + eta2 + xi2) * cos(2 * b - 2 * c) - 2 * y * cos(2 * b - c) - 2 * y * eta2 * cos(2 * b - c) - 2 * y * xi2 * cos(2 * b - c) + 2 * z * c2c + 2 * z * eta2 * c2c + 2 * z * xi2 * c2c + z * cos(2 * (b + c)) + z * eta2 * cos(2 * (b + c)) + z * xi2 * cos(2 * (b + c)) + 2 * y * cos(2 * b + c) + 2 * y * eta2 * cos(2 * b + c) + 2 * y * xi2 * cos(2 * b + c)) * w3 +
                    (-2 + 6 * eta2 + 6 * xi2 -
                     2 * (1 + eta2 + xi2) * cos(2 * b) +
                     (1 + eta2 + xi2) * cos(2 * b - 2 * c) + 2 * c2c +
                     2 * eta2 * c2c + 2 * xi2 * c2c + cos(2 * (b + c)) +
                     eta2 * cos(2 * (b + c)) + xi2 * cos(2 * (b + c))) *
                        w3_2 -
                    8 * w2 *
                        (y * (1 + eta2 + xi2) * Power(cb, 2) +
                         (y * (-3 + eta2 + xi2 + (1 + eta2 + xi2) * c2b)) / 2. -
                         z * (1 + eta2 + xi2) * s2b * sc +
                         (1 + eta2 + xi2) * s2b * sc * w3))) +
          4 * Sqrt(2) * eta * cc * w3 *
              Sqrt(
                  Power(y * eta * cb + z * xi * cc - z * eta * sb * sc -
                            eta * cb * w2 + (-(xi * cc) + eta * sb * sc) * w3,
                        2) *
                  (-4 * y2 - 2 * z2 + 4 * y2 * eta2 + 6 * z2 * eta2 +
                   4 * y2 * xi2 + 6 * z2 * xi2 + 4 * y2 * c2b - 2 * z2 * c2b +
                   4 * y2 * eta2 * c2b - 2 * z2 * eta2 * c2b +
                   4 * y2 * xi2 * c2b - 2 * z2 * xi2 * c2b +
                   z2 * cos(2 * b - 2 * c) + z2 * eta2 * cos(2 * b - 2 * c) +
                   z2 * xi2 * cos(2 * b - 2 * c) - 4 * y * z * cos(2 * b - c) -
                   4 * y * z * eta2 * cos(2 * b - c) -
                   4 * y * z * xi2 * cos(2 * b - c) + 2 * z2 * c2c +
                   2 * z2 * eta2 * c2c + 2 * z2 * xi2 * c2c +
                   z2 * cos(2 * (b + c)) + z2 * eta2 * cos(2 * (b + c)) +
                   z2 * xi2 * cos(2 * (b + c)) + 4 * y * z * cos(2 * b + c) +
                   4 * y * z * eta2 * cos(2 * b + c) +
                   4 * y * z * xi2 * cos(2 * b + c) +
                   4 * (-1 + eta2 + xi2 + (1 + eta2 + xi2) * c2b) * w2_2 -
                   2 * (-2 * z + 6 * z * eta2 + 6 * z * xi2 - 2 * z * (1 + eta2 + xi2) * c2b + z * (1 + eta2 + xi2) * cos(2 * b - 2 * c) - 2 * y * cos(2 * b - c) - 2 * y * eta2 * cos(2 * b - c) - 2 * y * xi2 * cos(2 * b - c) + 2 * z * c2c + 2 * z * eta2 * c2c + 2 * z * xi2 * c2c + z * cos(2 * (b + c)) + z * eta2 * cos(2 * (b + c)) + z * xi2 * cos(2 * (b + c)) + 2 * y * cos(2 * b + c) + 2 * y * eta2 * cos(2 * b + c) + 2 * y * xi2 * cos(2 * b + c)) *
                       w3 +
                   (-2 + 6 * eta2 + 6 * xi2 - 2 * (1 + eta2 + xi2) * c2b +
                    (1 + eta2 + xi2) * cos(2 * b - 2 * c) + 2 * c2c +
                    2 * eta2 * c2c + 2 * xi2 * c2c + cos(2 * (b + c)) +
                    eta2 * cos(2 * (b + c)) + xi2 * cos(2 * (b + c))) *
                       w3_2 -
                   8 * w2 *
                       (y * (1 + eta2 + xi2) * Power(cb, 2) +
                        (y * (-3 + eta2 + xi2 + (1 + eta2 + xi2) * c2b)) / 2. -
                        z * (1 + eta2 + xi2) * s2b * sc +
                        (1 + eta2 + xi2) * s2b * sc * w3))) +
          4 * Sqrt(2) * xi * sb * sc * w3 *
              Sqrt(Power(y * eta * cb + z * xi * cc - z * eta * sb * sc -
                             eta * cb * w2 + (-(xi * cc) + eta * sb * sc) * w3,
                         2) *
                   (-4 * y2 - 2 * z2 + 4 * y2 * eta2 + 6 * z2 * eta2 +
                    4 * y2 * xi2 + 6 * z2 * xi2 + 4 * y2 * c2b - 2 * z2 * c2b +
                    4 * y2 * eta2 * c2b - 2 * z2 * eta2 * c2b +
                    4 * y2 * xi2 * c2b - 2 * z2 * xi2 * c2b +
                    z2 * cos(2 * b - 2 * c) + z2 * eta2 * cos(2 * b - 2 * c) +
                    z2 * xi2 * cos(2 * b - 2 * c) - 4 * y * z * cos(2 * b - c) -
                    4 * y * z * eta2 * cos(2 * b - c) -
                    4 * y * z * xi2 * cos(2 * b - c) + 2 * z2 * c2c +
                    2 * z2 * eta2 * c2c + 2 * z2 * xi2 * c2c +
                    z2 * cos(2 * (b + c)) + z2 * eta2 * cos(2 * (b + c)) +
                    z2 * xi2 * cos(2 * (b + c)) + 4 * y * z * cos(2 * b + c) +
                    4 * y * z * eta2 * cos(2 * b + c) +
                    4 * y * z * xi2 * cos(2 * b + c) + 4 * (-1 + eta2 + xi2 + (1 + eta2 + xi2) * c2b) * w2_2 - 2 * (-2 * z + 6 * z * eta2 + 6 * z * xi2 - 2 * z * (1 + eta2 + xi2) * c2b + z * (1 + eta2 + xi2) * cos(2 * b - 2 * c) - 2 * y * cos(2 * b - c) - 2 * y * eta2 * cos(2 * b - c) - 2 * y * xi2 * cos(2 * b - c) + 2 * z * c2c + 2 * z * eta2 * c2c + 2 * z * xi2 * c2c + z * cos(2 * (b + c)) + z * eta2 * cos(2 * (b + c)) + z * xi2 * cos(2 * (b + c)) + 2 * y * cos(2 * b + c) + 2 * y * eta2 * cos(2 * b + c) + 2 * y * xi2 * cos(2 * b + c)) * w3 +
                    (-2 + 6 * eta2 + 6 * xi2 - 2 * (1 + eta2 + xi2) * c2b +
                     (1 + eta2 + xi2) * cos(2 * b - 2 * c) + 2 * c2c +
                     2 * eta2 * c2c + 2 * xi2 * c2c + cos(2 * (b + c)) +
                     eta2 * cos(2 * (b + c)) + xi2 * cos(2 * (b + c))) *
                        w3_2 -
                    8 * w2 *
                        (y * (1 + eta2 + xi2) * Power(cb, 2) +
                         (y * (-3 + eta2 + xi2 + (1 + eta2 + xi2) * c2b)) / 2. -
                         z * (1 + eta2 + xi2) * s2b * sc +
                         (1 + eta2 + xi2) * s2b * sc * w3)))) /
          (2. * (eta2 + xi2) *
           (y * eta * cb + z * xi * cc - z * eta * sb * sc - eta * cb * w2 +
            (-(xi * cc) + eta * sb * sc) * w3) *
           (4 * y2 + 6 * z2 + 4 * y2 * c2b - 2 * z2 * c2b +
            z2 * cos(2 * b - 2 * c) -
            4 * y * z * cos(2 * b - c) + 2 * z2 * c2c + z2 * cos(2 * (b + c)) + 4 * y * z * cos(2 * b + c) + 8 * Power(cb, 2) * w2_2 - 2 * (6 * z - 2 * z * c2b + z * cos(2 * b - 2 * c) - 2 * y * cos(2 * b - c) + 2 * z * c2c + z * cos(2 * (b + c)) + 2 * y * cos(2 * b + c)) * w3 +
            (6 - 2 * c2b + cos(2 * b - 2 * c) + 2 * c2c + cos(2 * (b + c))) *
                w3_2 -
            16 * cb * w2 * (y * cb - z * sb * sc + sb * sc * w3))));
  const float x2 =
      ((-6 + 2 * eta2 + 2 * xi2 + 2 * (1 + eta2 + xi2) * c2b +
        (1 + eta2 + xi2) * cos(2 * (b - c)) + 2 * c2c + 2 * eta2 * c2c +
        2 * xi2 * c2c + cos(2 * (b + c)) + eta2 * cos(2 * (b + c)) +
        xi2 * cos(2 * (b + c))) *
           w1 +
       (2 *
        (4 * z2 * xi * Power(cb, 3) * Power(cc, 2) * sb +
         4 * z2 * eta2 * xi * Power(cb, 3) * Power(cc, 2) * sb +
         4 * z2 * xi3 * Power(cb, 3) * Power(cc, 2) * sb +
         4 * y * z * eta * Power(cb, 2) * Power(cc, 3) * sb +
         4 * y * z * eta3 * Power(cb, 2) * Power(cc, 3) * sb +
         4 * y * z * eta * xi2 * Power(cb, 2) * Power(cc, 3) * sb +
         4 * z2 * xi * cb * Power(cc, 2) * Power(sb, 3) +
         4 * z2 * eta2 * xi * cb * Power(cc, 2) * Power(sb, 3) +
         4 * z2 * xi3 * cb * Power(cc, 2) * Power(sb, 3) -
         4 * y * z * xi * Power(cb, 2) * Power(cc, 2) * sc -
         4 * y * z * eta2 * xi * Power(cb, 4) * Power(cc, 2) * sc -
         4 * y * z * xi3 * Power(cb, 4) * Power(cc, 2) * sc -
         4 * y2 * eta * cb * Power(cc, 3) * sc -
         4 * y2 * eta3 * Power(cb, 3) * Power(cc, 3) * sc -
         4 * y2 * eta * xi2 * Power(cb, 3) * Power(cc, 3) * sc -
         4 * z2 * eta * cb * cc * Power(sb, 2) * sc -
         4 * z2 * eta3 * cb * cc * Power(sb, 2) * sc -
         4 * z2 * eta * xi2 * cb * cc * Power(sb, 2) * sc -
         4 * y * z * xi * Power(cc, 2) * Power(sb, 2) * sc +
         4 * y * z * xi * Power(cb, 2) * Power(cc, 2) * Power(sb, 2) * sc -
         4 * y * z * eta2 * xi * Power(cb, 2) * Power(cc, 2) * Power(sb, 2) *
             sc -
         4 * y * z * xi3 * Power(cb, 2) * Power(cc, 2) * Power(sb, 2) * sc +
         4 * y2 * eta * cb * Power(cc, 3) * Power(sb, 2) * sc +
         4 * y * z * xi * Power(cc, 2) * Power(sb, 4) * sc +
         4 * y * z * eta * cc * sb * Power(sc, 2) +
         4 * y * z * eta * Power(cb, 2) * cc * sb * Power(sc, 2) +
         8 * y * z * eta3 * Power(cb, 2) * cc * sb * Power(sc, 2) +
         8 * y * z * eta * xi2 * Power(cb, 2) * cc * sb * Power(sc, 2) -
         4 * y * z * eta * cc * Power(sb, 3) * Power(sc, 2) -
         4 * y2 * eta * cb * cc * Power(sc, 3) -
         4 * y2 * eta3 * Power(cb, 3) * cc * Power(sc, 3) -
         4 * y2 * eta * xi2 * Power(cb, 3) * cc * Power(sc, 3) +
         4 * y2 * eta * cb * cc * Power(sb, 2) * Power(sc, 3) -
         2 * eta * (1 + eta2 + xi2) * Power(cb, 3) * s2c * w2_2 +
         2 * (1 + eta2 + xi2) * cb * cc *
             (-2 * z * eta * Power(cb, 2) * sc +
              2 * z * (-2 * xi * cc * sb + eta * (1 + Power(sb, 2)) * sc) +
              y * cb * (eta * (-3 + c2c) * sb + xi * s2c)) *
             w3 -
         2 * (1 + eta2 + xi2) * cc * s2b * (-(xi * cc) + eta * sb * sc) * w3_2 +
         2 * (1 + eta2 + xi2) * Power(cb, 2) * cc * w2 *
             (z * eta * (-3 + c2c) * sb +
              2 * (2 * y * eta * cb + z * xi * cc) * sc -
              (eta * (-3 + c2c) * sb + xi * s2c) * w3) +
         Sqrt(2) *
             std::sqrt(
                 Power(y * eta * cb + z * xi * cc - z * eta * sb * sc -
                           eta * cb * w2 +
                           (-(xi * cc) + eta * sb * sc) * w3,
                       2) *
                 (-4 * y2 - 2 * z2 + 4 * y2 * eta2 + 6 * z2 * eta2 +
                  4 * y2 * xi2 + 6 * z2 * xi2 + 4 * y2 * c2b - 2 * z2 * c2b +
                  4 * y2 * eta2 * c2b - 2 * z2 * eta2 * c2b +
                  4 * y2 * xi2 * c2b - 2 * z2 * xi2 * c2b +
                  z2 * cos(2 * (b - c)) + z2 * eta2 * cos(2 * (b - c)) +
                  z2 * xi2 * cos(2 * (b - c)) - 4 * y * z * cos(2 * b - c) -
                  4 * y * z * eta2 * cos(2 * b - c) -
                  4 * y * z * xi2 * cos(2 * b - c) + 2 * z2 * c2c +
                  2 * z2 * eta2 * c2c + 2 * z2 * xi2 * c2c +
                  z2 * cos(2 * (b + c)) + z2 * eta2 * cos(2 * (b + c)) +
                  z2 * xi2 * cos(2 * (b + c)) + 4 * y * z * cos(2 * b + c) +
                  4 * y * z * eta2 * cos(2 * b + c) +
                  4 * y * z * xi2 * cos(2 * b + c) +
                  4 * (-1 + eta2 + xi2 + (1 + eta2 + xi2) * c2b) * w2_2 -
                  2 * (-2 * z + 6 * z * eta2 + 6 * z * xi2 - 2 * z * (1 + eta2 + xi2) * c2b + z * (1 + eta2 + xi2) * cos(2 * (b - c)) - 2 * y * cos(2 * b - c) - 2 * y * eta2 * cos(2 * b - c) - 2 * y * xi2 * cos(2 * b - c) + 2 * z * c2c + 2 * z * eta2 * c2c + 2 * z * xi2 * c2c + z * cos(2 * (b + c)) + z * eta2 * cos(2 * (b + c)) + z * xi2 * cos(2 * (b + c)) + 2 * y * cos(2 * b + c) + 2 * y * eta2 * cos(2 * b + c) + 2 * y * xi2 * cos(2 * b + c)) *
                      w3 +
                  (-2 + 6 * eta2 + 6 * xi2 - 2 * (1 + eta2 + xi2) * c2b +
                   (1 + eta2 + xi2) * cos(2 * (b - c)) + 2 * c2c +
                   2 * eta2 * c2c + 2 * xi2 * c2c + cos(2 * (b + c)) +
                   eta2 * cos(2 * (b + c)) + xi2 * cos(2 * (b + c))) *
                      w3_2 -
                  4 * w2 *
                      (-3 * y + y * eta2 + y * xi2 +
                       2 * y * (1 + eta2 + xi2) * Power(cb, 2) + y * c2b +
                       y * eta2 * c2b + y * xi2 * c2b - 2 * z * s2b * sc -
                       2 * z * eta2 * s2b * sc - 2 * z * xi2 * s2b * sc +
                       2 * (1 + eta2 + xi2) * s2b * sc * w3))))) /
           (y * eta * cb + z * xi * cc - z * eta * sb * sc - eta * cb * w2 +
            (-(xi * cc) + eta * sb * sc) * w3)) /
      (8. * ((eta2 + xi2) * Power(cb, 2) * Power(cc, 2) -
             Power(cc, 2) * Power(sb, 2) - Power(sc, 2)));
  const float a2 = ArcTan((4 * (-(z2 * xi * cb * sb) -
                                2 * y * z * eta * cc * sb +
                                z2 * xi * cb * Power(cc, 2) * sb +
                                y2 * xi * s2b +
                                2 * y * z * xi * Power(cb, 2) * sc -
                                2 * z2 * eta * cb * cc * sc -
                                2 * y * z * xi * Power(sb, 2) * sc -
                                z2 * xi * cb * sb * Power(sc, 2) +
                                xi * s2b * w2_2 + 2 * y * eta * cc * sb * w3 -
                                2 * z * xi * cb * Power(cc, 2) * sb * w3 +
                                z * xi * s2b * w3 -
                                2 * y * xi * Power(cb, 2) * sc * w3 +
                                4 * z * eta * cb * cc * sc * w3 +
                                2 * y * xi * Power(sb, 2) * sc * w3 +
                                z * xi * s2b * Power(sc, 2) * w3 -
                                xi * cb * sb * w3_2 +
                                xi * cb * Power(cc, 2) * sb * w3_2 -
                                2 * eta * cb * cc * sc * w3_2 -
                                xi * cb * sb * Power(sc, 2) * w3_2 +
                                2 * w2 *
                                    (z * eta * cc * sb -
                                     xi * (y * s2b + z * c2b * sc) +
                                     (-(eta * cc * sb) + xi * c2b * sc) * w3) +
                                std::sqrt(Power(
                                              y * eta * cb +
                                                  z * xi * cc - z * eta * sb * sc - eta * cb * w2 +
                                                  (-(xi * cc) + eta * sb * sc) * w3,
                                              2) *
                                          (-2 * y2 - z2 +
                                           2 * y2 * eta2 + 3 * z2 * eta2 + 2 * y2 * xi2 +
                                           3 * z2 * xi2 + 2 * y2 * c2b - z2 * c2b + 2 * y2 * eta2 * c2b -
                                           z2 * eta2 * c2b +
                                           2 * y2 * xi2 * c2b - z2 * xi2 * c2b + z2 * c2c + z2 * eta2 * c2c +
                                           z2 * xi2 * c2c + z2 * c2b * c2c + z2 * eta2 * c2b * c2c + z2 * xi2 * c2b * c2c -
                                           4 * y * z * s2b * sc -
                                           4 * y * z * eta2 * s2b * sc - 4 * y * z * xi2 * s2b * sc + 2 * (-1 + eta2 + xi2 + (1 + eta2 + xi2) * c2b) * w2_2 - 2 * (z * (1 + eta2 + xi2) * c2c - 2 * y * (1 + eta2 + xi2) * s2b * sc - z * (1 - 3 * eta2 - 3 * xi2 + 2 * (1 + eta2 + xi2) * c2b * Power(sc, 2))) * w3 + (-1 + 3 * eta2 + 3 * xi2 + (1 + eta2 + xi2) * c2c - 2 * (1 + eta2 + xi2) * c2b * Power(sc, 2)) * w3_2 -
                                           2 * w2 *
                                               (-3 * y + y * eta2 +
                                                y * xi2 + 2 * y * (1 + eta2 + xi2) * Power(cb, 2) +
                                                y * c2b + y * eta2 * c2b + y * xi2 * c2b -
                                                2 * z * s2b * sc -
                                                2 * z * eta2 * s2b * sc -
                                                2 * z * xi2 * s2b * sc +
                                                2 * (1 + eta2 + xi2) * s2b *
                                                    sc * w3))))) /
                              ((eta2 + xi2) * (4 * y2 + 6 * z2 + 4 * y2 * c2b -
                                               2 * z2 * c2b + z2 * cos(2 * b - 2 * c) -
                                               4 * y * z * cos(2 * b - c) +
                                               2 * z2 * c2c + z2 * cos(2 * (b + c)) +
                                               4 * y * z * cos(2 * b + c) +
                                               8 * Power(cb, 2) * w2_2 -
                                               2 * (6 * z - 2 * z * c2b + z * cos(2 * b - 2 * c) - 2 * y * cos(2 * b - c) + 2 * z * c2c + z * cos(2 * (b + c)) + 2 * y * cos(2 * b + c)) *
                                                   w3 +
                                               (6 - 2 * c2b +
                                                cos(2 * b - 2 * c) + 2 * c2c + cos(2 * (b + c))) *
                                                   w3_2 -
                                               16 * cb * w2 *
                                                   (y * cb - z * sb * sc +
                                                    sb * sc * w3))),
                          (
                              4 * Power(y, 3) * eta2 * sb +
                              2 * y * z2 * eta2 * sb + 8 * y * z2 * xi2 * sb +
                              12 * Power(y, 3) * eta2 * Power(cb, 2) * sb -
                              18 * y * z2 * eta2 * Power(cb, 2) * sb +
                              32 * y2 * z * eta * xi * cb * cc * sb -
                              8 * Power(z, 3) * eta * xi * cb * cc * sb -
                              2 * y * z2 * eta2 * Power(cc, 2) * sb +
                              8 * y * z2 * xi2 * Power(cc, 2) * sb +
                              18 * y *
                                  z2 * eta2 * Power(cb, 2) * Power(cc, 2) * sb +
                              8 * Power(z, 3) * eta * xi *
                                  cb * Power(cc, 3) * sb -
                              4 * Power(y, 3) * eta2 * Power(sb, 3) +
                              6 * y * z2 * eta2 * Power(sb, 3) -
                              6 * y * z2 * eta2 * Power(cc, 2) * Power(sb, 3) +
                              4 * y2 * z * eta2 * cb * sc +
                              3 * Power(z, 3) * eta2 * cb * sc +
                              4 * Power(z, 3) * xi2 * cb * sc +
                              12 * y2 * z * eta2 * Power(cb, 3) * sc -
                              3 * Power(z, 3) * eta2 * Power(cb, 3) * sc +
                              32 * y * z2 * eta * xi * Power(cb, 2) * cc * sc -
                              3 * Power(z, 3) * eta2 * cb * Power(cc, 2) * sc +
                              12 * Power(z, 3) * xi2 * cb * Power(cc, 2) * sc +
                              3 * Power(z, 3) * eta2 *
                                  Power(cb, 3) * Power(cc, 2) * sc -
                              36 * y2 * z * eta2 * cb * Power(sb, 2) * sc +
                              9 * Power(z, 3) * eta2 * cb * Power(sb, 2) * sc -
                              32 * y * z2 * eta * xi * cc * Power(sb, 2) * sc -
                              9 * Power(z, 3) * eta2 *
                                  cb * Power(cc, 2) * Power(sb, 2) * sc +
                              2 * y * z2 * eta2 * sb * Power(sc, 2) -
                              8 * y * z2 * xi2 * sb * Power(sc, 2) -
                              18 * y *
                                  z2 * eta2 * Power(cb, 2) * sb * Power(sc, 2) +
                              6 * y * z2 * eta2 * Power(sb, 3) * Power(sc, 2) +
                              Power(z, 3) * eta2 * cb * Power(sc, 3) -
                              4 * Power(z, 3) * xi2 * cb * Power(sc, 3) -
                              Power(z, 3) * eta2 * Power(cb, 3) * Power(sc, 3) +
                              3 * Power(z, 3) * eta2 *
                                  cb * Power(sb, 2) * Power(sc, 3) -
                              6 * Power(z, 3) * eta * xi * s2b * sc * s2c -
                              12 * y2 * eta2 * sb * w2 -
                              2 * z2 * eta2 * sb * w2 - 8 * z2 * xi2 * sb * w2 -
                              36 * y2 * eta2 * Power(cb, 2) * sb * w2 +
                              18 * z2 * eta2 * Power(cb, 2) * sb * w2 -
                              64 * y * z * eta * xi * cb * cc * sb * w2 +
                              2 * z2 * eta2 * Power(cc, 2) * sb * w2 -
                              8 * z2 * xi2 * Power(cc, 2) * sb * w2 -
                              18 * z2 * eta2 * Power(cb, 2) * Power(cc, 2) *
                                  sb * w2 +
                              12 * y2 * eta2 * Power(sb, 3) * w2 -
                              6 * z2 * eta2 * Power(sb, 3) * w2 +
                              6 * z2 * eta2 * Power(cc, 2) * Power(sb, 3) * w2 -
                              8 * y * z * eta2 * cb * sc * w2 -
                              24 * y * z * eta2 * Power(cb, 3) * sc * w2 -
                              32 * z2 * eta * xi * Power(cb, 2) * cc * sc * w2 +
                              72 * y * z * eta2 * cb * Power(sb, 2) * sc * w2 +
                              32 * z2 * eta * xi * cc * Power(sb, 2) * sc * w2 -
                              2 * z2 * eta2 * sb * Power(sc, 2) * w2 +
                              8 * z2 * xi2 * sb * Power(sc, 2) * w2 +
                              18 * z2 * eta2 * Power(cb, 2) *
                                  sb * Power(sc, 2) * w2 -
                              6 * z2 * eta2 * Power(sb, 3) * Power(sc, 2) * w2 +
                              12 * y * eta2 * sb * w2_2 +
                              36 * y * eta2 * Power(cb, 2) * sb * w2_2 +
                              32 * z * eta * xi * cb * cc * sb * w2_2 -
                              12 * y * eta2 * Power(sb, 3) * w2_2 +
                              4 * z * eta2 * cb * sc * w2_2 +
                              12 * z * eta2 * Power(cb, 3) * sc * w2_2 -
                              36 * z * eta2 * cb * Power(sb, 2) * sc * w2_2 -
                              4 * eta2 * sb * Power(w2, 3) -
                              12 * eta2 * Power(cb, 2) * sb * Power(w2, 3) +
                              4 * eta2 * Power(sb, 3) * Power(w2, 3) -
                              4 * y * z * eta2 * sb * w3 -
                              16 * y * z * xi2 * sb * w3 +
                              36 * y * z * eta2 * Power(cb, 2) * sb * w3 -
                              32 * y2 * eta * xi * cb * cc * sb * w3 +
                              24 * z2 * eta * xi * cb * cc * sb * w3 +
                              4 * y * z * eta2 * Power(cc, 2) * sb * w3 -
                              16 * y * z * xi2 * Power(cc, 2) * sb * w3 -
                              36 * y * z * eta2 * Power(cb, 2) * Power(cc, 2) *
                                  sb * w3 -
                              24 * z2 * eta * xi * cb * Power(cc, 3) * sb * w3 -
                              12 * y * z * eta2 * Power(sb, 3) * w3 +
                              12 * y * z * eta2 * Power(cc, 2) * Power(sb, 3) *
                                  w3 -
                              4 * y2 * eta2 * cb * sc * w3 -
                              9 * z2 * eta2 * cb * sc * w3 -
                              12 * z2 * xi2 * cb * sc * w3 -
                              12 * y2 * eta2 * Power(cb, 3) * sc * w3 +
                              9 * z2 * eta2 * Power(cb, 3) * sc * w3 -
                              64 * y * z * eta * xi * Power(cb, 2) * cc * sc *
                                  w3 +
                              9 * z2 * eta2 * cb * Power(cc, 2) * sc * w3 -
                              36 * z2 * xi2 * cb * Power(cc, 2) * sc * w3 -
                              9 * z2 * eta2 * Power(cb, 3) * Power(cc, 2) *
                                  sc * w3 +
                              36 * y2 * eta2 * cb * Power(sb, 2) * sc * w3 -
                              27 * z2 * eta2 * cb * Power(sb, 2) * sc * w3 +
                              64 * y * z * eta *
                                  xi * cc * Power(sb, 2) * sc * w3 +
                              27 * z2 *
                                  eta2 * cb * Power(cc, 2) * Power(sb, 2) *
                                  sc * w3 -
                              4 * y * z * eta2 * sb * Power(sc, 2) * w3 +
                              16 * y * z * xi2 * sb * Power(sc, 2) * w3 +
                              36 * y * z * eta2 * Power(cb, 2) *
                                  sb * Power(sc, 2) * w3 +
                              72 * z2 * eta * xi * cb * cc *
                                  sb * Power(sc, 2) * w3 -
                              12 * y * z * eta2 * Power(sb, 3) * Power(sc, 2) *
                                  w3 -
                              3 * z2 * eta2 * cb * Power(sc, 3) * w3 +
                              12 * z2 * xi2 * cb * Power(sc, 3) * w3 +
                              3 * z2 * eta2 * Power(cb, 3) * Power(sc, 3) * w3 -
                              9 * z2 * eta2 * cb * Power(sb, 2) * Power(sc, 3) *
                                  w3 +
                              4 * z * eta2 * sb * w2 * w3 +
                              16 * z * xi2 * sb * w2 * w3 -
                              36 * z * eta2 * Power(cb, 2) * sb * w2 * w3 +
                              64 * y * eta * xi * cb * cc * sb * w2 * w3 -
                              4 * z * eta2 * Power(cc, 2) * sb * w2 * w3 +
                              16 * z * xi2 * Power(cc, 2) * sb * w2 * w3 +
                              36 * z * eta2 * Power(cb, 2) * Power(cc, 2) *
                                  sb * w2 * w3 +
                              12 * z * eta2 * Power(sb, 3) * w2 * w3 -
                              12 * z * eta2 * Power(cc, 2) * Power(sb, 3) *
                                  w2 * w3 +
                              8 * y * eta2 * cb * sc * w2 * w3 +
                              24 * y * eta2 * Power(cb, 3) * sc * w2 * w3 +
                              64 * z * eta * xi * Power(cb, 2) * cc * sc * w2 *
                                  w3 -
                              72 * y * eta2 * cb * Power(sb, 2) * sc * w2 * w3 -
                              64 * z * eta * xi * cc * Power(sb, 2) * sc * w2 *
                                  w3 +
                              4 * z * eta2 * sb * Power(sc, 2) * w2 * w3 -
                              16 * z * xi2 * sb * Power(sc, 2) * w2 * w3 -
                              36 * z * eta2 * Power(cb, 2) *
                                  sb * Power(sc, 2) * w2 * w3 +
                              12 * z * eta2 * Power(sb, 3) * Power(sc, 2) *
                                  w2 * w3 -
                              32 * eta * xi * cb * cc * sb * w2_2 * w3 -
                              4 * eta2 * cb * sc * w2_2 * w3 -
                              12 * eta2 * Power(cb, 3) * sc * w2_2 * w3 +
                              36 * eta2 * cb * Power(sb, 2) * sc * w2_2 * w3 +
                              2 * y * eta2 * sb * w3_2 +
                              8 * y * xi2 * sb * w3_2 -
                              18 * y * eta2 * Power(cb, 2) * sb * w3_2 -
                              24 * z * eta * xi * cb * cc * sb * w3_2 -
                              2 * y * eta2 * Power(cc, 2) * sb * w3_2 +
                              8 * y * xi2 * Power(cc, 2) * sb * w3_2 +
                              18 * y * eta2 * Power(cb, 2) * Power(cc, 2) *
                                  sb * w3_2 +
                              24 * z * eta *
                                  xi * cb * Power(cc, 3) * sb * w3_2 +
                              6 * y * eta2 * Power(sb, 3) * w3_2 -
                              6 * y * eta2 * Power(cc, 2) * Power(sb, 3) *
                                  w3_2 +
                              9 * z * eta2 * cb * sc * w3_2 +
                              12 * z * xi2 * cb * sc * w3_2 -
                              9 * z * eta2 * Power(cb, 3) * sc * w3_2 +
                              32 * y * eta * xi * Power(cb, 2) * cc * sc *
                                  w3_2 -
                              9 * z * eta2 * cb * Power(cc, 2) * sc * w3_2 +
                              36 * z * xi2 * cb * Power(cc, 2) * sc * w3_2 +
                              9 * z * eta2 * Power(cb, 3) * Power(cc, 2) *
                                  sc * w3_2 +
                              27 * z * eta2 * cb * Power(sb, 2) * sc * w3_2 -
                              32 * y * eta *
                                  xi * cc * Power(sb, 2) * sc * w3_2 -
                              27 * z * eta2 * cb * Power(cc, 2) * Power(sb, 2) *
                                  sc * w3_2 +
                              2 * y * eta2 * sb * Power(sc, 2) * w3_2 -
                              8 * y * xi2 * sb * Power(sc, 2) * w3_2 -
                              18 * y * eta2 * Power(cb, 2) *
                                  sb * Power(sc, 2) * w3_2 -
                              72 * z * eta * xi * cb * cc *
                                  sb * Power(sc, 2) * w3_2 +
                              6 * y * eta2 * Power(sb, 3) * Power(sc, 2) *
                                  w3_2 +
                              3 * z * eta2 * cb * Power(sc, 3) * w3_2 -
                              12 * z * xi2 * cb * Power(sc, 3) * w3_2 -
                              3 * z * eta2 * Power(cb, 3) * Power(sc, 3) *
                                  w3_2 +
                              9 * z * eta2 * cb * Power(sb, 2) * Power(sc, 3) *
                                  w3_2 -
                              2 * eta2 * sb * w2 * w3_2 -
                              8 * xi2 * sb * w2 * w3_2 +
                              18 * eta2 * Power(cb, 2) * sb * w2 * w3_2 +
                              2 * eta2 * Power(cc, 2) * sb * w2 * w3_2 -
                              8 * xi2 * Power(cc, 2) * sb * w2 * w3_2 -
                              18 * eta2 * Power(cb, 2) * Power(cc, 2) *
                                  sb * w2 * w3_2 -
                              6 * eta2 * Power(sb, 3) * w2 * w3_2 +
                              6 * eta2 * Power(cc, 2) * Power(sb, 3) *
                                  w2 * w3_2 -
                              32 * eta * xi * Power(cb, 2) * cc * sc * w2 *
                                  w3_2 +
                              32 * eta * xi * cc * Power(sb, 2) * sc * w2 *
                                  w3_2 -
                              2 * eta2 * sb * Power(sc, 2) * w2 * w3_2 +
                              8 * xi2 * sb * Power(sc, 2) * w2 * w3_2 +
                              18 * eta2 *
                                  Power(cb, 2) * sb * Power(sc, 2) * w2 * w3_2 -
                              6 * eta2 * Power(sb, 3) * Power(sc, 2) *
                                  w2 * w3_2 +
                              8 * eta * xi * cb * cc * sb * Power(w3, 3) -
                              8 * eta * xi *
                                  cb * Power(cc, 3) * sb * Power(w3, 3) -
                              3 * eta2 * cb * sc * Power(w3, 3) -
                              4 * xi2 * cb * sc * Power(w3, 3) +
                              3 * eta2 * Power(cb, 3) * sc * Power(w3, 3) +
                              3 * eta2 * cb * Power(cc, 2) * sc * Power(w3, 3) -
                              12 * xi2 * cb * Power(cc, 2) * sc * Power(w3, 3) -
                              3 * eta2 * Power(cb, 3) * Power(cc, 2) *
                                  sc * Power(w3, 3) -
                              9 * eta2 * cb * Power(sb, 2) * sc * Power(w3, 3) +
                              9 * eta2 * cb * Power(cc, 2) * Power(sb, 2) *
                                  sc * Power(w3, 3) -
                              eta2 * cb * Power(sc, 3) * Power(w3, 3) +
                              4 * xi2 * cb * Power(sc, 3) * Power(w3, 3) +
                              eta2 * Power(cb, 3) * Power(sc, 3) *
                                  Power(w3, 3) -
                              3 * eta2 * cb * Power(sb, 2) * Power(sc, 3) *
                                  Power(w3, 3) +
                              6 * eta * xi * s2b * sc * s2c * Power(w3, 3) -
                              4 * Sqrt(2) * y * xi * cb *
                                  std::
                                      sqrt(Power(
                                               y * eta * cb +
                                                   z * xi * cc - z * eta * sb * sc - eta * cb * w2 + (-(xi * cc) + eta * sb * sc) * w3,
                                               2) *
                                           (-4 * y2 - 2 * z2 + 4 * y2 * eta2 + 6 * z2 * eta2 + 4 * y2 * xi2 + 6 * z2 * xi2 + 4 * y2 * c2b - 2 * z2 * c2b + 4 * y2 * eta2 * c2b - 2 * z2 * eta2 * c2b + 4 * y2 * xi2 * c2b - 2 * z2 * xi2 * c2b + z2 * cos(2 * b - 2 * c) +
                                            z2 * eta2 * cos(2 * b - 2 * c) + z2 * xi2 * cos(2 * b - 2 * c) -
                                            4 * y * z * cos(2 * b - c) -
                                            4 * y * z * eta2 * cos(2 * b - c) -
                                            4 * y * z * xi2 * cos(2 * b - c) +
                                            2 * z2 * c2c + 2 * z2 * eta2 * c2c + 2 * z2 * xi2 * c2c +
                                            z2 * cos(2 * (b + c)) + z2 * eta2 * cos(2 * (b + c)) +
                                            z2 *
                                                xi2 * cos(2 * (b + c)) +
                                            4 * y * z * cos(2 * b + c) +
                                            4 * y * z * eta2 * cos(2 * b + c) +
                                            4 * y * z * xi2 * cos(2 * b + c) +
                                            4 * (-1 + eta2 + xi2 + (1 + eta2 + xi2) * c2b) * w2_2 - 2 * (-2 * z + 6 * z * eta2 + 6 * z * xi2 - 2 * z * (1 + eta2 + xi2) * c2b + z * (1 + eta2 + xi2) * cos(2 * b - 2 * c) - 2 * y * cos(2 * b - c) - 2 * y * eta2 * cos(2 * b - c) - 2 * y * xi2 * cos(2 * b - c) + 2 * z * c2c + 2 * z * eta2 * c2c + 2 * z * xi2 * c2c + z * cos(2 * (b + c)) + z * eta2 * cos(2 * (b + c)) + z * xi2 * cos(2 * (b + c)) + 2 * y * cos(2 * b + c) + 2 * y * eta2 * cos(2 * b + c) + 2 * y * xi2 * cos(2 * b + c)) * w3 +
                                            (-2 + 6 * eta2 + 6 * xi2 -
                                             2 * (1 + eta2 + xi2) * c2b +
                                             (1 + eta2 + xi2) *
                                                 cos(2 * b - 2 * c) +
                                             2 * c2c + 2 * eta2 * c2c +
                                             2 * xi2 * c2c + cos(2 * (b + c)) +
                                             eta2 * cos(2 * (b + c)) +
                                             xi2 * cos(2 * (b + c))) *
                                                w3_2 -
                                            8 * w2 *
                                                (y * (1 + eta2 + xi2) *
                                                     Power(cb, 2) +
                                                 (y *
                                                  (-3 + eta2 + xi2 +
                                                   (1 + eta2 + xi2) * c2b)) /
                                                     2. -
                                                 z * (1 + eta2 + xi2) * s2b *
                                                     sc +
                                                 (1 + eta2 + xi2) * s2b * sc *
                                                     w3))) +
                              4 * Sqrt(2) * z * eta * cc *
                                  std::sqrt(Power(y * eta * cb +
                                                      z * xi * cc - z * eta * sb * sc - eta * cb * w2 +
                                                      (-(xi * cc) + eta * sb *
                                                                        sc) *
                                                          w3,
                                                  2) *
                                            (-4 * y2 - 2 * z2 + 4 * y2 * eta2 +
                                             6 * z2 * eta2 + 4 * y2 * xi2 +
                                             6 * z2 * xi2 + 4 * y2 * c2b -
                                             2 * z2 * c2b +
                                             4 * y2 * eta2 * c2b -
                                             2 * z2 * eta2 * c2b +
                                             4 * y2 * xi2 * c2b -
                                             2 * z2 * xi2 * c2b +
                                             z2 * cos(2 * b - 2 * c) +
                                             z2 * eta2 * cos(2 * b - 2 * c) +
                                             z2 * xi2 * cos(2 * b - 2 * c) -
                                             4 * y * z * cos(2 * b - c) -
                                             4 * y * z * eta2 * cos(2 * b - c) -
                                             4 * y * z * xi2 * cos(2 * b - c) +
                                             2 * z2 * c2c +
                                             2 * z2 * eta2 * c2c +
                                             2 * z2 * xi2 * c2c + z2 * cos(2 * (b + c)) + z2 * eta2 * cos(2 * (b + c)) + z2 * xi2 * cos(2 * (b + c)) + 4 * y * z * cos(2 * b + c) + 4 * y * z * eta2 * cos(2 * b + c) + 4 * y * z * xi2 * cos(2 * b + c) + 4 * (-1 + eta2 + xi2 + (1 + eta2 + xi2) * c2b) * w2_2 - 2 * (-2 * z + 6 * z * eta2 + 6 * z * xi2 - 2 * z * (1 + eta2 + xi2) * c2b + z * (1 + eta2 + xi2) * cos(2 * b - 2 * c) - 2 * y * cos(2 * b - c) - 2 * y * eta2 * cos(2 * b - c) - 2 * y * xi2 * cos(2 * b - c) + 2 * z * c2c + 2 * z * eta2 * c2c + 2 * z * xi2 * c2c + z * cos(2 * (b + c)) + z * eta2 * cos(2 * (b + c)) + z * xi2 * cos(2 * (b + c)) + 2 * y * cos(2 * b + c) + 2 * y * eta2 * cos(2 * b + c) + 2 * y * xi2 * cos(2 * b + c)) * w3 +
                                             (-2 + 6 * eta2 + 6 * xi2 -
                                              2 * (1 + eta2 + xi2) * c2b +
                                              (1 + eta2 + xi2) *
                                                  cos(2 * b - 2 * c) +
                                              2 * c2c + 2 * eta2 * c2c +
                                              2 * xi2 * c2c + cos(2 * (b + c)) +
                                              eta2 * cos(2 * (b + c)) +
                                              xi2 * cos(2 * (b + c))) *
                                                 w3_2 -
                                             8 * w2 *
                                                 (y * (1 + eta2 + xi2) *
                                                      Power(cb, 2) +
                                                  (y *
                                                   (-3 + eta2 + xi2 +
                                                    (1 + eta2 + xi2) * c2b)) /
                                                      2. -
                                                  z * (1 + eta2 + xi2) * s2b *
                                                      sc +
                                                  (1 + eta2 + xi2) * s2b * sc *
                                                      w3))) +
                              4 * Sqrt(2) * z * xi * sb * sc *
                                  Sqrt(Power(
                                           y * eta * cb + z * xi * cc -
                                               z * eta * sb * sc -
                                               eta * cb * w2 +
                                               (-(xi * cc) + eta * sb * sc) *
                                                   w3,
                                           2) *
                                       (-4 * y2 - 2 * z2 + 4 * y2 * eta2 +
                                        6 * z2 * eta2 + 4 * y2 * xi2 +
                                        6 * z2 * xi2 + 4 * y2 * c2b -
                                        2 * z2 * c2b + 4 * y2 * eta2 * c2b -
                                        2 * z2 * eta2 * c2b +
                                        4 * y2 * xi2 * c2b -
                                        2 * z2 * xi2 * c2b +
                                        z2 * cos(2 * b - 2 * c) +
                                        z2 * eta2 * cos(2 * b - 2 * c) +
                                        z2 * xi2 * cos(2 * b - 2 * c) -
                                        4 * y * z * cos(2 * b - c) -
                                        4 * y * z * eta2 * cos(2 * b - c) -
                                        4 * y * z * xi2 * cos(2 * b - c) +
                                        2 * z2 * c2c + 2 * z2 * eta2 * c2c +
                                        2 * z2 * xi2 * c2c +
                                        z2 * cos(2 * (b + c)) +
                                        z2 * eta2 * cos(2 * (b + c)) +
                                        z2 * xi2 * cos(2 * (b + c)) +
                                        4 * y * z * cos(2 * b + c) +
                                        4 * y * z * eta2 * cos(2 * b + c) +
                                        4 * y * z * xi2 * cos(2 * b + c) +
                                        4 * (-1 + eta2 + xi2 + (1 + eta2 + xi2) * c2b) *
                                            w2_2 -
                                        2 * (-2 * z + 6 * z * eta2 + 6 * z * xi2 - 2 * z * (1 + eta2 + xi2) * c2b + z * (1 + eta2 + xi2) * cos(2 * b - 2 * c) - 2 * y * cos(2 * b - c) - 2 * y * eta2 * cos(2 * b - c) - 2 * y * xi2 * cos(2 * b - c) + 2 * z * c2c + 2 * z * eta2 * c2c + 2 * z * xi2 * c2c + z * cos(2 * (b + c)) + z * eta2 * cos(2 * (b + c)) + z * xi2 * cos(2 * (b + c)) + 2 * y * cos(2 * b + c) + 2 * y * eta2 * cos(2 * b + c) + 2 * y * xi2 * cos(2 * b + c)) *
                                            w3 +
                                        (-2 + 6 * eta2 + 6 * xi2 -
                                         2 * (1 + eta2 + xi2) * c2b +
                                         (1 + eta2 + xi2) * cos(2 * b - 2 * c) +
                                         2 * c2c + 2 * eta2 * c2c +
                                         2 * xi2 * c2c + cos(2 * (b + c)) +
                                         eta2 * cos(2 * (b + c)) +
                                         xi2 * cos(2 * (b + c))) *
                                            w3_2 -
                                        8 * w2 *
                                            (y * (1 + eta2 + xi2) *
                                                 Power(cb, 2) +
                                             (y * (-3 + eta2 + xi2 +
                                                   (1 + eta2 + xi2) * c2b)) /
                                                 2. -
                                             z * (1 + eta2 + xi2) * s2b * sc +
                                             (1 + eta2 + xi2) * s2b * sc *
                                                 w3))) +
                              4 * Sqrt(2) * xi * cb * w2 *
                                  std::sqrt(
                                      Power(y * eta * cb +
                                                z * xi * cc - z * eta * sb * sc -
                                                eta * cb * w2 +
                                                (-(xi * cc) + eta * sb * sc) *
                                                    w3,
                                            2) *
                                      (-4 * y2 -
                                       2 * z2 + 4 * y2 * eta2 + 6 * z2 * eta2 +
                                       4 * y2 *
                                           xi2 +
                                       6 * z2 *
                                           xi2 +
                                       4 * y2 *
                                           c2b -
                                       2 * z2 *
                                           c2b +
                                       4 * y2 *
                                           eta2 * c2b -
                                       2 * z2 *
                                           eta2 * c2b +
                                       4 * y2 *
                                           xi2 * c2b -
                                       2 * z2 *
                                           xi2 * c2b +
                                       z2 * cos(2 * b - 2 * c) +
                                       z2 * eta2 * cos(2 * b - 2 * c) +
                                       z2 * xi2 * cos(2 * b - 2 * c) -
                                       4 * y * z * cos(2 * b - c) -
                                       4 * y * z * eta2 * cos(2 * b - c) -
                                       4 * y * z * xi2 * cos(2 * b - c) + 2 * z2 * c2c + 2 * z2 * eta2 * c2c + 2 * z2 * xi2 * c2c + z2 * cos(2 * (b + c)) + z2 * eta2 * cos(2 * (b + c)) + z2 * xi2 * cos(2 * (b + c)) + 4 * y * z * cos(2 * b + c) + 4 * y * z * eta2 * cos(2 * b + c) + 4 * y * z * xi2 * cos(2 * b + c) + 4 * (-1 + eta2 + xi2 + (1 + eta2 + xi2) * c2b) * w2_2 - 2 * (-2 * z + 6 * z * eta2 + 6 * z * xi2 - 2 * z * (1 + eta2 + xi2) * c2b + z * (1 + eta2 + xi2) * cos(2 * b - 2 * c) - 2 * y * cos(2 * b - c) - 2 * y * eta2 * cos(2 * b - c) - 2 * y * xi2 * cos(2 * b - c) + 2 * z * c2c + 2 * z * eta2 * c2c + 2 * z * xi2 * c2c + z * cos(2 * (b + c)) + z * eta2 * cos(2 * (b + c)) + z * xi2 * cos(2 * (b + c)) + 2 * y * cos(2 * b + c) + 2 * y * eta2 * cos(2 * b + c) + 2 * y * xi2 * cos(2 * b + c)) * w3 +
                                       (-2 + 6 * eta2 + 6 * xi2 -
                                        2 * (1 + eta2 + xi2) * c2b +
                                        (1 + eta2 + xi2) * cos(2 * b - 2 * c) +
                                        2 * c2c + 2 * eta2 * c2c +
                                        2 * xi2 * c2c + cos(2 * (b + c)) +
                                        eta2 * cos(2 * (b + c)) +
                                        xi2 * cos(2 * (b + c))) *
                                           w3_2 -
                                       8 * w2 *
                                           (y * (1 + eta2 + xi2) *
                                                Power(cb, 2) +
                                            (y * (-3 + eta2 + xi2 +
                                                  (1 + eta2 + xi2) * c2b)) /
                                                2. -
                                            z * (1 + eta2 + xi2) * s2b * sc +
                                            (1 + eta2 + xi2) * s2b * sc *
                                                w3))) -
                              4 * Sqrt(2) * eta * cc * w3 *
                                  std::
                                      sqrt(Power(
                                               y * eta * cb + z * xi * cc -
                                                   z * eta * sb * sc -
                                                   eta * cb * w2 +
                                                   (-(xi * cc) + eta * sb *
                                                                     sc) *
                                                       w3,
                                               2) *
                                           (-4 * y2 - 2 * z2 + 4 * y2 * eta2 +
                                            6 * z2 * eta2 + 4 * y2 * xi2 +
                                            6 * z2 * xi2 +
                                            4 * y2 * c2b - 2 * z2 * c2b + 4 * y2 * eta2 * c2b - 2 * z2 * eta2 * c2b + 4 * y2 * xi2 * c2b - 2 * z2 * xi2 * c2b + z2 * cos(2 * b - 2 * c) + z2 * eta2 * cos(2 * b - 2 * c) + z2 * xi2 * cos(2 * b - 2 * c) - 4 * y * z * cos(2 * b - c) - 4 * y * z * eta2 * cos(2 * b - c) - 4 * y * z * xi2 * cos(2 * b - c) + 2 * z2 * c2c + 2 * z2 * eta2 * c2c + 2 * z2 * xi2 * c2c + z2 * cos(2 * (b + c)) + z2 * eta2 * cos(2 * (b + c)) + z2 * xi2 * cos(2 * (b + c)) + 4 * y * z * cos(2 * b + c) + 4 * y * z * eta2 * cos(2 * b + c) + 4 * y * z * xi2 * cos(2 * b + c) + 4 * (-1 + eta2 + xi2 + (1 + eta2 + xi2) * c2b) * w2_2 - 2 * (-2 * z + 6 * z * eta2 + 6 * z * xi2 - 2 * z * (1 + eta2 + xi2) * c2b + z * (1 + eta2 + xi2) * cos(2 * b - 2 * c) - 2 * y * cos(2 * b - c) - 2 * y * eta2 * cos(2 * b - c) - 2 * y * xi2 * cos(2 * b - c) + 2 * z * c2c + 2 * z * eta2 * c2c + 2 * z * xi2 * c2c + z * cos(2 * (b + c)) + z * eta2 * cos(2 * (b + c)) + z * xi2 * cos(2 * (b + c)) + 2 * y * cos(2 * b + c) + 2 * y * eta2 * cos(2 * b + c) + 2 * y * xi2 * cos(2 * b + c)) * w3 +
                                            (-2 + 6 * eta2 + 6 * xi2 -
                                             2 * (1 + eta2 + xi2) * c2b +
                                             (1 + eta2 + xi2) *
                                                 cos(2 * b - 2 * c) +
                                             2 * c2c + 2 * eta2 * c2c +
                                             2 * xi2 * c2c + cos(2 * (b + c)) +
                                             eta2 * cos(2 * (b + c)) +
                                             xi2 * cos(2 * (b + c))) *
                                                w3_2 -
                                            8 * w2 *
                                                (y * (1 + eta2 + xi2) *
                                                     Power(cb, 2) +
                                                 (y *
                                                  (-3 + eta2 +
                                                   xi2 + (1 + eta2 + xi2) * c2b)) /
                                                     2. -
                                                 z * (1 + eta2 + xi2) * s2b *
                                                     sc +
                                                 (1 + eta2 + xi2) * s2b * sc *
                                                     w3))) -
                              4 * Sqrt(2) * xi * sb * sc * w3 *
                                  std::sqrt(
                                      Power(y * eta * cb +
                                                z * xi * cc - z * eta * sb * sc -
                                                eta * cb * w2 +
                                                (-(xi * cc) + eta * sb * sc) *
                                                    w3,
                                            2) *
                                      (-4 * y2 -
                                       2 * z2 + 4 * y2 * eta2 + 6 * z2 * eta2 +
                                       4 * y2 *
                                           xi2 +
                                       6 * z2 *
                                           xi2 +
                                       4 * y2 *
                                           c2b -
                                       2 * z2 *
                                           c2b +
                                       4 * y2 *
                                           eta2 * c2b -
                                       2 * z2 *
                                           eta2 * c2b +
                                       4 * y2 *
                                           xi2 * c2b -
                                       2 * z2 *
                                           xi2 * c2b +
                                       z2 * cos(2 * b - 2 * c) +
                                       z2 * eta2 * cos(2 * b - 2 * c) +
                                       z2 * xi2 * cos(2 * b - 2 * c) -
                                       4 * y * z * cos(2 * b - c) -
                                       4 * y * z * eta2 * cos(2 * b - c) -
                                       4 * y * z * xi2 * cos(2 * b - c) +
                                       2 * z2 *
                                           c2c +
                                       2 * z2 *
                                           eta2 * c2c +
                                       2 * z2 *
                                           xi2 * c2c +
                                       z2 * cos(2 * (b + c)) +
                                       z2 * eta2 * cos(2 * (b + c)) +
                                       z2 * xi2 * cos(2 * (b + c)) +
                                       4 * y * z * cos(2 * b + c) +
                                       4 * y * z * eta2 * cos(2 * b + c) +
                                       4 * y * z * xi2 * cos(2 * b + c) +
                                       4 * (-1 + eta2 + xi2 + (1 + eta2 + xi2) * c2b) *
                                           w2_2 -
                                       2 * (-2 * z + 6 * z * eta2 + 6 * z * xi2 - 2 * z * (1 + eta2 + xi2) * c2b + z * (1 + eta2 + xi2) * cos(2 * b - 2 * c) - 2 * y * cos(2 * b - c) - 2 * y * eta2 * cos(2 * b - c) - 2 * y * xi2 * cos(2 * b - c) + 2 * z * c2c + 2 * z * eta2 * c2c + 2 * z * xi2 * c2c + z * cos(2 * (b + c)) + z * eta2 * cos(2 * (b + c)) + z * xi2 * cos(2 * (b + c)) + 2 * y * cos(2 * b + c) + 2 * y * eta2 * cos(2 * b + c) + 2 * y * xi2 * cos(2 * b + c)) *
                                           w3 +
                                       (-2 + 6 * eta2 + 6 * xi2 -
                                        2 * (1 + eta2 + xi2) * c2b +
                                        (1 + eta2 + xi2) * cos(2 * b - 2 * c) +
                                        2 * c2c + 2 * eta2 * c2c +
                                        2 * xi2 * c2c + cos(2 * (b + c)) +
                                        eta2 * cos(2 * (b + c)) +
                                        xi2 * cos(2 * (b + c))) *
                                           Power(w3, 2) -
                                       8 * w2 *
                                           (y * (1 + eta2 + xi2) *
                                                Power(cb, 2) +
                                            (y * (-3 + eta2 + xi2 +
                                                  (1 + eta2 + xi2) * c2b)) /
                                                2. -
                                            z * (1 + eta2 + xi2) * s2b * sc +
                                            (1 + eta2 + xi2) * s2b * sc *
                                                w3)))) /
                              (2. * (eta2 + xi2) *
                               (y * eta * cb + z * xi * cc - z * eta * sb * sc -
                                eta * cb * w2 +
                                (-(xi * cc) + eta * sb * sc) * w3) *
                               (4 * y2 + 6 * z2 + 4 * y2 * c2b - 2 * z2 * c2b +
                                z2 * cos(2 * b - 2 * c) -
                                4 * y * z * cos(2 * b - c) + 2 * z2 * c2c +
                                z2 * cos(2 * (b + c)) +
                                4 * y * z * cos(2 * b + c) +
                                8 * Power(cb, 2) * w2_2 -
                                2 * (6 * z - 2 * z * c2b + z * cos(2 * b - 2 * c) - 2 * y * cos(2 * b - c) + 2 * z * c2c + z * cos(2 * (b + c)) + 2 * y * cos(2 * b + c)) *
                                    w3 +
                                (6 - 2 * c2b + cos(2 * b - 2 * c) + 2 * c2c +
                                 cos(2 * (b + c))) *
                                    w3_2 -
                                16 * cb * w2 *
                                    (y * cb - z * sb * sc + sb * sc * w3))));
  return {std::make_pair(x1, a1), std::make_pair(x2, a2)};
}

std::vector<std::pair<float, float>> ComputeYB(
    const Vector6f& v, const Vector4f& essential_parameters,
    const Vector2f& free_parameters) {
  const float c = v(2);
  const float a = v(0);
  const float z = v(3);
  const float x = v(4);
  const float w1 = free_parameters(0);
  const float w2 = free_parameters(1);
  const float w3 = essential_parameters(0);
  const float xi = essential_parameters(1) / essential_parameters(3);
  const float eta = essential_parameters(2) / essential_parameters(3);
  const float eta2 = Power(eta, 2);
  const float eta3 = Power(eta, 3);
  const float xi2 = Power(xi, 2);
  const float xi3 = Power(xi, 3);
  const float c2c = cos(2 * c);
  const float c2a = cos(2 * a);
  const float s2c = sin(2 * c);
  const float s2a = sin(2 * a);
  const float z2 = Power(z, 2);
  const float x2 = Power(x, 2);
  const float w1_2 = Power(w1, 2);
  const float w3_2 = Power(w3, 2);
  const float ca = cos(a);
  const float cc = cos(c);
  const float sa = sin(a);
  const float sc = sin(c);

  const float y1 =
      (x * z * eta * cos(a - 3 * c) + x * z * eta3 * cos(a - 3 * c) +
       x * z * eta * xi2 * cos(a - 3 * c) + x * z * eta * cos(a - c) +
       x * z * eta3 * cos(a - c) + x * z * eta * xi2 * cos(a - c) -
       x * z * eta * cos(a + c) - x * z * eta3 * cos(a + c) -
       x * z * eta * xi2 * cos(a + c) - x * z * eta * cos(a + 3 * c) -
       x * z * eta3 * cos(a + 3 * c) - x * z * eta * xi2 * cos(a + 3 * c) -
       x * z * xi * sin(a - 3 * c) - x * z * eta2 * xi * sin(a - 3 * c) -
       x * z * xi3 * sin(a - 3 * c) - x * z * xi * sin(a - c) -
       x * z * eta2 * xi * sin(a - c) - x * z * xi3 * sin(a - c) +
       4 * x2 * s2c + 4 * x2 * eta2 * s2c + 4 * x2 * xi2 * s2c +
       x * z * xi * sin(a + c) + x * z * eta2 * xi * sin(a + c) +
       x * z * xi3 * sin(a + c) + x * z * xi * sin(a + 3 * c) +
       x * z * eta2 * xi * sin(a + 3 * c) + x * z * xi3 * sin(a + 3 * c) +
       4 * (1 + eta2 + xi2) * s2c * w1_2 - x * eta * cos(a - 3 * c) * w3 -
       x * eta3 * cos(a - 3 * c) * w3 - x * eta * xi2 * cos(a - 3 * c) * w3 -
       x * eta * cos(a - c) * w3 - x * eta3 * cos(a - c) * w3 -
       x * eta * xi2 * cos(a - c) * w3 + x * eta * cos(a + c) * w3 +
       x * eta3 * cos(a + c) * w3 + x * eta * xi2 * cos(a + c) * w3 +
       x * eta * cos(a + 3 * c) * w3 + x * eta3 * cos(a + 3 * c) * w3 +
       x * eta * xi2 * cos(a + 3 * c) * w3 + x * xi * sin(a - 3 * c) * w3 +
       x * eta2 * xi * sin(a - 3 * c) * w3 + x * xi3 * sin(a - 3 * c) * w3 +
       x * xi * sin(a - c) * w3 + x * eta2 * xi * sin(a - c) * w3 +
       x * xi3 * sin(a - c) * w3 - x * xi * sin(a + c) * w3 -
       x * eta2 * xi * sin(a + c) * w3 - x * xi3 * sin(a + c) * w3 -
       x * xi * sin(a + 3 * c) * w3 - x * eta2 * xi * sin(a + 3 * c) * w3 -
       x * xi3 * sin(a + 3 * c) * w3 +
       4 *
           (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
            2 * eta * xi * s2a) *
           w2 *
           (x + z * xi * ca * cc + z * eta * cc * sa -
            cc * (xi * ca + eta * sa) * w3) -
       4 * Sqrt(2) * eta * ca *
           Sqrt(Power(x + z * xi * ca * cc + z * eta * cc * sa - w1 -
                          cc * (xi * ca + eta * sa) * w3,
                      2) *
                (2 * x2 + z2 + x2 * eta2 + x2 * xi2 - x2 * eta2 * c2a -
                 z2 * eta2 * c2a + x2 * xi2 * c2a + z2 * xi2 * c2a + z2 * c2c +
                 z2 * eta2 * c2c + z2 * xi2 * c2c + 2 * x2 * eta * xi * s2a +
                 2 * z2 * eta * xi * s2a +
                 2 * x *
                     (-2 - eta2 - xi2 + (eta2 - xi2) * c2a -
                      2 * eta * xi * s2a) *
                     w1 +
                 (2 + eta2 + xi2 + (-eta2 + xi2) * c2a + 2 * eta * xi * s2a) *
                     w1_2 -
                 2 * z *
                     (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                      2 * eta * xi * s2a) *
                     w3 +
                 (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                  2 * eta * xi * s2a) *
                     w3_2)) +
       4 * Sqrt(2) * xi * sa *
           Sqrt(Power(x + z * xi * ca * cc + z * eta * cc * sa - w1 -
                          cc * (xi * ca + eta * sa) * w3,
                      2) *
                (2 * x2 + z2 + x2 * eta2 + x2 * xi2 - x2 * eta2 * c2a -
                 z2 * eta2 * c2a + x2 * xi2 * c2a + z2 * xi2 * c2a + z2 * c2c +
                 z2 * eta2 * c2c + z2 * xi2 * c2c + 2 * x2 * eta * xi * s2a +
                 2 * z2 * eta * xi * s2a +
                 2 * x *
                     (-2 - eta2 - xi2 + (eta2 - xi2) * c2a -
                      2 * eta * xi * s2a) *
                     w1 +
                 (2 + eta2 + xi2 + (-eta2 + xi2) * c2a + 2 * eta * xi * s2a) *
                     w1_2 -
                 2 * z *
                     (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                      2 * eta * xi * s2a) *
                     w3 +
                 (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                  2 * eta * xi * s2a) *
                     w3_2)) -
       w1 * (4 *
                 (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                  2 * eta * xi * s2a) *
                 w2 +
             4 * (1 + eta2 + xi2) * s2c *
                 (2 * x + z * xi * ca * cc + z * eta * cc * sa -
                  cc * (xi * ca + eta * sa) * w3))) /
      (4. *
       (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c + 2 * eta * xi * s2a) *
       (x + z * xi * ca * cc + z * eta * cc * sa - w1 -
        cc * (xi * ca + eta * sa) * w3));
  const float b1 = (ArcTan(
      (-(z2 * xi * cos(a - 2 * c)) - x * z * eta2 * cos(2 * a - c) +
       x * z * xi2 * cos(2 * a - c) + x * z * eta2 * cos(2 * a + c) -
       x * z * xi2 * cos(2 * a + c) + z2 * xi * cos(a + 2 * c) -
       z2 * eta * sin(a - 2 * c) + 2 * x * z * eta * xi * sin(2 * a - c) -
       2 * x * z * eta * xi * sin(2 * a + c) + z2 * eta * sin(a + 2 * c) +
       2 * (2 * eta * xi * c2a + (eta2 - xi2) * s2a) * sc * w1 * (z - w3) +
       4 * (x * xi * ca - 2 * z * cc + x * eta * sa) * (eta * ca - xi * sa) *
           sc * w3 +
       2 * (eta * ca - xi * sa) * s2c * w3_2 -
       2 * Sqrt(2) *
           Sqrt(Power(x + z * xi * ca * cc + z * eta * cc * sa - w1 -
                          cc * (xi * ca + eta * sa) * w3,
                      2) *
                (2 * x2 + z2 + x2 * eta2 + x2 * xi2 - x2 * eta2 * c2a -
                 z2 * eta2 * c2a + x2 * xi2 * c2a + z2 * xi2 * c2a + z2 * c2c +
                 z2 * eta2 * c2c + z2 * xi2 * c2c + 2 * x2 * eta * xi * s2a +
                 2 * z2 * eta * xi * s2a +
                 2 * x *
                     (-2 - eta2 - xi2 + (eta2 - xi2) * c2a -
                      2 * eta * xi * s2a) *
                     w1 +
                 (2 + eta2 + xi2 + (-eta2 + xi2) * c2a + 2 * eta * xi * s2a) *
                     w1_2 -
                 2 * z *
                     (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                      2 * eta * xi * s2a) *
                     w3 +
                 (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                  2 * eta * xi * s2a) *
                     w3_2))) /
          ((2 + eta2 + xi2 + (-eta2 + xi2) * c2a + 2 * eta * xi * s2a) *
           (2 * x2 + z2 + z2 * c2c - 4 * x * w1 + 2 * w1_2 -
            4 * z * Power(cc, 2) * w3 + 2 * Power(cc, 2) * w3_2)),
      (-(Power(z, 3) * eta2 * xi * cos(a - 3 * c)) -
       Power(z, 3) * xi3 * cos(a - 3 * c) +
       3 * Power(z, 3) * eta2 * xi * cos(3 * a - 3 * c) -
       Power(z, 3) * xi3 * cos(3 * a - 3 * c) +
       8 * x * z2 * eta2 * cos(2 * a - 2 * c) -
       8 * x * z2 * xi2 * cos(2 * a - 2 * c) - 16 * x2 * z * xi * cos(a - c) -
       Power(z, 3) * eta2 * xi * cos(a - c) - Power(z, 3) * xi3 * cos(a - c) +
       3 * Power(z, 3) * eta2 * xi * cos(3 * a - c) -
       Power(z, 3) * xi3 * cos(3 * a - c) + 16 * x2 * z * xi * cos(a + c) +
       Power(z, 3) * eta2 * xi * cos(a + c) + Power(z, 3) * xi3 * cos(a + c) -
       8 * x * z2 * eta2 * cos(2 * (a + c)) +
       8 * x * z2 * xi2 * cos(2 * (a + c)) -
       3 * Power(z, 3) * eta2 * xi * cos(3 * (a + c)) +
       Power(z, 3) * xi3 * cos(3 * (a + c)) -
       3 * Power(z, 3) * eta2 * xi * cos(3 * a + c) +
       Power(z, 3) * xi3 * cos(3 * a + c) +
       Power(z, 3) * eta2 * xi * cos(a + 3 * c) +
       Power(z, 3) * xi3 * cos(a + 3 * c) -
       Power(z, 3) * eta3 * sin(a - 3 * c) -
       Power(z, 3) * eta * xi2 * sin(a - 3 * c) +
       Power(z, 3) * eta3 * sin(3 * a - 3 * c) -
       3 * Power(z, 3) * eta * xi2 * sin(3 * a - 3 * c) -
       16 * x * z2 * eta * xi * sin(2 * a - 2 * c) -
       16 * x2 * z * eta * sin(a - c) - Power(z, 3) * eta3 * sin(a - c) -
       Power(z, 3) * eta * xi2 * sin(a - c) +
       Power(z, 3) * eta3 * sin(3 * a - c) -
       3 * Power(z, 3) * eta * xi2 * sin(3 * a - c) +
       16 * x2 * z * eta * sin(a + c) + Power(z, 3) * eta3 * sin(a + c) +
       Power(z, 3) * eta * xi2 * sin(a + c) +
       16 * x * z2 * eta * xi * sin(2 * (a + c)) -
       Power(z, 3) * eta3 * sin(3 * (a + c)) +
       3 * Power(z, 3) * eta * xi2 * sin(3 * (a + c)) -
       Power(z, 3) * eta3 * sin(3 * a + c) +
       3 * Power(z, 3) * eta * xi2 * sin(3 * a + c) +
       Power(z, 3) * eta3 * sin(a + 3 * c) +
       Power(z, 3) * eta * xi2 * sin(a + 3 * c) +
       32 * (eta * ca - xi * sa) * sc * w1_2 * (z - w3) +
       4 * (4 * x + 6 * z * xi * ca * cc + 6 * z * eta * cc * sa) *
           (2 * eta * xi * c2a + (eta2 - xi2) * s2a) * s2c * w3_2 +
       32 * Power(cc, 2) * Power(xi * ca + eta * sa, 2) *
           (-(eta * ca) + xi * sa) * sc * Power(w3, 3) -
       16 * Sqrt(2) * x * xi * ca *
           Sqrt(Power(x + z * xi * ca * cc + z * eta * cc * sa - w1 -
                          cc * (xi * ca + eta * sa) * w3,
                      2) *
                (2 * x2 + z2 + x2 * eta2 + x2 * xi2 - x2 * eta2 * c2a -
                 z2 * eta2 * c2a + x2 * xi2 * c2a + z2 * xi2 * c2a + z2 * c2c +
                 z2 * eta2 * c2c + z2 * xi2 * c2c + 2 * x2 * eta * xi * s2a +
                 2 * z2 * eta * xi * s2a +
                 2 * x *
                     (-2 - eta2 - xi2 + (eta2 - xi2) * c2a -
                      2 * eta * xi * s2a) *
                     w1 +
                 (2 + eta2 + xi2 + (-eta2 + xi2) * c2a + 2 * eta * xi * s2a) *
                     w1_2 -
                 2 * z *
                     (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                      2 * eta * xi * s2a) *
                     w3 +
                 (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                  2 * eta * xi * s2a) *
                     w3_2)) +
       16 * Sqrt(2) * z * cc *
           Sqrt(Power(x + z * xi * ca * cc + z * eta * cc * sa - w1 -
                          cc * (xi * ca + eta * sa) * w3,
                      2) *
                (2 * x2 + z2 + x2 * eta2 + x2 * xi2 - x2 * eta2 * c2a -
                 z2 * eta2 * c2a + x2 * xi2 * c2a + z2 * xi2 * c2a + z2 * c2c +
                 z2 * eta2 * c2c + z2 * xi2 * c2c + 2 * x2 * eta * xi * s2a +
                 2 * z2 * eta * xi * s2a +
                 2 * x *
                     (-2 - eta2 - xi2 + (eta2 - xi2) * c2a -
                      2 * eta * xi * s2a) *
                     w1 +
                 (2 + eta2 + xi2 + (-eta2 + xi2) * c2a + 2 * eta * xi * s2a) *
                     w1_2 -
                 2 * z *
                     (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                      2 * eta * xi * s2a) *
                     w3 +
                 (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                  2 * eta * xi * s2a) *
                     w3_2)) -
       16 * Sqrt(2) * x * eta * sa *
           Sqrt(Power(x + z * xi * ca * cc + z * eta * cc * sa - w1 -
                          cc * (xi * ca + eta * sa) * w3,
                      2) *
                (2 * x2 + z2 + x2 * eta2 + x2 * xi2 - x2 * eta2 * c2a -
                 z2 * eta2 * c2a + x2 * xi2 * c2a + z2 * xi2 * c2a + z2 * c2c +
                 z2 * eta2 * c2c + z2 * xi2 * c2c + 2 * x2 * eta * xi * s2a +
                 2 * z2 * eta * xi * s2a +
                 2 * x *
                     (-2 - eta2 - xi2 + (eta2 - xi2) * c2a -
                      2 * eta * xi * s2a) *
                     w1 +
                 (2 + eta2 + xi2 + (-eta2 + xi2) * c2a + 2 * eta * xi * s2a) *
                     w1_2 -
                 2 * z *
                     (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                      2 * eta * xi * s2a) *
                     w3 +
                 (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                  2 * eta * xi * s2a) *
                     w3_2)) +
       w3 * (3 * z2 * eta2 * xi * cos(a - 3 * c) +
             3 * z2 * xi3 * cos(a - 3 * c) -
             9 * z2 * eta2 * xi * cos(3 * a - 3 * c) +
             3 * z2 * xi3 * cos(3 * a - 3 * c) -
             16 * x * z * eta2 * cos(2 * a - 2 * c) +
             16 * x * z * xi2 * cos(2 * a - 2 * c) + 16 * x2 * xi * cos(a - c) +
             3 * z2 * eta2 * xi * cos(a - c) + 3 * z2 * xi3 * cos(a - c) -
             9 * z2 * eta2 * xi * cos(3 * a - c) +
             3 * z2 * xi3 * cos(3 * a - c) - 16 * x2 * xi * cos(a + c) -
             3 * z2 * eta2 * xi * cos(a + c) - 3 * z2 * xi3 * cos(a + c) +
             16 * x * z * eta2 * cos(2 * (a + c)) -
             16 * x * z * xi2 * cos(2 * (a + c)) +
             9 * z2 * eta2 * xi * cos(3 * (a + c)) -
             3 * z2 * xi3 * cos(3 * (a + c)) +
             9 * z2 * eta2 * xi * cos(3 * a + c) -
             3 * z2 * xi3 * cos(3 * a + c) -
             3 * z2 * eta2 * xi * cos(a + 3 * c) -
             3 * z2 * xi3 * cos(a + 3 * c) + 3 * z2 * eta3 * sin(a - 3 * c) +
             3 * z2 * eta * xi2 * sin(a - 3 * c) -
             3 * z2 * eta3 * sin(3 * a - 3 * c) +
             9 * z2 * eta * xi2 * sin(3 * a - 3 * c) +
             32 * x * z * eta * xi * sin(2 * a - 2 * c) +
             16 * x2 * eta * sin(a - c) + 3 * z2 * eta3 * sin(a - c) +
             3 * z2 * eta * xi2 * sin(a - c) - 3 * z2 * eta3 * sin(3 * a - c) +
             9 * z2 * eta * xi2 * sin(3 * a - c) - 16 * x2 * eta * sin(a + c) -
             3 * z2 * eta3 * sin(a + c) - 3 * z2 * eta * xi2 * sin(a + c) -
             32 * x * z * eta * xi * sin(2 * (a + c)) +
             3 * z2 * eta3 * sin(3 * (a + c)) -
             9 * z2 * eta * xi2 * sin(3 * (a + c)) +
             3 * z2 * eta3 * sin(3 * a + c) -
             9 * z2 * eta * xi2 * sin(3 * a + c) -
             3 * z2 * eta3 * sin(a + 3 * c) -
             3 * z2 * eta * xi2 * sin(a + 3 * c) -
             16 * Sqrt(2) * cc *
                 Sqrt(Power(x + z * xi * ca * cc + z * eta * cc * sa - w1 -
                                cc * (xi * ca + eta * sa) * w3,
                            2) *
                      (2 * x2 + z2 + x2 * eta2 + x2 * xi2 - x2 * eta2 * c2a -
                       z2 * eta2 * c2a + x2 * xi2 * c2a + z2 * xi2 * c2a +
                       z2 * c2c + z2 * eta2 * c2c + z2 * xi2 * c2c +
                       2 * x2 * eta * xi * s2a + 2 * z2 * eta * xi * s2a +
                       2 * x *
                           (-2 - eta2 - xi2 + (eta2 - xi2) * c2a -
                            2 * eta * xi * s2a) *
                           w1 +
                       (2 + eta2 + xi2 + (-eta2 + xi2) * c2a +
                        2 * eta * xi * s2a) *
                           w1_2 -
                       2 * z *
                           (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                            2 * eta * xi * s2a) *
                           w3 +
                       (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                        2 * eta * xi * s2a) *
                           w3_2))) -
       8 * w1 *
           (z2 * eta2 * cos(2 * a - 2 * c) - z2 * xi2 * cos(2 * a - 2 * c) -
            4 * x * z * xi * cos(a - c) + 4 * x * z * xi * cos(a + c) -
            z2 * eta2 * cos(2 * (a + c)) + z2 * xi2 * cos(2 * (a + c)) -
            2 * z2 * eta * xi * sin(2 * a - 2 * c) -
            4 * x * z * eta * sin(a - c) + 4 * x * z * eta * sin(a + c) +
            2 * z2 * eta * xi * sin(2 * (a + c)) -
            8 * (eta * ca - xi * sa) *
                (x + 2 * z * xi * ca * cc + 2 * z * eta * cc * sa) * sc * w3 +
            2 * (2 * eta * xi * c2a + (eta2 - xi2) * s2a) * s2c * w3_2 -
            2 * Sqrt(2) * xi * ca *
                Sqrt(Power(x + z * xi * ca * cc + z * eta * cc * sa - w1 -
                               cc * (xi * ca + eta * sa) * w3,
                           2) *
                     (2 * x2 + z2 + x2 * eta2 + x2 * xi2 - x2 * eta2 * c2a -
                      z2 * eta2 * c2a + x2 * xi2 * c2a + z2 * xi2 * c2a +
                      z2 * c2c + z2 * eta2 * c2c + z2 * xi2 * c2c +
                      2 * x2 * eta * xi * s2a + 2 * z2 * eta * xi * s2a +
                      2 * x *
                          (-2 - eta2 - xi2 + (eta2 - xi2) * c2a -
                           2 * eta * xi * s2a) *
                          w1 +
                      (2 + eta2 + xi2 + (-eta2 + xi2) * c2a +
                       2 * eta * xi * s2a) *
                          w1_2 -
                      2 * z *
                          (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                           2 * eta * xi * s2a) *
                          w3 +
                      (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                       2 * eta * xi * s2a) *
                          w3_2)) -
            2 * Sqrt(2) * eta * sa *
                Sqrt(Power(x + z * xi * ca * cc + z * eta * cc * sa - w1 -
                               cc * (xi * ca + eta * sa) * w3,
                           2) *
                     (2 * x2 + z2 + x2 * eta2 + x2 * xi2 - x2 * eta2 * c2a -
                      z2 * eta2 * c2a + x2 * xi2 * c2a + z2 * xi2 * c2a +
                      z2 * c2c + z2 * eta2 * c2c + z2 * xi2 * c2c +
                      2 * x2 * eta * xi * s2a + 2 * z2 * eta * xi * s2a +
                      2 * x *
                          (-2 - eta2 - xi2 + (eta2 - xi2) * c2a -
                           2 * eta * xi * s2a) *
                          w1 +
                      (2 + eta2 + xi2 + (-eta2 + xi2) * c2a +
                       2 * eta * xi * s2a) *
                          w1_2 -
                      2 * z *
                          (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                           2 * eta * xi * s2a) *
                          w3 +
                      (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                       2 * eta * xi * s2a) *
                          w3_2)))) /
          (8. * (2 + eta2 + xi2 + (-eta2 + xi2) * c2a + 2 * eta * xi * s2a) *
           (x + z * xi * ca * cc + z * eta * cc * sa - w1 -
            cc * (xi * ca + eta * sa) * w3) *
           (2 * x2 + z2 + z2 * c2c - 4 * x * w1 + 2 * w1_2 -
            4 * z * Power(cc, 2) * w3 + 2 * Power(cc, 2) * w3_2))));
  const float y2 =
      -(-(x * z * eta * cos(a - 3 * c)) - x * z * eta3 * cos(a - 3 * c) -
        x * z * eta * xi2 * cos(a - 3 * c) - x * z * eta * cos(a - c) -
        x * z * eta3 * cos(a - c) - x * z * eta * xi2 * cos(a - c) +
        x * z * eta * cos(a + c) + x * z * eta3 * cos(a + c) +
        x * z * eta * xi2 * cos(a + c) + x * z * eta * cos(a + 3 * c) +
        x * z * eta3 * cos(a + 3 * c) + x * z * eta * xi2 * cos(a + 3 * c) +
        x * z * xi * sin(a - 3 * c) + x * z * eta2 * xi * sin(a - 3 * c) +
        x * z * xi3 * sin(a - 3 * c) + x * z * xi * sin(a - c) +
        x * z * eta2 * xi * sin(a - c) + x * z * xi3 * sin(a - c) -
        4 * x2 * s2c - 4 * x2 * eta2 * s2c - 4 * x2 * xi2 * s2c -
        x * z * xi * sin(a + c) - x * z * eta2 * xi * sin(a + c) -
        x * z * xi3 * sin(a + c) - x * z * xi * sin(a + 3 * c) -
        x * z * eta2 * xi * sin(a + 3 * c) - x * z * xi3 * sin(a + 3 * c) -
        4 * (1 + eta2 + xi2) * s2c * w1_2 + x * eta * cos(a - 3 * c) * w3 +
        x * eta3 * cos(a - 3 * c) * w3 + x * eta * xi2 * cos(a - 3 * c) * w3 +
        x * eta * cos(a - c) * w3 + x * eta3 * cos(a - c) * w3 +
        x * eta * xi2 * cos(a - c) * w3 - x * eta * cos(a + c) * w3 -
        x * eta3 * cos(a + c) * w3 - x * eta * xi2 * cos(a + c) * w3 -
        x * eta * cos(a + 3 * c) * w3 - x * eta3 * cos(a + 3 * c) * w3 -
        x * eta * xi2 * cos(a + 3 * c) * w3 - x * xi * sin(a - 3 * c) * w3 -
        x * eta2 * xi * sin(a - 3 * c) * w3 - x * xi3 * sin(a - 3 * c) * w3 -
        x * xi * sin(a - c) * w3 - x * eta2 * xi * sin(a - c) * w3 -
        x * xi3 * sin(a - c) * w3 + x * xi * sin(a + c) * w3 +
        x * eta2 * xi * sin(a + c) * w3 + x * xi3 * sin(a + c) * w3 +
        x * xi * sin(a + 3 * c) * w3 + x * eta2 * xi * sin(a + 3 * c) * w3 +
        x * xi3 * sin(a + 3 * c) * w3 -
        4 *
            (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
             2 * eta * xi * s2a) *
            w2 *
            (x + z * xi * ca * cc + z * eta * cc * sa -
             cc * (xi * ca + eta * sa) * w3) -
        4 * Sqrt(2) * eta * ca *
            Sqrt(Power(x + z * xi * ca * cc + z * eta * cc * sa - w1 -
                           cc * (xi * ca + eta * sa) * w3,
                       2) *
                 (2 * x2 + z2 + x2 * eta2 + x2 * xi2 - x2 * eta2 * c2a -
                  z2 * eta2 * c2a + x2 * xi2 * c2a + z2 * xi2 * c2a + z2 * c2c +
                  z2 * eta2 * c2c + z2 * xi2 * c2c + 2 * x2 * eta * xi * s2a +
                  2 * z2 * eta * xi * s2a +
                  2 * x *
                      (-2 - eta2 - xi2 + (eta2 - xi2) * c2a -
                       2 * eta * xi * s2a) *
                      w1 +
                  (2 + eta2 + xi2 + (-eta2 + xi2) * c2a + 2 * eta * xi * s2a) *
                      w1_2 -
                  2 * z *
                      (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                       2 * eta * xi * s2a) *
                      w3 +
                  (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                   2 * eta * xi * s2a) *
                      w3_2)) +
        4 * Sqrt(2) * xi * sa *
            Sqrt(Power(x + z * xi * ca * cc + z * eta * cc * sa - w1 -
                           cc * (xi * ca + eta * sa) * w3,
                       2) *
                 (2 * x2 + z2 + x2 * eta2 + x2 * xi2 - x2 * eta2 * c2a -
                  z2 * eta2 * c2a + x2 * xi2 * c2a + z2 * xi2 * c2a + z2 * c2c +
                  z2 * eta2 * c2c + z2 * xi2 * c2c + 2 * x2 * eta * xi * s2a +
                  2 * z2 * eta * xi * s2a +
                  2 * x *
                      (-2 - eta2 - xi2 + (eta2 - xi2) * c2a -
                       2 * eta * xi * s2a) *
                      w1 +
                  (2 + eta2 + xi2 + (-eta2 + xi2) * c2a + 2 * eta * xi * s2a) *
                      w1_2 -
                  2 * z *
                      (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                       2 * eta * xi * s2a) *
                      w3 +
                  (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                   2 * eta * xi * s2a) *
                      w3_2)) +
        4 * w1 *
            ((1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
              2 * eta * xi * s2a) *
                 w2 +
             (1 + eta2 + xi2) * s2c *
                 (2 * x + z * xi * ca * cc + z * eta * cc * sa -
                  cc * (xi * ca + eta * sa) * w3))) /
      (4. *
       (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c + 2 * eta * xi * s2a) *
       (x + z * xi * ca * cc + z * eta * cc * sa - w1 -
        cc * (xi * ca + eta * sa) * w3));
  const float b2 = (ArcTan(
      (-(z2 * xi * cos(a - 2 * c)) - x * z * eta2 * cos(2 * a - c) +
       x * z * xi2 * cos(2 * a - c) + x * z * eta2 * cos(2 * a + c) -
       x * z * xi2 * cos(2 * a + c) + z2 * xi * cos(a + 2 * c) -
       z2 * eta * sin(a - 2 * c) + 2 * x * z * eta * xi * sin(2 * a - c) -
       2 * x * z * eta * xi * sin(2 * a + c) + z2 * eta * sin(a + 2 * c) +
       2 * (2 * eta * xi * c2a + (eta2 - xi2) * s2a) * sc * w1 * (z - w3) +
       4 * (x * xi * ca - 2 * z * cc + x * eta * sa) * (eta * ca - xi * sa) *
           sc * w3 +
       2 * (eta * ca - xi * sa) * s2c * w3_2 +
       2 * Sqrt(2) *
           Sqrt(Power(x + z * xi * ca * cc + z * eta * cc * sa - w1 -
                          cc * (xi * ca + eta * sa) * w3,
                      2) *
                (2 * x2 + z2 + x2 * eta2 + x2 * xi2 - x2 * eta2 * c2a -
                 z2 * eta2 * c2a + x2 * xi2 * c2a + z2 * xi2 * c2a + z2 * c2c +
                 z2 * eta2 * c2c + z2 * xi2 * c2c + 2 * x2 * eta * xi * s2a +
                 2 * z2 * eta * xi * s2a +
                 2 * x *
                     (-2 - eta2 - xi2 + (eta2 - xi2) * c2a -
                      2 * eta * xi * s2a) *
                     w1 +
                 (2 + eta2 + xi2 + (-eta2 + xi2) * c2a + 2 * eta * xi * s2a) *
                     w1_2 -
                 2 * z *
                     (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                      2 * eta * xi * s2a) *
                     w3 +
                 (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                  2 * eta * xi * s2a) *
                     w3_2))) /
          ((2 + eta2 + xi2 + (-eta2 + xi2) * c2a + 2 * eta * xi * s2a) *
           (2 * x2 + z2 + z2 * c2c - 4 * x * w1 + 2 * w1_2 -
            4 * z * Power(cc, 2) * w3 + 2 * Power(cc, 2) * w3_2)),
      (-(Power(z, 3) * eta2 * xi * cos(a - 3 * c)) -
       Power(z, 3) * xi3 * cos(a - 3 * c) +
       3 * Power(z, 3) * eta2 * xi * cos(3 * a - 3 * c) -
       Power(z, 3) * xi3 * cos(3 * a - 3 * c) +
       8 * x * z2 * eta2 * cos(2 * a - 2 * c) -
       8 * x * z2 * xi2 * cos(2 * a - 2 * c) - 16 * x2 * z * xi * cos(a - c) -
       Power(z, 3) * eta2 * xi * cos(a - c) - Power(z, 3) * xi3 * cos(a - c) +
       3 * Power(z, 3) * eta2 * xi * cos(3 * a - c) -
       Power(z, 3) * xi3 * cos(3 * a - c) + 16 * x2 * z * xi * cos(a + c) +
       Power(z, 3) * eta2 * xi * cos(a + c) + Power(z, 3) * xi3 * cos(a + c) -
       8 * x * z2 * eta2 * cos(2 * (a + c)) +
       8 * x * z2 * xi2 * cos(2 * (a + c)) -
       3 * Power(z, 3) * eta2 * xi * cos(3 * (a + c)) +
       Power(z, 3) * xi3 * cos(3 * (a + c)) -
       3 * Power(z, 3) * eta2 * xi * cos(3 * a + c) +
       Power(z, 3) * xi3 * cos(3 * a + c) +
       Power(z, 3) * eta2 * xi * cos(a + 3 * c) +
       Power(z, 3) * xi3 * cos(a + 3 * c) -
       Power(z, 3) * eta3 * sin(a - 3 * c) -
       Power(z, 3) * eta * xi2 * sin(a - 3 * c) +
       Power(z, 3) * eta3 * sin(3 * a - 3 * c) -
       3 * Power(z, 3) * eta * xi2 * sin(3 * a - 3 * c) -
       16 * x * z2 * eta * xi * sin(2 * a - 2 * c) -
       16 * x2 * z * eta * sin(a - c) - Power(z, 3) * eta3 * sin(a - c) -
       Power(z, 3) * eta * xi2 * sin(a - c) +
       Power(z, 3) * eta3 * sin(3 * a - c) -
       3 * Power(z, 3) * eta * xi2 * sin(3 * a - c) +
       16 * x2 * z * eta * sin(a + c) + Power(z, 3) * eta3 * sin(a + c) +
       Power(z, 3) * eta * xi2 * sin(a + c) +
       16 * x * z2 * eta * xi * sin(2 * (a + c)) -
       Power(z, 3) * eta3 * sin(3 * (a + c)) +
       3 * Power(z, 3) * eta * xi2 * sin(3 * (a + c)) -
       Power(z, 3) * eta3 * sin(3 * a + c) +
       3 * Power(z, 3) * eta * xi2 * sin(3 * a + c) +
       Power(z, 3) * eta3 * sin(a + 3 * c) +
       Power(z, 3) * eta * xi2 * sin(a + 3 * c) +
       32 * (eta * ca - xi * sa) * sc * w1_2 * (z - w3) +
       4 * (4 * x + 6 * z * xi * ca * cc + 6 * z * eta * cc * sa) *
           (2 * eta * xi * c2a + (eta2 - xi2) * s2a) * s2c * w3_2 +
       32 * Power(cc, 2) * Power(xi * ca + eta * sa, 2) *
           (-(eta * ca) + xi * sa) * sc * Power(w3, 3) +
       16 * Sqrt(2) * x * xi * ca *
           Sqrt(Power(x + z * xi * ca * cc + z * eta * cc * sa - w1 -
                          cc * (xi * ca + eta * sa) * w3,
                      2) *
                (2 * x2 + z2 + x2 * eta2 + x2 * xi2 - x2 * eta2 * c2a -
                 z2 * eta2 * c2a + x2 * xi2 * c2a + z2 * xi2 * c2a + z2 * c2c +
                 z2 * eta2 * c2c + z2 * xi2 * c2c + 2 * x2 * eta * xi * s2a +
                 2 * z2 * eta * xi * s2a +
                 2 * x *
                     (-2 - eta2 - xi2 + (eta2 - xi2) * c2a -
                      2 * eta * xi * s2a) *
                     w1 +
                 (2 + eta2 + xi2 + (-eta2 + xi2) * c2a + 2 * eta * xi * s2a) *
                     w1_2 -
                 2 * z *
                     (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                      2 * eta * xi * s2a) *
                     w3 +
                 (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                  2 * eta * xi * s2a) *
                     w3_2)) -
       16 * Sqrt(2) * z * cc *
           Sqrt(Power(x + z * xi * ca * cc + z * eta * cc * sa - w1 -
                          cc * (xi * ca + eta * sa) * w3,
                      2) *
                (2 * x2 + z2 + x2 * eta2 + x2 * xi2 - x2 * eta2 * c2a -
                 z2 * eta2 * c2a + x2 * xi2 * c2a + z2 * xi2 * c2a + z2 * c2c +
                 z2 * eta2 * c2c + z2 * xi2 * c2c + 2 * x2 * eta * xi * s2a +
                 2 * z2 * eta * xi * s2a +
                 2 * x *
                     (-2 - eta2 - xi2 + (eta2 - xi2) * c2a -
                      2 * eta * xi * s2a) *
                     w1 +
                 (2 + eta2 + xi2 + (-eta2 + xi2) * c2a + 2 * eta * xi * s2a) *
                     w1_2 -
                 2 * z *
                     (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                      2 * eta * xi * s2a) *
                     w3 +
                 (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                  2 * eta * xi * s2a) *
                     w3_2)) +
       16 * Sqrt(2) * x * eta * sa *
           Sqrt(Power(x + z * xi * ca * cc + z * eta * cc * sa - w1 -
                          cc * (xi * ca + eta * sa) * w3,
                      2) *
                (2 * x2 + z2 + x2 * eta2 + x2 * xi2 - x2 * eta2 * c2a -
                 z2 * eta2 * c2a + x2 * xi2 * c2a + z2 * xi2 * c2a + z2 * c2c +
                 z2 * eta2 * c2c + z2 * xi2 * c2c + 2 * x2 * eta * xi * s2a +
                 2 * z2 * eta * xi * s2a +
                 2 * x *
                     (-2 - eta2 - xi2 + (eta2 - xi2) * c2a -
                      2 * eta * xi * s2a) *
                     w1 +
                 (2 + eta2 + xi2 + (-eta2 + xi2) * c2a + 2 * eta * xi * s2a) *
                     w1_2 -
                 2 * z *
                     (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                      2 * eta * xi * s2a) *
                     w3 +
                 (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                  2 * eta * xi * s2a) *
                     w3_2)) +
       w3 * (3 * z2 * eta2 * xi * cos(a - 3 * c) +
             3 * z2 * xi3 * cos(a - 3 * c) -
             9 * z2 * eta2 * xi * cos(3 * a - 3 * c) +
             3 * z2 * xi3 * cos(3 * a - 3 * c) -
             16 * x * z * eta2 * cos(2 * a - 2 * c) +
             16 * x * z * xi2 * cos(2 * a - 2 * c) + 16 * x2 * xi * cos(a - c) +
             3 * z2 * eta2 * xi * cos(a - c) + 3 * z2 * xi3 * cos(a - c) -
             9 * z2 * eta2 * xi * cos(3 * a - c) +
             3 * z2 * xi3 * cos(3 * a - c) - 16 * x2 * xi * cos(a + c) -
             3 * z2 * eta2 * xi * cos(a + c) - 3 * z2 * xi3 * cos(a + c) +
             16 * x * z * eta2 * cos(2 * (a + c)) -
             16 * x * z * xi2 * cos(2 * (a + c)) +
             9 * z2 * eta2 * xi * cos(3 * (a + c)) -
             3 * z2 * xi3 * cos(3 * (a + c)) +
             9 * z2 * eta2 * xi * cos(3 * a + c) -
             3 * z2 * xi3 * cos(3 * a + c) -
             3 * z2 * eta2 * xi * cos(a + 3 * c) -
             3 * z2 * xi3 * cos(a + 3 * c) + 3 * z2 * eta3 * sin(a - 3 * c) +
             3 * z2 * eta * xi2 * sin(a - 3 * c) -
             3 * z2 * eta3 * sin(3 * a - 3 * c) +
             9 * z2 * eta * xi2 * sin(3 * a - 3 * c) +
             32 * x * z * eta * xi * sin(2 * a - 2 * c) +
             16 * x2 * eta * sin(a - c) + 3 * z2 * eta3 * sin(a - c) +
             3 * z2 * eta * xi2 * sin(a - c) - 3 * z2 * eta3 * sin(3 * a - c) +
             9 * z2 * eta * xi2 * sin(3 * a - c) - 16 * x2 * eta * sin(a + c) -
             3 * z2 * eta3 * sin(a + c) - 3 * z2 * eta * xi2 * sin(a + c) -
             32 * x * z * eta * xi * sin(2 * (a + c)) +
             3 * z2 * eta3 * sin(3 * (a + c)) -
             9 * z2 * eta * xi2 * sin(3 * (a + c)) +
             3 * z2 * eta3 * sin(3 * a + c) -
             9 * z2 * eta * xi2 * sin(3 * a + c) -
             3 * z2 * eta3 * sin(a + 3 * c) -
             3 * z2 * eta * xi2 * sin(a + 3 * c) +
             16 * Sqrt(2) * cc *
                 Sqrt(Power(x + z * xi * ca * cc + z * eta * cc * sa - w1 -
                                cc * (xi * ca + eta * sa) * w3,
                            2) *
                      (2 * x2 + z2 + x2 * eta2 + x2 * xi2 - x2 * eta2 * c2a -
                       z2 * eta2 * c2a + x2 * xi2 * c2a + z2 * xi2 * c2a +
                       z2 * c2c + z2 * eta2 * c2c + z2 * xi2 * c2c +
                       2 * x2 * eta * xi * s2a + 2 * z2 * eta * xi * s2a +
                       2 * x *
                           (-2 - eta2 - xi2 + (eta2 - xi2) * c2a -
                            2 * eta * xi * s2a) *
                           w1 +
                       (2 + eta2 + xi2 + (-eta2 + xi2) * c2a +
                        2 * eta * xi * s2a) *
                           w1_2 -
                       2 * z *
                           (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                            2 * eta * xi * s2a) *
                           w3 +
                       (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                        2 * eta * xi * s2a) *
                           w3_2))) -
       8 * w1 *
           (z2 * eta2 * cos(2 * a - 2 * c) - z2 * xi2 * cos(2 * a - 2 * c) -
            4 * x * z * xi * cos(a - c) + 4 * x * z * xi * cos(a + c) -
            z2 * eta2 * cos(2 * (a + c)) + z2 * xi2 * cos(2 * (a + c)) -
            2 * z2 * eta * xi * sin(2 * a - 2 * c) -
            4 * x * z * eta * sin(a - c) + 4 * x * z * eta * sin(a + c) +
            2 * z2 * eta * xi * sin(2 * (a + c)) -
            8 * (eta * ca - xi * sa) *
                (x + 2 * z * xi * ca * cc + 2 * z * eta * cc * sa) * sc * w3 +
            2 * (2 * eta * xi * c2a + (eta2 - xi2) * s2a) * s2c * w3_2 +
            2 * Sqrt(2) * xi * ca *
                Sqrt(Power(x + z * xi * ca * cc + z * eta * cc * sa - w1 -
                               cc * (xi * ca + eta * sa) * w3,
                           2) *
                     (2 * x2 + z2 + x2 * eta2 + x2 * xi2 - x2 * eta2 * c2a -
                      z2 * eta2 * c2a + x2 * xi2 * c2a + z2 * xi2 * c2a +
                      z2 * c2c + z2 * eta2 * c2c + z2 * xi2 * c2c +
                      2 * x2 * eta * xi * s2a + 2 * z2 * eta * xi * s2a +
                      2 * x *
                          (-2 - eta2 - xi2 + (eta2 - xi2) * c2a -
                           2 * eta * xi * s2a) *
                          w1 +
                      (2 + eta2 + xi2 + (-eta2 + xi2) * c2a +
                       2 * eta * xi * s2a) *
                          w1_2 -
                      2 * z *
                          (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                           2 * eta * xi * s2a) *
                          w3 +
                      (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                       2 * eta * xi * s2a) *
                          w3_2)) +
            2 * Sqrt(2) * eta * sa *
                Sqrt(Power(x + z * xi * ca * cc + z * eta * cc * sa - w1 -
                               cc * (xi * ca + eta * sa) * w3,
                           2) *
                     (2 * x2 + z2 + x2 * eta2 + x2 * xi2 - x2 * eta2 * c2a -
                      z2 * eta2 * c2a + x2 * xi2 * c2a + z2 * xi2 * c2a +
                      z2 * c2c + z2 * eta2 * c2c + z2 * xi2 * c2c +
                      2 * x2 * eta * xi * s2a + 2 * z2 * eta * xi * s2a +
                      2 * x *
                          (-2 - eta2 - xi2 + (eta2 - xi2) * c2a -
                           2 * eta * xi * s2a) *
                          w1 +
                      (2 + eta2 + xi2 + (-eta2 + xi2) * c2a +
                       2 * eta * xi * s2a) *
                          w1_2 -
                      2 * z *
                          (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                           2 * eta * xi * s2a) *
                          w3 +
                      (1 + (-eta2 + xi2) * c2a + (1 + eta2 + xi2) * c2c +
                       2 * eta * xi * s2a) *
                          w3_2)))) /
          (8. * (2 + eta2 + xi2 + (-eta2 + xi2) * c2a + 2 * eta * xi * s2a) *
           (x + z * xi * ca * cc + z * eta * cc * sa - w1 -
            cc * (xi * ca + eta * sa) * w3) *
           (2 * x2 + z2 + z2 * c2c - 4 * x * w1 + 2 * w1_2 -
            4 * z * Power(cc, 2) * w3 + 2 * Power(cc, 2) * w3_2))));
  return {std::make_pair(y1, b1), std::make_pair(y2, b2)};
}

std::vector<std::pair<float, float>> ComputeZC(
    const Vector6f& v, const Vector4f& essential_parameters,
    const Vector2f& free_parameters) {
  const float a = v(0);
  const float b = v(1);
  const float y = v(5);
  const float x = v(4);
  const float w1 = free_parameters(0);
  const float w2 = free_parameters(1);
  const float w3 = essential_parameters(0);
  const float xi = essential_parameters(1) / essential_parameters(3);
  const float eta = essential_parameters(2) / essential_parameters(3);
  const float eta2 = Power(eta, 2);
  const float xi2 = Power(xi, 2);
  const float c2b = cos(2 * b);
  const float c2a = cos(2 * a);
  const float s2b = sin(2 * b);
  const float y2 = Power(y, 2);
  const float x2 = Power(x, 2);
  const float w1_2 = Power(w1, 2);
  const float w2_2 = Power(w2, 2);
  const float ca = cos(a);
  const float cb = cos(b);
  const float sa = sin(a);
  const float sb = sin(b);

  const float z1 =
      (2 * Sqrt(2) *
       (x2 * xi * ca * cb + y2 * xi * ca * cb + x2 * eta * cb * sa +
        y2 * eta * cb * sa - x2 * sb - y2 * sb -
        2 * x * (xi * ca * cb + eta * cb * sa - sb) * w1 +
        (xi * ca * cb + eta * cb * sa - sb) * w1_2 -
        2 * y * (xi * ca * cb + eta * cb * sa - sb) * w2 +
        (xi * ca * cb + eta * cb * sa - sb) * w2_2 +
        (Sqrt((2 + 3 * eta2 + 3 * xi2 + 2 * (eta2 - xi2) * c2a * Power(cb, 2) -
               2 * eta * xi * s2b -
               c2b * (-2 + eta2 + xi2 + 2 * eta * xi * s2b) +
               4 * xi * ca * s2b + 4 * eta * sa * s2b) *
              (x2 + y2 - 2 * x * w1 + w1_2 - 2 * y * w2 + w2_2)) *
         w3) /
            2.)) /
      Sqrt((4 + 6 * eta2 + 6 * xi2 + 2 * (eta2 - xi2) * c2a +
            4 * eta * cos(a - 2 * b) + eta2 * cos(2 * (a - b)) -
            xi2 * cos(2 * (a - b)) + 4 * c2b - 2 * eta2 * c2b - 2 * xi2 * c2b +
            eta2 * cos(2 * (a + b)) - xi2 * cos(2 * (a + b)) -
            4 * eta * cos(a + 2 * b) - 4 * eta * xi * s2b -
            4 * xi * sin(a - 2 * b) - 2 * eta * xi * sin(2 * (a - b)) -
            2 * eta * xi * sin(2 * (a + b)) + 4 * xi * sin(a + 2 * b)) *
           (x2 + y2 - 2 * x * w1 + w1_2 - 2 * y * w2 + w2_2));
  const float c1 = (ArcTan(
      (2 * Sqrt(2) *
       (y * eta * ca + x * cb - y * xi * sa + x * xi * ca * sb +
        x * eta * sa * sb - (cb + (xi * ca + eta * sa) * sb) * w1 +
        (-(eta * ca) + xi * sa) * w2)) /
          Sqrt((4 + 6 * eta2 + 6 * xi2 + 2 * (eta2 - xi2) * c2a +
                4 * eta * cos(a - 2 * b) + eta2 * cos(2 * a - 2 * b) -
                xi2 * cos(2 * a - 2 * b) + 4 * c2b - 2 * eta2 * c2b -
                2 * xi2 * c2b + eta2 * cos(2 * (a + b)) -
                xi2 * cos(2 * (a + b)) - 4 * eta * cos(a + 2 * b) -
                4 * eta * xi * s2b - 4 * xi * sin(a - 2 * b) -
                2 * eta * xi * sin(2 * a - 2 * b) -
                2 * eta * xi * sin(2 * (a + b)) + 4 * xi * sin(a + 2 * b)) *
               (x2 + y2 - 2 * x * w1 + w1_2 - 2 * y * w2 + w2_2)),
      (2 * Sqrt(2) *
       (-(x * eta * ca) + y * cb + x * xi * sa + y * xi * ca * sb +
        y * eta * sa * sb + (eta * ca - xi * sa) * w1 -
        (cb + (xi * ca + eta * sa) * sb) * w2)) /
          Sqrt((4 + 6 * eta2 + 6 * xi2 + 2 * (eta2 - xi2) * c2a +
                4 * eta * cos(a - 2 * b) + eta2 * cos(2 * a - 2 * b) -
                xi2 * cos(2 * a - 2 * b) + 4 * c2b - 2 * eta2 * c2b -
                2 * xi2 * c2b + eta2 * cos(2 * (a + b)) -
                xi2 * cos(2 * (a + b)) - 4 * eta * cos(a + 2 * b) -
                4 * eta * xi * s2b - 4 * xi * sin(a - 2 * b) -
                2 * eta * xi * sin(2 * a - 2 * b) -
                2 * eta * xi * sin(2 * (a + b)) + 4 * xi * sin(a + 2 * b)) *
               (x2 + y2 - 2 * x * w1 + w1_2 - 2 * y * w2 + w2_2))));
  const float z2 =
      (2 * Sqrt(2) *
       (-(x2 * xi * ca * cb) - y2 * xi * ca * cb - x2 * eta * cb * sa -
        y2 * eta * cb * sa + x2 * sb + y2 * sb +
        2 * x * (xi * ca * cb + eta * cb * sa - sb) * w1 +
        (-(xi * ca * cb) - eta * cb * sa + sb) * w1_2 +
        2 * y * (xi * ca * cb + eta * cb * sa - sb) * w2 +
        (-(xi * ca * cb) - eta * cb * sa + sb) * w2_2 +
        (Sqrt((2 + 3 * eta2 + 3 * xi2 + 2 * (eta2 - xi2) * c2a * Power(cb, 2) -
               2 * eta * xi * s2b -
               c2b * (-2 + eta2 + xi2 + 2 * eta * xi * s2b) +
               4 * xi * ca * s2b + 4 * eta * sa * s2b) *
              (x2 + y2 - 2 * x * w1 + w1_2 - 2 * y * w2 + w2_2)) *
         w3) /
            2.)) /
      Sqrt((4 + 6 * eta2 + 6 * xi2 + 2 * (eta2 - xi2) * c2a +
            4 * eta * cos(a - 2 * b) + eta2 * cos(2 * (a - b)) -
            xi2 * cos(2 * (a - b)) + 4 * c2b - 2 * eta2 * c2b - 2 * xi2 * c2b +
            eta2 * cos(2 * (a + b)) - xi2 * cos(2 * (a + b)) -
            4 * eta * cos(a + 2 * b) - 4 * eta * xi * s2b -
            4 * xi * sin(a - 2 * b) - 2 * eta * xi * sin(2 * (a - b)) -
            2 * eta * xi * sin(2 * (a + b)) + 4 * xi * sin(a + 2 * b)) *
           (x2 + y2 - 2 * x * w1 + w1_2 - 2 * y * w2 + w2_2));
  const float c2 = (ArcTan(
      (-2 * Sqrt(2) *
       (y * eta * ca + x * cb - y * xi * sa + x * xi * ca * sb +
        x * eta * sa * sb - (cb + (xi * ca + eta * sa) * sb) * w1 +
        (-(eta * ca) + xi * sa) * w2)) /
          Sqrt((4 + 6 * eta2 + 6 * xi2 + 2 * (eta2 - xi2) * c2a +
                4 * eta * cos(a - 2 * b) + eta2 * cos(2 * a - 2 * b) -
                xi2 * cos(2 * a - 2 * b) + 4 * c2b - 2 * eta2 * c2b -
                2 * xi2 * c2b + eta2 * cos(2 * (a + b)) -
                xi2 * cos(2 * (a + b)) - 4 * eta * cos(a + 2 * b) -
                4 * eta * xi * s2b - 4 * xi * sin(a - 2 * b) -
                2 * eta * xi * sin(2 * a - 2 * b) -
                2 * eta * xi * sin(2 * (a + b)) + 4 * xi * sin(a + 2 * b)) *
               (x2 + y2 - 2 * x * w1 + w1_2 - 2 * y * w2 + w2_2)),
      (2 * Sqrt(2) *
       (x * eta * ca - y * cb - x * xi * sa - y * xi * ca * sb -
        y * eta * sa * sb + (-(eta * ca) + xi * sa) * w1 +
        (cb + (xi * ca + eta * sa) * sb) * w2)) /
          Sqrt((4 + 6 * eta2 + 6 * xi2 + 2 * (eta2 - xi2) * c2a +
                4 * eta * cos(a - 2 * b) + eta2 * cos(2 * a - 2 * b) -
                xi2 * cos(2 * a - 2 * b) + 4 * c2b - 2 * eta2 * c2b -
                2 * xi2 * c2b + eta2 * cos(2 * (a + b)) -
                xi2 * cos(2 * (a + b)) - 4 * eta * cos(a + 2 * b) -
                4 * eta * xi * s2b - 4 * xi * sin(a - 2 * b) -
                2 * eta * xi * sin(2 * a - 2 * b) -
                2 * eta * xi * sin(2 * (a + b)) + 4 * xi * sin(a + 2 * b)) *
               (x2 + y2 - 2 * x * w1 + w1_2 - 2 * y * w2 + w2_2))));
  return {std::make_pair(z1, c1), std::make_pair(z2, c2)};
}

#endif  // EXPERIMENTAL_USERS_AIGERD_GENERAL_VOTING_ICCV_21_TOWARD_OPEN_SOURCE_X_H_H_
