// Copyright 2025 The Google Research Authors.
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

// Copyright 2023 The Google Research Authors.
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

#include "elo_fast_math.h"

#include <cstddef>

#include "Eigen/Core"
#include "Eigen/Dense"

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "elo_fast_math.cc"
#include "hwy/foreach_target.h"
// Keep in this order
#include "hwy/contrib/math/math-inl.h"
#include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace HWY_NAMESPACE {

template <typename D, typename V>
V LogWinProbabilityImpl(D d, V elo_a, V elo_b) {
  elo_a *= Set(d, kLog10 / kEloN);
  elo_b *= Set(d, kLog10 / kEloN);
  // log(exp(a) / (exp(a)+exp(b))) =     - log(1+exp(b-a))
  //                               = a-b - log(1+exp(a-b))
  // the top version is safe to compute if b < a, the bottom version otherwise.
  auto a_is_smaller = elo_a < elo_b;
  auto bma = elo_b - elo_a;
  auto amb = elo_a - elo_b;
  auto exp_arg = IfThenElse(a_is_smaller, amb, bma);
  auto bias = IfThenElseZero(a_is_smaller, amb);
  return bias - Log1p(d, Exp(d, exp_arg));
}

template <typename D, typename V>
V WinProbabilityImpl(D d, V elo_a, V elo_b) {
  return Exp(d, LogWinProbabilityImpl(d, elo_a, elo_b));
}

float LogWinProbability(float elo_a, float elo_b) {
  HWY_CAPPED(float, 1) d;
  return GetLane(LogWinProbabilityImpl(d, Set(d, elo_a), Set(d, elo_b)));
}

float WinProbability(float elo_a, float elo_b) {
  HWY_CAPPED(float, 1) d;
  return GetLane(WinProbabilityImpl(d, Set(d, elo_a), Set(d, elo_b)));
}

template <bool exp>
Eigen::MatrixXd ComputeWinProbabilityMatrix(const Eigen::VectorXd& elos) {
  Eigen::MatrixXd colwise_elos(elos.size(), elos.size());
  Eigen::MatrixXd rowwise_elos(elos.size(), elos.size());
  colwise_elos.colwise() = elos;
  rowwise_elos.rowwise() = elos.transpose();
  Eigen::MatrixXd result(elos.size(), elos.size());
  size_t count = elos.size() * elos.size();
  size_t i = 0;
  HWY_FULL(double) d;
  for (; i + Lanes(d) <= count; i += Lanes(d)) {
    auto elo_a = LoadU(d, colwise_elos.data() + i);
    auto elo_b = LoadU(d, rowwise_elos.data() + i);
    auto log_wp = LogWinProbabilityImpl(d, elo_a, elo_b);
    if (exp) {
      StoreU(Exp(d, log_wp), d, result.data() + i);
    } else {
      StoreU(log_wp, d, result.data() + i);
    }
  }
  for (; i < count; i++) {
    if (exp) {
      result.data()[i] =
          WinProbability(colwise_elos.data()[i], rowwise_elos.data()[i]);
    } else {
      result.data()[i] =
          LogWinProbability(colwise_elos.data()[i], rowwise_elos.data()[i]);
    }
  }
  return result;
}

Eigen::MatrixXd LogWinProbabilityMatrix(const Eigen::VectorXd& elos) {
  return ComputeWinProbabilityMatrix<false>(elos);
}

Eigen::MatrixXd WinProbabilityMatrix(const Eigen::VectorXd& elos) {
  return ComputeWinProbabilityMatrix<true>(elos);
}

template <typename D, typename V>
void AccumulateH2Impl(D d, V p, double weight, double* accum) {
  auto mplog2p = [](V p) {
    constexpr float kEps = 1e-38;
    p = Max(Set(D(), kEps), p);
    return Neg(p) * Log2(D(), p);
  };
  auto h2 = mplog2p(p) + mplog2p(Set(d, 1.0) - p);
  auto data = LoadU(d, accum);
  StoreU(MulAdd(Set(d, weight), h2, data), d, accum);
}

void AccumulateH2(const Eigen::MatrixXd& p, double weight,
                  Eigen::MatrixXd& accum) {
  size_t count = p.count();
  size_t i = 0;
  HWY_FULL(double) d;
  for (; i + Lanes(d) <= count; i += Lanes(d)) {
    AccumulateH2Impl(d, LoadU(d, p.data() + i), weight, accum.data() + i);
  }
  for (; i < count; i++) {
    HWY_CAPPED(double, 1) d;
    AccumulateH2Impl(d, LoadU(d, p.data() + i), weight, accum.data() + i);
  }
}

template <typename D, typename V>
V KLDivergenceImpl(D d, V true_p, V p) {
  auto kl = [&](V true_p, V p) {
    constexpr double kEps = 1e-5;
    auto eps = Set(d, kEps);
    auto true_p_log = Log(d, Max(eps, true_p));
    auto p_log = Log(d, Max(eps, p));
    return true_p * (true_p_log - p_log);
  };
  auto one = Set(d, 1.0);
  return kl(true_p, p) + kl(one - true_p, one - p);
}

double KLDivergence(const Eigen::MatrixXd& true_p, const Eigen::MatrixXd& p) {
  size_t count = p.count();
  size_t i = 0;
  HWY_FULL(double) d;
  auto simd_accum = Zero(d);
  for (; i + Lanes(d) <= count; i += Lanes(d)) {
    simd_accum += KLDivergenceImpl(d, LoadU(d, true_p.data() + i),
                                   LoadU(d, p.data() + i));
  }
  double accum = GetLane(SumOfLanes(d, simd_accum));
  for (; i < count; i++) {
    HWY_CAPPED(double, 1) d;
    accum += GetLane(KLDivergenceImpl(d, LoadU(d, true_p.data() + i),
                                      LoadU(d, p.data() + i)));
  }
  return accum;
}

}  // namespace HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

#if HWY_ONCE

HWY_EXPORT(LogWinProbability);
HWY_EXPORT(WinProbability);
HWY_EXPORT(LogWinProbabilityMatrix);
HWY_EXPORT(WinProbabilityMatrix);
HWY_EXPORT(AccumulateH2);
HWY_EXPORT(KLDivergence);

float LogWinProbability(float elo_a, float elo_b) {
  return HWY_STATIC_DISPATCH(LogWinProbability)(elo_a, elo_b);
}

float WinProbability(float elo_a, float elo_b) {
  return HWY_STATIC_DISPATCH(WinProbability)(elo_a, elo_b);
}

Eigen::MatrixXd LogWinProbabilityMatrix(const Eigen::VectorXd& elos) {
  return HWY_DYNAMIC_DISPATCH(LogWinProbabilityMatrix)(elos);
}

Eigen::MatrixXd WinProbabilityMatrix(const Eigen::VectorXd& elos) {
  return HWY_DYNAMIC_DISPATCH(WinProbabilityMatrix)(elos);
}

void AccumulateH2(const Eigen::MatrixXd& p, double weight,
                  Eigen::MatrixXd& accum) {
  return HWY_DYNAMIC_DISPATCH(AccumulateH2)(p, weight, accum);
}

double KLDivergence(const Eigen::MatrixXd& true_p, const Eigen::MatrixXd& p) {
  return HWY_DYNAMIC_DISPATCH(KLDivergence)(true_p, p);
}

#endif
