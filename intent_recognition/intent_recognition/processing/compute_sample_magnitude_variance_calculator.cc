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

// Calculator that returns the sample variance of the per sample magnitude of
// the input window. It uses Welford's online algorithm to compute the sample
// variance.

#include <math.h>

#include <cstdint>
#include <memory>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "absl/status/status.h"
#include "intent_recognition/processing/compute_sample_magnitude_variance_calculator.pb.h"


namespace ambient_sensing {

namespace {
constexpr char kInputWindowTag[] = "INPUT_WINDOW";
constexpr char kOutputVarianceTag[] = "OUTPUT_VARIANCE";

using ::mediapipe::CalculatorContext;
using ::mediapipe::CalculatorContract;

}  // namespace

class ComputeSampleMagnitudeVarianceCalculator
    : public mediapipe::CalculatorBase {
 public:
  ComputeSampleMagnitudeVarianceCalculator() = default;

  static absl::Status GetContract(CalculatorContract* cc) {
    int32_t num_inputs = cc->Inputs().NumEntries(kInputWindowTag);
    if (num_inputs != 1) {
      return absl::InvalidArgumentError(
          "Bad configuration: Exactly one input stream required.");
    }
    cc->Inputs().Tag(kInputWindowTag).Set<std::vector<float>>();

    if (cc->Outputs().NumEntries() != 1) {
      return absl::InvalidArgumentError(
          "Bad configuration: Exactly one output stream required.");
    }
    cc->Outputs().Tag(kOutputVarianceTag).Set<std::vector<float>>();

    if (!cc->Options().HasExtension(
            ComputeSampleMagnitudeVarianceCalculatorOptions::ext)) {
      return absl::InvalidArgumentError(
          "ComputeSampleMagnitudeVarianceCalculatorOptions must be specified.");
    }

    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    auto options = cc->Options().GetExtension(
        ComputeSampleMagnitudeVarianceCalculatorOptions::ext);
    if (!options.has_window_size()) {
      return absl::InvalidArgumentError("window_size has to be specified.");
    }
    if (options.window_size() < 1) {
      return absl::InvalidArgumentError("window_size has to be >= 1.");
    }
    window_size_ = options.window_size();
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    const auto& input_vector =
        cc->Inputs().Tag(kInputWindowTag).Get<std::vector<float>>();
    if (input_vector.size() % window_size_ != 0) {
      return absl::InvalidArgumentError(
          "Input size is not a multiple of window_size.");
    }
    int32_t n_dims = input_vector.size() / window_size_;

    double variance = 0;
    if (window_size_ > 1) {
      std::vector<double> magnitudes;
      double sum = 0;
      for (int timestep = 0; timestep < window_size_; timestep++) {
        double squared_magnitude = 0;
        int first_dim_at_timestep = timestep * n_dims;
        for (int dim = 0; dim < n_dims; dim++) {
          const double current_value =
              static_cast<double>(input_vector[first_dim_at_timestep + dim]);
          squared_magnitude += pow(current_value, 2);
        }

        double magnitude = sqrt(squared_magnitude);
        magnitudes.push_back(magnitude);
        sum += magnitude;
      }

      // Calculate the mean.
      double mean = sum / magnitudes.size();

      // Calculate variance.
      for (double magnitude : magnitudes) {
        double diff_squared = pow(magnitude - mean, 2);
        variance += diff_squared;
      }
      variance /= magnitudes.size() - 1;
    }

    std::vector<float> output_variance({static_cast<float>(variance)});
    cc->Outputs()
        .Tag(kOutputVarianceTag)
        .AddPacket(mediapipe::MakePacket<std::vector<float>>(output_variance)
                       .At(cc->InputTimestamp()));
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    return absl::OkStatus();
  }

 private:
  int32_t window_size_;
};
REGISTER_CALCULATOR(ComputeSampleMagnitudeVarianceCalculator);

}  // namespace ambient_sensing
