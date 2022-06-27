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

// Calculator that produces windows of values.
//
// If it operates on one input stream, it takes an input stream of items
// std::vector<float> and produces an output stream of windows
// std::vector<float> such that if items i1={i1_v1, i1_v2, i1_v3}, i2={i2_v1,
// i2_v2, i2_v3} are in the window, then the output value is {i1_v1, i1_v2,
// i1_v3, i2_v1, i2_v2, i2_v3}, i.e. it is a concatenation of i1 and i2.
//
// If there are many input streams, then for each input stream there will be a
// corresponding output stream where windows will be produced. Note that the
// windowing options (window size, window stride) are shared across all streams.
// If there are many input streams, then at each timepoint every stream must
// have a value, otherwise an error will be returned.
//
// The timestamp of the produced value is the timestamp of the most recent value
// among all included input items. In the example above, the produced value
// {i1_v1, i1_v2, ..., i2_v3} will have a timestamp of i2.
//
// The user can configure window size. Important: both size and
// stride refer to the number of items in the input stream! If each item in the
// input stream is a 3d vector (e.g. 3d accelerometer data), then setting window
// size 10 will result in producing 30d vector at each output.
//
// If the whole stream has been consumed, but no windows were generated, the
// only one and final window is then produced. It's padded on the right with
// either 0 or with the last seen value (which can be configured by setting the
// padding_strategy) side and its timestamp is the largets timestamp of the
// input sample. If the input stream is empty, no output is generated.
//
// This calculator uses O(window_size * datum_dimensionality *
// number_of_input_streams) memory for internal state.
//
// If input stream has N values, window size is W, and stride is S, then
// the output stream will have ⌊(N - W) / S + 1⌋ values.
#include <cstdint>
#include <deque>
#include <functional>
#include <iterator>
#include <vector>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/calculator_options.pb.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/optional.h"
#include "intent_recognition/processing/window_calculator.pb.h"

namespace ambient_sensing {
namespace {
using ::mediapipe::CalculatorContext;
using ::mediapipe::CalculatorContract;

constexpr char kInputStreamTag[] = "INPUT_STREAM";
constexpr char kOutputStreamTag[] = "WINDOWED_VALUES_OUTPUT_STREAM";
}  // namespace

class WindowCalculator : public mediapipe::CalculatorBase {
 public:
  WindowCalculator() = default;

  static absl::Status GetContract(CalculatorContract* cc) {
    if (cc->Inputs().NumEntries() != cc->Outputs().NumEntries()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Number of input streams must match number of output streams. ",
          cc->Inputs().NumEntries(), " != ", cc->Outputs().NumEntries()));
    }
    for (int i = 0; i < cc->Inputs().NumEntries(); i++) {
      cc->Inputs().Get(kInputStreamTag, i).Set<std::vector<float>>();
      cc->Outputs().Get(kOutputStreamTag, i).Set<std::vector<float>>();
    }
    if (!cc->Options().HasExtension(WindowCalculatorOptions::ext)) {
      return absl::InvalidArgumentError("WindowCalculatorOptions must be set.");
    }
    auto options = cc->Options().GetExtension(WindowCalculatorOptions::ext);
    if (options.window_size() <= 0) {
      return absl::InvalidArgumentError("window_size must be >= 1.");
    }
    if (options.enforce_input_dims_size() != cc->Inputs().NumEntries()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "enforce_input_dims_size must be number of input streams. ",
          options.enforce_input_dims_size(),
          " != ", cc->Inputs().NumEntries()));
    }
    if (!absl::c_all_of(options.enforce_input_dims(),
                        [](int64 x) { return x >= 1; })) {
      return absl::InvalidArgumentError("enforce_input_dims[i] must be >= 1.");
    }
    if (options.window_stride() <= 0) {
      return absl::InvalidArgumentError("window_stride must be >= 1.");
    }
    if (options.minimum_windows() < 0) {
      return absl::InvalidArgumentError("minimum_windows must be >= 0.");
    }
    return absl::OkStatus();
  }

  absl::Status Open(CalculatorContext* cc) override {
    options_ = cc->Options().GetExtension(WindowCalculatorOptions::ext);
    for (auto enforce_input_dims : options_.enforce_input_dims()) {
      output_dims_by_output_index_.push_back(1LL * enforce_input_dims *
                                             options_.window_size());
    }

    // Create a buffer for each output stream.
    buffer_by_output_index_.resize(output_dims_by_output_index_.size());
    buffer_input_count_.resize(output_dims_by_output_index_.size());
    return absl::OkStatus();
  }

  absl::Status Process(CalculatorContext* cc) override {
    DequeuePushBackWithoutExceedingCapacity(
        &timestamp_buffer_, cc->InputTimestamp(), kMaxTimestampCount);
    for (int i = 0; i < cc->Inputs().NumEntries(); i++) {
      auto& input_stream = cc->Inputs().Get(kInputStreamTag, i);
      if (input_stream.Value().IsEmpty()) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "All streams must have entries at a given timestamp, but stream %d "
            "does not. (Timestamp: %d ms)",
            i, cc->InputTimestamp().Microseconds()));
      }
      const auto& item = input_stream.Get<std::vector<float>>();
      if (item.size() != options_.enforce_input_dims(i)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Bad input dimensions at timestamp ",
            cc->InputTimestamp().Microseconds(), " - required ",
            options_.enforce_input_dims(i), ", actual ", item.size()));
      }
      auto& buffer = buffer_by_output_index_[i];
      AddValueToBuffer(item, &buffer, &buffer_input_count_[i]);
    }
    absl::Status status = MaybeProduceWindow(cc->InputTimestamp(), cc);
    if (status != absl::OkStatus()) return status;
    return absl::OkStatus();
  }

  absl::Status Close(CalculatorContext* cc) override {
    // The input stream didn't have any values.
    if (timestamp_buffer_.empty()) return absl::OkStatus();

    // Enough windows were produced already, so no need to make padded ones.
    if (output_count_ >= options_.minimum_windows()) return absl::OkStatus();

    // If we ended up here, then we need to produce more windows.
    // Determine padding values.
    constexpr int kNumMicrosPerMilli = 1000;
    mediapipe::TimestampDiff timestamp_increment_value =
        timestamp_buffer_.size() == kMaxTimestampCount
            ? timestamp_buffer_[1] - timestamp_buffer_[0]
            : kNumMicrosPerMilli;
    std::vector<std::vector<float>> pad;
    for (int i = 0; i < buffer_by_output_index_.size(); i++) {
      ASSIGN_OR_RETURN(std::vector<float> pad_val,
                       MakePadValue(options_.enforce_input_dims(i),
                                    options_.padding_strategy(),
                                    buffer_by_output_index_[i]));
      pad.push_back(pad_val);
    }
    // Continuously feed padding values in until we've output enough windows.
    while (output_count_ < options_.minimum_windows()) {
      for (int i = 0; i < buffer_by_output_index_.size(); i++) {
        AddValueToBuffer(pad[i], &buffer_by_output_index_[i],
                         &buffer_input_count_[i]);
      }
      DequeuePushBackWithoutExceedingCapacity(
          &timestamp_buffer_,
          timestamp_buffer_.back() + timestamp_increment_value,
          kMaxTimestampCount);
      absl::Status status = MaybeProduceWindow(timestamp_buffer_.back(), cc);
      if (status != absl::OkStatus()) return status;
    }
    return absl::OkStatus();
  }

 private:
  template <class T>
  void DequeuePushBackWithoutExceedingCapacity(std::deque<T>* deque, T value,
                                               int max_capacity) {
    if (deque->size() == max_capacity) {
      deque->pop_front();
    }
    deque->push_back(value);
  }

  absl::StatusOr<std::vector<float>> MakePadValue(
      int32_t dims, WindowCalculatorOptions_PaddingStrategy padding_strategy,
      const std::deque<std::vector<float>>& buffer) {
    switch (padding_strategy) {
      case WindowCalculatorOptions::RIGHT_ZERO:
        return std::vector<float>(/*n=*/dims, 0.0);
      case WindowCalculatorOptions::RIGHT_LAST_VALUE: {
        if (buffer.empty()) {
          return absl::InternalError(
              "Can't pad the empty buffer with RIGHT_LAST_VALUE strategy");
        }
        return std::vector<float>(buffer.back());
      }
      default:
        return absl::InternalError(absl::StrCat(
            "Unhandled padding strategy: ",
            WindowCalculatorOptions_PaddingStrategy_Name(padding_strategy)));
    }
  }

  void AddValueToBuffer(const std::vector<float>& item,
                        std::deque<std::vector<float>>* buffer_ptr,
                        int64_t* buffer_count) {
    DequeuePushBackWithoutExceedingCapacity(buffer_ptr, item,
                                            options_.window_size());
    *buffer_count += 1;
  }

  // Checks buffers and produces windows if 2 conditions are met:
  //   1. the buffer holds enough values
  //   2. current index in the input stream is divisible by the window_stride.
  // Additionally checks that window is either produced for each output stream,
  // or not produced for any output streams.
  absl::Status MaybeProduceWindow(const mediapipe::Timestamp& timestamp,
                                  CalculatorContext* cc) {
    int32_t windows_produced = 0;
    int32_t index = -1;
    for (auto& buffer : buffer_by_output_index_) {
      index += 1;
      // Only produce a window when there's been at least window_size inputs.
      if (buffer.size() != options_.window_size()) return absl::OkStatus();
      // Only produce windows where start_index = k * stride (k = 0, 1, ...).
      if ((buffer_input_count_[index] - options_.window_size()) %
              options_.window_stride() !=
          0) {
        return absl::OkStatus();
      }
      auto result = absl::make_unique<std::vector<float>>();
      for (const auto& r : buffer) {
        absl::c_copy(r, std::back_inserter(*result));
      }
      if (result->size() != output_dims_by_output_index_[index]) {
        return absl::InternalError(absl::StrCat(
            "Bad output dimensions at timestamp ", timestamp.Microseconds(),
            " - expected ", output_dims_by_output_index_[index], ", actual ",
            result->size(), " [output stream index ", index, "]"));
      }
      cc->Outputs()
          .Get(kOutputStreamTag, index)
          .Add(result.release(), timestamp);
      windows_produced += 1;
    }
    if (windows_produced != 0) {
      if (windows_produced != buffer_by_output_index_.size()) {
        return absl::InternalError(absl::StrCat(
            "Windows must be produced for all output streams at the same time. "
            "Produced ",
            windows_produced, " expected ", buffer_by_output_index_.size()));
      }
      output_count_++;
    }
    return absl::OkStatus();
  }

  // buffer_by_output_index_[i] is a buffer for i-th output stream.
  //
  // Values to the buffers should only be appened via AddValueToBuffer. In that
  // case there are invariants guaranteed:
  // - buffer_.size() is always at most options.window_size()
  // - buffer_input_count_[i] keeps track of how many inputs that buffer has
  // received.
  std::vector<std::deque<std::vector<float>>> buffer_by_output_index_;

  // Even though all elements in buffer_input_count_[i] should stay the same
  // before and after Process() and theoretically vector is not required here
  // the vector is still used for better organization and ease of future
  // refactorings.
  std::vector<int64_t> buffer_input_count_;

  WindowCalculatorOptions options_;
  std::deque<mediapipe::Timestamp> timestamp_buffer_;
  int output_count_ = 0;
  std::vector<int64_t> output_dims_by_output_index_;

  static constexpr int kMaxTimestampCount = 2;
};

REGISTER_CALCULATOR(WindowCalculator);

}  // namespace ambient_sensing
