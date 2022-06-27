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

#include <string>
#include <unordered_map>
#include <vector>

# Google 'internal' paths (although all go to TF or other open source code)
#include "xxx/node_hash_map.h"
#include "xxx/strings/str_cat.h"
#include "xxx/op.h"
#include "xxx/op_kernel.h"
#include "xxx/shape_inference.h"

namespace tensorflow {

namespace {
// Calculates the vector of all possible kmers up to length <= k_max.
std::vector<string> CalculateAllKmers(const int k_max) {
  // The order of these bases must remain the same as the order of
  // ORDERED_BASES in config.py.
  const std::vector<string> bases = {"A", "T", "G", "C"};
  std::vector<string> all_kmers = bases;
  std::vector<string> prev_kmers = bases;
  std::vector<string> next_kmers;

  for (int k = 2; k <= k_max; ++k) {
    for (const auto& prev_kmer : prev_kmers) {
      for (const auto& base : bases) {
        next_kmers.push_back(absl::StrCat(prev_kmer, base));
      }
    }
    for (const auto& kmer : next_kmers) {
      all_kmers.push_back(kmer);
    }
    prev_kmers = next_kmers;
    next_kmers.clear();
  }
  return all_kmers;
}
}  // namespace

REGISTER_OP("CountAllDnaKmers")
    .Input("sequences: string")
    .Attr("k_max: int >= 1")
    .Output("output: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::DimensionHandle batch_size = c->Dim(c->input(0), 0);
      int64 k_max;
      TF_RETURN_IF_ERROR(c->GetAttr("k_max", &k_max));
      int64 n_kmers = 0;
      for (int64 k = 1; k <= k_max; ++k) {
        // 4^k = 2^(2*k)
        n_kmers += (1 << (2 * k));
      }

      c->set_output(0, c->Matrix(batch_size, n_kmers));
      return Status::OK();
    })
    .Doc(R"doc(
Calculate occurrence counts for all kmers up size `k_max`.

The output shape is the same shape as the input, with an additional dimension
appended with size equal to the number of kmers.

Kmers appear in order A, T, G, C, AA, AT, ....

sequences: 1D tensor of strings.
k_max: int >= 1.
)doc");

class CountAllDnaKmersOp : public OpKernel {
 public:
  explicit CountAllDnaKmersOp(OpKernelConstruction* context) :
      OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("k_max", &k_max_));
    all_kmers_ = CalculateAllKmers(k_max_);
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<tensorflow::tstring>();

    Tensor* output_tensor = nullptr;
    const int num_all_kmers = all_kmers_.size();
    OP_REQUIRES_OK(context, context->allocate_output(
        0, {input.size(), num_all_kmers}, &output_tensor));
    auto output = output_tensor->tensor<int32, 2>();

    for (int b = 0; b < input.size(); ++b) {
      const string& sequence = input(b);

      // Count all kmer occurences
      absl::node_hash_map<string, int> kmers;
      for (int k = 1; k <= k_max_; ++k) {
        const int i_max = sequence.size() - k + 1;
        for (int i = 0; i < i_max; ++i) {
          string current_kmer = sequence.substr(i, k);
          if (kmers.count(current_kmer)) {
            kmers[current_kmer] += 1;
          } else {
            kmers[current_kmer] = 1;
          }
        }
      }

      // Fill in kmer counts in the output
      for (int i = 0; i < all_kmers_.size(); ++i) {
        const string& current_kmer = all_kmers_[i];
        if (kmers.count(current_kmer)) {
          output(b, i) = kmers[current_kmer];
        } else {
          output(b, i) = 0;
        }
      }
    }
  }

 private:
  int k_max_;
  std::vector<string> all_kmers_;
};

REGISTER_KERNEL_BUILDER(Name("CountAllDnaKmers").Device(DEVICE_CPU),
                        CountAllDnaKmersOp);


namespace {
// Encode a DNA base as an integer.
int EncodeDnaBase(const char base) {
  switch (base) {
    case 'A':
      return 0;
    case 'T':
      return 1;
    case 'G':
      return 2;
    case 'C':
      return 3;
    default:
      return -1;
  }
}
}  // namespace


REGISTER_OP("DnaSequenceToIndices")
    .Input("sequences: string")
    .Attr("sequence_length: int >= 0")
    .Attr("forward_primer: string = ''")
    .Attr("reverse_primer: string = ''")
    .Output("output: int32")
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      shape_inference::DimensionHandle batch_size = c->Dim(c->input(0), 0);
      int64 sequence_length;
      TF_RETURN_IF_ERROR(c->GetAttr("sequence_length", &sequence_length));

      string forward_primer;
      TF_RETURN_IF_ERROR(c->GetAttr("forward_primer", &forward_primer));
      string reverse_primer;
      TF_RETURN_IF_ERROR(c->GetAttr("reverse_primer", &reverse_primer));

      int64 full_sequence_length =
          sequence_length + forward_primer.size() + reverse_primer.size();
      c->set_output(0, c->Matrix(batch_size, full_sequence_length));
      return Status::OK();
    })
    .Doc(R"doc(
Encode DNA sequences into integer indexes.

The output maps characters in the sequence to integers suitable for feeding into
`tf.one_hot`:
A -> 0, T -> 1, G -> 2, C -> 3

Given input with shape `(batch_size,)`, the output has shape
`(batch_size, len(forward_primer) + sequence_length + len(reverse_primer))`.

sequences: 1D tensor of strings.
sequence_length: string length of each DNA sequence in `sequences`.
forward_primer: optional fixed prefix for all sequences.
reverse_primer: optional fixed suffix for all sequences.
)doc");

class DnaSequenceToIndicesOp : public OpKernel {
 public:
  explicit DnaSequenceToIndicesOp(OpKernelConstruction* context) :
      OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("sequence_length",
                                             &sequence_length_));
    OP_REQUIRES_OK(context, context->GetAttr("forward_primer",
                                             &forward_primer_));
    OP_REQUIRES_OK(context, context->GetAttr("reverse_primer",
                                             &reverse_primer_));
  }

  void Compute(OpKernelContext* context) override {
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<tensorflow::tstring>();

    const int full_sequence_length = forward_primer_.length() +
                                     reverse_primer_.length() +
                                     sequence_length_;

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(
        0, {input.size(), full_sequence_length}, &output_tensor));
    auto output = output_tensor->tensor<int32, 2>();

    for (int b = 0; b < input.size(); ++b) {
      const string& full_sequence = forward_primer_ + input(b) +
                                    reverse_primer_;
      OP_REQUIRES(
          context, full_sequence.length() == full_sequence_length,
          errors::InvalidArgument(
              "encountered sequence with length != ", sequence_length_,
              ": ", input(b)));
      for (int i = 0; i < full_sequence_length; ++i) {
        const int index = EncodeDnaBase(full_sequence[i]);
        OP_REQUIRES(
            context, index != -1,
            errors::InvalidArgument(
                "encountered invalid DNA sequence: ", full_sequence));
        output(b, i) = index;
      }
    }
  }

 private:
  int sequence_length_;
  string forward_primer_;
  string reverse_primer_;
};

REGISTER_KERNEL_BUILDER(Name("DnaSequenceToIndices").Device(DEVICE_CPU),
                        DnaSequenceToIndicesOp);

}  // namespace tensorflow
