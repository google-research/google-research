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

// Reads DNA sequences from the keys of an sstable and write another sstable
// with DNA sequence keys and values given by N-gram GenericFeatureVectors.
//
// Since the DNA alphabet has only four characters, and typical N-gram sizes are
// 4-6, we use a simple mathematical formula to map N-grams to sparse dimensions
// instead of hashing. We simply encode each character in the N-gram as two
// bits:
//
// A = 00
// C = 01
// G = 10
// T = 11
//
// Then, for example, the four-gram TCAG would be encoded as:
//
// (Leading zeroes) 11 01 00 10
//
// This allows up to 32-grams to be used, which should be more than sufficient.
#include "xxx/preprocess/scam_featurize.h"

#include <vector>

#include "absl/strings/string_view.h"
#include "xxx/scam/data_format/features.proto.h"
#include "xxx/scam/utils/gfv_normalization.h"

namespace research_biology {
namespace aptamers {

namespace {

std::vector<uint32> SequenceToNgrams(absl::string_view dna, size_t ngram_size) {
  std::vector<uint32> result;
  for (size_t i = 0; i + ngram_size <= dna.ssize(); ++i) {
    uint32 dim = 0;
    for (size_t j = 0; j < ngram_size; ++j) {
      const char c = dna[i + j];
      switch (c) {
        case 'A':
        case 'a':
          break;  // 0
        case 'C':
        case 'c':
          dim |= (1 << (2 * j));
          break;
        case 'G':
        case 'g':
          dim |= (2 << (2 * j));
          break;
        case 'T':
        case 't':
          dim |= (3 << (2 * j));
          break;
        default:
          LOG(FATAL) << "Non-DNA Character:  " << c;
      }
    }

    result.push_back(dim);
  }

  return result;
}

research_scam::GenericFeatureVector DnaToGfv(absl::string_view dna) {
  const size_t ngram_size = 6;
  research_scam::GenericFeatureVector gfv;
  std::vector<uint32> dims = SequenceToNgrams(dna, ngram_size);
  std::sort(dims.begin(), dims.end());
  gfv.set_feature_dim(std::pow(4, ngram_size));
  CHECK(!dims.empty()) << "No valid N-grams for sequence: " << dna;
  gfv.add_feature_index(dims[0]);
  gfv.add_feature_value_float(1);
  for (size_t i = 1; i < dims.size(); ++i) {
    if (dims[i] == dims[i - 1]) {
      ++(*gfv.mutable_feature_value_float()->rbegin());
    } else {
      gfv.add_feature_index(dims[i]);
      gfv.add_feature_value_float(1);
    }
  }

  gfv.set_feature_type(research_scam::GenericFeatureVector::FLOAT);
  return research_scam::NormalizeUnitL2(gfv);
}

class FeaturizeFn : public flume::MapFn<string, FeatureVectorEntry> {
 public:
  FeatureVectorEntry Map(const string& dna) override {
    return flume::make_kv(dna, DnaToGfv(dna));
  }

 private:
  REGISTER_AS_STATELESS_FN(FeaturizeFn);
};

}  // namespace

FeatureVectorTable FeaturizeSequences(const SequenceCollection& in) {
  return in.ParDo("featurize_dna", new FeaturizeFn());
}

}  // namespace aptamers
}  // namespace research_biology
