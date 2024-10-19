// Copyright 2024 The Google Research Authors.
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

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/strings/str_cat.h"
#include "scann/base/single_machine_base.h"
#include "scann/data_format/datapoint.h"
#include "scann/hashes/asymmetric_hashing2/indexing.h"
#include "scann/hashes/asymmetric_hashing2/querying.h"
#include "scann/hashes/asymmetric_hashing2/searcher.h"
#include "scann/oss_wrappers/scann_status.h"
#include "scann/proto/hash.pb.h"
#include "scann/utils/common.h"
#include "scann/utils/util_functions.h"

namespace research_scann {
namespace asymmetric_hashing2 {

class AHPrecomputedMutationArtifacts
    : public UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts {
 public:
  explicit AHPrecomputedMutationArtifacts(Datapoint<uint8_t> hashed)
      : hashed_(hashed) {}

  Datapoint<uint8_t>& hashed() { return hashed_; }

 private:
  Datapoint<uint8_t> hashed_;
};

namespace {

Status SetLUT16Hash(const DatapointPtr<uint8_t>& hashed, const size_t index,
                    PackedDataset* __restrict__ packed_struct) {
  MutableSpan<uint8_t> packed_dataset =
      MakeMutableSpan(packed_struct->bit_packed_data);

  const size_t hash_size = hashed.nonzero_entries();
  const size_t block_offset = index & 0x0f;

  const size_t offset = block_offset + (index & ~31) * hash_size / 2;
  SCANN_RET_CHECK_LE(offset + (hashed.nonzero_entries() - 1) * 16,
                     packed_dataset.size());
  SCANN_RET_CHECK_EQ(hashed.nonzero_entries(), packed_struct->num_blocks);

  if (index & 0x10) {
    for (int i = 0; i < hashed.nonzero_entries(); ++i) {
      packed_dataset[offset + i * 16] =
          (hashed.values()[i] << 4) | (packed_dataset[offset + i * 16] & 0x0f);
    }
  } else {
    for (int i = 0; i < hashed.nonzero_entries(); ++i) {
      packed_dataset[offset + i * 16] =
          hashed.values()[i] | (packed_dataset[offset + i * 16] & 0xf0);
    }
  }
  return OkStatus();
}

Datapoint<uint8_t> GetLUT16Hash(const size_t index,
                                const PackedDataset& packed_dataset) {
  const size_t hash_size = packed_dataset.num_blocks;
  const size_t block_offset = index & 0x0f;

  const size_t offset = block_offset + (index & ~31) * hash_size / 2;
  DCHECK_LE(offset + (hash_size - 1) * 16,
            packed_dataset.bit_packed_data.size());
  Datapoint<uint8_t> result;
  result.mutable_values()->reserve(hash_size);

  if (index & 0x10) {
    for (size_t i : Seq(hash_size)) {
      result.mutable_values()->push_back(
          (packed_dataset.bit_packed_data[offset + i * 16] >> 4));
    }
  } else {
    for (size_t i : Seq(hash_size)) {
      result.mutable_values()->push_back(
          (packed_dataset.bit_packed_data[offset + i * 16] & 0x0f));
    }
  }
  return result;
}

}  // namespace

template <typename T>
StatusOr<unique_ptr<typename Searcher<T>::Mutator>>
Searcher<T>::Mutator::Create(Searcher<T>* searcher) {
  const Indexer<T>* indexer = searcher->opts_.indexer_.get();
  if (!indexer) {
    return FailedPreconditionError(
        "research_scann::asymmetric_hashing2::Searcher has not been "
        "initialized "
        "with an indexer.");
  }

  PackedDataset* packed_dataset = nullptr;
  if (searcher->lut16_) {
    packed_dataset = &searcher->packed_dataset_;
  }
  auto result = absl::WrapUnique<typename Searcher<T>::Mutator>(
      new typename Searcher<T>::Mutator(searcher, indexer, packed_dataset));
  SCANN_RETURN_IF_ERROR(result->PrepareForBaseMutation(searcher));
  return std::move(result);
}

template <typename T>
void Searcher<T>::Mutator::Reserve(size_t size) {
  this->ReserveInBase(size);
}

template <typename T>
unique_ptr<UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts>
Searcher<T>::Mutator::ComputePrecomputedMutationArtifacts(
    const DatapointPtr<T>& maybe_residual,
    const DatapointPtr<T>& original) const {
  Datapoint<uint8_t> hashed;
  if (!Hash(maybe_residual, original, &hashed).ok()) return nullptr;
  return make_unique<AHPrecomputedMutationArtifacts>(std::move(hashed));
}

template <typename T>
unique_ptr<UntypedSingleMachineSearcherBase::PrecomputedMutationArtifacts>
Searcher<T>::Mutator::ComputePrecomputedMutationArtifacts(
    const DatapointPtr<T>& dptr) const {
  return ComputePrecomputedMutationArtifacts(dptr, dptr);
}

template <typename T>
Datapoint<uint8_t> Searcher<T>::Mutator::EnsureDatapointUnpacked(
    const Datapoint<uint8_t>& dp) {
  if (this->searcher_->opts_.quantization_scheme() ==
      AsymmetricHasherConfig::PRODUCT_AND_PACK) {
    Datapoint<uint8_t> unpacked;
    UnpackNibblesDatapoint(dp.ToPtr(), &unpacked);
    return unpacked;
  }
  return dp;
}

template <typename T>
StatusOr<Datapoint<T>> Searcher<T>::Mutator::GetDatapoint(
    DatapointIndex i) const {
  return UnimplementedError("GetDatapoint is not implemented.");
}

template <typename T>
StatusOr<DatapointIndex> Searcher<T>::Mutator::AddDatapoint(
    const DatapointPtr<T>& dptr, string_view docid, const MutationOptions& mo) {
  SCANN_RETURN_IF_ERROR(this->ValidateForAdd(dptr, docid, mo));
  Datapoint<uint8_t> hashed;
  PrecomputedMutationArtifacts* ma = mo.precomputed_mutation_artifacts;
  if (ma) {
    auto dc = dynamic_cast<AHPrecomputedMutationArtifacts*>(ma);
    if (!dc) {
      return InvalidArgumentError(
          "Invalid PrecomputedMutationArtifacts passed to "
          "asymmetric_hashing2::Searcher::Mutator::AddDatapoint.");
    }
    hashed = std::move(dc->hashed());
  } else {
    SCANN_RETURN_IF_ERROR(Hash(dptr, dptr, &hashed));
  }
  hashed = EnsureDatapointUnpacked(hashed);

  SCANN_ASSIGN_OR_RETURN(
      auto result2,
      this->AddDatapointToBase(dptr, docid,
                               MutateBaseOptions{.hashed = hashed.ToPtr()}));

  DatapointIndex result = kInvalidDatapointIndex;
  if (packed_dataset_) {
    result = packed_dataset_->num_datapoints++;
    const DimensionIndex hash_size = hashed.nonzero_entries();

    if (packed_dataset_->num_blocks == 0) {
      packed_dataset_->num_blocks = hashed.nonzero_entries();
    }

    if (!(result & 31)) {
      packed_dataset_->bit_packed_data.resize(
          packed_dataset_->bit_packed_data.size() + 16 * hash_size);
    }
    SCANN_RETURN_IF_ERROR(
        SetLUT16Hash(hashed.ToPtr(), result, packed_dataset_));
  }
  if (result == kInvalidDatapointIndex) {
    result = result2;
  } else if (result2 != kInvalidDatapointIndex) {
    SCANN_RET_CHECK_EQ(result, result2);
  }
  SCANN_RET_CHECK_NE(result, kInvalidDatapointIndex);
  return result;
}

template <typename T>
Status Searcher<T>::Mutator::RemoveDatapoint(DatapointIndex index) {
  SCANN_RETURN_IF_ERROR(this->ValidateForRemove(index));
  bool on_datapoint_index_rename_called = false;
  auto call_on_datapont_index_rename = [&](DatapointIndex old_idx,
                                           DatapointIndex new_idx) {
    if (on_datapoint_index_rename_called) return;
    this->OnDatapointIndexRename(old_idx, new_idx);
    on_datapoint_index_rename_called = true;
  };

  if (packed_dataset_) {
    const size_t new_size = --packed_dataset_->num_datapoints;
    Datapoint<uint8_t> hashed = GetLUT16Hash(new_size, *packed_dataset_);
    SCANN_RETURN_IF_ERROR(SetLUT16Hash(hashed.ToPtr(), index, packed_dataset_));

    if (!(new_size & 31)) {
      const DimensionIndex hash_size = hashed.nonzero_entries();
      packed_dataset_->bit_packed_data.resize(
          packed_dataset_->bit_packed_data.size() - 16 * hash_size);
    }
    call_on_datapont_index_rename(new_size, index);
  }
  SCANN_ASSIGN_OR_RETURN(const DatapointIndex swapped_in,
                         this->RemoveDatapointFromBase(index));
  call_on_datapont_index_rename(swapped_in, index);
  SCANN_RET_CHECK(on_datapoint_index_rename_called);
  return OkStatus();
}

template <typename T>
Status Searcher<T>::Mutator::RemoveDatapoint(string_view docid) {
  DatapointIndex index;
  if (!this->LookupDatapointIndex(docid, &index)) {
    return NotFoundError(absl::StrCat("Docid: ", docid, " is not found."));
  }
  return RemoveDatapoint(index);
}

template <typename T>
StatusOr<DatapointIndex> Searcher<T>::Mutator::UpdateDatapoint(
    const DatapointPtr<T>& dptr, string_view docid, const MutationOptions& mo) {
  DatapointIndex index;
  if (!this->LookupDatapointIndex(docid, &index)) {
    return NotFoundError(absl::StrCat("Docid: ", docid, " is not found."));
  }
  return UpdateDatapoint(dptr, index, mo);
}

template <typename T>
StatusOr<DatapointIndex> Searcher<T>::Mutator::UpdateDatapoint(
    const DatapointPtr<T>& dptr, DatapointIndex index,
    const MutationOptions& mo) {
  SCANN_RETURN_IF_ERROR(this->ValidateForUpdate(dptr, index, mo));

  Datapoint<uint8_t> hashed;
  const bool mutate_values_vector = true;
  if (mutate_values_vector) {
    PrecomputedMutationArtifacts* ma = mo.precomputed_mutation_artifacts;
    if (ma) {
      auto dc = dynamic_cast<AHPrecomputedMutationArtifacts*>(ma);
      if (!dc) {
        return InvalidArgumentError(
            "Invalid PrecomputedMutationArtifacts passed to "
            "asymmetric_hashing2::Searcher::Mutator::UpdateDatapoint.");
      }
      hashed = std::move(dc->hashed());
    } else {
      SCANN_RETURN_IF_ERROR(Hash(dptr, dptr, &hashed));
    }
    hashed = EnsureDatapointUnpacked(hashed);
    if (packed_dataset_) {
      SCANN_RETURN_IF_ERROR(
          SetLUT16Hash(hashed.ToPtr(), index, packed_dataset_));
    }
  }
  SCANN_RETURN_IF_ERROR(this->UpdateDatapointInBase(
      dptr, index, MutateBaseOptions{.hashed = hashed.ToPtr()}));
  return index;
}

template <typename T>
Status Searcher<T>::Mutator::Hash(const DatapointPtr<T>& maybe_residual,
                                  const DatapointPtr<T>& original,
                                  Datapoint<uint8_t>* result) const {
  const double noise_shaping_threshold =
      searcher_->opts_.noise_shaping_threshold_;
  if (std::isnan(noise_shaping_threshold)) {
    return indexer_->Hash(maybe_residual, result);
  } else {
    return indexer_->HashWithNoiseShaping(
        maybe_residual, original, result,
        {.threshold = noise_shaping_threshold});
  }
}

SCANN_INSTANTIATE_TYPED_CLASS(, Searcher);

}  // namespace asymmetric_hashing2
}  // namespace research_scann
