// Copyright 2020 The Google Research Authors.
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

#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_DATASET_UTIL_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_DATASET_UTIL_H_

#include <array>
#include <random>
#include <type_traits>

#include "file/base/path.h"
#include "dataset.h"
#include "datasets.proto.h"
#include "definitions.h"
#include "executor.h"
#include "generator.h"
#include "memory.h"
#include "random_generator.h"
#include "sstable/public/sstable.h"

namespace automl_zero {

using ::std::shuffle;  // NOLINT
using ::std::vector;  // NOLINT

const RandomSeedT kFirstParamSeedForTest = 9000;
const RandomSeedT kFirstDataSeedForTest = 19000;

// Fills `return_datasets` with `experiment_datasets` Datasets.
// `return_datasets` must be empty an empty vector.
void FillDatasets(
    const DatasetCollection& dataset_collection,
    std::vector<std::unique_ptr<DatasetInterface>>* return_datasets);

// Downcasts a DatasetInterface. Crashes if the downcast would have been
// incorrect.
template <FeatureIndexT F>
Dataset<F>* SafeDowncast(DatasetInterface* dataset) {
  CHECK(dataset != nullptr);
  CHECK_EQ(dataset->FeaturesSize(), F);
  return dynamic_cast<Dataset<F>*>(dataset);
}
template <FeatureIndexT F>
const Dataset<F>* SafeDowncast(const DatasetInterface* dataset) {
  CHECK(dataset != nullptr);
  CHECK_EQ(dataset->FeaturesSize(), F);
  return dynamic_cast<const Dataset<F>*>(dataset);
}
template <FeatureIndexT F>
std::unique_ptr<Dataset<F>> SafeDowncast(
    std::unique_ptr<DatasetInterface> dataset) {
  CHECK(dataset != nullptr);
  return std::unique_ptr<Dataset<F>>(SafeDowncast<F>(dataset.release()));
}

namespace test_only {

// Convenience method to generates a single dataset from a string containing a
// text-format DatasetSpec. Only generates 1 dataset, so `num_datasets` can be
// 1 or can be left unset. The features size is determined by the template
// argument `F`, so the `features_size` in the DatasetSpec can be equal to `F`
// or be left unset.
template <FeatureIndexT F>
Dataset<F> GenerateDataset(const std::string& dataset_spec_str) {
  DatasetCollection dataset_collection;
  DatasetSpec* dataset_spec = dataset_collection.add_datasets();
  CHECK(proto2::TextFormat::ParseFromString(dataset_spec_str, dataset_spec));
  if (!dataset_spec->has_features_size()) {
    dataset_spec->set_features_size(F);
  }
  CHECK_EQ(dataset_spec->features_size(), F);
  if (!dataset_spec->has_num_datasets()) {
    dataset_spec->set_num_datasets(1);
  }
  CHECK_EQ(dataset_spec->num_datasets(), 1);
  std::vector<std::unique_ptr<DatasetInterface>> datasets;
  FillDatasets(dataset_collection, &datasets);
  CHECK_EQ(datasets.size(), 1);

  // TODO(ereal): remove the std::move hack.
  std::unique_ptr<Dataset<F>> dataset = SafeDowncast<F>(std::move(datasets[0]));
  return std::move(*dataset);
}

}  // namespace test_only

template <FeatureIndexT F>
struct ClearAndResizeImpl {
  static void Call(const IntegerT num_train_examples,
                   const IntegerT num_valid_examples,
                   DatasetBuffer<F>* buffer) {
    buffer->train_features_.clear();
    buffer->train_labels_.clear();
    buffer->valid_features_.clear();
    buffer->valid_labels_.clear();
    buffer->train_features_.resize(num_train_examples);
    buffer->train_labels_.resize(num_train_examples);
    buffer->valid_features_.resize(num_valid_examples);
    buffer->valid_labels_.resize(num_valid_examples);

    for (Vector<F>& value : buffer->train_features_) {
      value.resize(F, 1);
    }
    for (Vector<F>& value : buffer->valid_features_) {
      value.resize(F, 1);
    }
  }
};

template <FeatureIndexT F>
void ClearAndResize(const IntegerT num_train_examples,
                    const IntegerT num_valid_examples,
                    DatasetBuffer<F>* buffer) {
  ClearAndResizeImpl<F>::Call(num_train_examples, num_valid_examples, buffer);
}

// Creates a dataset using the linear regressor with fixed weights. The
// weights are determined by the seed. Serves as a way to initialize the
// dataset.
template <FeatureIndexT F>
struct ScalarLinearRegressionDatasetCreator {
  static void Create(EvalType eval_type, IntegerT num_train_examples,
                     IntegerT num_valid_examples, RandomSeedT param_seed,
                     RandomSeedT data_seed, DatasetBuffer<F>* buffer) {
    ClearAndResize(num_train_examples, num_valid_examples, buffer);
    std::mt19937 data_bit_gen(data_seed + 939723201);
    RandomGenerator data_gen = RandomGenerator(&data_bit_gen);
    std::mt19937 param_bit_gen(param_seed + 997958712);
    RandomGenerator weights_gen = RandomGenerator(&param_bit_gen);
    Generator generator(NO_OP_ALGORITHM, 0, 0, 0, {}, {}, {}, nullptr,
                        nullptr);

    // Fill the features.
    for (Vector<F>& features : buffer->train_features_) {
      data_gen.FillGaussian<F>(0.0, 1.0, &features);
    }
    for (Vector<F>& features : buffer->valid_features_) {
      data_gen.FillGaussian<F>(0.0, 1.0, &features);
    }

    // Create a Algorithm and memory deterministically.
    Algorithm algorithm = generator.LinearModel(0.0);
    Memory<F> memory;
    memory.Wipe();
    weights_gen.FillGaussian<F>(
        0.0, 1.0, &memory.vector_[Generator::LINEAR_ALGORITHMWeightsAddress]);

    // Fill in the labels by executing the Algorithm.
    ExecuteAndFillLabels<F>(algorithm, &memory, buffer, &data_gen);
    CHECK_EQ(eval_type, RMS_ERROR);
  }
};

// Creates a dataset using the nonlinear regressor with fixed weights. The
// weights are determined by the seed. Serves as a way to initialize the
// dataset.
template <FeatureIndexT F>
struct Scalar2LayerNnRegressionDatasetCreator {
  static void Create(EvalType eval_type, IntegerT num_train_examples,
                     IntegerT num_valid_examples, RandomSeedT param_seed,
                     RandomSeedT data_seed, DatasetBuffer<F>* buffer) {
    ClearAndResize(num_train_examples, num_valid_examples, buffer);
    std::mt19937 data_bit_gen(data_seed + 865546086);
    RandomGenerator data_gen = RandomGenerator(&data_bit_gen);
    std::mt19937 param_bit_gen(param_seed + 174299604);
    RandomGenerator weights_gen = RandomGenerator(&param_bit_gen);
    Generator generator(NO_OP_ALGORITHM, 0, 0, 0, {}, {}, {}, nullptr,
                        nullptr);

    // Fill the features.
    for (Vector<F>& features : buffer->train_features_) {
      data_gen.FillGaussian<F>(0.0, 1.0, &features);
    }
    for (Vector<F>& features : buffer->valid_features_) {
      data_gen.FillGaussian<F>(0.0, 1.0, &features);
    }

    // Create a Algorithm and memory deterministically.
    Algorithm algorithm = generator.UnitTestNeuralNetNoBiasNoGradient(0.0);
    Memory<F> memory;
    memory.Wipe();
    weights_gen.FillGaussian<F>(
        0.0, 1.0, &memory.matrix_[
        Generator::kUnitTestNeuralNetNoBiasNoGradientFirstLayerWeightsAddress]);
    for (FeatureIndexT col = 0; col < F; ++col) {
      memory.matrix_[
          Generator::kUnitTestNeuralNetNoBiasNoGradientFirstLayerWeightsAddress]
          (0, col) = 0.0;
      memory.matrix_[
          Generator::kUnitTestNeuralNetNoBiasNoGradientFirstLayerWeightsAddress]
          (2, col) = 0.0;
    }
    weights_gen.FillGaussian<F>(
        0.0, 1.0,
        &memory.vector_[
        Generator::kUnitTestNeuralNetNoBiasNoGradientFinalLayerWeightsAddress]);
    memory.vector_[
        Generator::kUnitTestNeuralNetNoBiasNoGradientFinalLayerWeightsAddress]
        (0) = 0.0;
    memory.vector_[
        Generator::kUnitTestNeuralNetNoBiasNoGradientFinalLayerWeightsAddress]
        (2) = 0.0;

    // Fill in the labels by executing the Algorithm.
    ExecuteAndFillLabels<F>(algorithm, &memory, buffer, &data_gen);
    CHECK_EQ(eval_type, RMS_ERROR);
  }
};

// TODO(crazydonkey): make this function more readable.
template <FeatureIndexT F>
struct ProjectedBinaryClassificationDatasetCreator {
  static void Create(EvalType eval_type,
                     const ProjectedBinaryClassificationDataset& dataset_spec,
                     IntegerT num_train_examples, IntegerT num_valid_examples,
                     IntegerT features_size, RandomSeedT data_seed,
                     DatasetBuffer<F>* buffer) {
    ClearAndResize(num_train_examples, num_valid_examples, buffer);

    std::string dump_path;
    CHECK(dataset_spec.has_dump_path() & !dataset_spec.dump_path().empty())
        << "You have to specifiy the path to the dataset!" << std::endl;
    dump_path = dataset_spec.dump_path();

    std::unique_ptr<SSTable> sstable(
        SSTable::Open(
            dump_path,
            SSTable::ON_DISK()));

    IntegerT positive_class;
    IntegerT negative_class;

    if (dataset_spec.has_positive_class() &&
        dataset_spec.has_negative_class()) {
      positive_class = dataset_spec.positive_class();
      negative_class = dataset_spec.negative_class();
      IntegerT num_supported_data_seeds =
          dataset_spec.max_supported_data_seed() -
          dataset_spec.min_supported_data_seed();
      data_seed = static_cast<RandomSeedT>(
          dataset_spec.min_supported_data_seed() +
          data_seed % num_supported_data_seeds);
    } else if (!dataset_spec.has_positive_class() &&
               !dataset_spec.has_negative_class()) {
      std::mt19937 dataset_bit_gen(CustomHashMix(
          static_cast<RandomSeedT>(856572777), data_seed));
      RandomGenerator dataset_gen = RandomGenerator(
          &dataset_bit_gen);

      std::set<std::pair<IntegerT, IntegerT>> held_out_pairs_set;
      for (ClassPair class_pair : dataset_spec.held_out_pairs()) {
        held_out_pairs_set.insert(std::pair<IntegerT, IntegerT>(
            std::min(class_pair.positive_class(),
                     class_pair.negative_class()),
            std::max(class_pair.positive_class(),
                     class_pair.negative_class())));
      }

      std::vector<std::pair<IntegerT, IntegerT>> search_pairs;
      // Assumming the classes are in [0, 10).
      for (IntegerT i = 0; i < 10; i++) {
        for (IntegerT j = i+1; j < 10; j++){
          std::pair<IntegerT, IntegerT> class_pair(i, j);
          // Collect all the pairs that is not held out.
          if (held_out_pairs_set.count(class_pair) == 0)
            search_pairs.push_back(class_pair);
        }
      }

      CHECK(!search_pairs.empty())
          << "All the pairs are held out!" << std::endl;

      std::pair<IntegerT, IntegerT> selected_pair =
          search_pairs[dataset_gen.UniformInteger(0, (search_pairs.size()))];
      positive_class = selected_pair.first;
      negative_class = selected_pair.second;
      data_seed = static_cast<RandomSeedT>(
          dataset_gen.UniformInteger(
              dataset_spec.min_supported_data_seed(),
              dataset_spec.max_supported_data_seed()));
    } else {
      LOG(FATAL) << ("You should either provide both or none of the positive"
                     " and negative classes.") << std::endl;
    }

    // Generate the key using the dataset_spec.
    std::string key = absl::StrCat(dataset_spec.dataset_name(), "-pos_",
                                   positive_class, "-neg_", negative_class,
                                   "-dim_", features_size, "-seed_", data_seed);

    // Create SSTable::Iterator object and Seek() to the search term in the
    // SSTable.
    std::unique_ptr<SSTable::Iterator> iter(sstable->GetIterator());
    iter->Seek(key);
    ProjectedBinaryClassificationDataset saved_dataset;
    while (!iter->done() && (iter->key() == key)) {
      CHECK(saved_dataset.ParseFromString(iter->value())) << iter->key();
      iter->Next();
    }

    // Check there is enough data saved in the sstable.
    CHECK_GE(saved_dataset.dumped_dataset().train_features_size(),
             buffer->train_features_.size());
    CHECK_GE(saved_dataset.dumped_dataset().train_labels_size(),
             buffer->train_labels_.size());

    for (IntegerT k = 0; k < buffer->train_features_.size(); ++k)  {
      for (IntegerT i_dim = 0; i_dim < F; ++i_dim) {
       buffer->train_features_[k][i_dim] =
           saved_dataset.dumped_dataset().train_features(k).features(i_dim);
      }
      buffer->train_labels_[k] =
           saved_dataset.dumped_dataset().train_labels(k);
    }

    CHECK_GE(saved_dataset.dumped_dataset().valid_features_size(),
             buffer->valid_features_.size());
    CHECK_GE(saved_dataset.dumped_dataset().valid_labels_size(),
             buffer->valid_labels_.size());
    for (IntegerT k = 0; k < buffer->valid_features_.size(); ++k)  {
      for (IntegerT i_dim = 0; i_dim < F; ++i_dim) {
       buffer->valid_features_[k][i_dim] =
           saved_dataset.dumped_dataset().valid_features(k).features(i_dim);
      }
      buffer->valid_labels_[k] =
           saved_dataset.dumped_dataset().valid_labels(k);
    }

    CHECK(eval_type == ACCURACY);
  }
};

template<FeatureIndexT F>
void CopyUnitTestFixedDatasetVector(
    const proto2::RepeatedField<double>& src, Scalar* dest) {
  LOG(FATAL) << "Not allowed." << std::endl;
}
template<FeatureIndexT F>
void CopyUnitTestFixedDatasetVector(
    const proto2::RepeatedField<double>& src, Vector<F>* dest) {
  CHECK_EQ(src.size(), F);
  for (IntegerT index = 0; index < F; ++index) {
    (*dest)(index) = src.at(index);
  }
}

template <FeatureIndexT F>
struct UnitTestFixedDatasetCreator {
  static void Create(const UnitTestFixedDataset& dataset_spec,
                     DatasetBuffer<F>* buffer) {
    const IntegerT num_train_examples = dataset_spec.train_features_size();
    CHECK_EQ(dataset_spec.train_labels_size(), num_train_examples);
    const IntegerT num_valid_examples = dataset_spec.valid_features_size();
    CHECK_EQ(dataset_spec.valid_labels_size(), num_valid_examples);
    ClearAndResize(num_train_examples, num_valid_examples, buffer);

    // Copy the training features and labels.
    for (IntegerT example = 0; example < num_train_examples; ++example) {
      CopyUnitTestFixedDatasetVector<F>(
          dataset_spec.train_features(example).elements(),
          &buffer->train_features_[example]);
      CHECK_EQ(dataset_spec.train_labels(example).elements_size(), 1);
      buffer->train_labels_[example] =
          dataset_spec.train_labels(example).elements(0);
    }

    // Copy the validation features and labels.
    for (IntegerT example = 0; example < num_valid_examples; ++example) {
      CopyUnitTestFixedDatasetVector<F>(
          dataset_spec.valid_features(example).elements(),
          &buffer->valid_features_[example]);
      CHECK_EQ(dataset_spec.valid_labels(example).elements_size(), 1);
      buffer->valid_labels_[example] =
          dataset_spec.valid_labels(example).elements(0);
    }
  }
};

template <FeatureIndexT F>
struct UnitTestZerosDatasetCreator {
  static void Create(const IntegerT num_train_examples,
                     const IntegerT num_valid_examples,
                     const UnitTestZerosDatasetSpec& dataset_spec,
                     DatasetBuffer<F>* buffer) {
    ClearAndResize(num_train_examples, num_valid_examples, buffer);
    for (Vector<F>& feature : buffer->train_features_) {
      feature.setZero();
    }
    for (Scalar& label : buffer->train_labels_) {
      label = 0.0;
    }
    for (Vector<F>& feature : buffer->valid_features_) {
      feature.setZero();
    }
    for (Scalar& label : buffer->valid_labels_) {
      label = 0.0;
    }
  }
};

template <FeatureIndexT F>
struct UnitTestOnesDatasetCreator {
  static void Create(const IntegerT num_train_examples,
                     const IntegerT num_valid_examples,
                     const UnitTestOnesDatasetSpec& dataset_spec,
                     DatasetBuffer<F>* buffer) {
    ClearAndResize(num_train_examples, num_valid_examples, buffer);
    for (Vector<F>& feature : buffer->train_features_) {
      feature.setOnes();
    }
    for (Scalar& label : buffer->train_labels_) {
      label = 1.0;
    }
    for (Vector<F>& feature : buffer->valid_features_) {
      feature.setOnes();
    }
    for (Scalar& label : buffer->valid_labels_) {
      label = 1.0;
    }
  }
};

template <FeatureIndexT F>
struct UnitTestIncrementDatasetCreator {
  static void Create(const IntegerT num_train_examples,
                     const IntegerT num_valid_examples,
                     const UnitTestIncrementDatasetSpec& dataset_spec,
                     DatasetBuffer<F>* buffer) {
    ClearAndResize(num_train_examples, num_valid_examples, buffer);

    const double increment = dataset_spec.increment();
    Scalar incrementing_scalar = 0.0;
    Vector<F> incrementing_vector = Vector<F>::Zero(F, 1);
    Vector<F> ones_vector = Vector<F>::Ones(F, 1);
    ones_vector *= increment;

    for (Vector<F>& feature : buffer->train_features_) {
      feature = incrementing_vector;
      incrementing_vector += ones_vector;
    }
    for (Scalar& label : buffer->train_labels_) {
      label = incrementing_scalar;
      incrementing_scalar += increment;
    }

    incrementing_scalar = 0.0;
    incrementing_vector.setZero();

    for (Vector<F>& feature : buffer->valid_features_) {
      feature = incrementing_vector;
      incrementing_vector += ones_vector;
    }
    for (Scalar& label : buffer->valid_labels_) {
      label = incrementing_scalar;
      incrementing_scalar += increment;
    }
  }
};

template <FeatureIndexT F>
std::unique_ptr<Dataset<F>> CreateDataset(const IntegerT dataset_index,
                                          const RandomSeedT param_seed,
                                          const RandomSeedT data_seed,
                                          const DatasetSpec& dataset_spec) {
  CHECK_GT(dataset_spec.num_train_examples(), 0);
  CHECK_GT(dataset_spec.num_valid_examples(), 0);
  DatasetBuffer<F> buffer;
  switch (dataset_spec.dataset_type_case()) {
    case (DatasetSpec::kScalarLinearRegressionDataset):
      ScalarLinearRegressionDatasetCreator<F>::Create(
          dataset_spec.eval_type(), dataset_spec.num_train_examples(),
          dataset_spec.num_valid_examples(), param_seed, data_seed, &buffer);
      break;
    case (DatasetSpec::kScalar2LayerNnRegressionDataset):
      Scalar2LayerNnRegressionDatasetCreator<F>::Create(
          dataset_spec.eval_type(), dataset_spec.num_train_examples(),
          dataset_spec.num_valid_examples(), param_seed, data_seed, &buffer);
      break;
    case (DatasetSpec::kProjectedBinaryClassificationDataset):
      ProjectedBinaryClassificationDatasetCreator<F>::Create(
          dataset_spec.eval_type(),
          dataset_spec.projected_binary_classification_dataset(),
          dataset_spec.num_train_examples(), dataset_spec.num_valid_examples(),
          dataset_spec.features_size(), data_seed, &buffer);
      break;
    case (DatasetSpec::kUnitTestFixedDataset):
      UnitTestFixedDatasetCreator<F>::Create(
          dataset_spec.unit_test_fixed_dataset(), &buffer);
      break;
    case (DatasetSpec::kUnitTestZerosDataset):
      UnitTestZerosDatasetCreator<F>::Create(
          dataset_spec.num_train_examples(),
          dataset_spec.num_valid_examples(),
          dataset_spec.unit_test_zeros_dataset(),
          &buffer);
      break;
    case (DatasetSpec::kUnitTestOnesDataset):
      UnitTestOnesDatasetCreator<F>::Create(
          dataset_spec.num_train_examples(),
          dataset_spec.num_valid_examples(),
          dataset_spec.unit_test_ones_dataset(),
          &buffer);
      break;
    case (DatasetSpec::kUnitTestIncrementDataset):
      UnitTestIncrementDatasetCreator<F>::Create(
          dataset_spec.num_train_examples(), dataset_spec.num_valid_examples(),
          dataset_spec.unit_test_increment_dataset(), &buffer);
      break;
    default:
      LOG(FATAL) << "Unknown dataset type\n";
      break;
  }

  std::mt19937 data_bit_gen(data_seed + 3274582109);
  CHECK_EQ(buffer.train_features_.size(), dataset_spec.num_train_examples());
  CHECK_EQ(buffer.train_labels_.size(), dataset_spec.num_train_examples());
  CHECK_EQ(buffer.valid_features_.size(), dataset_spec.num_valid_examples());
  CHECK_EQ(buffer.valid_labels_.size(), dataset_spec.num_valid_examples());

  CHECK(dataset_spec.has_eval_type());
  return absl::make_unique<Dataset<F>>(
      dataset_index, dataset_spec.eval_type(),
      dataset_spec.num_train_epochs(), &data_bit_gen, &buffer);
}

// Randomizes all the seeds given a base seed. See "internal workflow" comment
// in datasets.proto.
// TODO(crazydonkey): make sure the random seed is never 0.
void RandomizeDatasetSeeds(DatasetCollection* dataset_collection,
                           RandomSeedT seed);

}  // namespace automl_zero

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_AUTOML_ZERO_DATASET_UTIL_H_
