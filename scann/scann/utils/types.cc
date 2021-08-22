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

#include "scann/utils/types.h"

#include "absl/flags/flag.h"

ABSL_RETIRED_FLAG(
    bool, experimental_use_fast_top_neighbors, true,
    "RETIRED!  Previously:  If true, uses FastTopNeighbors in "
    "traditional ScaNN server.  FastTopNeighbors is always used in DR server.");

namespace research_scann {

Status DisabledTypeError(TypeTag type_tag) {
  return FailedPreconditionError(
      "The '%s' type (type_tag=%d) has been disabled with the "
      "-DSCANN_DISABLE_UNCOMMON_TYPES compile-time flag. Recompile without "
      "this "
      "flag if you wish to use types other than {float, uint8}",
      TypeNameFromTag(type_tag), type_tag);
}

StatusOr<TypeTag> TypeTagFromName(string_view type_name) {
  const std::string type_name_lc = absl::AsciiStrToLower(type_name);

  if (type_name_lc == "float") {
    return InputOutputConfig::FLOAT;
  }
  if (type_name_lc == "uint8") {
    return InputOutputConfig::UINT8;
  }
  auto err_if_uncommon_disabled = [&](TypeTag tag) {
#ifdef SCANN_DISABLE_UNCOMMON_TYPES
    return DisabledTypeError(tag);
#else
    return tag;
#endif
  };

  if (type_name_lc == "int8") {
    return err_if_uncommon_disabled(InputOutputConfig::INT8);
  }
  if (type_name_lc == "int16") {
    return err_if_uncommon_disabled(InputOutputConfig::INT16);
  }
  if (type_name_lc == "uint16") {
    return err_if_uncommon_disabled(InputOutputConfig::UINT16);
  }
  if (type_name_lc == "int32") {
    return err_if_uncommon_disabled(InputOutputConfig::INT32);
  }
  if (type_name_lc == "uint32") {
    return err_if_uncommon_disabled(InputOutputConfig::UINT32);
  }
  if (type_name_lc == "int64") {
    return err_if_uncommon_disabled(InputOutputConfig::INT64);
  }
  if (type_name_lc == "uint64") {
    return err_if_uncommon_disabled(InputOutputConfig::UINT64);
  }
  if (type_name_lc == "double") {
    return err_if_uncommon_disabled(InputOutputConfig::DOUBLE);
  }
  return InvalidArgumentError(
      absl::StrCat("Invalid type name: '", type_name, "'"));
}

string_view TypeNameFromTag(TypeTag type_tag) {
  switch (type_tag) {
    case InputOutputConfig::IN_MEMORY_DATA_TYPE_NOT_SPECIFIED:
      return "NoValue";
    case InputOutputConfig::INT8:
      return "int8";
    case InputOutputConfig::UINT8:
      return "uint8";
    case InputOutputConfig::INT16:
      return "int16";
    case InputOutputConfig::UINT16:
      return "uint16";
    case InputOutputConfig::INT32:
      return "int32";
    case InputOutputConfig::UINT32:
      return "uint32";
    case InputOutputConfig::INT64:
      return "int64";
    case InputOutputConfig::UINT64:
      return "uint64";
    case InputOutputConfig::FLOAT:
      return "float";
    case InputOutputConfig::DOUBLE:
      return "double";
    default:
      return "INVALID_SCANN_TYPE_TAG";
  }
}

}  // namespace research_scann
