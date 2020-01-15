#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_EDF_OSS_FILE_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_EDF_OSS_FILE_H_

#include <string>

#include "absl/strings/string_view.h"
#include "edf/base/status.h"
#include "edf/edf_file.h"

using std::string;

namespace eeg_modelling {

class EdfOssFile : public EdfFile {
 public:
  EdfOssFile(FILE* fp);

  EdfOssFile(const char* filename, const char* mode);

  ~EdfOssFile();

  size_t Tell() const;

  Status SeekFromBegin(size_t position);

  size_t Read(void* ptr, size_t n) const;

  size_t ReadToString(string* str, size_t n) const;

  size_t Write(const void* ptr, size_t n);

 private:
  FILE* fp_;
};

}  // namespace eeg_modelling

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_EDF_OSS_FILE_H_
