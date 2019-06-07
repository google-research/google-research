#ifndef THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_EDF_FILE_H_
#define THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_EDF_FILE_H_

#include <string>

#include "edf/base/status.h"

namespace eeg_modelling {

// Interface to bridge open source and google internal file APIs.
class EdfFile {
 public:
  EdfFile();
  virtual ~EdfFile();

  // Tells the current position of the file, -1 in case
  virtual size_t Tell() const = 0;
  // Seek from the beginning of the file.
  virtual Status SeekFromBegin(size_t position) = 0;
  virtual size_t Read(void* ptr, size_t n) const = 0;
  virtual size_t ReadToString(std::string* str, size_t n) const = 0;
  virtual size_t Write(const void* ptr, size_t n) = 0;
};

}  // namespace eeg_modelling

#endif  // THIRD_PARTY_GOOGLE_RESEARCH_GOOGLE_RESEARCH_EEG_MODELLING_EDF_EDF_FILE_H_
