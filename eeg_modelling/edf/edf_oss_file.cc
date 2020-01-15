#include "edf/edf_oss_file.h"

#include <memory>

#include "absl/base/internal/raw_logging.h"
#include "edf/base/status.h"

namespace eeg_modelling {

EdfOssFile::EdfOssFile(FILE* fp) : fp_(fp) {}

EdfOssFile::~EdfOssFile() { fclose(fp_); }

EdfOssFile::EdfOssFile(const char* filename, const char* mode) {
  fp_ = fopen(filename, mode);
  if (fp_ == nullptr) {
    ABSL_RAW_LOG(FATAL, "File open failed %s", filename);
  }
}

size_t EdfOssFile::Tell() const { return std::ftell(fp_); }

Status EdfOssFile::SeekFromBegin(size_t position) {
  auto ret = std::fseek(fp_, position, SEEK_SET);
  if (ret != 0) {
    return Status(StatusCode::kUnknown, "Seek failed");
  }
  return OkStatus();
}

size_t EdfOssFile::Read(void* ptr, size_t n) const {
  return std::fread(ptr, 1, n, fp_);
}

size_t EdfOssFile::ReadToString(string* str, size_t n) const {
  std::unique_ptr<char[]> buf(new char[n]);
  auto bytes_read = std::fread(buf.get(), 1, n, fp_);
  if (bytes_read <= 0) {
    return bytes_read;
  }
  str->assign(buf.get(), bytes_read);
  return bytes_read;
}

size_t EdfOssFile::Write(const void* ptr, size_t n) {
  return std::fwrite(ptr, 1, n, fp_);
}

}  // namespace eeg_modelling
