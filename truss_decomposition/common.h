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

#ifndef COMMON_H
#define COMMON_H

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CHECK(x)                                                              \
  do {                                                                        \
    if (!(x)) {                                                               \
      fprintf(stderr, "Assertion failure at %s:%d: %s\n", __FILE__, __LINE__, \
              #x);                                                            \
      abort();                                                                \
    }                                                                         \
  } while (0);

#define CHECK_NE(a, b) CHECK(a != b)

#define CHECK_M(x, msg)                                                \
  do {                                                                 \
    if (!(x)) {                                                        \
      fprintf(stderr, "Assertion failure at %s:%d: %s %s\n", __FILE__, \
              __LINE__, #x, msg);                                      \
      abort();                                                         \
    }                                                                  \
  } while (0);

inline void FullFence() {
  std::atomic_thread_fence(std::memory_order_seq_cst);
  asm volatile("" : : : "memory");
#pragma omp barrier
}

template <typename T>
void ReadBinaryOrDie(FILE *f, T *out, size_t count = 1) {
  fread(out, sizeof(T), count, f);
}

template <typename T>
T ReadBinaryOrDie(FILE *f) {
  T ans;
  ReadBinaryOrDie(f, &ans);
  return ans;
}

template <typename T>
T ReadBase10Fast(FILE *f) {
  T n = 0;
  int ch = fgetc_unlocked(f);
  while (ch != EOF && !('0' <= ch && ch <= '9')) ch = fgetc_unlocked(f);
  if (ch == EOF) return EOF;
  while ('0' <= ch && ch <= '9') {
    n = 10 * n + ch - '0';
    ch = fgetc_unlocked(f);
  }
  return n;
}

class FastWriter {
  static constexpr size_t kBufSize = 1 << 16;

 public:
  FastWriter(const char *filename) { CHECK(out_ = fopen(filename, "w")); }
  FastWriter(int fd) { CHECK(out_ = fdopen(fd, "w")); }
  ~FastWriter() {
    Flush();
    CHECK_M(fclose(out_) == 0, strerror(errno));
  }

  FastWriter(const FastWriter &) = delete;
  FastWriter(FastWriter &&) = delete;

  void Flush() {
    if (buf_position_ > 0) {
      CHECK_M(fwrite(buf_, 1, buf_position_, out_), strerror(errno));
    }
    buf_position_ = 0;
  }

  void AddToBuffer(char c) {
    buf_[buf_position_++] = c;
    if (buf_position_ >= kBufSize) {
      Flush();
    }
  }

  void AddToBuffer(const char *data, size_t len) {
    for (size_t i = 0; i < len; i++) {
      buf_[buf_position_++] = data[i];
      if (buf_position_ >= kBufSize) {
        Flush();
      }
    }
  }

  void AddToBuffer(const std::string &s) { AddToBuffer(s.c_str(), s.size()); }

  template <typename T>
  void Write(const T &param) {
    AddToBuffer(std::to_string(param));
  }

  void Write(const char param) { AddToBuffer(param); }

 private:
  FILE *out_ = nullptr;  // owned

  char buf_[kBufSize] = {};
  size_t buf_position_ = 0;
};

std::string FileFromStorageDir(const std::string &storage_dir,
                               const std::string &filename) {
  if (storage_dir.empty()) return "";
  return storage_dir + "/" + filename;
}

#endif
