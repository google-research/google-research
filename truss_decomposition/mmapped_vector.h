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

#ifndef MMAPPED_VECTOR_HPP
#define MMAPPED_VECTOR_HPP
#include <fcntl.h>
#include <memory.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstring>
#include <iostream>

#include "common.h"  // NOLINT

// vector-like class that may be backed by a file.
// Note that this will create multiple version of various methods for each
// instantiation of the mmapped_vector with a different T. This could be
// avoided by moving methods independent from T outside of the template.
template <typename T>
class mmapped_vector {
 public:
  // Construct and initialize.
  mmapped_vector(std::string backing_file, size_t initial_capacity = 0,
                 bool reserve_only = false) {
    init(std::move(backing_file), initial_capacity, reserve_only);
  }

  // Construct an empty vector not backed by a file.
  mmapped_vector() { init(""); }

  // Initialize the vector. If `backing_file` is the empty string, data will
  // reside in memory. If `reserve_only` is true, the size of the vector is left
  // at 0 (so that `.push_back()` works as expected).
  void init(std::string backing_file, size_t initial_capacity = 0,
            bool reserve_only = false) {
    if (initial_capacity == 0) {
      reserve_only = true;
      initial_capacity = 16;
    }
    clear();
    backing_file_ = std::move(backing_file);
    capacity_ = initial_capacity;
    size_ = 0;
    fd_ = -1;
    data_ = nullptr;
    change_capacity(initial_capacity);
    if (!reserve_only) {
      resize(initial_capacity);
    }
  }

  ~mmapped_vector() { clear(); }

  // std::vector-like interface.
  size_t size() const { return size_; }

  void resize(size_t sz) {
    size_t size = size_;
    resize_without_initializing(sz);
    if (size_ > size) {
      memset(data_ + size, 0, byte_size(size_) - byte_size(size));
    }
  }

  void clear() {
    resize(0);
    shrink();
  }

  void reserve(size_t cap) {
    if (cap > capacity_) {
      change_capacity(cap);
    }
  }

  void shrink() { change_capacity(size()); }

  void push_back(const T &val) {
    resize_without_initializing(size() + 1);
    data_[size() - 1] = val;
  }

  T &operator[](size_t pos) { return data_[pos]; }
  T *data() { return data_; }
  T *begin() { return data(); }
  T *end() { return begin() + size(); }
  T &back() { return (*this)[size() - 1]; }
  T &front() { return *data(); }

  // Move from another vector with a different type of data.
  template <typename U>
  void reinterpret(mmapped_vector<U> &&other) {
    static_assert(sizeof(U) % sizeof(T) == 0, "Invalid reinterpret");
    fd_ = other.fd_;
    backing_file_ = std::move(other.backing_file_);
    data_ = (T *)other.data_;
    size_ = other.size_ * sizeof(U) / sizeof(T);
    capacity_ = other.capacity_ * sizeof(U) / sizeof(T);
    other.fd_ = -1;
    other.data_ = nullptr;
  }

  mmapped_vector(const mmapped_vector &) = delete;
  mmapped_vector(mmapped_vector &&) = delete;
  mmapped_vector operator=(const mmapped_vector &) = delete;
  mmapped_vector operator=(mmapped_vector &&) = delete;

  template <typename U>
  friend class mmapped_vector;

  const std::string &BackingFile() const { return backing_file_; }

 private:
  std::string backing_file_;
  int fd_ = -1;
  T *data_ = nullptr;
  size_t capacity_;
  size_t size_;

  size_t byte_size(size_t cap) { return cap * sizeof(T); }

  void resize_without_initializing(size_t sz) {
    if (sz > capacity_) {
      reserve(std::max(2 * capacity_, sz));
    }
    size_ = sz;
  }

  void change_capacity(size_t cap) {
    if (cap == 0) {
      if (fd_ != -1) {
        CHECK_NE(unlink(backing_file_.c_str()), -1);
      }
      if (data_ != nullptr) {
        CHECK_NE(munmap(data_, byte_size(capacity_)), -1);
      }
      fd_ = -1;
      data_ = nullptr;
      capacity_ = 0;
      return;
    }
    size_t final_capacity = byte_size(cap);
    // No mapping was present, create a new one.
    if (data_ == nullptr) {
      if (!backing_file_.empty()) {
        fd_ = open(backing_file_.c_str(), O_RDWR | O_CREAT, S_IRUSR | S_IWUSR);
        CHECK_NE(fd_, -1);
        CHECK_NE(ftruncate(fd_, final_capacity), -1);
      }
      int mmap_flags =
          backing_file_.empty() ? MAP_ANONYMOUS | MAP_PRIVATE : MAP_SHARED;
      data_ = (T *)mmap(nullptr, final_capacity, PROT_READ | PROT_WRITE,
                        mmap_flags, fd_, 0);
    } else {
      // Resize existing mapping.
      size_t initial_capacity = byte_size(capacity_);
      if (fd_ != -1) {
        CHECK_M(ftruncate(fd_, final_capacity) != -1, strerror(errno));
      }
      data_ =
          (T *)mremap(data_, initial_capacity, final_capacity, MREMAP_MAYMOVE);
    }
    CHECK_NE(data_, MAP_FAILED);
    madvise(data_, final_capacity, MADV_HUGEPAGE);
    capacity_ = cap;
  }
};

#endif
