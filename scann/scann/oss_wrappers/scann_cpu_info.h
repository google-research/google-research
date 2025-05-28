// Copyright 2025 The Google Research Authors.
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



#ifndef SCANN_OSS_WRAPPERS_SCANN_CPU_INFO_H_
#define SCANN_OSS_WRAPPERS_SCANN_CPU_INFO_H_

#include <string>

#if defined(_MSC_VER)

#include <intrin.h>
#endif

namespace research_scann {
namespace port {

int NumSchedulableCPUs();

int MaxParallelism();

int MaxParallelism(int numa_node);

static constexpr int kUnknownCPU = -1;
int NumTotalCPUs();

int GetCurrentCPU();

int NumHyperthreadsPerCore();

enum CPUFeature {

  MMX = 0,
  SSE = 1,
  SSE2 = 2,
  SSE3 = 3,
  SSSE3 = 4,
  SSE4_1 = 5,
  SSE4_2 = 6,
  CMOV = 7,
  CMPXCHG8B = 8,
  CMPXCHG16B = 9,
  POPCNT = 10,
  AES = 11,
  AVX = 12,
  RDRAND = 13,
  AVX2 = 14,
  FMA = 15,
  F16C = 16,
  PCLMULQDQ = 17,
  RDSEED = 18,
  ADX = 19,
  SMAP = 20,

  PREFETCHWT1 = 21,

  BMI1 = 22,
  BMI2 = 23,
  HYPERVISOR = 25,

  PREFETCHW = 26,

  AVX512F = 27,
  AVX512CD = 28,
  AVX512ER = 29,
  AVX512PF = 30,
  AVX512VL = 31,
  AVX512BW = 32,
  AVX512DQ = 33,
  AVX512VBMI = 34,
  AVX512IFMA = 35,
  AVX512_4VNNIW = 36,
  AVX512_4FMAPS = 37,
  AVX512_VNNI = 38,
  AVX512_BF16 = 39,

  AVX_VNNI = 40,

  AMX_TILE = 41,
  AMX_INT8 = 42,
  AMX_BF16 = 43,

  AVX512_FP16 = 44,
  AMX_FP16 = 45,
  AVX_NE_CONVERT = 46,
  AVX_VNNI_INT8 = 47,
};

enum Aarch64CPU {
  ARM_NEOVERSE_N1 = 0,
  ARM_NEOVERSE_V1 = 1,
};

bool TestAarch64CPU(Aarch64CPU cpu);

bool TestCPUFeature(CPUFeature feature);

constexpr bool IsX86CPU() {
#ifdef PLATFORM_IS_X86
  return true;
#else
  return false;
#endif
}

constexpr bool IsAarch64CPU() {
#if defined(PLATFORM_IS_ARM64) && !defined(__APPLE__) && !defined(__OpenBSD__)
  return true;
#else
  return false;
#endif
}

std::string CPUVendorIDString();

int CPUFamily();

int CPUModelNum();

double NominalCPUFrequency();

int CPUIDNumSMT();

}  // namespace port
}  // namespace research_scann

#endif
