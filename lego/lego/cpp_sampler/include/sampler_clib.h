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

#ifndef SAMPLER_CLIB
#define SAMPLER_CLIB


extern "C" void load_binary_file(const char* fname, void* _mem_ptr,
                                 unsigned long long n_ints, // NOLINT
                                 const char* dtype);

extern "C" void load_kg_from_binary(void* _kg_ptr, void* mem_ptr,
                                    unsigned long long n_ints,  // NOLINT
                                    const char* dtype);

extern "C" void load_kg_from_numpy(void* _kg_ptr, void* triplets,
                                   long long n_triplest,  // NOLINT
                                   bool has_reverse_edges, const char* dtype);

#endif
