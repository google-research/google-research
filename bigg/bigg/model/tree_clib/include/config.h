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

#ifndef cfg_H
#define cfg_H

#include <iostream>
#include <cstring>
#include <random>
#include <fstream>
#include <set>
#include <map>

typedef float Dtype;

struct cfg
{
    static int max_num_nodes;
    static bool directed, self_loop, bfs_permute;
    static int bits_compress;
    static int dim_embed;
    static int gpu;
    static int seed;

    static std::default_random_engine generator;

    static void LoadParams(const int argc, const char** argv);

    static void SetRandom();
};

#endif
