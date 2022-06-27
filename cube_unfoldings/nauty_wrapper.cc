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

#include "nauty_wrapper.h"

#define MAXN WORDSIZE /* Define this before including nauty.h */

#include "nauty/gtools.h"
#include "nauty/nauty.h"

namespace cube_unfoldings {
namespace {
// Workaround for nauty's inherently thread-unsafe callback model.
static std::vector<Generator>*& Generators() {
  thread_local std::vector<Generator>* store;
  return store;
}

// g has n nodes and m edges.
std::vector<Edge> GetAllEdges(graph* g, int m, int n) {
  std::vector<Edge> edges;
  edges.reserve(m);

  int v, w;
  for (v = 0; v < n; ++v) {
    for (w = v + 1; w < n; ++w) {
      if ((g[v] >> (m * WORDSIZE - 1 - w)) % 2) {
        edges.push_back({v, w});
      }
    }
  }
  return edges;
}
}  // namespace

TreeData TreeData::FromString(const char* tree) {
  TreeData ret;
  Generators() = &ret.generators;
  graph g[MAXN * MAXM] = {};
  int lab[MAXN], ptn[MAXN], orbits[MAXN];
  DEFAULTOPTIONS_GRAPH(options);

  options.userautomproc = [](int count, int* perm, int* orbits, int numorbits,
                             int stabvertex, int n) {
    Generators()->emplace_back(perm, perm + n);
  };
  statsblk stats;
  int n, m;
  size_t e;
  // tree is not actually modified, but the nauty interface is not
  // const-correct.
  stringcounts(const_cast<char*>(tree), &n, &e);
  m = SETWORDSNEEDED(n);
  stringtograph(const_cast<char*>(tree), g, m);

  options.getcanon = FALSE;
  options.defaultptn = TRUE;
  densenauty(g, lab, ptn, orbits, &options, &stats, m, n, nullptr);

  complement(g, m, n);
  ret.complement_edges = GetAllEdges(g, m, n);
  ret.n = n;
  return ret;
}
}  // namespace cube_unfoldings
