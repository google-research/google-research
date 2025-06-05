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

#ifndef LINK_CUT_TREE_H_
#define LINK_CUT_TREE_H_

#include <algorithm>
#include <vector>

namespace geo_algorithms {

// Copied from Bard
class LCTNode {
 public:
  LCTNode(int value) : value(value) {
    child[0] = child[1] = parent = nullptr;
    flip = false;
  }

  void push();

  void attach(int d, LCTNode *y);

  LCTNode *child[2];
  LCTNode *parent;
  int value;  // You can store any value here
  bool flip;
};

class LinkCutTree {
 public:
  LinkCutTree(int n);

  void link(int u, int v);

  void cut(int u);

  int find_root(int u);

 private:
  int dir(LCTNode *x);

  void rotate(LCTNode *x);

  void splay(LCTNode *x);

  LCTNode *access(LCTNode *x);

  void make_root(LCTNode *x);

  std::vector<LCTNode> T;
};

}  // namespace geo_algorithms

#endif  // LINK_CUT_TREE_H_
