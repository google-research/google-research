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

#include "link_cut_tree.h"

#include <iostream>

namespace geo_algorithms {

void LCTNode::push() {
  if (!flip) return;
  std::swap(child[0], child[1]);
  if (child[0]) child[0]->flip ^= 1;
  if (child[1]) child[1]->flip ^= 1;
  flip = 0;
}

void LCTNode::attach(int d, LCTNode *y) {
  child[d] = y;
  if (y) y->parent = this;
}

LinkCutTree::LinkCutTree(int n) {
  for (int i = 0; i < n; i++) {
    T.push_back(LCTNode(i));
  }
}

void LinkCutTree::link(int u, int v) {
  // std::cout << "Linking " << u << " to " << v << "\n";
  LCTNode *x = &T[u], *y = &T[v];
  make_root(x);
  x->parent = y;
}

void LinkCutTree::cut(int u) {
  // std::cout << "Cutting " << u << "\n";
  LCTNode *x = &T[u];
  access(x);
  x->child[0]->parent = nullptr;
  x->child[0] = nullptr;
}

int LinkCutTree::find_root(int u) {
  LCTNode *x = &T[u];
  access(x);
  while (x->child[0]) x = x->child[0];
  splay(x);
  return x->value;
}

int LinkCutTree::dir(LCTNode *x) {
  if (!x->parent) return -1;
  return x->parent->child[0] == x ? 0 : x->parent->child[1] == x ? 1 : -1;
}

void LinkCutTree::rotate(LCTNode *x) {
  LCTNode *y = x->parent, *z = y->parent;
  int dx = dir(x), dy = dir(y);
  y->attach(dx, x->child[!dx]);
  x->attach(!dx, y);
  if (~dy) z->attach(dy, x);
  x->parent = z;
}

void LinkCutTree::splay(LCTNode *x) {
  for (x->push(); ~dir(x);) {
    LCTNode *y = x->parent, *z = y->parent;
    y->push();
    x->push();
    if (~dir(y)) rotate(dir(x) != dir(y) ? x : y);
    rotate(x);
  }
  x->push();
}

LCTNode *LinkCutTree::access(LCTNode *x) {
  LCTNode *last = nullptr;
  for (LCTNode *y = x; y; y = y->parent) {
    splay(y);
    y->attach(1, last);
    last = y;
  }
  splay(x);
  return last;
}

void LinkCutTree::make_root(LCTNode *x) {
  access(x);
  x->flip ^= 1;
}

}  // namespace geo_algorithms
