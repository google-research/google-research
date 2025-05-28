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

#ifndef TRIE_H_
#define TRIE_H_

#include <algorithm>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <vector>

class TrieNode {
 public:
  TrieNode() {
    parent_ = nullptr;
    frequency_ = 0;
    is_terminal_ = false;
    data_ = 0;
    key_ = -1;
  }
  // Returns all descendants of this node, where the keys on the path from this
  // node to the descendant appear in the candidate set.
  // The result can include the node itself.
  std::vector<TrieNode*> match(const std::unordered_set<int>& candidate_set);
  // Checks if all ancestor keys of this node appear in the candidate set.
  bool matchParents(const std::unordered_set<int>& candidate_set);
  void prune_infrequent(const int min_frequency);

  void prune_nonmatching_nodes(const std::unordered_set<TrieNode*> matches);
  bool isRoot() const { return parent_ == nullptr; }

  std::unordered_map<int, TrieNode> children_;
  TrieNode* parent_;
  int frequency_;
  bool is_terminal_;
  int key_;
  int data_;
};

class Trie {
 public:
  void IncreaseCount(const std::unordered_set<int>& candidate_set);

  TrieNode* insert(const std::unordered_set<int>& candidate_set);
  TrieNode* insert(const std::vector<int>& candidate_set);
  TrieNode* find(const std::unordered_set<int>& candidate_set);

  Trie propose_candidates(const int size);

  int countTerminalNodes();

  TrieNode root_;
};

template<typename F> void traverse_nodes(
    const TrieNode& node, const std::vector<int>& prefix, int depth, F f) {
  f(node, prefix, depth);
  for (const auto& child : node.children_) {
    std::vector<int> new_prefix(prefix.begin(), prefix.end());
    new_prefix.push_back(child.first);
    traverse_nodes(child.second, new_prefix, depth+1, f);
  }
}

template<typename F> void traverse_nodes(
    TrieNode* node, const std::vector<int>& prefix, int depth, F f) {
  f(node, prefix, depth);
  for (auto& child : node->children_) {
    std::vector<int> new_prefix(prefix.begin(), prefix.end());
    new_prefix.push_back(child.first);
    traverse_nodes(&(child.second), new_prefix, depth+1, f);
  }
}

int Trie::countTerminalNodes() {
  int result = 0;
  traverse_nodes(root_, {}, 0, [&result](
      const TrieNode& node, const std::vector<int>& prefix, int depth) {
    if (node.is_terminal_) {
      result += 1;
    }
  });
  return result;
}

Trie FrequentItemSets(const std::vector<std::unordered_set<int>>& data,
                      int min_frequency, int num_items) {
  Trie result;
  Trie candidates;
  for (int i = 0; i < num_items; ++i) {
    candidates.insert(std::vector<int>{i});
  }
  int candidate_size = 1;
  int unique_id = 0;
  while (!candidates.root_.children_.empty()) {
    for (const auto& d : data) {
      candidates.IncreaseCount(d);
    }
    candidates.root_.prune_infrequent(min_frequency);
    // Copy candidates to the result trie.
    traverse_nodes(candidates.root_, {}, 0, [&result, &unique_id](
        const TrieNode& node, const std::vector<int>& prefix, int depth) {
      if (node.is_terminal_) {
        TrieNode* n = result.insert(prefix);
        n->frequency_ = node.frequency_;
        n->data_ = unique_id++;
      }
    });
    ++candidate_size;
    candidates = candidates.propose_candidates(candidate_size);
  }
  return result;
}

Trie Trie::propose_candidates(const int size) {
  Trie result;
  traverse_nodes(root_, {}, 0, [size, &result, this](
      const TrieNode& node, const std::vector<int>& prefix, int depth){
    if (depth == size-2) {
      for (const auto& child_A : node.children_) {
        for (const auto& child_B : node.children_) {
          if (child_A.first < child_B.first) {
            std::unordered_set<int> candidate_set(prefix.begin(),
                                                  prefix.end());
            candidate_set.insert(child_A.first);
            candidate_set.insert(child_B.first);

            // Check that all subsets of length-1 of the candidate are
            // present.
            bool all_subsets_exist = true;
            for (int c : candidate_set) {
              std::unordered_set<int> all_but_c(candidate_set.begin(),
                                                candidate_set.end());
              all_but_c.erase(c);
              if (find(all_but_c) == nullptr) {
                all_subsets_exist = false;
                break;
              }
            }
            if (all_subsets_exist) {
              result.insert(candidate_set);
            }
          }
        }
      }
    }
  });
  return result;
}


void Trie::IncreaseCount(const std::unordered_set<int>& candidate_set) {
  for (auto node : root_.match(candidate_set)) {
    node->frequency_++;
  }
}

TrieNode* Trie::insert(const std::unordered_set<int>& candidate_set) {
  std::vector<int> sorted_candidates(candidate_set.begin(),
                                     candidate_set.end());
  std::sort(sorted_candidates.begin(), sorted_candidates.end());

  return insert(sorted_candidates);
}

TrieNode* Trie::insert(const std::vector<int>& sorted_candidates) {
  TrieNode* node = &root_;
  for (const int candidate : sorted_candidates) {
    auto child = node->children_.find(candidate);
    if (child != node->children_.end()) {
      node = &(child->second);
    } else {
      TrieNode new_node;
      new_node.parent_ = node;
      new_node.frequency_ = 0;
      new_node.is_terminal_ = false;
      new_node.key_ = candidate;
      node->children_[candidate] = new_node;
      node = &(node->children_[candidate]);
    }
  }
  node->is_terminal_ = true;
  return node;
}

TrieNode* Trie::find(const std::unordered_set<int>& candidate_set) {
  std::vector<int> sorted_candidates(candidate_set.begin(),
                                     candidate_set.end());
  std::sort(sorted_candidates.begin(), sorted_candidates.end());

  TrieNode* node = &root_;
  for (const int candidate : sorted_candidates) {
    auto child = node->children_.find(candidate);
    if (child != node->children_.end()) {
      node = &(child->second);
    } else {
      return nullptr;
    }
  }
  return node;
}

void TrieNode::prune_infrequent(const int min_frequency) {
  is_terminal_ = (frequency_ >= min_frequency) && (parent_ != nullptr);

  auto child_it = children_.begin();
  while (child_it != children_.end()) {
    child_it->second.prune_infrequent(min_frequency);
    if (child_it->second.children_.empty() && !child_it->second.is_terminal_) {
      child_it = children_.erase(child_it);
    } else {
      ++child_it;
    }
  }
}

void TrieNode::prune_nonmatching_nodes(
    const std::unordered_set<TrieNode*> matches) {
  if (is_terminal_) {
    is_terminal_ = ((matches.find(this) != matches.end()) && (!isRoot()));
  }

  auto child_it = children_.begin();
  while (child_it != children_.end()) {
    child_it->second.prune_nonmatching_nodes(matches);
    if (child_it->second.children_.empty() && !child_it->second.is_terminal_) {
      child_it = children_.erase(child_it);
    } else {
      ++child_it;
    }
  }
}

std::unordered_set<TrieNode*> find_n_most_frequent_terminal_nodes(
    TrieNode* root, const int num_nodes) {
  using WeightAndNode = std::pair<float, TrieNode*>;
  std::priority_queue<WeightAndNode, std::vector<WeightAndNode>,
      std::greater<WeightAndNode>> best_nodes;
  if (num_nodes > 0) {
    traverse_nodes(root, {}, 0, [&best_nodes, &num_nodes](
        TrieNode* node, const std::vector<int>& prefix, int depth) {
      if (!node->is_terminal_) {
        return;
      }
      if ((best_nodes.size() < (size_t)num_nodes) ||
          (node->frequency_ > best_nodes.top().first)) {
        best_nodes.push({node->frequency_, node});
        while (best_nodes.size() > (size_t)num_nodes) {
          best_nodes.pop();
        }
      }
    });
  }
  std::unordered_set<TrieNode*> result;
  while (!best_nodes.empty()) {
    result.insert(best_nodes.top().second);
    best_nodes.pop();
  }
  return result;
}



std::vector<TrieNode*> TrieNode::match(
    const std::unordered_set<int>& candidate_set) {
  std::vector<TrieNode*> result;
  if (is_terminal_) {
    result.push_back(this);
  }
  if (!children_.empty()) {
    for (const auto& candidate_value : candidate_set) {
      const auto child = children_.find(candidate_value);
      if (child != children_.end()) {
        std::vector<TrieNode*> matches =
            child->second.match(candidate_set);  // note: candidate is not
                                                 // removed.
        result.insert(result.end(), matches.begin(), matches.end());
      }
    }
  }
  return result;
}

bool TrieNode::matchParents(const std::unordered_set<int>& candidate_set) {
  TrieNode* n = parent_;
  while (!n->isRoot()) {
    if (candidate_set.find(n->key_) == candidate_set.end()) {
      return false;
    }
    n = n->parent_;
  }
  return true;
}

#endif  // TRIE_H_
