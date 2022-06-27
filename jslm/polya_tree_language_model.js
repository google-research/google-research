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

/**
 * @fileoverview Pólya Tree (PT) language model.
 *
 * The Pólya tree method uses a balanced binary search tree whose leaf nodes
 * contain the symbols of the vocabulary V, such that each symbol v \in V can be
 * identified by a sequence of (at most \log_2(|V|)) binary branching decisions
 * from the root of the tree. The leafs of the tree represent predictive
 * probabilities for the respective symbols. The tree has N=|V|−1 internal
 * nodes, each containing a value \theta_i that represents the probability of
 * choosing between its two children [1]. The probability of a given symbol v is
 * then defined as [2]:
 *
 *   P(v) = \prod_{i \in PATH(v)} Bernoulli(b_i | \theta_i)
 *        = \prod_{i \in PATH(v)} \theta_i^{b_i} (1 - \theta_i)^{1 - b_i} ,
 *
 * where PATH(v) denotes the set of nodes i that belong to the path from the
 * root to node v, b_i \in {0, 1} is the branching decision at node i and
 * \theta_i is the Bernoulli distribution bias at node i. See documentation of
 * the getProbs() API to see how the \theta is approximated via conjugate priors
 * using beta distribution, in other words, \theta_i ~ Beta(\alpha, \beta).
 *
 * This language model can be used as a prior in a more sophisticated
 * context-based model.
 *
 * References:
 * -----------
 *   [1] Steinruecken, Christian (2015): "Lossless Data Compression", PhD
 *       dissertation, University of Cambridge.
 *   [2] Gleave, Adam and Steinruecken, Christian (2017): "Making compression
 *       algorithms for Unicode text", arXiv preprint arXiv:1701.04047.
 *   [3] Mauldin, R. Daniel and Sudderth, William D. and Williams, S. C. (1992):
 *       "Polya Trees and Random Distributions", The Annals of Statistics,
 *       pp. 1203--1221.
 */

const assert = require("assert");

const vocab = require("./vocabulary");

/**
 * Hard-coded parameters for the beta distribution. In Beta(\alpha, \beta), the
 * \alpha and \beta hyperparameters represent the left and right splits,
 * respectively.
 */
const betaDistrAlpha = 0.5;  // $\alpha$.
const betaDistrBeta = 0.5;   // $\beta$.

// Epsilon for sanity checks.
const epsilon = 1E-12;

/**
 * Node in the Pólya tree.
 *
 * @final
 */
class Node {
  /**
   * Constructor.
   */
  constructor() {
    // Number of times the left branch was taken.
    this.numBranchLeft_ = 0;
    // Number of times the right branch was taken.
    this.numBranchRight_ = 0;
  }
}

/**
 * Node in a path from the root to the leaf of a Pólya tree.
 *
 * @final
 */
class PathNode {
  /**
   * Constructor.
   */
  constructor() {
    // Identity of the node in the array encoding the tree.
    this.id_ = 0;
    // Flag indicating whether this is a left or right branch in relation to
    // this node's parent.
    this.leftBranch_ = false;
  }
}

/**
 * Handle encapsulating the search context. Since the PT models are
 * context-less, this class is intentionally left as an empty handle to comply
 * with the interface of the context-based models.
 * @final
 */
class Context {}

/**
 * Pólya tree language model.
 *
 * @final
 */
class PolyaTreeLanguageModel {
  /**
   * Constructor.
   * @param {?Vocabulary} vocab Symbol vocabulary object.
   */
  constructor(vocab) {
    this.totalObservations_ = 0;  // Total number of observations.
    this.vocab_ = vocab;
    assert(this.vocab_.size() > 1,
           "Expecting at least two symbols in the vocabulary");

    this.nodes_ = null;  // Array representation of a tree.
    this.rootProbs_ = null;  // Probabilities at the root.
    this.buildTree_();
  }

  /**
   * Creates new context which is initially empty.
   * @return {?Context} Context object.
   * @final
   */
  createContext() {
    // Nothing to do here.
    return new Context();
  }

  /**
   * Clones existing context.
   * @param {?Context} context Existing context object.
   * @return {?Context} Cloned context object.
   * @final
   */
  cloneContext(context) {
    // Nothing to do here.
    return new Context();
  }

  /**
   * Adds symbol to the supplied context. Does not update the model.
   * @param {?Context} context Context object.
   * @param {number} symbol Integer symbol.
   * @final
   */
  addSymbolToContext(context, symbol) {
    // Nothing to do here.
  }

  /**
   * Adds symbol to the supplied context and updates the model.
   * @param {?Context} context Context object.
   * @param {number} symbol Integer symbol.
   * @final
   */
  addSymbolAndUpdate(context, symbol) {
    if (symbol <= vocab.rootSymbol) {  // Only add valid symbols.
      return;
    }
    assert(symbol < this.vocab_.size(), "Invalid symbol: " + symbol);
    const path = this.getPath_(symbol);
    assert(path.length > 1,
           "Expected more than one node in the path for symbol " + symbol);
    const numInternalNodes = path.length - 1;
    for (let i = 0; i < numInternalNodes; ++i) {
      // Update the branch counts for each node.
      const pathNode = path[i];
      let treeNode = this.nodes_[pathNode.id_];
      if (pathNode.leftBranch_) {  // Update left branch counts.
        treeNode.numBranchLeft_++;
      } else {  // Update right branch counts.
        treeNode.numBranchRight_++;
      }

      // Make sure at each step the total frequency of each node corresponds to
      // to the branching frequency of the parent, left or right, respectively.
      const nodeTotal = treeNode.numBranchLeft_ + treeNode.numBranchRight_;
      if (i > 0) {
        const prevPathNode = path[i - 1];
        const prevTreeNode = this.nodes_[prevPathNode.id_];
        if (prevPathNode.leftBranch_) {
          assert(prevTreeNode.numBranchLeft_ == nodeTotal,
                 "Expected the total count for the current node (" + nodeTotal +
                 ") to be equal to the parent left branch count (" +
                 prevTreeNode.numBranchLeft_ + ")");
        } else {
          assert(prevTreeNode.numBranchRight_ == nodeTotal,
                 "Expected the total count for the current node (" + nodeTotal +
                 ") to be equal to the parent right branch count (" +
                 prevTreeNode.numBranchRight_ + ")");
        }
      }
    }
    this.totalObservations_++;
  }

  /**
   * Returns probabilities for all the symbols in the vocabulary given the
   * context.
   *
   * Using Bernoulli likelihoods, the probability of symbol v is
   *
   *   P(v) = \prod_{i \in PATH(v)} \theta_i^{b_i} (1 - \theta_i)^{1 - b_i} ,
   *
   * where the Bernoulli bias \theta_i at node i in the path can be expressed
   * using the parameters of the beta distribution (\alpha and \beta) and the
   * branching counts n^l_i (left) and n^r_i (right) at each node i in the path:
   *
   *   \theta_i = \frac{\alpha + n^l_i}{\alpha + \beta + n^l_i + n^r_i}.
   *
   * @param {?Context} context Context symbols.
   * @return {?array} Array of floating point probabilities corresponding to all
   *                  the symbols in the vocabulary plus the 0th element
   *                  representing the root of the tree that should be ignored.
   * @final
   */
  getProbs(context) {
    const numSymbols = this.vocab_.size();
    const numValidSymbols = numSymbols - 1;  // Minus the first symbol.
    let probs = new Array(numSymbols);
    probs[0] = 0.0;  // Ignore first symbol.

    // Compute the distribution.
    let totalMass = 1.0;
    for (let i = 1; i < numSymbols; ++i) {
      const path = this.getPath_(i);
      assert(path.length > 1,
             "Expected more than one node in the path for symbol " + i);
      probs[i] = this.rootProbs_[i];
      const numInternalNodes = path.length - 1;
      for (let j = 0; j < numInternalNodes; ++j) {
        const pathNode = path[j];
        const treeNode = this.nodes_[pathNode.id_];
        const theta = (betaDistrAlpha + treeNode.numBranchLeft_) /
              (betaDistrAlpha + betaDistrBeta +
               treeNode.numBranchLeft_ + treeNode.numBranchRight_);
        if (pathNode.leftBranch_) {  // Follow left branch.
          probs[i] *= theta;
        } else {  // Follow right branch.
          probs[i] *= (1.0 - theta);
        }
      }
      totalMass -= probs[i];
    }
    assert(totalMass > 0.0, "Negative probability mass");

    // Adjust the remaining probability mass, if any.
    const delta = totalMass / numValidSymbols;
    let newProbMass = 0.0;
    for (let i = 1; i < numSymbols; ++i) {
      probs[i] += delta;
      totalMass -= delta;
      newProbMass += probs[i];
    }
    assert(Math.abs(1.0 - newProbMass) < epsilon);
    return probs;
  }

  /**
   * Constructs Pólya tree from the given vocabulary.
   *
   * Note, the Pólya Tree does not correspond exactly to a classical Binary
   * Search Tree (BST) because of the requirement to have |V|-1 extra nodes.
   *
   * @final @private
   */
  buildTree_() {
    // Fill in the array representing the level-order traversal of the tree. Let
    // |V| be the cardinality of the vocabulary. The first part of the array,
    // corresponding to |V| - 1 elements, represents the internal nodes. The
    // second part consisting of |V| represents the leaf nodes.
    //
    // The root node is stored at 0th element. For a given node in position i,
    // its left child is at position (2i + 1) and its right child is at position
    // (2i + 2).
    const numSymbols = this.vocab_.size() - 1;
    const numNodes = numSymbols * 2 - 1;
    this.nodes_ = new Array(numNodes);
    for (let i = 0; i < numSymbols - 1; ++i) {  // internal nodes
      this.nodes_[i] = new Node();
    }
    for (let i = 0; i < numSymbols; ++i) {
      this.nodes_[i + numSymbols - 1] = new Node();
    }

    // Compute probabilities at the root node in the absence of branching
    // frequencies, in other words, given
    //
    //   \theta_0 = \frac{\alpha}{\alpha + \beta}
    //
    // compute
    //
    //   P_0() = \prod_{i \in PATH(v)} \theta_0^{b_i} (1 - \theta_0)^{1 - b_i} .
    this.rootProbs_ = new Array(numSymbols);
    this.rootProbs_[0] = 0.0;  // Ignore first symbol.
    const theta = betaDistrAlpha / (betaDistrAlpha + betaDistrBeta);
    let totalMass = 1.0;
    for (let i = 1; i < this.vocab_.size(); ++i) {
      const path = this.getPath_(i);
      assert(path.length > 1,
             "Expected more than one node in the path for symbol " + i);
      let p = 1.0;
      const numInternalNodes = path.length - 1;
      for (let j = 0; j < numInternalNodes; ++j) {
        const pathNode = path[j];
        if (pathNode.leftBranch_) {  // Follow left branch.
          p *= theta;
        } else {  // Follow right branch.
          p *= (1.0 - theta);
        }
      }
      this.rootProbs_[i] = p;
      totalMass -= p;
    }
    assert(totalMass == 0.0,
           "Invalid total mass for initial probabilities: " + totalMass);
  }

  /**
   * Returns a list of indices of the nodes representing the path from the root
   * of the tree to the leaf node containing the symbol.
   *
   * @param {number} symbol Actual symbol.
   * @return {?PathNode} Array of elements representing the path.
   * @final @private
   */
  getPath_(symbol) {
    assert(symbol < this.vocab_.size(), "Invalid symbol: " + symbol);

    // Compute a list of IDs representing the path.
    const numSymbols = this.vocab_.size() - 1;
    let path = new Array();
    const symbolNodeId = numSymbols - 1 + symbol - 1;
    path.push(symbolNodeId);
    let parent = Math.trunc((symbolNodeId - 1) / 2);
    while (parent >= 1) {
      path.push(parent);
      parent = Math.trunc((parent - 1) / 2);
    }
    path.push(0);  // Add root.
    path = path.reverse();

    // Convert the IDs to the full-fledged path representation. Node, the
    // branching information for the last (leaf) node is omitted.
    let nodes = new Array(path.length);
    for (let i = 0; i < path.length; ++i) {
      let node = new PathNode();
      node.id_ = path[i];
      if (i > 0) {
        if (node.id_ == 2 * path[i - 1] + 1) {
          nodes[i - 1].leftBranch_ = true;   // Left branch.
        } else if (node.id_ == 2 * path[i - 1] + 2) {
          nodes[i - 1].leftBranch_ = false;  // Right branch.
        } else {
          assert(false, "Invalid node ID: " + node.id_);
        }
      }
      nodes[i] = node;
    }
    return nodes;
  }

  /**
   * Prints given level of a tree to console using pre-order traversal.
   *
   * @param {number} nodeId Identity of a node.
   * @param {string} indent Indentation prefix.
   * @final @private
   */
  printToConsole_(nodeId, indent) {
    const numNodes = this.nodes_.length;
    if (nodeId >= numNodes) {
      return;
    }
    let info = nodeId;
    if (2 * nodeId >= numNodes - 1) {  // Leaf node.
      const numSymbols = this.vocab_.size() - 1;
      const symbol = nodeId - numSymbols + 2;
      assert(symbol > 0, "Invalid symbol: " + symbol);
      info += ": " + this.vocab_.symbols_[symbol] + " (" + symbol + ")";
    } else {  // Internal node.
      const treeNode = this.nodes_[nodeId];
      info += " [" + treeNode.numBranchLeft_ + "-" +
          treeNode.numBranchRight_ + "]";
    }
    console.log(indent + info);
    indent += "  ";
    this.printToConsole_(2 * nodeId + 1, indent);  // Left.
    this.printToConsole_(2 * nodeId + 2, indent);  // Right.
  }

  /**
   * Prints the tree to console.
   *
   * Uses level-order traversal.
   * @final
   */
  printToConsole() {
    this.printToConsole_(0, "");
  }
}

/**
 * Exported APIs.
 */
exports.PolyaTreeLanguageModel = PolyaTreeLanguageModel;
