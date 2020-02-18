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

/**
 * @fileoverview Pólya Tree (PT) language model.
 */

const assert = require("assert");

const vocab = require("./vocabulary");

/**
 * Hard-coded beta distribution parameters.
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
 * @final
 */
class PolyaTreeLanguageModel {
  /**
   * Constructor.
   * @param {?Vocabulary} vocab Symbol vocabulary object.
   */
  constructor(vocab) {
    this.vocab_ = vocab;
    assert(this.vocab_.size() > 1,
           "Expecting at least two symbols in the vocabulary");
    this.nodes_ = null;  // Array representation of a tree.
    this.buildTree_();
    this.totalObservations_ = 0;  // Total number of observations.
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
    assert(path.length > 1, "Expected more than one node in the path");
    for (let i = 0; i < path.length; ++i) {
      const pathNode = path[i];
      if (pathNode.leftBranch_) {
        this.nodes_[pathNode.id_].numBranchLeft_++;
      } else {
        this.nodes_[pathNode.id_].numBranchRight_++;
      }
    }
    this.totalObservations_++;
  }

  /**
   * Returns probabilities for all the symbols in the vocabulary given the
   * context.
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
      probs[i] = 1.0 / numValidSymbols;  // Uniform distribution.
      const numInternalNodes = path.length - 1;
      for (let j = 0; j < numInternalNodes; ++j) {
        const pathNode = path[j];
        const treeNode = this.nodes_[pathNode.id_];
        const theta = (betaDistrAlpha + treeNode.numBranchLeft_) /
              (betaDistrAlpha + betaDistrBeta +
               treeNode.numBranchLeft_ + treeNode.numBranchRight_);
        if (pathNode.leftBranch_) {
          probs[i] *= theta;
        } else {
          probs[i] *= (1.0 - theta);
        }
      }
      totalMass -= probs[i];
    }

    // Adjust the remaining probability mass, if any.
    let newProbMass = 0.0;
    let leftSymbols = numValidSymbols;
    for (let i = 1; i < numSymbols; ++i) {
      const p = totalMass / leftSymbols;
      probs[i] += p;
      totalMass -= p;
      newProbMass += probs[i];
      --leftSymbols;
    }
    assert(totalMass == 0.0, "Expected remaining probability mass to be zero!");
    assert(Math.abs(1.0 - newProbMass) < epsilon);
    return probs;
  }

  /**
   * Constructs Pólya tree from the given vocabulary.
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
