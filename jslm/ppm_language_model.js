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

/**
 * @fileoverview Prediction by Partial Matching (PPM) language model.
 *
 * The original PPM algorithm is described in [1]. This particular
 * implementation has been inspired by the PPM model used by Dasher, an
 * Augmentative and alternative communication (AAC) input method developed by
 * the Inference Group at University of Cambridge. The overview of the system
 * is provided in [2]. The details of this algorithm, which is different from
 * the standard PPM, are outlined in general terms in [3]. Please also see [4]
 * for an excellent overview of various PPM variants.
 *
 * References:
 * -----------
 *   [1] Cleary, John G. and Witten, Ian H. (1984): “Data Compression Using
 *       Adaptive Coding and Partial String Matching”, IEEE Transactions on
 *       Communications, vol. 32, no. 4, pp. 396–402.
 *   [2] Ward, David J. and Blackwell, Alan F. and MacKay, David J. C. (2000):
 *       “Dasher - A Data Entry Interface Using Continuous Gestures and
 *       Language Models”, UIST'00 Proceedings of the 13th annual ACM symposium
 *       on User interface software and technology, pp. 129–137, November, San
 *       Diego, USA.
 *   [3] Cowans, Phil (2005): “Language Modelling In Dasher -- A Tutorial”,
 *       June, Inference Lab, Cambridge University (presentation).
 *   [4] Jin Hu Huang and David Powers (2004): "Adaptive Compression-based
 *       Approach for Chinese Pinyin Input." Proceedings of the Third SIGHAN
 *       Workshop on Chinese Language Processing, pp. 24--27, Barcelona, Spain,
 *       ACL.
 * Please also consult the references in README.md file in this directory.
 */

const assert = require("assert");

const vocab = require("./vocabulary");

/**
 * Kneser-Ney "-like" smoothing parameters.
 *
 * These hardcoded values are copied from Dasher. Please see the documentation
 * for PPMLanguageModel.getProbs() below for more information.
 */
const knAlpha = 0.49;
const knBeta = 0.77;

/* Epsilon for sanity checks. */
const epsilon = 1E-10;

/**
 * Node in a search tree, which is implemented as a suffix trie that represents
 * every suffix of a sequence used during its construction. Please see
 *   [1] Moffat, Alistair (1990): "Implementing the PPM data compression
 *       scheme", IEEE Transactions on Communications, vol. 38, no. 11, pp.
 *       1917--1921.
 *   [2] Esko Ukknonen (1995): "On-line construction of suffix trees",
 *       Algorithmica, volume 14, pp. 249--260, Springer, 1995.
 *   [3] Kennington, C. (2011): "Application of Suffix Trees as an
 *       Implementation Technique for Varied-Length N-gram Language Models",
 *       MSc. Thesis, Saarland University.
 *
 * @final
 */
class Node {
  constructor() {
    // Leftmost child node for the current node.
    this.child_ = null;
    // Next node.
    this.next_ = null;
    // Node in the backoff structure, also known as "vine" structure (see [1]
    // above) and "suffix link" (see [2] above). The backoff for the given node
    // points at the node representing the shorter context. For example, if the
    // current node in the trie represents string "AA" (corresponding to the
    // branch "[R] -> [A] -> [*A*]" in the trie, where [R] stands for root),
    // then its backoff points at the node "A" (represented by "[R] ->
    // [*A*]"). In this case both nodes are in the same branch but they don't
    // need to be. For example, for the node "B" in the trie path for the string
    // "AB" ("[R] -> [A] -> [*B*]") the backoff points at the child node of a
    // different path "[R] -> [*B*]".
    this.backoff_ = null;
    // Frequency count for this node. Number of times the suffix symbol stored
    // in this node was observed.
    this.count_ = 1;
    // Symbol that this node stores.
    this.symbol_ = vocab.rootSymbol;
  }

  /**
   * Finds child of the current node with a specified symbol.
   * @param {number} symbol Integer symbol.
   * @return {?Node} Node with the symbol.
   * @final
   */
  findChildWithSymbol(symbol) {
    let current = this.child_;
    while (current != null) {
      if (current.symbol_ == symbol) {
        return current;
      }
      current = current.next_;
    }
    return current;
  }

  /**
   * Total number of observations for all the children of this node. This
   * counts all the events observed in this context.
   *
   * Note: This API is used at inference time. A possible alternative that will
   * speed up the inference is to store the number of children in each node as
   * originally proposed by Moffat for PPMB in
   *   Moffat, Alistair (1990): "Implementing the PPM data compression scheme",
   *   IEEE Transactions on Communications, vol. 38, no. 11, pp. 1917--1921.
   * This however will increase the memory use of the algorithm which is already
   * quite substantial.
   *
   * @param {!array} exclusionMask Boolean exclusion mask for all the symbols.
   *                 Can be 'null', in which case no exclusion happens.
   * @return {number} Total number of observations under this node.
   * @final
   */
  totalChildrenCounts(exclusionMask) {
    let childNode = this.child_;
    let count = 0;
    while (childNode != null) {
      if (!exclusionMask || !exclusionMask[childNode.symbol_]) {
        count += childNode.count_;
      }
      childNode = childNode.next_;
    }
    return count;
  }
}

/**
 * Handle encapsulating the search context.
 * @final
 */
class Context {
  /**
   * Constructor.
   * @param {?Node} head Head node of the context.
   * @param {number} order Length of the context.
   */
  constructor(head, order) {
    // Current node.
    this.head_ = head;
    // The order corresponding to length of the context.
    this.order_ = order;
  }
}

/**
 * Prediction by Partial Matching (PPM) Language Model.
 * @final
 */
class PPMLanguageModel {
  /**
   * @param {?Vocabulary} vocab Symbol vocabulary object.
   * @param {number} maxOrder Maximum length of the context.
   */
  constructor(vocab, maxOrder) {
    this.vocab_ = vocab;
    assert(this.vocab_.size() > 1,
           "Expecting at least two symbols in the vocabulary");

    this.maxOrder_ = maxOrder;
    this.root_ = new Node();
    this.rootContext_ = new Context();
    this.rootContext_.head_ = this.root_;
    this.rootContext_.order_ = 0;
    this.numNodes_ = 1;

    // Exclusion mechanism: Off by default, but can be enabled during the
    // run-time once the constructed suffix tree contains reliable counts.
    this.useExclusion_ = false;
  }

  /**
   * Adds symbol to the supplied node.
   * @param {?Node} node Tree node which to grow.
   * @param {number} symbol Symbol.
   * @return {?Node} Node with the symbol.
   * @final @private
   */
  addSymbolToNode_(node, symbol) {
    let symbolNode = node.findChildWithSymbol(symbol);
    if (symbolNode != null) {
      // Update the counts for the given node and also for all the backoff nodes
      // representing shorter contexts.
      symbolNode.count_++;
      let backoffNode = symbolNode.backoff_;
      assert(backoffNode != null, "Expected valid backoff node!");
      while (backoffNode != null) {
        assert(backoffNode == this.root_ || backoffNode.symbol_ == symbol,
               "Expected backoff node to be root or to contain " + symbol +
               ". Found " + backoffNode.symbol_ + " instead");
        backoffNode.count_++;
        backoffNode = backoffNode.backoff_;
      }
    } else {
      // Symbol does not exist under the given node. Create a new child node
      // and update the backoff structure for lower contexts.
      symbolNode = new Node();
      symbolNode.symbol_ = symbol;
      symbolNode.next_ = node.child_;
      node.child_ = symbolNode;
      this.numNodes_++;
      if (node == this.root_) {
        // Shortest possible context.
        symbolNode.backoff_ = this.root_;
      } else {
        assert(node.backoff_ != null, "Expected valid backoff node");
        symbolNode.backoff_ = this.addSymbolToNode_(node.backoff_, symbol);
      }
    }
    return symbolNode;
  }

  /**
   * Creates new context which is initially empty.
   * @return {?Context} Context object.
   * @final
   */
  createContext() {
    return new Context(this.rootContext_.head_, this.rootContext_.order_);
  }

  /**
   * Clones existing context.
   * @param {?Context} context Existing context object.
   * @return {?Context} Cloned context object.
   * @final
   */
  cloneContext(context) {
    return new Context(context.head_, context.order_);
  }

  /**
   * Adds symbol to the supplied context. Does not update the model.
   * @param {?Context} context Context object.
   * @param {number} symbol Integer symbol.
   * @final
   */
  addSymbolToContext(context, symbol) {
    if (symbol <= vocab.rootSymbol) {  // Only add valid symbols.
      return;
    }
    assert(symbol < this.vocab_.size(), "Invalid symbol: " + symbol);
    while (context.head_ != null) {
      if (context.order_ < this.maxOrder_) {
        // Extend the current context.
        const childNode = context.head_.findChildWithSymbol(symbol);
        if (childNode != null) {
          context.head_ = childNode;
          context.order_++;
          return;
        }
      }
      // Try to extend the shorter context.
      context.order_--;
      context.head_ = context.head_.backoff_;
    }
    if (context.head_ == null) {
      context.head_ = this.root_;
      context.order_ = 0;
    }
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
    const symbolNode = this.addSymbolToNode_(context.head_, symbol);
    assert(symbolNode == context.head_.findChildWithSymbol(symbol));
    context.head_ = symbolNode;
    context.order_++;
    while (context.order_ > this.maxOrder_) {
      context.head_ = context.head_.backoff_;
      context.order_--;
    }
  }

  /**
   * Returns probabilities for all the symbols in the vocabulary given the
   * context.
   *
   * Notation:
   * ---------
   *         $x_h$ : Context representing history, $x_{h-1}$ shorter context.
   *   $n(w, x_h)$ : Count of symbol $w$ in context $x_h$.
   *      $T(x_h)$ : Total count in context $x_h$.
   *      $q(x_h)$ : Number of symbols with non-zero counts seen in context
   *                 $x_h$, i.e. |{w' : c(x_h, w') > 0}|. Alternatively, this
   *                 represents the number of distinct extensions of history
   *                 $x_h$ in the training data.
   *
   * Standard Kneser-Ney method (aka Absolute Discounting):
   * ------------------------------------------------------
   * Subtracting \beta (in [0, 1)) from all counts.
   *   P_{kn}(w | x_h) = \frac{\max(n(w, x_h) - \beta, 0)}{T(x_h)} +
   *                     \beta * \frac{q(x_h)}{T(x_h)} * P_{kn}(w | x_{h-1}),
   * where the second term in summation represents escaping to lower-order
   * context.
   *
   * See: Ney, Reinhard and Kneser, Hermann (1995): “Improved backing-off for
   * M-gram language modeling”, Proc. of Acoustics, Speech, and Signal
   * Processing (ICASSP), May, pp. 181–184.
   *
   * Modified Kneser-Ney method (Dasher version [3]):
   * ------------------------------------------------
   * Introducing \alpha parameter (in [0, 1)) and estimating as
   *   P_{kn}(w | x_h) = \frac{\max(n(w, x_h) - \beta, 0)}{T(x_h) + \alpha} +
   *                     \frac{\alpha + \beta * q(x_h)}{T(x_h) + \alpha} *
   *                     P_{kn}(w | x_{h-1}) .
   *
   * Additional details on the above version are provided in Sections 3 and 4
   * of:
   *   Steinruecken, Christian and Ghahramani, Zoubin and MacKay, David (2016):
   *   "Improving PPM with dynamic parameter updates", In Proc. Data
   *   Compression Conference (DCC-2015), pp. 193--202, April, Snowbird, UT,
   *   USA. IEEE.
   *
   * @param {?Context} context Context symbols.
   * @return {?array} Array of floating point probabilities corresponding to all
   *                  the symbols in the vocabulary plus the 0th element
   *                  representing the root of the tree that should be ignored.
   * @final
   */
  getProbs(context) {
    // Initialize the initial estimates. Note, we don't use uniform
    // distribution here.
    const numSymbols = this.vocab_.size();
    let probs = new Array(numSymbols);
    for (let i = 0; i < numSymbols; ++i) {
      probs[i] = 0.0;
    }

    // Initialize the exclusion mask.
    let exclusionMask = null;
    if (this.useExclusion_) {
      exclusionMask = new Array(numSymbols);
      for (let i = 0; i < numSymbols; ++i) {
        exclusionMask[i] = false;
      }
    }

    // Estimate the probabilities for all the symbols in the supplied context.
    // This runs over all the symbols in the context and over all the suffixes
    // (orders) of the context. If the exclusion mechanism is enabled, the
    // estimate for a higher-order ngram is fully trusted and is excluded from
    // further interpolation with lower-order estimates.
    //
    // Exclusion mechanism is disabled by default. Enable it with care: it has
    // been shown to work well on large corpora, but may in theory degrade the
    // performance on smaller sets (as we observed with default Dasher English
    // training data).
    let totalMass = 1.0;
    let node = context.head_;
    let gamma = totalMass;
    while (node != null) {
      const count = node.totalChildrenCounts(exclusionMask);
      if (count > 0) {
        let childNode = node.child_;
        while (childNode != null) {
          const symbol = childNode.symbol_;
          if (!exclusionMask || !exclusionMask[symbol]) {
            const p = gamma * (childNode.count_ - knBeta) / (count + knAlpha);
            probs[symbol] += p;
            totalMass -= p;
            if (exclusionMask) {
              exclusionMask[symbol] = true;
            }
          }
          childNode = childNode.next_;
        }
      }

      // Backoff to lower-order context. The $\gamma$ factor represents the
      // total probability mass after processing the current $n$-th order before
      // backing off to $n-1$-th order. It roughly corresponds to the usual
      // interpolation parameter, as used in the literature, e.g. in
      //   Stanley F. Chen and Joshua Goodman (1999): "An empirical study of
      //   smoothing techniques for language modeling", Computer Speech and
      //   Language, vol. 13, pp. 359-–394.
      //
      // Note on computing $gamma$:
      // --------------------------
      // According to the PPM papers, and in particular the Section 4 of
      //   Steinruecken, Christian and Ghahramani, Zoubin and MacKay,
      //   David (2016): "Improving PPM with dynamic parameter updates", In
      //   Proc. Data Compression Conference (DCC-2015), pp. 193--202, April,
      //   Snowbird, UT, USA. IEEE,
      // that describes blending (i.e. interpolation), the second multiplying
      // factor in the interpolation $\lambda$ for a given suffix node $x_h$ in
      // the tree is given by
      //   \lambda(x_h) = \frac{q(x_h) * \beta + \alpha}{T(x_h) + \alpha} .
      // It can be shown that
      //   \gamma(x_h) = 1.0 - \sum_{w'}
      //      \frac{\max(n(w', x_h) - \beta, 0)}{T(x_h) + \alpha} =
      //      \lambda(x_h)
      // and, in the update below, the following is equivalent:
      //   \gamma = \gamma * \gamma(x_h) = totalMass .
      //
      // Since gamma *= (numChildren * knBeta + knAlpha) / (count + knAlpha) is
      // expensive, we assign the equivalent totalMass value to gamma.
      node = node.backoff_;
      gamma = totalMass;
    }
    assert(totalMass >= 0.0,
           "Invalid remaining probability mass: " + totalMass);

    // Count the total number of symbols that should have their estimates
    // blended with the uniform distribution representing the zero context.
    // When exclusion mechanism is enabled (by enabling this.useExclusion_)
    // this number will represent the number of symbols not seen during the
    // training, otherwise, this number will be equal to total number of
    // symbols because we always interpolate with the estimates for an empty
    // context.
    let numUnseenSymbols = 0;
    for (let i = 1; i < numSymbols; ++i) {
      if (!exclusionMask || !exclusionMask[i]) {
        numUnseenSymbols++;
      }
    }

    // Adjust the probability mass for all the symbols.
    const remainingMass = totalMass;
    for (let i = 1; i < numSymbols; ++i) {
      // Following is estimated according to a uniform distribution
      // corresponding to the context length of zero.
      if (!exclusionMask || !exclusionMask[i]) {
        const p = remainingMass / numUnseenSymbols;
        probs[i] += p;
        totalMass -= p;
      }
    }
    let leftSymbols = numSymbols - 1;
    let newProbMass = 0.0;
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
   * Prints the trie to console.
   * @param {?Node} node Current trie node.
   * @param {string} indent Indentation prefix.
   * @final @private
   */
  printToConsole_(node, indent) {
    console.log(indent + "  " + this.vocab_.symbols_[node.symbol_] +
                "(" + node.symbol_ + ") [" + node.count_ + "]");
    indent += "  ";
    let child = node.child_;
    while (child != null) {
      this.printToConsole_(child, indent);
      child = child.next_;
    }
  }

  /**
   * Prints the trie to console.
   * @final
   */
  printToConsole() {
    this.printToConsole_(this.root_, "");
  }
}

/**
 * Exported APIs.
 */
exports.PPMLanguageModel = PPMLanguageModel;
