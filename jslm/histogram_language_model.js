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
 * @fileoverview Histogram language model.
 *
 * This is an adaptive context-less language model. It estimates the
 * probability of the $n$th symbol $w_n$ to be equal to value $s$ as
 * a function of the number of times $s$ was observed in the preceding
 * sequence { $w_1$, ..., $w_{n-1}$ }. Since the model is context-less,
 * it is unlikely to perform as well as more sophisticated models,
 * however the main advantage of this approach over the alternatives
 * is its simplicity resulting in low memory and CPU usage.
 *
 * This language model can be used as a prior in a more sophisticated
 * context-based model.
 *
 * References:
 * -----------
 *   [1] Steinruecken, Christian (2015): "Lossless Data Compression", PhD
 *       dissertation, University of Cambridge.
 *   [2] Pitman, Jim and Yor, Marc (1997): "The two-parameter Poisson–Dirichlet
 *       distribution derived from a stable subordinator.", The Annals of
 *       Probability, vol. 25, no. 2, pp. 855-–900.
 *   [3] Stanley F. Chen and Joshua Goodman (1999): "An empirical study of
 *       smoothing techniques for language modeling", Computer Speech and
 *       Language, vol. 13, pp. 359-–394.
 */

const assert = require("assert");

const vocab = require("./vocabulary");

/**
 * Kneser-Ney "-like" smoothing parameters: $\alpha$ and $\beta$
 * controlling the two factors in the final posterior estimate:
 *   (1) the empirical distribution and
 *   (2) the arbitrary base distribution.
 * Higher values of $\alpha$ give more weight to (2), low values
 * make distribution (1) the dominant distribution.
 *
 * This parameters describe the two-parameter Chinese restaurant process or
 * the (discrete) Pitman-Yor process.
 */
// Strength parameter $\alpha$ that controls how quickly the base
// prior distribution is influenced by empirical symbol counts.
const pyAlpha = 0.50;

// Discounting parameter.
const pyBeta = 0.77;

// Epsilon for sanity checks.
const epsilon = 1E-12;

/**
 * Handle encapsulating the search context. Since the histogram models are
 * context-less, this class is intentionally left as an empty handle to
 * comply with the interface of the context-based models.
 * @final
 */
class Context {}

/**
 * Histogram language model.
 * @final
 */
class HistogramLanguageModel {
  /**
   * Constructor.
   * @param {?Vocabulary} vocab Symbol vocabulary object.
   */
  constructor(vocab) {
    this.vocab_ = vocab;
    assert(this.vocab_.size() > 1,
           "Expecting at least two symbols in the vocabulary");
    // Total number of symbols observed during the training.
    this.totalObservations_ = 0;

    // Histogram of symbol counts. The first element is ignored.
    const numSymbols = this.vocab_.size();
    this.histogram_ = new Array(numSymbols);
    for (let i = 0; i < numSymbols; ++i) {
      this.histogram_[i] = 0;
    }
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
    this.histogram_[symbol]++;
    this.totalObservations_++;
  }

  /**
   * Returns probabilities for all the symbols in the vocabulary given the
   * context.
   *
   * This particular formulation can be seen as modified Kneser-Ney smoothing,
   * a two-parameter Chinese Restaurant process or a discrete Pitman-Yor
   * process. See
   *
   *   Steinruecken, Christian (2015): "Lossless Data Compression", PhD
   *   dissertation, University of Cambridge. Section 4.2.3 (pp. 65--67).
   *
   * The distribution is computed as follows:
   *   P(w_{N+1} = s | w_1, ..., w_N) =
   *     \frac{n_s - \beta}{N + \alpha} * 1[n_s > 0] +
   *     \frac{\alpha + \beta * U}{N + \alpha} * P_b(s) ,
   * where
   *   - "s" is the symbol to predict and "n_s" is its count in the data,
   *   - 1[n_s > 0]: an indicator boolean function,
   *   - U is the number of unique seen symbols and
   *   - P_b(s) is the base distribution, which in our implementation is
   *     a uniform distribution.
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

    // Figure out the number of unique (seen) symbols.
    let numUniqueSeenSymbols = 0;
    for (let i = 1; i < numSymbols; ++i) {
      if (this.histogram_[i] > 0) {
        numUniqueSeenSymbols++;
      }
    }

    // Compute the distribution.
    const denominator = this.totalObservations_ + pyAlpha;
    const baseFactor = (pyAlpha + pyBeta * numUniqueSeenSymbols) / denominator;
    const uniformPrior = 1.0 / numValidSymbols;
    let totalMass = 1.0;
    for (let i = 1; i < numSymbols; ++i) {
      let empirical = 0.0;
      if (this.histogram_[i] > 0) {
        empirical = (this.histogram_[i] - pyBeta) / denominator;
      }
      probs[i] = empirical + baseFactor * uniformPrior;
      totalMass -= probs[i];
    }
    assert(Math.abs(totalMass) < epsilon,
           "Invalid remaining probability mass: " + totalMass);

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
   * Prints the histogram to console.
   * @final
   */
  printToConsole() {
    console.log("Histogram of counts (total: " +
                this.totalObservations_ + "): ");
    for (let i = 1; i < this.histogram_.length; ++i) {
      console.log("\t" + this.vocab_.symbols_[i] + ": " + this.histogram_[i]);
    }
  }
}

/**
 * Exported APIs.
 */
exports.HistogramLanguageModel = HistogramLanguageModel;
