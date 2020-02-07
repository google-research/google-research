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
 * @fileoverview Simple vocabulary abstraction.
 *
 * This is used to store symbols and map them to contiguous integers.
 */

// Symbol name of the root symbol, also used for out-of-vocabulary symbols.
const rootSymbolName = "Root";

// The special out-of-vocabulary (OOV) symbol.
const oovSymbol = "OOV";

/**
 * Vocabulary of symbols, which is a set of symbols that map one-to-one to
 * unique integers.
 * @final
 */
class Vocabulary {
  constructor() {
    this.symbols_ = Array();
    this.symbols_.push(rootSymbolName);
    this.oovSymbol_ = -1;
  }

  /**
   * Adds symbol to the vocabulary returning its unique ID.
   * @param {string} symbol Symbol to be added.
   * @return {number} Symbol ID.
   * @final
   */
  addSymbol(symbol) {
    let pos = this.symbols_.indexOf(symbol);
    if (pos >= 0) {
      return pos;
    }
    this.symbols_.push(symbol);
    return this.symbols_.length - 1;  // Minus the root.
  }

  /**
   * Returns the vocabulary symbol ID if it exists, otherwise maps the supplied
   * symbol to out-of-vocabulary (OOV) symbol. Note, this method is *only* used
   * for testing.
   * @param {string} symbol Symbol to be looked up.
   * @return {number} Symbol ID.
   * @final
   */
  getSymbolOrOOV(symbol) {
    let pos = this.symbols_.indexOf(symbol);
    if (pos >= 0) {
      return pos;
    }
    this.oovSymbol_ = this.addSymbol(oovSymbol);
    return this.oovSymbol_;
  }

  /**
   * Returns cardinality of the vocabulary.
   * @return {number} Size.
   * @final
   */
  size() {
    return this.symbols_.length;
  }
}

/**
 * Exported APIs.
 */
exports.Vocabulary = Vocabulary;
