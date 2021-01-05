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
 * @fileoverview Simple vocabulary abstraction.
 *
 * This is used to store symbols and map them to contiguous integers.
 */

// Special symbol denoting the root node.
const rootSymbol = 0;

// Symbol name of the root symbol, also used for out-of-vocabulary symbols.
const rootSymbolName = "<R>";

// The special out-of-vocabulary (OOV) symbol.
const oovSymbol = "<OOV>";

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
    // The current symbol container length is used as a unique ID. Because
    // the symbol IDs are used to index the array directly, the symbol ID is
    // assigned before updating the array.
    const symbol_id = this.symbols_.length;
    this.symbols_.push(symbol);
    return symbol_id;
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
exports.rootSymbol = rootSymbol;
exports.Vocabulary = Vocabulary;
