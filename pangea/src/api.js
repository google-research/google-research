// Copyright 2019 The Google Research Authors.
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
 * @fileoverview Mock crowd annotation API (e.g., Amazon Mechanical Turk).
 */

var API;

if (typeof API == undefined) {
  /**
   * Mock crowd annotation API (e.g., Amazon Mechanical Turk).
   */
  API = class {
    /**
     * Returns init arguments.
     * @return {!Object} init arguments.
     */
    getInitArgs() {
      return {};
    }

    /**
     * Initializes the plugin.
     * @param {!Function} onInit Accepts this as an argument.
     */
    init(onInit) {
      onInit(this.getInitArgs());
    }

    /**
     * Submits the answer.
     * @param {!Object} answer An object.
     */
    submitAnswer(answer) {
      window.alert(answer);
    }
  };
}

(global => {
  if (typeof define === 'function' && define.amd) {
    define([], () => API);

  } else if (typeof module !== 'undefined' && typeof exports === 'object') {
    module.exports = API;

  } else if (global !== undefined) {
    global.API = API;
  }
})(this);
