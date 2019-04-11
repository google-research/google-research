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
 * @fileoverview Provide util functions.
 *
 * The toggle/hide/show element functions are not used in the whole
 * code, since the code is rather straight forward and can be written directly.
 * However, using these functions is preferred sometimes to improve readability,
 * when there are many elements being hidden or shown.
 */

goog.module('eeg_modelling.eeg_viewer.utils');

/**
 * Hides or shows an HTML element.
 * @param {string} elementId HTML Id of the element.
 * @param {boolean} hide Whether to hide or show the element.
 */
function toggleElement(elementId, hide) {
  const element = document.getElementById(elementId);
  element.classList.toggle('hidden', hide);
}

/**
 * Hides an HTML element.
 * @param {string} elementId HTML Id of the element.
 */
function hideElement(elementId) {
  toggleElement(elementId, true);
}

/**
 * Shows an HTML element.
 * @param {string} elementId HTML Id of the element.
 */
function showElement(elementId) {
  toggleElement(elementId, false);
}


/**
 * Hides or show a Material Design Lite spinner.
 * @param {string} elementId HTML Id of the spinner.
 * @param {boolean} hide Whether to hide or show the spinner.
 */
function toggleSpinner(elementId, hide) {
  const element = document.getElementById(elementId);
  element.classList.toggle('hidden', hide);
  element.classList.toggle('is-active', !hide);
}

/**
 * Shows a spinner.
 * @param {string} elementId HTML Id of the spinner.
 */
function showSpinner(elementId) {
  toggleSpinner(elementId, false);
}

/**
 * Hides a spinner.
 * @param {string} elementId HTML Id of the spinner.
 */
function hideSpinner(elementId) {
  toggleSpinner(elementId, true);
}


exports = {
  toggleElement,
  hideElement,
  showElement,
  toggleSpinner,
  showSpinner,
  hideSpinner,
};
