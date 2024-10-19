// Copyright 2024 The Google Research Authors.
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

const JspbMap = goog.require('jspb.Map');
const {safeElement} = goog.require('safevalues.dom');
const {sanitizeHtml} = goog.require('safevalues');

/**
 * @typedef {{
 *   componentHandler: {
 *     upgradeElement: function(!Element):void,
 *   },
 * }}
 */
let MDLEnhancedWindow;

/**
 * @typedef {{
 *   MaterialCheckbox: {
 *     check: function():void,
 *     uncheck: function():void,
 *   },
 * }}
 */
let MaterialCheckboxElement;

/**
 * @typedef {{
 *   elementId: string,
 *   elementOffsetX: number,
 *   elementOffsetY: number,
 * }}
 */
let DragOptions;


/**
 * Returns a cast HTML Input element.
 * @param {string} id The HTML id of the element.
 * @return {!HTMLInputElement} The input element.
 */
function getInputElement(id) {
  return /** @type {!HTMLInputElement} */ (document.getElementById(id));
}

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
function toggleMDLSpinner(elementId, hide) {
  const element = document.getElementById(elementId);
  element.classList.toggle('hidden', hide);
  element.classList.toggle('is-active', !hide);
}

/**
 * Shows a spinner.
 * @param {string} elementId HTML Id of the spinner.
 */
function showMDLSpinner(elementId) {
  toggleMDLSpinner(elementId, false);
}

/**
 * Hides a spinner.
 * @param {string} elementId HTML Id of the spinner.
 */
function hideMDLSpinner(elementId) {
  toggleMDLSpinner(elementId, true);
}


/**
 * Run MDL function upgradeElement on a HTML Element.
 * Run this function on any MDL element created in JS, to enable all MDL
 * functionalities. The element must be inserted in the DOM before calling this
 * function.
 * @param {!Element} element Element to upgrade.
 */
function upgradeMDLElement(element) {
  const mdlCastWindow = /** @type {!MDLEnhancedWindow} */ (window);
  mdlCastWindow.componentHandler.upgradeElement(element);
}

/**
 * Add a MDL tooltip element to display on hover of a target element.
 * @param {!Element} parentElement HTML element to add the tooltip element.
 *     Must be inserted in the DOM before calling this function
 * @param {string} targetId HTML id of the element targeted by the tooltip.
 * @param {string} tooltipText Text to display on the tooltip. HTML is allowed.
 */
function addMDLTooltip(parentElement, targetId, tooltipText) {
  const tooltip = document.createElement('div');
  tooltip.setAttribute('for', targetId);
  tooltip.className = 'mdl-tooltip mdl-tooltip--large';
  safeElement.setInnerHtml(tooltip, sanitizeHtml(tooltipText));
  parentElement.appendChild(tooltip);

  upgradeMDLElement(tooltip);
}

/**
 * Add a MDL checkbox element to another HTML element.
 * @param {!Element} parentElement HTML element to add the checkbox element.
 *     Must be inserted in the DOM before calling this function.
 * @param {string} checkboxId HTML id to use in the checkbox.
 * @param {string} text Text to display in the checkbox.
 * @param {function(boolean):void} onChange Callback to trigger when the
 *     checkbox changes.
 */
function addMDLCheckbox(parentElement, checkboxId, text, onChange) {
  const labelElement = document.createElement('label');
  labelElement.className = 'mdl-checkbox mdl-js-checkbox';
  labelElement.setAttribute('for', checkboxId);

  const inputElement =
      /** @type {!HTMLInputElement} */ (document.createElement('input'));
  inputElement.id = checkboxId;
  inputElement.setAttribute('type', 'checkbox');
  inputElement.className = 'mdl-checkbox__input';

  inputElement.onchange = () => onChange(inputElement.checked);

  const spanElement = document.createElement('span');
  spanElement.className = 'mdl-checkbox__label';
  spanElement.textContent = text;

  labelElement.appendChild(/** @type {?Node} */ (inputElement));
  labelElement.appendChild(spanElement);
  parentElement.appendChild(labelElement);

  upgradeMDLElement(labelElement);
}


/**
 * Check or uncheck a MDL checkbox element.
 * See here for MDL checkbox definition:
 * https://getmdl.io/components/#toggles-section
 * @param {!Element} labelElement Label element container of the checkbox.
 * @param {boolean} checked Indicates if should be checked or unchecked.
 */
function toggleMDLCheckbox(labelElement, checked) {
  const labelMDL = /** @type {!MaterialCheckboxElement} */ (labelElement);
  if (checked) {
    labelMDL.MaterialCheckbox.check();
  } else {
    labelMDL.MaterialCheckbox.uncheck();
  }
}

/**
 * Return the keys of a proto map.
 * @param {!JspbMap} protoMap map to extract the keys from.
 * @return {!Array<string>} Array with Map keys.
 */
function getProtoMapKeys(protoMap) {
  const keys = [];
  const keyIter = protoMap.keys();
  let key = keyIter.next();
  while (!key.done) {
    keys.push(key.value);
    key = keyIter.next();
  }
  return keys;
}

/**
 * Gets the minimum value of an array.
 * This function is preferred over Math.min(...array), since said function will
 * throw overflow with an array too large.
 * @param {!Array<number>} array with values.
 * @return {number} minimum number of the array.
 */
function getArrayMin(array) {
  return array.reduce(
      (val, currentMin) => (val < currentMin ? val : currentMin),
      array[0],
  );
}

/**
 * Gets the maximum value of an array.
 * This function is preferred over Math.max(...array), since said function will
 * throw overflow with an array too large.
 * @param {!Array<number>} array with values.
 * @return {number} maximum number of the array.
 */
function getArrayMax(array) {
  return array.reduce(
      (val, currentMax) => (val > currentMax ? val : currentMax),
      array[0],
  );
}

exports = {
  DragOptions,
  getInputElement,
  toggleElement,
  hideElement,
  showElement,
  toggleMDLSpinner,
  showMDLSpinner,
  hideMDLSpinner,
  upgradeMDLElement,
  addMDLTooltip,
  addMDLCheckbox,
  toggleMDLCheckbox,
  getProtoMapKeys,
  getArrayMin,
  getArrayMax,
};
