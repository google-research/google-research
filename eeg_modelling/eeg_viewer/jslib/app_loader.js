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
 * @fileoverview Wrapper of the JS app to load Google Charts Lib
 * dynamically.
 */

/**
 * Load a js script by adding a script tag to the document.
 * @param {string} src JS script source
 * @param {?Function} callback Function to call after the script loads
 */
function loadScript(src, callback) {
  const script = document.createElement('script');
  if (callback) {
    script.onload = callback;
  }
  script.src = src;
  document.head.appendChild(script);
}

const CHARTS_URL = 'https://www.gstatic.com/charts/loader.js';
const CHARTS_VERSION = 'current';
const COMPILED_APP = 'static/js/compiled_app.js';

loadScript(CHARTS_URL, () => {
  google.charts.load(CHARTS_VERSION, {
    packages: ['corechart'],
  });
  google.charts.setOnLoadCallback(() => loadScript(COMPILED_APP));
});
