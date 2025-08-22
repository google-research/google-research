// Copyright 2025 The Google Research Authors.
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
 * To run this test, visit the URL:
 *     http://.../anthea.html?test=test-mqm-ja.js
 * and then click around through the rating process.
 */
const testProjectTSVData = `{"source_language":"en","target_language":"ja"}

This is the first​-​sentence​.	これが最初の文です。	doc-enja	system-GL
Sentence 4​.	文4。	doc-enja	system-GL
      `;
const testProjectName = 'Google-MQM-Test-JA';
const testTemplateName = 'MQM';
const activeName = antheaManager.activeName(testProjectName, testTemplateName);
try {
  const activeDataKey = antheaManager.ACTIVE_KEY_PREFIX_ + activeName;
  window.localStorage.removeItem(activeDataKey);
  const activeResultsKey =
      antheaManager.ACTIVE_RESULTS_KEY_PREFIX_ + activeName;
  window.localStorage.removeItem(activeResultsKey);
} catch (err) {
  console.log('Caught exception (harmless if for "no such item": ' + err);
}
antheaManager.createActive(
    testTemplateName, testProjectName, testProjectTSVData,
    50 /* High HOTW rate, for testing */);
