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
 *     http://.../anthea.html?test=test-paragraph.js
 * and then click around through the rating process.
 */
const testProjectTSVData = `{"source_language":"en","target_language":"en"}
This is the first​-​sentence​.​​ This is the second sentence. It includes this string​.​​ The third sentence​.​​ Sentence 4​.	This is its translation​.​​ This is the translation (​of the second sentence​)​.​​ A translation of the 3rd sentence​.​​ Translated sentence 4​.	doc-42	system-GL

A second paragraph​. ​​ This is a long sentence with meaninglessness embedded as an essential artifact that requires the reader to comtemplate their exact place in the vast expanse of existence​. ​​ Translater had no clue on this one​. ​​ The last line​. ​​ The last word​.	A very flawed translation​.​​ Waiting for whom​?​​ Given the existence as uttered forth in the public works of Puncher and Wattmann of a personal God quaquaquaqua with white beard quaquaquaqua outside time without extension who from the heights of divine apathia divine athambia divine aphasia loves us dearly with some exceptions for reasons unknown but time will tell and suffers like the divine Miranda with those who for reasons unknown .​.​.	doc-42	system-GL
      `;
const testProjectName = 'Google-MQM-Test-Paragraph';
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
    4 /* HOTW rate */);
