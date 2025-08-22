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
 *     http://.../anthea.html?test=test-hotw-pretend.js
 * and then click around through the rating process.
 */
const testProjectTSVData = `{"source_language":"en","target_language":"en","hotw_percent":60,"hotw_pretend":1}
This is the first​-​sentence​.	This is its translation​.	doc-42	system-GL
This is the second sentence​.	This is the translation (​of the second sentence​)​.	doc-42	system-GL
The third sentence​.	A translation of the 3rd sentence​.	doc-42	system-GL
Sentence 4​.	Translated sentence 4​.	doc-42	system-GL

A second paragraph​. This is a long sentence with meaninglessness embedded as an essential artifact that requires the reader to comtemplate their exact place in the vast expanse of existence​.	Translater had no clue on this one​.	doc-42	system-GL
The nth sentence in this document​.	The translation of the nth sentence in the document​.	doc-42	system-GL
The n+1th sentence in this document​.	The translation of the weird sentence in the weird document​.	doc-42	system-GL
The penultimate sentence in this document​.	The translation​, of the penultimate sentence in the document​.	doc-42	system-GL
The last line​. The last word​. Waiting for whom​?	Given the existence as uttered forth in the public works of Puncher and Wattmann of a personal God quaquaquaqua with white beard quaquaquaqua outside time without extension who from the heights of divine apathia divine athambia divine aphasia loves us dearly with some exceptions for reasons unknown but time will tell and suffers like the divine Miranda with those who for reasons unknown .​.​.	doc-42	system-GL
      `;
const testProjectName = 'Google-MQM-HOTW-Pretend-Test';
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
/** The 0 hotw_percent parameter gets overriden by the project JSON params */
antheaManager.createActive(
    testTemplateName, testProjectName, testProjectTSVData, 0);
