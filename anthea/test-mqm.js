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
 *     http://.../anthea.html?test=test-mqm.js
 * and then click around through the rating process.
 */
const testProjectTSVData = `{"source_language":"en","target_language":"en"}

This is the first​-​sentence​. 	This\\tis its\\n\u200b\u200btranslation​.	doc-42	system-GL
This is the second sentence. It includes this long string that tests text-wrapping: http://01234567890123456789012345678901234567890123456789012345678901234567890123456789012345678901234567890123456789​.	This is the translation (​of the second sentence​)​.	doc-42	system-GL
The third sentence​. 	A translation of the 3rd sentence​. 	doc-42	system-GL
# A sentence beginning with # 4​.	Translated sentence 4​.	doc-42	system-GL

A second paragraph​. This is a long sentence with meaninglessness and an embedded\\n\\n\u200b\u200bParagraph break embedded as an essential artifact that requires the reader to comtemplate their exact place in the vast expanse of existence​.	Translater had no clue on this one​.	doc-42	system-GL
The first sentence in the second document​.	The translation of the first sentence with an internal line\\n\u200b\u200bbreak in the second document​.	doc-99	system-DL
The 2nd sentence in the second document​.	The translation of the doosra sentence in the second document​.	doc-99	system-DL
The third and final document​.	The translation​, of the opening sentence in the third document​.	doc-1531	system-DL
The last line​. The last word​. Waiting for whom​?	Given the existence as uttered forth in the public works of Puncher and Wattmann of a personal God quaquaquaqua with white beard quaquaquaqua outside time without extension who from the heights of divine apathia divine athambia divine aphasia loves us dearly with some exceptions for reasons unknown but time will tell and suffers like the divine Miranda with those who for reasons unknown .​.​.	doc-1531	system-DL
      `;
const testProjectName = 'Google-MQM-Test-Dev-Only-42';
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
