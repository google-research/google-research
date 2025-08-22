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
const testProjectTSVData = `{"source_language":"en","target_language":"en","shuffle_seed":937563829}
This is the first​-​sentence​. 	This\\tis its\\n\u200b\u200bfirst translation​ .	doc-42	system-GL
This is the second sentence.	This is the translation (​of the second sentence​) which is a little longer​.	doc-42	system-GL
The third sentence​.	A translation of the 3rd sentence​. 	doc-42	system-GL
# A sentence beginning with # 4​.	Translated sentence 4​.	doc-42	system-GL

A second paragraph​. It includes this long string that tests text-wrapping​.	Translater had no clue on this one​.	doc-42	system-GL
This is the first​-​sentence​. 	THIS\\tIS ITS\\n\u200b\u200bSECOND TRANSLATION.	doc-42	system-AL
This is the second sentence.	THIS IS THE TRANSLATION (​OF THE SECOND SENTENCE​)​.	doc-42	system-AL
The third sentence​.	A TRANSLATION OF THE 3RD sentence​. 	doc-42	system-AL
# A sentence beginning with # 4​.	TRANSLATED SENTENCE 4​.	doc-42	system-AL

A second paragraph​. It includes this long string that tests text-wrapping​.	TRANSLATER HAD no clue on this one​.	doc-42	system-AL
The first sentence in the second document​.	The translation of the first sentence with an internal line\\n\u200b\u200bbreak in the second document which is very very very vvery lonoooooooooooog​.	doc-99	system-GL
The 2nd sentence in the second document​.	The translation of the doosra sentence in the second document​.	doc-99	system-GL
The first sentence in the second document​.	THE TRANSLATION of the first sentence with an internal line\\n\u200b\u200bbreak in the second document​.	doc-99	system-AL
The 2nd sentence in the second document​.	THE TRANSLATION of the doosra sentence in the second document​.	doc-99	system-AL
This is the first sentence in the third document.	This is the translation of the first sentence in the third document.	doc-100	system-GL
This is the 2nd sentence.	This is the 2nd sentence translation.	doc-100	system-GL
This is the first sentence in the third document.	THIS IS THE TRANSLATION OF THE FIRST SENTENCE IN THE THIRD DOCUMENT.	doc-100	system-AL
This is the 2nd sentence.	THIS IS THE 2ND SENTENCE TRANSLATION.	doc-100	system-AL
      `;
const testProjectName = 'Google-MQM-Test-Dev-Only-42';
const testTemplateName = 'MQM-SxS';
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
    80 /* High HOTW rate, for testing */);
