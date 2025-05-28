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
 *     http://.../anthea.html?test=test-mqm-source-media.js
 * and then click around through the rating process.
 */
const testProjectTSVData = `{"source_language":"en","target_language":"en"}
 	A translation of the 1st sentence in the video​.	doc-42	system-GL	{"source_media": {"url": "https://www.youtube.com/embed/jNQXAC9IVRw?si=5biYW2R90rsuwGe5", "type": "video"}}
	This is the translation (​of the second sentence​) of the video​.	doc-42	system-GL

 	A translation of the 3rd sentence (new paragraph)​. 	doc-42	system-GL
	Translated sentence 4​.	doc-42	system-GL
 	This\\tis another\\n\u200b\u200btranslation, of an image​.	doc-43	system-GL	{"source_media": {"url": "https://images.unsplash.com/vector-1741034409454-14c337369933", "type": "image"}}
	This is the translation (​of the second sentence​) in an image​.	doc-43	system-GL

 	A translation of the 3rd sentence (new paragraph)​. 	doc-43	system-GL
      `;
const testProjectName = 'Google-MQM-Source-Media-Test-Dev-Only-42';
const testTemplateName = 'MQM-Source-Media';
const activeName = antheaManager.activeName(testProjectName, testTemplateName);
try {
  const activeDataKey = antheaManager.ACTIVE_KEY_PREFIX_ + activeName;
  window.localStorage.removeItem(activeDataKey);
  const activeResultsKey =
      antheaManager.ACTIVE_RESULTS_KEY_PREFIX_ + activeName;
  window.localStorage.removeItem(activeResultsKey);
} catch (err) {
  console.log('Caught exception (harmless if for "no such item"): ' + err);
}
antheaManager.createActive(
    testTemplateName, testProjectName, testProjectTSVData,
    50 /* High HOTW rate, for testing */);