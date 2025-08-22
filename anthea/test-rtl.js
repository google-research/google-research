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
 *     http://.../anthea.html?test=test-rtl.js
 * and then click around through the rating process.
 */
const testProjectTSVData = `{"source-language":"en", "target-language": "ur"}
One 1​, two 2​, three 3​, four 4​.	ایک 1​، دو 2​، تین 3​، چار 4​۔	Test-en-ur	Sys-en-ur
		
Five 5​, six seven eight (​6​, 7​, 8​)​.	پانچ 5​، چھ سات آٹھ (​6​، 7​، 8​)​۔	Test-en-ur	Sys-en-ur
`;
const testProjectName = 'Google-MQM-Test-RTL';
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
    testTemplateName, testProjectName, testProjectTSVData, 0);
