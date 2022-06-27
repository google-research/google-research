// Copyright 2022 The Google Research Authors.
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

goog.module('eeg_modelling.eeg_viewer.Annotations.tests');
goog.setTestOnly();

const Annotations = goog.require('eeg_modelling.eeg_viewer.Annotations');
const MockControl = goog.require('goog.testing.MockControl');
const dom = goog.require('goog.dom');
const testSuite = goog.require('goog.testing.testSuite');

let mockControl;
let storeData;


testSuite({

  setUp() {
    mockControl = new MockControl();

    storeData = {
      absStart: 13,
      chunkStart: 7,
      chunkDuration: 3,
      fileInputDirty: true,
      numSecs: 100,
      requestStatus: 'RETURNED',
      annotations: [{labelText: 'Douglas Adams', startTime: 0}],
      patientId: 'KENOBI',
      sstableKey: 'hello/THERE.edf#0',
    };

    const panel = document.createElement('div');
    panel.id = 'labels-panel';
    const title = document.createElement('div');
    title.classList.add('mdl-card__title-text');
    panel.appendChild(title);
    const annotationTable = document.createElement('table');
    annotationTable.classList.add('annotation');
    const tableBody = document.createElement('tbody');
    annotationTable.appendChild(tableBody);
    document.body.appendChild(annotationTable);
    const annotationItem = document.createElement('li');
    annotationItem.classList.add('annotation');
    panel.appendChild(annotationItem);
    document.body.appendChild(panel);
  },

  testHandleAnnotations() {
    Annotations.getInstance().handleAnnotations(storeData);

    const annotationTableCells = document.querySelectorAll('.annotation td');
    assertEquals('00:00:13', annotationTableCells.item(1).innerHTML);
  },

  tearDown() {
    dom.removeChildren(dom.getDocument().body);
    mockControl.$tearDown();
  },

});
