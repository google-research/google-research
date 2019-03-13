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

goog.module('eeg_modelling.eeg_viewer.Menus.tests');
goog.setTestOnly();

const Menus = goog.require('eeg_modelling.eeg_viewer.Menus');
const MockControl = goog.require('goog.testing.MockControl');
const dom = goog.require('goog.dom');
const testSuite = goog.require('goog.testing.testSuite');

let mockControl;
let storeData;


testSuite({

  setUp() {
    mockControl = new MockControl();

    storeData = {
      'tfExSSTablePath': 'tf example sstable',
      'predictionSSTablePath': null,
      'sstableKey': 'sstable key',
      'edfPath': 'edf file',
    };

    const fileMenu = document.createElement('div');
    fileMenu.id = 'file-menu';

    const addInput = (elementId) => {
      const inputElement = document.createElement('input');
      inputElement.id = `${elementId}`;
      fileMenu.appendChild(inputElement);
    };
    addInput('input-tfex-sstable');
    addInput('input-prediction-sstable');
    addInput('input-key');
    addInput('input-edf');
    addInput('input-tfex-path');
    addInput('input-prediction-path');

    document.body.appendChild(fileMenu);

    const displayFilePath = document.createElement('div');
    displayFilePath.id = 'display-file-path';
    document.body.appendChild(displayFilePath);
  },

  testHandleFileParams() {
    Menus.getInstance().handleFileParams(storeData);

    const tfExSSTable = document.querySelector('#input-tfex-sstable');
    assertEquals('tf example sstable', tfExSSTable.value);

    const predSSTable = document.querySelector('#input-prediction-sstable');
    assertEquals('', predSSTable.value);

    const key = document.querySelector('#input-key');
    assertEquals('sstable key', key.value);

    const edfPath = document.querySelector('#input-edf');
    assertEquals('edf file', edfPath.value);
  },

  tearDown() {
    dom.removeChildren(dom.getDocument().body);
    mockControl.$tearDown();
  },
});
