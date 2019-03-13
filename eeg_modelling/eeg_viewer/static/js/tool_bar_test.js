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

goog.module('eeg_modelling.eeg_viewer.ToolBar.tests');
goog.setTestOnly();

const MockControl = goog.require('goog.testing.MockControl');
const ToolBar = goog.require('eeg_modelling.eeg_viewer.ToolBar');
const WaveformMetadata = goog.require('proto.eeg_modelling.protos.WaveformMetadata');
const dom = goog.require('goog.dom');
const testSuite = goog.require('goog.testing.testSuite');

let mockControl;
let prevButton;
let nextButton;
let prevSecButton;
let nextSecButton;

const toolBar = ToolBar.getInstance();


testSuite({

  setUp() {
    mockControl = new MockControl();
    const toolBar = document.createElement('div');
    toolBar.id = 'tool-bar';
    document.body.appendChild(toolBar);
    prevButton = document.createElement('button');
    prevButton.id = 'prev-button';
    toolBar.appendChild(prevButton);
    prevSecButton = document.createElement('button');
    prevSecButton.id = 'prev-sec-button';
    toolBar.appendChild(prevSecButton);
    nextButton = document.createElement('button');
    nextButton.id = 'next-button';
    toolBar.appendChild(nextButton);
    nextSecButton = document.createElement('button');
    nextSecButton.id = 'next-sec-button';
    toolBar.appendChild(nextSecButton);

    const montageList = document.createElement('ul');
    montageList.setAttribute('for', 'montage-dropdown');
    const montageOption = document.createElement('li');
    montageOption.id = 'test-montage';
    montageList.appendChild(montageOption);
    toolBar.appendChild(montageList);
  },

  testHandleIndexChannelMap() {
    const metadata = new WaveformMetadata();
    metadata.getChannelDictMap()
        .set(2, 'B')
        .set(3, 'D');
    const storeData = {
      indexChannelMap: metadata.getChannelDictMap(),
    };

    const mockMontages = mockControl.createMethodMock(toolBar, 'getMontages');
    mockMontages().$returns({'test-montage': ['A-B', 'B|C-D']});
    const mockSelect = mockControl.createMethodMock(toolBar, 'selectDropdown');
    mockSelect('montage-dropdown', null, null).$once();
    mockSelect('montage-dropdown', 'test-montage', ['2-3']).$once();

    mockControl.$replayAll();

    toolBar.handleIndexChannelMap(storeData);
    document.getElementById('test-montage').click();

    mockControl.$verifyAll();
  },

  testHandleChunkNavigation() {
    const storeData = {
      chunkStart: 0,
      chunkDuration: 10,
      chunkGraphData: 'fake data',
      highCut: 90,
      lowCut: 0.8,
      notch: 42,
      numSecs: 20,
      sensitivity: 4,
      timeScale: 1,
    };

    const mockSelect = mockControl.createMethodMock(toolBar, 'selectDropdown');
    mockSelect('low-cut-dropdown', '0.8 Hz', null).$once();
    mockSelect('high-cut-dropdown', '90 Hz', null).$once();
    mockSelect('notch-dropdown', '42 Hz', null).$once();
    mockSelect('sensitivity-dropdown',
        `4 ${String.fromCharCode(956)}V`, null).$once();
    mockSelect('time-frame-dropdown', '10 sec', null).$once();
    mockSelect('grid-dropdown', '1 / sec', null).$once();

    mockControl.$replayAll();

    toolBar.handleChunkNavigation(storeData);

    mockControl.$verifyAll();

    assertTrue(prevButton.disabled);
    assertFalse(nextButton.disabled);
  },

  testHandleChunkNavigation_DisableAll() {
    const storeData = {
      chunkStart: 1,
      chunkDuration: 5,
      numSecs: 10,
    };
    toolBar.handleChunkNavigation(storeData);
    assertTrue(prevButton.disabled);
    assertTrue(nextButton.disabled);
  },

  tearDown() {
    dom.removeChildren(dom.getDocument().body);
    mockControl.$tearDown();
  },

});
