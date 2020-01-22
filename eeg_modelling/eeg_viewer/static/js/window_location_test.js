// Copyright 2020 The Google Research Authors.
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

goog.module('eeg_modelling.eeg_viewer.WindowLocation.tests');
goog.setTestOnly();

const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const MockControl = goog.require('goog.testing.MockControl');
const PropertyReplacer = goog.require('goog.testing.PropertyReplacer');
const WindowLocation = goog.require('eeg_modelling.eeg_viewer.WindowLocation');
const mockmatchers = goog.require('goog.testing.mockmatchers');
const testSuite = goog.require('goog.testing.testSuite');

let mockControl;
let storeData;

const _ = mockmatchers.ignoreArgument;

const propertyReplacer = new PropertyReplacer();
const windowLocation = WindowLocation.getInstance();


testSuite({

  setUp() {
    mockControl = new MockControl();
    location.hash = '';

    storeData = {
      chunkStart: '42',
      chunkDuration: '3',
      chunkGraphData: 'high ground',
      tfExSSTablePath: '/hello/there',
      predictionSSTablePath: 'general/kenobi',
      tfExFilePath: '/good/bye',
      predictionFilePath: 'hello/world',
      sstableKey: 'grievous',
      channelIds: ['0', '1', '1', '2', '3', '5'],
      lowCut: 0,
      highCut: 120,
      notch: 50,
      queryDataDirty: false,
      edfPath: '/edf/file',
    };
  },

  testParseFragment() {
    location.hash =
        ('#chunkstart=42&chunkduration=3&' +
         'tfexsstablepath=/hello/there&' +
         'predictionsstablepath=general/kenobi&' +
         'sstablekey=grievous&' +
         'edfpath=/edf/file&' +
         'tfexfilepath=/good/bye&' +
         'predictionfilepath=hello/world&' +
         'channelids=0,1,1,2,3,5');
    const mockFragmentData = {
      chunkstart: '42',
      chunkduration: '3',
      tfexsstablepath: '/hello/there',
      predictionsstablepath: 'general/kenobi',
      tfexfilepath: '/good/bye',
      predictionfilepath: 'hello/world',
      sstablekey: 'grievous',
      edfpath: '/edf/file',
      channelids: '0,1,1,2,3,5',
      lowcut: null,
      highcut: null,
      notch: null,
    };

    assertObjectEquals(mockFragmentData, windowLocation.parseFragment());
  },

  testMakeDataRequest() {
    const mockDispatcher = mockControl.createLooseMock(Dispatcher);
    mockDispatcher.sendAction(_).$once();
    propertyReplacer.replace(Dispatcher, 'getInstance', () => mockDispatcher);

    mockControl.$replayAll();
    windowLocation.makeDataRequest();
    mockControl.$verifyAll();
  },

  testHandleRequestParams() {
    const mockHash =
        ('#chunkstart=42&chunkduration=3&' +
         'lowcut=0&highcut=120&notch=50&' +
         'tfexsstablepath=/hello/there&' +
         'predictionsstablepath=general/kenobi&' +
         'sstablekey=grievous&' +
         'edfpath=/edf/file&' +
         'tfexfilepath=/good/bye&' +
         'predictionfilepath=hello/world&' +
         'channelids=0,1,1,2,3,5');

    windowLocation.handleRequestParams(storeData, ['tfExSSTablePath']);
    assertEquals(mockHash, location.hash);
  },

  testHandleRequestParams_EmptyValues() {
    const mockHash =
        ('#chunkstart=42&chunkduration=3&' +
         'lowcut=0&highcut=120&notch=50&' +
         'tfexsstablepath=/hello/there&' +
         'predictionsstablepath=&' +
         'sstablekey=grievous&' +
         'edfpath=&' +
         'tfexfilepath=&' +
         'predictionfilepath=&' +
         'channelids=0,1,1,2,3,5');

    storeData.predictionSSTablePath = null;
    storeData.edfPath = null;
    storeData.tfExFilePath = null;
    storeData.predictionFilePath = null;

    windowLocation.handleRequestParams(storeData, ['tfExFilePath']);
    assertEquals(mockHash, location.hash);
  },

  tearDown() {
    mockControl.$tearDown();
    location.hash = '';
  },

});
