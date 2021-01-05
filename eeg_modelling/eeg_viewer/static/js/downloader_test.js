// Copyright 2021 The Google Research Authors.
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
 * @fileoverview Provide tests for the Downloader.
 */

goog.module('eeg_modelling.eeg_viewer.Downloader.tests');
goog.setTestOnly();

const DownloadHelper = goog.require('jslib.DownloadHelper');
const Downloader = goog.require('eeg_modelling.eeg_viewer.Downloader');
const MockControl = goog.require('goog.testing.MockControl');
const googJson = goog.require('goog.json');
const mockmatchers = goog.require('goog.testing.mockmatchers');
const singleton = goog.require('goog.testing.singleton');
const testSuite = goog.require('goog.testing.testSuite');

let mockControl;
let downloader;
let storeData;
let fileParams;

testSuite({

  setUp() {
    mockControl = new MockControl();

    singleton.resetAll();
    downloader = Downloader.getInstance();

    fileParams = {
      tfExSSTablePath: 'sstablePath',
      sstableKey: 'key',
      predictionSSTablePath: null,
      edfPath: null,
      predictionFilePath: null,
      tfExFilePath: null,
    };

    storeData = Object.assign({}, fileParams, {
      absStart: 13,
      chunkStart: 7,
      chunkDuration: 3,
      numSecs: 100,
      downloadData: null,
    });
  },

  testHandleDownloadData() {
    const downloadData = {
      properties: [
        'chunkStart',
        'sstableKey',
      ],
      name: 'arbitrary-name',
      timestamp: 'any',
    };

    storeData.downloadData = downloadData;

    const formattedDataMatcher = new mockmatchers.ArgumentMatcher((data) => {
      if (!googJson.isValid(data)) {
        return false;
      }
      const parsedData = JSON.parse(data);
      assertEquals(downloadData.timestamp, parsedData.timestamp);
      assertObjectEquals(fileParams, parsedData.fileParams);
      assertObjectEquals({chunkStart: 7, sstableKey: 'key'}, parsedData.store);
      return true;
    });

    const filenameMatcher = new mockmatchers.ArgumentMatcher((filename) => {
      return typeof filename === 'string' &&
          filename.includes(downloadData.name) &&
          filename.includes(storeData.sstableKey) && filename.endsWith('.json');
    });

    const jsonTypeMatcher = new mockmatchers.ArgumentMatcher((contentType) => {
      return typeof contentType === 'string' && contentType.includes('json');
    });

    const downloadMock =
        mockControl.createMethodMock(DownloadHelper, 'download');
    downloadMock(formattedDataMatcher, filenameMatcher, jsonTypeMatcher)
        .$once();

    mockControl.$replayAll();

    downloader.handleDownloadData(storeData);

    mockControl.$verifyAll();
  },

  tearDown() {
    mockControl.$tearDown();
  },

});
