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

/**
 * @fileoverview Provide tests for the Uploader.
 */

goog.module('eeg_modelling.eeg_viewer.Uploader.tests');
goog.setTestOnly();

const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const MockControl = goog.require('goog.testing.MockControl');
const Uploader = goog.require('eeg_modelling.eeg_viewer.Uploader');
const mockmatchers = goog.require('goog.testing.mockmatchers');
const singleton = goog.require('goog.testing.singleton');
const testSuite = goog.require('goog.testing.testSuite');

let mockControl;
let uploader;


/**
 * Mocks the DOM's function getElementById, returning a fake fileInput with
 * a specified content.
 * Returns an empty element for other html id's provided.
 * @param {string} fileContent String that will be passed as file content.
 */
function mockDomWithFileInput(fileContent) {
  const mockFileBlob = new Blob([fileContent], {
    type: 'application/json'
  });
  const mockInput = {
    files: [mockFileBlob],
  };
  const mockGetElement =
      mockControl.createMethodMock(document, 'getElementById');

  mockGetElement(mockmatchers.isString).$anyTimes().$does((id) => {
    if (id === 'uploader-file-input') {
      return mockInput;
    }
    switch (id) {
      case 'uploader-file-input':
        return mockInput;
      case 'uploader-text-display':
        return document.createElement('input');
      case 'uploader-modal':
      default:
        return document.createElement('div');
    }
  });
}


testSuite({

  setUp() {
    mockControl = new MockControl();

    singleton.reset();
    uploader = Uploader.getInstance();
  },

  async testUpload() {
    const mockFileContent = JSON.stringify({
      store: {
        chunkStart: 3,
        highCut: 10,
      },
      fileParams: {},
      timestamp: 'any',
    });
    mockDomWithFileInput(mockFileContent);

    const mockDispatcher = mockControl.createStrictMock(Dispatcher);
    mockDispatcher.sendAction({
      actionType: Dispatcher.ActionType.IMPORT_STORE,
      data: {
        chunkStart: 3,
        highCut: 10,
      },
    }).$once();
    const mockGetInstance = mockControl.createMethodMock(Dispatcher,
        'getInstance');
    mockGetInstance().$returns(mockDispatcher);

    mockControl.$replayAll();

    await uploader.upload();

    mockControl.$verifyAll();
  },

  async testUpload_noInjection() {
    const mockFileContent = JSON.stringify({
      store: {
        fileType: '<script>alert("malicious code")</script>',
      },
      fileParams: {},
      timestamp: 'any',
    });
    mockDomWithFileInput(mockFileContent);

    const actionArgMatcher = new mockmatchers.ArgumentMatcher((actionArg) => {
      assertEquals(Dispatcher.ActionType.IMPORT_STORE, actionArg.actionType);
      assertNotContains('script', JSON.stringify(actionArg.data));
      return true;
    });

    const mockDispatcher = mockControl.createStrictMock(Dispatcher);
    mockDispatcher.sendAction(actionArgMatcher).$once();
    const mockGetInstance = mockControl.createMethodMock(Dispatcher,
        'getInstance');
    mockGetInstance().$returns(mockDispatcher);

    mockControl.$replayAll();

    await uploader.upload();

    mockControl.$verifyAll();
  },


  tearDown() {
    mockControl.$tearDown();
  },

});
