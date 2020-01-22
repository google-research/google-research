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

goog.module('eeg_modelling.eeg_viewer.Dispatcher.tests');
goog.setTestOnly();

const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const MockControl = goog.require('goog.testing.MockControl');
const testSuite = goog.require('goog.testing.testSuite');

let mockControl;
let dispatcher;

let testingActionType = Dispatcher.ActionType.ANNOTATION_SELECTION;


testSuite({

  setUp() {
    mockControl = new MockControl();
    dispatcher = Dispatcher.getInstance();
  },

  testSendAction() {
    const testData = 'testData';

    const mockCallback = mockControl.createFunctionMock();
    mockCallback(testData).$once();
    mockControl.$replayAll();
    dispatcher.registerCallback(testingActionType, mockCallback);

    const actionEvent = {
      actionType: testingActionType,
      data: testData,
    };
    dispatcher.sendAction(actionEvent);
    mockControl.$verifyAll();
  },

  tearDown() {
    mockControl.$tearDown();
  },

});
