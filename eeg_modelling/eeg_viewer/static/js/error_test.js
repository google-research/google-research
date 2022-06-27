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

goog.module('eeg_modelling.eeg_viewer.Error.tests');
goog.setTestOnly();

const Error = goog.require('eeg_modelling.eeg_viewer.Error');
const MockControl = goog.require('goog.testing.MockControl');
const mockmatchers = goog.require('goog.testing.mockmatchers');
const testSuite = goog.require('goog.testing.testSuite');

let mockControl;
let error;
const _ = mockmatchers.ignoreArgument;

const mockMaterialSnackbar = {
  showSnackbar: () => {},
};

testSuite({

  setUp() {
    mockControl = new MockControl();
    error = Error.getInstance();

    const notificationSnackbar = document.createElement('div');
    notificationSnackbar.id = 'notification-snackbar';
    notificationSnackbar.MaterialSnackbar = mockMaterialSnackbar;

    document.body.append(notificationSnackbar);
  },

  testHandleError() {
    const storeData = {
      error: {
        message: 'Test error',
        timestamp: 1,
      },
    };

    const mockShowSnackbar = mockControl.createMethodMock(mockMaterialSnackbar, 'showSnackbar');

    const snackbarArgMatcher = new mockmatchers.ArgumentMatcher((arg) =>
      arg.message === 'Test error' && arg.timeout >= 3000);

    mockShowSnackbar(snackbarArgMatcher).$once();

    mockControl.$replayAll();
    error.handleError(storeData);
    mockControl.$verifyAll();
  },

  tearDown() {
    mockControl.$tearDown();
  },

});
