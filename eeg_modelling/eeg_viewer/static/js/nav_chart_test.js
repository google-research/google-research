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

goog.module('eeg_modelling.eeg_viewer.NavChart.tests');
goog.setTestOnly();

const Dispatcher = goog.require('eeg_modelling.eeg_viewer.Dispatcher');
const MockControl = goog.require('goog.testing.MockControl');
const NavChart = goog.require('eeg_modelling.eeg_viewer.NavChart');
const dom = goog.require('goog.dom');
const gvizEvents = goog.require('google.visualization.events');
const mockmatchers = goog.require('goog.testing.mockmatchers');
const singleton = goog.require('goog.testing.singleton');
const testSuite = goog.require('goog.testing.testSuite');

let mockControl;

let navChart;
let storeData;
let navContainer;


testSuite({

  setUp() {
    mockControl = new MockControl();

    singleton.resetAll();
    navChart = NavChart.getInstance();

    storeData = {
      absStart: 2,
      chunkStart: 4,
      chunkDuration: 3,
      chunkGraphData: {
        cols: [0, 1, 1, 2, 3, 5],
      },
      numSecs: 10,
      predictionMode: 'None',
      samplingFreq: 1,
      seriesHeight: 42,
      waveEvents: [],
    };

    const parentContainer = document.createElement('div');
    parentContainer.id = navChart.parentId;
    navContainer = document.createElement('div');
    navContainer.id = navChart.containerId;
    navContainer.style.position = 'absolute';
    navContainer.style.top = '5px';
    navContainer.style.left = '11px';
    const overlay = document.createElement('canvas');
    overlay.id = navChart.overlayId;
    parentContainer.appendChild(navContainer);
    parentContainer.appendChild(overlay);
    document.body.appendChild(parentContainer);
  },

  testGetHTickValues() {
    mockControl.$replayAll();

    assertArrayEquals([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        navChart.getHTickValues(storeData));

    mockControl.$verifyAll();
  },

  testUpdateChartOptions() {
    const mockGetParent = mockControl.createMethodMock(navChart, 'getParent');
    mockGetParent().$returns({clientWidth: 500});

    const mockNumSecs = mockControl.createMethodMock(navChart, 'getNumSecs');
    mockNumSecs(storeData).$returns(10).$times(2);
    const mockStart = mockControl.createMethodMock(navChart, 'getStart');
    mockStart(storeData).$returns(0).$times(12);

    const mockSet = mockControl.createMethodMock(navChart, 'setOption');
    mockSet('width', 500);
    mockSet('hAxis.viewWindow', {min: 0, max: 10}).$once();
    mockSet('hAxis.ticks',
        [{v: 0, f: '00:00:02'}, {v: 1, f: '00:00:03'},
         {v: 2, f: '00:00:04'}, {v: 3, f: '00:00:05'},
         {v: 4, f: '00:00:06'}, {v: 5, f: '00:00:07'},
         {v: 6, f: '00:00:08'}, {v: 7, f: '00:00:09'},
         {v: 8, f: '00:00:10'}, {v: 9, f: '00:00:11'}]).$once();
    mockSet('colors', ['transparent']).$once();
    mockControl.$replayAll();

    navChart.updateChartOptions(storeData);

    mockControl.$verifyAll();
  },

  testHandleChartData_highlightsViewport() {
    const numMatcher = new mockmatchers.ArgumentMatcher(
        (num) => typeof num === 'number' && num >= 0);

    const mockRect = mockControl.createFunctionMock('fillRect');
    const mockClearRect = mockControl.createFunctionMock('clearRect');
    const mockGetContext = mockControl.createMethodMock(navChart,
        'getContext');
    mockGetContext().$returns({
      setLineDash: () => null,
      fillRect: mockRect,
      clearRect: mockClearRect,
    });
    mockClearRect(numMatcher, numMatcher, numMatcher, numMatcher).$once();
    mockRect(numMatcher, numMatcher, numMatcher, numMatcher).$once();

    mockControl.$replayAll();

    navChart.handleChartData(storeData, ['numSecs', 'chunkStart']);

    mockControl.$verifyAll();
  },

  testChartListeners() {
    const mockCli = mockControl.createMethodMock(navChart,
        'getChartLayoutInterface');
    mockCli().$returns({
      getHAxisValue: (x) => x,
      getChartAreaBoundingBox: () => ({top: 1, left: 3, height: 3, width: 5}),
    });

    const mockDispatcher = mockControl.createStrictMock(Dispatcher);
    mockDispatcher.sendAction({
      actionType: Dispatcher.ActionType.NAV_BAR_CHUNK_REQUEST,
      data: {
        time: 6.4,
      }
    }).$once();
    const mockGetInstance = mockControl.createMethodMock(Dispatcher,
        'getInstance');
    mockGetInstance().$returns(mockDispatcher);

    mockControl.$replayAll();

    navChart.initChart();
    navChart.addChartEventListeners();
    gvizEvents.trigger(navChart.getChart(), 'click', {
      targetID: 'point#1#1',
      x: 6.4,
      y: 0,
    });

    mockControl.$verifyAll();
  },

  tearDown() {
    dom.removeChildren(dom.getDocument().body);
    mockControl.$tearDown();
  },
});
