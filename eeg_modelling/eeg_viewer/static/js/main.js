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

goog.module('eeg_modelling.eeg_viewer.views');

const Annotations = goog.require('eeg_modelling.eeg_viewer.Annotations');
const ChartController = goog.require('eeg_modelling.eeg_viewer.ChartController');
const Console = goog.require('goog.debug.Console');
const Downloader = goog.require('eeg_modelling.eeg_viewer.Downloader');
const Error = goog.require('eeg_modelling.eeg_viewer.Error');
const Graph = goog.require('eeg_modelling.eeg_viewer.Graph');
const Menus = goog.require('eeg_modelling.eeg_viewer.Menus');
const NavChart = goog.require('eeg_modelling.eeg_viewer.NavChart');
const Predictions = goog.require('eeg_modelling.eeg_viewer.Predictions');
const Requests = goog.require('eeg_modelling.eeg_viewer.Requests');
const ToolBar = goog.require('eeg_modelling.eeg_viewer.ToolBar');
const Uploader = goog.require('eeg_modelling.eeg_viewer.Uploader');
const WaveEventForm = goog.require('eeg_modelling.eeg_viewer.WaveEventForm');
const WaveEvents = goog.require('eeg_modelling.eeg_viewer.WaveEvents');
const WindowLocation = goog.require('eeg_modelling.eeg_viewer.WindowLocation');

/** @const {!Console} */
const console = new Console();
console.setCapturing(true);

Annotations.getInstance();
ChartController.getInstance();
Downloader.getInstance();
Error.getInstance();

/** @const {!Graph} */
const graph = Graph.getInstance();
goog.exportSymbol(
    'graph.closeSensitivityMenu',
    goog.bind(graph.closeSensitivityMenu, graph));
goog.exportSymbol(
    'graph.increaseSensitivity',
    goog.bind(graph.increaseSensitivity, graph));
goog.exportSymbol(
    'graph.decreaseSensitivity',
    goog.bind(graph.decreaseSensitivity, graph));
goog.exportSymbol(
    'graph.allowDrop',
    goog.bind(graph.allowDrop, graph));
goog.exportSymbol(
    'graph.drop',
    goog.bind(graph.drop, graph));

/** @const {!Menus} */
const menus = Menus.getInstance();
goog.exportSymbol('menus.loadFile', goog.bind(menus.loadFile, menus));
goog.exportSymbol(
    'menus.toggleFileInfo', goog.bind(menus.toggleFileInfo, menus));
goog.exportSymbol('menus.toggleMenu', goog.bind(menus.handleMenuToggle, menus));
goog.exportSymbol(
    'menus.selectFileType', goog.bind(menus.selectFileType, menus));

NavChart.getInstance();

/** @const {!Predictions} */
const predictions = Predictions.getInstance();
goog.exportSymbol(
    'predictions.selectPredictionMode',
    goog.bind(predictions.handlePredictionModeSelection, predictions));
goog.exportSymbol(
    'predictions.toggleFilter',
    goog.bind(predictions.toggleFilter, predictions));
goog.exportSymbol(
    'predictions.selectPredictionLabel',
    goog.bind(predictions.handlePredictionLabelSelection, predictions));

Requests.getInstance();

/** @const {!ToolBar} */
const toolBar = ToolBar.getInstance();
goog.exportSymbol(
    'toolBar.toggleMenu', goog.bind(toolBar.toggleMenu, toolBar));
goog.exportSymbol(
    'toolBar.selectDropdown', goog.bind(toolBar.selectDropdown, toolBar));
goog.exportSymbol('toolBar.nextChunk', goog.bind(toolBar.nextChunk, toolBar));
goog.exportSymbol('toolBar.nextSec', goog.bind(toolBar.nextSec, toolBar));
goog.exportSymbol('toolBar.prevChunk', goog.bind(toolBar.prevChunk, toolBar));
goog.exportSymbol('toolBar.prevSec', goog.bind(toolBar.prevSec, toolBar));

/** @const {!Uploader} */
const uploader = Uploader.getInstance();
goog.exportSymbol(
    'uploader.closeMenu', goog.bind(uploader.closeMenu, uploader));
goog.exportSymbol(
    'uploader.openMenu', goog.bind(uploader.openMenu, uploader));
goog.exportSymbol(
    'uploader.handleFileChange',
    goog.bind(uploader.handleFileChange, uploader));
goog.exportSymbol(
    'uploader.upload', goog.bind(uploader.upload, uploader));

/** @const {!WindowLocation} */
const windowLocation = WindowLocation.getInstance();

/** @const {!WaveEventForm} */
const waveEventForm = WaveEventForm.getInstance();
goog.exportSymbol(
    'waveEventForm.selectType',
    goog.bind(waveEventForm.selectType, waveEventForm));
goog.exportSymbol(
    'waveEventForm.close',
    goog.bind(waveEventForm.close, waveEventForm));
goog.exportSymbol(
    'waveEventForm.save',
    goog.bind(waveEventForm.save, waveEventForm));
goog.exportSymbol(
    'waveEventForm.toggleAllChannels',
    goog.bind(waveEventForm.toggleAllChannels, waveEventForm));
goog.exportSymbol(
    'waveEventForm.clickInput',
    goog.bind(waveEventForm.clickInput, waveEventForm));
goog.exportSymbol(
    'waveEventForm.drag',
    goog.bind(waveEventForm.drag, waveEventForm));

/** @const {!WaveEvents} */
const waveEvents = WaveEvents.getInstance();
goog.exportSymbol(
    'waveEvents.closeWaveEventMenu',
    goog.bind(waveEvents.closeWaveEventMenu, waveEvents));
goog.exportSymbol(
    'waveEvents.deleteWaveEvent',
    goog.bind(waveEvents.deleteWaveEvent, waveEvents));
goog.exportSymbol(
    'waveEvents.navigateToWaveEvent',
    goog.bind(waveEvents.navigateToWaveEvent, waveEvents));
goog.exportSymbol(
    'waveEvents.searchSimilarPatterns',
    goog.bind(waveEvents.searchSimilarPatterns, waveEvents));
goog.exportSymbol(
    'waveEvents.clearCurve', goog.bind(waveEvents.clearCurve, waveEvents));
goog.exportSymbol(
    'waveEvents.getSimilarityCurve',
    goog.bind(waveEvents.getSimilarityCurve, waveEvents));
goog.exportSymbol(
    'waveEvents.searchAndGetCurve',
    goog.bind(waveEvents.searchAndGetCurve, waveEvents));
goog.exportSymbol(
    'waveEvents.acceptSimilarPattern',
    goog.bind(waveEvents.acceptSimilarPattern, waveEvents));
goog.exportSymbol(
    'waveEvents.editSimilarPattern',
    goog.bind(waveEvents.editSimilarPattern, waveEvents));
goog.exportSymbol(
    'waveEvents.closeSimilarPatternMenu',
    goog.bind(waveEvents.closeSimilarPatternMenu, waveEvents));
goog.exportSymbol(
    'waveEvents.navigateToPattern',
    goog.bind(waveEvents.navigateToPattern, waveEvents));
goog.exportSymbol(
    'waveEvents.rejectSimilarPattern',
    goog.bind(waveEvents.rejectSimilarPattern, waveEvents));
goog.exportSymbol(
    'waveEvents.downloadEvents',
    goog.bind(waveEvents.downloadEvents, waveEvents));
goog.exportSymbol(
    'waveEvents.searchMore',
    goog.bind(waveEvents.searchMore, waveEvents));
goog.exportSymbol(
    'waveEvents.rejectAll',
    goog.bind(waveEvents.rejectAll, waveEvents));
goog.exportSymbol(
    'waveEvents.toggleSimilaritySettings',
    goog.bind(waveEvents.toggleSimilaritySettings, waveEvents));
goog.exportSymbol(
    'waveEvents.saveSimilaritySettings',
    goog.bind(waveEvents.saveSimilaritySettings, waveEvents));

/**
 * Make a request to the server when the URL changes.
 */
window.onhashchange = () => windowLocation.makeDataRequest();

windowLocation.makeDataRequest();
