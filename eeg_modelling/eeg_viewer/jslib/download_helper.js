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

/**
 * @fileoverview Provide a class to handle file downloads in the browser.
 */

goog.module('jslib.DownloadHelper');

class DownloadHelper {
  /**
   * Downloads string content to a file.
   * @param {string} data Data to download as string.
   * @param {string} filename Name of the local file to download.
   * @param {string} contentType Content type.
   */
  static download(data, filename, contentType) {
    const blob = new Blob([data], { type: contentType });

    const anchor = document.createElement('a');
    anchor.style.display = 'none';
    document.body.appendChild(anchor);

    const url = URL.createObjectURL(blob);
    anchor.href = url;
    anchor.download = filename;
    anchor.click();

    window.URL.revokeObjectURL(url);
    document.body.removeChild(anchor);
  }
}

exports = DownloadHelper;
