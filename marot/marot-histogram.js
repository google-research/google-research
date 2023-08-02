// Copyright 2023 The Google Research Authors.
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
 * This file contains the JavaScript code for MarotHistogram, a component of
 * Marot that can show a segment-wise histogram of scores or score differences.
 */

/**
 * Helper class for building a segment scores histogram. Call addSegment() on it
 * multiple times to record segment scores. Then call display().
 *
 * In most cases, the rendered histogram treats X = 0 differently, showing a
 * slim gray bar for that exact value. This is always done for difference
 * histograms. For non-difference histograms, this is only done for metrics for
 * which 0 means a perfect score (such as MQM).
 */
class MarotHistogram {
  /**
   * @param {number} m The index of the metric in marot.metrics.
   * @param {string} sys The name of the system.
   * @param {string} color The color to use for system.
   * @param {string=} sysCmp if the histogram is for diffs, then the name of
   *     the system being compared against.
   * @param {string=} colorCmp if the histogram is for diffs, then the color of
   *     the system being compared against.
   */
   constructor(m, sys, color, sysCmp='', colorCmp='') {
    this.metricIndex = m;
    this.sys = sys;
    this.sysCmp = sysCmp;
    this.hasDiffs = sysCmp ? true : false;
    this.metric = marot.metrics[m];
    this.color = color;
    this.colorCmp = colorCmp;

    const metricInfo = marot.metricsInfo[this.metric];
    this.lowerBetter = metricInfo.lowerBetter || false;

    /**
     * Is there a dedicated bin for value == 0?
     */
    this.hasZeroBin = this.hasDiffs ? true : (this.lowerBetter ? true : false);

    /** @const {number} Width of a histogram bin, in score units */
    this.BIN_WIDTH = (this.metric == 'MQM') ? 0.5 : 0.05;
    this.BIN_PRECISION = (this.metric == 'MQM') ? 1 : 2;

    /** @const {number} Width of a histogram bin, in pixels */
    this.BIN_WIDTH_PIXELS = 10 + (this.BIN_PRECISION * 3);

    this.PIXELS_PER_UNIT = this.BIN_WIDTH_PIXELS / this.BIN_WIDTH;

    /** @const {number} Width of the special "zero" bin, in pixels */
    this.ZERO_BIN_WIDTH_PIXELS = 6;

    this.LOG_MULTIPLIER = 1.0 / Math.LN2;
    this.LOG_UNIT_HEIGHT_PIXELS = 25;
    this.TOP_OFFSET_PIXELS = this.hasDiffs ? 49 : 19;
    this.BOTTOM_OFFSET_PIXELS = 50;
    this.X_OFFSET_PIXELS = 50;

    this.COLOR_ZERO = 'rgba(211,211,211,0.5)';
    this.COLOR_OUTLINE = 'black';
    this.COLOR_LEGEND = 'black';
    this.COLOR_LABELS = 'black';
    this.COLOR_LINES = 'lightgray';

    /**
     * @const {!Object} Dict keyed by bin. Each bin has an array of doc-seg keys.
     *    The only non-numeric key possibly present is 'zero' (when
     *    this.hasZeroBin is true).
     */
    this.segsInBin = {};

    /**
     * @const {number} The largest bin visible on the X-axis.
     */
    this.maxBin = 0;
    /**
     * @const {number} The smallest bin visible on the X-axis.
     */
    this.minBin = 0;

    /** {number} The largest count in bin (used to determine height of plot) */
    this.maxCount = 8;

    this.totalCount = 0;
    this.sys1BetterCount = 0;
    this.sys2BetterCount = 0;
  }

  /**
   * Returns the bin for a particular value. We return the left end-point (except
   * for the special 'zero' bin).
   * @param {number} value
   * @return {string}
   */
  binOf(value) {
    if (this.hasZeroBin && value == 0) {
      return 'zero';
    }
    const absValue = Math.abs(value);
    const absBin = Math.floor(absValue / this.BIN_WIDTH) * this.BIN_WIDTH;
    const leftVal = (value < 0) ? (0 - absBin - this.BIN_WIDTH) : absBin;
    return leftVal.toFixed(this.BIN_PRECISION);
  }

  /**
   * Adds a segment to the histogram, updating the appropriate bin.
   * @param {string} doc
   * @param {string|number} docSegId
   * @param {number} value The score for the first system
   */
  addSegment(doc, docSegId, value) {
    const bin = this.binOf(value);
    const numericBin = (bin == 'zero') ? 0 : parseFloat(bin);
    if (numericBin < this.minBin) this.minBin = numericBin;
    if (numericBin > this.maxBin) this.maxBin = numericBin;
    const docsegKey = marot.docsegKey(doc, docSegId);
    if (!this.segsInBin.hasOwnProperty(bin)) {
      this.segsInBin[bin] = [];
    }
    this.segsInBin[bin].push(docsegKey);
    if (this.segsInBin[bin].length > this.maxCount) {
      this.maxCount = this.segsInBin[bin].length;
    }
    this.totalCount++;
    if (this.hasDiffs && bin != 'zero') {
      const firstLower = (numericBin < 0);
      const firstBetter = (firstLower && this.lowerBetter) ||
                          (!firstLower && !this.lowerBetter);
      if (firstBetter) {
        this.sys1BetterCount++;
      } else {
        this.sys2BetterCount++;
      }
    }
  }

  /**
   * Creates and returns an SVG rect.
   * @param {number} x
   * @param {number} y
   * @param {number} w
   * @param {number} h
   * @param {string} color
   * @return {!Element}
   */
  getRect(x, y, w, h, color) {
    const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    rect.setAttributeNS(null, 'x', x);
    rect.setAttributeNS(null, 'y', y + this.TOP_OFFSET_PIXELS);
    rect.setAttributeNS(null, 'width', w);
    rect.setAttributeNS(null, 'height', h);
    rect.style.fill = color;
    rect.style.stroke = this.COLOR_OUTLINE;
    return rect;
  };

  /**
   * Creates a histogram bar with a given description, makes it clickable to
   * constrain the view to the docsegs passed.
   * @param {!Element} plot
   * @param {number} x
   * @param {number} y
   * @param {number} w
   * @param {number} h
   * @param {string} color
   * @param {string} desc
   * @param {!Array<string>} docsegs
   */
  makeHistBar(plot, x, y, w, h, color, desc, docsegs) {
    /**
     * Need to wrap the rect in a g (group) element to be able to show
     * the description when hovering ("title" attribute does not work with SVG
     * elements).
     */
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttributeNS(null, 'class', 'marot-sys-v-sys-hist');
    g.insertAdjacentHTML('beforeend',
                         `<title>Click to see examples of ${desc}</title>`);
    const rect = this.getRect(x, y, w, h, color);
    g.appendChild(rect);
    const viewingConstraints = {};
    for (let ds of docsegs) {
      viewingConstraints[ds] = true;
    }
    viewingConstraints.description = desc;
    viewingConstraints.color = color;
    g.addEventListener('click', (e) => {
      marot.show(viewingConstraints);
    });
    plot.appendChild(g);
  }

  /**
   * Creates a line on the plot.
   * @param {!Element} plot
   * @param {number} x1
   * @param {number} y1
   * @param {number} x2
   * @param {number} y2
   * @param {string} color
   */
  makeLine(plot, x1, y1, x2, y2, color) {
    const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    line.setAttributeNS(null, 'x1', x1);
    line.setAttributeNS(null, 'y1', y1 + this.TOP_OFFSET_PIXELS);
    line.setAttributeNS(null, 'x2', x2);
    line.setAttributeNS(null, 'y2', y2 + this.TOP_OFFSET_PIXELS);
    line.style.stroke = color;
    plot.appendChild(line);
  }

  /**
   * Writes some text on the plot.
   * @param {!Element} plot
   * @param {number} x
   * @param {number} y
   * @param {string} s
   * @param {string} color
   */
  makeText(plot, x, y, s, color) {
    const text = document.createElementNS('http://www.w3.org/2000/svg', 'text');
    text.setAttributeNS(null, 'x', x);
    text.setAttributeNS(null, 'y', y + this.TOP_OFFSET_PIXELS);
    text.innerHTML = s;
    text.style.fill = color;
    text.style.fontSize = '10px';
    plot.appendChild(text);
  }

  /**
   * Returns height in pixels for a histogram bar with the given count.
   * @param {number} count
   * @return {number}
   */
  heightInPixels(count) {
    if (count == 0) return 0;
    return this.LOG_UNIT_HEIGHT_PIXELS *
           ((Math.log(count) * this.LOG_MULTIPLIER) + 1);
  }

  /**
   * Returns the color to use for the bin's histogram rectangle.
   * @param {string} bin
   * @param {number} numericBin
   * @return {string}
   */
  binColor(bin, numericBin) {
    if (bin == 'zero') {
      return this.COLOR_ZERO;
    }
    if (!this.hasDiffs) {
      return this.color;
    } else {
      const firstLower = (numericBin < 0);
      const firstBetter = (firstLower && this.lowerBetter) ||
                          (!firstLower && !this.lowerBetter);
      return firstBetter ? this.color : this.colorCmp;
    }
  }

  /**
   * Returns a description of the bin.
   * @param {string} bin
   * @param {number} numericBin
   * @param {number} count
   * @return {string}
   */
  binDesc(bin, numericBin, count) {
    if (!this.hasDiffs) {
      if (bin == 'zero') {
        return '' + count + ' segment(s) where ' + this.sys +
               ' has ' + this.metric + ' score exactly equal to 0';
      }
      const binLeft = numericBin;
      const binRight = numericBin + this.BIN_WIDTH;
      let leftParen = (numericBin < 0) ? '(' :
                      ((numericBin == 0 && this.hasZeroBin) ? '(' : '[');
      let rightParen = (numericBin < 0) ?
                       ((binRight == 0 && this.hasZeroBin) ? ')' : ']') : ')';
      return '' + count + ' segment(s) where ' + this.sys + ' has ' +
             this.metric + ' score in ' + 'range ' + leftParen +
             this.binDisplay(binLeft) +
             ',' + this.binDisplay(binRight) + rightParen;
    } else {
      if (bin == 'zero') {
        return '' + count + ' segment(s) where ' + this.sys + ' and ' +
               this.sysCmp + ' have identical ' + this.metric + ' scores';
      }
      const firstLower = (numericBin < 0);
      const firstBetter = (firstLower && this.lowerBetter) ||
                          (!firstLower && !this.lowerBetter);
      const betterSys = firstBetter ? this.sys : this.sysCmp;
      const worseSys = firstBetter ? this.sysCmp : this.sys;
      const binLeft = numericBin;
      const binRight = numericBin + this.BIN_WIDTH;
      const absBinLeft = (numericBin < 0) ? (0 - binRight) : binLeft;
      const absBinRight = absBinLeft + this.BIN_WIDTH;
      const firstParen = (absBinLeft == 0 && this.hasZeroBin) ? '(' : '[';
      return '' + count + ' segment(s) where ' + betterSys + ' is better than ' +
             worseSys + ' with ' + this.metric + ' score diff in range ' +
             firstParen + this.binDisplay(absBinLeft) + ',' +
             this.binDisplay(absBinRight) + ')';
    }
  }

  /**
   * Returns the x coordinate in pixels for a particular metric value.
   * @param {number} value
   * @return {number}
   */
  xPixels(value) {
    return this.X_OFFSET_PIXELS + ((value - this.minBin) * this.PIXELS_PER_UNIT);
  }

  /**
   * Returns a string suitable to display, for a floating-point number. Strips
   * trailing zeros and then a trailing decimal point.
   * @param {number} value
   * @return {string}
   */
  binDisplay(value) {
    return value.toFixed(
        this.BIN_PRECISION).replace(/0+$/, '').replace(/\.$/, '');
  }

  /**
   * Displays the histogram using the data collected through prior addSegment()
   * calls.
   * @param {!Element} plot
   */
  display(plot) {
    /** Create some buffer space above the plot. */
    this.maxCount += 10;

    const binKeys = Object.keys(this.segsInBin);
    /** Sort so that 'zero' bin is drawn at the end. */
    binKeys.sort((a, b) => {
      let a2 = (a == 'zero') ? Number.MAX_VALUE : a;
      let b2 = (b == 'zero') ? Number.MAX_VALUE : b;
      return a2 - b2;
    });
    const plotWidth = Math.max(
        400, (2 * this.X_OFFSET_PIXELS) +
             ((this.maxBin - this.minBin) * this.PIXELS_PER_UNIT));
    const plotHeight = this.heightInPixels(this.maxCount);
    const svgWidth = plotWidth;
    const svgHeight = plotHeight +
                      (this.TOP_OFFSET_PIXELS + this.BOTTOM_OFFSET_PIXELS);
    plot.innerHTML = '';
    plot.setAttributeNS(null, 'viewBox', `0 0 ${svgWidth} ${svgHeight}`);
    plot.setAttributeNS(null, 'width', svgWidth);
    plot.setAttributeNS(null, 'height', svgHeight);

    /* y axis labels */
    this.makeLine(plot, 0, plotHeight, plotWidth, plotHeight, this.COLOR_LINES);
    this.makeText(plot, 5, plotHeight - 2, '0', this.COLOR_LABELS);
    for (let l = 1; l <= this.maxCount; l *= 2) {
      const h = this.heightInPixels(l);
      this.makeLine(plot, 0, plotHeight - h, plotWidth, plotHeight - h,
                    this.COLOR_LINES);
      this.makeText(plot, 5, plotHeight - h - 2, '' + l, this.COLOR_LABELS);
    }

    if (this.hasDiffs) {
      /* legend, shown in the area above the plot */
      const legends = [
        {
          color: this.color,
          desc: this.sys1BetterCount + ' better segments for ' + this.sys,
        },
        {
          color: this.colorCmp,
          desc: this.sys2BetterCount + ' better segments for ' + this.sysCmp,
        },
      ];
      for (let s = 0; s < legends.length; s++) {
        const legend = legends[s];
        const y = -30 + (s * (this.BIN_WIDTH_PIXELS + 10));
        const x = 25;
        plot.appendChild(this.getRect(
            x, y, this.BIN_WIDTH_PIXELS, this.BIN_WIDTH_PIXELS, legend.color));
        this.makeText(plot, x + this.BIN_WIDTH_PIXELS + 5, y + 10,
                      legend.desc, this.COLOR_LEGEND);
      }
    }

    for (let bin of binKeys) {
      const segs = this.segsInBin[bin];
      if (segs.length == 0) continue;
      const numericBin = (bin == 'zero') ? 0 : parseFloat(bin);
      let x = this.xPixels(numericBin);
      const binWidth = (bin == 'zero') ? this.ZERO_BIN_WIDTH_PIXELS :
                       this.BIN_WIDTH_PIXELS;
      if (bin == 'zero') {
        x -= (binWidth / 2.0);
      }
      const color = this.binColor(bin, numericBin);
      const desc = this.binDesc(bin, numericBin, segs.length);
      const h = this.heightInPixels(segs.length);
      this.makeHistBar(
            plot, x, plotHeight - h, binWidth, h,
            color, desc, segs);
    }

    /** Draw x-axis labels */
    const maxV = Math.max(Math.abs(this.minBin), Math.abs(this.maxBin));
    const step = 2 * this.BIN_WIDTH;
    for (let v = 0; v <= maxV + this.BIN_WIDTH; v += step) {
      if (v >= 0 && v <= this.maxBin + this.BIN_WIDTH) {
        const vDisp = this.binDisplay(v);
        const x = this.xPixels(v);
        const xDelta = 3 * vDisp.length;
        this.makeLine(plot, x, plotHeight, x, plotHeight + 8, this.COLOR_LINES);
        this.makeText(plot, x - xDelta, plotHeight + 20,
                      vDisp, this.COLOR_LABELS);
      }
      const negV = 0 - v;
      if (v == 0 || negV < this.minBin) {
        continue;
      }
      const negVDisp = this.binDisplay(negV);
      const x = this.xPixels(negV);
      const xDelta = 3 * (negVDisp.length + 1);
      this.makeLine(plot, x, plotHeight, x, plotHeight + 8, this.COLOR_LINES);
      this.makeText(plot, x - xDelta, plotHeight + 20,
                    negVDisp, this.COLOR_LABELS);
    }
    /* X-axis name */
    this.makeText(plot, this.X_OFFSET_PIXELS, plotHeight + 40,
                  (this.hasDiffs ? this.metric + ' score differences' :
                   this.sys + ': ' + this.totalCount + ' segments with ' +
                   this.metric + ' scores'),
                  this.COLOR_LEGEND);
  }
}