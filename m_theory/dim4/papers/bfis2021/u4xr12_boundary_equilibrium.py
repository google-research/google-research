# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""An almost-equilibrium for the U(4)|xR12 gauging.

This solution was found via:

  python3 -i -m dim4.papers.bfis2021.so3n1_on_sl2x7 scan_u4xr12

...using an increased number of samples. This showed up as...:

1715,-30.116318500106615,3.939412900384883e-21,-0.1598880237012632,...

Despite stationarity-violation being essentially-zero, there are
additional flat directions, and apparently, this solution sits in a
very flat basin (perhaps even truly flat) of the
stationarity-violation. Minimizing the stationarity-violation with
float64 numerical accuracy is not good enough to accurately determine
the location and properties of this solution (unless this is a
manifold of solutions with different physics). This shows in other
discoveries of this critical point having similar but nevertheless
substantially different gravitino masses: There always is a
naive-mass^2 = 1.0 gravitino, indicating unbroken N=1 supersymmetry,
as well as one gravitino with mass^2 = 41/9 and two with mass^2 = 4,
but the other masses are noisy throughout, and no two discoveries have
the same mass there. This indicates that we would need higher
numerical accuracy to actually pinpoint the solution.

However, given that we know the expected target mass spectrum from analytic
omega-continuation, which should be 41/9x3 + 4x4 + 1x1, we can try to refine
this solution to another one that has precisely this gravitino mass spectrum and
also satisfies the stationarity condition |grad V| = 0 to numerical accuracy.

Doing this then shows that the entire mass spectrum nicely aligns with the one
we observe for the limit boundary gauging.

"""

v70 = (
    -0.1598880237012632, -0.21633850549854772, -0.12846683080095253,
    -0.14855391665395948, -0.037508476890801015, 0.8618378686129694,
    -0.04853769215375507, 0.063089843286208, 0.015301975685707061,
    0.07070544634080797, 0.06706842631384814, 0.0495557045381229,
    0.003518210373335068, 0.018732785611496584, 0.00555199085738167,
    0.005157266782825642, -0.04502054250920104, 0.13503877230797373,
    -0.045496829074317534, 0.020209321201322725, 0.04345326532600796,
    0.061392888242964475, -0.18004549057558195, -0.04599593105047841,
    0.007152278135800403, 0.17146523200818917, -0.31959795567868193,
    -0.08712718522948705, -0.11974358480824067, -0.31321041903440644,
    -0.14702936384068166, 0.2521793506838656, 0.34123583177874345,
    0.13585190495504634, -0.2463082064361732, 0.10004082269633535,
    0.04371508977825264, -0.0011170953749372232, -0.08286813693177862,
    -0.07114735006336236, 0.06501396798364605, 0.050811133724784276,
    -0.1311017972966881, -0.12127402408856404, -0.03870601951274827,
    0.040578586141946094, -0.010710608593345444, -0.019657994203529625,
    -0.00773766258136694, -0.028691625114436865, -0.0711195567839736,
    -0.10935744629619007, 0.023251763775933576, 0.11656943522459028,
    -0.053208450577868126, 0.08478164436205092, 0.052098264565329795,
    0.05715504507456131, -0.12139357691021292, 0.059699770374204104,
    -0.031277812444966614, -0.07654178579418122, -0.06829505891209509,
    0.08000808393919134, 0.10426776261784464, -0.02670869848484579,
    -0.004126450187771894, -0.005529280276502518, -0.045643339638804514,
    -0.05363654041986461)
