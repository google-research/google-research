% Copyright 2019 Google LLC % Copyright 2019 Google LLC
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%     https://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

function TrotImpact2Constraints(nlp, src, tar, bounds, varargin)

plant = nlp.Plant;

% first call the class method
plant.rigidImpactConstraint(nlp, src, tar, bounds, varargin{:});

% Don't need time continuity constraint
removeConstraint(nlp, 'tContDomain');

% the relabeling of joint coordinate is no longer valid
removeConstraint(nlp, 'xDiscreteMapTrotImpact2');

% the configuration only depends on the relabeling matrix
R = plant.R;
x = plant.States.x;
xn = plant.States.xn;
x_diff = R * x - xn;
x_map = SymFunction(['xDiscreteMap', plant.Name], x_diff(3:end), {x, xn});

% Need to add leg synchronization constraint

addNodeConstraint(nlp, x_map, {'x', 'xn'}, 'first', 0, 0, 'NonLinear');

end