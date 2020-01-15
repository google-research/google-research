% Copyright 2019 Google LLC
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

function rad = ext_cm2rad(cm_vec)
off = 0.0240;
l1 = 0.1036;
l2 = 0.216;
foot_pos = l1 + l2 + off;

if all(size(cm_vec)-[1, 1]) > 0
  error('Input must be a scalar or vector, NOT a matrix');
end

rad = zeros(size(cm_vec));

for i = 1:length(rad)
  cm = cm_vec(i);
  cm = cm + foot_pos;
  cm = cm - off;
  rad(i) = acos((l2^2 - l1^2 - cm^2)/(2 * l1 * cm));
end

end