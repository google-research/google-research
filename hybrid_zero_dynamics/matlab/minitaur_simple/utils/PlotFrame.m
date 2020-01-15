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


function PlotFrame(pos, eulerAngles)


r = eulerAngles(1);
p = eulerAngles(2);
y = eulerAngles(3);

lineSize = 0.1;
lineWidth = 2;

Rotm = eul2rotm([r, p, y], 'XYZ');

x = lineSize * Rotm(:, 1);
y = lineSize * Rotm(:, 2);
z = lineSize * Rotm(:, 3);

px = pos(1);
py = pos(2);
pz = pos(3);

colors = ['r'; 'g'; 'b'];

quiver3(px, py, pz, x(1), x(2), x(3), 'color', colors(1, :), ...
  'LineWidth', lineWidth)
hold on
quiver3(px, py, pz, y(1), y(2), y(3), 'color', colors(2, :), ...
  'LineWidth', lineWidth)
quiver3(px, py, pz, z(1), z(2), z(3), 'color', colors(3, :), ...
  'LineWidth', lineWidth)
axis equal