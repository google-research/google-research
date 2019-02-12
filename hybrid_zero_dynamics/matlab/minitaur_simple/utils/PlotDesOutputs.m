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

function PlotDesOutputs(model, nlp, data, indices)

[M, N] = size(nlp.Phase(1).Plant.Params.atime);

if nargin < 4
  indices = 1:length(model.Joints);
else
  if isempty(indices), return;
  end
end

t = [];
y_des = [];
dy_des = [];
ddy_des = [];
y_act = [];
dy_act = [];
ddy_act = [];


if isa(data, 'struct')
  gait = data;
  cont_domain_idx = find(cellfun(@(x)isa(x, 'ContinuousDynamics'), ...
    {nlp.Phase.Plant}));

  for j = cont_domain_idx

    if j == 1
      t = [t, gait(j).tspan];
    else
      t = [t, t(end) + gait(j).tspan];
    end % if

    nodes = 1:nlp.Phase(j).NumNode;

    for node = nodes
      x = gait(j).states.x(:, node);
      dx = gait(j).states.dx(:, node);
      ddx = gait(j).states.ddx(:, node);

      yd = nlp.Phase(j).Plant.VirtualConstraints.time.calcDesired( ...
        gait(j).tspan(node), x, dx, ...
        reshape(gait(j).params.atime, [M, N]), gait(j).params.ptime);

      ya = nlp.Phase(j).Plant.VirtualConstraints.time.calcActual( ...
        x, dx);

      y_des = [y_des, yd{1}];
      dy_des = [dy_des, yd{2}];
      ddy_des = [ddy_des, yd{3}];

      y_act = [y_act, ya{1}];
      dy_act = [dy_act, ya{2}];
      ddy_act = [ddy_act, ya{3} * [dx; ddx]];
    end % for nodes
  end % for  cont_domain_idx

else
  logger = data;
  for j = 1:length(logger)
    if j == 1
      t = [t, logger(j).flow.t];
    else
      t = [t, t(end) + logger(j).flow.t];
    end % if

    x = logger(j).flow.states.x;
    dx = logger(j).flow.states.dx;
    yd = logger(j).flow.yd_time;
    ya = logger(j).flow.ya_time;
    d1yd = logger(j).flow.d1yd_time;
    d1ya = logger(j).flow.d1ya_time;
    d2ya = zeros(size(ya));
    d2yd = zeros(size(yd));

    y_des = [y_des, yd];
    dy_des = [dy_des, d1yd];
    ddy_des = [ddy_des, d2yd];

    y_act = [y_act, ya];
    dy_act = [dy_act, d1ya];
    ddy_act = [ddy_act, d2ya];
  end % for steps

end % if

ax = [];
for i = 7:indices(end)
  j = i - 6;
  f = figure;
  clf;
  set(f, 'WindowStyle', 'docked');
  ax = [ax, subplot(3, 1, 1)]; %#ok<*AGROW>
  hold on;
  plot(t, y_des(j, :), 'r-');
  plot(t, y_act(j, :), 'g--');
  title('Output Position');
  legend('des', 'act');

  ax = [ax, subplot(3, 1, 2)];
  hold on;
  plot(t, dy_des(j, :), 'r-');
  plot(t, dy_act(j, :), 'g--');
  title('Output Velocity');
  legend('des', 'act');

  ax = [ax, subplot(3, 1, 3)];
  hold on;
  plot(t, ddy_des(j, :), 'r-');
  plot(t, ddy_act(j, :), 'g--');
  title('Output Acceleration');
  legend('des', 'act');

  f.Name = [nlp.Phase(1).Plant.VirtualConstraints.time.OutputLabel{j}];
end

linkaxes(ax, 'x');


end
