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

function PlotOptStates(model, nlp, data, indices)


if nargin < 4
  indices = 1:length(model.Joints);
else
  if isempty(indices), return;
  end
end
joint_names = {model.Joints.Name};


t = [];
x = [];
x_lb = [];
x_ub = [];
dx = [];
dx_lb = [];
dx_ub = [];
ddx = [];
ddx_lb = [];
ddx_ub = [];

if isa(data, 'struct')
  gait = data;
  cont_domain_idx = find(cellfun(@(x)isa(x, 'ContinuousDynamics'), ...
    {nlp.Phase.Plant}));


  for j = cont_domain_idx

    if j == 1
      t = [t, gait(j).tspan];
    else
      t = [t, t(end) + gait(j).tspan];
    end

    x = [x, gait(j).states.x];
    x_lb = [x_lb, [nlp.Phase(j).OptVarTable.x.LowerBound]];
    x_ub = [x_ub, [nlp.Phase(j).OptVarTable.x.UpperBound]];

    dx = [dx, gait(j).states.dx];
    dx_lb = [dx_lb, [nlp.Phase(j).OptVarTable.dx.LowerBound]];
    dx_ub = [dx_ub, [nlp.Phase(j).OptVarTable.dx.UpperBound]];

    ddx = [ddx, gait(j).states.ddx];
    ddx_lb = [ddx_lb, [nlp.Phase(j).OptVarTable.ddx.LowerBound]];
    ddx_ub = [ddx_ub, [nlp.Phase(j).OptVarTable.ddx.UpperBound]];
  end % for  cont_domain_idx

else
  logger = data;
  x_lb_vec = nlp.Phase(1).OptVarTable.x(1, 1).LowerBound;
  x_ub_vec = nlp.Phase(1).OptVarTable.x(1, 1).UpperBound;
  dx_lb_vec = nlp.Phase(1).OptVarTable.dx(1, 1).LowerBound;
  dx_ub_vec = nlp.Phase(1).OptVarTable.dx(1, 1).UpperBound;
  ddx_lb_vec = nlp.Phase(1).OptVarTable.ddx(1, 1).LowerBound;
  ddx_ub_vec = nlp.Phase(1).OptVarTable.ddx(1, 1).UpperBound;


  for j = 1:length(logger)

    t_size = length(logger(j).flow.t);

    if j == 1
      t = [t, logger(j).flow.t];
    else
      t = [t, t(end) + logger(j).flow.t];
    end % if

    x = [x, logger(j).flow.states.x];
    x_lb = [x_lb, repmat(x_lb_vec, [1, t_size])];
    x_ub = [x_ub, repmat(x_ub_vec, [1, t_size])];

    dx = [dx, logger(j).flow.states.dx];
    dx_lb = [dx_lb, repmat(dx_lb_vec, [1, t_size])];
    dx_ub = [dx_ub, repmat(dx_ub_vec, [1, t_size])];

    ddx = [ddx, logger(j).flow.states.ddx];
    ddx_lb = [ddx_lb, repmat(ddx_lb_vec, [1, t_size])];
    ddx_ub = [ddx_ub, repmat(ddx_ub_vec, [1, t_size])];

  end % for steps

end % if


ax = [];
for i = indices
  f = figure;
  clf;
  set(f, 'WindowStyle', 'docked');

  ax = [ax, subplot(3, 1, 1)]; %#ok<*AGROW>
  hold on;
  plot(t, x(i, :), 'b');
  plot(t, x_lb(i, :), 'r--');
  plot(t, x_ub(i, :), 'g--');
  title('Joint Displacement');
  legend('q', 'lb', 'ub');

  ax = [ax, subplot(3, 1, 2)];
  hold on;
  plot(t, dx(i, :), 'b');
  plot(t, dx_lb(i, :), 'r--');
  plot(t, dx_ub(i, :), 'g--');
  title('Joint Velocity');
  legend('dq', 'lb', 'ub');

  ax = [ax, subplot(3, 1, 3)];
  hold on;
  plot(t, ddx(i, :), 'b');
  plot(t, ddx_lb(i, :), 'r--');
  plot(t, ddx_ub(i, :), 'g--');
  title('Joint Acceleration');
  legend('ddq', 'lb', 'ub');


  f.Name = [joint_names{i}, '_state'];
end

linkaxes(ax, 'x');


end
