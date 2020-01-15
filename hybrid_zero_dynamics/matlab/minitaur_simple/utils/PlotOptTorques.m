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

function PlotOptTorques(model, nlp, data, indices)

act_joint_idx = find(arrayfun(@(x) ~isempty(x.Actuator), model.Joints));

if nargin < 4
  indices = act_joint_idx;
else
  if isempty(indices), return;
  end
end

joint_names = {model.Joints.Name};

t = [];
u = [];
u_lb = [];
u_ub = [];

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

    u = [u, gait(j).inputs.u];
    u_lb = [u_lb, [nlp.Phase(j).OptVarTable.u.LowerBound]];
    u_ub = [u_ub, [nlp.Phase(j).OptVarTable.u.UpperBound]];
  end

else
  logger = data;
  u_lb_vec = nlp.Phase(1).OptVarTable.u(1, 1).LowerBound;
  u_ub_vec = nlp.Phase(1).OptVarTable.u(1, 1).UpperBound;

  for j = 1:length(logger)

    t_size = length(logger(j).flow.t);

    if j == 1
      t = [t, logger(j).flow.t];
    else
      t = [t, t(end) + logger(j).flow.t];
    end % if

    u = [u, logger(j).flow.u];
    u_lb = [u_lb, repmat(u_lb_vec, [1, t_size])];
    u_ub = [u_ub, repmat(u_ub_vec, [1, t_size])];
  end %
end

ax = [];
for i = 1:length(indices)
  idx = indices(i);
  if ~ismember(idx, act_joint_idx)
    continue;
  end
  f = figure;
  clf;
  set(f, 'WindowStyle', 'docked');
  %         f.Position = [680 558 560 420];
  ax = [ax, axes(f)]; %#ok<LAXES,*AGROW>
  hold on;
  plot(t, u(i, :), 'b');
  plot(t, u_lb(i, :), 'r--');
  plot(t, u_ub(i, :), 'g--');

  title('Torque');
  legend('u', 'lb', 'ub');


  f.Name = [joint_names{idx}, '_torque'];
end

linkaxes(ax, 'x');

% In needed: multiplot in a figure
% figure
% subplot(2,1,1)
% plot(t, u_opt([1 4],:)); hold on
% plot(t, u_lb(1,:), 'k-x');
% plot(t, u_ub(1,:), 'k-o');
% title('Torque for FrontRight and BackLeft leg swing');
% legend('u_{FR}', 'u_{BL}', 'lb', 'ub');
% subplot(2,1,2)
% plot(t, u_opt([2 3],:)); hold on
% plot(t, u_lb(1,:), 'k-x');
% plot(t, u_ub(1,:), 'k-o');
% title('Torque for FrontLeft and BackRight leg swing');
% legend('u_{FL}', 'u_{BR}', 'lb', 'ub');
%


end
