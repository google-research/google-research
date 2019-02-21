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

gait = struct( ...
  'tspan', tspan, ...
  'states', states, ...
  'inputs', inputs, ...
  'params', params);

%% For Two Step Hybrid System Optimization
figure
subplot(2, 1, 1)
plot([linspace(0, 1, degree), linspace(1, 2, degree)], ...
  [vc1(1:4, :), vc3(1:4, :)], '-o')
title('phase2 - hip swing bezier coefficients')
% legend('FR','BR','BL','FL')
l1 = legend('FL', 'BL', 'FR', 'BR');
set(l1, 'Orientation', 'Horizontal', 'Location', 'BestOutside');

subplot(2, 1, 2)
plot([linspace(0, 1, degree), linspace(1, 2, degree)], ...
  [vc1(5:8, :), vc3(5:8, :)], '-o')
title('phase2 - knee extension bezier coefficients')
% legend('FR','BR','BL','FL')
% l2 = legend('FL', 'BL', 'FR', 'BR');
% set(l2, 'Orientation', 'Horizontal', 'Location', 'BestOutside');

%

%% Visualize
t_log = [tspan{1, 1}, (tspan{1}(end) + tspan{3, 1})];
q_log = [states{1, 1}.x, states{3, 1}.x];
v_log = [states{1, 1}.dx, states{3, 1}.dx];


nodes = nlp.Phase(1).NumNode;

nodes_list = 1:1:nodes;

%%% Phase 1

for node = 1:nodes
  %%% Constraints Evaluation

  % Foot Left Foot Height (FL)
  swgft1_ht(node) = swing_foot1_height_TrotStance1(q_log(:, node));
  % Back Right Foot Height (BR)
  swgft2_ht(node) = swing_foot2_height_TrotStance1(q_log(:, node));

  %% Lower Bounds
  swgft1_ht_lb(node) = ...
    nlp.Phase(1).ConstrTable.swing_foot1_height_TrotStance1(node).LowerBound;
  swgft2_ht_lb(node) = ...
    nlp.Phase(1).ConstrTable.swing_foot2_height_TrotStance1(node).LowerBound;
  %% Upper Bounds
  swgft1_ht_ub(node) = ...
    nlp.Phase(1).ConstrTable.swing_foot1_height_TrotStance1(node).UpperBound;
  swgft2_ht_ub(node) = ...
    nlp.Phase(1).ConstrTable.swing_foot2_height_TrotStance1(node).UpperBound;
  %%
  vel_com_x(:, node) = CoMXVel_TrotStance1(q_log(:, node), v_log(:, node));

end

%%% Phase 2

for node = (nodes + 1):2 * nodes
  node0 = node - nodes;
  %%% Constraints Evaluation

  % Front Right Foot Height (FR)
  swgft1_ht(node) = swing_foot1_height_TrotStance2(q_log(:, node));
  % Back Left Foot Height (BL)
  swgft2_ht(node) = swing_foot2_height_TrotStance2(q_log(:, node));

  %% Lower Bounds
  swgft1_ht_lb(node) = ...
    nlp.Phase(3).ConstrTable.swing_foot1_height_TrotStance2(node0).LowerBound;
  swgft2_ht_lb(node) = ...
    nlp.Phase(3).ConstrTable.swing_foot2_height_TrotStance2(node0).LowerBound;
  %% Upper Bounds
  swgft1_ht_ub(node) = ...
    nlp.Phase(3).ConstrTable.swing_foot1_height_TrotStance2(node0).UpperBound;
  swgft2_ht_ub(node) = ...
    nlp.Phase(3).ConstrTable.swing_foot2_height_TrotStance2(node0).UpperBound;
  %%
  vel_com_x(:, node) = CoMXVel_TrotStance1(q_log(:, node), v_log(:, node));
end

%% Plotting
figure

subplot(2, 1, 1)
plot([1:1:2 * nodes], swgft1_ht, 'b-');
hold on;
plot([1:1:2 * nodes], swgft1_ht_lb, 'ro');
plot([1:1:2 * nodes], swgft1_ht_ub, 'go');
ylabel('swing foot height')
title('Trot Stance 1')

subplot(2, 1, 2)
plot([1:1:2 * nodes], swgft2_ht, 'b-');
hold on;
plot([1:1:2 * nodes], swgft2_ht_lb, 'ro');
plot([1:1:2 * nodes], swgft2_ht_ub, 'go');
xlabel('node number')
ylabel('swing foot height')
title('Trot Stance 2')
l = legend('opt value', 'lower bound', 'upper bound');
set(l, 'Orientation', 'Horizontal', 'Location', 'BestOutside')


figure
plot([1:1:2 * nodes], vel_com_x)
xlabel('node number')
ylabel('xcom velocity')


for i = 1:21
  q10 = q_log(:, i);
  p_mFR = p_motor_front_rightR_joint(q10);
  p_mBL = p_motor_back_leftL_joint(q10);
  p_FR = p_FrontRightToe(q10);
  p_BL = p_BackLeftToe(q10);
  p_FR_rel(:, i) = p_FR - 0 * p_mFR;
  p_BL_rel(:, i) = p_BL - 0 * p_mBL;
end

disp('Checking Foot Pos Symmetry in Trot Stance 1')
figure
subplot(3, 1, 1)
plot(1:21, [p_FR_rel(1, :); p_BL_rel(1, :)])
title('FootPosX')
subplot(3, 1, 2)
plot(1:21, [p_FR_rel(2, :); p_BL_rel(2, :)])
title('FootPosY')
subplot(3, 1, 3)
plot(1:21, [p_FR_rel(3, :); p_BL_rel(3, :)])
title('FootPosZ')
legend('Front', 'Back');

figure
subplot(3, 1, 1)
plot(1:21, (p_FR_rel(1, :) + p_BL_rel(1, :)))
title('FootPosX')
subplot(3, 1, 2)
plot(1:21, (p_FR_rel(2, :) - p_BL_rel(2, :)))
title('FootPosY')
subplot(3, 1, 3)
plot(1:21, (p_FR_rel(3, :) - p_BL_rel(3, :)))
title('FootPosZ')


disp('Displaying Initial and Final Conditions for Phase 1 and Phase 2');
[q_log(3:end, nodes), q_log(3:end, nodes+1), q_log(3:end, 2*nodes), ...
  q_log(3:end, 1)]
[v_log(3:end, nodes), v_log(3:end, nodes+1), v_log(3:end, 2*nodes), ...
  v_log(3:end, 1)]

%% xMap and dxMap TS1
disp('Displaying Impact Map Error');
disp('Impact1_x0');
x0_err1 = q_log(3:end, nodes) - q_log(3:end, nodes+1);
x0_err1'
disp('Impact1_dx0');
dx0_err1 = v_log(:, nodes) - v_log(:, nodes+1);
dx0_err1'

%% xMap and dxMap TS2
disp('Impact2_x0');
x0_err2 = q_log(3:end, 2*nodes) - q_log(3:end, 1);
x0_err2'
disp('Impact2_dx0');
dx0_err2 = v_log(:, 2*nodes) - v_log(:, 1);
dx0_err2'

% %% 1) Bezier Leg Symmetry
% disp('TS1 Leg Symmetry Error');
% disp(BezierLegSymmetry_TrotStance1(params{1}.atime)')
% disp('TS2 Leg Symmetry Error');
% disp(BezierLegSymmetry_TrotStance2(params{3}.atime)')

%% 1) Swing Toe Symmetry
disp('TS1 SwingToe Symmetry Error');
disp(SwingToeSymmetry_TrotStance1(states{1}.x(:,end)))
disp('TS2 SwingToe Symmetry Error');
disp(SwingToeSymmetry_TrotStance2(states{3}.x(:,end)))


%% 2) Bezier Phase Symmetry
disp('Trot Phase Symmetry Error');
disp(BezierPhaseSymmetry(params{1}.atime, params{3}.atime)')

%% 3) Average StepLen
disp('TS1 First Node Avg StepLen');
average_steplen_TrotStance1(q_log(:, 1))
disp('TS1 Last Node Avg StepLen');
average_steplen_TrotStance1(q_log(:, nodes))
disp('TS2 First Node Avg StepLen');
average_steplen_TrotStance2(q_log(:, nodes+1))
disp('TS2 Last Node Avg StepLen');
average_steplen_TrotStance2(q_log(:, 2*nodes))

%% 4) Final time
disp('TS1 Duration');
params{1}.ptime(1)
disp('TS2 Duration');
params{3}.ptime(1)
