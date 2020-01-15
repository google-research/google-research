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

function TrotStance1Constraints(nlp, bounds, varargin)

domain = nlp.Plant;

% relative degree 2 outputs
domain.VirtualConstraints.time.imposeNLPConstraint(nlp, ...
  [bounds.time.kp, bounds.time.kd], [1, 1]);

% tau boundary [0,1]
T = SymVariable('t', [nlp.OptVarTable.T(1).Dimension, 1]);
p_name = nlp.OptVarTable.ptime(1).Name;
p = SymVariable(p_name, [nlp.OptVarTable.ptime(1).Dimension, 1]);
tau_0 = SymFunction(['tau_0_', domain.Name], T(1)-p(2), {T, p});
tau_F = SymFunction(['tau_F_', domain.Name], T(2)-p(1), {T, p});

tau0_cstr = NlpFunction('Name', 'tau0', ...
  'Dimension', 1, ...
  'lb', 0, ...
  'ub', 0, ...
  'Type', 'Linear', ...
  'SymFun', tau_0, ...
  'DepVariables', [nlp.OptVarTable.T(1); nlp.OptVarTable.ptime(1)]);

addConstraint(nlp, 'tau0', 'first', tau0_cstr);

tauF_cstr = NlpFunction('Name', 'tauF', ...
  'Dimension', 1, ...
  'lb', 0, ...
  'ub', 0, ...
  'Type', 'Linear', ...
  'SymFun', tau_F, ...
  'DepVariables', [nlp.OptVarTable.T(end); nlp.OptVarTable.ptime(end)]);

addConstraint(nlp, 'tauF', 'last', tauF_cstr);


pFR = getCartesianPosition(domain, domain.ContactPoints.FrontRightToe);
pFL = getCartesianPosition(domain, domain.ContactPoints.FrontLeftToe);
pBR = getCartesianPosition(domain, domain.ContactPoints.BackRightToe);
pBL = getCartesianPosition(domain, domain.ContactPoints.BackLeftToe);
steplen_front = sqrt((pFL(1) - pFR(1)).^2);
steplen_back = sqrt((pBL(1) - pBR(1)).^2);
avg_steplen = 0.5 * (steplen_front + steplen_back);

steplen_func = SymFunction(['average_steplen_', domain.Name], avg_steplen, ...
  {domain.States.x});

steplen_des_lb = bounds.constraints.average_steplen.lb;
steplen_des_ub = bounds.constraints.average_steplen.ub;
addNodeConstraint(nlp, steplen_func, {'x'}, 'last', steplen_des_lb, ...
  steplen_des_ub, 'Nonlinear');

%% Define a trapezium lower bound for swing foot height
swing_ht_max = bounds.constraints.min_swingfoot_lift.desired;
num_nodes = nlp.NumNode;
flat_len = 3;
ramp_len = 7;
rem_len = num_nodes - 2 * flat_len - 2 * ramp_len;
node_list = 1:num_nodes;
swg_ht_list = [zeros(1, flat_len), ...
  linspace(0, swing_ht_max, ramp_len), ...
  swing_ht_max * ones(1, rem_len), ...
  linspace(swing_ht_max, 0, ramp_len), ...
  zeros(1, flat_len)];

for node = 1:num_nodes
  % Front Left Foot should be 0 at last node and have clearance during
  % stance phase
  X = SymVariable('x', [nlp.OptVarTable.x(1).Dimension, 1]);
  swingFootHeight1 = SymFunction(['swing_foot1_height_', domain.Name], ...
    nlp.Plant.EventFuncs.frontleftFootHeight.ConstrExpr, {X});
  addNodeConstraint(nlp, swingFootHeight1, {'x'}, node_list(node), ...
    swg_ht_list(node), Inf, 'Nonlinear');

  % Back Right Foot should be 0 at last node and have clearance during
  % stance phase
  X = SymVariable('x', [nlp.OptVarTable.x(1).Dimension, 1]);
  swingFootHeight2_Expr = ...
    nlp.Plant.getCartesianPosition(nlp.Plant.ContactPoints.BackRightToe, ...
    [0; 0; 0]');
  swingFootHeight2 = SymFunction(['swing_foot2_height_', domain.Name], ...
    swingFootHeight2_Expr(3), {X});
  addNodeConstraint(nlp, swingFootHeight2, {'x'}, node_list(node), ...
    swg_ht_list(node), Inf, 'Nonlinear');

end

% Swing feet velocity at impact (last node) must be zero
swingtoe1 = domain.ContactPoints.FrontLeftToe;
J_st1 = swingtoe1.computeBodyJacobian(domain.numState);
v_st1 = J_st1 * domain.States.dx;
swingtoe1_linvel_fun = SymFunction(['swingtoe1_linearvelocity_', ...
  nlp.Plant.Name], v_st1(1:3), {nlp.Plant.States.x, nlp.Plant.States.dx});
addNodeConstraint(nlp, swingtoe1_linvel_fun, {'x', 'dx'}, 'last', 0, 0, ...
  'Nonlinear');

swingtoe2 = domain.ContactPoints.BackRightToe;
J_st2 = swingtoe2.computeBodyJacobian(domain.numState);
v_st2 = J_st2 * domain.States.dx;
swingtoe2_linvel_fun = SymFunction(['swingtoe2_linearvelocity_', ...
  nlp.Plant.Name], v_st2(1:3), {nlp.Plant.States.x, nlp.Plant.States.dx});
addNodeConstraint(nlp, swingtoe2_linvel_fun, {'x', 'dx'}, 'last', 0, 0, ...
  'Nonlinear');


%% Swing Toe Trajectory Symmetry

% Get local origin at the hip motor
p_motorFL = getCartesianPosition(domain, domain.Joints(7));
p_motorBR = getCartesianPosition(domain, domain.Joints(10));

pFL_rel = pFL - p_motorFL;
pBR_rel = pBR - p_motorBR;

swingtoe_symmetry = pFL_rel(3) - pBR_rel(3);
swingtoe_symmetry_fn = SymFunction(['SwingToeSymmetry_', domain.Name], ...
  swingtoe_symmetry, domain.States.x);
addNodeConstraint(nlp, swingtoe_symmetry_fn, {'x'}, 'all', 0, 0, 'NonLinear');


end
