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

%% Setup
clear;
restoredefaultpath;
clc;
cur = pwd;
addpath(genpath(cur));

%% Add Frost to path
if ispc
   FROST_ROOT = addpath('C:\Users\Avinash Siravuru\Box\repos\frost-dev\');
   slash = '\';
else
   FROST_ROOT = addpath('/data/repos/frost-dev/');
   slash = '/';
end
addpath(FROST_ROOT);

% Then call the fost_addpath script.
frost_addpath;

% Declare Constants
STEPLEN_DES = 0.0;
STEPLEN_MARGIN = 0.005;
if STEPLEN_DES <= 0.01
  MAX_VELCOM_X_ENDPTS = 1e-3;
else
  MAX_VELCOM_X_ENDPTS = Inf;
end
SWINGFOOTLIFT_DES = 0.07;

% 
% KP = 100;
% KD = 20; % FROST Default Gains

KP = 10*1.1;
KD = 10*0.07; % Mintaur Gains

BEZIER_DEG = 4;
OPT_TYPE = ['two_step_deg', num2str(BEZIER_DEG)];


OPTIMIZE = true;
COMPILE = false;
MAX_ITERS = 1000; % maximum number of optimization iterations.

export_path = fullfile(cur, ['gen_', OPT_TYPE, slash]);
load_path = '';%[export_path,'dynamics/'];
delay_set = true;

solution_path = fullfile(cur, ['sol',slash]);
if ~exist(solution_path, 'dir')
  mkdir(solution_path)
end

if OPTIMIZE
  tag = datestr(now, 30);
end

%% Start the parallel pool is not already 'ON'.
if COMPILE && OPTIMIZE
  if isempty(gcp('nocreate'))
    parpool
  end
end

%% Load Model
minitaur = MinitaurSimple(['urdf',slash,'minitaur_simple.urdf']);

if isempty(load_path)
  minitaur.configureDynamics('DelayCoriolisSet', delay_set, ...
    'OmitCoriolisSet', true);
else
  minitaur.loadDynamics(load_path, delay_set);
end

%% Visualize robot
% ext_disp = DisplayMinitaur(minitaur, 'ExportPath', [export_path, 'anim/']);

%% Define domains and the hybrid system
trot_stance1 = TrotStance1(minitaur, load_path, BEZIER_DEG);
trot_stance2 = TrotStance2(minitaur, load_path, BEZIER_DEG);

trot_impact1 = TrotImpact1(trot_stance1, load_path);
trot_impact2 = TrotImpact2(trot_stance2, load_path);

minitaur_trot = HybridSystem('Minitaur_Trot');
ctrl = IOFeedback('IO');
ctrl.Param.kp = KP;
ctrl.Param.kd = KD;
minitaur_trot = addVertex(minitaur_trot, 'TrotStance1', 'Domain', ...
  trot_stance1, 'Control', ctrl);
minitaur_trot = addVertex(minitaur_trot, 'TrotStance2', 'Domain', ...
  trot_stance2, 'Control', ctrl);


srcs = {'TrotStance1', 'TrotStance2'};
tars = {'TrotStance2', 'TrotStance1'};

minitaur_trot = addEdge(minitaur_trot, srcs, tars);
minitaur_trot = setEdgeProperties(minitaur_trot, srcs, tars, ...
  'Guard', {trot_impact1, trot_impact2});

% Add User Constraints
trot_stance1.UserNlpConstraint = str2func('TrotStance1Constraints');
trot_stance2.UserNlpConstraint = str2func('TrotStance2Constraints');

trot_impact1.UserNlpConstraint = str2func('TrotImpact1Constraints');
trot_impact2.UserNlpConstraint = str2func('TrotImpact2Constraints');

% Define User Costs
u1 = trot_stance1.Inputs.Control.u;
u2 = trot_stance2.Inputs.Control.u;

wt_s = 10;
wt_e = 1;

u1_s = tomatrix(u1(1:4));
u1_s_ = norm(u1_s).^2;
u1_e = tomatrix(u1(5:8));
u1_e_ = norm(u1_e).^2;
u1_ = tovector(wt_e*(u1_e_ / (4 * (40^2)))+wt_s*(u1_s_ / (4 * (40^2))));
u1_fun = SymFunction(['torque_', trot_stance1.Name], u1_, {u1});

u2_s = tomatrix(u2(1:4));
u2_s_ = norm(u2_s).^2;
u2_e = tomatrix(u2(5:8));
u2_e_ = norm(u2_e).^2;
u2_ = tovector(wt_e*(u2_e_ / (4 * (40^2)))+wt_s*(u2_s_ / (4 * (40^2))));
u2_fun = SymFunction(['torque_', trot_stance2.Name], u2_, {u2});

%% Create optimization problem

num_grid.TrotStance1 = 10;
num_grid.TrotStance2 = 10;
nlp = HybridTrajectoryOptimization('Minitaur_Trot_Opt', minitaur_trot, ...
  num_grid, [], 'EqualityConstraintBoundary', 1e-4);

% Configure bounds
SetBounds;

% load some optimization related expressions here
if ~isempty(load_path)
  nlp.configure(bounds, 'LoadPath', load_path);
else
  nlp.configure(bounds);
end

nlp.update;

% Add costs and update

addRunningCost(nlp.Phase(getPhaseIndex(nlp, 'TrotStance1')), u1_fun, 'u');
nlp.update;
addRunningCost(nlp.Phase(getPhaseIndex(nlp, 'TrotStance2')), u2_fun, 'u');
nlp.update;

% CoM Velocity constraints
p_com1 = trot_stance1.getComPosition;
v_com1 = jacobian(p_com1, trot_stance1.States.x) * trot_stance1.States.dx;
vcom_func1 = SymFunction(['CoMVel_', trot_stance1.Name], v_com1, ...
  {trot_stance1.States.x, trot_stance1.States.dx});
vxcom_func1 = SymFunction(['CoMXVel_', trot_stance1.Name], v_com1(1), ...
  {trot_stance1.States.x, trot_stance1.States.dx});

p_com2 = trot_stance2.getComPosition;
v_com2 = jacobian(p_com2, trot_stance2.States.x) * trot_stance2.States.dx;
vcom_func2 = SymFunction(['CoMVel_', trot_stance2.Name], v_com2, ...
  {trot_stance2.States.x, trot_stance2.States.dx});
vxcom_func2 = SymFunction(['CoMXVel_', trot_stance2.Name], v_com2(1), ...
  {trot_stance2.States.x, trot_stance2.States.dx});


% addNodeConstraint(nlp.Phase(1), vxcom_func1, {'x', 'dx'}, 'all', 0, Inf, ...
%   'Nonlinear');
% addNodeConstraint(nlp.Phase(3), vxcom_func2, {'x', 'dx'}, 'all', 0, Inf, ...
%   'Nonlinear');

addNodeConstraint(nlp.Phase(1), vxcom_func1, {'x', 'dx'}, 'except-terminal', 0, Inf, ...
  'Nonlinear');
addNodeConstraint(nlp.Phase(3), vxcom_func2, {'x', 'dx'}, 'except-terminal', 0, Inf, ...
  'Nonlinear');
addNodeConstraint(nlp.Phase(1), vxcom_func1, {'x', 'dx'}, 'terminal', 0, MAX_VELCOM_X_ENDPTS, ...
  'Nonlinear');
addNodeConstraint(nlp.Phase(3), vxcom_func2, {'x', 'dx'}, 'terminal', 0, MAX_VELCOM_X_ENDPTS, ...
  'Nonlinear');

% Add common constraints
[N, M] = size(trot_stance1.Params.atime);

% Phase Symmetry (Symmetry in Leg motion across the two trot phases)
a1 = SymVariable('a1time', [N, M]);
a2 = SymVariable('a2time', [N, M]);
a1_vec = a1.tovector;
a2_vec = a2.tovector;

%symmetry_matrix = a2 - a1([2 1 4 3 6 5 8 7],:);
end_pts_idx = 1:M; %[1 2 (M-1) M];
phase_symmetry_matrix = a2(:, end_pts_idx) - diag([-1, -1, -1, -1, ...
  1, 1, 1, 1]) * a1([3, 4, 1, 2, 7, 8, 5, 6], end_pts_idx);
phase_symmetry_vector = phase_symmetry_matrix.tovector;

bezier_phase_symmetry_fn = SymFunction('BezierPhaseSymmetry', ...
  phase_symmetry_vector, {a1_vec, a2_vec});

bezier_phase_symmetry_cstr = NlpFunction('Name', 'BezierPhaseSymmetryConstr',...
  'Dimension', N*length(end_pts_idx), ...
  'lb', 0, ...
  'ub', 0, ...
  'Type', 'Linear', ...
  'SymFun', bezier_phase_symmetry_fn, ...
  'DepVariables', [nlp.Phase(getPhaseIndex(nlp, ...
  'TrotStance1')).OptVarTable.atime(1), nlp.Phase(getPhaseIndex(nlp, ...
  'TrotStance2')).OptVarTable.atime(1)]);

addConstraint(nlp.Phase(getPhaseIndex(nlp, 'TrotStance1')), ...
  'BezierPhaseSymmetry', 'first', bezier_phase_symmetry_cstr);
addConstraint(nlp.Phase(getPhaseIndex(nlp, 'TrotStance2')), ...
  'BezierPhaseSymmetry', 'first', bezier_phase_symmetry_cstr);

% Make both step durations same
T_name = lower(nlp.Phase(1).OptVarTable.T(end).Name);
T1f = SymVariable([T_name, '1f'], ...
  [nlp.Phase(1).OptVarTable.T(end).Dimension, 1]);
T2f = SymVariable([T_name, '2f'], ...
  [nlp.Phase(3).OptVarTable.T(end).Dimension, 1]);
Tf = SymFunction('uniform_time_duration', T1f-T2f, {T1f, T2f});

uniform_time_cstr = NlpFunction('Name', 'UniformTimePerStep', ...
  'Dimension', nlp.Phase(1).OptVarTable.T(end).Dimension, ...
  'lb', 0, ...
  'ub', 0, ...
  'Type', 'Linear', ...
  'SymFun', Tf, ...
  'DepVariables', [nlp.Phase(getPhaseIndex(nlp, ...
  'TrotStance1')).OptVarTable.T(end), nlp.Phase(getPhaseIndex(nlp, ...
  'TrotStance2')).OptVarTable.T(end)]);

addConstraint(nlp.Phase(getPhaseIndex(nlp, 'TrotStance1')), ...
  'UniformTimePerStep', 'last', uniform_time_cstr);
addConstraint(nlp.Phase(getPhaseIndex(nlp, 'TrotStance2')), ...
  'UniformTimePerStep', 'last', uniform_time_cstr);


nlp.update;

%% Compile
if COMPILE == true
  if ~exist([export_path, 'opt',slash], 'dir')
    mkdir([export_path, 'opt',slash])
  end
  minitaur.ExportKinematics([export_path, 'kinematics',slash]);
  trot_stance1.compile([export_path, 'dynamics',slash]);
  trot_stance2.compile([export_path, 'dynamics',slash]);
  trot_impact1.compile([export_path, 'dynamics',slash]);
  trot_impact2.compile([export_path, 'dynamics',slash]);
  compileConstraint(nlp, [], [], [export_path, 'opt',slash]);
  compileObjective(nlp, [], [], [export_path, 'opt',slash]);
  compileConstraint(nlp.Phase(1), 'CoMXVel_TrotStance1', [export_path, 'opt',slash]);
  compileConstraint(nlp.Phase(3), 'CoMXVel_TrotStance2', [export_path, 'opt',slash]);
  compileConstraint(nlp.Phase(1), 'BezierPhaseSymmetry', [export_path, 'opt',slash]);
  compileConstraint(nlp.Phase(3), 'BezierPhaseSymmetry', [export_path, 'opt',slash]);
  compileConstraint(nlp.Phase(1), 'UniformTimePerStep', [export_path, 'opt',slash]);
  compileConstraint(nlp.Phase(3), 'UniformTimePerStep', [export_path, 'opt',slash]);
  compileConstraint(nlp.Phase(1), 'average_steplen_TrotStance1', ...
    [export_path, 'opt',slash]);
  compileConstraint(nlp.Phase(3), 'average_steplen_TrotStance2', ...
    [export_path, 'opt',slash]);
  compileConstraint(nlp.Phase(1), 'BezierLegSymmetry_TrotStance1', ...
    [export_path, 'opt',slash]);
  compileConstraint(nlp.Phase(3), 'BezierLegSymmetry_TrotStance2', ...
    [export_path, 'opt',slash]);
  compileConstraint(nlp.Phase(1), 'SwingToeSymmetry_TrotStance1', ...
    [export_path, 'opt',slash]);
  compileConstraint(nlp.Phase(3), 'SwingToeSymmetry_TrotStance2', ...
    [export_path, 'opt',slash]);
  compileConstraint(nlp.Phase(1), 'pd_feedback_TrotStance1', ...
    [export_path, 'opt',slash]);
  compileConstraint(nlp.Phase(3), 'pd_feedback_TrotStance2', ...
    [export_path, 'opt',slash]);
end

% Example constraint removal
%
% removeConstraint(nlp.Phase(1), 'BezierLegSymmetry_TrotStance1');
% removeConstraint(nlp.Phase(3), 'BezierLegSymmetry_TrotStance2');
% removeConstraint(nlp.Phase(3), 'UniformTimePerStep');
% removeConstraint(nlp.Phase(1), 'CoMVel_TrotStance1')
% removeConstraint(nlp.Phase(3), 'CoMVel_TrotStance2')

% nlp.update;

%% Create Ipopt solver
addpath(genpath(export_path));
SetBounds;
nlp.updateVariableBounds(bounds);
nlp.update;
solver = IpoptApplication(nlp);
solver.Options.ipopt.max_iter = MAX_ITERS;

%% Check ConstrTable finally!
% nlp.Phase(1).ConstrTable{1,:}.Name
% nlp.Phase(3).ConstrTable{1,:}.Name
disp(nlp.Phase(1).ConstrTable(1, 23).average_steplen_TrotStance1.UpperBound)
disp(nlp.Phase(3).ConstrTable(21, 23).average_steplen_TrotStance2.UpperBound)
disp(nlp.Phase(1).ConstrTable(1, 23).average_steplen_TrotStance1.LowerBound)
disp(nlp.Phase(3).ConstrTable(21, 23).average_steplen_TrotStance2.LowerBound)


%% Run Optimization.
if OPTIMIZE == true
  tic
  [sol, info] = optimize(solver);
  toc

  checkConstraints(nlp, sol, 1e-4, [solution_path, 'cons_log_', tag, '.txt']);

  [tspan, states, inputs, params] = exportSolution(nlp, sol);

  % Save results
  save([solution_path, 'solndata_', tag], 'tspan', 'states', 'inputs', ...
    'params', 'sol', 'bounds', 'nlp');

  % Some optimization analysis
  vc1 = reshape(params{1}.atime, [num_hol, degree]);
  vc3 = reshape(params{3}.atime, [num_hol, degree]);
  save([solution_path, 'bezcoeffs_', tag, '.txt'], 'vc1', 'vc3', '-ascii');
  disp('Saved optimized gait parameters');
else
  keyboard;
end


gait = struct('tspan', tspan, 'states', states, 'inputs', inputs, 'params', ...
  params);

%% Plotting

% PlotOptStates(minitaur, nlp, gait);
% PlotOptTorques(minitaur, nlp, gait);
% PlotDesOutputs(minitaur, nlp, gait);

PlotOptConstraints;
q10 = q_log(:, 1);
q1F = q_log(:, 21);
q20 = q_log(:, 22);
q2F = q_log(:, 42);
v10 = v_log(:, 1);
v1F = v_log(:, 21);
v20 = v_log(:, 22);
v2F = v_log(:, 42);


%% Animation
cycles = 3;
anim = Animator.MinitaurSimpleAnimator(t_log, q_log);
anim.pov = Animator.AnimatorPointOfView.Free;
anim.Animate(true);
anim.isLooping = false;
anim.updateWorldPosition = false;
anim.endTime = cycles * (params{1}.ptime(1) + params{3}.ptime(1));
conGUI = Animator.AnimatorControls();
conGUI.anim = anim;
% %
keyboard;

make_vid.cycles = 2;
make_vid.flag = true;
if make_vid.flag
  % To save gifs: 'gifs/optimalgait_Bez4_3d.gif';
  video_path = strrep(solution_path,['sol',slash],['video',slash]);
  if ~exist(video_path, 'dir')
      mkdir(video_path);
  end
  make_vid.filename = [video_path, ...
                       'optimalgait_25cm_steplen_',tag,'.gif'];
else
  make_vid.filename = '';
end
make_vid.visibility = 'off';
make_vid.pov = [35, 15];
plot_frames = false;
PlotMinitaurSimple(t_log, q_log, make_vid, plot_frames);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Simulate
tic

% Defaults all zeros
x0 = zeros(2*26,1);

% Initial Optimal Condn
x0 = [states{1}.x(:, 1); states{1}.dx(:, 1)];

% load holonomic outputs' bezier params
params{1}.atime = reshape(params{1}.atime, [num_hol, degree]);
params{3}.atime = reshape(params{3}.atime, [num_hol, degree]);

trot_stance1_params = params{1};
trot_impact1_params = params{2};
trot_stance2_params = params{3};
trot_impact2_params = params{4};
trot_stance1_params.ktime = 2 * [50, 10];
trot_stance2_params.ktime = 2 * [50, 10];

minitaur_trot = setVertexProperties(minitaur_trot, 'TrotStance1', 'Param', ...
  trot_stance1_params);
minitaur_trot = setVertexProperties(minitaur_trot, 'TrotStance2', 'Param', ...
  trot_stance2_params);

% Remove some holonomic constraints to avoid decoupling matrix singularities.
% trot_stance1.removeHolonomicConstraint('BackLeftToe');
% trot_stance2.removeHolonomicConstraint('BackRightToe');

steps = 0.6;
cycles = 1;
tf = steps * (params{1}.ptime(1) + params{3}.ptime(1));
minitaur_trot.setOption('OdeSolver', @ode15s);
logger = minitaur_trot.simulate(0, x0, tf, [], 'NumCycle', cycles);

toc

steps = length(logger);
sim_states = cell(steps, 1);
sim_tspan = cell(steps, 1);
sim_params = cell(steps, 1);
sim_inputs = cell(steps, 1);
for i = 1:steps
  sim_states{i} = logger(i).flow.states;
  sim_tspan{i} = logger(i).flow.t;
  sim_inputs{i} = logger(i).flow.u;
  sim_params{i} = logger(i).static.params;
end

sim_gait = struct( ...
  'tspan', sim_tspan, ...
  'states', sim_states, ...
  'inputs', sim_inputs, ...
  'params', sim_params);

% plotOptStates(minitaur, nlp, logger);
% plotOptTorques(minitaur, nlp, logger);
% plotDesOutputs(minitaur, nlp, logger);

q_log_sim = [];
t_log_sim = [];
dq_log_sim = [];
for i = 1:length(logger)
  q_step = logger(1, i).flow.states.x;
  dq_step = logger(1, i).flow.states.dx;
  t_step = logger(1, i).flow.t;
  q_log_sim = [q_log_sim, q_step]; %#ok<AGROW>
  t_log_sim = [t_log_sim, t_step]; %#ok<AGROW>
  dq_log_sim = [dq_log_sim, dq_step]; %#ok<AGROW>
end


anim = Animator.MinitaurSimpleAnimator(t_log_sim, q_log_sim);
anim.pov = Animator.AnimatorPointOfView.Front;
anim.Animate(true);
anim.isLooping = false;
anim.updateWorldPosition = false;
anim.endTime = tf;
conGUI = Animator.AnimatorControls();
conGUI.anim = anim;

make_vid.cycles = 1;
make_vid.flag = false;
make_vid.filename = ''; %'avi/simulatedgait_Bez4_front.avi';
make_vid.pov = [0, 0];
plot_frames = true;
plotMinitaurSimple(t_log_sim, q_log_sim, make_vid, plot_frames);
